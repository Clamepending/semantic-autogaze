"""r/teacher-mse-distill cycle 1: train ImageLevelFilterHead over L3 14x14 with
MSE loss on the full 118K teacher heatmap (continuous prob mass), instead of
per-patch BCE or image-level BCE.

Hypothesis: MSE directly supervises spatial mass distribution; per-patch-ranking-
vs-heatmap-iou insight predicts this should help contiguity. Compare vs:
  - r/filter-head-retrain cycle 2 (same head, image-level BCE): 1/20 wins+ties vs GT
  - r/per-patch-supervised-tinyhead (richer head, per-patch BCE): 2/20 wins+ties
  - best AutoGaze-frozen head so far under GT-IoU: 2/20

Training:
  input  : L3 features (B, 196, 192) from cached features_gaze_layers_val
  target : teacher_14x14_val[img_id][cat_id] -> (14, 14) in [0,1]
  loss   : F.mse_loss( sigmoid(head.patch_logits(feats, text)[:, 0, :].view(14,14)),
                       teacher_heatmap )
  metric : val MSE on held-out 1000-image split; also image-level AUROC (max over
           patch probs) for backward-compat comparison.

Eval: cycle 2 audits under audit_all.py. Cycle 1 is training-only.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ImageLevelFilterHead(nn.Module):
    """Same head class as r/filter-head-retrain / r/grid-upscale-frozen."""
    def __init__(self, patch_dim=192, text_dim=512, proj_dim=128, hidden=256,
                 aggregator="attn", attn_temp=4.0):
        super().__init__()
        self.aggregator = aggregator
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, proj_dim),
        )
        self.text_proj = nn.Linear(text_dim, proj_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        if aggregator == "attn":
            self.attn_log_temp = nn.Parameter(torch.log(torch.tensor(attn_temp)))

    def patch_logits(self, patches, text):
        zp = F.normalize(self.patch_proj(patches), dim=-1)
        zt = F.normalize(self.text_proj(text), dim=-1)
        return torch.einsum("bnd,qd->bqn", zp, zt) * self.scale + self.bias


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    flat_s = scores.flatten().astype(np.float64)
    flat_y = labels.flatten().astype(np.int32)
    n_pos = int(flat_y.sum()); n_neg = len(flat_y) - n_pos
    if n_pos == 0 or n_neg == 0: return float("nan")
    order = np.argsort(-flat_s)
    sorted_y = flat_y[order]
    cum_tp = np.cumsum(sorted_y); cum_fp = np.cumsum(1 - sorted_y)
    tpr = cum_tp / n_pos; fpr = cum_fp / n_neg
    return float(np.trapezoid(tpr, fpr))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer-cache-dir", default="results/autogaze_probe/features_gaze_layers_val")
    p.add_argument("--teacher-cache-dir", default="results/autogaze_probe/teacher_14x14_val")
    p.add_argument("--clip-text", default="results/icon_student_B_native_train/clip_text_embeddings.pt")
    p.add_argument("--out-dir", default="results/teacher_mse_distill")
    p.add_argument("--device", default="cpu")
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--n-val-imgs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    layer_cache = Path(args.layer_cache_dir)
    teacher_cache = Path(args.teacher_cache_dir)

    print("[diag] loading CLIP text...", flush=True)
    clip_text = torch.load(args.clip_text, map_location="cpu", weights_only=True)

    # --- Intersect images that have BOTH L3 features AND teacher heatmap ---
    feat_imgs = {int(p.stem) for p in layer_cache.glob("*.pt")}
    teach_imgs = {int(p.stem) for p in teacher_cache.glob("*.pt")}
    cached_imgs = sorted(feat_imgs & teach_imgs)
    print(f"[diag] {len(cached_imgs)} images with both L3 and teacher", flush=True)

    # --- Build (img_row, cat_id, teacher_14x14) triples ---
    print("[diag] scanning teacher cache for (img, cat, heatmap) triples...", flush=True)
    img_to_row = {img_id: i for i, img_id in enumerate(cached_imgs)}
    triples = []  # (img_row, cat_id, teacher_flat_196)
    for img_id in tqdm(cached_imgs, desc="teacher-scan"):
        obj = torch.load(teacher_cache / f"{img_id}.pt", weights_only=True)
        for cat_id, hm in obj.items():
            cat_id = int(cat_id)
            if str(cat_id) not in clip_text:
                continue
            triples.append((img_to_row[img_id], cat_id, hm.flatten().float().clamp(0, 1)))
    print(f"[diag] {len(triples)} (img, cat) teacher triples", flush=True)

    # --- Preload L3 features into RAM ---
    print("[diag] preloading L3 features...", flush=True)
    t0 = time.time()
    feats = torch.zeros((len(cached_imgs), 196, 192), dtype=torch.float32)
    for i, img_id in enumerate(tqdm(cached_imgs, desc="preload-feat")):
        stack = torch.load(layer_cache / f"{img_id}.pt", weights_only=True)
        feats[i] = stack[3].float()
    print(f"[diag] preloaded in {time.time()-t0:.1f}s ({feats.numel()*4/(1024**2):.0f} MB)", flush=True)

    # --- Split by image (not by triple) ---
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(cached_imgs))
    val_imgs = set(perm[: args.n_val_imgs].tolist())
    trip_tr = [t for t in triples if t[0] not in val_imgs]
    trip_val = [t for t in triples if t[0] in val_imgs]
    print(f"[diag] split: train={len(trip_tr)} triples / val={len(trip_val)} triples", flush=True)

    # --- Collect triples into tensors (rows x 196, text_idx, teacher_14x14) ---
    def pack(trips):
        rows = torch.tensor([t[0] for t in trips], dtype=torch.long)
        text = torch.stack([clip_text[str(t[1])].float() for t in trips])  # (T, 512)
        teach = torch.stack([t[2] for t in trips])  # (T, 196)
        cat_ids = torch.tensor([t[1] for t in trips], dtype=torch.long)
        return rows, text, teach, cat_ids

    rows_tr, text_tr, teach_tr, cat_tr = pack(trip_tr)
    rows_val, text_val, teach_val, cat_val = pack(trip_val)
    print(f"[diag] teacher_mean_mass (tr)={float(teach_tr.mean()):.4f} "
          f"pos_rate>=0.5 (tr)={float((teach_tr>=0.5).float().mean()):.4f}", flush=True)

    feats = feats.to(device)
    text_tr = text_tr.to(device); teach_tr = teach_tr.to(device); rows_tr = rows_tr.to(device)
    text_val = text_val.to(device); teach_val = teach_val.to(device); rows_val = rows_val.to(device)

    head = ImageLevelFilterHead(
        patch_dim=192, text_dim=512,
        proj_dim=args.proj_dim, hidden=args.hidden,
        aggregator="attn",
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[diag] head params: {n_params}", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)

    def mse_loss(logits_patch: torch.Tensor, teacher_patch: torch.Tensor) -> torch.Tensor:
        # logits_patch: (T, 196); teacher_patch: (T, 196) in [0,1]
        return F.mse_loss(torch.sigmoid(logits_patch), teacher_patch)

    # Per-row paired forward: since each triple has its own (patches, text), do it manually.
    def forward_rows(rows, text, teach, eval_mode=False):
        logits_out = torch.empty((len(rows), 196), device=device)
        teach_out = teach
        for s in range(0, len(rows), args.batch_size):
            r = rows[s:s+args.batch_size]
            x = feats[r]                        # (B, 196, 192)
            t = text[s:s+args.batch_size]       # (B, 512)
            # We want per-row cosine against its own text: patch_logits with Q=B treats
            # each text against each image; we only need the diagonal. Cheaper path:
            zp = F.normalize(head.patch_proj(x), dim=-1)        # (B, 196, D)
            zt = F.normalize(head.text_proj(t), dim=-1)         # (B, D)
            lg = (zp * zt.unsqueeze(1)).sum(-1) * head.scale + head.bias  # (B, 196)
            logits_out[s:s+args.batch_size] = lg
        return logits_out, teach_out

    history = []
    best_mse = float("inf")
    best_path = out_dir / "best.pt"

    for epoch in range(args.epochs):
        head.train()
        perm_e = torch.randperm(len(rows_tr), device=device)
        loss_sum = 0.0; nb = 0
        for s in range(0, len(rows_tr), args.batch_size):
            sel = perm_e[s:s+args.batch_size]
            r = rows_tr[sel]
            x = feats[r]
            t = text_tr[sel]
            y = teach_tr[sel]
            zp = F.normalize(head.patch_proj(x), dim=-1)
            zt = F.normalize(head.text_proj(t), dim=-1)
            logits = (zp * zt.unsqueeze(1)).sum(-1) * head.scale + head.bias  # (B,196)
            loss = mse_loss(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            loss_sum += float(loss); nb += 1

        head.eval()
        with torch.inference_mode():
            l_val, t_val = forward_rows(rows_val, text_val, teach_val)
            val_mse = float(F.mse_loss(torch.sigmoid(l_val), t_val))
            # image-level per-pair "AUROC": max-prob over patches vs "has any mass" label
            max_p = torch.sigmoid(l_val).max(dim=-1).values.cpu().numpy()
            has_mass = (t_val.max(dim=-1).values > 0.1).cpu().numpy().astype(np.int32)
        avg_loss = loss_sum / max(nb, 1)
        # image AUROC only meaningful if both pos & neg present (but teach_val has only "pos" rows)
        history.append({"epoch": epoch, "train_mse": avg_loss, "val_mse": val_mse,
                        "val_max_prob_mean": float(max_p.mean())})
        print(f"[ep {epoch:02d}] train_mse={avg_loss:.5f}  val_mse={val_mse:.5f}  "
              f"val_max_prob_mean={max_p.mean():.3f}", flush=True)
        if val_mse < best_mse:
            best_mse = val_mse
            torch.save({
                "state_dict": head.state_dict(),
                "config": {"patch_dim": 192, "text_dim": 512,
                           "proj_dim": args.proj_dim, "hidden": args.hidden,
                           "aggregator": "attn"},
                "epoch": epoch, "val_mse": val_mse, "train_mse": avg_loss,
            }, best_path)

    print(f"[done] best val_mse={best_mse:.5f} (saved to {best_path})", flush=True)
    with open(out_dir / "history.json", "w") as f:
        json.dump({"args": vars(args), "n_params": n_params,
                   "best_val_mse": best_mse, "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
