"""r/coco-mask-distill cycle 1: train ImageLevelFilterHead over L3 14x14 with
per-patch BCE on COCO GT masks downsampled to 14x14.

Hypothesis: removing the peak-magnitude requirement (vs r/teacher-mse-distill
which falsified because L3 head couldn't reach teacher 0.99 peaks) lets a frozen
L3 head clear 6/20 wins+ties. Binary target also aligns train and eval target.

Target build:
  for each (img_id, cat_id) with COCO annotation:
    full-res GT mask -> mean-pool to 14x14 -> binarize at threshold 0.25
    cell value = 1 if >=25% of patch is covered by GT, else 0
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
from pycocotools.coco import COCO
from tqdm import tqdm


class ImageLevelFilterHead(nn.Module):
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


def per_patch_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    flat_s = scores.flatten().astype(np.float64)
    flat_y = labels.flatten().astype(np.int32)
    n_pos = int(flat_y.sum()); n_neg = len(flat_y) - n_pos
    if n_pos == 0 or n_neg == 0: return float("nan")
    order = np.argsort(-flat_s)
    sorted_y = flat_y[order]
    cum_tp = np.cumsum(sorted_y); cum_fp = np.cumsum(1 - sorted_y)
    tpr = cum_tp / n_pos; fpr = cum_fp / n_neg
    return float(np.trapezoid(tpr, fpr))


def build_target_14x14(coco: COCO, img_id: int, cat_id: int, thresh: float = 0.25) -> torch.Tensor | None:
    """Return (14, 14) binary tensor {0,1}, or None if no annotations."""
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=0)
    if not ann_ids:
        return None
    info = coco.imgs[img_id]
    h, w = info["height"], info["width"]
    full = np.zeros((h, w), dtype=np.uint8)
    for a in coco.loadAnns(ann_ids):
        m = coco.annToMask(a)
        if m.shape != (h, w):  # rare crowd annotation mismatch; skip
            continue
        full |= m.astype(np.uint8)
    if full.sum() == 0:
        return None
    t = torch.from_numpy(full).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    pooled = F.adaptive_avg_pool2d(t, output_size=(14, 14))[0, 0]  # (14,14) in [0,1]
    return (pooled >= thresh).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer-cache-dir", default="results/autogaze_probe/features_gaze_layers_val")
    p.add_argument("--clip-text", default="results/icon_student_B_native_train/clip_text_embeddings.pt")
    p.add_argument("--data-dir", default="data/coco")
    p.add_argument("--ann", default="annotations/instances_val2017.json")
    p.add_argument("--out-dir", default="results/coco_mask_distill")
    p.add_argument("--device", default="cpu")
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--patch-thresh", type=float, default=0.25,
                   help="patch is positive if >= this fraction is covered by GT")
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

    print("[diag] loading CLIP text + COCO ann...", flush=True)
    clip_text = torch.load(args.clip_text, map_location="cpu", weights_only=True)
    coco = COCO(os.path.join(args.data_dir, args.ann))
    cat_ids_in_text = {int(k) for k in clip_text.keys()}

    cached_imgs = sorted(int(p.stem) for p in layer_cache.glob("*.pt"))
    print(f"[diag] {len(cached_imgs)} cached L3 feature images", flush=True)

    # Build (img_row, cat_id, mask_196) triples
    print(f"[diag] building (img, cat, mask_14x14) triples (patch-thresh={args.patch_thresh})...", flush=True)
    img_to_row = {img_id: i for i, img_id in enumerate(cached_imgs)}
    triples = []
    for img_id in tqdm(cached_imgs, desc="build-targets"):
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=0)
        cats_here = sorted({coco.anns[a]["category_id"] for a in ann_ids
                            if coco.anns[a]["category_id"] in cat_ids_in_text})
        for cat_id in cats_here:
            mask = build_target_14x14(coco, img_id, cat_id, thresh=args.patch_thresh)
            if mask is not None:
                triples.append((img_to_row[img_id], cat_id, mask.flatten()))
    print(f"[diag] {len(triples)} (img, cat) triples", flush=True)
    if len(triples) == 0:
        raise RuntimeError("no triples built — check COCO/clip_text")

    # Preload L3 features
    print("[diag] preloading L3 features...", flush=True)
    t0 = time.time()
    feats = torch.zeros((len(cached_imgs), 196, 192), dtype=torch.float32)
    for i, img_id in enumerate(tqdm(cached_imgs, desc="preload-feat")):
        stack = torch.load(layer_cache / f"{img_id}.pt", weights_only=True)
        feats[i] = stack[3].float()
    print(f"[diag] preloaded in {time.time()-t0:.1f}s ({feats.numel()*4/(1024**2):.0f} MB)", flush=True)

    # Split by image
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(cached_imgs))
    val_imgs = set(perm[: args.n_val_imgs].tolist())
    trip_tr = [t for t in triples if t[0] not in val_imgs]
    trip_val = [t for t in triples if t[0] in val_imgs]
    print(f"[diag] split: train={len(trip_tr)} triples / val={len(trip_val)} triples", flush=True)

    def pack(trips):
        rows = torch.tensor([t[0] for t in trips], dtype=torch.long)
        text = torch.stack([clip_text[str(t[1])].float() for t in trips])
        target = torch.stack([t[2] for t in trips])
        return rows, text, target

    rows_tr, text_tr, tgt_tr = pack(trip_tr)
    rows_val, text_val, tgt_val = pack(trip_val)
    pos_rate = float(tgt_tr.mean())
    print(f"[diag] per-patch positive rate (tr) = {pos_rate:.4f}", flush=True)
    pos_weight = torch.tensor((1.0 - pos_rate) / max(pos_rate, 1e-8), device=device)
    print(f"[diag] BCE pos_weight = {float(pos_weight):.2f}", flush=True)

    feats = feats.to(device)
    text_tr = text_tr.to(device); tgt_tr = tgt_tr.to(device); rows_tr = rows_tr.to(device)
    text_val = text_val.to(device); tgt_val = tgt_val.to(device); rows_val = rows_val.to(device)

    head = ImageLevelFilterHead(
        patch_dim=192, text_dim=512,
        proj_dim=args.proj_dim, hidden=args.hidden,
        aggregator="attn",
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[diag] head params: {n_params}", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward_rows(rows, text):
        out = torch.empty((len(rows), 196), device=device)
        for s in range(0, len(rows), args.batch_size):
            r = rows[s:s+args.batch_size]
            x = feats[r]
            t = text[s:s+args.batch_size]
            zp = F.normalize(head.patch_proj(x), dim=-1)
            zt = F.normalize(head.text_proj(t), dim=-1)
            out[s:s+args.batch_size] = (zp * zt.unsqueeze(1)).sum(-1) * head.scale + head.bias
        return out

    history = []
    best_auc = -1.0
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
            y = tgt_tr[sel]
            zp = F.normalize(head.patch_proj(x), dim=-1)
            zt = F.normalize(head.text_proj(t), dim=-1)
            logits = (zp * zt.unsqueeze(1)).sum(-1) * head.scale + head.bias
            loss = bce(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            loss_sum += float(loss); nb += 1

        head.eval()
        with torch.inference_mode():
            l_val = forward_rows(rows_val, text_val)
            scores = torch.sigmoid(l_val).cpu().numpy()
            labels = tgt_val.cpu().numpy()
            val_auc = per_patch_auroc(scores, labels)
            val_max_prob_mean = float(scores.max(axis=1).mean())
        avg_loss = loss_sum / max(nb, 1)
        history.append({"epoch": epoch, "train_loss": avg_loss,
                        "val_per_patch_auroc": val_auc,
                        "val_max_prob_mean": val_max_prob_mean})
        print(f"[ep {epoch:02d}] train_loss={avg_loss:.4f}  val_per_patch_auroc={val_auc:.4f}  "
              f"val_max_prob_mean={val_max_prob_mean:.3f}", flush=True)
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "state_dict": head.state_dict(),
                "config": {"patch_dim": 192, "text_dim": 512,
                           "proj_dim": args.proj_dim, "hidden": args.hidden,
                           "aggregator": "attn"},
                "epoch": epoch, "val_per_patch_auroc": val_auc,
                "val_max_prob_mean": val_max_prob_mean,
            }, best_path)

    print(f"[done] best val_per_patch_auroc={best_auc:.4f} (saved to {best_path})", flush=True)
    with open(out_dir / "history.json", "w") as f:
        json.dump({"args": vars(args), "n_params": n_params,
                   "best_val_per_patch_auroc": best_auc,
                   "pos_rate": pos_rate, "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
