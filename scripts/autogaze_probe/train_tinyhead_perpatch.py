"""Train a tiny transformer head on AutoGaze L3 patches with per-patch BCE
distillation against IconStudent-118K teacher heatmaps.

Architecture:
  patches (B,196,192) -> patch_proj (192->128) -> 1x TransformerEncoderLayer
                      -> refined patches (B,196,128)
  text     (Q,512)    -> text_proj (512->128)
  patch_logits = einsum('bnd,qd->bqn', refined_norm, text_norm) * scale + bias

Training: per-patch BCE against teacher (img_id, cat_id) -> 14x14 heatmap from
results/autogaze_probe/teacher_14x14_val/. Negatives: each batch sample
includes 5 random absent categories with all-zero teacher.

Distinguishes from r/autogaze-frozen-head (BilinearCosineHead, no self-attn): the
self-attention layer lets each patch's representation incorporate context from
neighboring patches, which we hypothesize will produce more contiguous heatmaps.
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


class PerPatchTinyHead(nn.Module):
    def __init__(self, patch_dim=192, text_dim=512, proj_dim=128,
                 n_heads=4, ff_dim=256, n_layers=1, dropout=0.0):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, proj_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.text_proj = nn.Linear(text_dim, proj_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, patches, text):
        # patches: (B, 196, patch_dim) ; text: (Q, text_dim) -> (B, Q, 196)
        zp = self.transformer(self.patch_proj(patches))
        zp = F.normalize(zp, dim=-1)
        zt = F.normalize(self.text_proj(text), dim=-1)
        return torch.einsum("bnd,qd->bqn", zp, zt) * self.scale + self.bias


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer-cache-dir", default="results/autogaze_probe/features_gaze_layers_val")
    p.add_argument("--teacher-cache-dir", default="results/autogaze_probe/teacher_14x14_val")
    p.add_argument("--clip-text", default="results/icon_student_B_native_train/clip_text_embeddings.pt")
    p.add_argument("--data-dir", default="data/coco")
    p.add_argument("--ann", default="annotations/instances_val2017.json")
    p.add_argument("--out-dir", default="results/perpatch_tinyhead")
    p.add_argument("--device", default="cpu")
    p.add_argument("--ckpt-name", default="best.pt")
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--neg-per-image", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    layer_cache = Path(args.layer_cache_dir)
    teacher_cache = Path(args.teacher_cache_dir)

    print("[diag] loading text + COCO ann...")
    clip_text = torch.load(args.clip_text, map_location="cpu", weights_only=True)
    coco = COCO(os.path.join(args.data_dir, args.ann))
    cat_ids_all = sorted(int(k) for k in clip_text.keys())
    cat_id_to_col = {c: i for i, c in enumerate(cat_ids_all)}
    Q = len(cat_ids_all)
    text_mat = torch.stack([clip_text[str(c)].float() for c in cat_ids_all]).to(device)
    print(f"[diag] Q={Q} categories")

    # Restrict to images that have BOTH features AND teacher
    feat_imgs = {int(p.stem) for p in layer_cache.glob("*.pt")}
    teach_imgs = {int(p.stem) for p in teacher_cache.glob("*.pt")}
    cached_imgs = sorted(feat_imgs & teach_imgs)
    print(f"[diag] {len(cached_imgs)} images with both features and teacher")

    print("[diag] preloading L3 features...")
    t0 = time.time()
    feats = torch.zeros((len(cached_imgs), 196, 192), dtype=torch.float32)
    img_id_to_row = {}
    for i, img_id in enumerate(tqdm(cached_imgs, desc="preload feats")):
        stack = torch.load(layer_cache / f"{img_id}.pt", weights_only=True)
        feats[i] = stack[3].float()
        img_id_to_row[img_id] = i
    print(f"[diag] preloaded {len(cached_imgs)} feats in {time.time()-t0:.1f}s")

    print("[diag] preloading teacher heatmaps + building positive pair list...")
    t0 = time.time()
    pos_pairs = []  # list of (row_idx, cat_idx)
    teach_data = {}  # (row_idx, cat_idx) -> 14x14 fp32 tensor
    for i, img_id in enumerate(tqdm(cached_imgs, desc="preload teacher")):
        d = torch.load(teacher_cache / f"{img_id}.pt", weights_only=False)
        for cat_id, hm in d.items():
            cat_id = int(cat_id)
            if cat_id not in cat_id_to_col:
                continue
            cat_idx = cat_id_to_col[cat_id]
            pos_pairs.append((i, cat_idx))
            teach_data[(i, cat_idx)] = hm.float().flatten()  # (196,)
    print(f"[diag] preloaded {len(pos_pairs)} positive pairs in {time.time()-t0:.1f}s")

    pos_pairs = np.array(pos_pairs, dtype=np.int64)

    # Per-image positive cat sets for random-negative sampling
    pos_cats_by_row = {}
    for r, c in pos_pairs:
        pos_cats_by_row.setdefault(r, set()).add(c)

    rng = np.random.default_rng(args.seed)

    # Val split: by image so test images are disjoint
    img_perm = rng.permutation(len(cached_imgs))
    val_rows = set(img_perm[: args.n_val].tolist())
    tr_rows = set(img_perm[args.n_val :].tolist())
    pos_pairs_tr = pos_pairs[np.isin(pos_pairs[:, 0], list(tr_rows))]
    pos_pairs_val = pos_pairs[np.isin(pos_pairs[:, 0], list(val_rows))]
    print(f"[diag] split: train={len(pos_pairs_tr)} pairs ({len(tr_rows)} imgs)  "
          f"val={len(pos_pairs_val)} pairs ({len(val_rows)} imgs)")

    # Move features to device
    feats = feats.to(device)

    # Mean teacher activation (positive labels only) for pos_weight tuning
    teach_mean = float(np.mean([teach_data[(int(r), int(c))].mean().item() for r, c in pos_pairs[:1000]]))
    print(f"[diag] mean teacher activation (positive only) = {teach_mean:.4f}")
    # When we mix in K negatives per positive, effective pos_rate per patch ≈
    #   teach_mean / (1 + K). Tune pos_weight accordingly.
    eff_pos_rate = teach_mean / (1.0 + args.neg_per_image)
    pos_weight_val = (1.0 - eff_pos_rate) / max(eff_pos_rate, 1e-8)
    pos_weight = torch.tensor(pos_weight_val, device=device)
    print(f"[diag] effective pos_rate (per patch) = {eff_pos_rate:.4f}, pos_weight={pos_weight_val:.2f}")

    head = PerPatchTinyHead(
        patch_dim=192, text_dim=512, proj_dim=args.proj_dim,
        n_heads=args.n_heads, ff_dim=args.ff_dim, n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[diag] head params: {n_params}")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = []
    best_auc = -1.0
    best_path = out_dir / args.ckpt_name

    pos_pairs_tr_arr = pos_pairs_tr  # (N,2)
    pos_pairs_val_arr = pos_pairs_val

    def build_batch(pair_indices, K_neg, max_neg_cands=20):
        """Given indices into pos_pairs_tr_arr, return:
           - rows: (B,) image-row indices
           - cat_indices: (B, 1+K_neg) cat indices [pos, neg1, neg2, ...]
           - targets: (B, 1+K_neg, 196) per-patch teacher (zeros for negatives)
        """
        B = len(pair_indices)
        sel = pos_pairs_tr_arr[pair_indices]  # (B,2)
        rows = torch.from_numpy(sel[:, 0]).to(device)
        pos_cats = sel[:, 1]  # (B,)

        # Sample K random negative cats per row from cats not in image
        all_cats = np.arange(Q)
        neg_cats = np.zeros((B, K_neg), dtype=np.int64)
        for b in range(B):
            r = int(sel[b, 0])
            taken = pos_cats_by_row[r]
            # Rejection-sample K negatives from candidates pool
            cand = rng.integers(0, Q, size=max_neg_cands)
            cand = [int(c) for c in cand if int(c) not in taken][:K_neg]
            while len(cand) < K_neg:
                c = int(rng.integers(0, Q))
                if c not in taken:
                    cand.append(c)
            neg_cats[b] = cand
        cat_idx_t = np.concatenate([pos_cats[:, None], neg_cats], axis=1)  # (B,1+K)
        cat_idx_t_torch = torch.from_numpy(cat_idx_t).to(device)

        # Targets: positive heatmap for col 0; zeros for cols 1..K
        targets = torch.zeros((B, 1 + K_neg, 196), dtype=torch.float32, device=device)
        for b in range(B):
            t = teach_data[(int(sel[b, 0]), int(sel[b, 1]))]  # (196,)
            targets[b, 0] = t.to(device)
        return rows, cat_idx_t_torch, targets

    def per_patch_auc(scores: np.ndarray, labels: np.ndarray) -> float:
        """ROC-AUC over (batch, query, patch) triples. Threshold labels at 0.5."""
        flat_s = scores.flatten().astype(np.float64)
        flat_y = (labels.flatten() > 0.5).astype(np.int32)
        n_pos = int(flat_y.sum())
        n_neg = len(flat_y) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        order = np.argsort(-flat_s)
        sorted_y = flat_y[order]
        cum_tp = np.cumsum(sorted_y)
        cum_fp = np.cumsum(1 - sorted_y)
        tpr = cum_tp / n_pos
        fpr = cum_fp / n_neg
        return float(np.trapezoid(tpr, fpr))

    for epoch in range(args.epochs):
        head.train()
        perm_e = rng.permutation(len(pos_pairs_tr_arr))
        loss_sum = 0.0; nb = 0
        for s in range(0, len(perm_e), args.batch_size):
            sel = perm_e[s : s + args.batch_size]
            rows, cat_idx_t, targets = build_batch(sel, args.neg_per_image)
            patches = feats[rows]                              # (B, 196, 192)
            text_b = text_mat[cat_idx_t]                       # (B, 1+K, 512)
            # Per-batch text projection: do (B, 1+K, 196) logits via einsum
            zp = head.transformer(head.patch_proj(patches))    # (B, 196, D)
            zp = F.normalize(zp, dim=-1)
            zt = F.normalize(head.text_proj(text_b), dim=-1)   # (B, 1+K, D)
            logits = torch.einsum("bnd,bqd->bqn", zp, zt) * head.scale + head.bias
            loss = bce(logits, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss); nb += 1

        # validate: per-patch AUROC over val POSITIVE pairs only (K=5 negs from full Q pool)
        head.eval()
        with torch.inference_mode():
            scores_all = []; labels_all = []
            val_sub = pos_pairs_val_arr[: min(2000, len(pos_pairs_val_arr))]
            for s in range(0, len(val_sub), args.batch_size):
                sel = val_sub[s : s + args.batch_size]
                B = len(sel)
                rows = torch.from_numpy(sel[:, 0]).to(device)
                pos_cats = sel[:, 1]
                # K=5 random absent cats per row
                neg_cats = np.zeros((B, 5), dtype=np.int64)
                for b in range(B):
                    r = int(sel[b, 0])
                    taken = pos_cats_by_row.get(r, set())
                    cand = []
                    while len(cand) < 5:
                        c = int(rng.integers(0, Q))
                        if c not in taken:
                            cand.append(c)
                    neg_cats[b] = cand
                cat_idx_t = np.concatenate([pos_cats[:, None], neg_cats], axis=1)
                cat_idx_t_torch = torch.from_numpy(cat_idx_t).to(device)

                patches = feats[rows]
                text_b = text_mat[cat_idx_t_torch]
                zp = head.transformer(head.patch_proj(patches))
                zp = F.normalize(zp, dim=-1)
                zt = F.normalize(head.text_proj(text_b), dim=-1)
                logits = torch.einsum("bnd,bqd->bqn", zp, zt) * head.scale + head.bias

                targets = torch.zeros((B, 6, 196), dtype=torch.float32, device=device)
                for b in range(B):
                    t = teach_data[(int(sel[b, 0]), int(sel[b, 1]))]
                    targets[b, 0] = t.to(device)
                scores_all.append(logits.cpu().numpy())
                labels_all.append(targets.cpu().numpy())
            scores_np = np.concatenate(scores_all, axis=0)
            labels_np = np.concatenate(labels_all, axis=0)
        auc = per_patch_auc(scores_np, labels_np)
        avg_loss = loss_sum / max(nb, 1)
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_auc": auc})
        print(f"[ep {epoch:02d}] loss={avg_loss:.4f}  val_auc={auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc = auc
            torch.save({
                "state_dict": head.state_dict(),
                "config": {"patch_dim": 192, "text_dim": 512,
                           "proj_dim": args.proj_dim,
                           "n_heads": args.n_heads, "ff_dim": args.ff_dim,
                           "n_layers": args.n_layers},
                "epoch": epoch, "val_auc": auc,
            }, best_path)

    print(f"[done] best val_auc={best_auc:.4f} (saved to {best_path})")
    with open(out_dir / "history.json", "w") as f:
        json.dump({
            "args": vars(args), "n_params": n_params,
            "best_val_auc": best_auc,
            "teach_mean": teach_mean,
            "history": history,
        }, f, indent=2)


if __name__ == "__main__":
    main()
