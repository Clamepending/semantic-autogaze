"""Train an image-level attn-pool filter head over BILINEAR-UPSCALED L3 features.

Cycle 2 of r/filter-head-retrain plateaued at val_auc=0.8473 with attn-pool over
the native 14x14 grid. r/filter-head-multi-layer (concat L0..L3 = 768 dims) hit
0.8435 — feature-side richer-features hypothesis falsified. r/per-patch-supervised-
tinyhead lifted per-patch AUROC to 0.8838 with a transformer head but lost the
v11 audit 1W/18L/1T — head-capacity hypothesis falsified.

Last untested AutoGaze-frozen lever: GRID RESOLUTION. The 14x14 grid forces small
categories' teacher mass into 1-2 patches (per-pair pos_rate ~3.66%). Bilinear-
upscale L3 from 14x14 -> 28x28 (4x patches, 784 total) and re-train the cycle-2
attn-pool head. If this still plateaus, the AutoGaze-frozen direction is
comprehensively exhausted.

Architecture: same as train_filter_head.py — ImageLevelFilterHead with attn-pool.
The only change is N=784 instead of 196.
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

from train_filter_head import ImageLevelFilterHead, roc_auc, average_precision


def upscale_patches(patches_196: torch.Tensor, target_hw: int = 28) -> torch.Tensor:
    """Bilinear-upscale (B, 196, D) -> (B, target_hw**2, D)."""
    B, N, D = patches_196.shape
    assert N == 196, f"expected 196 patches, got {N}"
    grid = patches_196.reshape(B, 14, 14, D).permute(0, 3, 1, 2)  # (B, D, 14, 14)
    up = F.interpolate(grid, size=(target_hw, target_hw),
                       mode="bilinear", align_corners=False)  # (B, D, T, T)
    return up.permute(0, 2, 3, 1).reshape(B, target_hw * target_hw, D)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer-cache-dir", default="results/autogaze_probe/features_gaze_layers_val")
    p.add_argument("--clip-text", default="results/icon_student_B_native_train/clip_text_embeddings.pt")
    p.add_argument("--data-dir", default="data/coco")
    p.add_argument("--ann", default="annotations/instances_val2017.json")
    p.add_argument("--out-dir", default="results/filter_head_upscale")
    p.add_argument("--device", default="cpu")
    p.add_argument("--reduction", default="attn", choices=["max", "mean", "topk", "attn"])
    p.add_argument("--ckpt-name", default="best.pt")
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--target-hw", type=int, default=28,
                   help="upscaled grid side (28 means 784 patches; 21 means 441)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
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

    print("[diag] loading text + COCO ann...")
    clip_text = torch.load(args.clip_text, map_location="cpu", weights_only=True)
    coco = COCO(os.path.join(args.data_dir, args.ann))
    cat_ids = sorted(int(k) for k in clip_text.keys())
    cat_id_to_col = {c: i for i, c in enumerate(cat_ids)}
    Q = len(cat_ids)
    text_mat = torch.stack([clip_text[str(c)].float() for c in cat_ids]).to(device)
    print(f"[diag] Q={Q} categories")

    cached_imgs = sorted(int(p.stem) for p in layer_cache.glob("*.pt"))
    print(f"[diag] {len(cached_imgs)} cached images")

    N = args.target_hw * args.target_hw
    print(f"[diag] upscaling 14x14 -> {args.target_hw}x{args.target_hw} (N={N})")

    print("[diag] preloading + upscaling L3 features into memory...")
    t0 = time.time()
    feats = torch.zeros((len(cached_imgs), N, 192), dtype=torch.float32)
    labels = torch.zeros((len(cached_imgs), Q), dtype=torch.float32)
    BATCH = 256
    buf = []
    buf_idx = []
    for i, img_id in enumerate(tqdm(cached_imgs, desc="preload")):
        stack = torch.load(layer_cache / f"{img_id}.pt", weights_only=True)
        buf.append(stack[3].float())
        buf_idx.append(i)
        if len(buf) == BATCH:
            up = upscale_patches(torch.stack(buf), target_hw=args.target_hw)
            for k, idx in enumerate(buf_idx):
                feats[idx] = up[k]
            buf.clear(); buf_idx.clear()
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        for a in ann_ids:
            c = coco.anns[a]["category_id"]
            if c in cat_id_to_col:
                labels[i, cat_id_to_col[c]] = 1.0
    if buf:
        up = upscale_patches(torch.stack(buf), target_hw=args.target_hw)
        for k, idx in enumerate(buf_idx):
            feats[idx] = up[k]
        buf.clear(); buf_idx.clear()
    print(f"[diag] preloaded {len(cached_imgs)} feats in {time.time()-t0:.1f}s "
          f"(feats tensor {feats.numel()*4/(1024**2):.0f} MB)")

    pos_rate = float(labels.mean())
    print(f"[diag] image-level pos_rate (per-pair) = {pos_rate:.4f}")
    pos_weight = torch.tensor((1.0 - pos_rate) / max(pos_rate, 1e-8), device=device)
    print(f"[diag] BCE pos_weight = {float(pos_weight):.2f}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(cached_imgs))
    val_idx = perm[: args.n_val]
    tr_idx = perm[args.n_val :]
    print(f"[diag] split: train={len(tr_idx)} val={len(val_idx)}")

    feats_tr = feats[tr_idx].to(device)
    labels_tr = labels[tr_idx].to(device)
    feats_val = feats[val_idx].to(device)
    labels_val = labels[val_idx].to(device)

    head = ImageLevelFilterHead(
        patch_dim=192, text_dim=512,
        proj_dim=args.proj_dim, hidden=args.hidden,
        aggregator=args.reduction,
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[diag] head params: {n_params}")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = []
    best_auc = -1.0
    best_path = out_dir / args.ckpt_name

    for epoch in range(args.epochs):
        head.train()
        perm_e = torch.randperm(len(tr_idx), device=device)
        loss_sum = 0.0; nb = 0
        for s in range(0, len(tr_idx), args.batch_size):
            sel = perm_e[s : s + args.batch_size]
            x = feats_tr[sel]; y = labels_tr[sel]
            logits = head.image_logits(x, text_mat, reduction=args.reduction)
            loss = bce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss); nb += 1

        head.eval()
        with torch.inference_mode():
            scores_val = []
            for s in range(0, len(val_idx), args.batch_size):
                xv = feats_val[s : s + args.batch_size]
                lv = head.image_logits(xv, text_mat, reduction=args.reduction)
                scores_val.append(lv.cpu().numpy())
            scores_val = np.concatenate(scores_val, axis=0)
        labels_val_np = labels_val.cpu().numpy()
        auc = roc_auc(scores_val, labels_val_np)
        ap = average_precision(scores_val, labels_val_np)
        avg_loss = loss_sum / max(nb, 1)
        history.append({"epoch": epoch, "train_loss": avg_loss,
                        "val_auc": auc, "val_ap": ap})
        print(f"[ep {epoch:02d}] loss={avg_loss:.4f}  val_auc={auc:.4f}  val_ap={ap:.4f}",
              flush=True)
        if auc > best_auc:
            best_auc = auc
            torch.save({
                "state_dict": head.state_dict(),
                "config": {"patch_dim": 192, "text_dim": 512,
                           "proj_dim": args.proj_dim, "hidden": args.hidden,
                           "aggregator": args.reduction},
                "epoch": epoch, "val_auc": auc, "val_ap": ap,
                "reduction": args.reduction,
                "target_hw": args.target_hw,
            }, best_path)

    print(f"[done] best val_auc={best_auc:.4f} (saved to {best_path})")
    with open(out_dir / "history.json", "w") as f:
        json.dump({
            "args": vars(args), "n_params": n_params,
            "best_val_auc": best_auc,
            "pos_rate": pos_rate,
            "target_hw": args.target_hw,
            "history": history,
        }, f, indent=2)


if __name__ == "__main__":
    main()
