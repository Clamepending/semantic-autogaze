"""Train an image-level contrastive head over cached L3 features.

Goal: a small head that takes (L3_patches, text_emb) -> scalar image-level
score that discriminates "category present in image" vs "absent". Falsifier
diagnosis from r/filter-use-formalize: the cycle-2 frozen-head was trained
per-patch-BCE against IconStudent-118K teacher heatmaps and so excels at
within-image patch ranking but fails at cross-image discrimination
(ROC-AUC ~ 0.55). This move trains the head with explicit image-level
positives vs negatives via multi-label BCE on max-pooled patch logits.

Architecture:
  patches (B,196,192) -> patch MLP -> (B,196,128) (l2-normed)
  text     (Q,512)    -> linear     -> (Q,128)    (l2-normed)
  patch_logits = einsum('bnd,qd->bqn', zp, zt) * scale + bias
  image_logits = max(patch_logits, dim=-1)         # MIL: any patch suffices

Training: 4000 train / 1000 val from val2017 cached features. Multi-label
BCE on image_logits with pos_weight tuned to COCO val pos_rate (~3.66%).
Per-epoch validation: ROC-AUC over (img, cat) pairs on the 1000-image holdout.
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
                 aggregator="max", attn_temp=4.0):
        super().__init__()
        self.aggregator = aggregator
        self.attn_temp_init = attn_temp
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
            # learnable inverse-temperature for attention softmax over patches
            self.attn_log_temp = nn.Parameter(torch.log(torch.tensor(attn_temp)))

    def patch_logits(self, patches, text):
        # patches: (B, N, patch_dim) ; text: (Q, text_dim) -> (B, Q, N)
        zp = F.normalize(self.patch_proj(patches), dim=-1)
        zt = F.normalize(self.text_proj(text), dim=-1)
        return torch.einsum("bnd,qd->bqn", zp, zt) * self.scale + self.bias

    def image_logits(self, patches, text, reduction=None):
        red = reduction or self.aggregator
        log = self.patch_logits(patches, text)  # (B, Q, N)
        if red == "max":
            return log.max(dim=-1).values
        if red == "mean":
            return log.mean(dim=-1)
        if red == "topk":
            k = max(1, log.shape[-1] // 20)
            top, _ = log.topk(k, dim=-1)
            return top.mean(dim=-1)
        if red == "attn":
            # text-conditioned soft attention over patches.
            # Use log itself (cosine_sim * scale + bias) divided by a learnable temp
            # as the attention logits; weighted sum produces a per-(image,query) score.
            # For aggregation we sum the soft-weighted logits — equivalent to a softmax-weighted
            # mean of patch_logits, which strictly interpolates between max (low temp) and mean (high temp).
            t = torch.exp(self.attn_log_temp)
            w = F.softmax(log / t, dim=-1)  # (B, Q, N)
            return (w * log).sum(dim=-1)
        raise ValueError(red)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    flat_s = scores.flatten().astype(np.float64)
    flat_y = labels.flatten().astype(np.int32)
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


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    flat_s = scores.flatten().astype(np.float64)
    flat_y = labels.flatten().astype(np.int32)
    n_pos = int(flat_y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-flat_s)
    sorted_y = flat_y[order]
    cum_tp = np.cumsum(sorted_y)
    precision = cum_tp / np.arange(1, len(sorted_y) + 1)
    recall = cum_tp / n_pos
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer-cache-dir", default="results/autogaze_probe/features_gaze_layers_val")
    p.add_argument("--clip-text", default="results/icon_student_B_native_train/clip_text_embeddings.pt")
    p.add_argument("--data-dir", default="data/coco")
    p.add_argument("--ann", default="annotations/instances_val2017.json")
    p.add_argument("--out-dir", default="results/filter_head_retrain")
    p.add_argument("--device", default="cpu")
    p.add_argument("--reduction", default="max", choices=["max", "mean", "topk", "attn"])
    p.add_argument("--ckpt-name", default="best.pt")
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
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
    text_mat = torch.stack([clip_text[str(c)].float() for c in cat_ids]).to(device)  # (Q, 512)
    print(f"[diag] Q={Q} categories")

    cached_imgs = sorted(int(p.stem) for p in layer_cache.glob("*.pt"))
    print(f"[diag] {len(cached_imgs)} cached images")

    print("[diag] preloading L3 features into memory...")
    t0 = time.time()
    feats = torch.zeros((len(cached_imgs), 196, 192), dtype=torch.float32)
    labels = torch.zeros((len(cached_imgs), Q), dtype=torch.float32)
    for i, img_id in enumerate(tqdm(cached_imgs, desc="preload")):
        stack = torch.load(layer_cache / f"{img_id}.pt", weights_only=True)
        feats[i] = stack[3].float()
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        for a in ann_ids:
            c = coco.anns[a]["category_id"]
            if c in cat_id_to_col:
                labels[i, cat_id_to_col[c]] = 1.0
    print(f"[diag] preloaded {len(cached_imgs)} feats in {time.time()-t0:.1f}s")

    pos_rate = float(labels.mean())
    print(f"[diag] image-level pos_rate (per-pair) = {pos_rate:.4f}")
    pos_weight = torch.tensor((1.0 - pos_rate) / max(pos_rate, 1e-8), device=device)
    print(f"[diag] BCE pos_weight = {float(pos_weight):.2f}")

    # Random shuffle for val split
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

        # validate
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
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_auc": auc, "val_ap": ap})
        print(f"[ep {epoch:02d}] loss={avg_loss:.4f}  val_auc={auc:.4f}  val_ap={ap:.4f}", flush=True)
        if auc > best_auc:
            best_auc = auc
            torch.save({
                "state_dict": head.state_dict(),
                "config": {"patch_dim": 192, "text_dim": 512,
                           "proj_dim": args.proj_dim, "hidden": args.hidden,
                           "aggregator": args.reduction},
                "epoch": epoch, "val_auc": auc, "val_ap": ap,
                "reduction": args.reduction,
            }, best_path)

    print(f"[done] best val_auc={best_auc:.4f} (saved to {best_path})")
    with open(out_dir / "history.json", "w") as f:
        json.dump({
            "args": vars(args), "n_params": n_params,
            "best_val_auc": best_auc,
            "pos_rate": pos_rate,
            "history": history,
        }, f, indent=2)


if __name__ == "__main__":
    main()
