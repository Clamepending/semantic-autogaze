"""r/autogaze-deep-finetune cycle 1: unfreeze gaze_decoder layers 1..3 + final norm + ImageLevelFilterHead.

The last untested AutoGaze axis. Light unfreeze (layer 3 only, r/autogaze-light-finetune cycle 2)
moved AUROC by Δ=-0.0009 vs L3-frozen (within noise). This move escalates to 3× the trainable
surface across the deeper portion of the LLaMA stack.

Architecture:
  L0 cached  (B,196,192)  -> trainable layer 1 -> trainable layer 2 -> trainable layer 3
                          -> frozen final RMSNorm
                          -> trainable ImageLevelFilterHead(attn pool)  -> image_logits (B,Q)

Loss: multi-label BCE on aggregated patch logits, pos_weight tuned to COCO val pos_rate.
Compare-against: r/filter-head-retrain cycle 2 (frozen attn-pool over L3): val_auc=0.8473.
Falsifier: best val_auc < 0.86 -> direction falsified, no qual cycle.
"""
from __future__ import annotations

import argparse
import copy
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

from semantic_autogaze.inference import load_autogaze


class ImageLevelFilterHead(nn.Module):
    def __init__(self, patch_dim=192, text_dim=512, proj_dim=128, hidden=256,
                 aggregator="attn", attn_temp=4.0):
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
            self.attn_log_temp = nn.Parameter(torch.log(torch.tensor(attn_temp)))

    def patch_logits(self, patches, text):
        zp = F.normalize(self.patch_proj(patches), dim=-1)
        zt = F.normalize(self.text_proj(text), dim=-1)
        return torch.einsum("bnd,qd->bqn", zp, zt) * self.scale + self.bias

    def image_logits(self, patches, text, reduction=None):
        red = reduction or self.aggregator
        log = self.patch_logits(patches, text)
        if red == "max":
            return log.max(dim=-1).values
        if red == "mean":
            return log.mean(dim=-1)
        if red == "attn":
            t = torch.exp(self.attn_log_temp)
            w = F.softmax(log / t, dim=-1)
            return (w * log).sum(dim=-1)
        raise ValueError(red)


class DeepUnfreezeBackbone(nn.Module):
    """3 trainable LLaMA layers + frozen final RMSNorm. Reads L0 cached features as input."""

    def __init__(self, autogaze, n_patches=196):
        super().__init__()
        gm = autogaze.gazing_model
        self.layer1 = copy.deepcopy(gm.gaze_decoder.model.layers[1])
        self.layer2 = copy.deepcopy(gm.gaze_decoder.model.layers[2])
        self.layer3 = copy.deepcopy(gm.gaze_decoder.model.layers[3])
        for layer in (self.layer1, self.layer2, self.layer3):
            for p in layer.parameters():
                p.requires_grad = True
            layer.self_attn.config._attn_implementation = "eager"

        self.norm = copy.deepcopy(gm.gaze_decoder.model.norm)
        for p in self.norm.parameters():
            p.requires_grad = False
        self.rotary_emb = copy.deepcopy(gm.gaze_decoder.model.rotary_emb)
        for p in self.rotary_emb.parameters():
            p.requires_grad = False

        cm = torch.full((n_patches, n_patches), float("-inf"))
        cm = torch.triu(cm, diagonal=1)
        self.register_buffer("causal_mask", cm.unsqueeze(0).unsqueeze(0))
        pos = torch.arange(n_patches).unsqueeze(0)
        self.register_buffer("position_ids", pos)
        self.n_patches = n_patches

    def forward(self, l0_patches: torch.Tensor) -> torch.Tensor:
        B, T, D = l0_patches.shape
        position_ids = self.position_ids.expand(B, -1)
        cos, sin = self.rotary_emb(l0_patches, position_ids)
        attn_mask = self.causal_mask.expand(B, -1, -1, -1)
        x = l0_patches
        for layer in (self.layer1, self.layer2, self.layer3):
            out = layer(
                x,
                attention_mask=attn_mask,
                position_ids=position_ids,
                position_embeddings=(cos, sin),
            )
            x = out[0] if isinstance(out, tuple) else out
        return self.norm(x)


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
    p.add_argument("--out-dir", default="results/autogaze_deep_finetune/cycle1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--reduction", default="attn", choices=["max", "mean", "attn"])
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-layer", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--max-images", type=int, default=None,
                   help="Optional cap for fast iteration; default uses all cached.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    layer_cache = Path(args.layer_cache_dir)

    print("[diag] loading text + COCO ann...", flush=True)
    clip_text = torch.load(args.clip_text, map_location="cpu", weights_only=True)
    coco = COCO(os.path.join(args.data_dir, args.ann))
    cat_ids = sorted(int(k) for k in clip_text.keys())
    cat_id_to_col = {c: i for i, c in enumerate(cat_ids)}
    Q = len(cat_ids)
    text_mat = torch.stack([clip_text[str(c)].float() for c in cat_ids]).to(device)
    print(f"[diag] Q={Q} categories", flush=True)

    cached_imgs = sorted(int(p.stem) for p in layer_cache.glob("*.pt"))
    if args.max_images is not None:
        cached_imgs = cached_imgs[: args.max_images]
    print(f"[diag] using {len(cached_imgs)} cached images", flush=True)

    print("[diag] preloading L0 features into memory (fp32)...", flush=True)
    t0 = time.time()
    feats = torch.zeros((len(cached_imgs), 196, 192), dtype=torch.float32)
    labels = torch.zeros((len(cached_imgs), Q), dtype=torch.float32)
    for i, img_id in enumerate(tqdm(cached_imgs, desc="preload")):
        stack = torch.load(layer_cache / f"{img_id}.pt", weights_only=True)  # (4,196,192) fp16
        feats[i] = stack[0].float()  # L0 = output of layer 0 = input to layer 1
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        for a in ann_ids:
            c = coco.anns[a]["category_id"]
            if c in cat_id_to_col:
                labels[i, cat_id_to_col[c]] = 1.0
    print(f"[diag] preloaded {len(cached_imgs)} feats in {time.time()-t0:.1f}s", flush=True)

    pos_rate = float(labels.mean())
    pos_weight = torch.tensor((1.0 - pos_rate) / max(pos_rate, 1e-8), device=device)
    print(f"[diag] pos_rate={pos_rate:.4f} pos_weight={float(pos_weight):.2f}", flush=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(cached_imgs))
    val_idx = perm[: args.n_val]
    tr_idx = perm[args.n_val :]
    print(f"[diag] split: train={len(tr_idx)} val={len(val_idx)}", flush=True)

    feats_tr = feats[tr_idx].to(device)
    labels_tr = labels[tr_idx].to(device)
    feats_val = feats[val_idx].to(device)
    labels_val = labels[val_idx].to(device)

    print("[diag] loading AutoGaze (for layer weight extraction)...", flush=True)
    ag = load_autogaze(device=device)

    backbone = DeepUnfreezeBackbone(ag).to(device)
    head = ImageLevelFilterHead(
        patch_dim=192, text_dim=512,
        proj_dim=args.proj_dim, hidden=args.hidden,
        aggregator=args.reduction,
    ).to(device)

    n_layer = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[diag] trainable: layers={n_layer}, head={n_head}, total={n_layer + n_head}", flush=True)

    opt = torch.optim.AdamW([
        {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": args.lr_layer},
        {"params": head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.wd)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = []
    best_auc = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(args.epochs):
        backbone.train(); backbone.norm.eval(); backbone.rotary_emb.eval()
        head.train()
        perm_e = torch.randperm(len(tr_idx), device=device)
        loss_sum = 0.0; nb = 0
        ep_t0 = time.time()
        for s in range(0, len(tr_idx), args.batch_size):
            sel = perm_e[s : s + args.batch_size]
            x = feats_tr[sel]; y = labels_tr[sel]
            feats_out = backbone(x)
            logits = head.image_logits(feats_out, text_mat, reduction=args.reduction)
            loss = bce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss); nb += 1

        backbone.eval(); head.eval()
        with torch.inference_mode():
            scores_val = []
            for s in range(0, len(val_idx), args.batch_size):
                xv = feats_val[s : s + args.batch_size]
                fv = backbone(xv)
                lv = head.image_logits(fv, text_mat, reduction=args.reduction)
                scores_val.append(lv.cpu().numpy())
            scores_val = np.concatenate(scores_val, axis=0)
        labels_val_np = labels_val.cpu().numpy()
        auc = roc_auc(scores_val, labels_val_np)
        ap = average_precision(scores_val, labels_val_np)
        avg_loss = loss_sum / max(nb, 1)
        ep_t = time.time() - ep_t0
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_auc": auc, "val_ap": ap, "t_sec": ep_t})
        print(f"[ep {epoch:02d}] loss={avg_loss:.4f} val_auc={auc:.4f} val_ap={ap:.4f} ({ep_t:.0f}s)", flush=True)
        if auc > best_auc:
            best_auc = auc
            torch.save({
                "backbone_state_dict": backbone.state_dict(),
                "head_state_dict": head.state_dict(),
                "head_config": {"patch_dim": 192, "text_dim": 512,
                                "proj_dim": args.proj_dim, "hidden": args.hidden,
                                "aggregator": args.reduction},
                "epoch": epoch, "val_auc": auc, "val_ap": ap,
                "reduction": args.reduction,
            }, best_path)

    print(f"[done] best val_auc={best_auc:.4f} (saved to {best_path})", flush=True)
    print(f"[done] vs r/filter-head-retrain cycle 2 (L3-frozen attn-pool, val_auc=0.8473): "
          f"Δ={best_auc - 0.8473:+.4f}", flush=True)
    with open(out_dir / "history.json", "w") as f:
        json.dump({
            "args": vars(args), "n_layer_params": n_layer, "n_head_params": n_head,
            "best_val_auc": best_auc, "pos_rate": pos_rate, "history": history,
            "baseline_l3_frozen_attn_pool": 0.8473,
        }, f, indent=2)


if __name__ == "__main__":
    main()
