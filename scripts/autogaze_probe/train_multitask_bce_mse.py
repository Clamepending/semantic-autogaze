"""r/multitask-bce-mse cycle 1: joint per-patch BCE on COCO GT mask 14x14
+ MSE on 118K teacher heatmap, both supervising the same per-patch logits over
frozen L3 features.

Hypothesis: combining the two strongest single-loss signals (BCE on binary GT —
2/20, mean iou=0.168 in r/coco-mask-distill) and (MSE on continuous teacher —
1/20 in r/teacher-mse-distill) might co-optimize ranking + spatial mass without
either cannibalizing the other.

  loss = w_bce * BCE(logits, gt_mask_14x14)       (binary, pos_weight balanced)
       + w_mse * MSE(sigmoid(logits), teacher_14x14)  (continuous in [0,1])

Default weights: w_bce=1.0, w_mse=1.0.
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


def build_target_14x14(coco: COCO, img_id: int, cat_id: int, thresh: float = 0.25):
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=0)
    if not ann_ids:
        return None
    info = coco.imgs[img_id]
    h, w = info["height"], info["width"]
    full = np.zeros((h, w), dtype=np.uint8)
    for a in coco.loadAnns(ann_ids):
        m = coco.annToMask(a)
        if m.shape != (h, w):
            continue
        full |= m.astype(np.uint8)
    if full.sum() == 0:
        return None
    t = torch.from_numpy(full).float().unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(t, output_size=(14, 14))[0, 0]
    return (pooled >= thresh).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer-cache-dir", default="results/autogaze_probe/features_gaze_layers_val")
    p.add_argument("--teacher-cache-dir", default="results/autogaze_probe/teacher_14x14_val")
    p.add_argument("--clip-text", default="results/icon_student_B_native_train/clip_text_embeddings.pt")
    p.add_argument("--data-dir", default="data/coco")
    p.add_argument("--ann", default="annotations/instances_val2017.json")
    p.add_argument("--out-dir", default="results/multitask_bce_mse")
    p.add_argument("--device", default="cpu")
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--patch-thresh", type=float, default=0.25)
    p.add_argument("--w-bce", type=float, default=1.0)
    p.add_argument("--w-mse", type=float, default=1.0)
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

    print("[diag] loading CLIP text + COCO ann...", flush=True)
    clip_text = torch.load(args.clip_text, map_location="cpu", weights_only=True)
    coco = COCO(os.path.join(args.data_dir, args.ann))
    cat_ids_in_text = {int(k) for k in clip_text.keys()}

    feat_imgs = {int(p.stem) for p in layer_cache.glob("*.pt")}
    teach_imgs = {int(p.stem) for p in teacher_cache.glob("*.pt")}
    cached_imgs = sorted(feat_imgs & teach_imgs)
    print(f"[diag] {len(cached_imgs)} images with both L3 and teacher", flush=True)

    print(f"[diag] building (img, cat, gt_mask, teacher) quads (patch-thresh={args.patch_thresh})...", flush=True)
    img_to_row = {img_id: i for i, img_id in enumerate(cached_imgs)}
    quads = []  # (img_row, cat_id, gt_mask_flat_196, teacher_flat_196)
    for img_id in tqdm(cached_imgs, desc="build-targets"):
        teacher_obj = torch.load(teacher_cache / f"{img_id}.pt", weights_only=True)
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=0)
        cats_here = sorted({coco.anns[a]["category_id"] for a in ann_ids
                            if coco.anns[a]["category_id"] in cat_ids_in_text})
        for cat_id in cats_here:
            if cat_id not in teacher_obj and str(cat_id) not in teacher_obj:
                continue
            mask = build_target_14x14(coco, img_id, cat_id, thresh=args.patch_thresh)
            if mask is None:
                continue
            hm = teacher_obj[cat_id] if cat_id in teacher_obj else teacher_obj[str(cat_id)]
            quads.append((img_to_row[img_id], cat_id,
                          mask.flatten(),
                          hm.flatten().float().clamp(0, 1)))
    print(f"[diag] {len(quads)} (img, cat) quads", flush=True)
    if len(quads) == 0:
        raise RuntimeError("no quads built — check COCO/clip_text/teacher_cache")

    print("[diag] preloading L3 features...", flush=True)
    t0 = time.time()
    feats = torch.zeros((len(cached_imgs), 196, 192), dtype=torch.float32)
    for i, img_id in enumerate(tqdm(cached_imgs, desc="preload-feat")):
        stack = torch.load(layer_cache / f"{img_id}.pt", weights_only=True)
        feats[i] = stack[3].float()
    print(f"[diag] preloaded in {time.time()-t0:.1f}s ({feats.numel()*4/(1024**2):.0f} MB)", flush=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(cached_imgs))
    val_imgs = set(perm[: args.n_val_imgs].tolist())
    quad_tr = [t for t in quads if t[0] not in val_imgs]
    quad_val = [t for t in quads if t[0] in val_imgs]
    print(f"[diag] split: train={len(quad_tr)} quads / val={len(quad_val)} quads", flush=True)

    def pack(quads):
        rows = torch.tensor([t[0] for t in quads], dtype=torch.long)
        text = torch.stack([clip_text[str(t[1])].float() for t in quads])
        gt   = torch.stack([t[2] for t in quads])
        teach = torch.stack([t[3] for t in quads])
        return rows, text, gt, teach

    rows_tr, text_tr, gt_tr, teach_tr = pack(quad_tr)
    rows_val, text_val, gt_val, teach_val = pack(quad_val)
    pos_rate = float(gt_tr.mean())
    print(f"[diag] per-patch pos_rate (gt, tr) = {pos_rate:.4f}; "
          f"teacher_mean_mass (tr) = {float(teach_tr.mean()):.4f}", flush=True)
    pos_weight = torch.tensor((1.0 - pos_rate) / max(pos_rate, 1e-8), device=device)

    feats = feats.to(device)
    text_tr = text_tr.to(device); gt_tr = gt_tr.to(device); teach_tr = teach_tr.to(device); rows_tr = rows_tr.to(device)
    text_val = text_val.to(device); gt_val = gt_val.to(device); teach_val = teach_val.to(device); rows_val = rows_val.to(device)

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
    best_score = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(args.epochs):
        head.train()
        perm_e = torch.randperm(len(rows_tr), device=device)
        loss_sum = 0.0; bce_sum = 0.0; mse_sum = 0.0; nb = 0
        for s in range(0, len(rows_tr), args.batch_size):
            sel = perm_e[s:s+args.batch_size]
            r = rows_tr[sel]
            x = feats[r]
            t = text_tr[sel]
            y_gt = gt_tr[sel]
            y_te = teach_tr[sel]
            zp = F.normalize(head.patch_proj(x), dim=-1)
            zt = F.normalize(head.text_proj(t), dim=-1)
            logits = (zp * zt.unsqueeze(1)).sum(-1) * head.scale + head.bias
            l_bce = bce(logits, y_gt)
            l_mse = F.mse_loss(torch.sigmoid(logits), y_te)
            loss = args.w_bce * l_bce + args.w_mse * l_mse
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            loss_sum += float(loss); bce_sum += float(l_bce); mse_sum += float(l_mse); nb += 1

        head.eval()
        with torch.inference_mode():
            l_val = forward_rows(rows_val, text_val)
            scores = torch.sigmoid(l_val).cpu().numpy()
            gt_np = gt_val.cpu().numpy()
            te_np = teach_val.cpu().numpy()
            val_auc = per_patch_auroc(scores, gt_np)
            val_mse = float(((scores - te_np) ** 2).mean())
            val_max = float(scores.max(axis=1).mean())
        avg_loss = loss_sum / max(nb, 1)
        avg_bce = bce_sum / max(nb, 1)
        avg_mse = mse_sum / max(nb, 1)
        history.append({
            "epoch": epoch, "train_loss": avg_loss,
            "train_bce": avg_bce, "train_mse": avg_mse,
            "val_per_patch_auroc": val_auc, "val_mse_vs_teacher": val_mse,
            "val_max_prob_mean": val_max,
        })
        print(f"[ep {epoch:02d}] loss={avg_loss:.4f} (bce={avg_bce:.4f} mse={avg_mse:.4f}) "
              f"val_auc={val_auc:.4f} val_mse={val_mse:.4f} val_max={val_max:.3f}", flush=True)
        # Use val_per_patch_auroc as the "best" criterion (binarized GT side; the qual gate)
        if val_auc > best_score:
            best_score = val_auc
            torch.save({
                "state_dict": head.state_dict(),
                "config": {"patch_dim": 192, "text_dim": 512,
                           "proj_dim": args.proj_dim, "hidden": args.hidden,
                           "aggregator": "attn"},
                "epoch": epoch,
                "val_per_patch_auroc": val_auc,
                "val_mse_vs_teacher": val_mse,
                "val_max_prob_mean": val_max,
            }, best_path)

    print(f"[done] best val_per_patch_auroc={best_score:.4f} (saved to {best_path})", flush=True)
    with open(out_dir / "history.json", "w") as f:
        json.dump({"args": vars(args), "n_params": n_params,
                   "best_val_per_patch_auroc": best_score,
                   "pos_rate": pos_rate, "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
