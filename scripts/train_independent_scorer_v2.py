"""r/independent-text-scorer-v1 cycle 1b — distillation to a Pi-class backbone.

Same architecture as cycle 1 but with a smaller frozen backbone in place of
CLIP ViT-B/16. Adds a "soft target" loss term using cycle-1 (Ours v1) as
teacher (the v1 head's outputs on the same image+query are an additional
supervision signal alongside COCO GT and CLIPSeg-soft).

Usage:
  python -m scripts.train_independent_scorer_v2 --backbone tiny --epochs 5 \
    --output_dir results/independent_scorer_v2_tiny
"""
from __future__ import annotations
import os, sys, time, json, argparse, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")
from pycocotools.coco import COCO
from train_independent_scorer import (
    QUAL_GRID_HOLDOUT, MAX_CATS_PER_IMAGE, MIN_AREA_FRAC, GRID, N_PATCH,
    CLIPSEG_NAME, CocoCatScorerDataset, coco_collate,
    clipseg_target_14x14, iou14, TextScorerHead,
)


# ---- Backbone registry ----
BACKBONES = {
    "tiny":      "vit_tiny_patch16_224.augreg_in21k_ft_in1k",   # 5.5M, ViT, embed=192, 196 patches
    "small":     "vit_small_patch16_224.augreg_in21k_ft_in1k",  # 21.7M, ViT, embed=384, 196 patches
    "mobilenet": "mobilenetv3_small_100",                        # 1.5M, CNN, 576 ch, 7x7 → upsample 14x14
}
# Standard ImageNet normalization for timm models.
IM_MEAN = (0.485, 0.456, 0.406)
IM_STD = (0.229, 0.224, 0.225)


def _is_cnn_backbone(name: str) -> bool:
    return "mobilenet" in name.lower() or "efficientnet" in name.lower()


@torch.no_grad()
def encode_timm_patches(timm_model, pil_images, device, mean, std):
    """Run timm visual on a list of PIL images, return (B, 196, embed_dim).

    For ViT backbones (patch=16 at 224 res) the token dimension is naturally 196
    (drop CLS token if present). For CNN backbones (e.g. MobileNet-V3-Small),
    forward_features returns (B, C, 7, 7) — bilinear-upsample to (B, C, 14, 14)
    and flatten to (B, 196, C) so the same head can consume the features.
    """
    imgs = []
    for pil in pil_images:
        arr = np.array(pil)
        t = torch.from_numpy(arr).permute(2, 0, 1).float().to(device) / 255.0
        t = F.interpolate(t.unsqueeze(0), size=(224, 224), mode="bicubic", align_corners=False).squeeze(0)
        t = (t - mean[:, None, None]) / std[:, None, None]
        imgs.append(t)
    img_batch = torch.stack(imgs, dim=0)
    feats = timm_model.forward_features(img_batch)
    if feats.dim() == 4:
        # CNN feature map (B, C, h, w) — bilinear upsample to 14x14, flatten
        feats = F.interpolate(feats, size=(GRID, GRID), mode="bilinear", align_corners=False)
        feats = feats.permute(0, 2, 3, 1).reshape(feats.shape[0], GRID * GRID, feats.shape[1])
    elif feats.shape[1] == 197:
        feats = feats[:, 1:, :]  # ViT with CLS token, drop CLS
    return feats


def main(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] backbone = {BACKBONES[args.backbone]}", flush=True)
    bb = timm.create_model(BACKBONES[args.backbone], pretrained=True, num_classes=0).to(device).eval()
    for p in bb.parameters(): p.requires_grad_(False)
    # Probe embed_dim by running a dummy forward (handles CNN vs ViT uniformly).
    with torch.no_grad():
        _probe = bb.forward_features(torch.zeros(1, 3, 224, 224, device=device))
    if _probe.dim() == 4:  # CNN
        embed_dim = _probe.shape[1]
    else:  # ViT
        embed_dim = _probe.shape[-1]
    print(f"  feature shape={tuple(_probe.shape)}, channel/embed_dim={embed_dim}, "
          f"params={sum(p.numel() for p in bb.parameters())/1e6:.2f}M", flush=True)

    print("[setup] CLIP text encoder for query embeddings (frozen) ...", flush=True)
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters(): p.requires_grad_(False)

    print("[setup] CLIPSeg (frozen, distillation target) ...", flush=True)
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    clipseg_proc = CLIPSegProcessor.from_pretrained(CLIPSEG_NAME)
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_NAME).to(device).eval()
    for p in clipseg_model.parameters(): p.requires_grad_(False)

    teacher = None
    if args.distill_v1:
        print("[setup] Ours v1 teacher (frozen) ...", flush=True)
        from train_independent_scorer import TextScorerHead as V1Head
        v1_head = V1Head(patch_dim=768, text_dim=512, hidden_dim=384,
                         n_attn_heads=6, n_attn_layers=2, grid_size=GRID).to(device).eval()
        v1_ckpt = torch.load("/home/ogata/semantic-autogaze/results/independent_scorer/best_v1.pt",
                             map_location=device)
        v1_head.load_state_dict(v1_ckpt["head"] if isinstance(v1_ckpt, dict) and "head" in v1_ckpt else v1_ckpt)
        for p in v1_head.parameters(): p.requires_grad_(False)
        # v1 needs CLIP ViT-B/16 features; for this distillation we run CLIP visual when computing
        # teacher targets per-batch.
        teacher = ("v1", v1_head)

    print("[setup] head + datasets ...", flush=True)
    head = TextScorerHead(patch_dim=embed_dim, text_dim=512,
                          hidden_dim=args.head_hidden_dim,
                          n_attn_heads=args.head_attn_heads,
                          n_attn_layers=args.head_attn_layers,
                          grid_size=GRID,
                          use_spatial=args.head_use_spatial).to(device)
    print(f"  head: {sum(p.numel() for p in head.parameters())/1e6:.2f}M params "
          f"(patch_dim={embed_dim}, hidden={args.head_hidden_dim}, "
          f"n_attn={args.head_attn_layers}, spatial={args.head_use_spatial})", flush=True)

    train_ds = CocoCatScorerDataset(args.coco_root, split="train")
    val_ds = CocoCatScorerDataset(args.coco_root, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, collate_fn=coco_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, collate_fn=coco_collate)

    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    IM_MEAN_T = torch.tensor(IM_MEAN, device=device)
    IM_STD_T = torch.tensor(IM_STD, device=device)
    CLIP_MEAN_T = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
    CLIP_STD_T  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)

    best_val_iou = -1.0; best_path = out_dir / "best.pt"
    metrics = []
    for ep in range(args.epochs):
        head.train(); t0 = time.time()
        loss_sum = 0.0; n_batches = 0
        for batch in train_loader:
            with torch.no_grad():
                # timm-backbone patches (used by the v2 head)
                patches = encode_timm_patches(bb, batch["image_pil"], device, IM_MEAN_T, IM_STD_T)
                # CLIP text embedding for query
                toks = clip_tok(batch["query"]).to(device)
                text_emb = F.normalize(clip_model.encode_text(toks), dim=-1)
                # CLIPSeg soft target
                clipseg_t = clipseg_target_14x14(clipseg_model, clipseg_proc,
                                                 batch["image_pil"], batch["query"], device)
                # v1 teacher targets (run CLIP ViT-B/16 + v1 head)
                if teacher is not None:
                    imgs = []
                    for pil in batch["image_pil"]:
                        arr = np.array(pil)
                        t = torch.from_numpy(arr).permute(2, 0, 1).float().to(device) / 255.0
                        t = F.interpolate(t.unsqueeze(0), size=(224, 224), mode="bicubic", align_corners=False).squeeze(0)
                        t = (t - CLIP_MEAN_T[:, None, None]) / CLIP_STD_T[:, None, None]
                        imgs.append(t)
                    img_b = torch.stack(imgs, dim=0)
                    clip_model.visual.output_tokens = True
                    _, clip_patches = clip_model.visual(img_b)  # (B, 196, 768)
                    clip_model.visual.output_tokens = False
                    v1_logits = teacher[1](clip_patches, text_emb)  # (B, 196)
                    v1_soft = torch.sigmoid(v1_logits).reshape(-1, GRID, GRID)
            gt = batch["gt14"].to(device).float()
            pred = head(patches, text_emb)
            pred_grid = pred.reshape(-1, GRID, GRID)
            l_bce = bce(pred_grid, gt)
            l_mse_clipseg = mse(torch.sigmoid(pred_grid), clipseg_t)
            loss = args.alpha * l_bce + args.beta * l_mse_clipseg
            if teacher is not None:
                l_mse_v1 = mse(torch.sigmoid(pred_grid), v1_soft)
                loss = loss + args.gamma * l_mse_v1
            optim.zero_grad(); loss.backward(); optim.step()
            loss_sum += loss.item(); n_batches += 1

        head.eval(); ious = []
        with torch.no_grad():
            for batch in val_loader:
                patches = encode_timm_patches(bb, batch["image_pil"], device, IM_MEAN_T, IM_STD_T)
                toks = clip_tok(batch["query"]).to(device)
                text_emb = F.normalize(clip_model.encode_text(toks), dim=-1)
                pred = head(patches, text_emb)
                gt = batch["gt14"].to(device).bool()
                ious.extend(iou14(pred, gt))
        val_iou = float(np.nanmean(ious))
        sched.step()
        line = (f"[ep {ep+1}/{args.epochs}] train_loss={loss_sum/max(1,n_batches):.4f} "
                f"val_mIoU={val_iou:.3f}  wall={time.time()-t0:.1f}s")
        print(line, flush=True)
        metrics.append({"epoch": ep+1, "train_loss": loss_sum/max(1,n_batches), "val_mIoU": val_iou})
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({"head": head.state_dict(), "args": vars(args),
                        "val_iou": val_iou, "epoch": ep+1, "backbone": BACKBONES[args.backbone],
                        "embed_dim": embed_dim}, best_path)
            print(f"  saved best to {best_path}  val_mIoU={val_iou:.3f}", flush=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"backbone": BACKBONES[args.backbone],
                   "best_val_iou": best_val_iou, "epochs": metrics,
                   "head_params_M": sum(p.numel() for p in head.parameters())/1e6,
                   "backbone_params_M": sum(p.numel() for p in bb.parameters())/1e6}, f, indent=2)
    print(f"[done] best val mIoU = {best_val_iou:.3f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backbone", choices=list(BACKBONES), default="tiny")
    p.add_argument("--coco_root", default="/home/ogata/semantic-autogaze/data/coco_val2017")
    p.add_argument("--output_dir", default="/home/ogata/semantic-autogaze/results/independent_scorer_v2_tiny")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.5, help="BCE on COCO GT weight")
    p.add_argument("--beta", type=float, default=0.3, help="MSE on CLIPSeg-soft weight")
    p.add_argument("--gamma", type=float, default=0.4, help="MSE on Ours-v1-soft (distillation) weight; 0 disables")
    p.add_argument("--distill_v1", action="store_true", default=True,
                   help="Add Ours v1 outputs as a soft-target loss (distillation)")
    p.add_argument("--head_hidden_dim", type=int, default=384)
    p.add_argument("--head_attn_layers", type=int, default=2)
    p.add_argument("--head_attn_heads", type=int, default=6)
    p.add_argument("--head_use_spatial", action="store_true", default=False,
                   help="Enable 3-conv spatial-refinement layer in head")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
