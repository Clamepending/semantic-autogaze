"""Phase 2b: SigLIP-style dense per-pair contrastive distillation from Grounded-SAM.

Loss formulation (from lit review — novel combination):
  For each minibatch of B images and Q queries (Q includes a mix of
  per-image positives + cross-batch negatives):
    let logit[b, q, h, w] = t * cosine(patch_emb[b, h, w], text_emb[q]) + bias
    let target[b, q, h, w] = grounded_sam_mask if (b, q) is a "positive" else 0
    L_dense = mean( BCEWithLogits(logit, target) )
    L_pool  = BCEWithLogits( meanpool_hw(logit), present_label[b, q] )
    L_dice  = soft_dice_loss(sigmoid(logit), target) on present pairs only
    L = L_dense + 0.5 * L_pool + 0.3 * L_dice  (+ optional L2 feature aux)

Training data: Phase 2a outputs (.npz files) at results/phase2_targets/.
Each npz is a single (image, query) pair with mask14 (14x14), gate_passed,
presence flag.

Within each minibatch we want a mix of:
  - on-diagonal positives: (image_b, query_b) where Grounded-SAM said "yes"
  - "absent" negatives: (image_b, query_b) where Grounded-SAM said "no" (gated
     out OR gated through but no box). Target = zero mask.
  - off-diagonal negatives: pairs from other images in the same batch that
     are confirmed-absent for cross-image generalization.

Backbones: v1 (CLIP-B/16 visual frozen), v2-tiny (timm ViT-Tiny/16 frozen),
d-mobile (timm MobileNet-V3-Small frozen). Last-2 transformer blocks of
backbone are tunable when --finetune_backbone_blocks > 0 (CLIPSelf-style).
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")
from train_independent_scorer import TextScorerHead, GRID
from train_independent_scorer_v2 import IM_MEAN, IM_STD

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ---- Dataset ----

class TargetDataset(Dataset):
    """Streams (image, query, mask14, presence, clip_sim) from phase2 npz files.

    Each __getitem__ returns ONE pair. Batches mix positive + negative pairs
    naturally because we sample uniformly from the npz directory.
    """

    def __init__(self, target_dir: str, image_dir: str,
                 image_size: int = 224, mean=CLIP_MEAN, std=CLIP_STD,
                 limit: int | None = None):
        self.target_dir = Path(target_dir)
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        files = sorted(self.target_dir.glob("*.npz"))
        if limit:
            files = files[:limit]
        self.files = files
        # Precompute (img_id, query_slug) splits for fast lookup
        print(f"[dataset] {len(self.files)} npz files in {target_dir}", flush=True)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        npz_path = self.files[idx]
        d = np.load(npz_path, allow_pickle=True)
        img_id = str(d["img_id"]) if "img_id" in d else npz_path.stem.split("__")[0]
        query = str(d["query"]) if "query" in d else npz_path.stem.split("__", 1)[1]
        img_path = self.image_dir / f"{img_id}.jpg"
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception:
            return None
        # Resize + normalize
        arr = np.array(pil.resize((self.image_size, self.image_size), Image.BICUBIC))
        x = (arr.astype(np.float32) / 255.0 - self.mean) / self.std
        x = torch.from_numpy(x).permute(2, 0, 1).float()  # (3, H, W)
        mask14 = torch.from_numpy(d["mask14"].astype(np.float32) / 255.0)  # (14, 14) {0,1}
        return {
            "image": x,
            "query": query,
            "mask14": mask14,
            "presence": bool(d["presence"]),
            "clip_sim": float(d["clip_sim"]),
            "gate_passed": bool(d["gate_passed"]),
        }


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "query": [b["query"] for b in batch],
        "mask14": torch.stack([b["mask14"] for b in batch]),
        "presence": torch.tensor([b["presence"] for b in batch], dtype=torch.float32),
        "clip_sim": torch.tensor([b["clip_sim"] for b in batch]),
        "gate_passed": torch.tensor([b["gate_passed"] for b in batch], dtype=torch.bool),
    }


# ---- Backbone factory ----

def build_backbone(model: str, device, finetune_blocks: int = 0):
    """Returns (backbone_callable, patch_dim, mean, std, kind)."""
    if model == "v1":
        # CLIP-B/16 visual encoder. Patches at 14x14 grid, 768-d.
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        clip_model = clip_model.to(device)
        clip_model.visual.output_tokens = True
        # Freeze all, then unfreeze last N transformer blocks
        for p in clip_model.parameters(): p.requires_grad_(False)
        if finetune_blocks > 0:
            blocks = clip_model.visual.transformer.resblocks
            for blk in blocks[-finetune_blocks:]:
                for p in blk.parameters(): p.requires_grad_(True)
            print(f"[backbone] v1: unfrozen last {finetune_blocks} transformer blocks "
                  f"(of {len(blocks)})", flush=True)
        # Wrap as a callable that returns (B, 196, 768)
        def fn(x):
            _, patches = clip_model.visual(x)  # (B, 196, 768)
            return patches
        return fn, 768, CLIP_MEAN, CLIP_STD, "clip-visual", clip_model

    import timm
    if model == "v2-tiny":
        bb = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k",
                               pretrained=True, num_classes=0).to(device)
        for p in bb.parameters(): p.requires_grad_(False)
        if finetune_blocks > 0:
            for blk in bb.blocks[-finetune_blocks:]:
                for p in blk.parameters(): p.requires_grad_(True)
        def fn(x):
            f = bb.forward_features(x)  # (B, 197, 192) — has CLS
            return f[:, 1:, :]
        return fn, 192, IM_MEAN, IM_STD, "timm-vit", bb
    if model == "d-mobile":
        bb = timm.create_model("mobilenetv3_small_100",
                               pretrained=True, num_classes=0).to(device)
        for p in bb.parameters(): p.requires_grad_(False)
        if finetune_blocks > 0:
            # MobileNet doesn't have block-level easily; unfreeze stem-3 + classifier
            print(f"[backbone] d-mobile: finetune_blocks not supported, keeping frozen", flush=True)
        def fn(x):
            f = bb.forward_features(x)  # (B, 576, 7, 7) — 4D
            f = F.interpolate(f, size=(GRID, GRID), mode="bilinear", align_corners=False)
            return f.permute(0, 2, 3, 1).reshape(f.shape[0], GRID * GRID, f.shape[1])
        return fn, 576, IM_MEAN, IM_STD, "timm-cnn", bb
    raise ValueError(f"unknown model {model}")


# ---- Loss ----

class SiglipBias(nn.Module):
    """Learnable temperature + bias used in SigLIP-style sigmoid contrastive."""
    def __init__(self, t_init=10.0, bias_init=-4.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(float(np.log(t_init))))
        self.bias = nn.Parameter(torch.tensor(float(bias_init)))

    def forward(self, cosine):
        return cosine * self.log_t.exp() + self.bias


def soft_dice_loss(probs, target, eps=1e-6):
    """probs, target: (..., H*W) or (..., H, W). Returns scalar."""
    p = probs.flatten(-2, -1) if probs.dim() >= 3 else probs
    t = target.flatten(-2, -1) if target.dim() >= 3 else target
    inter = (p * t).sum(-1)
    union = p.sum(-1) + t.sum(-1)
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def encode_text_batch(clip_model, clip_tok, queries, device):
    """Returns (Q, 512) text embeddings, normalized."""
    toks = clip_tok(queries).to(device)
    with torch.no_grad():
        emb = F.normalize(clip_model.encode_text(toks), dim=-1)
    return emb


# ---- Training ----

def train(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[device] {device}", flush=True)

    # Always need CLIP text encoder for queries
    print(f"[clip] loading text encoder ...", flush=True)
    import open_clip
    clip_model_text, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_model_text = clip_model_text.to(device).eval()
    for p in clip_model_text.parameters(): p.requires_grad_(False)

    # Backbone
    bb_fn, patch_dim, mean, std, kind, bb_module = build_backbone(
        args.model, device, finetune_blocks=args.finetune_backbone_blocks)

    # Head
    head = TextScorerHead(
        patch_dim=patch_dim, text_dim=512,
        hidden_dim=args.head_hidden_dim, n_attn_heads=args.head_attn_heads,
        n_attn_layers=args.head_attn_layers, grid_size=GRID,
        use_spatial=args.head_use_spatial,
    ).to(device)
    head.train()

    # SigLIP bias / temperature
    sb = SiglipBias(t_init=args.t_init, bias_init=args.bias_init).to(device)

    # Optimizer
    params = [p for p in head.parameters()] + [p for p in sb.parameters()]
    if args.finetune_backbone_blocks > 0:
        params += [p for p in bb_module.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    n_train_params = sum(p.numel() for p in params)
    print(f"[opt] {n_train_params/1e6:.2f} M trainable params (lr={args.lr})", flush=True)

    # Data
    ds = TargetDataset(args.target_dir, args.image_dir,
                       image_size=224, mean=mean, std=std,
                       limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate,
                        drop_last=True, pin_memory=True)
    print(f"[data] {len(ds)} pairs, batch_size={args.batch_size}, "
          f"{len(loader)} steps/epoch", flush=True)

    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    log_f = open(log_path, "a")

    step = 0
    best_loss = float("inf")
    t_start = time.time()
    for epoch in range(args.epochs):
        for batch in loader:
            if batch is None: continue
            B = batch["image"].shape[0]
            images = batch["image"].to(device)
            queries = batch["query"]
            mask14 = batch["mask14"].to(device)        # (B, 14, 14)
            presence = batch["presence"].to(device)    # (B,)
            gate_passed = batch["gate_passed"].to(device)

            # 1) text embs for ALL queries in batch (Q = B since 1 query per pair)
            text_embs = encode_text_batch(clip_model_text, clip_tok, queries, device)  # (B, 512)

            # 2) backbone -> patches
            if args.finetune_backbone_blocks > 0:
                patches = bb_fn(images)  # (B, 196, patch_dim)
            else:
                with torch.no_grad():
                    patches = bb_fn(images)

            # 3) Run the head Q times = B times for each row (image) against ALL queries.
            # We construct the (B, Q, 196) heatmap as: for each (b, q), run head(patches[b], text_embs[q]).
            # Vectorize via broadcasting: replicate patches B times, replicate text Q times, run.
            patches_rep = patches.unsqueeze(1).expand(B, B, -1, -1).reshape(B * B, GRID * GRID, patch_dim)
            text_rep = text_embs.unsqueeze(0).expand(B, B, -1).reshape(B * B, -1)
            logits_flat = head(patches_rep, text_rep)  # (B*B, 196)
            logits = logits_flat.reshape(B, B, GRID, GRID)  # (B, Q, H, W)

            # 4) Build target: (B, Q, 14, 14)
            #    On-diagonal: mask14[b]
            #    Off-diagonal: zeros
            target = torch.zeros(B, B, GRID, GRID, device=device)
            for b in range(B):
                target[b, b] = mask14[b]
            target_present = torch.zeros(B, B, device=device)
            for b in range(B):
                target_present[b, b] = presence[b]

            # 5) SigLIP bias on logits (treats as cosine-like; logits ARE arbitrary-scale
            # already from the head, so we apply learnable t/bias as a calibration)
            cal_logits = sb(logits)

            # 6) L_dense: per-patch BCEWithLogits over (B, Q, H, W)
            L_dense = F.binary_cross_entropy_with_logits(cal_logits, target, reduction="mean")

            # 7) L_pool: image-level presence loss using mean-pooled logits
            pooled = cal_logits.mean(dim=(-2, -1))  # (B, Q)
            L_pool = F.binary_cross_entropy_with_logits(pooled, target_present, reduction="mean")

            # 8) L_dice on diagonal pairs only (where mask is meaningful)
            diag_idx = torch.arange(B, device=device)
            diag_logits = cal_logits[diag_idx, diag_idx]   # (B, H, W)
            diag_target = target[diag_idx, diag_idx]       # (B, H, W)
            L_dice = soft_dice_loss(torch.sigmoid(diag_logits), diag_target)

            L = L_dense + args.lambda_pool * L_pool + args.lambda_dice * L_dice

            opt.zero_grad(); L.backward(); opt.step()
            step += 1

            if step % args.log_every == 0:
                with torch.no_grad():
                    diag_iou = ((torch.sigmoid(diag_logits) > 0.5) & (diag_target > 0.5)).sum() / max(
                        1, ((torch.sigmoid(diag_logits) > 0.5) | (diag_target > 0.5)).sum())
                line = {
                    "step": step, "epoch": epoch, "L": float(L.item()),
                    "L_dense": float(L_dense.item()), "L_pool": float(L_pool.item()),
                    "L_dice": float(L_dice.item()), "diag_iou": float(diag_iou.item()),
                    "t": float(sb.log_t.exp().item()), "bias": float(sb.bias.item()),
                    "lr": float(opt.param_groups[0]['lr']),
                    "elapsed_min": (time.time() - t_start) / 60,
                }
                print(f"  step {step:5d} | L={L:.3f} (dense={L_dense:.3f} pool={L_pool:.3f} dice={L_dice:.3f}) | diag_iou={diag_iou:.3f} | t={line['t']:.1f} bias={line['bias']:.2f}",
                      flush=True)
                log_f.write(json.dumps(line) + "\n"); log_f.flush()

            if step % args.save_every == 0:
                ckpt = {
                    "step": step, "epoch": epoch, "model": args.model,
                    "head": head.state_dict(),
                    "sb": sb.state_dict(),
                    "args": vars(args),
                    "embed_dim": patch_dim,
                    "backbone": (
                        "ViT-B-16" if args.model == "v1" else
                        "vit_tiny_patch16_224.augreg_in21k_ft_in1k" if args.model == "v2-tiny" else
                        "mobilenetv3_small_100"
                    ),
                }
                if args.finetune_backbone_blocks > 0:
                    ckpt["backbone_state"] = bb_module.state_dict()
                save_path = os.path.join(args.output_dir, f"ckpt_step{step}.pt")
                torch.save(ckpt, save_path)
                # Also "best" if loss is best
                if L.item() < best_loss:
                    best_loss = L.item()
                    torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))
                print(f"    [save] {save_path}", flush=True)

            if args.max_steps and step >= args.max_steps:
                print(f"[stop] reached max_steps={args.max_steps}", flush=True)
                break
        if args.max_steps and step >= args.max_steps:
            break

    log_f.close()
    print(f"\n[done] training finished in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model", required=True, choices=["v1", "v2-tiny", "d-mobile"])
    p.add_argument("--target_dir", default="/home/ogata/semantic-autogaze/results/phase2_targets")
    p.add_argument("--image_dir", default="/home/ogata/semantic-autogaze/data/coco_val2017/val2017")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="limit dataset size for fast iteration")

    # Training
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--finetune_backbone_blocks", type=int, default=0,
                   help="ViT only: unfreeze last N transformer blocks")

    # Loss weights
    p.add_argument("--lambda_pool", type=float, default=0.5)
    p.add_argument("--lambda_dice", type=float, default=0.3)
    p.add_argument("--t_init", type=float, default=10.0)
    p.add_argument("--bias_init", type=float, default=-4.0)

    # Head config
    p.add_argument("--head_hidden_dim", type=int, default=384)
    p.add_argument("--head_attn_heads", type=int, default=6)
    p.add_argument("--head_attn_layers", type=int, default=2)
    p.add_argument("--head_use_spatial", action="store_true", default=True)

    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=200)
    args = p.parse_args()
    train(args)
