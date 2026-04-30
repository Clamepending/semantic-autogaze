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

    Pre-scans all files to identify which are presence=True so we can sample
    a balanced positive-rate per batch (avoids the trivial 'predict zero
    everywhere' attractor when off-diagonal pairs dominate).
    """

    def __init__(self, target_dir: str, image_dir: str,
                 image_size: int = 224, mean=CLIP_MEAN, std=CLIP_STD,
                 limit: int | None = None, positive_only: bool = False,
                 build_presence_lookup: bool = False):
        self.target_dir = Path(target_dir)
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        files = sorted(self.target_dir.glob("*.npz"))
        if limit:
            files = files[:limit]
        # Pre-scan presence flags. ~5k files takes ~3s.
        print(f"[dataset] scanning {len(files)} npz files for presence ...", flush=True)
        keep = []
        n_pos = 0
        # presence_lookup: (img_id, query_str) → True if Grounded-SAM said yes.
        # Used to filter false-negative off-diagonal pairs in SigLIP loss.
        presence_lookup: dict = {}
        for f in files:
            try:
                d = np.load(f, allow_pickle=False)
                pres = bool(d["presence"])
            except Exception:
                continue
            if pres: n_pos += 1
            if build_presence_lookup:
                stem = f.stem
                img_id, _, query_slug = stem.partition("__")
                presence_lookup[(img_id, query_slug)] = pres
            if positive_only and not pres:
                continue
            keep.append((f, pres))
        self.files = [t[0] for t in keep]
        self.is_pos = np.array([t[1] for t in keep], dtype=bool)
        self.pos_indices = np.where(self.is_pos)[0]
        self.neg_indices = np.where(~self.is_pos)[0]
        self.presence_lookup = presence_lookup
        # Category slug per index — used by BalancedSampler for category-weighted sampling
        self.cat_per_index = [f.stem.partition("__")[2] for f in self.files]
        print(f"[dataset] kept {len(self.files)} (positive={int(self.is_pos.sum())}, "
              f"negative={int((~self.is_pos).sum())}) | original positive rate {n_pos}/{len(files)}",
              flush=True)
        if build_presence_lookup:
            print(f"[dataset] presence_lookup has {len(presence_lookup)} (img_id, query) keys",
                  flush=True)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        npz_path = self.files[idx]
        d = np.load(npz_path, allow_pickle=True)
        img_id = str(d["img_id"]) if "img_id" in d else npz_path.stem.split("__")[0]
        query = str(d["query"]) if "query" in d else npz_path.stem.split("__", 1)[1]
        # Slug stored in filename for cross-pair presence_lookup
        query_slug = npz_path.stem.split("__", 1)[1]
        img_path = self.image_dir / f"{img_id}.jpg"
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception:
            return None
        arr = np.array(pil.resize((self.image_size, self.image_size), Image.BICUBIC))
        x = (arr.astype(np.float32) / 255.0 - self.mean) / self.std
        x = torch.from_numpy(x).permute(2, 0, 1).float()
        mask14 = torch.from_numpy(d["mask14"].astype(np.float32) / 255.0)
        return {
            "image": x,
            "query": query,
            "query_slug": query_slug,
            "img_id": img_id,
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
        "query_slug": [b.get("query_slug", "") for b in batch],
        "img_id": [b.get("img_id", "") for b in batch],
        "mask14": torch.stack([b["mask14"] for b in batch]),
        "presence": torch.tensor([b["presence"] for b in batch], dtype=torch.float32),
        "clip_sim": torch.tensor([b["clip_sim"] for b in batch]),
        "gate_passed": torch.tensor([b["gate_passed"] for b in batch], dtype=torch.bool),
    }


from torch.utils.data import Sampler

class BalancedSampler(Sampler):
    """Samples 'pos_frac' positives + (1-pos_frac) negatives per epoch.

    With category_alpha > 0, also re-weights positives so rare categories are
    oversampled: weight ∝ 1 / (cat_freq ** category_alpha).
      alpha=0   → uniform per-positive (default; Phase 1-4)
      alpha=0.5 → sqrt-balanced (gentle oversample)
      alpha=1.0 → fully cat-balanced (each cat has equal expected count)

    Iterates through the larger pool (negatives or positives) once; pads with
    sampled-from the smaller pool to maintain target rate.
    """
    def __init__(self, pos_indices, neg_indices, pos_frac=0.7, seed=42,
                 cat_per_index=None, category_alpha=0.0):
        self.pos = np.asarray(pos_indices)
        self.neg = np.asarray(neg_indices)
        self.pos_frac = pos_frac
        self.seed = seed
        self._epoch = 0
        self.category_alpha = float(category_alpha)
        self.cat_per_index = cat_per_index  # list[str] or None
        self.pos_weights = None
        if self.category_alpha > 0 and cat_per_index is not None and len(self.pos) > 0:
            from collections import Counter
            pos_cats = [cat_per_index[i] for i in self.pos.tolist()]
            cat_freq = Counter(pos_cats)
            inv = np.array([(1.0 / cat_freq[c]) ** self.category_alpha for c in pos_cats])
            self.pos_weights = inv / inv.sum()
            print(f"[sampler] category-weighted positives enabled: alpha={self.category_alpha} "
                  f"({len(cat_freq)} unique cats; rarest={min(cat_freq.values())} freq, "
                  f"most-common={max(cat_freq.values())} freq)", flush=True)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        n_pos = len(self.pos); n_neg = len(self.neg)
        if n_pos == 0:
            yield from rng.permutation(self.neg).tolist(); return
        if n_neg == 0:
            yield from rng.permutation(self.pos).tolist(); return
        pos_target = n_pos
        neg_target = int(round(pos_target * (1 - self.pos_frac) / max(self.pos_frac, 1e-9)))
        # Positive sampling: weighted if alpha>0, else use all once
        if self.pos_weights is not None:
            sampled_pos = rng.choice(self.pos, size=pos_target, replace=True, p=self.pos_weights)
        else:
            sampled_pos = self.pos
        sampled_neg = rng.choice(self.neg, size=min(neg_target, n_neg * 5), replace=(n_neg < neg_target))
        all_idx = np.concatenate([sampled_pos, sampled_neg])
        rng.shuffle(all_idx)
        self._epoch += 1
        yield from all_idx.tolist()

    def __len__(self):
        n_pos = len(self.pos); n_neg = len(self.neg)
        if n_pos == 0: return n_neg
        if n_neg == 0: return n_pos
        neg_target = int(round(n_pos * (1 - self.pos_frac) / max(self.pos_frac, 1e-9)))
        return n_pos + min(neg_target, n_neg * 5)


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
    if model == "dinov2-s":
        # DINOv2 ViT-Small/14 — self-supervised, very strong patch features.
        # 22M params. Patch 14 → 16x16 patches at 224 input.
        bb = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                               pretrained=True, num_classes=0,
                               img_size=224, dynamic_img_size=True).to(device)
        for p in bb.parameters(): p.requires_grad_(False)
        if finetune_blocks > 0:
            for blk in bb.blocks[-finetune_blocks:]:
                for p in blk.parameters(): p.requires_grad_(True)
        DINOV2_MEAN = (0.485, 0.456, 0.406)
        DINOV2_STD = (0.229, 0.224, 0.225)
        def fn(x):
            f = bb.forward_features(x)  # (B, 257, 384) — 16x16 patches + cls
            patches = f[:, 1:, :]  # (B, 256, 384) drop cls
            # Reshape to spatial grid + interpolate to 14x14
            B, N, D = patches.shape
            side = int(N ** 0.5)
            grid = patches.permute(0, 2, 1).reshape(B, D, side, side)
            grid = F.interpolate(grid, size=(GRID, GRID), mode="bilinear", align_corners=False)
            return grid.permute(0, 2, 3, 1).reshape(B, GRID * GRID, D)
        return fn, 384, DINOV2_MEAN, DINOV2_STD, "timm-dinov2", bb
    if model == "mobileclip-s2":
        # Apple MobileCLIP-S2 — text-aware, mobile-optimized.
        # 35.8M visual params, 256x256 input.
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("MobileCLIP-S2",
                                                                   pretrained="datacompdr")
        clip_model = clip_model.to(device)
        for p in clip_model.parameters(): p.requires_grad_(False)
        # MobileCLIP visual returns (B, embed_dim) by default. We need patch tokens.
        # Hook into forward to extract before pooling. The backbone is a ViT
        # variant — we reuse forward_intermediates if available, or hook.
        # Easier: extract via timm-style forward_features on the underlying trunk.
        v = clip_model.visual
        # MobileCLIP-S2 visual: FastViT or hybrid. Inspect structure.
        # For simplicity: run trunk forward and grab last conv features.
        def fn(x):
            # MobileCLIP visual.trunk is the backbone returning patches/spatial.
            with torch.no_grad():
                if hasattr(v, "trunk"):
                    feats = v.trunk.forward_features(x)
                    if feats.dim() == 4:
                        feats = F.interpolate(feats, size=(GRID, GRID),
                                              mode="bilinear", align_corners=False)
                        return feats.permute(0, 2, 3, 1).reshape(feats.shape[0], GRID*GRID, feats.shape[1])
                    elif feats.dim() == 3:
                        # (B, N, D) — interp to GRID*GRID
                        side = int(feats.shape[1] ** 0.5)
                        if side * side == feats.shape[1]:
                            B, N, D = feats.shape
                            g = feats.permute(0, 2, 1).reshape(B, D, side, side)
                            g = F.interpolate(g, size=(GRID, GRID),
                                              mode="bilinear", align_corners=False)
                            return g.permute(0, 2, 3, 1).reshape(B, GRID*GRID, D)
                        # Drop cls if present
                        return feats[:, 1:, :]
                # Fallback
                return v(x)
        # MobileCLIP uses standard ImageNet normalization
        # Get embed dim by probing
        x_probe = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            out = fn(x_probe)
        embed_dim = out.shape[-1]
        print(f"[backbone] mobileclip-s2 patch_dim={embed_dim}", flush=True)
        return fn, embed_dim, IM_MEAN, IM_STD, "open_clip-mobileclip", clip_model.visual
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

    # Optional: load distillation teacher (frozen, eval).
    teacher = None
    if args.distill_teacher_ckpt:
        print(f"[distill] loading teacher {args.distill_teacher_ckpt} ...", flush=True)
        ck = torch.load(args.distill_teacher_ckpt, map_location=device, weights_only=False)
        t_args = ck.get("args", {})
        t_model = t_args.get("model", "v1")
        t_bb_fn, t_pd, t_mean, t_std, t_kind, t_bb_module = build_backbone(
            t_model, device, finetune_blocks=0)
        t_head = TextScorerHead(
            patch_dim=t_pd, text_dim=512,
            hidden_dim=t_args.get("head_hidden_dim", 384),
            n_attn_heads=t_args.get("head_attn_heads", 6),
            n_attn_layers=t_args.get("head_attn_layers", 2),
            grid_size=GRID, use_spatial=t_args.get("head_use_spatial", True),
        ).to(device).eval()
        t_head.load_state_dict(ck["head"])
        t_sb = SiglipBias().to(device).eval(); t_sb.load_state_dict(ck["sb"])
        for p in t_head.parameters(): p.requires_grad_(False)
        for p in t_sb.parameters(): p.requires_grad_(False)
        teacher = {
            "bb_fn": t_bb_fn, "head": t_head, "sb": t_sb,
            "mean": np.array(t_mean, dtype=np.float32),
            "std": np.array(t_std, dtype=np.float32),
            "model": t_model,
        }
        print(f"  teacher: model={t_model}, patch_dim={t_pd}, mean={t_mean}", flush=True)

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
                       limit=args.limit, positive_only=args.positive_only,
                       build_presence_lookup=args.fn_filter)
    if args.balanced_pos_frac > 0:
        sampler = BalancedSampler(ds.pos_indices, ds.neg_indices,
                                  pos_frac=args.balanced_pos_frac,
                                  cat_per_index=ds.cat_per_index,
                                  category_alpha=args.category_alpha)
        loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.num_workers, collate_fn=collate,
                            drop_last=True, pin_memory=True)
        print(f"[data] {len(ds)} pairs (pos_frac={args.balanced_pos_frac:.2f}), "
              f"batch_size={args.batch_size}, {len(loader)} steps/epoch", flush=True)
    else:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate,
                            drop_last=True, pin_memory=True)
        print(f"[data] {len(ds)} pairs, batch_size={args.batch_size}, "
              f"{len(loader)} steps/epoch", flush=True)

    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    log_f = open(log_path, "a")

    # Optional: wandb logging
    wb = None
    if args.wandb_project:
        try:
            import wandb as _wb
            run_name = args.wandb_run_name or os.path.basename(args.output_dir.rstrip("/"))
            wb = _wb.init(project=args.wandb_project, name=run_name,
                          config=vars(args), reinit=True, dir=args.output_dir)
            print(f"[wandb] initialised → {wb.url}", flush=True)
        except Exception as e:
            print(f"[wandb] init failed ({e}); continuing without wandb", flush=True)
            wb = None

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

            # FN filter: if off-diagonal pair (img_i, query_j) has presence=True in the
            # offline lookup, mask out that loss term — it's actually a positive that was
            # mislabeled as zero. Cleans up the SigLIP off-diagonal supervision signal.
            fn_keep = torch.ones(B, B, device=device)
            n_fn = 0
            if args.fn_filter and ds.presence_lookup:
                img_ids = batch.get("img_id", [])
                slugs = batch.get("query_slug", [])
                for i in range(B):
                    for j in range(B):
                        if i == j: continue
                        if ds.presence_lookup.get((img_ids[i], slugs[j]), False):
                            fn_keep[i, j] = 0.0
                            n_fn += 1

            # 6) L_dense: per-patch BCEWithLogits with pos_weight to fight class imbalance.
            pos_weight = torch.tensor(args.bce_pos_weight, device=device)
            # Combined per-pair weight = lambda_off_diagonal weighting * fn_keep
            diag_mask = torch.zeros(B, B, device=device)
            for b in range(B): diag_mask[b, b] = 1.0
            weight_pair = diag_mask + (1.0 - diag_mask) * args.lambda_off_diagonal
            weight_pair = weight_pair * fn_keep  # zero out FN-filtered off-diagonals
            bce_per_pos = F.binary_cross_entropy_with_logits(
                cal_logits, target, reduction="none", pos_weight=pos_weight)
            w_full = weight_pair.unsqueeze(-1).unsqueeze(-1).expand_as(bce_per_pos)
            L_dense = (bce_per_pos * w_full).sum() / w_full.sum().clamp(min=1)

            # 7) L_pool: image-level presence loss using mean-pooled logits.
            # FN-filter applies here too — don't push off-diagonal toward "absent" if
            # the query is actually in the image.
            pool_pos_weight = torch.tensor(args.pool_pos_weight, device=device)
            pooled = cal_logits.mean(dim=(-2, -1))  # (B, Q)
            pool_bce = F.binary_cross_entropy_with_logits(
                pooled, target_present, reduction="none", pos_weight=pool_pos_weight)
            L_pool = (pool_bce * fn_keep).sum() / fn_keep.sum().clamp(min=1)

            # 8) L_dice on diagonal pairs only (where mask is meaningful)
            diag_idx = torch.arange(B, device=device)
            diag_logits = cal_logits[diag_idx, diag_idx]   # (B, H, W)
            diag_target = target[diag_idx, diag_idx]       # (B, H, W)
            L_dice = soft_dice_loss(torch.sigmoid(diag_logits), diag_target)

            # 9) Optional distillation: MSE(student logits, teacher logits) on cal_logits.
            L_distill = torch.tensor(0.0, device=device)
            if teacher is not None and args.lambda_distill > 0:
                # Re-normalize images for teacher
                with torch.no_grad():
                    s_mean = torch.tensor(np.array(mean, dtype=np.float32), device=device).view(1,3,1,1)
                    s_std = torch.tensor(np.array(std, dtype=np.float32), device=device).view(1,3,1,1)
                    t_mean_t = torch.tensor(teacher["mean"], device=device).view(1,3,1,1)
                    t_std_t = torch.tensor(teacher["std"], device=device).view(1,3,1,1)
                    images_01 = images * s_std + s_mean
                    images_t = (images_01 - t_mean_t) / t_std_t
                    t_patches = teacher["bb_fn"](images_t)
                    t_patches_rep = t_patches.unsqueeze(1).expand(B, B, -1, -1).reshape(B * B, GRID * GRID, t_patches.shape[-1])
                    t_logits_flat = teacher["head"](t_patches_rep, text_rep)
                    t_logits = t_logits_flat.reshape(B, B, GRID, GRID)
                    t_cal = teacher["sb"](t_logits)
                    t_probs = torch.sigmoid(t_cal)
                # Student probs
                s_probs = torch.sigmoid(cal_logits)
                # FN-filter applies to off-diagonal — we don't distill on FN-flagged pairs
                w = weight_pair.unsqueeze(-1).unsqueeze(-1).expand_as(s_probs)
                L_distill = ((s_probs - t_probs) ** 2 * w).sum() / w.sum().clamp(min=1)

            L = L_dense + args.lambda_pool * L_pool + args.lambda_dice * L_dice + args.lambda_distill * L_distill

            opt.zero_grad(); L.backward(); opt.step()
            step += 1

            if step == 1 and args.fn_filter:
                print(f"  [fn_filter] step 1: n_fn={n_fn} of {B*(B-1)} off-diagonal pairs masked", flush=True)
            if step % args.log_every == 0:
                with torch.no_grad():
                    diag_iou = ((torch.sigmoid(diag_logits) > 0.5) & (diag_target > 0.5)).sum() / max(
                        1, ((torch.sigmoid(diag_logits) > 0.5) | (diag_target > 0.5)).sum())
                line = {
                    "step": step, "epoch": epoch, "L": float(L.item()),
                    "L_dense": float(L_dense.item()), "L_pool": float(L_pool.item()),
                    "L_dice": float(L_dice.item()),
                    "L_distill": float(L_distill.item()) if isinstance(L_distill, torch.Tensor) else 0.0,
                    "diag_iou": float(diag_iou.item()),
                    "t": float(sb.log_t.exp().item()), "bias": float(sb.bias.item()),
                    "lr": float(opt.param_groups[0]['lr']),
                    "elapsed_min": (time.time() - t_start) / 60,
                }
                print(f"  step {step:5d} | L={L:.3f} (dense={L_dense:.3f} pool={L_pool:.3f} dice={L_dice:.3f} distill={line['L_distill']:.3f}) | diag_iou={diag_iou:.3f} | t={line['t']:.1f} bias={line['bias']:.2f}",
                      flush=True)
                log_f.write(json.dumps(line) + "\n"); log_f.flush()
                if wb is not None:
                    try: wb.log(line, step=step)
                    except Exception: pass

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
    p.add_argument("--model", required=True,
                   choices=["v1", "v2-tiny", "d-mobile", "dinov2-s", "mobileclip-s2"])
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
    p.add_argument("--bce_pos_weight", type=float, default=20.0,
                   help="upweight positive patches in dense BCE; ~20-50 useful for sparse masks")
    p.add_argument("--wandb_project", default="semantic-autogaze",
                   help="W&B project name; pass empty string to disable.")
    p.add_argument("--wandb_run_name", default=None,
                   help="W&B run display name; defaults to output_dir basename.")
    p.add_argument("--distill_teacher_ckpt", default=None,
                   help="Path to teacher ckpt (e.g. DINOv2-small). Adds MSE-on-teacher-prob loss.")
    p.add_argument("--lambda_distill", type=float, default=0.5,
                   help="Weight on the teacher-MSE distillation loss term.")
    p.add_argument("--category_alpha", type=float, default=0.0,
                   help="Category-rebalance exponent for BalancedSampler. 0=uniform per-positive (Phase 1-4), 0.5=sqrt-balance, 1.0=full inverse-frequency.")
    p.add_argument("--lambda_off_diagonal", type=float, default=1.0,
                   help="weight on off-diagonal pair losses (0=non-contrastive direct regression, 1=full SigLIP)")
    p.add_argument("--fn_filter", action="store_true",
                   help="filter false-negative off-diagonal pairs (img_i actually contains query_j)"
                        " using the offline presence lookup. Critical for clean SigLIP supervision"
                        " when training on >50k pairs.")
    p.add_argument("--pool_pos_weight", type=float, default=5.0,
                   help="upweight diagonal positives in image-level pool BCE; ~B is right scale")
    p.add_argument("--positive_only", action="store_true",
                   help="filter dataset to presence=True only; ensures every diagonal slot has a real mask")
    p.add_argument("--balanced_pos_frac", type=float, default=0.7,
                   help="fraction of positive samples per epoch (0=disable, use shuffle); default 0.7")

    # Head config
    p.add_argument("--head_hidden_dim", type=int, default=384)
    p.add_argument("--head_attn_heads", type=int, default=6)
    p.add_argument("--head_attn_layers", type=int, default=2)
    p.add_argument("--head_use_spatial", action="store_true", default=True)

    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=200)
    args = p.parse_args()
    train(args)
