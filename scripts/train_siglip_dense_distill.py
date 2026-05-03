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

import cv2
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


# ---- Higher-grid head variant ----

class TextScorerHeadGrid28(TextScorerHead):
    """TextScorerHead variant whose forward output is upsampled from 14x14 to 28x28.

    Architecture is IDENTICAL to TextScorerHead (same patch_proj / pos_embed at
    14*14=196 cells / self-attention / cross-attention / score_mlp / spatial
    refiner). The ONLY difference is a final F.interpolate(..., size=28,
    mode='bilinear') applied to the (B, 1, 14, 14) score grid. This keeps the
    head's parameter count, learnable-weight set, and state_dict keys
    BIT-IDENTICAL to the parent class -- so a v0.5.0 ckpt trained at grid=14
    loads cleanly into this subclass via head.load_state_dict(rk["head"]).

    Motivation: thin-object localization (knife/skis/baseball-bat/sports-ball)
    fails at 14x14 because each cell is ~16px on a 224 input and thin objects
    span <1 cell. 28x28 halves the cell footprint to ~8px, recovering thin
    objects without retraining the head from scratch.

    Note: forward() returns the SAME shape contract as the parent
    (flattened (B, G_out*G_out)), so the calling code's
    `logits_flat.reshape(B, B, G_out, G_out)` continues to work when G_out=28.
    """

    def __init__(self, *args, grid_size_out: int = 28, **kwargs):
        # Force the inherited grid_size to 14 so positional embed / self-attn
        # / cross-attn shapes match the saved v0.5.0 head.
        kwargs["grid_size"] = GRID
        super().__init__(*args, **kwargs)
        self.grid_size_out = int(grid_size_out)

    def forward(self, patch_feats, text_emb):
        B = patch_feats.shape[0]
        # Parent returns (B, 14*14) flat logits.
        scores_flat = super().forward(patch_feats, text_emb)  # (B, 196)
        scores = scores_flat.reshape(B, 1, GRID, GRID)        # (B, 1, 14, 14)
        scores_up = F.interpolate(
            scores, size=(self.grid_size_out, self.grid_size_out),
            mode="bilinear", align_corners=False,
        )  # (B, 1, 28, 28)
        return scores_up.reshape(B, self.grid_size_out * self.grid_size_out)


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
                 build_presence_lookup: bool = False,
                 augment: bool = False,
                 augment_aggressive: bool = False,
                 grid_size_out: int = 14):
        self.target_dir = Path(target_dir)
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        # Output grid size for the per-query mask target. 14 = legacy (load
        # mask14 directly from the npz). 28 = on-the-fly resample: prefer
        # max-pool of mask_full > 0.5 (mirrors generate_clean_targets.pool_to_14
        # at higher resolution); fall back to nearest-upsample of mask14 when
        # the npz lacks mask_full (legacy files).
        self.grid_size_out = int(grid_size_out)
        # augment_aggressive supersedes augment when both are passed: the
        # aggressive pipeline is strictly stronger (it includes h-flip + color
        # jitter as its final stages).
        self.augment = augment or augment_aggressive
        self.augment_aggressive = augment_aggressive
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
        # Source per index — derived from slug prefix:
        #   pp_*    → pascal_part (body parts)
        #   stuff_* → coco-stuff / panoptic stuff
        #   lvis_*  → LVIS rare/common categories
        #   else    → COCO 80 things
        self.source_per_index = []
        for slug in self.cat_per_index:
            if slug.startswith("pp_"): self.source_per_index.append("pp")
            elif slug.startswith("stuff_"): self.source_per_index.append("stuff")
            elif slug.startswith("lvis_"): self.source_per_index.append("lvis")
            else: self.source_per_index.append("coco")
        print(f"[dataset] kept {len(self.files)} (positive={int(self.is_pos.sum())}, "
              f"negative={int((~self.is_pos).sum())}) | original positive rate {n_pos}/{len(files)}",
              flush=True)
        if build_presence_lookup:
            print(f"[dataset] presence_lookup has {len(presence_lookup)} (img_id, query) keys",
                  flush=True)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        # ------------------------------------------------------------------
        # Augmentation pipeline.
        #
        # When self.augment_aggressive is True we apply a JOINT geometric
        # transform to the (image, mask_full) pair: random h-flip, +/-20-deg
        # rotation, perspective warp with up to 5% corner displacement,
        # random uniform scale 0.7-1.5x, then center- or random-crop to
        # (image_size, image_size). After the geometric stage we re-pool
        # the transformed full-resolution mask down to 14x14 with the same
        # max-pool > 0.5 recipe used by scripts/generate_clean_targets.py
        # (`pool_to_14`). Color jitter (brightness + contrast) is applied
        # last, on the post-geometric image only.
        #
        # Why: addresses C1 + C2 in
        #   /home/ogata/mac-brain/projects/semantic-autogaze/CONCERNS.md
        # -- the head currently overfits to the canonical (centered,
        # upright, un-warped) viewpoint of COCO/LVIS crops, hurting mIoU
        # on EgoSchema / VQA-style frames where objects appear at oblique
        # angles, varying scales, and partial visibility. Aggressive
        # geometric augmentation forces viewpoint/scale invariance into
        # the head representations.
        #
        # Caveats:
        #  - Falls back to the legacy (h-flip + color-jitter) path when
        #    `mask_full` is missing from the npz, since the geometric warp
        #    needs the full-resolution mask to re-pool correctly.
        #  - If a random-scale + crop happens to drop the entire object out
        #    of frame, mask14 will be all-zero. That is fine: the trainer
        #    already handles all-zero masks (presence-style negatives),
        #    and such samples teach the head "this view does not show the
        #    queried object" -- a useful supervision signal.
        # ------------------------------------------------------------------
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

        S = self.image_size

        # Decide whether we can run the aggressive pipeline for this sample.
        # It requires `mask_full` in the npz so the joint geometric transform
        # can be re-pooled to 14x14 cleanly. If absent, fall back to legacy.
        can_aggressive = (getattr(self, "augment_aggressive", False)
                          and ("mask_full" in d.files))

        if can_aggressive:
            # --- Aggressive joint geometric transform ---
            # Work in the original image resolution so the mask warp is
            # pixel-accurate, then crop down to (S, S).
            arr_full = np.array(pil)  # (H, W, 3) uint8
            H, W = arr_full.shape[:2]
            mask_full = d["mask_full"]
            # mask_full is uint8 0/255 at (H, W). If shape disagrees with
            # the loaded image (rare -- happens when the image was
            # re-encoded since the npz was written), resize the mask to
            # the image's resolution with nearest-neighbour to preserve
            # binary semantics.
            if mask_full.shape[:2] != (H, W):
                mask_full = cv2.resize(mask_full, (W, H),
                                       interpolation=cv2.INTER_NEAREST)
            mask_full = (mask_full > 127).astype(np.uint8) * 255

            # 1) Random horizontal flip (p=0.5)
            if np.random.rand() < 0.5:
                arr_full = arr_full[:, ::-1, :].copy()
                mask_full = mask_full[:, ::-1].copy()

            # 2) Random rotation in +/-20 degrees about image center
            angle = float(np.random.uniform(-20.0, 20.0))
            R = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
            arr_full = cv2.warpAffine(arr_full, R, (W, H),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))
            mask_full = cv2.warpAffine(mask_full, R, (W, H),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)

            # 3) Perspective warp with up to 5% corner displacement
            disp = 0.05
            src_pts = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
            jitter = np.random.uniform(-disp, disp, size=(4, 2)).astype(np.float32)
            jitter[:, 0] *= W
            jitter[:, 1] *= H
            dst_pts = src_pts + jitter
            P = cv2.getPerspectiveTransform(src_pts, dst_pts)
            arr_full = cv2.warpPerspective(arr_full, P, (W, H),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=(0, 0, 0))
            mask_full = cv2.warpPerspective(mask_full, P, (W, H),
                                            flags=cv2.INTER_NEAREST,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=0)

            # 4) Random uniform scale 0.7x-1.5x then random/center crop
            #    to S x S. Anchoring the scaled canvas on S keeps the crop
            #    window well-defined regardless of the source resolution
            #    while keeping (image, mask) pixel-aligned at every step.
            scale = float(np.random.uniform(0.7, 1.5))
            new_w = max(1, int(round(S * scale)))
            new_h = max(1, int(round(S * scale)))
            arr_scaled = cv2.resize(arr_full, (new_w, new_h),
                                    interpolation=cv2.INTER_LINEAR)
            mask_scaled = cv2.resize(mask_full, (new_w, new_h),
                                     interpolation=cv2.INTER_NEAREST)

            # Pad if smaller than crop, else random-crop
            if new_h < S or new_w < S:
                pad_h = max(0, S - new_h)
                pad_w = max(0, S - new_w)
                top = pad_h // 2; bot = pad_h - top
                left = pad_w // 2; right = pad_w - left
                arr_scaled = cv2.copyMakeBorder(arr_scaled, top, bot, left, right,
                                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
                mask_scaled = cv2.copyMakeBorder(mask_scaled, top, bot, left, right,
                                                 cv2.BORDER_CONSTANT, value=0)
                new_h, new_w = arr_scaled.shape[:2]
            y0 = int(np.random.randint(0, new_h - S + 1)) if new_h > S else 0
            x0 = int(np.random.randint(0, new_w - S + 1)) if new_w > S else 0
            arr = arr_scaled[y0:y0 + S, x0:x0 + S, :].copy()
            mask_cropped = mask_scaled[y0:y0 + S, x0:x0 + S].copy()

            # 5) Re-pool post-transform mask to grid_size_out x grid_size_out
            #    with the canonical max-pool > 0.5 recipe (matches
            #    generate_clean_targets.pool_to_14 at higher resolution when
            #    grid_size_out > 14).
            mt = torch.from_numpy(mask_cropped.astype(np.float32) / 255.0)
            mt = mt.unsqueeze(0).unsqueeze(0)
            G_out = self.grid_size_out
            pooled = F.adaptive_max_pool2d(mt, (G_out, G_out)).squeeze().numpy()
            mask14 = (pooled > 0.5).astype(np.float32)

            # 6) Color jitter on the post-geometric image (same as legacy).
            if np.random.rand() < 0.5:
                b = 0.8 + 0.4 * np.random.rand()
                arr = np.clip(arr.astype(np.float32) * b, 0, 255).astype(np.uint8)
            if np.random.rand() < 0.5:
                c = 0.8 + 0.4 * np.random.rand()
                m = arr.mean(axis=(0, 1), keepdims=True)
                arr = np.clip((arr.astype(np.float32) - m) * c + m, 0, 255).astype(np.uint8)
        else:
            # --- Legacy path: simple resize, optional h-flip + color jitter. ---
            arr = np.array(pil.resize((S, S), Image.BICUBIC))
            G_out = self.grid_size_out
            if G_out == 14:
                # Direct load — the npz already stores mask14 at the canonical
                # 14x14 max-pool > 0.5 resolution.
                mask14 = d["mask14"].astype(np.float32) / 255.0  # (14, 14)
            else:
                # On-the-fly resample to G_out x G_out. Prefer mask_full +
                # max-pool > 0.5 (matches generate_clean_targets.pool_to_14
                # recipe at higher resolution). Fall back to nearest-upsample
                # of mask14 when mask_full is absent (legacy npz files).
                if "mask_full" in d.files:
                    mf = d["mask_full"].astype(np.float32)
                    if mf.max() > 1.0:
                        mf = mf / 255.0
                    mt = torch.from_numpy(mf).unsqueeze(0).unsqueeze(0)
                    pooled = F.adaptive_max_pool2d(mt, (G_out, G_out)).squeeze().numpy()
                    mask14 = (pooled > 0.5).astype(np.float32)
                else:
                    m14 = d["mask14"].astype(np.float32) / 255.0  # (14, 14)
                    mt = torch.from_numpy(m14).unsqueeze(0).unsqueeze(0)
                    up = F.interpolate(mt, size=(G_out, G_out), mode="nearest")
                    mask14 = (up.squeeze().numpy() > 0.5).astype(np.float32)
            if self.augment:
                # Horizontal flip with p=0.5
                if np.random.rand() < 0.5:
                    arr = arr[:, ::-1, :].copy()
                    mask14 = mask14[:, ::-1].copy()
                # Color jitter: gentle brightness/contrast/saturation
                if np.random.rand() < 0.5:
                    # brightness in [0.8, 1.2]
                    b = 0.8 + 0.4 * np.random.rand()
                    arr = np.clip(arr.astype(np.float32) * b, 0, 255).astype(np.uint8)
                if np.random.rand() < 0.5:
                    # contrast in [0.8, 1.2]
                    c = 0.8 + 0.4 * np.random.rand()
                    m = arr.mean(axis=(0,1), keepdims=True)
                    arr = np.clip((arr.astype(np.float32) - m) * c + m, 0, 255).astype(np.uint8)

        x = (arr.astype(np.float32) / 255.0 - self.mean) / self.std
        x = torch.from_numpy(x).permute(2, 0, 1).float()
        mask14 = torch.from_numpy(np.ascontiguousarray(mask14)).float()
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
                 cat_per_index=None, category_alpha=0.0,
                 source_per_index=None, source_weights=None):
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
        # Source-based weighting: e.g. {"pp": 5.0, "stuff": 1.0, "lvis": 1.0, "coco": 1.0}
        # Multiplies the existing weight by source_weights[source]; then renormalizes.
        if source_weights and source_per_index is not None and len(self.pos) > 0:
            src = [source_per_index[i] for i in self.pos.tolist()]
            mult = np.array([float(source_weights.get(s, 1.0)) for s in src])
            base = self.pos_weights if self.pos_weights is not None else np.ones(len(self.pos))
            w = base * mult
            self.pos_weights = w / w.sum()
            from collections import Counter
            sc = Counter(src)
            print(f"[sampler] source-weighted positives: {dict(sc)} → weights {source_weights}",
                  flush=True)

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
    if model in ("fastvit-t8", "mobilevit-xs", "convnext-atto", "convnext-femto",
                  "convnext-pico", "convnext-nano", "convnext-tiny",
                  "repvit-m1", "efficientformerv2-s0"):
        # Small/mobile architectures from timm. All output 4D spatial features
        # which we interpolate to GRID×GRID.
        timm_name_map = {
            "fastvit-t8":          "fastvit_t8.apple_in1k",
            "mobilevit-xs":        "mobilevit_xs.cvnets_in1k",
            "convnext-atto":       "convnext_atto.d2_in1k",
            "convnext-femto":      "convnext_femto.d1_in1k",
            "convnext-pico":       "convnext_pico.d1_in1k",
            "convnext-nano":       "convnext_nano.in12k_ft_in1k",
            "convnext-tiny":       "convnext_tiny.in12k_ft_in1k",
            "repvit-m1":           "repvit_m1.dist_in1k",
            "efficientformerv2-s0": "efficientformerv2_s0.snap_dist_in1k",
        }
        bb = timm.create_model(timm_name_map[model], pretrained=True,
                               num_classes=0).to(device)
        for p in bb.parameters(): p.requires_grad_(False)
        # Probe shape
        # Most of these expect 224 or 256
        in_size = bb.default_cfg.get("input_size", (3, 224, 224))[1]
        # Always use 224 to keep the rest of the pipeline consistent (dataset resizes to 224)
        with torch.no_grad():
            x_probe = torch.randn(1, 3, 224, 224, device=device)
            try:
                feat_probe = bb.forward_features(x_probe)
            except Exception:
                # Some models need exact native size
                bb_native = timm.create_model(timm_name_map[model], pretrained=True,
                                              num_classes=0, img_size=in_size).to(device)
                bb = bb_native
                for p in bb.parameters(): p.requires_grad_(False)
                x_probe = torch.randn(1, 3, in_size, in_size, device=device)
                feat_probe = bb.forward_features(x_probe)
        embed_dim = feat_probe.shape[1] if feat_probe.dim() == 4 else feat_probe.shape[-1]
        print(f"[backbone] {model}: timm={timm_name_map[model]} feat_shape={tuple(feat_probe.shape)} → embed_dim={embed_dim}", flush=True)
        def fn(x):
            f = bb.forward_features(x)
            if f.dim() == 4:
                f = F.interpolate(f, size=(GRID, GRID), mode="bilinear", align_corners=False)
                return f.permute(0, 2, 3, 1).reshape(f.shape[0], GRID * GRID, f.shape[1])
            elif f.dim() == 3:
                # (B, N, D) — drop CLS if present, interpolate spatial
                if f.shape[1] == 197:  # 14x14 + cls
                    return f[:, 1:, :]
                side = int(f.shape[1] ** 0.5)
                if side * side == f.shape[1]:
                    B, N, D = f.shape
                    g = f.permute(0, 2, 1).reshape(B, D, side, side)
                    g = F.interpolate(g, size=(GRID, GRID), mode="bilinear", align_corners=False)
                    return g.permute(0, 2, 3, 1).reshape(B, GRID * GRID, D)
                return f[:, :GRID*GRID, :]  # last resort
            raise RuntimeError(f"unexpected feat shape: {f.shape}")
        return fn, embed_dim, IM_MEAN, IM_STD, "timm-mobile-hybrid", bb
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

class ObjectnessHead(nn.Module):
    """OWLv2-style query-agnostic objectness head.

    Predicts a per-cell scalar "is there ANY queryable object in this image
    cell?" — independent of any text query. At inference, the per-query dense
    score can be gated by sigmoid(objectness) to suppress activations on
    background cells (ceiling, sky, blurred background, etc.) that share
    scene-context features with positives but contain no actual object.

    Architecture mirrors the prefix of TextScorerHead: a small patch projector,
    a learned positional embedding, and 1-2 layers of self-attention over
    patches, followed by a Linear(hidden, 1). The last Linear's bias is
    initialised to `bias_init` (default -2.0) so that at step 0 the head emits
    sigmoid(-2) ~= 0.12 everywhere — a low constant prior rather than a noisy
    output.

    Input:  patch_feats (B, 196, patch_dim)
    Output: per-cell objectness logits (B, 14, 14)
    """
    def __init__(self, patch_dim, hidden_dim=256, n_attn_heads=4,
                 n_attn_layers=1, grid_size=GRID, bias_init=-2.0):
        super().__init__()
        self.grid_size = grid_size
        self.bias_init = float(bias_init)

        self.patch_proj = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, grid_size * grid_size, hidden_dim) * 0.02
        )

        self.self_attn_layers = nn.ModuleList()
        for _ in range(n_attn_layers):
            self.self_attn_layers.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(hidden_dim, n_attn_heads, batch_first=True),
                "norm1": nn.LayerNorm(hidden_dim),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                ),
                "norm2": nn.LayerNorm(hidden_dim),
            }))

        self.score = nn.Linear(hidden_dim, 1)
        # Init last Linear so initial output == bias_init for any input —
        # head emits sigmoid(bias_init) everywhere at step 0.
        with torch.no_grad():
            self.score.weight.zero_()
            self.score.bias.fill_(self.bias_init)

    def forward(self, patch_feats):
        B = patch_feats.shape[0]
        G = self.grid_size
        x = self.patch_proj(patch_feats)  # (B, 196, hidden)
        x = x + self.pos_embed
        for layer in self.self_attn_layers:
            r = x; x = layer["norm1"](x)
            x_a, _ = layer["attn"](x, x, x); x = r + x_a
            r = x; x = layer["norm2"](x); x = r + layer["ffn"](x)
        logits = self.score(x).squeeze(-1)  # (B, 196)
        return logits.reshape(B, G, G)


class SiglipBias(nn.Module):
    """Learnable temperature + bias used in SigLIP-style sigmoid contrastive."""
    def __init__(self, t_init=10.0, bias_init=-4.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(float(np.log(t_init))))
        self.bias = nn.Parameter(torch.tensor(float(bias_init)))

    def forward(self, cosine):
        return cosine * self.log_t.exp() + self.bias


class SiglipBiasPerQuery(nn.Module):
    """Per-query learnable bias: one global temperature + a small MLP that
    maps each query's text_emb (512-d) -> a scalar bias. Lets the head shift
    each text query to a common operating point (calibration fix).

    forward(logits, text_emb):
        logits   : (B, B, H, W)  -- pair grid; first B is image, second B is query
        text_emb : (B, 512)      -- per-query text embeddings
    Returns:
        cal_logits : (B, B, H, W) where
            cal_logits[i, j, :, :] = t * logits[i, j, :, :] + b(text_emb_j)
    """
    def __init__(self, t_init=10.0, bias_init=-4.0, text_dim=512, hidden=64):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(float(np.log(t_init))))
        self.bias_init = float(bias_init)
        self.bias_mlp = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Initialize last Linear so initial output == bias_init for any input.
        # (weight ~= 0, bias = bias_init -> module reproduces global-bias behavior at step 0.)
        with torch.no_grad():
            self.bias_mlp[-1].weight.zero_()
            self.bias_mlp[-1].bias.fill_(self.bias_init)

    def per_query_bias(self, text_emb):
        """text_emb: (B, 512) -> (B,) scalar bias per query."""
        return self.bias_mlp(text_emb).squeeze(-1)

    def forward(self, logits, text_emb):
        # b_q: (B,) one bias per query (the "j" / second-B / query axis of logits)
        b_q = self.per_query_bias(text_emb)
        # Broadcast: bias is constant across the image dim (first B) and across (H, W);
        # only varies along the query dim. Shape (1, B, 1, 1) handles that for (B, B, H, W).
        b_bcast = b_q.view(1, -1, 1, 1)
        return logits * self.log_t.exp() + b_bcast


class SiglipBiasPerQueryLinear(nn.Module):
    """Per-query bias as a single Linear(text_dim -> 1) (no hidden, no nonlinearity).
    Ablation of SiglipBiasPerQuery's MLP — tests whether the GELU + 64-d hidden
    layer is load-bearing or whether per-query bias is captured by a linear
    projection of text_emb. Same forward semantics as SiglipBiasPerQuery."""
    def __init__(self, t_init=10.0, bias_init=-4.0, text_dim=512):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(float(np.log(t_init))))
        self.bias_init = float(bias_init)
        self.bias_linear = nn.Linear(text_dim, 1)
        # init: weight = 0, bias = bias_init -> step-0 output equals global bias.
        with torch.no_grad():
            self.bias_linear.weight.zero_()
            self.bias_linear.bias.fill_(self.bias_init)

    def per_query_bias(self, text_emb):
        return self.bias_linear(text_emb).squeeze(-1)

    def forward(self, logits, text_emb):
        b_q = self.per_query_bias(text_emb)
        b_bcast = b_q.view(1, -1, 1, 1)
        return logits * self.log_t.exp() + b_bcast


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
    if args.grid_size_out == GRID:
        head = TextScorerHead(
            patch_dim=patch_dim, text_dim=512,
            hidden_dim=args.head_hidden_dim, n_attn_heads=args.head_attn_heads,
            n_attn_layers=args.head_attn_layers, grid_size=GRID,
            use_spatial=args.head_use_spatial,
        ).to(device)
    else:
        # Higher-grid head: identical params to TextScorerHead (positional
        # embed still 14*14=196), but forward() upsamples logits to
        # grid_size_out via bilinear. State_dict keys/shapes are bit-identical
        # to TextScorerHead, so a v0.5.0 ckpt loads cleanly below.
        head = TextScorerHeadGrid28(
            patch_dim=patch_dim, text_dim=512,
            hidden_dim=args.head_hidden_dim, n_attn_heads=args.head_attn_heads,
            n_attn_layers=args.head_attn_layers,
            use_spatial=args.head_use_spatial,
            grid_size_out=args.grid_size_out,
        ).to(device)
        print(f"[head] TextScorerHeadGrid28 (internal grid={GRID}, "
              f"output upsampled to {args.grid_size_out}x{args.grid_size_out} "
              f"via bilinear; same params as TextScorerHead)", flush=True)
    head.train()

    # SigLIP bias / temperature
    if args.per_query_bias:
        if args.per_query_bias_kind == "linear":
            sb = SiglipBiasPerQueryLinear(t_init=args.t_init, bias_init=args.bias_init).to(device)
            print(f"[sb] using SiglipBiasPerQueryLinear (Linear 512->1, init bias={args.bias_init})", flush=True)
        else:
            sb = SiglipBiasPerQuery(t_init=args.t_init, bias_init=args.bias_init).to(device)
            print(f"[sb] using SiglipBiasPerQuery (MLP 512->64->1, init bias={args.bias_init})", flush=True)
    else:
        sb = SiglipBias(t_init=args.t_init, bias_init=args.bias_init).to(device)

    # Optional: OWLv2-style query-agnostic objectness head. Only constructed
    # when --objectness_weight > 0; saves params + compute otherwise.
    objectness_head = None
    if args.objectness_weight > 0.0:
        objectness_head = ObjectnessHead(patch_dim=patch_dim).to(device)
        objectness_head.train()
        print(f"[objectness] head constructed (weight={args.objectness_weight}, "
              f"pos_weight={args.objectness_pos_weight}, bias_init=-2.0)", flush=True)

    # Optimizer
    params = [p for p in head.parameters()] + [p for p in sb.parameters()]
    if objectness_head is not None:
        params += [p for p in objectness_head.parameters()]
    if args.finetune_backbone_blocks > 0:
        params += [p for p in bb_module.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    n_train_params = sum(p.numel() for p in params)
    print(f"[opt] {n_train_params/1e6:.2f} M trainable params (lr={args.lr})", flush=True)

    # Optional: resume head + sb (and backbone if finetuning) from an existing ckpt.
    if args.resume_from:
        rk = torch.load(args.resume_from, map_location=device, weights_only=False)
        # TextScorerHeadGrid28 sets the parent's grid_size to 14 in __init__,
        # so its state_dict keys/shapes are bit-identical to a v0.5.0
        # TextScorerHead trained at grid=14. The bilinear upsample at the
        # tail has zero learnable parameters, so this load_state_dict is
        # always strict-clean even when args.grid_size_out == 28.
        head.load_state_dict(rk["head"])
        # If we're using SiglipBiasPerQuery but the resume ckpt was saved from
        # global-bias SiglipBias, the state_dict keys/shapes don't match
        # (old: "log_t", "bias"; new: "log_t", "bias_mlp.0.weight", ...).
        # Fall back to loading just the temperature; the MLP starts fresh, but
        # was initialized so its constant output == args.bias_init, matching
        # the old global-bias operating point.
        try:
            sb.load_state_dict(rk["sb"])
            print(f"[resume] loaded head+sb from {args.resume_from}", flush=True)
        except (RuntimeError, KeyError) as _e:
            print(f"[resume][warn] sb state_dict mismatch ({_e!r}); "
                  f"loading log_t only and initializing MLP fresh "
                  f"(bias_init={args.bias_init} matches old global bias).", flush=True)
            old_sb = rk.get("sb", {})
            if isinstance(old_sb, dict) and "log_t" in old_sb and hasattr(sb, "log_t"):
                with torch.no_grad():
                    sb.log_t.copy_(old_sb["log_t"].to(sb.log_t.device))
            print(f"[resume] loaded head from {args.resume_from} (sb partial)", flush=True)
        if args.finetune_backbone_blocks > 0 and "backbone_state" in rk:
            bb_module.load_state_dict(rk["backbone_state"])
        # Optionally resume the objectness head. Old (pre-phase21) ckpts have
        # no "obj" key -> init fresh; phase21+ ckpts do -> load. Either is
        # fine; never crash.
        if objectness_head is not None:
            try:
                if "obj" in rk and rk["obj"] is not None:
                    objectness_head.load_state_dict(rk["obj"])
                    print(f"[resume] loaded objectness head from {args.resume_from}", flush=True)
                else:
                    print(f"[resume] no 'obj' key in ckpt; objectness head starts fresh", flush=True)
            except (RuntimeError, KeyError) as _e:
                print(f"[resume][warn] objectness head load failed ({_e!r}); starting fresh",
                      flush=True)

    # Data
    ds = TargetDataset(args.target_dir, args.image_dir,
                       image_size=args.image_size, mean=mean, std=std,
                       limit=args.limit, positive_only=args.positive_only,
                       build_presence_lookup=args.fn_filter,
                       augment=args.augment,
                       augment_aggressive=args.augment_aggressive,
                       grid_size_out=args.grid_size_out)

    # ---- Train/val split by image_id ----
    # Hold out fraction of unique image_ids so val images are NEVER seen in training pairs.
    # This is the right unit because the dataset is (image, query) pairs with images shared.
    train_ds = ds
    val_ds = None
    if args.val_split_frac > 0:
        unique_imgs = sorted({f.stem.partition("__")[0] for f in ds.files})
        rng_split = np.random.default_rng(2026)
        rng_split.shuffle(unique_imgs)
        n_val = int(len(unique_imgs) * args.val_split_frac)
        val_imgs = set(unique_imgs[:n_val])
        train_mask = np.array([f.stem.partition("__")[0] not in val_imgs for f in ds.files])
        val_mask = ~train_mask
        # Build train_ds and val_ds as Subset-style filtered indexers without
        # re-creating the dataset (saves another scan).
        # Simpler: replace ds.files in-place to train-only, expose val files separately.
        all_files = ds.files; all_is_pos = ds.is_pos; all_cat = ds.cat_per_index; all_src = ds.source_per_index
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        ds.files = [all_files[i] for i in train_idx]
        ds.is_pos = all_is_pos[train_idx]
        ds.pos_indices = np.where(ds.is_pos)[0]
        ds.neg_indices = np.where(~ds.is_pos)[0]
        ds.cat_per_index = [all_cat[i] for i in train_idx]
        ds.source_per_index = [all_src[i] for i in train_idx]
        # Build a lightweight val_ds sharing presence_lookup but with val files only
        class _ValSubset(TargetDataset):
            def __init__(self, parent, idxs):
                self.target_dir = parent.target_dir
                self.image_dir = parent.image_dir
                self.image_size = parent.image_size
                self.mean = parent.mean
                self.std = parent.std
                self.augment = False  # never augment val
                self.augment_aggressive = False  # never augment val (gates aggressive pipeline too)
                self.grid_size_out = parent.grid_size_out
                self.files = [all_files[i] for i in idxs]
                self.is_pos = all_is_pos[idxs]
                self.pos_indices = np.where(self.is_pos)[0]
                self.neg_indices = np.where(~self.is_pos)[0]
                self.cat_per_index = [all_cat[i] for i in idxs]
                self.source_per_index = [all_src[i] for i in idxs]
                self.presence_lookup = parent.presence_lookup
            def __len__(self): return len(self.files)
            __getitem__ = TargetDataset.__getitem__
        val_ds = _ValSubset(ds, val_idx)
        print(f"[split] val_frac={args.val_split_frac:.2f} → train_imgs={len(unique_imgs)-n_val} val_imgs={n_val}",
              flush=True)
        print(f"  train pairs={len(ds.files)} (pos={int(ds.is_pos.sum())})  val pairs={len(val_ds.files)} (pos={int(val_ds.is_pos.sum())})",
              flush=True)
    train_ds = ds  # update reference
    if args.balanced_pos_frac > 0:
        # Parse --source_weights "pp:5,stuff:1,lvis:1,coco:1" → dict
        source_w = None
        if args.source_weights:
            source_w = {}
            for kv in args.source_weights.split(","):
                k, _, v = kv.partition(":")
                source_w[k.strip()] = float(v.strip())
        sampler = BalancedSampler(ds.pos_indices, ds.neg_indices,
                                  pos_frac=args.balanced_pos_frac,
                                  cat_per_index=ds.cat_per_index,
                                  category_alpha=args.category_alpha,
                                  source_per_index=ds.source_per_index,
                                  source_weights=source_w)
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

    # Pick a fixed B-positive validation batch for matrix logging (deterministic seed)
    val_batch = None
    if wb is not None and args.wandb_log_matrix_every > 0 and len(ds.pos_indices) > 0:
        rng_val = np.random.default_rng(2026)
        # Sample B distinct positive indices with distinct queries + images
        seen_q, seen_i, chosen_idx = set(), set(), []
        perm = rng_val.permutation(ds.pos_indices)
        for ix in perm:
            item = ds[int(ix)]
            if item is None: continue
            if item["query"] in seen_q or item["img_id"] in seen_i: continue
            seen_q.add(item["query"]); seen_i.add(item["img_id"])
            chosen_idx.append(int(ix))
            if len(chosen_idx) >= args.wandb_matrix_batch_size: break
        if len(chosen_idx) >= 2:
            B_val = len(chosen_idx)
            items = [ds[int(i)] for i in chosen_idx]
            val_batch = {
                "images": torch.stack([it["image"] for it in items]).to(device),
                "queries": [it["query"] for it in items],
                "img_ids": [it["img_id"] for it in items],
                "slugs": [it["query_slug"] for it in items],
                "diag_full": [],  # filled below
                "raw_pil": [],
                "B": B_val,
            }
            from PIL import Image as PILImage
            for it in items:
                # Reload mask_full at full resolution for crisp contour
                npz_p = ds.target_dir / f"{it['img_id']}__{it['query_slug']}.npz"
                d_full = np.load(npz_p, allow_pickle=False)
                m = d_full["mask_full"].astype(np.float32)
                if m.max() > 1: m /= 255.0
                m224 = np.array(PILImage.fromarray((m * 255).astype(np.uint8))
                                .resize((224, 224), PILImage.NEAREST)) / 255.0
                val_batch["diag_full"].append(m224)
                # Cache the input image as displayable RGB
                arr = it["image"].cpu().numpy().transpose(1, 2, 0)
                arr = arr * np.array(std) + np.array(mean)
                arr = np.clip(arr, 0, 1)
                val_batch["raw_pil"].append(arr)
            print(f"[wandb-matrix] cached B={B_val} validation batch (seed=2026)", flush=True)

    def render_and_log_matrix(step_n):
        """Render BxB cross-pair prediction matrix and log to wandb."""
        if wb is None or val_batch is None: return
        import matplotlib.pyplot as _plt
        import io
        head.eval()
        # Head's output grid (14 by default, 28 with --grid_size_out 28).
        G_render = int(args.grid_size_out)
        try:
            with torch.no_grad():
                B_v = val_batch["B"]
                v_patches = bb_fn(val_batch["images"])
                # Build text embs for the val queries
                v_text = encode_text_batch(clip_model_text, clip_tok, val_batch["queries"], device)
                pred = np.zeros((B_v, B_v, G_render, G_render), dtype=np.float32)
                for i in range(B_v):
                    for j in range(B_v):
                        logits = head(v_patches[i:i+1], v_text[j:j+1]).reshape(G_render, G_render)
                        if args.per_query_bias:
                            # SiglipBiasPerQuery expects (B, B, H, W) + text_emb (B, 512).
                            cal = sb(logits.view(1, 1, G_render, G_render),
                                     v_text[j:j+1]).view(G_render, G_render)
                        else:
                            cal = sb(logits)
                        pred[i, j] = torch.sigmoid(cal).cpu().numpy()
                # FN-flagged off-diag pairs
                fn_flags = np.zeros((B_v, B_v), dtype=bool)
                for i in range(B_v):
                    for j in range(B_v):
                        if i == j: continue
                        if ds.presence_lookup.get((val_batch["img_ids"][i], val_batch["slugs"][j]), False):
                            fn_flags[i, j] = True
                # Render
                fig = _plt.figure(figsize=(2.0 * (B_v + 1), 2.0 * (B_v + 1)))
                gs = fig.add_gridspec(B_v + 1, B_v + 1, hspace=0.05, wspace=0.05)
                # Column headers
                for j in range(B_v):
                    ax = fig.add_subplot(gs[0, j + 1])
                    ax.imshow(val_batch["raw_pil"][j])
                    ax.contour(val_batch["diag_full"][j], levels=[0.5], colors="lime", linewidths=1.0)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(f"q{j}: {val_batch['queries'][j][:14]}", fontsize=7)
                ax = fig.add_subplot(gs[0, 0])
                ax.text(0.5, 0.5, f"step {step_n}", ha="center", va="center", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                # Body
                for i in range(B_v):
                    ax_l = fig.add_subplot(gs[i + 1, 0])
                    ax_l.imshow(val_batch["raw_pil"][i])
                    ax_l.contour(val_batch["diag_full"][i], levels=[0.5], colors="lime", linewidths=1.0)
                    ax_l.set_xticks([]); ax_l.set_yticks([])
                    ax_l.set_title(f"i{i}: {val_batch['queries'][i][:14]}", fontsize=7)
                    arr_i = val_batch["raw_pil"][i]
                    for j in range(B_v):
                        ax_c = fig.add_subplot(gs[i + 1, j + 1])
                        ax_c.imshow(arr_i, alpha=0.55)
                        h = pred[i, j]
                        h_up = np.kron(h, np.ones((arr_i.shape[0] // G_render + 1,
                                                    arr_i.shape[1] // G_render + 1)))[:arr_i.shape[0], :arr_i.shape[1]]
                        ax_c.imshow(h_up, alpha=0.6, cmap="hot", vmin=0, vmax=1)
                        ax_c.set_xticks([]); ax_c.set_yticks([])
                        if i == j:
                            border = "limegreen"
                        elif fn_flags[i, j]:
                            border = "red"
                        else:
                            border = "dimgray"
                        for s_e in ax_c.spines.values():
                            s_e.set_visible(True); s_e.set_edgecolor(border); s_e.set_linewidth(2.0)
                        ax_c.text(0.03, 0.97, f"{h.max():.2f}", color="white", fontsize=6,
                                  transform=ax_c.transAxes, va="top",
                                  bbox=dict(facecolor="black", alpha=0.55, pad=1, edgecolor="none"))
                _plt.tight_layout()
                buf = io.BytesIO()
                _plt.savefig(buf, dpi=90, bbox_inches="tight"); _plt.close(fig)
                buf.seek(0)
                import wandb as _wb
                wb.log({"validation_matrix": _wb.Image(PILImage.open(buf))}, step=step_n)
        except Exception as _e:
            print(f"[wandb-matrix] render failed: {_e}", flush=True)
        finally:
            head.train()

    step = 0
    best_loss = float("inf")
    best_val_iou = -1.0
    latest_val_iou = None
    t_start = time.time()
    # Per-query head output spatial side. The backbone still produces GRID*GRID
    # patch tokens (==196); only the head's output grid is variable.
    G_out = int(args.grid_size_out)
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
            # We construct the (B, Q, G_out*G_out) heatmap as: for each (b, q), run head(patches[b], text_embs[q]).
            # Vectorize via broadcasting: replicate patches B times, replicate text Q times, run.
            # NOTE: patches always have GRID*GRID=196 tokens (backbone-side spatial
            # side is fixed at 14). Only the head OUTPUT grid varies with G_out.
            patches_rep = patches.unsqueeze(1).expand(B, B, -1, -1).reshape(B * B, GRID * GRID, patch_dim)
            text_rep = text_embs.unsqueeze(0).expand(B, B, -1).reshape(B * B, -1)
            logits_flat = head(patches_rep, text_rep)  # (B*B, G_out*G_out)
            logits = logits_flat.reshape(B, B, G_out, G_out)  # (B, Q, H, W)

            # 4) Build target: (B, Q, G_out, G_out)
            #    On-diagonal: mask14[b]  (which is already at G_out resolution
            #                              when args.grid_size_out != 14;
            #                              the dataset resamples on-the-fly.)
            #    Off-diagonal: zeros
            target = torch.zeros(B, B, G_out, G_out, device=device)
            for b in range(B):
                target[b, b] = mask14[b]
            target_present = torch.zeros(B, B, device=device)
            for b in range(B):
                target_present[b, b] = presence[b]

            # 5) SigLIP bias on logits (treats as cosine-like; logits ARE arbitrary-scale
            # already from the head, so we apply learnable t/bias as a calibration)
            if args.per_query_bias:
                # Per-query bias: each query j gets its own bias from a small MLP
                # over text_embs[j]. Bias broadcasts across the image dim and (H, W).
                cal_logits = sb(logits, text_embs)
            else:
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

            # 8b) L_hard_neg: hard-negative mining on positive (on-diagonal) pairs.
            # The standard L_dense weighs positive cells `pos_weight` (default 30) higher
            # than negative cells; that catches thin/small objects but makes false
            # positives "cheap." On categories with strong scene context (snowboard, knife,
            # skateboard), the head learns to fire broadly when the text matches the scene
            # rather than the object. To fight this, on each positive (on-diagonal) sample
            # we mine the top-K negative cells by predicted score (where mask14 < 0.5) and
            # add an extra BCE-toward-zero penalty with weight `hard_neg_weight`.
            if args.hard_neg_weight > 0:
                # Per-sample hard negatives within the diagonal pairs only.
                # diag_logits / diag_target shape: (B, H, W) for the diagonal queries.
                B_d, H_d, W_d = diag_logits.shape
                diag_pres_mask = presence > 0.5  # only mine on present-class pairs
                if diag_pres_mask.any():
                    sel_logits = diag_logits[diag_pres_mask]
                    sel_target = diag_target[diag_pres_mask]
                    # Negative cells = mask14 < 0.5 within positive samples.
                    is_neg = (sel_target < 0.5).float()  # (B', H, W)
                    # Predicted prob, cells outside negatives masked to -inf so they
                    # don't get picked as "hard negatives".
                    pred_prob = torch.sigmoid(sel_logits)
                    masked_score = pred_prob * is_neg + (1 - is_neg) * (-1e9)
                    # Top-K hard negatives per sample.
                    k = max(1, int(args.hard_neg_topk))
                    flat = masked_score.view(masked_score.shape[0], -1)
                    # Cells available (>=0) — exclude positives (now -1e9) and dont overshoot
                    n_neg_per = is_neg.view(is_neg.shape[0], -1).sum(dim=1).clamp(min=1)
                    k_per = torch.minimum(
                        torch.full_like(n_neg_per, float(k)), n_neg_per
                    ).long()
                    # Pick top-k indices per sample; we pad k uniformly with the global k
                    # then mask by k_per. Simpler: take top-k everywhere; the mask is_neg
                    # already excluded positives.
                    topk_vals, topk_idx = flat.topk(k=k, dim=1)
                    # Gather logits at topk_idx
                    sel_logits_flat = sel_logits.view(sel_logits.shape[0], -1)
                    hn_logits = torch.gather(sel_logits_flat, 1, topk_idx)
                    hn_target = torch.zeros_like(hn_logits)
                    L_hard_neg = F.binary_cross_entropy_with_logits(
                        hn_logits, hn_target, reduction="mean"
                    )
                else:
                    L_hard_neg = torch.tensor(0.0, device=device)
            else:
                L_hard_neg = torch.tensor(0.0, device=device)

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
                    # Teacher is always a v0.5.0-style head at GRID=14. If the
                    # student outputs at G_out>14, bilinearly upsample teacher
                    # logits to match before MSE.
                    t_logits = t_logits_flat.reshape(B, B, GRID, GRID)
                    t_cal = teacher["sb"](t_logits)
                    if G_out != GRID:
                        t_cal = F.interpolate(
                            t_cal.reshape(B * B, 1, GRID, GRID),
                            size=(G_out, G_out), mode="bilinear",
                            align_corners=False,
                        ).reshape(B, B, G_out, G_out)
                    t_probs = torch.sigmoid(t_cal)
                # Student probs
                s_probs = torch.sigmoid(cal_logits)
                # FN-filter applies to off-diagonal — we don't distill on FN-flagged pairs
                w = weight_pair.unsqueeze(-1).unsqueeze(-1).expand_as(s_probs)
                L_distill = ((s_probs - t_probs) ** 2 * w).sum() / w.sum().clamp(min=1)

            # 9b) L_calib: clamp the worst false-positive cell on clean off-diagonal
            # (absent) pairs. For each (i, j) with i != j AND fn_keep[i,j] == 1, take
            # m_ij = sigmoid(cal_logits[i,j]).amax() over (H, W) — the per-pair worst
            # cell — and penalize relu(m_ij - tau)^2 so small overshoots are mild and
            # large overshoots are heavy. Avg over the kept off-diagonal pairs only.
            # On-diagonal pairs are excluded; they are supervised by L_dense / L_dice.
            if args.lambda_calib > 0:
                offdiag_mask = (1.0 - diag_mask) * fn_keep  # (B, B), 1 on clean absent pairs
                if offdiag_mask.sum() > 0:
                    pair_max = torch.sigmoid(cal_logits).amax(dim=(-2, -1))  # (B, Q)
                    overshoot = F.relu(pair_max - args.calib_target_max)
                    L_calib = (overshoot.pow(2) * offdiag_mask).sum() / offdiag_mask.sum().clamp(min=1)
                else:
                    L_calib = torch.tensor(0.0, device=device)
            else:
                L_calib = torch.tensor(0.0, device=device)

            # 9d) L_bias_var: penalizes variance of per-query biases across the
            # batch when --per_query_bias is on. Reduces the "tennis racket
            # query → MLP learns +0.5 bias → fires everywhere" overshoot
            # observed qualitatively in v0.6.0 review. Encourages the MLP to
            # only deviate from the mean bias when the query genuinely needs it.
            # No-op when per_query_bias is off or lambda is 0.
            if args.lambda_bias_var > 0 and args.per_query_bias:
                with torch.no_grad():
                    pass
                # text_embs is (B, 512), already encoded above for L_dense.
                b_q_train = sb.per_query_bias(text_embs)  # (B,)
                if b_q_train.numel() > 1:
                    L_bias_var = b_q_train.var(unbiased=False)
                else:
                    L_bias_var = torch.tensor(0.0, device=device)
            else:
                L_bias_var = torch.tensor(0.0, device=device)

            # 9c) L_objectness: query-agnostic per-cell "is there any object?" head.
            # Supervises objectness_head(patches) -> (B, 14, 14) against the on-
            # diagonal mask14[b]: cells positive for at least one query in this
            # batch are treated as "object-here". Disabled (zero tensor) when
            # --objectness_weight == 0 (head is also not constructed).
            if objectness_head is not None:
                obj_logits = objectness_head(patches)  # (B, 14, 14)
                obj_pos_weight = torch.tensor(args.objectness_pos_weight, device=device)
                L_objectness = F.binary_cross_entropy_with_logits(
                    obj_logits, mask14, reduction="mean", pos_weight=obj_pos_weight)
            else:
                L_objectness = torch.tensor(0.0, device=device)

            L = L_dense + args.lambda_pool * L_pool + args.lambda_dice * L_dice + args.lambda_distill * L_distill + args.hard_neg_weight * L_hard_neg + args.lambda_calib * L_calib + args.objectness_weight * L_objectness + args.lambda_bias_var * L_bias_var

            opt.zero_grad(); L.backward(); opt.step()
            step += 1

            if step == 1 and args.fn_filter:
                print(f"  [fn_filter] step 1: n_fn={n_fn} of {B*(B-1)} off-diagonal pairs masked", flush=True)
            if step % args.log_every == 0:
                with torch.no_grad():
                    diag_iou = ((torch.sigmoid(diag_logits) > 0.5) & (diag_target > 0.5)).sum() / max(
                        1, ((torch.sigmoid(diag_logits) > 0.5) | (diag_target > 0.5)).sum())
                    # Per-query bias stats: mean/std of the bias across the batch's queries.
                    # For global bias we report bias_mean = scalar bias, bias_std = 0.
                    if args.per_query_bias:
                        b_q_now = sb.per_query_bias(text_embs).detach()  # (B,)
                        b_mean = float(b_q_now.mean().item())
                        b_std = float(b_q_now.std(unbiased=False).item()) if b_q_now.numel() > 1 else 0.0
                    else:
                        b_mean = float(sb.bias.item())
                        b_std = 0.0
                line = {
                    "step": step, "epoch": epoch, "L": float(L.item()),
                    "L_dense": float(L_dense.item()), "L_pool": float(L_pool.item()),
                    "L_dice": float(L_dice.item()),
                    "L_hard_neg": float(L_hard_neg.item()) if isinstance(L_hard_neg, torch.Tensor) else 0.0,
                    "L_distill": float(L_distill.item()) if isinstance(L_distill, torch.Tensor) else 0.0,
                    "L_calib": float(L_calib.item()) if isinstance(L_calib, torch.Tensor) else 0.0,
                    "L_bias_var": float(L_bias_var.item()) if isinstance(L_bias_var, torch.Tensor) else 0.0,
                    "L_objectness": float(L_objectness.item()) if isinstance(L_objectness, torch.Tensor) else 0.0,
                    "diag_iou": float(diag_iou.item()),
                    "t": float(sb.log_t.exp().item()),
                    "bias": b_mean,
                    "b_mean": b_mean,
                    "b_std": b_std,
                    "lr": float(opt.param_groups[0]['lr']),
                    "elapsed_min": (time.time() - t_start) / 60,
                }
                print(f"  step {step:5d} | L={L:.3f} (dense={L_dense:.3f} pool={L_pool:.3f} dice={L_dice:.3f} hn={line['L_hard_neg']:.3f} distill={line['L_distill']:.3f}) | diag_iou={diag_iou:.3f} | t={line['t']:.1f} b_mean={b_mean:.2f} b_std={b_std:.2f}",
                      flush=True)
                log_f.write(json.dumps(line) + "\n"); log_f.flush()
                if wb is not None:
                    try: wb.log(line, step=step)
                    except Exception: pass

            # Periodic validation matrix to wandb
            if (wb is not None and args.wandb_log_matrix_every > 0
                and step % args.wandb_log_matrix_every == 0):
                render_and_log_matrix(step)

            # Periodic val-set eval (held-out images)
            if val_ds is not None and args.val_eval_every > 0 and step % args.val_eval_every == 0:
                head.eval()
                with torch.no_grad():
                    # Build a quick val DataLoader on the fly, capped at val_max_batches
                    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=2, collate_fn=collate, drop_last=True)
                    iou_sum, iou_n = 0.0, 0
                    bce_sum, bce_n = 0.0, 0
                    for v_idx, v_batch in enumerate(val_loader):
                        if v_idx >= args.val_max_batches: break
                        if v_batch is None: continue
                        Bv = v_batch["image"].shape[0]
                        v_images = v_batch["image"].to(device)
                        v_queries = v_batch["query"]
                        v_mask14 = v_batch["mask14"].to(device)
                        v_pres = v_batch["presence"].to(device)
                        v_text = encode_text_batch(clip_model_text, clip_tok, v_queries, device)
                        v_patches = bb_fn(v_images)
                        # Diagonal-only forward (cheaper than BxB; enough for diag_iou)
                        v_diag_logits = []
                        for k in range(Bv):
                            ll = head(v_patches[k:k+1], v_text[k:k+1]).reshape(G_out, G_out)
                            if args.per_query_bias:
                                v_diag_logits.append(
                                    sb(ll.view(1, 1, G_out, G_out),
                                       v_text[k:k+1]).view(G_out, G_out))
                            else:
                                v_diag_logits.append(sb(ll))
                        v_dl = torch.stack(v_diag_logits)  # (Bv, G_out, G_out)
                        v_p = torch.sigmoid(v_dl)
                        gt = v_mask14
                        # Per-positive diag IoU (only positives have meaningful masks)
                        pos_mask = (v_pres > 0.5)
                        if pos_mask.any():
                            gt_pos = gt[pos_mask] > 0.5
                            pred_pos = v_p[pos_mask] > 0.5
                            inter = (gt_pos & pred_pos).sum().item()
                            union = (gt_pos | pred_pos).sum().item()
                            if union > 0:
                                iou_sum += inter / union; iou_n += 1
                        bce = F.binary_cross_entropy_with_logits(v_dl, gt, reduction="mean")
                        bce_sum += bce.item(); bce_n += 1
                    val_diag_iou = iou_sum / max(1, iou_n)
                    val_bce = bce_sum / max(1, bce_n)
                    latest_val_iou = val_diag_iou
                    print(f"  [val] step {step} val_diag_iou={val_diag_iou:.3f}  val_bce={val_bce:.3f}  ({iou_n} batches)",
                          flush=True)
                    if wb is not None:
                        try:
                            wb.log({"val/diag_iou": val_diag_iou, "val/bce": val_bce}, step=step)
                        except Exception: pass
                    # Save best-by-val-iou whenever we have a new high
                    if val_diag_iou > best_val_iou:
                        best_val_iou = val_diag_iou
                        ckpt = {
                            "step": step, "epoch": epoch, "model": args.model,
                            "head": head.state_dict(), "sb": sb.state_dict(),
                            "args": vars(args), "val_diag_iou": val_diag_iou,
                        }
                        if objectness_head is not None:
                            ckpt["obj"] = objectness_head.state_dict()
                        if args.finetune_backbone_blocks > 0:
                            ckpt["backbone_state"] = bb_module.state_dict()
                        torch.save(ckpt, os.path.join(args.output_dir, "best_val.pt"))
                        print(f"    [save] best_val.pt  val_diag_iou={val_diag_iou:.3f}", flush=True)
                head.train()

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
                if objectness_head is not None:
                    ckpt["obj"] = objectness_head.state_dict()
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
                   choices=["v1", "v2-tiny", "d-mobile", "dinov2-s", "mobileclip-s2",
                            "fastvit-t8", "mobilevit-xs", "convnext-atto",
                            "convnext-femto", "convnext-pico", "convnext-nano", "convnext-tiny",
                            "repvit-m1", "efficientformerv2-s0"])
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
    p.add_argument("--per_query_bias_kind", default="mlp", choices=["mlp", "linear"],
                   help="Form of the per-query bias module when --per_query_bias is on. "
                        "'mlp' = SiglipBiasPerQuery (Linear 512->64 + GELU + Linear 64->1, the v0.6.0 default). "
                        "'linear' = SiglipBiasPerQueryLinear (single Linear 512->1, ablation of the GELU+hidden).")
    p.add_argument("--per_query_bias", action="store_true", default=False,
                   help="Replace global SigLIP (t, b) with per-query bias: a small "
                        "MLP from text_emb (512) -> scalar bias per query. "
                        "Lets the head shift each query to a common operating point.")
    p.add_argument("--bce_pos_weight", type=float, default=20.0,
                   help="upweight positive patches in dense BCE; ~20-50 useful for sparse masks")
    p.add_argument("--hard_neg_weight", type=float, default=0.0,
                   help="weight on hard-negative-mining loss (top-K negative cells per positive sample, BCE-toward-zero); attacks scene-context shortcut learning. 0 = disabled. Try 0.5-2.0.")
    p.add_argument("--hard_neg_topk", type=int, default=20,
                   help="Per-sample K for hard-negative mining (top-K predicted-prob cells where mask14<0.5).")
    p.add_argument("--resume_from", default=None,
                   help="Optional ckpt path to resume head+sb (and backbone if finetuning) from. Used for fine-tuning experiments on top of an existing release ckpt.")
    p.add_argument("--wandb_project", default="semantic-autogaze",
                   help="W&B project name; pass empty string to disable.")
    p.add_argument("--wandb_run_name", default=None,
                   help="W&B run display name; defaults to output_dir basename.")
    p.add_argument("--wandb_log_matrix_every", type=int, default=1000,
                   help="Render BxB cross-pair matrix on a fixed val batch and log to wandb every N steps. 0 = disable.")
    p.add_argument("--wandb_matrix_batch_size", type=int, default=5,
                   help="Size of the BxB validation matrix logged to wandb.")
    p.add_argument("--val_split_frac", type=float, default=0.05,
                   help="Fraction of unique image_ids to hold out as validation. 0 = no split.")
    p.add_argument("--val_eval_every", type=int, default=2000,
                   help="Compute val-set diag_iou + per-pair BCE every N steps. 0 = disable.")
    p.add_argument("--val_max_batches", type=int, default=20,
                   help="Cap number of val batches per eval (speed control).")
    p.add_argument("--distill_teacher_ckpt", default=None,
                   help="Path to teacher ckpt (e.g. DINOv2-small). Adds MSE-on-teacher-prob loss.")
    p.add_argument("--lambda_distill", type=float, default=0.5,
                   help="Weight on the teacher-MSE distillation loss term.")
    p.add_argument("--lambda_calib", type=float, default=0.0,
                   help="Weight on the absent-pair max-prob calibration loss. Penalizes relu(max_cell_prob - calib_target_max)^2 over clean off-diagonal (absent) pairs to clamp worst false-positive cell. 0 = disabled.")
    p.add_argument("--calib_target_max", type=float, default=0.30,
                   help="Target ceiling for max predicted prob on absent (off-diagonal, fn-kept) pairs; cells above this incur quadratic penalty.")
    p.add_argument("--lambda_bias_var", type=float, default=0.0,
                   help="Weight on per-query bias variance regularizer (only with --per_query_bias). Penalizes Var[b_q] across the batch — discourages the MLP from learning extreme positive/negative biases for individual queries (the 'tennis racket fires everywhere' overshoot in v0.6.0 review). 0 = disabled. Try 0.05-0.5.")
    p.add_argument("--category_alpha", type=float, default=0.0,
                   help="Category-rebalance exponent for BalancedSampler. 0=uniform per-positive (Phase 1-4), 0.5=sqrt-balance, 1.0=full inverse-frequency.")
    p.add_argument("--source_weights", default=None,
                   help="Per-source sampling-weight multipliers, e.g. 'pp:5,stuff:1,lvis:1,coco:1'. Multiplies the existing per-positive weight. Slug-prefix-based: pp_*=pascal_part, stuff_*=coco-stuff/panoptic, lvis_*=LVIS, else=COCO.")
    p.add_argument("--augment", action="store_true",
                   help="Enable horizontal-flip + color jitter (brightness/contrast) augmentation in TargetDataset.")
    p.add_argument("--augment_aggressive", action="store_true", default=False,
                   help="Enable AGGRESSIVE geometric augmentation: joint h-flip, +/-20deg rotation, "
                        "perspective warp (5%% corner displacement), random scale 0.7-1.5x, "
                        "and random crop applied to (image, mask_full). The mask is then "
                        "re-pooled to 14x14 (max-pool > 0.5). Color jitter from --augment is "
                        "applied last on the post-geometric image. This flag supersedes --augment "
                        "(strictly stronger) and is disabled for validation regardless. Falls "
                        "back to legacy aug for npz files missing the `mask_full` field.")
    p.add_argument("--image_size", type=int, default=224,
                   help="Input image size. Default 224. Use 288/336 for higher spatial resolution at higher compute cost.")
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
    p.add_argument("--grid_size_out", type=int, default=14, choices=[14, 28],
                   help="Output spatial grid for the per-query dense head. "
                        "14 (default) preserves v0.5.0 behaviour. 28 activates "
                        "TextScorerHeadGrid28 — same parameters as TextScorerHead, "
                        "but final logits are bilinearly upsampled from 14x14 to "
                        "28x28. Targets are resampled on-the-fly: max-pool of "
                        "mask_full > 0.5 (preferred), or nearest-upsample of "
                        "mask14 (fallback for legacy npz). Addresses the "
                        "thin-object localization failure (knife/skis/baseball-bat/"
                        "sports-ball) where pixels span <1 cell at 14x14. "
                        "v0.5.0 ckpts load cleanly into the 28-grid head via "
                        "--resume_from since the parameter set is identical.")

    # OWLv2-style query-agnostic objectness head (optional, training-only).
    # When >0, an ObjectnessHead is constructed alongside the main per-query
    # head, supervised by mask14 (the on-diagonal positive mask) with
    # BCEWithLogits(pos_weight=objectness_pos_weight). At inference, the head
    # produces a per-cell "is there ANY queryable object here?" score that
    # can gate the per-query heatmap. 0.0 = head not constructed.
    p.add_argument("--objectness_weight", type=float, default=0.0,
                   help="Loss weight for the OWLv2-style query-agnostic objectness head. "
                        "0.0 disables (head is not constructed). Try 0.1-1.0.")
    p.add_argument("--objectness_pos_weight", type=float, default=5.0,
                   help="pos_weight for objectness BCE. Lower than per-query bce_pos_weight "
                        "(default 30) since 'any object anywhere' has a much higher base rate.")

    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=200)
    args = p.parse_args()
    train(args)
