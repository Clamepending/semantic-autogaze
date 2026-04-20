"""
Train BigHead on COCO instance segmentation masks for crisp heatmaps.

Pipeline:
  1. Download COCO val2017 images + instance annotations (if not present)
  2. Cache frozen AutoGaze hidden states per image
  3. Cache CLIP text embeddings per category name
  4. Train BigHead with BCE(logits, mask_14x14) supervision
  5. Optionally mix CLIPSeg soft targets for open-vocab coverage

Run:
    cd /Users/mark/code/semantic-autogaze
    source .venv/bin/activate
    python -m semantic_autogaze.train_coco_seg --device mps
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from semantic_autogaze.bighead import BigHead, BigHeadDecoder

PATCH_GRID = 14
NUM_PATCHES = PATCH_GRID * PATCH_GRID  # 196


# ── Data download ──────────────────────────────────────────────────────


def download_coco_val(data_dir: str):
    """Download COCO val2017 images + instance annotations if missing."""
    import urllib.request

    img_dir = os.path.join(data_dir, "val2017")
    ann_file = os.path.join(data_dir, "annotations", "instances_val2017.json")

    if os.path.isdir(img_dir) and os.path.isfile(ann_file):
        print(f"[data] COCO val2017 already present at {data_dir}")
        return img_dir, ann_file

    os.makedirs(data_dir, exist_ok=True)

    if not os.path.isfile(ann_file):
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        ann_zip = os.path.join(data_dir, "annotations_trainval2017.zip")
        if not os.path.isfile(ann_zip):
            print(f"[data] Downloading COCO annotations...")
            urllib.request.urlretrieve(ann_url, ann_zip)
        print(f"[data] Extracting annotations...")
        with zipfile.ZipFile(ann_zip) as zf:
            for member in zf.namelist():
                if "instances_val2017" in member or "instances_train2017" not in member:
                    if member.startswith("annotations/"):
                        zf.extract(member, data_dir)
        os.remove(ann_zip)

    if not os.path.isdir(img_dir):
        img_url = "http://images.cocodataset.org/zips/val2017.zip"
        img_zip = os.path.join(data_dir, "val2017.zip")
        if not os.path.isfile(img_zip):
            print(f"[data] Downloading COCO val2017 images (~800MB)...")
            urllib.request.urlretrieve(img_url, img_zip)
        print(f"[data] Extracting images...")
        with zipfile.ZipFile(img_zip) as zf:
            zf.extractall(data_dir)
        os.remove(img_zip)

    return img_dir, ann_file


def download_coco_train(data_dir: str):
    """Download COCO train2017 images + instance annotations if missing."""
    import urllib.request

    img_dir = os.path.join(data_dir, "train2017")
    ann_file = os.path.join(data_dir, "annotations", "instances_train2017.json")

    if os.path.isdir(img_dir) and os.path.isfile(ann_file):
        print(f"[data] COCO train2017 already present at {data_dir}")
        return img_dir, ann_file

    os.makedirs(data_dir, exist_ok=True)

    if not os.path.isfile(ann_file):
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        ann_zip = os.path.join(data_dir, "annotations_trainval2017.zip")
        if not os.path.isfile(ann_zip):
            print(f"[data] Downloading COCO annotations...")
            urllib.request.urlretrieve(ann_url, ann_zip)
        print(f"[data] Extracting annotations...")
        with zipfile.ZipFile(ann_zip) as zf:
            zf.extract("annotations/instances_train2017.json", data_dir)
        os.remove(ann_zip)

    if not os.path.isdir(img_dir):
        img_url = "http://images.cocodataset.org/zips/train2017.zip"
        img_zip = os.path.join(data_dir, "train2017.zip")
        if not os.path.isfile(img_zip):
            print(f"[data] Downloading COCO train2017 images (~18GB)...")
            urllib.request.urlretrieve(img_url, img_zip)
        print(f"[data] Extracting images...")
        with zipfile.ZipFile(img_zip) as zf:
            zf.extractall(data_dir)
        os.remove(img_zip)

    return img_dir, ann_file


# ── Mask utilities ─────────────────────────────────────────────────────


def mask_to_patch_target(mask: np.ndarray, grid: int = PATCH_GRID) -> np.ndarray:
    """Downsample a binary mask to (grid, grid) patch-level targets.

    Uses average pooling: patch target = fraction of pixels in that patch
    that belong to the mask. Values in [0, 1].
    """
    h, w = mask.shape
    t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(t, (grid, grid))
    return pooled.squeeze().numpy()


def get_image_categories(coco: COCO, img_id: int, min_area_frac: float = 0.005):
    """Get categories present in image with at least min_area_frac coverage."""
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=0)
    anns = coco.loadAnns(ann_ids)
    img_info = coco.imgs[img_id]
    img_area = img_info["width"] * img_info["height"]

    cats = {}
    for ann in anns:
        cat_id = ann["category_id"]
        area = ann["area"]
        if area / img_area < min_area_frac:
            continue
        if cat_id not in cats:
            cats[cat_id] = []
        cats[cat_id].append(ann)
    return cats


def pick_hard_semantic_negatives(
    present_cat_ids,
    absent_cat_ids: list[int],
    clip_embeddings: dict[str, torch.Tensor],
    k: int,
) -> list[int]:
    """Pick k absent categories most similar (max CLIP cos-sim) to any positive.

    For each absent category, score = max over positives of cos(absent, positive).
    Highest scores = hardest negatives (semantically closest to something in
    the image). This is the gradient signal that forces fine-grained query
    routing instead of "predict objectness on any kitchen-utensil query".
    """
    if k <= 0 or not absent_cat_ids or not present_cat_ids:
        return []
    pos_embs = torch.stack(
        [clip_embeddings[str(c)].float() for c in present_cat_ids], dim=0
    )  # (P, D)
    pos_embs = F.normalize(pos_embs, dim=-1)
    abs_embs = torch.stack(
        [clip_embeddings[str(c)].float() for c in absent_cat_ids], dim=0
    )  # (A, D)
    abs_embs = F.normalize(abs_embs, dim=-1)
    sims = abs_embs @ pos_embs.T  # (A, P)
    scores = sims.max(dim=-1).values  # (A,)
    topk = min(k, len(absent_cat_ids))
    idx = torch.topk(scores, topk).indices.tolist()
    return [absent_cat_ids[i] for i in idx]


def build_category_mask(coco: COCO, img_id: int, cat_id: int, anns: list) -> np.ndarray:
    """Build binary mask for all instances of a category in an image."""
    img_info = coco.imgs[img_id]
    h, w = img_info["height"], img_info["width"]
    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        m = coco.annToMask(ann)
        mask = np.maximum(mask, m)
    return mask


def build_other_present_target(
    coco: COCO,
    img_id: int,
    self_cat_id: int,
    cats_for_img: dict,
    grid: int,
    self_target: np.ndarray,
) -> np.ndarray:
    """Patch-grid weight map of OTHER present categories' coverage in the image.

    For each patch, value = max occupancy across all present categories ≠ self.
    Then subtract self_target so we never penalise the model for firing where
    the positive itself overlaps another category (e.g. person on bicycle).

    Used by v13e (within-image hard negatives): we want the model trained on
    (image, "fork") to be heavily discouraged from firing on the bottle's
    patches in the same image. The weight = "how-other-it-is" per patch.
    """
    other = np.zeros((grid, grid), dtype=np.float32)
    for cat_id, anns in cats_for_img.items():
        if cat_id == self_cat_id or not anns:
            continue
        m = build_category_mask(coco, img_id, cat_id, anns)
        t = mask_to_patch_target(m, grid)
        other = np.maximum(other, t)
    other = np.clip(other - self_target, 0.0, 1.0)
    return other


# ── Caching ────────────────────────────────────────────────────────────


@torch.inference_mode()
def cache_autogaze_hidden_states(
    img_dir: str,
    coco: COCO,
    cache_dir: str,
    device: torch.device,
    max_images: Optional[int] = None,
):
    """Cache AutoGaze post-decoder hidden states for each COCO image."""
    from semantic_autogaze.inference import load_autogaze
    from semantic_autogaze.model import SemanticAutoGaze

    os.makedirs(cache_dir, exist_ok=True)
    img_ids = sorted(coco.getImgIds())
    if max_images:
        img_ids = img_ids[:max_images]

    already = sum(1 for i in img_ids if os.path.exists(os.path.join(cache_dir, f"{i}.pt")))
    if already == len(img_ids):
        print(f"[cache] All {already} hidden states already cached")
        return

    print(f"[cache] Loading AutoGaze for hidden state caching...")
    autogaze = load_autogaze(device=device)
    wrapper = SemanticAutoGaze(autogaze).to(device).eval()

    for img_id in tqdm(img_ids, desc="Caching AutoGaze hidden states"):
        out_path = os.path.join(cache_dir, f"{img_id}.pt")
        if os.path.exists(out_path):
            continue

        img_info = coco.imgs[img_id]
        img_path = os.path.join(img_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img.resize((224, 224), Image.BILINEAR))

        # (C, H, W) → (1, 1, C, H, W) — single frame
        t = torch.from_numpy(img_np).permute(2, 0, 1).float().div_(255.0)
        video = t.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3, 224, 224)

        hidden = wrapper.get_patch_hidden_states(video)  # (1, 196, 192)
        torch.save(hidden[0].cpu().to(torch.float16), out_path)

        if device.type == "mps" and img_id % 50 == 0:
            torch.mps.synchronize()

    del wrapper, autogaze
    if device.type == "mps":
        torch.mps.synchronize()
    torch.mps.empty_cache() if device.type == "mps" else None
    print(f"[cache] Done caching hidden states")


@torch.inference_mode()
def cache_clip_vision_features(
    img_dir: str,
    coco: COCO,
    cache_dir: str,
    device: torch.device,
    max_images: Optional[int] = None,
):
    """Cache CLIP ViT-B/16 per-patch vision features (196×768) for each COCO image."""
    import open_clip

    os.makedirs(cache_dir, exist_ok=True)
    img_ids = sorted(coco.getImgIds())
    if max_images:
        img_ids = img_ids[:max_images]

    already = sum(1 for i in img_ids if os.path.exists(os.path.join(cache_dir, f"{i}.pt")))
    if already == len(img_ids):
        print(f"[cache] All {already} CLIP vision features already cached")
        return

    print(f"[cache] Loading CLIP ViT-B/16 for vision feature caching...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    vision = model.visual.to(device).eval()

    for img_id in tqdm(img_ids, desc="Caching CLIP vision features"):
        out_path = os.path.join(cache_dir, f"{img_id}.pt")
        if os.path.exists(out_path):
            continue

        img_info = coco.imgs[img_id]
        img_path = os.path.join(img_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        img_t = preprocess(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

        # Extract patch features before pooling
        x = vision.conv1(img_t)  # (1, 768, 14, 14)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (1, 196, 768)
        # Add class token
        cls = vision.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)  # (1, 197, 768)
        x = x + vision.positional_embedding.unsqueeze(0)
        x = vision.patch_dropout(x) if hasattr(vision, 'patch_dropout') else x
        x = vision.ln_pre(x)
        x = vision.transformer(x)  # (1, 197, 768)
        patch_features = x[:, 1:, :]  # (1, 196, 768) — exclude class token

        torch.save(patch_features[0].cpu().to(torch.float16), out_path)

        if device.type == "mps" and img_id % 50 == 0:
            torch.mps.synchronize()

    del vision, model
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    print(f"[cache] Done caching CLIP vision features")


class CLIPVisionOnline:
    """Holds a frozen CLIP ViT-B/16 vision encoder for on-the-fly patch feature extraction.

    Computes 196×768 patch features per image without caching to disk.
    Designed for training on large datasets (e.g. train2017) where disk-cached
    CLIP vision features would be ~35GB.
    """

    def __init__(self, device: torch.device):
        import open_clip
        print("[clip_online] Loading CLIP ViT-B/16 for on-the-fly vision features...")
        model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        self.vision = model.visual.to(device).eval()
        self.device = device
        # Freeze all parameters
        for p in self.vision.parameters():
            p.requires_grad = False
        del model
        print("[clip_online] CLIP vision encoder loaded (frozen)")

    @torch.inference_mode()
    def extract_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch features for a batch of preprocessed images.

        Args:
            images: (B, 3, 224, 224) preprocessed CLIP input tensors.

        Returns:
            (B, 196, 768) patch features (float32).
        """
        x = self.vision.conv1(images)  # (B, 768, 14, 14)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, 196, 768)
        cls = self.vision.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 197, 768)
        x = x + self.vision.positional_embedding.unsqueeze(0)
        x = self.vision.patch_dropout(x) if hasattr(self.vision, 'patch_dropout') else x
        x = self.vision.ln_pre(x)
        x = self.vision.transformer(x)  # (B, 197, 768)
        return x[:, 1:, :].float()  # (B, 196, 768) — exclude class token


@torch.inference_mode()
def cache_clip_text_embeddings(
    categories: dict[int, str],
    cache_path: str,
    device: torch.device,
):
    """Cache CLIP ViT-B/16 text embeddings for all category names."""
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if set(cached.keys()) >= set(str(k) for k in categories.keys()):
            print(f"[cache] CLIP text embeddings already cached ({len(cached)} categories)")
            return cached

    import open_clip

    print(f"[cache] Encoding {len(categories)} category names with CLIP...")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model = model.to(device).eval()

    embeddings = {}
    for cat_id, cat_name in categories.items():
        prompts = [
            cat_name,
            f"a photo of a {cat_name}",
            f"a {cat_name} in a scene",
        ]
        toks = tokenizer(prompts).to(device)
        embs = model.encode_text(toks).float()
        emb = F.normalize(embs.mean(dim=0, keepdim=True), dim=-1)
        embeddings[str(cat_id)] = emb.cpu().squeeze(0)

    del model
    torch.save(embeddings, cache_path)
    print(f"[cache] Saved {len(embeddings)} text embeddings to {cache_path}")
    return embeddings


# ── Dataset ────────────────────────────────────────────────────────────


class CocoSegDataset(Dataset):
    """COCO images with instance segmentation masks as patch-level targets."""

    def __init__(
        self,
        coco: COCO,
        img_ids: list[int],
        hidden_cache_dir: str,
        clip_embeddings: dict[str, torch.Tensor],
        categories: dict[int, str],
        target_grid: int = PATCH_GRID,
        neg_per_image: int = 1,
        hard_semantic_neg_per_image: int = 0,
        within_image_neg: bool = False,
        clip_vision_cache_dir: Optional[str] = None,
        img_dir: Optional[str] = None,
        clip_preprocess=None,
    ):
        self.coco = coco
        self.hidden_cache_dir = hidden_cache_dir
        self.clip_embeddings = clip_embeddings
        self.categories = categories
        self.target_grid = target_grid
        self.target_n = target_grid * target_grid
        self.clip_vision_cache_dir = clip_vision_cache_dir
        self.img_dir = img_dir
        self.clip_preprocess = clip_preprocess
        self.within_image_neg = within_image_neg

        # Build index: (img_id, cat_id, [anns]) for all valid pairs
        # Also build a per-image map of all present (cat_id -> anns) for v13e
        # within-image hard negatives.
        self.samples = []
        self.cats_for_img: dict[int, dict] = {}
        for img_id in img_ids:
            cache_path = os.path.join(hidden_cache_dir, f"{img_id}.pt")
            if not os.path.exists(cache_path):
                continue
            cats = get_image_categories(coco, img_id)
            present = {}
            for cat_id, anns in cats.items():
                if str(cat_id) in clip_embeddings:
                    self.samples.append((img_id, cat_id, anns))
                    present[cat_id] = anns
            if present:
                self.cats_for_img[img_id] = present

        # v13e: precompute per-image per-category patch-occupancy maps once,
        # so __getitem__ never has to rasterize polygons. Without this, every
        # (img_id, query_cat) sample pays the cost of re-building all present
        # categories' masks → 130× training slowdown vs v11 (measured: 9.5s/it
        # vs 0.07s/it on Modal T4 at bs=32).
        self.cat_patch_occupancy: dict[int, dict[int, np.ndarray]] = {}
        if within_image_neg:
            print(f"[data] precomputing within-image-neg patch occupancy for {len(self.cats_for_img)} images at grid {target_grid}…")
            for img_id, present in tqdm(
                self.cats_for_img.items(),
                desc="cat patch occupancy",
            ):
                per_cat = {}
                for cat_id, anns in present.items():
                    if not anns:
                        continue
                    m = build_category_mask(coco, img_id, cat_id, anns)
                    per_cat[cat_id] = mask_to_patch_target(m, target_grid)
                if per_cat:
                    self.cat_patch_occupancy[img_id] = per_cat

        # Also add negative samples: categories NOT present in image
        self._add_negatives(
            img_ids,
            neg_per_image=neg_per_image,
            hard_semantic_neg_per_image=hard_semantic_neg_per_image,
        )

    def _add_negatives(
        self,
        img_ids: list[int],
        neg_per_image: int = 1,
        hard_semantic_neg_per_image: int = 0,
    ):
        """Add negative category samples (category not in image → all-zero mask).

        - `neg_per_image`: random absent categories (the easy negatives).
        - `hard_semantic_neg_per_image`: absent categories most similar in CLIP
          text-embedding space to a positive in the image. Forces fine-grained
          query routing because the head can't tell "knife" from "fork" without
          actually using the query.
        """
        all_cat_ids = [c for c in self.categories.keys() if str(c) in self.clip_embeddings]
        neg_samples = []
        present_by_img = {}
        for img_id, cat_id, _ in self.samples:
            present_by_img.setdefault(img_id, set()).add(cat_id)

        for img_id in img_ids:
            cache_path = os.path.join(self.hidden_cache_dir, f"{img_id}.pt")
            if not os.path.exists(cache_path):
                continue
            present_cats = present_by_img.get(img_id, set())
            absent = [c for c in all_cat_ids if c not in present_cats]
            if not absent:
                continue

            # Random absent negatives (existing behavior)
            n_rand = min(neg_per_image, len(absent))
            if n_rand > 0:
                chosen = random.sample(absent, n_rand)
                for neg_cat in chosen:
                    neg_samples.append((img_id, neg_cat, []))

            # Hard semantic negatives: most-similar-to-any-positive absent cats
            n_hard = min(hard_semantic_neg_per_image, len(absent))
            if n_hard > 0 and present_cats:
                hard_picks = pick_hard_semantic_negatives(
                    present_cats, absent, self.clip_embeddings, n_hard
                )
                for neg_cat in hard_picks:
                    neg_samples.append((img_id, neg_cat, []))

        self.samples.extend(neg_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, cat_id, anns = self.samples[idx]

        # Load cached hidden states
        hidden = torch.load(
            os.path.join(self.hidden_cache_dir, f"{img_id}.pt"),
            map_location="cpu",
            weights_only=True,
        ).float()  # (196, 192) — stored as float16

        # Optionally concatenate CLIP vision features
        if self.clip_vision_cache_dir:
            clip_vis_path = os.path.join(self.clip_vision_cache_dir, f"{img_id}.pt")
            if os.path.exists(clip_vis_path):
                clip_vis = torch.load(clip_vis_path, map_location="cpu", weights_only=True).float()
                hidden = torch.cat([hidden, clip_vis], dim=-1)  # (196, 192+768)

        # CLIP text embedding for this category
        query = self.clip_embeddings[str(cat_id)]  # (512,)

        # Build mask target at the configured target grid
        if anns:
            mask = build_category_mask(self.coco, img_id, cat_id, anns)
            target = mask_to_patch_target(mask, self.target_grid)
        else:
            target = np.zeros((self.target_grid, self.target_grid), dtype=np.float32)

        # v13e: per-patch weight map for OTHER present categories (in-image
        # hard negatives). Uses the patch-occupancy maps precomputed once at
        # dataset init — no polygon rasterization in the hot path.
        if self.within_image_neg and img_id in self.cat_patch_occupancy:
            occ = self.cat_patch_occupancy[img_id]
            other = np.zeros((self.target_grid, self.target_grid), dtype=np.float32)
            for c, t in occ.items():
                if c == cat_id:
                    continue
                np.maximum(other, t, out=other)
            np.subtract(other, target, out=other)
            np.clip(other, 0.0, 1.0, out=other)
        else:
            other = np.zeros((self.target_grid, self.target_grid), dtype=np.float32)

        target_t = torch.from_numpy(target).reshape(self.target_n)
        other_t = torch.from_numpy(other).reshape(self.target_n)

        result = {
            "hidden": hidden,
            "query": query,
            "target": target_t,
            "other_target": other_t,
            "img_id": img_id,
            "cat_id": cat_id,
        }

        # On-the-fly CLIP vision: return preprocessed image tensor
        if self.clip_preprocess is not None and self.img_dir is not None:
            img_info = self.coco.imgs[img_id]
            img_path = os.path.join(self.img_dir, img_info["file_name"])
            img = Image.open(img_path).convert("RGB")
            result["clip_image"] = self.clip_preprocess(img)

        return result


# ── CLIPSeg soft targets ──────────────────────────────────────────────


@torch.inference_mode()
def generate_clipseg_targets(
    img_dir: str,
    coco: COCO,
    img_ids: list[int],
    categories: dict[int, str],
    cache_dir: str,
    device: torch.device,
    target_grid: int = PATCH_GRID,
    max_cats_per_image: int = 5,
    include_negatives: bool = False,
):
    """Generate CLIPSeg soft heatmaps for mixing with hard mask supervision.

    `target_grid` controls the spatial resolution of cached targets. Cache
    files are versioned by grid size so 14x and 28x and 56x can coexist.

    `include_negatives`: if True, also generate CLIPSeg targets for a few
    categories not present in the image. This gives the student CLIPSeg's
    "this object is absent" signal for open-vocab.
    """
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    os.makedirs(cache_dir, exist_ok=True)
    suffix = f"_g{target_grid}.pt"

    done = sum(1 for i in img_ids if os.path.exists(os.path.join(cache_dir, f"{i}{suffix}")))
    if done == len(img_ids):
        print(f"[clipseg] All {done} images already cached at grid {target_grid}")
        return

    print(f"[clipseg] Loading CLIPSeg model (target grid {target_grid})...")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = model.to(device).eval()

    all_cat_ids = list(categories.keys())

    from semantic_autogaze.letterbox import letterbox_image

    for img_id in tqdm(img_ids, desc=f"CLIPSeg soft targets g{target_grid}"):
        out_path = os.path.join(cache_dir, f"{img_id}{suffix}")
        if os.path.exists(out_path):
            continue

        img_info = coco.imgs[img_id]
        img = Image.open(os.path.join(img_dir, img_info["file_name"])).convert("RGB")
        # Letterbox to square before CLIPSeg sees it. CLIPSegProcessor will
        # resize to 352x352; with a square input that resize is aspect-
        # preserving and the resulting target lives in letterbox-square frame
        # consistent with the DINOv2 cache.
        img, _ = letterbox_image(img)

        present_cats = get_image_categories(coco, img_id)
        query_cat_ids = list(present_cats.keys())[:max_cats_per_image]
        if include_negatives:
            absent = [c for c in all_cat_ids if c not in present_cats]
            random.shuffle(absent)
            query_cat_ids = list(query_cat_ids) + absent[:2]
        query_names = [categories[c] for c in query_cat_ids if c in categories]
        if not query_names:
            continue

        inputs = processor(
            text=query_names,
            images=[img] * len(query_names),
            return_tensors="pt",
            padding=True,
        ).to(device)

        outputs = model(**inputs)
        logits = outputs.logits  # (N_queries, H, W)
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)  # single-query edge case

        soft = torch.sigmoid(logits).unsqueeze(1).cpu()  # (N, 1, H, W)
        # Bilinear interpolate to target_grid (works for any non-divisible H/W)
        soft_g = F.interpolate(soft, size=(target_grid, target_grid),
                               mode="bilinear", align_corners=False)
        soft_g = soft_g.squeeze(1)  # (N, target_grid, target_grid)

        result = {}
        for i, cat_id in enumerate(query_cat_ids):
            if i < len(soft_g):
                result[str(cat_id)] = soft_g[i].to(torch.float16)

        torch.save(result, out_path)

    del model, processor
    print(f"[clipseg] Done")


class CocoSegWithClipSegDataset(Dataset):
    """Dataset that mixes hard mask targets with CLIPSeg soft targets."""

    def __init__(
        self,
        coco: COCO,
        img_ids: list[int],
        hidden_cache_dir: str,
        clip_embeddings: dict[str, torch.Tensor],
        categories: dict[int, str],
        clipseg_cache_dir: Optional[str] = None,
        clipseg_mix_ratio: float = 0.3,
        target_grid: int = PATCH_GRID,
        neg_per_image: int = 1,
        hard_semantic_neg_per_image: int = 0,
        within_image_neg: bool = False,
        clip_vision_cache_dir: Optional[str] = None,
        img_dir: Optional[str] = None,
        clip_preprocess=None,
    ):
        self.coco = coco
        self.hidden_cache_dir = hidden_cache_dir
        self.clip_embeddings = clip_embeddings
        self.categories = categories
        self.clipseg_cache_dir = clipseg_cache_dir
        self.clipseg_mix_ratio = clipseg_mix_ratio
        self.target_grid = target_grid
        self.target_n = target_grid * target_grid
        self.clipseg_suffix = f"_g{target_grid}.pt"
        self.clip_vision_cache_dir = clip_vision_cache_dir
        self.img_dir = img_dir
        self.clip_preprocess = clip_preprocess
        self.within_image_neg = within_image_neg

        # Build index of (img_id, cat_id, anns, source) tuples
        # Also cache per-image (cat_id -> anns) for v13e in-image negatives.
        self.samples = []
        self.cats_for_img: dict[int, dict] = {}
        all_cat_ids = [c for c in categories.keys() if str(c) in clip_embeddings]
        for img_id in img_ids:
            cache_path = os.path.join(hidden_cache_dir, f"{img_id}.pt")
            if not os.path.exists(cache_path):
                continue
            cats = get_image_categories(coco, img_id)
            present_for_neg = []
            present_anns: dict = {}
            for cat_id, anns in cats.items():
                if str(cat_id) in clip_embeddings:
                    self.samples.append((img_id, cat_id, anns, "mask"))
                    present_for_neg.append(cat_id)
                    present_anns[cat_id] = anns
            if present_anns:
                self.cats_for_img[img_id] = present_anns

            # Random absent negatives
            present = set(cats.keys())
            absent = [c for c in all_cat_ids if c not in present]
            if absent and neg_per_image > 0:
                n = min(neg_per_image, len(absent))
                chosen = random.sample(absent, n)
                for neg in chosen:
                    self.samples.append((img_id, neg, [], "mask"))

            # Hard semantic negatives
            if absent and hard_semantic_neg_per_image > 0 and present_for_neg:
                n_hard = min(hard_semantic_neg_per_image, len(absent))
                hard_picks = pick_hard_semantic_negatives(
                    present_for_neg, absent, clip_embeddings, n_hard
                )
                for neg in hard_picks:
                    self.samples.append((img_id, neg, [], "mask"))

        # v13e: precompute per-image per-category patch-occupancy maps once
        # (same fix as in CocoSegDataset — without it, each __getitem__ would
        # rasterize every present category's polygons, blowing up training
        # time ~130×).
        self.cat_patch_occupancy: dict[int, dict[int, np.ndarray]] = {}
        if within_image_neg:
            print(f"[data] precomputing within-image-neg patch occupancy for {len(self.cats_for_img)} images at grid {target_grid}…")
            for img_id, present in tqdm(
                self.cats_for_img.items(),
                desc="cat patch occupancy (clipseg dataset)",
            ):
                per_cat = {}
                for cat_id, anns in present.items():
                    if not anns:
                        continue
                    m = build_category_mask(coco, img_id, cat_id, anns)
                    per_cat[cat_id] = mask_to_patch_target(m, target_grid)
                if per_cat:
                    self.cat_patch_occupancy[img_id] = per_cat

    def __len__(self):
        return len(self.samples)

    def _maybe_load_clip_image(self, img_id: int, result: dict) -> dict:
        """Add preprocessed CLIP image to result dict if online mode is active."""
        if self.clip_preprocess is not None and self.img_dir is not None:
            img_info = self.coco.imgs[img_id]
            img_path = os.path.join(self.img_dir, img_info["file_name"])
            img = Image.open(img_path).convert("RGB")
            result["clip_image"] = self.clip_preprocess(img)
        return result

    def __getitem__(self, idx):
        img_id, cat_id, anns, source = self.samples[idx]

        hidden = torch.load(
            os.path.join(self.hidden_cache_dir, f"{img_id}.pt"),
            map_location="cpu",
            weights_only=True,
        ).float()

        # Optionally concatenate CLIP vision features (cached mode)
        if self.clip_vision_cache_dir:
            clip_vis_path = os.path.join(self.clip_vision_cache_dir, f"{img_id}.pt")
            if os.path.exists(clip_vis_path):
                clip_vis = torch.load(clip_vis_path, map_location="cpu", weights_only=True).float()
                hidden = torch.cat([hidden, clip_vis], dim=-1)

        query = self.clip_embeddings[str(cat_id)]

        # Decide whether to use CLIPSeg soft target or hard mask
        use_clipseg = (
            self.clipseg_cache_dir
            and random.random() < self.clipseg_mix_ratio
            and anns  # only for positive samples
        )

        target_np: Optional[np.ndarray] = None
        target_t: Optional[torch.Tensor] = None
        if use_clipseg:
            clipseg_path = os.path.join(self.clipseg_cache_dir, f"{img_id}{self.clipseg_suffix}")
            if os.path.exists(clipseg_path):
                clipseg_data = torch.load(clipseg_path, map_location="cpu", weights_only=True)
                if str(cat_id) in clipseg_data:
                    soft = clipseg_data[str(cat_id)].float()
                    target_np = soft.reshape(self.target_grid, self.target_grid).numpy()
                    target_t = soft.reshape(self.target_n)

        if target_t is None:
            if anns:
                mask = build_category_mask(self.coco, img_id, cat_id, anns)
                target_np = mask_to_patch_target(mask, self.target_grid)
            else:
                target_np = np.zeros((self.target_grid, self.target_grid), dtype=np.float32)
            target_t = torch.from_numpy(target_np).reshape(self.target_n)

        # v13e other-present weight map. Uses precomputed patch-occupancy
        # maps so the hot path never rasterizes polygons.
        if self.within_image_neg and anns and img_id in self.cat_patch_occupancy:
            occ = self.cat_patch_occupancy[img_id]
            other = np.zeros((self.target_grid, self.target_grid), dtype=np.float32)
            for c, t in occ.items():
                if c == cat_id:
                    continue
                np.maximum(other, t, out=other)
            np.subtract(other, target_np, out=other)
            np.clip(other, 0.0, 1.0, out=other)
        else:
            other = np.zeros((self.target_grid, self.target_grid), dtype=np.float32)
        other_t = torch.from_numpy(other).reshape(self.target_n)

        return self._maybe_load_clip_image(img_id, {
            "hidden": hidden, "query": query, "target": target_t,
            "other_target": other_t,
            "img_id": img_id, "cat_id": cat_id})


# ── Training ───────────────────────────────────────────────────────────


def soft_dice_loss(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # probs, target: (B, N) in [0,1]. Per-sample soft dice, mean across batch.
    intersection = (probs * target).sum(dim=1)
    denom = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss (Lin et al. 2017) for imbalanced binary classification.

    alpha_t = alpha for positive (target=1), 1-alpha for negative (target=0).
    With alpha=0.25: foreground weighted 0.25, background 0.75 (standard RetinaNet).
    With alpha=0.75: foreground weighted 0.75, background 0.25 (better for low-FG tasks).
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pt = torch.exp(-bce)  # p_t
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal = alpha_t * (1 - pt) ** gamma * bce
    return focal.mean()


def train_epoch(head, loader, optimizer, device, epoch, total_epochs, dice_weight=0.0, use_focal=False, focal_alpha=0.25, focal_gamma=2.0, other_neg_weight=0.0, clip_vision_online: Optional[CLIPVisionOnline] = None):
    head.train()
    bce_losses, dice_losses, other_losses, total_losses = [], [], [], []
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [train]")
    for batch in pbar:
        hidden = batch["hidden"].to(device)
        query = batch["query"].to(device)
        target = batch["target"].to(device)

        # On-the-fly CLIP vision feature extraction
        if clip_vision_online is not None and "clip_image" in batch:
            clip_vis = clip_vision_online.extract_batch(batch["clip_image"].to(device))
            hidden = torch.cat([hidden, clip_vis], dim=-1)

        logits = head(hidden, query)  # (B, N)

        # Compute loss term (focal or BCE)
        if use_focal:
            bce = focal_loss(logits, target, alpha=focal_alpha, gamma=focal_gamma)
        else:
            bce = F.binary_cross_entropy_with_logits(logits, target)

        if dice_weight > 0:
            probs = torch.sigmoid(logits)
            dice = soft_dice_loss(probs, target)
            loss = bce + dice_weight * dice
        else:
            dice = torch.zeros((), device=device)
            loss = bce

        # v13e: weighted BCE pushing pred to 0 inside other present categories.
        # weight = how-other-it-is per patch (in [0,1]); normalised by sum so
        # the term is interpretable as "average BCE on other-cat patches".
        if other_neg_weight > 0 and "other_target" in batch:
            other_target = batch["other_target"].to(device)
            denom = other_target.sum().clamp(min=1.0)
            per_pixel = F.binary_cross_entropy_with_logits(
                logits, torch.zeros_like(logits), reduction="none"
            )
            other_loss = (per_pixel * other_target).sum() / denom
            loss = loss + other_neg_weight * other_loss
        else:
            other_loss = torch.zeros((), device=device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

        bce_losses.append(bce.item())
        dice_losses.append(dice.item())
        other_losses.append(other_loss.item())
        total_losses.append(loss.item())
        pbar.set_postfix(
            bce=f"{bce.item():.4f}", dice=f"{dice.item():.4f}",
            otr=f"{other_loss.item():.4f}",
        )

    return {
        "bce": sum(bce_losses) / len(bce_losses),
        "dice": sum(dice_losses) / len(dice_losses),
        "other": sum(other_losses) / len(other_losses),
        "total": sum(total_losses) / len(total_losses),
    }


@torch.inference_mode()
def validate(head, loader, device, use_focal=False, clip_vision_online: Optional[CLIPVisionOnline] = None):
    # Always pure BCE for cross-experiment comparability.
    head.eval()
    losses = []
    for batch in loader:
        hidden = batch["hidden"].to(device)
        query = batch["query"].to(device)
        target = batch["target"].to(device)

        # On-the-fly CLIP vision feature extraction
        if clip_vision_online is not None and "clip_image" in batch:
            clip_vis = clip_vision_online.extract_batch(batch["clip_image"].to(device))
            hidden = torch.cat([hidden, clip_vis], dim=-1)

        logits = head(hidden, query)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        losses.append(loss.item())

    return sum(losses) / max(len(losses), 1)


@torch.inference_mode()
def save_qualitative_examples(
    head,
    coco: COCO,
    img_dir: str,
    hidden_cache_dir: str,
    clip_embeddings: dict[str, torch.Tensor],
    categories: dict[int, str],
    device: torch.device,
    output_dir: str,
    num_examples: int = 20,
    clip_vision_cache_dir: Optional[str] = None,
    clip_vision_online: Optional[CLIPVisionOnline] = None,
):
    """Save side-by-side comparisons: original image + GT mask + predicted heatmap."""
    head.eval()
    os.makedirs(output_dir, exist_ok=True)
    cmap = matplotlib.colormaps["jet"]

    img_ids = sorted(coco.getImgIds())
    random.shuffle(img_ids)
    saved = 0

    for img_id in img_ids:
        if saved >= num_examples:
            break

        cache_path = os.path.join(hidden_cache_dir, f"{img_id}.pt")
        if not os.path.exists(cache_path):
            continue

        cats = get_image_categories(coco, img_id)
        if not cats:
            continue

        cat_id = random.choice(list(cats.keys()))
        if str(cat_id) not in clip_embeddings:
            continue

        cat_name = categories.get(cat_id, f"cat_{cat_id}")
        anns = cats[cat_id]

        # Load image
        img_info = coco.imgs[img_id]
        img = Image.open(os.path.join(img_dir, img_info["file_name"])).convert("RGB")
        img_np = np.array(img)

        # GT mask
        mask = build_category_mask(coco, img_id, cat_id, anns)
        gt_14 = mask_to_patch_target(mask, PATCH_GRID)

        # Predict (output may be at decoder grid, not 14x14)
        hidden = torch.load(cache_path, map_location="cpu", weights_only=True).float()
        if clip_vision_cache_dir:
            cv_path = os.path.join(clip_vision_cache_dir, f"{img_id}.pt")
            if os.path.exists(cv_path):
                clip_vis = torch.load(cv_path, map_location="cpu", weights_only=True).float()
                hidden = torch.cat([hidden, clip_vis], dim=-1)
        elif clip_vision_online is not None:
            clip_img = clip_vision_online.preprocess(img).unsqueeze(0).to(device)
            clip_vis = clip_vision_online.extract_batch(clip_img)[0].cpu()
            hidden = torch.cat([hidden, clip_vis], dim=-1)
        query = clip_embeddings[str(cat_id)]
        logits = head(hidden.unsqueeze(0).to(device), query.unsqueeze(0).to(device))
        n_out = logits.shape[-1]
        out_grid = int(round(n_out ** 0.5))
        probs = torch.sigmoid(logits).reshape(out_grid, out_grid).cpu().numpy()

        # Upsample for overlay
        h, w = img_np.shape[:2]
        probs_up = F.interpolate(
            torch.from_numpy(probs).unsqueeze(0).unsqueeze(0),
            size=(h, w), mode="bilinear", align_corners=False,
        )[0, 0].numpy()
        probs_up = (probs_up - probs_up.min()) / (probs_up.max() - probs_up.min() + 1e-8)

        heat_rgb = (cmap(probs_up)[..., :3] * 255).astype(np.uint8)
        overlay = (0.5 * img_np.astype(np.float32) + 0.5 * heat_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)

        # GT overlay
        gt_up = F.interpolate(
            torch.from_numpy(gt_14).unsqueeze(0).unsqueeze(0),
            size=(h, w), mode="nearest",
        )[0, 0].numpy()
        gt_rgb = (cmap(gt_up)[..., :3] * 255).astype(np.uint8)
        gt_overlay = (0.5 * img_np.astype(np.float32) + 0.5 * gt_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img_np)
        axes[0].set_title(f"Image {img_id}")
        axes[0].axis("off")
        axes[1].imshow(gt_overlay)
        axes[1].set_title(f"GT: \"{cat_name}\"")
        axes[1].axis("off")
        axes[2].imshow(overlay)
        axes[2].set_title(f"Pred: \"{cat_name}\"")
        axes[2].axis("off")
        plt.suptitle(f"Query: \"{cat_name}\" | Image {img_id}", fontsize=14)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"{saved:03d}_{img_id}_{cat_name.replace(' ', '_')}.png")
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close()
        saved += 1

    print(f"[eval] Saved {saved} qualitative examples to {output_dir}")


# ── Main ───────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/coco")
    p.add_argument("--output_dir", default="results/coco_seg")
    p.add_argument("--device", default="mps")
    p.add_argument("--max_images", type=int, default=None,
                   help="Limit images for faster iteration (None = all)")
    p.add_argument("--train_split", default="val",
                   help="COCO split to train on: 'val' (5K) or 'train' (118K). "
                        "When 'train', val2017 is used for validation.")
    p.add_argument("--expanded_dim", type=int, default=512)
    p.add_argument("--n_attn_heads", type=int, default=8)
    p.add_argument("--n_attn_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--clipseg_mix", type=float, default=0.0,
                   help="Fraction of samples using CLIPSeg soft targets (0=disabled)")
    p.add_argument("--dice_weight", type=float, default=0.0,
                   help="Soft-dice loss weight added to BCE (0=disabled). Encourages crisp boundaries.")
    p.add_argument("--focal", action="store_true",
                   help="Use focal loss instead of BCE.")
    p.add_argument("--focal_alpha", type=float, default=0.25,
                   help="Focal loss alpha parameter (default 0.25).")
    p.add_argument("--focal_gamma", type=float, default=2.0,
                   help="Focal loss gamma parameter (default 2.0).")
    p.add_argument("--decoder", action="store_true",
                   help="Use BigHeadDecoder with upsampling for higher-resolution output.")
    p.add_argument("--out_grid", type=int, default=14,
                   help="Output spatial grid size. With --decoder must be 14, 28, or 56.")
    p.add_argument("--decoder_dim", type=int, default=128,
                   help="Decoder bottleneck channel dim (BigHeadDecoder only).")
    p.add_argument("--clipseg_neg", action="store_true",
                   help="When generating CLIPSeg targets, also include negative categories.")
    p.add_argument("--neg_per_image", type=int, default=1,
                   help="Number of negative category samples per image (default 1).")
    p.add_argument("--hard_semantic_neg_per_image", type=int, default=0,
                   help="Number of HARD semantic negatives per image (absent categories most "
                        "similar in CLIP text-embedding space to a positive). Forces fine-grained "
                        "query routing. v13c. Additive to --neg_per_image.")
    p.add_argument("--within_image_neg", action="store_true",
                   help="v13e: emit per-patch weight map of OTHER present categories so "
                        "the loss can directly penalise firing on co-occurring objects. "
                        "Requires --other_neg_weight > 0 to take effect.")
    p.add_argument("--other_neg_weight", type=float, default=0.0,
                   help="v13e: weight on the within-image hard-negative BCE term. "
                        "Tuned: 1.0 = roughly comparable to main BCE on small-object cases; "
                        "5.0 = aggressive routing pressure.")
    p.add_argument("--clip_vision", action="store_true",
                   help="Concatenate CLIP ViT-B/16 patch features (768-dim) with AutoGaze features (cached to disk).")
    p.add_argument("--clip_vision_online", action="store_true",
                   help="Compute CLIP vision features on-the-fly (no disk caching). Use for large datasets where caching is infeasible.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", default="semantic-autogaze")
    p.add_argument("--run_name", default=None)
    p.add_argument("--num_qual_examples", type=int, default=20)
    p.add_argument("--init_from", default=None,
                   help="Path to a .pt checkpoint to initialise the head from "
                        "(state_dict only). Lets you fine-tune from a prior run.")
    p.add_argument("--hidden_cache_dir", default=None,
                   help="Override the train-split AutoGaze hidden-state cache dir. "
                        "Default: <output_dir>/hidden_cache. Point at a prior run's cache "
                        "to skip the multi-hour re-cache when re-using the same split.")
    p.add_argument("--val_hidden_cache_dir", default=None,
                   help="Override the val-split AutoGaze hidden-state cache dir. "
                        "Default: <output_dir>/val_hidden_cache (only used when train_split=train).")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader worker processes. 0 keeps loading on the main thread "
                        "(GPU starves; only useful for debugging). Default 4.")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── 1. Download data ──
    if args.train_split == "train":
        img_dir, ann_file = download_coco_train(args.data_dir)
        # Use val2017 for validation
        val_img_dir, val_ann_file = download_coco_val(args.data_dir)
    else:
        img_dir, ann_file = download_coco_val(args.data_dir)
        val_img_dir, val_ann_file = None, None  # val split from same set

    print(f"[data] Loading COCO annotations from {ann_file}")
    coco = COCO(ann_file)
    if val_ann_file and val_ann_file != ann_file:
        print(f"[data] Loading COCO val annotations from {val_ann_file}")
        coco_val = COCO(val_ann_file)
    else:
        coco_val = None

    # Category mapping
    cat_info = coco.loadCats(coco.getCatIds())
    categories = {c["id"]: c["name"] for c in cat_info}
    print(f"[data] {len(categories)} categories: {list(categories.values())[:10]}...")

    # ── 2. Cache hidden states ──
    hidden_cache_dir = args.hidden_cache_dir or os.path.join(args.output_dir, "hidden_cache")
    os.makedirs(hidden_cache_dir, exist_ok=True)
    print(f"[cache] train hidden-state cache dir: {hidden_cache_dir}")
    cache_autogaze_hidden_states(
        img_dir, coco, hidden_cache_dir, device, max_images=args.max_images,
    )
    if coco_val is not None:
        val_hidden_cache_dir = args.val_hidden_cache_dir or os.path.join(args.output_dir, "val_hidden_cache")
        os.makedirs(val_hidden_cache_dir, exist_ok=True)
        print(f"[cache] val hidden-state cache dir: {val_hidden_cache_dir}")
        cache_autogaze_hidden_states(
            val_img_dir, coco_val, val_hidden_cache_dir, device,
        )
    else:
        val_hidden_cache_dir = hidden_cache_dir

    # ── 2b. Optional: Cache CLIP vision features ──
    clip_vision_cache_dir = None
    val_clip_vision_cache_dir = None
    if args.clip_vision:
        clip_vision_cache_dir = os.path.join(args.output_dir, "clip_vision_cache")
        cache_clip_vision_features(img_dir, coco, clip_vision_cache_dir, device, max_images=args.max_images)
        if coco_val is not None:
            val_clip_vision_cache_dir = os.path.join(args.output_dir, "val_clip_vision_cache")
            cache_clip_vision_features(val_img_dir, coco_val, val_clip_vision_cache_dir, device)
        else:
            val_clip_vision_cache_dir = clip_vision_cache_dir

    # ── 2c. Optional: On-the-fly CLIP vision features ──
    clip_vision_online = None
    clip_preprocess = None
    if args.clip_vision_online:
        if args.clip_vision:
            print("[warn] --clip_vision and --clip_vision_online both set; using online mode")
        clip_vision_online = CLIPVisionOnline(device)
        clip_preprocess = clip_vision_online.preprocess

    # ── 3. Cache CLIP text embeddings ──
    clip_cache_path = os.path.join(args.output_dir, "clip_text_embeddings.pt")
    clip_embeddings = cache_clip_text_embeddings(categories, clip_cache_path, device)

    # ── 4. Optional: CLIPSeg soft targets ──
    clipseg_cache_dir = None
    if args.clipseg_mix > 0:
        # Reuse a top-level CLIPSeg cache so multiple runs share generation cost.
        clipseg_cache_dir = os.path.join(args.data_dir, "clipseg_cache")
        all_img_ids = sorted(coco.getImgIds())
        if args.max_images:
            all_img_ids = all_img_ids[:args.max_images]
        generate_clipseg_targets(
            img_dir, coco, all_img_ids, categories, clipseg_cache_dir, device,
            target_grid=args.out_grid, include_negatives=args.clipseg_neg,
        )

    # ── 5. Train/val split ──
    if coco_val is not None:
        # Separate train/val COCO sets (e.g. train2017 + val2017)
        train_ids = sorted(coco.getImgIds())
        if args.max_images:
            train_ids = train_ids[:args.max_images]
        val_ids = sorted(coco_val.getImgIds())
        val_coco_ref = coco_val
        val_cache_dir = val_hidden_cache_dir
        print(f"[data] Split: {len(train_ids)} train (train2017), {len(val_ids)} val (val2017)")
    else:
        all_img_ids = sorted(coco.getImgIds())
        if args.max_images:
            all_img_ids = all_img_ids[:args.max_images]
        random.shuffle(all_img_ids)
        split = int(len(all_img_ids) * (1 - args.val_split))
        train_ids = all_img_ids[:split]
        val_ids = all_img_ids[split:]
        val_coco_ref = coco
        val_cache_dir = hidden_cache_dir
        print(f"[data] Split: {len(train_ids)} train, {len(val_ids)} val images")

    # Resolve CLIP vision cache dirs for val
    train_cv_dir = clip_vision_cache_dir
    val_cv_dir = val_clip_vision_cache_dir if coco_val is not None else clip_vision_cache_dir

    # Resolve image dirs for on-the-fly CLIP vision (None when not using online mode)
    train_img_dir_for_clip = img_dir if clip_preprocess else None
    val_img_dir_for_clip = (val_img_dir if coco_val is not None else img_dir) if clip_preprocess else None

    if args.clipseg_mix > 0:
        train_dataset = CocoSegWithClipSegDataset(
            coco, train_ids, hidden_cache_dir, clip_embeddings, categories,
            clipseg_cache_dir=clipseg_cache_dir,
            clipseg_mix_ratio=args.clipseg_mix,
            target_grid=args.out_grid,
            neg_per_image=args.neg_per_image,
            hard_semantic_neg_per_image=args.hard_semantic_neg_per_image,
            within_image_neg=args.within_image_neg,
            clip_vision_cache_dir=train_cv_dir,
            img_dir=train_img_dir_for_clip,
            clip_preprocess=clip_preprocess,
        )
        val_dataset = CocoSegDataset(
            val_coco_ref, val_ids, val_cache_dir, clip_embeddings, categories,
            target_grid=args.out_grid,
            neg_per_image=args.neg_per_image,
            hard_semantic_neg_per_image=args.hard_semantic_neg_per_image,
            within_image_neg=args.within_image_neg,
            clip_vision_cache_dir=val_cv_dir,
            img_dir=val_img_dir_for_clip,
            clip_preprocess=clip_preprocess,
        )
    else:
        train_dataset = CocoSegDataset(
            coco, train_ids, hidden_cache_dir, clip_embeddings, categories,
            target_grid=args.out_grid,
            neg_per_image=args.neg_per_image,
            hard_semantic_neg_per_image=args.hard_semantic_neg_per_image,
            within_image_neg=args.within_image_neg,
            clip_vision_cache_dir=train_cv_dir,
            img_dir=train_img_dir_for_clip,
            clip_preprocess=clip_preprocess,
        )
        val_dataset = CocoSegDataset(
            val_coco_ref, val_ids, val_cache_dir, clip_embeddings, categories,
            target_grid=args.out_grid,
            neg_per_image=args.neg_per_image,
            hard_semantic_neg_per_image=args.hard_semantic_neg_per_image,
            within_image_neg=args.within_image_neg,
            clip_vision_cache_dir=val_cv_dir,
            img_dir=val_img_dir_for_clip,
            clip_preprocess=clip_preprocess,
        )

    print(f"[data] {len(train_dataset)} train samples, {len(val_dataset)} val samples")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # ── 6. Model ──
    use_clip_vision = args.clip_vision or args.clip_vision_online
    hidden_dim = 192 + (768 if use_clip_vision else 0)
    if args.decoder:
        head = BigHeadDecoder(
            hidden_dim=hidden_dim,
            embedding_dim=512,
            expanded_dim=args.expanded_dim,
            n_attn_heads=args.n_attn_heads,
            n_attn_layers=args.n_attn_layers,
            decoder_dim=args.decoder_dim,
            out_grid=args.out_grid,
            in_grid=PATCH_GRID,
            dropout=args.dropout,
        ).to(device)
        model_kind = "BigHeadDecoder"
    else:
        assert args.out_grid == PATCH_GRID, "BigHead only supports out_grid=14; pass --decoder for higher."
        head = BigHead(
            hidden_dim=hidden_dim,
            embedding_dim=512,
            expanded_dim=args.expanded_dim,
            n_attn_heads=args.n_attn_heads,
            n_attn_layers=args.n_attn_layers,
            dropout=args.dropout,
        ).to(device)
        model_kind = "BigHead"
    param_count = sum(p.numel() for p in head.parameters()) / 1e3
    print(f"[model] {model_kind}: {param_count:.1f}K params "
          f"(e={args.expanded_dim}, L={args.n_attn_layers}, h={args.n_attn_heads}, "
          f"out_grid={args.out_grid})")

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=True)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        missing, unexpected = head.load_state_dict(sd, strict=False)
        print(f"[init] Loaded head from {args.init_from}; "
              f"missing={len(missing)} unexpected={len(unexpected)}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # ── 7. wandb ──
    run_name = args.run_name or f"coco-seg-e{args.expanded_dim}-L{args.n_attn_layers}"
    if "/" in args.wandb_project:
        entity, project = args.wandb_project.split("/", 1)
    else:
        entity, project = None, args.wandb_project
    wandb.init(entity=entity, project=project, name=run_name, config=vars(args))

    # ── 8. Training loop ──
    best_val_bce = float("inf")
    print(f"\n[train] Starting {args.num_epochs} epochs...")
    t0 = time.time()

    for epoch in range(args.num_epochs):
        train_metrics = train_epoch(
            head, train_loader, optimizer, device, epoch, args.num_epochs,
            dice_weight=args.dice_weight,
            use_focal=args.focal,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            other_neg_weight=args.other_neg_weight,
            clip_vision_online=clip_vision_online,
        )
        val_loss = validate(head, val_loader, device, clip_vision_online=clip_vision_online)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_metrics["bce"]
        print(f"  Epoch {epoch+1}: train_bce={train_loss:.4f} "
              f"train_dice={train_metrics['dice']:.4f} val_bce={val_loss:.4f} lr={lr:.6f}")
        wandb.log({
            "train/bce": train_loss,
            "train/dice": train_metrics["dice"],
            "train/other": train_metrics.get("other", 0.0),
            "train/total": train_metrics["total"],
            "val/bce": val_loss,
            "train/lr": lr,
            "epoch": epoch + 1,
        })

        if val_loss < best_val_bce:
            best_val_bce = val_loss
            ckpt = {
                "state_dict": head.state_dict(),
                "config": {
                    "hidden_dim": hidden_dim,
                    "embedding_dim": 512,
                    "expanded_dim": args.expanded_dim,
                    "n_attn_heads": args.n_attn_heads,
                    "n_attn_layers": args.n_attn_layers,
                    "model_kind": model_kind,
                    "out_grid": args.out_grid,
                    "decoder_dim": args.decoder_dim if args.decoder else None,
                    "clip_vision": use_clip_vision,
                },
                "epoch": epoch + 1,
                "val_bce": val_loss,
            }
            torch.save(ckpt, os.path.join(args.output_dir, "best_head.pt"))
            wandb.log({"val/best_bce": best_val_bce})
            print(f"  ↑ New best val_bce={best_val_bce:.4f}")

        torch.save(
            {"state_dict": head.state_dict(), "epoch": epoch + 1, "val_bce": val_loss,
             "config": {"hidden_dim": hidden_dim, "embedding_dim": 512,
                        "expanded_dim": args.expanded_dim,
                        "n_attn_heads": args.n_attn_heads,
                        "n_attn_layers": args.n_attn_layers}},
            os.path.join(args.output_dir, "latest_head.pt"),
        )

    elapsed = time.time() - t0
    print(f"\n[train] Done in {elapsed/60:.1f} min. Best val_bce={best_val_bce:.4f}")

    # ── 9. Qualitative evaluation (on val set) ──
    print(f"\n[eval] Generating {args.num_qual_examples} qualitative examples...")
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_head.pt"), map_location=device)
    head.load_state_dict(best_ckpt["state_dict"])
    qual_dir = os.path.join(args.output_dir, "qualitative")
    qual_coco = coco_val if coco_val is not None else coco
    qual_img_dir = val_img_dir if val_img_dir is not None else img_dir
    save_qualitative_examples(
        head, qual_coco, qual_img_dir, val_cache_dir, clip_embeddings,
        categories, device, qual_dir, num_examples=args.num_qual_examples,
        clip_vision_cache_dir=val_clip_vision_cache_dir,
        clip_vision_online=clip_vision_online,
    )

    wandb.finish()
    print(f"\n[done] Checkpoints in {args.output_dir}/")
    print(f"[done] Qualitative examples in {qual_dir}/")
    print(f"[done] Best val BCE: {best_val_bce:.4f}")


if __name__ == "__main__":
    main()
