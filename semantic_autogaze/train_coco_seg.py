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

from semantic_autogaze.bighead import BigHead

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


def build_category_mask(coco: COCO, img_id: int, cat_id: int, anns: list) -> np.ndarray:
    """Build binary mask for all instances of a category in an image."""
    img_info = coco.imgs[img_id]
    h, w = img_info["height"], img_info["width"]
    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        m = coco.annToMask(ann)
        mask = np.maximum(mask, m)
    return mask


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
    ):
        self.coco = coco
        self.hidden_cache_dir = hidden_cache_dir
        self.clip_embeddings = clip_embeddings
        self.categories = categories

        # Build index: (img_id, cat_id, [anns]) for all valid pairs
        self.samples = []
        for img_id in img_ids:
            cache_path = os.path.join(hidden_cache_dir, f"{img_id}.pt")
            if not os.path.exists(cache_path):
                continue
            cats = get_image_categories(coco, img_id)
            for cat_id, anns in cats.items():
                if str(cat_id) in clip_embeddings:
                    self.samples.append((img_id, cat_id, anns))

        # Also add negative samples: categories NOT present in image
        self._add_negatives(img_ids)

    def _add_negatives(self, img_ids: list[int]):
        """Add negative category samples (category not in image → all-zero mask)."""
        all_cat_ids = list(self.categories.keys())
        neg_samples = []
        for img_id in img_ids:
            cache_path = os.path.join(self.hidden_cache_dir, f"{img_id}.pt")
            if not os.path.exists(cache_path):
                continue
            present_cats = set(c for i, c, _ in self.samples if i == img_id)
            absent = [c for c in all_cat_ids if c not in present_cats and str(c) in self.clip_embeddings]
            if absent:
                neg_cat = random.choice(absent)
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

        # CLIP text embedding for this category
        query = self.clip_embeddings[str(cat_id)]  # (512,)

        # Build mask target
        if anns:
            mask = build_category_mask(self.coco, img_id, cat_id, anns)
            target = mask_to_patch_target(mask, PATCH_GRID)  # (14, 14)
        else:
            target = np.zeros((PATCH_GRID, PATCH_GRID), dtype=np.float32)

        target = torch.from_numpy(target).reshape(NUM_PATCHES)  # (196,)

        return {
            "hidden": hidden,
            "query": query,
            "target": target,
            "img_id": img_id,
            "cat_id": cat_id,
        }


# ── CLIPSeg soft targets ──────────────────────────────────────────────


@torch.inference_mode()
def generate_clipseg_targets(
    img_dir: str,
    coco: COCO,
    img_ids: list[int],
    categories: dict[int, str],
    cache_dir: str,
    device: torch.device,
    max_cats_per_image: int = 5,
):
    """Generate CLIPSeg soft heatmaps for mixing with hard mask supervision."""
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    os.makedirs(cache_dir, exist_ok=True)

    done = sum(1 for i in img_ids if os.path.exists(os.path.join(cache_dir, f"{i}.pt")))
    if done == len(img_ids):
        print(f"[clipseg] All {done} images already cached")
        return

    print(f"[clipseg] Loading CLIPSeg model...")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = model.to(device).eval()

    cat_names = list(categories.values())
    cat_ids = list(categories.keys())

    for img_id in tqdm(img_ids, desc="CLIPSeg soft targets"):
        out_path = os.path.join(cache_dir, f"{img_id}.pt")
        if os.path.exists(out_path):
            continue

        img_info = coco.imgs[img_id]
        img = Image.open(os.path.join(img_dir, img_info["file_name"])).convert("RGB")

        present_cats = get_image_categories(coco, img_id)
        query_cat_ids = list(present_cats.keys())[:max_cats_per_image]
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

        # Downsample to 14x14
        soft = torch.sigmoid(logits)
        soft_14 = F.adaptive_avg_pool2d(soft.unsqueeze(1), (PATCH_GRID, PATCH_GRID))
        soft_14 = soft_14.squeeze(1).cpu()  # (N_queries, 14, 14)

        result = {}
        for i, cat_id in enumerate(query_cat_ids):
            if i < len(soft_14):
                result[str(cat_id)] = soft_14[i].to(torch.float16)

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
    ):
        self.coco = coco
        self.hidden_cache_dir = hidden_cache_dir
        self.clip_embeddings = clip_embeddings
        self.categories = categories
        self.clipseg_cache_dir = clipseg_cache_dir
        self.clipseg_mix_ratio = clipseg_mix_ratio

        # Build index of (img_id, cat_id, anns, source) tuples
        self.samples = []
        for img_id in img_ids:
            cache_path = os.path.join(hidden_cache_dir, f"{img_id}.pt")
            if not os.path.exists(cache_path):
                continue
            cats = get_image_categories(coco, img_id)
            for cat_id, anns in cats.items():
                if str(cat_id) in clip_embeddings:
                    self.samples.append((img_id, cat_id, anns, "mask"))

            # Add negative sample
            all_cat_ids = list(categories.keys())
            present = set(cats.keys())
            absent = [c for c in all_cat_ids if c not in present and str(c) in clip_embeddings]
            if absent:
                neg = random.choice(absent)
                self.samples.append((img_id, neg, [], "mask"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, cat_id, anns, source = self.samples[idx]

        hidden = torch.load(
            os.path.join(self.hidden_cache_dir, f"{img_id}.pt"),
            map_location="cpu",
            weights_only=True,
        ).float()

        query = self.clip_embeddings[str(cat_id)]

        # Decide whether to use CLIPSeg soft target or hard mask
        use_clipseg = (
            self.clipseg_cache_dir
            and random.random() < self.clipseg_mix_ratio
            and anns  # only for positive samples
        )

        if use_clipseg:
            clipseg_path = os.path.join(self.clipseg_cache_dir, f"{img_id}.pt")
            if os.path.exists(clipseg_path):
                clipseg_data = torch.load(clipseg_path, map_location="cpu", weights_only=True)
                if str(cat_id) in clipseg_data:
                    target = clipseg_data[str(cat_id)].float().reshape(NUM_PATCHES)
                    return {"hidden": hidden, "query": query, "target": target,
                            "img_id": img_id, "cat_id": cat_id}

        # Fall back to hard mask
        if anns:
            mask = build_category_mask(self.coco, img_id, cat_id, anns)
            target = mask_to_patch_target(mask, PATCH_GRID)
        else:
            target = np.zeros((PATCH_GRID, PATCH_GRID), dtype=np.float32)

        target = torch.from_numpy(target).reshape(NUM_PATCHES)

        return {"hidden": hidden, "query": query, "target": target,
                "img_id": img_id, "cat_id": cat_id}


# ── Training ───────────────────────────────────────────────────────────


def train_epoch(head, loader, optimizer, device, epoch, total_epochs):
    head.train()
    losses = []
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [train]")
    for batch in pbar:
        hidden = batch["hidden"].to(device)
        query = batch["query"].to(device)
        target = batch["target"].to(device)

        logits = head(hidden, query)  # (B, 196)
        loss = F.binary_cross_entropy_with_logits(logits, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return sum(losses) / len(losses)


@torch.inference_mode()
def validate(head, loader, device):
    head.eval()
    losses = []
    for batch in loader:
        hidden = batch["hidden"].to(device)
        query = batch["query"].to(device)
        target = batch["target"].to(device)

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

        # Predict
        hidden = torch.load(cache_path, map_location="cpu", weights_only=True).float()
        query = clip_embeddings[str(cat_id)]
        logits = head(hidden.unsqueeze(0).to(device), query.unsqueeze(0).to(device))
        probs = torch.sigmoid(logits).reshape(PATCH_GRID, PATCH_GRID).cpu().numpy()

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
                   help="Limit images for faster iteration (None = all 5K val)")
    p.add_argument("--expanded_dim", type=int, default=512)
    p.add_argument("--n_attn_heads", type=int, default=8)
    p.add_argument("--n_attn_layers", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--clipseg_mix", type=float, default=0.0,
                   help="Fraction of samples using CLIPSeg soft targets (0=disabled)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", default="semantic-autogaze")
    p.add_argument("--run_name", default=None)
    p.add_argument("--num_qual_examples", type=int, default=20)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── 1. Download data ──
    img_dir, ann_file = download_coco_val(args.data_dir)
    print(f"[data] Loading COCO annotations from {ann_file}")
    coco = COCO(ann_file)

    # Category mapping
    cat_info = coco.loadCats(coco.getCatIds())
    categories = {c["id"]: c["name"] for c in cat_info}
    print(f"[data] {len(categories)} categories: {list(categories.values())[:10]}...")

    # ── 2. Cache hidden states ──
    hidden_cache_dir = os.path.join(args.output_dir, "hidden_cache")
    cache_autogaze_hidden_states(
        img_dir, coco, hidden_cache_dir, device, max_images=args.max_images,
    )

    # ── 3. Cache CLIP text embeddings ──
    clip_cache_path = os.path.join(args.output_dir, "clip_text_embeddings.pt")
    clip_embeddings = cache_clip_text_embeddings(categories, clip_cache_path, device)

    # ── 4. Optional: CLIPSeg soft targets ──
    clipseg_cache_dir = None
    if args.clipseg_mix > 0:
        clipseg_cache_dir = os.path.join(args.output_dir, "clipseg_cache")
        all_img_ids = sorted(coco.getImgIds())
        if args.max_images:
            all_img_ids = all_img_ids[:args.max_images]
        generate_clipseg_targets(
            img_dir, coco, all_img_ids, categories, clipseg_cache_dir, device,
        )

    # ── 5. Train/val split ──
    all_img_ids = sorted(coco.getImgIds())
    if args.max_images:
        all_img_ids = all_img_ids[:args.max_images]
    random.shuffle(all_img_ids)
    split = int(len(all_img_ids) * (1 - args.val_split))
    train_ids = all_img_ids[:split]
    val_ids = all_img_ids[split:]
    print(f"[data] Split: {len(train_ids)} train, {len(val_ids)} val images")

    DatasetClass = CocoSegWithClipSegDataset if args.clipseg_mix > 0 else CocoSegDataset

    if args.clipseg_mix > 0:
        train_dataset = DatasetClass(
            coco, train_ids, hidden_cache_dir, clip_embeddings, categories,
            clipseg_cache_dir=clipseg_cache_dir,
            clipseg_mix_ratio=args.clipseg_mix,
        )
        val_dataset = CocoSegDataset(
            coco, val_ids, hidden_cache_dir, clip_embeddings, categories,
        )
    else:
        train_dataset = CocoSegDataset(
            coco, train_ids, hidden_cache_dir, clip_embeddings, categories,
        )
        val_dataset = CocoSegDataset(
            coco, val_ids, hidden_cache_dir, clip_embeddings, categories,
        )

    print(f"[data] {len(train_dataset)} train samples, {len(val_dataset)} val samples")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0,
    )

    # ── 6. Model ──
    head = BigHead(
        hidden_dim=192,
        embedding_dim=512,
        expanded_dim=args.expanded_dim,
        n_attn_heads=args.n_attn_heads,
        n_attn_layers=args.n_attn_layers,
    ).to(device)
    param_count = sum(p.numel() for p in head.parameters()) / 1e3
    print(f"[model] BigHead: {param_count:.1f}K params "
          f"(e={args.expanded_dim}, L={args.n_attn_layers}, h={args.n_attn_heads})")

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
        train_loss = train_epoch(head, train_loader, optimizer, device, epoch, args.num_epochs)
        val_loss = validate(head, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1}: train_bce={train_loss:.4f} val_bce={val_loss:.4f} lr={lr:.6f}")
        wandb.log({
            "train/bce": train_loss,
            "val/bce": val_loss,
            "train/lr": lr,
            "epoch": epoch + 1,
        })

        if val_loss < best_val_bce:
            best_val_bce = val_loss
            ckpt = {
                "state_dict": head.state_dict(),
                "config": {
                    "hidden_dim": 192,
                    "embedding_dim": 512,
                    "expanded_dim": args.expanded_dim,
                    "n_attn_heads": args.n_attn_heads,
                    "n_attn_layers": args.n_attn_layers,
                },
                "epoch": epoch + 1,
                "val_bce": val_loss,
            }
            torch.save(ckpt, os.path.join(args.output_dir, "best_head.pt"))
            wandb.log({"val/best_bce": best_val_bce})
            print(f"  ↑ New best val_bce={best_val_bce:.4f}")

        torch.save(
            {"state_dict": head.state_dict(), "epoch": epoch + 1, "val_bce": val_loss,
             "config": {"hidden_dim": 192, "embedding_dim": 512,
                        "expanded_dim": args.expanded_dim,
                        "n_attn_heads": args.n_attn_heads,
                        "n_attn_layers": args.n_attn_layers}},
            os.path.join(args.output_dir, "latest_head.pt"),
        )

    elapsed = time.time() - t0
    print(f"\n[train] Done in {elapsed/60:.1f} min. Best val_bce={best_val_bce:.4f}")

    # ── 9. Qualitative evaluation ──
    print(f"\n[eval] Generating {args.num_qual_examples} qualitative examples...")
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_head.pt"), map_location=device)
    head.load_state_dict(best_ckpt["state_dict"])
    qual_dir = os.path.join(args.output_dir, "qualitative")
    save_qualitative_examples(
        head, coco, img_dir, hidden_cache_dir, clip_embeddings,
        categories, device, qual_dir, num_examples=args.num_qual_examples,
    )

    wandb.finish()
    print(f"\n[done] Checkpoints in {args.output_dir}/")
    print(f"[done] Qualitative examples in {qual_dir}/")
    print(f"[done] Best val BCE: {best_val_bce:.4f}")


if __name__ == "__main__":
    main()
