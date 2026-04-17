"""
Quantitative evaluation of a trained head checkpoint on COCO val2017.

Metrics:
  - IoU at multiple thresholds (0.1, 0.2, ..., 0.5)
  - Precision and recall at those thresholds
  - Mean foreground vs background sigmoid probability (separation gap)
  - Max sigmoid probability
  - False positive rate on negative samples (categories NOT present)

Results saved to JSON: {output_dir}/eval_metrics.json

Run:
    cd /Users/mark/code/semantic-autogaze
    python scripts/eval_quantitative.py \
        --head-ckpt results/coco_seg/best_head.pt \
        --hidden-cache-dir results/coco_seg/hidden_cache \
        --output-dir results/eval
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from tqdm import tqdm

# Import utilities from train_coco_seg
from semantic_autogaze.train_coco_seg import (
    PATCH_GRID,
    CLIPVisionOnline,
    build_category_mask,
    cache_clip_text_embeddings,
    download_coco_val,
    get_image_categories,
    mask_to_patch_target,
)
from semantic_autogaze.inference import load_head


def compute_iou_at_threshold(pred_probs: np.ndarray, target: np.ndarray, threshold: float) -> float:
    """
    Compute IoU at a specific probability threshold.

    Args:
        pred_probs: (grid, grid) sigmoid probabilities
        target: (grid, grid) binary ground truth mask (0 or 1)
        threshold: probability threshold for foreground

    Returns:
        IoU value in [0, 1]
    """
    pred_binary = (pred_probs >= threshold).astype(np.float32)
    intersection = np.sum(pred_binary * target)
    union = np.sum(np.maximum(pred_binary, target))

    if union == 0:
        # Both empty -> IoU = 1
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_precision_recall_at_threshold(
    pred_probs: np.ndarray, target: np.ndarray, threshold: float
) -> tuple[float, float]:
    """
    Compute precision and recall at a specific probability threshold.

    Args:
        pred_probs: (grid, grid) sigmoid probabilities
        target: (grid, grid) binary ground truth mask (0 or 1)
        threshold: probability threshold for foreground

    Returns:
        (precision, recall)
    """
    pred_binary = (pred_probs >= threshold).astype(np.float32)

    tp = np.sum(pred_binary * target)
    fp = np.sum(pred_binary * (1 - target))
    fn = np.sum((1 - pred_binary) * target)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return float(precision), float(recall)


def upsample_to_patch_grid(
    logits: torch.Tensor, target_grid: int = PATCH_GRID
) -> np.ndarray:
    """
    Upsample logits to the target patch grid if necessary.

    Args:
        logits: (output_size,) or output is at some grid resolution
        target_grid: desired output grid size (14)

    Returns:
        (target_grid, target_grid) numpy array
    """
    # Infer current grid from logit count
    current_size = int(np.sqrt(logits.shape[-1]))

    # Reshape to spatial grid
    logits_grid = logits.reshape(current_size, current_size).unsqueeze(0).unsqueeze(0)

    if current_size == target_grid:
        return logits_grid.squeeze(0).squeeze(0).numpy()

    # Interpolate to target grid
    upsampled = F.interpolate(
        logits_grid,
        size=(target_grid, target_grid),
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.squeeze(0).squeeze(0).numpy()


@torch.inference_mode()
def evaluate_head(
    head: torch.nn.Module,
    coco: COCO,
    img_ids: list[int],
    hidden_cache_dir: str,
    clip_embeddings: dict[str, torch.Tensor],
    categories: dict[int, str],
    device: torch.device,
    thresholds: list[float] = None,
    max_images: Optional[int] = None,
    clip_vision_cache_dir: Optional[str] = None,
    clip_vision_online: Optional[CLIPVisionOnline] = None,
    img_dir: Optional[str] = None,
) -> dict:
    """
    Evaluate head on COCO val2017 data.

    Args:
        head: loaded head module (BigHead or BigHeadDecoder)
        coco: COCO dataset object
        img_ids: list of image IDs to evaluate
        hidden_cache_dir: directory with cached hidden states
        clip_embeddings: dict of category_id -> embedding tensor
        categories: dict of category_id -> category_name
        device: torch device
        thresholds: list of IoU thresholds to evaluate (default: [0.1, 0.2, 0.3, 0.4, 0.5])
        max_images: if set, limit evaluation to first N images

    Returns:
        dict of aggregated metrics
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    head.eval()

    # Initialize metric accumulators
    ious_by_threshold = {t: [] for t in thresholds}
    prec_by_threshold = {t: [] for t in thresholds}
    rec_by_threshold = {t: [] for t in thresholds}

    fg_probs = []  # sigmoid probs on foreground patches
    bg_probs = []  # sigmoid probs on background patches
    max_probs = []

    # Track false positives on negative samples
    neg_max_probs = []  # max probs when category is absent

    positive_samples = 0
    negative_samples = 0

    eval_img_ids = img_ids[:max_images] if max_images else img_ids

    pbar = tqdm(eval_img_ids, desc="Evaluating")
    for img_id in pbar:
        cache_path = os.path.join(hidden_cache_dir, f"{img_id}.pt")
        if not os.path.exists(cache_path):
            continue

        # Load cached hidden states
        hidden = torch.load(cache_path, map_location="cpu", weights_only=True).float()

        # Optionally concatenate CLIP vision features
        if clip_vision_cache_dir:
            cv_path = os.path.join(clip_vision_cache_dir, f"{img_id}.pt")
            if os.path.exists(cv_path):
                clip_vis = torch.load(cv_path, map_location="cpu", weights_only=True).float()
                hidden = torch.cat([hidden, clip_vis], dim=-1)
        elif clip_vision_online is not None and img_dir is not None:
            from PIL import Image
            img_info = coco.imgs[img_id]
            img_path = os.path.join(img_dir, img_info["file_name"])
            img = Image.open(img_path).convert("RGB")
            clip_img = clip_vision_online.preprocess(img).unsqueeze(0).to(device)
            clip_vis = clip_vision_online.extract_batch(clip_img)[0].cpu()
            hidden = torch.cat([hidden, clip_vis], dim=-1)

        # Get categories present in this image
        present_cats = get_image_categories(coco, img_id)
        if not present_cats:
            continue

        # ── Positive samples: categories present ──
        for cat_id, anns in present_cats.items():
            if str(cat_id) not in clip_embeddings:
                continue

            # Build ground truth mask
            mask = build_category_mask(coco, img_id, cat_id, anns)
            target_14 = mask_to_patch_target(mask, PATCH_GRID)

            # Run head forward pass
            query = clip_embeddings[str(cat_id)].to(device)
            logits = head(hidden.unsqueeze(0).to(device), query.unsqueeze(0)).squeeze(0)
            probs = torch.sigmoid(logits)

            # Upsample to patch grid if necessary
            probs_np = upsample_to_patch_grid(probs.cpu(), PATCH_GRID)

            # Compute metrics at each threshold
            for threshold in thresholds:
                iou = compute_iou_at_threshold(probs_np, target_14, threshold)
                precision, recall = compute_precision_recall_at_threshold(
                    probs_np, target_14, threshold
                )
                ious_by_threshold[threshold].append(iou)
                prec_by_threshold[threshold].append(precision)
                rec_by_threshold[threshold].append(recall)

            # Foreground vs background separation
            fg_mask = target_14 > 0.5
            bg_mask = target_14 <= 0.5

            if fg_mask.sum() > 0:
                fg_probs.append(probs_np[fg_mask].mean())
            if bg_mask.sum() > 0:
                bg_probs.append(probs_np[bg_mask].mean())

            max_probs.append(probs_np.max())
            positive_samples += 1

        # ── Negative samples: categories absent ──
        all_cat_ids = set(categories.keys())
        absent_cats = list(all_cat_ids - set(present_cats.keys()))

        # Sample a few negatives per image
        num_neg_to_eval = min(3, len(absent_cats))
        if num_neg_to_eval > 0:
            neg_sample = np.random.choice(absent_cats, size=num_neg_to_eval, replace=False)
            for neg_cat_id in neg_sample:
                if str(neg_cat_id) not in clip_embeddings:
                    continue

                query = clip_embeddings[str(neg_cat_id)].to(device)
                logits = head(hidden.unsqueeze(0).to(device), query.unsqueeze(0)).squeeze(0)
                probs = torch.sigmoid(logits)

                probs_np = upsample_to_patch_grid(probs.cpu(), PATCH_GRID)
                neg_max_probs.append(probs_np.max())
                negative_samples += 1

    # Aggregate results
    metrics = {
        "num_positive_samples": positive_samples,
        "num_negative_samples": negative_samples,
        "iou_by_threshold": {
            float(t): float(np.mean(ious_by_threshold[t])) if ious_by_threshold[t] else 0.0
            for t in thresholds
        },
        "precision_by_threshold": {
            float(t): float(np.mean(prec_by_threshold[t])) if prec_by_threshold[t] else 0.0
            for t in thresholds
        },
        "recall_by_threshold": {
            float(t): float(np.mean(rec_by_threshold[t])) if rec_by_threshold[t] else 0.0
            for t in thresholds
        },
        "mean_fg_prob": float(np.mean(fg_probs)) if fg_probs else 0.0,
        "mean_bg_prob": float(np.mean(bg_probs)) if bg_probs else 0.0,
        "fg_bg_separation_gap": float(np.mean(fg_probs) - np.mean(bg_probs))
        if (fg_probs and bg_probs)
        else 0.0,
        "mean_max_prob": float(np.mean(max_probs)) if max_probs else 0.0,
        "mean_neg_max_prob": float(np.mean(neg_max_probs)) if neg_max_probs else 0.0,
    }

    return metrics


def main():
    p = argparse.ArgumentParser(description="Evaluate head on COCO val2017")
    p.add_argument(
        "--head-ckpt",
        type=str,
        required=True,
        help="Path to head checkpoint (.pt file)",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/coco",
        help="COCO data directory",
    )
    p.add_argument(
        "--hidden-cache-dir",
        type=str,
        required=True,
        help="Directory with cached AutoGaze hidden states",
    )
    p.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device: mps, cuda, or cpu",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/eval",
        help="Output directory for results",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit evaluation to first N images (None = all)",
    )
    p.add_argument(
        "--clip-vision-cache-dir",
        type=str,
        default=None,
        help="Directory with cached CLIP vision features (for v10d+ models)",
    )
    p.add_argument(
        "--clip-vision-online",
        action="store_true",
        help="Compute CLIP vision features on-the-fly (no disk cache needed)",
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Load data ──
    print("[data] Downloading/verifying COCO val2017...")
    img_dir, ann_file = download_coco_val(args.data_dir)
    coco = COCO(ann_file)

    # Category mapping
    cat_info = coco.loadCats(coco.getCatIds())
    categories = {c["id"]: c["name"] for c in cat_info}
    print(f"[data] {len(categories)} categories")

    # ── Load/cache CLIP embeddings ──
    clip_cache_path = os.path.join(args.output_dir, "clip_text_embeddings.pt")
    print("[cache] Loading/caching CLIP text embeddings...")
    clip_embeddings = cache_clip_text_embeddings(categories, clip_cache_path, device)

    # ── Load head ──
    print(f"[model] Loading head from {args.head_ckpt}...")
    if not os.path.exists(args.head_ckpt):
        raise FileNotFoundError(f"Head checkpoint not found: {args.head_ckpt}")

    head, head_type = load_head(args.head_ckpt, device)
    print(f"[model] Loaded {head_type}")

    # ── Optional: on-the-fly CLIP vision ──
    clip_vision_online = None
    if args.clip_vision_online:
        clip_vision_online = CLIPVisionOnline(device)

    # ── Run evaluation ──
    print("[eval] Starting evaluation...")
    img_ids = sorted(coco.getImgIds())

    metrics = evaluate_head(
        head,
        coco,
        img_ids,
        args.hidden_cache_dir,
        clip_embeddings,
        categories,
        device,
        thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
        max_images=args.max_images,
        clip_vision_cache_dir=args.clip_vision_cache_dir,
        clip_vision_online=clip_vision_online,
        img_dir=img_dir,
    )

    # ── Save and print results ──
    output_json = os.path.join(args.output_dir, "eval_metrics.json")
    with open(output_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Positive samples: {metrics['num_positive_samples']}")
    print(f"Negative samples: {metrics['num_negative_samples']}")
    print()

    print("IoU by threshold:")
    for t, iou in sorted(metrics["iou_by_threshold"].items()):
        print(f"  threshold {t:3.1f}: {iou:.4f}")
    print()

    print("Precision by threshold:")
    for t, prec in sorted(metrics["precision_by_threshold"].items()):
        print(f"  threshold {t:3.1f}: {prec:.4f}")
    print()

    print("Recall by threshold:")
    for t, rec in sorted(metrics["recall_by_threshold"].items()):
        print(f"  threshold {t:3.1f}: {rec:.4f}")
    print()

    print("Foreground/Background Separation:")
    print(f"  Mean foreground prob:     {metrics['mean_fg_prob']:.4f}")
    print(f"  Mean background prob:     {metrics['mean_bg_prob']:.4f}")
    print(f"  Separation gap (fg - bg): {metrics['fg_bg_separation_gap']:.4f}")
    print()

    print("Probability Statistics:")
    print(f"  Mean max prob (positive): {metrics['mean_max_prob']:.4f}")
    print(f"  Mean max prob (negative): {metrics['mean_neg_max_prob']:.4f}")
    print()

    print("=" * 70)
    print(f"Results saved to {output_json}")


if __name__ == "__main__":
    main()
