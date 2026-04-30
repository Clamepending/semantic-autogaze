"""Visualize what the v0.5.0 ckpt produces on the failure categories
(knife / skis / tie / baseball bat / sports ball / spoon — all with
n=50 mIoU < 0.40).

For each failure category, find one COCO val2017 image that contains
the category, run the model, and save a side-by-side panel:
  [input] [GT 14x14] [model heatmap] [diff]

This tells us whether the model is:
  (a) producing zero-everywhere (under-trained / class imbalance)
  (b) producing a wrong-but-non-zero heatmap (capacity / supervision quality)
  (c) producing the right shape but at the wrong scale (calibration / threshold)

Usage:
  CUDA_VISIBLE_DEVICES="" python -m scripts.qual_failure_categories \
      --ckpt results/phase15_atto_long_50k/ckpt_step35000.pt \
      --output_dir /home/ogata/mac-brain/projects/semantic-autogaze/figures/v050_failure_qual
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")

from pycocotools.coco import COCO
from scripts.eval_phase2_ckpt import load_ckpt, heatmap_one, GRID

COCO_ROOT = "/home/ogata/semantic-autogaze/data/coco_val2017"

# Failure categories from the n=50 50-image eval (mIoU < 0.40)
FAILURE_CATEGORIES = [
    "knife", "skis", "tie", "baseball bat",
    "snowboard", "sports ball", "skateboard", "spoon",
]
# Strong categories for comparison
STRONG_CATEGORIES = ["dog", "cat", "bear"]


def main(args):
    device = torch.device(args.device)
    bb_fn, head, sb, mean, std, model, kind, bb_module = load_ckpt(args.ckpt, device)
    print(f"[ckpt] loaded {args.ckpt}", flush=True)

    import open_clip
    text_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai")
    text_model = text_model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-B-16")

    coco = COCO(os.path.join(COCO_ROOT, "annotations", "instances_val2017.json"))
    name2cat = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cats_to_plot = FAILURE_CATEGORIES + STRONG_CATEGORIES

    n_cats = len(cats_to_plot)
    fig, axs = plt.subplots(n_cats, 4, figsize=(13, 3.0 * n_cats))
    if n_cats == 1:
        axs = axs[None, :]

    for row, cat_name in enumerate(cats_to_plot):
        if cat_name not in name2cat:
            print(f"[skip] {cat_name} not in COCO categories")
            for c in range(4):
                axs[row, c].axis("off")
            continue
        cat_id = name2cat[cat_name]
        img_ids = coco.getImgIds(catIds=[cat_id])
        if not img_ids:
            print(f"[skip] {cat_name} no images")
            continue
        # Deterministic pick: smallest img_id
        img_id = min(img_ids)
        img_info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(COCO_ROOT, "val2017", img_info["file_name"])
        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size

        # Build GT mask at 14x14
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cat_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        full_mask = np.zeros((H, W), dtype=np.uint8)
        for a in anns:
            m = coco.annToMask(a)
            full_mask = np.maximum(full_mask, m)
        # Downsample to 14x14
        from PIL import Image as _PIL
        gt14 = np.array(_PIL.fromarray(full_mask * 255).resize((GRID, GRID), _PIL.BILINEAR), dtype=np.float32) / 255.0

        # Model heatmap
        heat_np = heatmap_one(pil, cat_name, bb_fn, head, sb, mean, std, text_model, tok, device)

        # IoU at top-K (K = number of GT-positive cells)
        n_gt = int((gt14 > 0.5).sum())
        if n_gt > 0:
            thresh = np.partition(heat_np.ravel(), -n_gt)[-n_gt]
            pred_bin = (heat_np >= thresh).astype(np.float32)
            gt_bin = (gt14 > 0.5).astype(np.float32)
            inter = float((pred_bin * gt_bin).sum())
            union = float(((pred_bin + gt_bin) > 0).sum())
            iou = inter / max(union, 1)
        else:
            iou = float("nan")

        # Plot
        axs[row, 0].imshow(pil)
        axs[row, 0].set_title(f"{cat_name}\n(img_id={img_id})", fontsize=10)
        axs[row, 0].axis("off")

        axs[row, 1].imshow(gt14, cmap="gray", vmin=0, vmax=1)
        axs[row, 1].set_title(f"GT 14×14\n({n_gt}/196 cells)", fontsize=10)
        axs[row, 1].axis("off")

        axs[row, 2].imshow(heat_np, cmap="jet", vmin=0, vmax=max(heat_np.max(), 1e-3))
        hmax = float(heat_np.max())
        hmean = float(heat_np.mean())
        axs[row, 2].set_title(
            f"Model heatmap\nmax={hmax:.2f} mean={hmean:.2f}",
            fontsize=10,
        )
        axs[row, 2].axis("off")

        # Overlap visualization (red=miss, green=hit, blue=GT-only)
        if n_gt > 0:
            rgb = np.zeros((GRID, GRID, 3), dtype=np.float32)
            rgb[..., 1] = pred_bin * gt_bin            # green = TP
            rgb[..., 0] = pred_bin * (1 - gt_bin)      # red = FP
            rgb[..., 2] = (1 - pred_bin) * gt_bin      # blue = FN
            axs[row, 3].imshow(rgb)
            axs[row, 3].set_title(f"top-{n_gt} match\nIoU = {iou:.3f}", fontsize=10)
        else:
            axs[row, 3].imshow(np.zeros((GRID, GRID)), cmap="gray")
            axs[row, 3].set_title("no GT", fontsize=10)
        axs[row, 3].axis("off")

        print(f"[{cat_name:18s}] img={img_id} n_gt={n_gt} hmax={hmax:.3f} hmean={hmean:.3f} iou={iou:.3f}", flush=True)

    fig.suptitle(
        f"v0.5.0 phase15 atto step 35000 — qualitative output on FAILURE categories (top 8) + STRONG controls (bottom 3)",
        fontsize=11, y=0.998,
    )
    fig.tight_layout()
    out_path = out_dir / "v050_failure_qual.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_dir", required=True)
    main(p.parse_args())
