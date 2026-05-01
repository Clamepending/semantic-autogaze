"""Render v0.6.0 qualitative review panels:
  1. REGRESSION categories where v0.6.0 lost vs v0.5.0 (frisbee, tennis racket,
     wine glass, stop sign, umbrella). Each: input | GT | v0.6.0 heatmap | overlap.
  2. ABSTENTION test: pair each of 8 COCO val images with a query that is
     NOT in the image (we know its actual category, so we query something
     different). Heatmap should produce LOW scores everywhere when the head
     correctly abstains; ANY high-confidence cell on an absent query is a
     visible false positive.

Output:
  /home/ogata/mac-brain/projects/semantic-autogaze/figures/v060_review/
    regression_cats.png
    abstention_grid.png
"""
from __future__ import annotations
import argparse, os, sys, random
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image as _PIL

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")

from pycocotools.coco import COCO
from scripts.eval_phase2_ckpt import load_ckpt, heatmap_one, GRID

COCO_ROOT = "/home/ogata/semantic-autogaze/data/coco_val2017"

REGRESSION_CATS = ["frisbee", "tennis racket", "wine glass", "stop sign", "umbrella"]

# (image_category, absent_query) — query is something NOT in the image
ABSTENTION_PAIRS = [
    # image is a bear scene → query = laptop (clearly absent)
    ("bear", "laptop"),
    # image is a dog scene → query = bicycle
    ("dog", "bicycle"),
    # image is a cat scene → query = car
    ("cat", "car"),
    # image is a person scene → query = elephant
    ("person", "elephant"),
    # image is a pizza scene → query = airplane
    ("pizza", "airplane"),
    # image is a giraffe → query = phone
    ("giraffe", "phone"),
    # image is a stop sign → query = banana
    ("stop sign", "banana"),
    # image is a frisbee → query = pen (small / never trained for)
    ("frisbee", "pen"),
]


def render_panel(rows, save_path, title, bb_fn, head, sb, mean, std, device,
                 text_model, tok, coco, name2cat, mode="regression"):
    """rows = list of (header_label, cat_for_image_pick, query_text)."""
    n = len(rows)
    fig, axs = plt.subplots(n, 4, figsize=(13, 3.0 * n))
    if n == 1: axs = axs[None, :]

    for r, (label, img_cat, query_text) in enumerate(rows):
        if img_cat not in name2cat:
            for c in range(4): axs[r, c].axis("off")
            continue
        cat_id = name2cat[img_cat]
        img_ids = coco.getImgIds(catIds=[cat_id])
        if not img_ids: continue
        # Use a deterministic mid-pick rather than smallest, so we get a
        # representative image not the one with edge-case annotations.
        img_id = sorted(img_ids)[len(img_ids) // 2]
        info = coco.loadImgs([img_id])[0]
        pil = _PIL.open(os.path.join(COCO_ROOT, "val2017", info["file_name"])).convert("RGB")
        W, H = pil.size

        # GT mask for the IMAGE'S actual category (img_cat) at 14×14
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cat_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        full_mask = np.zeros((H, W), dtype=np.uint8)
        for a in anns: full_mask = np.maximum(full_mask, coco.annToMask(a))
        gt14 = np.array(_PIL.fromarray(full_mask * 255).resize((GRID, GRID), _PIL.BILINEAR),
                        dtype=np.float32) / 255.0

        # Run head on the QUERY text (which may differ from img_cat for abstention)
        heat = heatmap_one(pil, query_text, bb_fn, head, sb, mean, std, text_model, tok, device)

        # Stats
        hmax = float(heat.max()); hmean = float(heat.mean())
        n_gt = int((gt14 > 0.5).sum())
        if mode == "regression" and n_gt > 0:
            thresh = np.partition(heat.ravel(), -n_gt)[-n_gt]
            pred_bin = (heat >= thresh).astype(np.float32)
            gt_bin = (gt14 > 0.5).astype(np.float32)
            inter = float((pred_bin * gt_bin).sum())
            union = float(((pred_bin + gt_bin) > 0).sum())
            iou = inter / max(union, 1)
        else:
            iou = float("nan")

        axs[r, 0].imshow(pil)
        axs[r, 0].set_title(f"{label}\n(img={img_cat}, q='{query_text}')", fontsize=9)
        axs[r, 0].axis("off")

        axs[r, 1].imshow(gt14, cmap="gray", vmin=0, vmax=1)
        axs[r, 1].set_title(f"GT for img_cat={img_cat}\n({n_gt}/196 cells)", fontsize=9)
        axs[r, 1].axis("off")

        axs[r, 2].imshow(heat, cmap="jet", vmin=0, vmax=1)
        axs[r, 2].set_title(f"v0.6.0 heatmap (q='{query_text}')\nmax={hmax:.2f} mean={hmean:.2f}",
                            fontsize=9)
        axs[r, 2].axis("off")

        if mode == "regression" and n_gt > 0:
            rgb = np.zeros((GRID, GRID, 3), dtype=np.float32)
            rgb[..., 1] = pred_bin * gt_bin
            rgb[..., 0] = pred_bin * (1 - gt_bin)
            rgb[..., 2] = (1 - pred_bin) * gt_bin
            axs[r, 3].imshow(rgb)
            axs[r, 3].set_title(f"top-{n_gt} match\nIoU = {iou:.3f}", fontsize=9)
        else:
            # Abstention mode: show the heatmap with absolute threshold 0.45
            kept = (heat >= 0.45).astype(np.float32)
            axs[r, 3].imshow(kept, cmap="hot", vmin=0, vmax=1)
            n_above = int(kept.sum())
            axs[r, 3].set_title(f"@ τ=0.45: {n_above}/196 cells fire\n"
                                f"({'CLEAN abstain' if n_above == 0 else 'FALSE POSITIVE'})",
                                fontsize=9)
        axs[r, 3].axis("off")

        iou_str = f"{iou:.3f}" if iou == iou else "n/a"
        print(f"  [{label:25s}] img={info['file_name']:18s} hmax={hmax:.3f} hmean={hmean:.3f} iou={iou_str}",
              flush=True)

    fig.suptitle(title, fontsize=11, y=0.998)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    print(f"[saved] {save_path}", flush=True)


def main(args):
    device = torch.device(args.device)
    bb_fn, head, sb, mean, std, model, kind, bb_module = load_ckpt(args.ckpt, device)
    print(f"[ckpt] loaded {args.ckpt}", flush=True)

    import open_clip
    text_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    text_model = text_model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-B-16")

    coco = COCO(os.path.join(COCO_ROOT, "annotations", "instances_val2017.json"))
    name2cat = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Panel 1: REGRESSION categories (positive case — query matches image)
    print(f"\n=== regression categories (positive: query = image_cat) ===")
    rows1 = [(c, c, c) for c in REGRESSION_CATS]
    render_panel(rows1, out_dir / "regression_cats.png",
                 "v0.6.0 phase19b on REGRESSION categories — model lost vs v0.5.0 here",
                 bb_fn, head, sb, mean, std, device, text_model, tok, coco, name2cat,
                 mode="regression")

    # Panel 2: ABSTENTION (negative case — query is absent)
    print(f"\n=== abstention pairs (query NOT in image) ===")
    rows2 = [(f"{img_cat} img + '{q}' query", img_cat, q) for (img_cat, q) in ABSTENTION_PAIRS]
    render_panel(rows2, out_dir / "abstention_grid.png",
                 "v0.6.0 phase19b ABSTENTION — does the head stay quiet when query is absent?",
                 bb_fn, head, sb, mean, std, device, text_model, tok, coco, name2cat,
                 mode="abstention")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_dir", required=True)
    main(p.parse_args())
