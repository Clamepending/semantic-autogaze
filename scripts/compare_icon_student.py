"""Side-by-side comparison: IconStudent recipe A vs recipe B vs CLIPSeg GT vs COCO GT.

Uses the same v11 audit (img_id, cat) pairs so we can directly judge whether
either recipe matches CLIPSeg's heatmap quality on hard cases.

Panels: [image | COCO GT | CLIPSeg GT (teacher) | A pred | B pred]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO

from semantic_autogaze.icon_student import IconStudent
from semantic_autogaze.letterbox import (
    LetterboxInfo,
    compute_info,
    heatmap_to_original,
    letterbox_mask,
)
from semantic_autogaze.train_coco_seg import build_category_mask, get_image_categories


FILENAME_RE = re.compile(r"^\d{3}_(\d+)_(.+)\.png$")


def parse_qual_dir(qual_dir: Path) -> list[tuple[int, str]]:
    pairs = []
    for p in sorted(qual_dir.iterdir()):
        m = FILENAME_RE.match(p.name)
        if m:
            pairs.append((int(m.group(1)), m.group(2)))
    return pairs


def load_student(ckpt_path: str, device: torch.device) -> IconStudent:
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = raw["config"]
    model = IconStudent(
        patch_dim=cfg["patch_dim"],
        query_dim=cfg["query_dim"],
        decoder_dim=cfg["decoder_dim"],
        in_grid=cfg["in_grid"],
        out_grid=cfg["out_grid"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(raw["state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def predict(model: IconStudent, patches: torch.Tensor, query: torch.Tensor,
            device: torch.device, info: LetterboxInfo) -> np.ndarray:
    """Predict in letterbox-square frame, un-letterbox to (h, w), keep absolute [0, 1] scale."""
    logits = model(patches.unsqueeze(0).to(device), query.unsqueeze(0).to(device))
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    hw = heatmap_to_original(probs, info, mode="bilinear")
    return np.clip(hw, 0.0, 1.0)


def target_to_original(t: torch.Tensor, info: LetterboxInfo) -> np.ndarray:
    """Cached target lives in letterbox-square frame; un-letterbox to (h, w), keep absolute [0, 1]."""
    arr = t.float().numpy()
    hw = heatmap_to_original(arr, info, mode="bilinear")
    return np.clip(hw, 0.0, 1.0)


def overlay(img_np: np.ndarray, gray: np.ndarray, cmap) -> np.ndarray:
    rgb = (cmap(gray)[..., :3] * 255).astype(np.uint8)
    return (0.5 * img_np.astype(np.float32) + 0.5 * rgb).clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qual-dir", required=True)
    p.add_argument("--ckpt-a", required=True, help="recipe A checkpoint (CLIPSeg distill)")
    p.add_argument("--ckpt-b", required=True, help="recipe B checkpoint (COCO masks)")
    p.add_argument("--label-a", default="A: distill CLIPSeg")
    p.add_argument("--label-b", default="B: COCO masks")
    p.add_argument("--dinov2-cache", required=True)
    p.add_argument("--clipseg-cache", required=True, help="CLIPSeg target cache (g128)")
    p.add_argument("--clip-text-embeddings", required=True)
    p.add_argument("--target-grid", type=int, default=128)
    p.add_argument("--data-dir", default="/Users/mark/code/semantic-autogaze/data/coco")
    p.add_argument("--ann", default="annotations/instances_val2017.json")
    p.add_argument("--img-subdir", default="val2017")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = parse_qual_dir(Path(args.qual_dir))
    print(f"[diag] mined {len(pairs)} pairs from {args.qual_dir}")

    coco = COCO(os.path.join(args.data_dir, args.ann))
    img_dir = os.path.join(args.data_dir, args.img_subdir)
    name_to_id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}

    clip_text = torch.load(args.clip_text_embeddings, map_location="cpu", weights_only=True)
    print(f"[diag] loading {args.label_a} from {args.ckpt_a}")
    model_a = load_student(args.ckpt_a, device)
    print(f"[diag] loading {args.label_b} from {args.ckpt_b}")
    model_b = load_student(args.ckpt_b, device)

    cmap = matplotlib.colormaps["jet"]
    summary = []

    for idx, (img_id, cat_slug) in enumerate(pairs):
        cat_name = cat_slug.replace("_", " ")
        if cat_name not in name_to_id:
            print(f"  [skip] {idx:03d}: {cat_name!r} unknown")
            continue
        cat_id = name_to_id[cat_name]

        patch_path = Path(args.dinov2_cache) / f"{img_id}.pt"
        if not patch_path.exists():
            print(f"  [skip] {idx:03d}: no DINOv2 cache for {img_id}")
            continue
        patches = torch.load(patch_path, map_location="cpu", weights_only=True).float()

        img_info = coco.imgs[img_id]
        img = Image.open(os.path.join(img_dir, img_info["file_name"])).convert("RGB")
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        lb = compute_info(h, w)

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=0))
        if not anns:
            print(f"  [skip] {idx:03d}: no GT anns for {cat_name} in {img_id}")
            continue
        # COCO GT is in native (h, w) frame already; binary mask in [0, 1].
        gt_mask = build_category_mask(coco, img_id, cat_id, anns).astype(np.float32)
        gt_norm = np.clip(gt_mask, 0.0, 1.0)

        # CLIPSeg teacher target (cached at target_grid in letterbox-square frame)
        cs_path = Path(args.clipseg_cache) / f"{img_id}_g{args.target_grid}.pt"
        if cs_path.exists():
            cs_dict = torch.load(cs_path, map_location="cpu", weights_only=True)
            cs_t = cs_dict.get(str(cat_id))
            cs_up = target_to_original(cs_t, lb) if cs_t is not None else np.zeros((h, w))
        else:
            cs_up = np.zeros((h, w))

        other_cats = [
            coco.cats[c]["name"]
            for c in get_image_categories(coco, img_id).keys()
            if c != cat_id
        ]

        query = clip_text[str(cat_id)].float()
        pa = predict(model_a, patches, query, device, lb)
        pb = predict(model_b, patches, query, device, lb)

        fig, axes = plt.subplots(1, 5, figsize=(30, 6))
        axes[0].imshow(img_np); axes[0].set_title(f"image {img_id}"); axes[0].axis("off")
        axes[1].imshow(overlay(img_np, gt_norm, cmap)); axes[1].set_title(f"COCO GT \"{cat_name}\""); axes[1].axis("off")
        axes[2].imshow(overlay(img_np, cs_up, cmap)); axes[2].set_title(f"CLIPSeg teacher (max={cs_up.max():.2f})"); axes[2].axis("off")
        axes[3].imshow(overlay(img_np, pa, cmap)); axes[3].set_title(f"{args.label_a} (max={pa.max():.2f})"); axes[3].axis("off")
        axes[4].imshow(overlay(img_np, pb, cmap)); axes[4].set_title(f"{args.label_b} (max={pb.max():.2f})"); axes[4].axis("off")
        others_str = ", ".join(other_cats[:8]) + ("…" if len(other_cats) > 8 else "")
        plt.suptitle(f"query: \"{cat_name}\"  |  also present: {others_str}", fontsize=12)
        plt.tight_layout()

        out = out_dir / f"{idx:03d}_{img_id}_{cat_slug}.png"
        plt.savefig(out, dpi=110, bbox_inches="tight")
        plt.close()
        summary.append({
            "idx": idx, "img_id": img_id, "cat": cat_name,
            "other_cats": other_cats, "out": str(out),
        })
        print(f"  {idx:03d}: {cat_name:14s} on {img_id} → {out.name}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[diag] wrote {len(summary)} comparison panels to {out_dir}")


if __name__ == "__main__":
    main()
