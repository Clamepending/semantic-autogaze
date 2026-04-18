"""
Side-by-side comparison: v11 vs v13e_ft on the SAME (img_id, cat_name) pairs.

Parses v11's qualitative filenames (e.g. `001_141821_bowl.png`) to get the exact
20 (img_id, category) pairs, then runs both checkpoints on the same hidden-state
caches and renders [image | GT | v11 pred | v13e pred] panels.

This is the only rendering that lets us judge whether the within-image
hard-negative loss actually shifted the saliency-override pattern.
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

from semantic_autogaze.bighead import BigHeadDecoder
from semantic_autogaze.train_coco_seg import (
    PATCH_GRID,
    build_category_mask,
    get_image_categories,
)


FILENAME_RE = re.compile(r"^\d{3}_(\d+)_(.+)\.png$")


def parse_qual_dir(qual_dir: Path) -> list[tuple[int, str]]:
    """Returns list of (img_id, cat_name_with_underscores) from v11 qual filenames."""
    pairs = []
    for p in sorted(qual_dir.iterdir()):
        m = FILENAME_RE.match(p.name)
        if m:
            pairs.append((int(m.group(1)), m.group(2)))
    return pairs


def load_head(ckpt_path: str, device: torch.device) -> BigHeadDecoder:
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = raw["config"]
    state = raw["state_dict"]
    head = BigHeadDecoder(
        hidden_dim=cfg["hidden_dim"],
        embedding_dim=cfg["embedding_dim"],
        expanded_dim=cfg["expanded_dim"],
        n_attn_heads=cfg["n_attn_heads"],
        n_attn_layers=cfg["n_attn_layers"],
        decoder_dim=cfg["decoder_dim"],
        out_grid=cfg["out_grid"],
        in_grid=PATCH_GRID,
        dropout=0.0,
    ).to(device)
    head.load_state_dict(state)
    head.eval()
    return head


@torch.inference_mode()
def predict(head, hidden, query, device, h, w):
    logits = head(hidden.unsqueeze(0).to(device), query.unsqueeze(0).to(device))
    n_out = logits.shape[-1]
    out_grid = int(round(n_out ** 0.5))
    probs = torch.sigmoid(logits).reshape(out_grid, out_grid).cpu()
    up = F.interpolate(
        probs.unsqueeze(0).unsqueeze(0),
        size=(h, w), mode="bilinear", align_corners=False,
    )[0, 0].numpy()
    return (up - up.min()) / (up.max() - up.min() + 1e-8)


def overlay(img_np, gray, cmap):
    rgb = (cmap(gray)[..., :3] * 255).astype(np.uint8)
    return (0.5 * img_np.astype(np.float32) + 0.5 * rgb).clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qual-dir", required=True, help="v11 qualitative dir to mine for pairs")
    p.add_argument("--head-a", required=True, help="ckpt A (v11)")
    p.add_argument("--head-b", required=True, help="ckpt B (v13e_ft)")
    p.add_argument("--label-a", default="v11")
    p.add_argument("--label-b", default="v13e_ft")
    p.add_argument("--hidden-cache", required=True)
    p.add_argument("--clip-text-embeddings", required=True)
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
    print(f"[diag] mined {len(pairs)} (img_id, cat) pairs from {args.qual_dir}")

    print("[diag] loading COCO…")
    coco = COCO(os.path.join(args.data_dir, args.ann))
    img_dir = os.path.join(args.data_dir, args.img_subdir)
    name_to_id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}

    clip_text = torch.load(args.clip_text_embeddings, map_location="cpu", weights_only=True)

    print(f"[diag] loading {args.label_a} from {args.head_a}…")
    head_a = load_head(args.head_a, device)
    print(f"[diag] loading {args.label_b} from {args.head_b}…")
    head_b = load_head(args.head_b, device)

    cmap = matplotlib.colormaps["jet"]
    summary = []

    for idx, (img_id, cat_slug) in enumerate(pairs):
        cat_name = cat_slug.replace("_", " ")
        if cat_name not in name_to_id:
            print(f"  [skip] {idx:03d}: {cat_name!r} not in name_to_id")
            continue
        cat_id = name_to_id[cat_name]

        cache_path = os.path.join(args.hidden_cache, f"{img_id}.pt")
        if not os.path.exists(cache_path):
            print(f"  [skip] {idx:03d}: no hidden cache for {img_id}")
            continue
        hidden = torch.load(cache_path, map_location="cpu", weights_only=True).float()

        info = coco.imgs[img_id]
        img_path = os.path.join(img_dir, info["file_name"])
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=0))
        if not anns:
            print(f"  [skip] {idx:03d}: no anns for {cat_name} in {img_id}")
            continue
        mask = build_category_mask(coco, img_id, cat_id, anns)
        gt_norm = mask.astype(np.float32)
        gt_norm = gt_norm / (gt_norm.max() + 1e-8)

        # Cross-reference: which OTHER cats are present in this image?
        other_cats = [
            coco.cats[c]["name"]
            for c in get_image_categories(coco, img_id).keys()
            if c != cat_id
        ]

        query = clip_text[str(cat_id)].float()
        pa = predict(head_a, hidden, query, device, h, w)
        pb = predict(head_b, hidden, query, device, h, w)

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        axes[0].imshow(img_np)
        axes[0].set_title(f"image {img_id}")
        axes[0].axis("off")
        axes[1].imshow(overlay(img_np, gt_norm, cmap))
        axes[1].set_title(f"GT \"{cat_name}\"")
        axes[1].axis("off")
        axes[2].imshow(overlay(img_np, pa, cmap))
        axes[2].set_title(f"{args.label_a} pred \"{cat_name}\"")
        axes[2].axis("off")
        axes[3].imshow(overlay(img_np, pb, cmap))
        axes[3].set_title(f"{args.label_b} pred \"{cat_name}\"")
        axes[3].axis("off")
        others_str = ", ".join(other_cats[:8]) + ("…" if len(other_cats) > 8 else "")
        plt.suptitle(
            f"query: \"{cat_name}\"  |  also present: {others_str}",
            fontsize=12,
        )
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
