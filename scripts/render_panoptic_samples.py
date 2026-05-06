"""Render qualitative samples from COCO Panoptic to verify (image, query, mask)
alignment for the 53 stuff categories that Panoptic adds on top of COCO instances.
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def rgb2id(rgb):
    """Panoptic encoding: id = R + G*256 + B*256^2"""
    return rgb[..., 0].astype(np.int64) + (rgb[..., 1].astype(np.int64) << 8) + (rgb[..., 2].astype(np.int64) << 16)


def main(args):
    rng = random.Random(args.seed)
    print(f"[load] panoptic JSON ...")
    with open(args.ann_path) as f:
        pan = json.load(f)

    cats_by_id = {c['id']: c for c in pan['categories']}
    cat_names = [c['name'] for c in pan['categories']]
    print(f"  {len(cat_names)} categories ({sum(c['isthing']==1 for c in pan['categories'])} things, {sum(c['isthing']==0 for c in pan['categories'])} stuff)")

    # Bucket annotations by category — we want stuff samples primarily
    print(f"[index] building category buckets ...")
    buckets: dict = {}  # category_id -> list of (image_info, segment)
    img_by_id = {i['id']: i for i in pan['images']}
    for ann in pan['annotations']:
        for seg in ann['segments_info']:
            cid = seg['category_id']
            buckets.setdefault(cid, []).append((ann['image_id'], ann['file_name'], seg))

    print(f"[scan] yielding samples per category ...")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Render strips: one strip per category we care about (focus on stuff + a few thing)
    sample_cats = [c for c in pan['categories'] if c['isthing'] == 0]  # stuff focus
    if args.include_things:
        # Sample a few popular thing classes too
        sample_cats += [c for c in pan['categories'] if c['name'] in ('person', 'dog', 'cat', 'car')]

    for cat in sample_cats:
        cid = cat['id']
        name = cat['name']
        if cid not in buckets or not buckets[cid]: continue
        chosen = rng.sample(buckets[cid], k=min(len(buckets[cid]), args.n_per_query))
        n = len(chosen)
        fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
        if n == 1: axes = axes[None, :]
        for i, (img_id, file_name, seg) in enumerate(chosen):
            info = img_by_id[img_id]
            jpg_name = info['file_name']
            img_path = Path(args.image_dir) / jpg_name
            png_path = Path(args.panoptic_png_dir) / file_name
            if not img_path.exists() or not png_path.exists():
                for c in range(3): axes[i, c].axis("off")
                axes[i, 0].text(0.5, 0.5, f"missing: {img_path.name}", ha="center", va="center")
                continue
            pil = Image.open(img_path).convert("RGB")
            png = np.array(Image.open(png_path).convert("RGB"))
            seg_id = seg['id']
            ids = rgb2id(png)
            mask = (ids == seg_id).astype(np.float32)

            axes[i, 0].imshow(pil); axes[i, 0].axis("off")
            axes[i, 0].set_title(f"{img_id} {pil.size[0]}x{pil.size[1]}", fontsize=9)
            axes[i, 1].imshow(pil)
            color = "lime" if cat['isthing'] == 0 else "cyan"
            axes[i, 1].imshow(mask, alpha=0.55, cmap="Greens" if cat['isthing'] == 0 else "Blues", vmin=0, vmax=1)
            axes[i, 1].contour(mask, levels=[0.5], colors=color, linewidths=1.2)
            axes[i, 1].axis("off")
            axes[i, 1].set_title(f"query='{name}' (isthing={cat['isthing']}, area={seg['area']})", fontsize=9)
            axes[i, 2].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[i, 2].axis("off")
            axes[i, 2].set_title(f"mask  pixels={int(mask.sum())}", fontsize=9)

        plt.suptitle(f"COCO Panoptic: '{name}' on val2017 — {len(buckets[cid])} segments total",
                     fontsize=11, y=0.995)
        plt.tight_layout()
        slug = name.replace(" ", "_").replace("-", "_").replace("/", "_")
        out_path = out_dir / f"panoptic-samples-{slug}.png"
        plt.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")

    if args.library_figures_dir:
        import shutil
        ldir = Path(args.library_figures_dir); ldir.mkdir(parents=True, exist_ok=True)
        for f in out_dir.glob("panoptic-samples-*.png"):
            shutil.copy(f, ldir / f.name)
            print(f"  -> {ldir / f.name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ann_path", default="/home/ogata/semantic-autogaze/data/coco_val2017/panoptic/annotations/panoptic_val2017.json")
    p.add_argument("--panoptic_png_dir", default="/home/ogata/semantic-autogaze/data/coco_val2017/panoptic/annotations/panoptic_val2017")
    p.add_argument("--image_dir", default="/home/ogata/semantic-autogaze/data/coco_val2017/val2017")
    p.add_argument("--n_per_query", type=int, default=4)
    p.add_argument("--include_things", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="/home/ogata/semantic-autogaze/results/panoptic_samples")
    p.add_argument("--library_figures_dir", default="/home/ogata/mac-brain/projects/semantic-autogaze/figures")
    args = p.parse_args()
    main(args)
