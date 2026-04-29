"""Render qualitative samples from Pascal-Part to verify (image, query, mask)
alignment for body-part annotations on VOC2010 images.

Targeted at the deployment-vocab gap: "person:hand", "person:arm", "person:head"
weren't covered by COCO+LVIS but are gold-labeled in Pascal-Part.

Picks N random samples per target part and renders 4-col strips:
  [image] [image + mask overlay] [mask alone] [provenance metadata]
"""
from __future__ import annotations
import argparse, os, random
from pathlib import Path
import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt


def parse_mat(mat_path):
    """Returns list of (class, [(part_name, mask_full)])"""
    mat = sio.loadmat(mat_path)
    anno = mat.get('anno')
    if anno is None: return []
    objects = anno[0,0]['objects']
    out = []
    for obj in objects[0]:
        cls = str(obj['class'][0]) if 'class' in obj.dtype.names else 'unk'
        parts = obj['parts'] if 'parts' in obj.dtype.names else None
        part_masks = []
        if parts is not None and parts.size > 0:
            for p in parts[0]:
                pname = str(p['part_name'][0])
                pmask = p['mask']
                part_masks.append((pname, pmask))
        out.append((cls, part_masks))
    return out


def union_parts(part_masks, predicate):
    """Union all part masks whose part_name matches predicate."""
    masks = [m for n, m in part_masks if predicate(n)]
    if not masks: return None
    union = masks[0].astype(bool)
    for m in masks[1:]:
        union |= m.astype(bool)
    return union


# Queries to render: combine left+right naming variants under unified part
QUERY_RULES = [
    ("hand",       "person", lambda n: n in ("lhand", "rhand")),
    ("arm",        "person", lambda n: n in ("luarm", "ruarm", "llarm", "rlarm")),
    ("upper arm",  "person", lambda n: n in ("luarm", "ruarm")),
    ("forearm",    "person", lambda n: n in ("llarm", "rlarm")),
    ("head",       "person", lambda n: n == "head"),
    ("face/eye",   "person", lambda n: n in ("leye", "reye")),
    ("nose",       "person", lambda n: n == "nose"),
    ("mouth",      "person", lambda n: n == "mouth"),
    ("hair",       "person", lambda n: n == "hair"),
    ("torso",      "person", lambda n: n == "torso"),
    ("leg",        "person", lambda n: n in ("luleg", "ruleg", "llleg", "rlleg")),
    ("foot",       "person", lambda n: n in ("lfoot", "rfoot")),
    ("dog:head",   "dog",    lambda n: n == "head"),
    ("dog:paw",    "dog",    lambda n: n in ("lfpa", "rfpa", "lbpa", "rbpa")),
    ("cat:head",   "cat",    lambda n: n == "head"),
    ("bird:wing",  "bird",   lambda n: n in ("lwing", "rwing")),
    ("car:wheel",  "car",    lambda n: n.startswith("wheel")),
    ("bicycle:wheel", "bicycle", lambda n: n in ("fwheel", "bwheel")),
    ("aeroplane:wing", "aeroplane", lambda n: n in ("lwing", "rwing")),
    ("tvmonitor:screen", "tvmonitor", lambda n: n == "screen"),
]


def main(args):
    rng = random.Random(args.seed)
    ann_dir = Path(args.ann_dir)
    img_dir = Path(args.image_dir)

    print(f"[scan] scanning Pascal-Part .mat files for queries ...")
    mat_files = sorted(ann_dir.glob("*.mat"))
    # Bucket: (query_label, [(image_id, target_class, mask_full)])
    buckets: dict = {q: [] for q, _, _ in QUERY_RULES}

    for f in mat_files:
        img_id = f.stem  # like "2008_000002"
        objs = parse_mat(f)
        for cls, part_masks in objs:
            for q, target_cls, pred in QUERY_RULES:
                if cls != target_cls: continue
                m = union_parts(part_masks, pred)
                if m is None or m.sum() < 50: continue
                buckets[q].append((img_id, cls, m))

    print(f"[scan] yields:")
    for q, lst in buckets.items():
        print(f"  {q:25s}  {len(lst)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Render one strip per query (top N samples)
    for q, lst in buckets.items():
        if len(lst) == 0: continue
        chosen = rng.sample(lst, k=min(len(lst), args.n_per_query))
        n = len(chosen)
        fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
        if n == 1: axes = axes[None, :]
        for i, (img_id, cls, m) in enumerate(chosen):
            img_path = img_dir / f"{img_id}.jpg"
            if not img_path.exists():
                axes[i, 0].text(0.5, 0.5, f"missing: {img_path}",
                                ha="center", va="center")
                for c in range(3): axes[i, c].axis("off")
                continue
            pil = Image.open(img_path).convert("RGB")
            axes[i, 0].imshow(pil); axes[i, 0].axis("off")
            axes[i, 0].set_title(f"{img_id}  ({pil.size[0]}x{pil.size[1]})", fontsize=9)
            axes[i, 1].imshow(pil)
            axes[i, 1].imshow(m.astype(float), alpha=0.55, cmap="Greens", vmin=0, vmax=1)
            axes[i, 1].contour(m.astype(float), levels=[0.5], colors="lime", linewidths=1.2)
            axes[i, 1].axis("off")
            axes[i, 1].set_title(f"query='{q}' object='{cls}'", fontsize=9)
            axes[i, 2].imshow(m, cmap="gray", vmin=0, vmax=1)
            axes[i, 2].axis("off")
            axes[i, 2].set_title(f"mask  pixels={int(m.sum())}", fontsize=9)
        plt.suptitle(f"Pascal-Part: '{q}' on VOC2010 — {len(lst)} total samples", fontsize=11, y=0.995)
        plt.tight_layout()
        slug = q.replace(" ", "_").replace(":", "_").replace("/", "_")
        out_path = out_dir / f"pascal-part-samples-{slug}.png"
        plt.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")

    if args.library_figures_dir:
        import shutil
        ldir = Path(args.library_figures_dir); ldir.mkdir(parents=True, exist_ok=True)
        for f in out_dir.glob("pascal-part-samples-*.png"):
            shutil.copy(f, ldir / f.name)
            print(f"  -> {ldir / f.name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ann_dir", default="/home/ogata/semantic-autogaze/data/pascal_part/Annotations_Part")
    p.add_argument("--image_dir", default="/home/ogata/semantic-autogaze/data/voc2010/VOCdevkit/VOC2010/JPEGImages")
    p.add_argument("--n_per_query", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="/home/ogata/semantic-autogaze/results/pascal_part_samples")
    p.add_argument("--library_figures_dir", default="/home/ogata/mac-brain/projects/semantic-autogaze/figures")
    args = p.parse_args()
    main(args)
