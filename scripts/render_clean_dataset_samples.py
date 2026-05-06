"""Render qualitative samples from a clean (COCO/LVIS) targets directory
to visually verify (image, query, mask) alignment is correct.

Picks N random POSITIVE samples + M random VERIFIED-NEGATIVE samples + K
random UNVERIFIED-MISSING samples and lays them out as a 4-column grid:
  [image] [image + mask overlay] [mask alone] [provenance metadata]

Output: figures/clean-dataset-samples-<tag>.png
"""
from __future__ import annotations
import argparse, os, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    return {
        "img_id": str(d["img_id"]),
        "query": str(d["query"]),
        "presence": bool(d["presence"]),
        "mask_full": np.array(d["mask_full"]),
        "mask14": np.array(d["mask14"]),
        "H": int(d["H"]), "W": int(d["W"]),
        "n_boxes": int(d["n_boxes"]),
        "source": str(d.get("source", "")) if "source" in d.files else "?",
        "lvis_neg_verified": bool(d["lvis_neg_verified"])
            if "lvis_neg_verified" in d.files else False,
        "lvis_not_exhaustive": bool(d["lvis_not_exhaustive"])
            if "lvis_not_exhaustive" in d.files else False,
        "clip_sim": float(d["clip_sim"]) if "clip_sim" in d.files else 1.0,
        "top_box_score": float(d["top_box_score"]) if "top_box_score" in d.files else 1.0,
    }


def render_strip(samples, image_dir, title, out_path):
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.0 * n))
    if n == 1: axes = axes[None, :]

    for i, (s, npz_path) in enumerate(samples):
        img_path = Path(image_dir) / f"{s['img_id']}.jpg"
        if not img_path.exists():
            for c in range(4):
                axes[i, c].axis("off")
                if c == 0: axes[i, c].text(0.5, 0.5, f"image not found:\n{img_path}",
                                           ha="center", va="center")
            continue
        pil = Image.open(img_path).convert("RGB")

        mask_full = s["mask_full"].astype(float)
        if mask_full.max() > 1: mask_full /= 255.0

        # Col 0: input image
        axes[i, 0].imshow(pil); axes[i, 0].axis("off")
        axes[i, 0].set_title(f"{s['img_id']} ({pil.size[0]}x{pil.size[1]})", fontsize=9)

        # Col 1: image + mask_full overlay (green for positive, red for verified-neg)
        axes[i, 1].imshow(pil)
        if s["presence"]:
            cmap = "Greens"; tag_color = "lime"
        elif s["lvis_neg_verified"]:
            cmap = "Reds"; tag_color = "red"
        else:
            cmap = "Greys"; tag_color = "gray"
        axes[i, 1].imshow(mask_full, alpha=0.55, cmap=cmap, vmin=0, vmax=1)
        # Mask outline for crispness
        if mask_full.max() > 0:
            axes[i, 1].contour(mask_full, levels=[0.5], colors=tag_color, linewidths=1.4)
        axes[i, 1].axis("off")
        axes[i, 1].set_title(f"query='{s['query']}' (presence={s['presence']})", fontsize=9)

        # Col 2: mask alone (binary)
        if mask_full.max() > 0:
            axes[i, 2].imshow(mask_full, cmap="gray", vmin=0, vmax=1)
        else:
            # Show a flat zero mask explicitly
            axes[i, 2].imshow(np.zeros_like(mask_full), cmap="gray", vmin=0, vmax=1)
            axes[i, 2].text(0.5, 0.5, "ZERO MASK\n(absent target)",
                            ha="center", va="center", color="white", fontsize=10,
                            transform=axes[i, 2].transAxes)
        axes[i, 2].axis("off")
        axes[i, 2].set_title(f"mask_full ({s['H']}x{s['W']})  ·  mask14 sum={s['mask14'].sum()/255:.0f}",
                             fontsize=9)

        # Col 3: provenance metadata
        meta = (
            f"npz: {Path(npz_path).name}\n\n"
            f"source        : {s['source']}\n"
            f"presence      : {s['presence']}\n"
            f"n_boxes       : {s['n_boxes']}\n"
            f"clip_sim      : {s['clip_sim']:.3f}\n"
            f"top_box_score : {s['top_box_score']:.3f}\n"
            f"lvis_neg_verified  : {s['lvis_neg_verified']}\n"
            f"lvis_not_exhaustive: {s['lvis_not_exhaustive']}\n"
        )
        axes[i, 3].text(0.02, 0.98, meta, ha="left", va="top",
                        fontsize=8, family="monospace",
                        transform=axes[i, 3].transAxes)
        axes[i, 3].axis("off")

    plt.suptitle(title, fontsize=12, y=0.99)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def categorize(npz_path):
    d = np.load(npz_path, allow_pickle=False)
    pres = bool(d["presence"])
    lvis_neg = bool(d["lvis_neg_verified"]) if "lvis_neg_verified" in d.files else False
    if pres: return "pos"
    if lvis_neg: return "verified_neg"
    return "weak_neg"


def main(args):
    rng = random.Random(args.seed)
    targets = sorted(Path(args.target_dir).glob("*.npz"))
    print(f"[input] {len(targets)} npz files in {args.target_dir}")

    # Subsample for category scan
    scan = rng.sample(targets, k=min(len(targets), args.scan_size))

    pos_paths = []; vneg_paths = []; wneg_paths = []
    for f in scan:
        c = categorize(f)
        if c == "pos": pos_paths.append(f)
        elif c == "verified_neg": vneg_paths.append(f)
        else: wneg_paths.append(f)
    print(f"[scan] in {len(scan)} sampled: pos={len(pos_paths)} verified_neg={len(vneg_paths)} weak_neg={len(wneg_paths)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # POSITIVES strip
    if pos_paths:
        chosen = rng.sample(pos_paths, k=min(len(pos_paths), args.n_per_category))
        loaded = [(load_npz(p), p) for p in chosen]
        render_strip(
            loaded, args.image_dir,
            f"CLEAN POSITIVES — {args.tag} (presence=True, GT mask is union of instance masks)",
            out_dir / f"clean-samples-{args.tag}-positives.png",
        )

    # VERIFIED NEGATIVES (LVIS-only)
    if vneg_paths:
        chosen = rng.sample(vneg_paths, k=min(len(vneg_paths), args.n_per_category))
        loaded = [(load_npz(p), p) for p in chosen]
        render_strip(
            loaded, args.image_dir,
            f"VERIFIED NEGATIVES — {args.tag} (LVIS neg_category_ids — human-verified absent; GT mask is zeros)",
            out_dir / f"clean-samples-{args.tag}-verified-negatives.png",
        )

    # WEAK NEGATIVES (COCO-style "not annotated" — usually but not always absent)
    if wneg_paths:
        chosen = rng.sample(wneg_paths, k=min(len(wneg_paths), args.n_per_category))
        loaded = [(load_npz(p), p) for p in chosen]
        render_strip(
            loaded, args.image_dir,
            f"WEAK NEGATIVES — {args.tag} (COCO 'not in instance annotations'; usually absent but not verified)",
            out_dir / f"clean-samples-{args.tag}-weak-negatives.png",
        )

    if args.library_figures_dir:
        import shutil
        ldir = Path(args.library_figures_dir); ldir.mkdir(parents=True, exist_ok=True)
        for f in out_dir.glob(f"clean-samples-{args.tag}-*.png"):
            shutil.copy(f, ldir / f.name)
            print(f"  -> {ldir / f.name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target_dir", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--tag", required=True, help="Short tag like 'val2017' or 'train2017+lvis'")
    p.add_argument("--n_per_category", type=int, default=8)
    p.add_argument("--scan_size", type=int, default=4000,
                   help="Random subset to categorize for sampling buckets.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="/home/ogata/semantic-autogaze/results/clean_dataset_samples")
    p.add_argument("--library_figures_dir",
                   default="/home/ogata/mac-brain/projects/semantic-autogaze/figures")
    args = p.parse_args()
    main(args)
