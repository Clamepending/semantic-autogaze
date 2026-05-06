"""Render a sample BxB SigLIP-style training batch SUPERVISION matrix
(no model needed — just shows what the loss is computed against).

Each (image_i, query_j) cell shows:
  image_i with the supervision target overlay for (i, j):
    - Diagonal (i==j): the actual Grounded-SAM/COCO/LVIS positive mask
                       (lime green fill). Border = lime.
    - Off-diagonal (i!=j) genuine negative: GT = zeros. Border = dimgray.
                       Label "neg→0" — model trained to predict zeros here.
    - Off-diagonal FN-pollution (presence_lookup[(i, j)] == True):
                       GT = zeros at training time IF --fn_filter is off,
                       ELSE loss is masked out so target is "don't care".
                       Border = red. Label "FN-mask".

This is the cleanest view of the SigLIP per-pair supervision signal at
training time — independent of model state.

Output: figures/clean-training-batch-<tag>-seed<seed>.png
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


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
        "source": str(d.get("source", "?")) if "source" in d.files else "?",
    }


def main(args):
    rng = np.random.default_rng(args.seed)
    target_dir = Path(args.target_dir)
    image_dir = Path(args.image_dir)

    # Build presence_lookup: which (img_id, query) pairs are positive in the dataset
    print(f"[scan] building presence_lookup over {target_dir} ...", flush=True)
    all_npz = sorted(target_dir.glob("*.npz"))
    print(f"  {len(all_npz)} npz total", flush=True)
    presence_lookup: dict = {}
    pos_paths = []
    for f in all_npz:
        # Parse stem: <img_id>__<query_slug>.npz
        stem = f.stem
        if "__" not in stem: continue
        img_id, _, slug = stem.partition("__")
        # Quick presence read: it's stored as bool(d["presence"]) which is a
        # 0-d numpy array. Read just that key for speed.
        try:
            d = np.load(f, allow_pickle=False)
            pres = bool(d["presence"])
        except Exception:
            continue
        presence_lookup[(img_id, slug)] = pres
        if pres:
            pos_paths.append(f)
    n_pos = len(pos_paths)
    print(f"[scan] {len(presence_lookup)} (img, query) keys, {n_pos} positives", flush=True)

    # Pick B distinct positive (image, query) samples with distinct images and queries
    perm = rng.permutation(len(pos_paths))
    seen_q: set = set(); seen_i: set = set(); chosen = []
    for ix in perm:
        f = pos_paths[ix]
        s = load_npz(f)
        if s["query"] in seen_q or s["img_id"] in seen_i:
            continue
        seen_q.add(s["query"]); seen_i.add(s["img_id"])
        chosen.append((s, f))
        if len(chosen) >= args.batch_size:
            break
    B = len(chosen)
    print(f"[batch] sampled {B} (image, query) positives:")
    for i, (s, f) in enumerate(chosen):
        print(f"  [{i}] {s['img_id']} '{s['query']}' source={s['source']} "
              f"n_boxes={s['n_boxes']}")

    # Resize all masks to args.image_size for crisp display
    sz = args.image_size
    diag_full_resized = []
    images_pil = []
    img_ids = []; queries = []; slugs = []
    for s, f in chosen:
        img_path = image_dir / f"{s['img_id']}.jpg"
        pil = Image.open(img_path).convert("RGB").resize((sz, sz), Image.BICUBIC)
        images_pil.append(np.array(pil))
        img_ids.append(s["img_id"])
        queries.append(s["query"])
        slugs.append(f.stem.partition("__")[2])
        m = s["mask_full"].astype(float)
        if m.max() > 1: m /= 255.0
        m_r = np.array(Image.fromarray((m * 255).astype(np.uint8))
                       .resize((sz, sz), Image.NEAREST)) / 255.0
        diag_full_resized.append(m_r)

    # Build BxB target matrix
    targets = np.zeros((B, B, sz, sz), dtype=np.float32)
    fn_overlay = np.zeros((B, B, sz, sz), dtype=np.float32)
    fn_flags = np.zeros((B, B), dtype=bool)
    for i in range(B):
        for j in range(B):
            if i == j:
                targets[i, j] = diag_full_resized[i]
            elif presence_lookup.get((img_ids[i], slugs[j]), False):
                fn_flags[i, j] = True
                # Load the mask for the FN pair to display what's masked out
                npz_fn = target_dir / f"{img_ids[i]}__{slugs[j]}.npz"
                if npz_fn.exists():
                    d_fn = np.load(npz_fn, allow_pickle=False)
                    m = d_fn["mask_full"].astype(float)
                    if m.max() > 1: m /= 255.0
                    m_r = np.array(Image.fromarray((m * 255).astype(np.uint8))
                                   .resize((sz, sz), Image.NEAREST)) / 255.0
                    fn_overlay[i, j] = m_r

    # ---- Render ----
    cell = 1.9
    header_h = 1.4
    fig_w = (B + 1) * cell + 0.6
    fig_h = B * cell + header_h + 0.8
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        B + 1, B + 1,
        height_ratios=[header_h] + [cell] * B,
        width_ratios=[cell * 1.15] + [cell] * B,
        hspace=0.08, wspace=0.06, left=0.04, right=0.99, top=0.92, bottom=0.02,
    )

    # Header row: queries + their source images
    for j in range(B):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.imshow(images_pil[j])
        ax.contour(diag_full_resized[j], levels=[0.5], colors="lime", linewidths=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"query j={j}: '{queries[j]}'\n(source img {img_ids[j]})",
                     fontsize=9, pad=4)
        for s in ax.spines.values(): s.set_visible(False)
    ax = fig.add_subplot(gs[0, 0])
    n_fn = int(fn_flags.sum()); n_pos_diag = B; n_neg = B*B - n_pos_diag - n_fn
    ax.text(0.5, 0.5,
            "queries →\n↓ images\n\n"
            "SigLIP per-pair\nsupervision target\n\n"
            f"diag POS: {n_pos_diag}\n"
            f"FN-pollution: {n_fn}\n"
            f"genuine neg: {n_neg}",
            ha="center", va="center", fontsize=8, transform=ax.transAxes,
            family="monospace")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    # Body
    for i in range(B):
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.imshow(images_pil[i])
        ax.contour(diag_full_resized[i], levels=[0.5], colors="lime", linewidths=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"image i={i}: {img_ids[i]}\ndiag query: '{queries[i]}'",
                     fontsize=9, pad=2)
        for s in ax.spines.values(): s.set_visible(False)

        for j in range(B):
            if i == j:
                border = "limegreen"; tag = "POS"
                cmap = "Greens"
            elif fn_flags[i, j]:
                border = "red"; tag = "FN-mask"
                cmap = "Reds"
            else:
                border = "dimgray"; tag = "neg→0"
                cmap = "Greys"

            ax_c = fig.add_subplot(gs[i + 1, j + 1])
            ax_c.imshow(images_pil[i], alpha=0.55)
            tgt = targets[i, j] if i == j else (fn_overlay[i, j] if fn_flags[i, j] else None)
            if tgt is not None and tgt.max() > 0:
                ax_c.imshow(tgt, alpha=0.55, cmap=cmap, vmin=0, vmax=1)
                ax_c.contour(tgt, levels=[0.5], colors=border, linewidths=1.6,
                             linestyles=("solid" if i == j else "dashed"))
            else:
                # genuine neg: GT is zeros — show flat black overlay
                ax_c.imshow(np.zeros_like(images_pil[i][:, :, 0]),
                            alpha=0.5, cmap="Greys", vmin=0, vmax=1)
            ax_c.set_xticks([]); ax_c.set_yticks([])
            ax_c.text(0.97, 0.03, tag, color="white", fontsize=8,
                      fontweight="bold", transform=ax_c.transAxes,
                      ha="right", va="bottom",
                      bbox=dict(facecolor=border, alpha=0.85, pad=2, edgecolor="none"))
            for s in ax_c.spines.values():
                s.set_visible(True); s.set_edgecolor(border); s.set_linewidth(3)

    fig.suptitle(
        f"SigLIP training batch SUPERVISION — clean COCO+LVIS, {B}×{B}, seed={args.seed}.  "
        f"Each cell = image_i with the GT target for query_j.  "
        f"Diagonal POS={n_pos_diag} (lime), off-diag FN-pollution={n_fn} (red dashed; "
        f"loss-masked under --fn_filter), genuine neg→0={n_neg} (gray, target zeros).",
        fontsize=10, y=0.985, wrap=True,
    )

    out_path = Path(args.output_dir) / f"clean-training-batch-{args.tag}-seed{args.seed}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    if args.library_figures_dir:
        import shutil
        ldir = Path(args.library_figures_dir); ldir.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_path, ldir / out_path.name)
        print(f"  -> {ldir / out_path.name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target_dir",
                   default="/home/ogata/semantic-autogaze/results/clean_targets_val2017")
    p.add_argument("--image_dir",
                   default="/home/ogata/semantic-autogaze/data/coco_val2017/val2017")
    p.add_argument("--tag", default="clean-val2017")
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir",
                   default="/home/ogata/semantic-autogaze/results/clean_training_batch")
    p.add_argument("--library_figures_dir",
                   default="/home/ogata/mac-brain/projects/semantic-autogaze/figures")
    args = p.parse_args()
    main(args)
