"""Render a validation BxB cross-pair matrix for SigLIP-style training.

Samples B (image, query) positives from the phase2_targets dataset, runs the
trained head on every cross-pair (image_i, query_j), and lays out a BxB grid.
Each cell shows predicted heatmap (row=image, col=query). Above the grid: the
B input images. Left of the grid: the B queries. Below each predicted cell:
the ground-truth target — Grounded-SAM mask on the diagonal, all-zero off-
diagonal (with red border + 'FN' annotation when presence_lookup flags the
off-diagonal pair as a false negative — image_i actually contains query_j).

Output: figures/siglip-validation-matrix-<model>.png (also saved to results/).
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")
from train_independent_scorer import TextScorerHead, GRID
from train_siglip_dense_distill import (
    TargetDataset, build_backbone, SiglipBias, CLIP_MEAN, CLIP_STD,
)


def load_ckpt(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ck.get("args", {})
    model = args.get("model", "v2-tiny")
    bb_fn, patch_dim, mean, std, kind, bb_module = build_backbone(
        model, device, finetune_blocks=args.get("finetune_backbone_blocks", 0)
    )
    head = TextScorerHead(
        patch_dim=patch_dim, text_dim=512,
        hidden_dim=args.get("head_hidden_dim", 384),
        n_attn_heads=args.get("head_attn_heads", 6),
        n_attn_layers=args.get("head_attn_layers", 2),
        grid_size=GRID, use_spatial=args.get("head_use_spatial", True),
    ).to(device).eval()
    head.load_state_dict(ck["head"])
    sb = SiglipBias().to(device).eval()
    sb.load_state_dict(ck["sb"])
    if "backbone_state" in ck:
        bb_module.load_state_dict(ck["backbone_state"])
    return bb_fn, head, sb, mean, std, model


@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[ckpt] loading {args.ckpt}", flush=True)
    bb_fn, head, sb, mean, std, model = load_ckpt(args.ckpt, device)
    print(f"  model={model}", flush=True)

    print(f"[clip-text] loading...", flush=True)
    import open_clip
    clip_text, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_text = clip_text.to(device).eval()

    print(f"[dataset] scanning targets ...", flush=True)
    ds = TargetDataset(
        target_dir=args.target_dir,
        image_dir=args.image_dir,
        image_size=224,
        mean=mean, std=std,
        positive_only=True,
        build_presence_lookup=True,
    )

    # Pick B distinct (image_id, query_slug) positives, with distinct queries.
    rng = np.random.default_rng(args.seed)
    seen_queries: set = set()
    seen_imgs: set = set()
    chosen = []
    perm = rng.permutation(len(ds))
    for idx in perm:
        item = ds[int(idx)]
        if item is None:
            continue
        q = item["query"]; iid = item["img_id"]
        if q in seen_queries or iid in seen_imgs:
            continue
        seen_queries.add(q); seen_imgs.add(iid)
        chosen.append(item)
        if len(chosen) >= args.batch_size:
            break
    B = len(chosen)
    print(f"[batch] sampled {B} (image, query) positives", flush=True)
    for i, b in enumerate(chosen):
        print(f"  [{i}] img={b['img_id']} query='{b['query']}' "
              f"(presence={b['presence']}, clip_sim={b['clip_sim']:.3f})")

    # Stack into (B, 3, 224, 224) tensor
    x = torch.stack([b["image"] for b in chosen]).to(device)
    queries = [b["query"] for b in chosen]
    img_ids = [b["img_id"] for b in chosen]
    slugs = [b["query_slug"] for b in chosen]
    masks_diag = torch.stack([b["mask14"] for b in chosen]).cpu().numpy()  # (B, 14, 14)

    # Encode patches once per image, text once per query
    patches = bb_fn(x)  # (B, 196, D)
    toks = clip_tok(queries).to(device)
    text = F.normalize(clip_text.encode_text(toks), dim=-1)  # (B, 512)

    # Forward (B images) x (B queries) — head is per (image, text) so we run
    # B * B forward calls. With B<=8 this is cheap.
    pred = np.zeros((B, B, GRID, GRID), dtype=np.float32)
    for i in range(B):
        for j in range(B):
            logits = head(patches[i:i+1], text[j:j+1]).reshape(GRID, GRID)
            cal = sb(logits)
            pred[i, j] = torch.sigmoid(cal).cpu().numpy()
    print(f"[pred] computed {B}x{B}={B*B} cross-pair heatmaps", flush=True)

    # Build GT matrix using full-resolution masks resized to 224 (better than
    # upscaling 14x14 for visual clarity). Diagonal cells use the actual mask
    # for image_i; off-diagonal cells use zero mask, but we additionally store
    # the *would-be* mask if image_i contains query_j (FN-pollution diagnostic).
    gt = np.zeros((B, B, args.image_size, args.image_size), dtype=np.float32)
    fn_mask_overlay = np.zeros((B, B, args.image_size, args.image_size), dtype=np.float32)
    fn_mask = np.zeros((B, B), dtype=bool)
    diag_full = []
    for i in range(B):
        # Reload the full-res mask for image i, query i
        npz_diag = os.path.join(args.target_dir, f"{img_ids[i]}__{slugs[i]}.npz")
        d_diag = np.load(npz_diag, allow_pickle=False)
        m_full = d_diag["mask_full"].astype(np.float32)
        if m_full.max() > 1: m_full /= 255.0
        m224 = np.array(Image.fromarray((m_full * 255).astype(np.uint8))
                        .resize((args.image_size, args.image_size), Image.NEAREST)) / 255.0
        diag_full.append(m224)

    for i in range(B):
        for j in range(B):
            if i == j:
                gt[i, j] = diag_full[i]
            elif ds.presence_lookup.get((img_ids[i], slugs[j]), False):
                fn_mask[i, j] = True
                # Try to load the FN-pair's mask to show what we're masking out
                npz_fn = os.path.join(args.target_dir, f"{img_ids[i]}__{slugs[j]}.npz")
                if os.path.exists(npz_fn):
                    d_fn = np.load(npz_fn, allow_pickle=False)
                    m_full = d_fn["mask_full"].astype(np.float32)
                    if m_full.max() > 1: m_full /= 255.0
                    m224 = np.array(Image.fromarray((m_full * 255).astype(np.uint8))
                                    .resize((args.image_size, args.image_size), Image.NEAREST)) / 255.0
                    fn_mask_overlay[i, j] = m224

    # ---- Render matrix ----
    # Layout: ONE cell per (image_i, query_j) — predicted heatmap (red/yellow
    # hot overlay) AND GT mask contour (lime green outline) on top of the
    # image_i. The heatmap fill shows what the model OUTPUTS; the green
    # contour shows what it SHOULD output (zero-area outline = "predict zero").
    # Cell border color: lime=diag POS, red=FN-pollution, dimgray=genuine neg.
    cell = 1.9
    header_h = 1.4
    fig_w = (B + 1) * cell + 0.6
    fig_h = B * cell + header_h + 0.8
    fig = plt.figure(figsize=(fig_w, fig_h))
    height_ratios = [header_h] + [cell] * B
    width_ratios = [cell * 1.15] + [cell] * B
    gs = fig.add_gridspec(B + 1, B + 1,
                          height_ratios=height_ratios,
                          width_ratios=width_ratios,
                          hspace=0.08, wspace=0.06,
                          left=0.04, right=0.99, top=0.92, bottom=0.02)

    # Top header row: column = query j with its source image as thumbnail
    for j, b in enumerate(chosen):
        ax = fig.add_subplot(gs[0, j + 1])
        arr = b["image"].cpu().numpy().transpose(1, 2, 0)
        arr = arr * np.array(std) + np.array(mean)
        arr = np.clip(arr, 0, 1)
        ax.imshow(arr)
        # Overlay GT mask outline (where this query SHOULD fire, on its source img)
        gt_src = diag_full[j]
        ax.contour(gt_src, levels=[0.5], colors="lime", linewidths=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"query j={j}: '{b['query']}'\n(source img {b['img_id']})",
                     fontsize=9, pad=4)
        for s in ax.spines.values(): s.set_visible(False)

    # Top-left corner legend
    ax = fig.add_subplot(gs[0, 0])
    ax.text(0.5, 0.5,
            "queries →\n↓ images\n\n"
            "fill = pred\n"
            "lime contour = GT\n"
            "border:\n"
            "  green = diag POS\n"
            "  red   = FN-pollution\n"
            "  gray  = genuine neg",
            ha="center", va="center", fontsize=8, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    # Body: B rows of B cells, each cell = pred fill + GT contour over image_i
    n_fn = 0; n_diag = 0; n_neg = 0
    for i, b in enumerate(chosen):
        # Left header cell: image i + its diagonal query
        ax = fig.add_subplot(gs[i + 1, 0])
        arr = b["image"].cpu().numpy().transpose(1, 2, 0)
        arr = arr * np.array(std) + np.array(mean)
        arr = np.clip(arr, 0, 1)
        ax.imshow(arr)
        ax.contour(diag_full[i], levels=[0.5], colors="lime", linewidths=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"image i={i}: {b['img_id']}\ndiag query: '{b['query']}'",
                     fontsize=9, pad=2)
        for s in ax.spines.values(): s.set_visible(False)

        for j in range(B):
            # Cell type
            if i == j:
                border = "limegreen"; tag = "POS"; n_diag += 1
            elif fn_mask[i, j]:
                border = "red"; tag = "FN-mask"; n_fn += 1
            else:
                border = "dimgray"; tag = "neg→0"; n_neg += 1

            ax_c = fig.add_subplot(gs[i + 1, j + 1])
            ax_c.imshow(arr, alpha=0.55)
            h = pred[i, j]
            h_up = np.kron(h, np.ones((arr.shape[0] // GRID + 1,
                                       arr.shape[1] // GRID + 1)))[:arr.shape[0], :arr.shape[1]]
            ax_c.imshow(h_up, alpha=0.55, cmap="hot", vmin=0, vmax=1)
            ax_c.set_xticks([]); ax_c.set_yticks([])

            # GT contour overlay
            if i == j:
                # Diagonal: GT is the actual mask for image i, query j (=i)
                ax_c.contour(gt[i, j], levels=[0.5], colors="lime", linewidths=1.6)
            elif fn_mask[i, j] and fn_mask_overlay[i, j].max() > 0:
                # FN-pollution: there IS a mask in the dataset for this pair,
                # but training treats it as zero (without --fn_filter) or
                # masks the loss (with --fn_filter). Show the would-be GT in
                # red dashed so reader sees what's being thrown away.
                ax_c.contour(fn_mask_overlay[i, j], levels=[0.5],
                             colors="red", linewidths=1.4, linestyles="dashed")
            # genuine neg: no contour (target is zero everywhere)

            # Score annotation
            ax_c.text(0.03, 0.97, f"max={h.max():.2f}",
                      color="white", fontsize=8, fontweight="bold",
                      transform=ax_c.transAxes, va="top",
                      bbox=dict(facecolor="black", alpha=0.65, pad=2, edgecolor="none"))
            ax_c.text(0.97, 0.03, tag, color="white", fontsize=8,
                      fontweight="bold", transform=ax_c.transAxes,
                      ha="right", va="bottom",
                      bbox=dict(facecolor=border, alpha=0.85, pad=2, edgecolor="none"))
            for s in ax_c.spines.values():
                s.set_visible(True); s.set_edgecolor(border); s.set_linewidth(3)

    fig.suptitle(
        f"SigLIP {model} — {B}×{B} cross-pair validation matrix.  "
        f"Each cell shows the model's prediction for (image_i, query_j) "
        f"as a hot overlay on image_i; lime contour = supervision GT (diagonal); "
        f"red dashed = would-be GT on FN-pollution pairs (loss-masked under --fn_filter).  "
        f"counts: POS={n_diag} · FN={n_fn} · neg={n_neg}",
        fontsize=10, y=0.99, wrap=True,
    )

    out_png = os.path.join(args.output_dir, f"siglip-validation-matrix-{model}.png")
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}")

    if args.library_figures_dir:
        os.makedirs(args.library_figures_dir, exist_ok=True)
        import shutil
        shutil.copy(out_png, os.path.join(args.library_figures_dir, os.path.basename(out_png)))
        print(f"  -> {args.library_figures_dir}/{os.path.basename(out_png)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--target_dir", default="/home/ogata/semantic-autogaze/results/phase2_targets")
    p.add_argument("--image_dir", default="/home/ogata/semantic-autogaze/data/coco_val2017/val2017")
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="/home/ogata/semantic-autogaze/results/phase2_validation_matrix")
    p.add_argument("--library_figures_dir",
                   default="/home/ogata/mac-brain/projects/semantic-autogaze/figures")
    args = p.parse_args()
    main(args)
