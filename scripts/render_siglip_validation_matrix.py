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

    # Build GT matrix and FN mask
    gt = np.zeros((B, B, GRID, GRID), dtype=np.float32)
    fn_mask = np.zeros((B, B), dtype=bool)
    for i in range(B):
        for j in range(B):
            if i == j:
                gt[i, j] = masks_diag[i]
            elif ds.presence_lookup.get((img_ids[i], slugs[j]), False):
                fn_mask[i, j] = True  # image_i actually contains query_j

    # ---- Render matrix ----
    # Layout: 2 rows per image-row (predicted, GT) × B columns of queries.
    # Top row of column headers = query labels above input thumbnails.
    # Left column of row headers = image thumbnail with image-id + diag query.
    cell = 1.7
    header_h = 1.1
    fig_w = (B + 1) * cell + 0.6
    fig_h = (2 * B) * cell + header_h + 0.8
    fig = plt.figure(figsize=(fig_w, fig_h))
    height_ratios = [header_h] + [cell] * (2 * B)
    width_ratios = [cell * 1.1] + [cell] * B
    gs = fig.add_gridspec(2 * B + 1, B + 1,
                          height_ratios=height_ratios,
                          width_ratios=width_ratios,
                          hspace=0.08, wspace=0.06,
                          left=0.04, right=0.99, top=0.94, bottom=0.02)

    # Top header row: query labels above query-image thumbnails
    for j, b in enumerate(chosen):
        ax = fig.add_subplot(gs[0, j + 1])
        arr = b["image"].cpu().numpy().transpose(1, 2, 0)
        arr = arr * np.array(std) + np.array(mean)
        arr = np.clip(arr, 0, 1)
        ax.imshow(arr)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"query j={j}\n'{b['query']}'", fontsize=10, pad=4)
        for s in ax.spines.values(): s.set_visible(False)

    # Top-left corner cell label
    ax = fig.add_subplot(gs[0, 0])
    ax.text(0.5, 0.5, "queries  →\n\n↓\nimages",
            ha="center", va="center", fontsize=10, transform=ax.transAxes,
            fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    # Body rows: for each image i, two rows (pred, GT)
    for i, b in enumerate(chosen):
        # Image label cell on the left, spanning both pred + gt rows
        ax = fig.add_subplot(gs[2*i + 1: 2*i + 3, 0])
        arr = b["image"].cpu().numpy().transpose(1, 2, 0)
        arr = arr * np.array(std) + np.array(mean)
        arr = np.clip(arr, 0, 1)
        ax.imshow(arr)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"image i={i}\n'{b['query']}'", fontsize=9, pad=2)
        for s in ax.spines.values(): s.set_visible(False)

        for j in range(B):
            # Cell type
            if i == j:
                color = "limegreen"; tag = "POS"
            elif fn_mask[i, j]:
                color = "red";       tag = "FN"
            else:
                color = "lightgray"; tag = "neg"

            arr_i = chosen[i]["image"].cpu().numpy().transpose(1, 2, 0)
            arr_i = arr_i * np.array(std) + np.array(mean)
            arr_i = np.clip(arr_i, 0, 1)

            # Predicted heatmap row
            ax_p = fig.add_subplot(gs[2*i + 1, j + 1])
            ax_p.imshow(arr_i, alpha=0.35)
            h = pred[i, j]
            h_up = np.kron(h, np.ones((arr_i.shape[0] // GRID + 1,
                                       arr_i.shape[1] // GRID + 1)))[:arr_i.shape[0], :arr_i.shape[1]]
            ax_p.imshow(h_up, alpha=0.65, cmap="hot", vmin=0, vmax=1)
            ax_p.set_xticks([]); ax_p.set_yticks([])
            ax_p.text(0.03, 0.97, f"max={h.max():.2f}", color="white", fontsize=8,
                      transform=ax_p.transAxes, va="top",
                      bbox=dict(facecolor="black", alpha=0.7, pad=2, edgecolor="none"))
            for s in ax_p.spines.values():
                s.set_visible(True); s.set_edgecolor(color); s.set_linewidth(2.5)
            if j == 0:
                ax_p.set_ylabel("PRED", fontsize=9, rotation=90,
                                ha="center", va="center", labelpad=8, fontweight="bold")

            # GT row
            ax_g = fig.add_subplot(gs[2*i + 2, j + 1])
            ax_g.imshow(arr_i, alpha=0.35)
            g = gt[i, j]
            g_up = np.kron(g, np.ones((arr_i.shape[0] // GRID + 1,
                                       arr_i.shape[1] // GRID + 1)))[:arr_i.shape[0], :arr_i.shape[1]]
            cmap_gt = "Greens" if i == j else ("Reds" if fn_mask[i, j] else "Greys")
            ax_g.imshow(g_up, alpha=0.65, cmap=cmap_gt, vmin=0, vmax=1)
            ax_g.set_xticks([]); ax_g.set_yticks([])
            ax_g.text(0.03, 0.97, tag, color="white", fontsize=9,
                      transform=ax_g.transAxes, va="top", fontweight="bold",
                      bbox=dict(facecolor=color, alpha=0.85, pad=2, edgecolor="none"))
            for s in ax_g.spines.values():
                s.set_visible(True); s.set_edgecolor(color); s.set_linewidth(2.5)
            if j == 0:
                ax_g.set_ylabel("GT", fontsize=9, rotation=90,
                                ha="center", va="center", labelpad=8, fontweight="bold")

    # Title
    n_fn = int(fn_mask.sum())
    n_diag = B
    n_neg = B * B - n_diag - n_fn
    fig.suptitle(
        f"SigLIP {model} — {B}×{B} cross-pair validation matrix      "
        f"diagonal POS={n_diag} (green) · off-diag FN-pollution={n_fn} (red, masked under --fn_filter) · genuine off-diag neg={n_neg} (gray)",
        fontsize=11, y=0.985, fontweight="bold",
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="/home/ogata/semantic-autogaze/results/phase2_validation_matrix")
    p.add_argument("--library_figures_dir",
                   default="/home/ogata/mac-brain/projects/semantic-autogaze/figures")
    args = p.parse_args()
    main(args)
