"""Render an open-vocab probe grid: N images × Q queries for a chosen ckpt.

Lets us see how the Pi-deployable scorer responds across a wide variety of
queries (things, stuff, body parts, OOD) on diverse scenes.
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
from train_independent_scorer import TextScorerHead, GRID
from train_siglip_dense_distill import build_backbone, SiglipBias


def load_ckpt(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ck.get("args", {})
    model = args.get("model", "v2-tiny")
    bb_fn, patch_dim, mean, std, kind, bb_module = build_backbone(model, device, finetune_blocks=0)
    head = TextScorerHead(
        patch_dim=patch_dim, text_dim=512,
        hidden_dim=args.get("head_hidden_dim", 384),
        n_attn_heads=args.get("head_attn_heads", 6),
        n_attn_layers=args.get("head_attn_layers", 2),
        grid_size=GRID, use_spatial=args.get("head_use_spatial", True),
    ).to(device).eval()
    head.load_state_dict(ck["head"])
    sb = SiglipBias().to(device).eval(); sb.load_state_dict(ck["sb"])
    if "backbone_state" in ck:
        bb_module.load_state_dict(ck["backbone_state"])
    return bb_fn, head, sb, mean, std, model


@torch.no_grad()
def heatmap(pil, query, bb_fn, head, sb, mean, std, clip_text, clip_tok, device):
    arr = np.array(pil.resize((224, 224), Image.BICUBIC))
    x = (arr.astype(np.float32) / 255.0 - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(device)
    patches = bb_fn(x)
    toks = clip_tok([query]).to(device)
    text = F.normalize(clip_text.encode_text(toks), dim=-1)
    logits = head(patches, text).reshape(GRID, GRID)
    cal = sb(logits)
    return torch.sigmoid(cal).cpu().numpy()


def main(args):
    device = torch.device(args.device)
    bb_fn, head, sb, mean, std, model = load_ckpt(args.ckpt, device)
    print(f"[ckpt] {args.ckpt}  model={model}", flush=True)

    import open_clip
    clip_text, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_text = clip_text.to(device).eval()

    # Pick diverse images from val2017
    coco_root = "/home/ogata/semantic-autogaze/data/coco_val2017/val2017"
    image_ids = [
        ("000000050380", "family scene with dogs/people"),
        ("000000176037", "city bus at intersection"),
        ("000000223747", "cat on bed"),
        ("000000549738", "kite/sky scene"),
        ("000000397133", "kitchen interior"),
        ("000000458109", "train at platform"),
    ]

    # Diverse query set: things + stuff + body parts + OOD
    queries = [
        "person", "dog", "cat", "car", "bus",        # things in dist
        "sky", "road", "wall", "grass", "tree",       # stuff
        "hand", "face", "head",                       # body parts
        "pen", "laptop", "book",                      # OOD-ish
    ]

    n_imgs = len(image_ids)
    n_queries = len(queries)

    # Compute heatmaps
    all_h = np.zeros((n_imgs, n_queries, GRID, GRID), dtype=np.float32)
    pils = []
    for i, (img_id, _) in enumerate(image_ids):
        ip = os.path.join(coco_root, f"{img_id}.jpg")
        pil = Image.open(ip).convert("RGB")
        pils.append(pil)
        for j, q in enumerate(queries):
            all_h[i, j] = heatmap(pil, q, bb_fn, head, sb, mean, std,
                                  clip_text, clip_tok, device)
            print(f"  [{i},{j}] {img_id} '{q}' max={all_h[i,j].max():.2f}", flush=True)

    # Render
    cell = 1.6
    header_h = 1.2
    fig_w = (n_queries + 1) * cell + 0.5
    fig_h = n_imgs * cell + header_h + 0.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_imgs + 1, n_queries + 1,
        height_ratios=[header_h] + [cell] * n_imgs,
        width_ratios=[cell * 1.1] + [cell] * n_queries,
        hspace=0.05, wspace=0.05, left=0.04, right=0.99, top=0.94, bottom=0.02,
    )

    # Top header: query labels
    ax = fig.add_subplot(gs[0, 0])
    ax.text(0.5, 0.5, "queries →\n↓ images", ha="center", va="center", fontsize=10,
            transform=ax.transAxes, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    for j, q in enumerate(queries):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.text(0.5, 0.5, q, ha="center", va="center", fontsize=10, fontweight="bold",
                transform=ax.transAxes, rotation=0)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

    # Body
    for i, (img_id, desc) in enumerate(image_ids):
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.imshow(pils[i])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"{img_id}\n{desc}", fontsize=8, rotation=0,
                      ha="right", va="center", labelpad=18)
        for s in ax.spines.values(): s.set_visible(False)

        arr = np.array(pils[i].resize((224, 224), Image.BICUBIC))
        for j, q in enumerate(queries):
            ax = fig.add_subplot(gs[i + 1, j + 1])
            ax.imshow(arr, alpha=0.55)
            h = all_h[i, j]
            h_up = np.kron(h, np.ones((arr.shape[0] // GRID + 1,
                                        arr.shape[1] // GRID + 1)))[:arr.shape[0], :arr.shape[1]]
            ax.imshow(h_up, alpha=0.65, cmap="hot", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.03, 0.97, f"{h.max():.2f}", color="white", fontsize=7,
                    fontweight="bold", transform=ax.transAxes, va="top",
                    bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none"))
            for s in ax.spines.values():
                s.set_visible(True); s.set_edgecolor("dimgray"); s.set_linewidth(0.5)

    fig.suptitle(f"Open-vocab probe: {model} ckpt — {n_imgs} images × {n_queries} queries.  "
                 f"Hot fill = predicted heatmap; max-score in top-left of each cell.",
                 fontsize=10, y=0.985)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output_path", required=True)
    args = p.parse_args()
    main(args)
