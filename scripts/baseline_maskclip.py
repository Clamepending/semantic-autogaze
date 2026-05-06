"""MaskCLIP (Zhou et al., ECCV 2022, arxiv 2112.01071) — training-free
open-vocabulary heatmap baseline.

Trick: take a stock CLIP ViT-B/16 (`open_clip` with `pretrained="openai"`).
For all transformer blocks except the last, run normally. For the *last*
self-attention, replace `softmax(Q@K^T/sqrt(d)) @ V` with just `V` (after
the V projection and `out_proj`). Optionally apply or skip the FFN — we
default to applying the FFN (variant 2 in the paper) but expose a flag.
After the last block: drop the CLS token, apply `ln_post` and `visual.proj`
per-patch -> 14x14 patch features in CLIP's 512-d shared space.

Heatmap = patch_features @ text_emb (cosine in [-1, 1]).
We render `sigmoid(t * cosine + b)` with `t=100, b=-1.0` to map to [0, 1]
in a way that makes "high cosine" = "fires" without any per-image scaling.
A `--score linear` fallback uses `(cos+1)/2` for visual diff.

Outputs match `scripts/qual_openvocab_eval.py:render_panel` so the existing
`scripts/compare_openvocab_sweep.py` aggregator picks up `summary.csv`.

Run:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.baseline_maskclip \
    --image_dir /home/ogata/semantic-autogaze/data/eval_openvocab_streets \
    --output_dir /home/ogata/mac-brain/projects/semantic-autogaze/figures/openvocab_eval/baseline_maskclip_vitb16 \
    --kw_set street
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image as _PIL

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")
# Reuse the canonical KEYWORDS lists so cross-comparison stays honest.
from scripts.qual_openvocab_eval import KEYWORDS as STREET_KEYWORDS
from scripts.qual_envvideo_eval import KEYWORDS as ENVVIDEO_KEYWORDS

GRID = 14  # ViT-B/16 at 224x224 -> 14x14 patches

# CLIP / OpenAI normalization
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@torch.no_grad()
def maskclip_patch_features(visual, image_tensor, apply_ffn: bool = True):
    """Run a CLIP ViT visual encoder up to and through a value-only last block.

    Returns per-patch features in the 512-d CLIP shared space, shape (B, 14, 14, 512).
    """
    x = visual.conv1(image_tensor)  # (B, 768, 14, 14)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, 196, 768)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls, x], dim=1)  # (B, 197, 768)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    # open_clip ResidualAttentionBlock expects (B, N, D) and uses batch_first
    # on its MultiheadAttention if `batch_first=True`. To stay safe we feed
    # (N, B, D) (the open_clip pre-3.0 convention) — open_clip 3.x's block
    # accepts both via `_call_attn` which handles permutation internally.
    # Easiest: just call `block(x)` for normal blocks.

    # Run all but last block via the standard forward.
    for blk in visual.transformer.resblocks[:-1]:
        x = blk(x)

    last = visual.transformer.resblocks[-1]

    # MaskCLIP last-block: pre-norm, then value-only attention, then optional FFN.
    x_norm = last.ln_1(x)
    in_w = last.attn.in_proj_weight  # (3*D, D)
    in_b = last.attn.in_proj_bias    # (3*D,)
    D = x_norm.shape[-1]
    # V slab is rows [2D:3D]
    v = F.linear(x_norm, in_w[2 * D:3 * D, :], in_b[2 * D:3 * D])  # (B, N, D)
    v = last.attn.out_proj(v)  # (B, N, D)
    x = x + v  # residual

    if apply_ffn:
        x = x + last.mlp(last.ln_2(x))

    # Drop CLS, apply ln_post and proj per-patch
    patch_tokens = x[:, 1:, :]  # (B, 196, 768)
    patch_tokens = visual.ln_post(patch_tokens)
    if visual.proj is not None:
        patch_tokens = patch_tokens @ visual.proj  # (B, 196, 512)
    B = patch_tokens.shape[0]
    return patch_tokens.reshape(B, GRID, GRID, patch_tokens.shape[-1])


def preprocess_image(pil) -> torch.Tensor:
    arr = np.array(pil.resize((224, 224), _PIL.BICUBIC)).astype(np.float32) / 255.0
    arr = (arr - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()  # (1,3,224,224)


def render_panel(pil, image_label, heats, save_path):
    """Mirrors scripts/qual_openvocab_eval.py:render_panel."""
    n = len(heats)
    cols = 4
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 3.0, (rows + 1) * 3.0))
    gs = fig.add_gridspec(rows + 1, cols)

    ax_in = fig.add_subplot(gs[0, :])
    ax_in.imshow(pil)
    ax_in.set_title(f"input: {image_label}", fontsize=10)
    ax_in.set_xticks([]); ax_in.set_yticks([])

    arr = np.array(pil)
    H, W = arr.shape[:2]
    for i, (q, heat, hmax, hmean) in enumerate(heats):
        r = 1 + (i // cols); c = i % cols
        ax = fig.add_subplot(gs[r, c])
        heat_up = np.kron(heat, np.ones((H // GRID + 1, W // GRID + 1)))[:H, :W]
        ax.imshow(arr, alpha=0.55)
        ax.imshow(heat_up, alpha=0.55, cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"'{q}'  max={hmax:.2f} mean={hmean:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def cosine_to_score(cos: torch.Tensor, mode: str, t: float = 100.0, b: float = -1.0) -> torch.Tensor:
    """Map cosine in [-1, 1] -> score in [0, 1] for visualization + thresholding.

    Modes:
      sigmoid : sigmoid(t * cos + b). For CLIP-style cosines (~0.1-0.4) with
                t=100, b=+1.0 this saturates fully (~1.0 everywhere) — degenerate.
                Kept for reference / spec compliance.
      linear  : (cos + 1) / 2. Monotonic; range typically [0.55, 0.7] for
                CLIP cosines on natural images. Documented spec fallback.
      cliptemp: sigmoid(20*(cos - 0.22)). Empirical "fair" temperature for CLIP
                ViT-B/16 cosines so that the natural cosine spread (0.1-0.4)
                covers most of [0, 1]. Lets the aggregator's 0.45 threshold
                actually discriminate. NOT a published method — pragmatic.
      minmax  : per-(image,query) min-max scaling. Standard MaskCLIP visualization.
                Pure spatial structure; max=1.0 for every query so abstention
                is undefined. Useful for visual inspection only.
    """
    if mode == "sigmoid":
        return torch.sigmoid(t * cos + b)
    elif mode == "linear":
        return torch.clamp((cos + 1.0) * 0.5, 0.0, 1.0)
    elif mode == "cliptemp":
        return torch.sigmoid(20.0 * (cos - 0.22))
    elif mode == "minmax":
        orig_shape = cos.shape
        H, W, K = orig_shape[-3], orig_shape[-2], orig_shape[-1]
        flat = cos.reshape(-1, H * W, K)
        cos_min = flat.min(dim=1, keepdim=True).values
        cos_max = flat.max(dim=1, keepdim=True).values
        cos_range = (cos_max - cos_min).clamp_min(1e-6)
        spread = (flat - cos_min) / cos_range
        return spread.reshape(orig_shape)
    else:
        raise ValueError(f"unknown score mode: {mode}")


def main(args):
    device = torch.device(args.device)
    import open_clip

    print(f"[load] open_clip ViT-B-16 pretrained=openai (force_quick_gelu=True)", flush=True)
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai", force_quick_gelu=True
    )
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-B-16")
    visual = model.visual

    # Pick keyword set
    if args.kw_set == "street":
        KEYWORDS = STREET_KEYWORDS
    elif args.kw_set == "envvideo":
        KEYWORDS = ENVVIDEO_KEYWORDS
    else:
        raise SystemExit(f"--kw_set must be 'street' or 'envvideo' (got {args.kw_set})")

    # Pre-encode text once
    print(f"[text] encoding {len(KEYWORDS)} queries", flush=True)
    text_tokens = tok(list(KEYWORDS)).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens)  # (K, 512)
    text_emb = F.normalize(text_emb, dim=-1)  # (K, 512)

    img_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    if args.frames:
        keep = set(int(f) for f in args.frames.split(","))
        image_paths = [p for p in image_paths
                       if (p.stem.startswith("frame_") and int(p.stem.split("_")[1]) in keep)
                       or not p.stem.startswith("frame_")]

    aggregate = []
    for img_p in image_paths:
        pil = _PIL.open(img_p).convert("RGB")
        img_t = preprocess_image(pil).to(device)
        with torch.no_grad():
            patch_feats = maskclip_patch_features(visual, img_t, apply_ffn=args.apply_ffn)  # (1, 14, 14, 512)
        patch_feats = F.normalize(patch_feats, dim=-1)
        # cosine: (1, 14, 14, 512) x (K, 512) -> (1, 14, 14, K)
        cos = torch.einsum("bhwd,kd->bhwk", patch_feats, text_emb)  # in [-1, 1]
        scores = cosine_to_score(cos, args.score, t=args.sigmoid_t, b=args.sigmoid_b)
        scores = scores.squeeze(0).cpu().numpy()  # (14, 14, K)

        heats = []
        for i, q in enumerate(KEYWORDS):
            heat = scores[..., i]
            hmax = float(heat.max()); hmean = float(heat.mean())
            heats.append((q, heat, hmax, hmean))
            aggregate.append((img_p.stem, q, hmax, hmean))
            print(f"  [{img_p.stem:35s}] '{q:14s}' max={hmax:.3f} mean={hmean:.3f}", flush=True)
        save_path = out_dir / f"{img_p.stem}.png"
        render_panel(pil, img_p.name, heats, save_path)
        print(f"  [saved] {save_path}", flush=True)

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("image,query,hmax,hmean\n")
        for img, q, hmax, hmean in aggregate:
            f.write(f"{img},{q},{hmax:.4f},{hmean:.4f}\n")
    print(f"[saved] {csv_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--kw_set", choices=["street", "envvideo"], required=True,
                   help="Pick KEYWORDS list: 'street' from qual_openvocab_eval, "
                        "'envvideo' from qual_envvideo_eval. Order is preserved.")
    p.add_argument("--score", choices=["sigmoid", "linear", "cliptemp", "minmax"],
                   default="cliptemp",
                   help="How to map cosine -> [0,1]. 'sigmoid' = sigmoid(t*cos+b) "
                        "(saturates for CLIP cosines; spec value, kept for ref). "
                        "'linear' = (cos+1)/2 (documented fallback; bunched). "
                        "'cliptemp' = sigmoid(20*(cos-0.22)) — empirical centering "
                        "on CLIP's natural cosine spread; default. "
                        "'minmax' = per-(img,query) min-max scaling (no abstention).")
    p.add_argument("--sigmoid_t", type=float, default=100.0)
    p.add_argument("--sigmoid_b", type=float, default=-1.0)
    p.add_argument("--apply_ffn", action="store_true", default=True,
                   help="Apply the last block's FFN (paper variant 2). "
                        "Default ON. Pass --no_ffn to disable (variant 1).")
    p.add_argument("--no_ffn", dest="apply_ffn", action="store_false")
    p.add_argument("--frames", default="",
                   help="Comma-separated frame numbers to keep (envvideo only). "
                        "Empty => all images.")
    main(p.parse_args())
