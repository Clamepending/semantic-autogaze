"""One-shot multi-model qual eval for the user-supplied validation image at
/home/ogata/mac-brain/projects/semantic-autogaze/validation/.

Renders side-by-side heatmap panels per query for:
  - phase24d (current Pi-class outdoor leader, v0.7.0)
  - phase19b v0.6.0 (older Pi default)
  - phase25 DINOv2-s (heavier indoor recipe)
  - CLIPSeg (the indoor-winning teacher we're distilling from)
  - MaskCLIP (lit baseline)

Query set is tailored to indoor desk scenes (the 20260501_215915 image):
common things, fine objects, multi-word, stuff, abstention.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image as _PIL

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")

from scripts.eval_phase2_ckpt import load_ckpt, heatmap_one, GRID

# Tailored to the desk scene from inspection of the image:
# laptop (floor lower-left), Acer monitor (left of desk), stack of books with
# playing cards on top, "UPSIDE DOWN LABS" cardboard box, "Seeed Studio AI
# Robotics" teal mat, white 3D-printed robot arm + cables, person in grey
# Champion hoodie holding a VR controller, black backpack on floor, door
# with vent on right, wood floor (right) vs white desk surface.
QUERIES = [
    # POSITIVE - present in image
    # Easy / obvious
    "person", "books", "monitor", "laptop", "backpack",
    "wall", "door", "wood floor",
    # Medium / specific objects
    "white desk", "cables", "playing cards", "robotic arm",
    "cardboard box",
    # Hard / fine-grained
    "hoodie", "white sock", "VR controller",
    # NEGATIVE - must abstain (clean negatives)
    "tree", "mountain", "ocean", "car", "bicycle", "elephant",
    # NEGATIVE - misleading near-miss (most diagnostic)
    "wooden desk",   # desk is white plastic; FALSE
    "blue carpet",   # no carpet at all; FALSE (envvideo carry-over)
    "green wall",    # wall is white; FALSE
    "tennis racket", # near-miss for the VR controller; FALSE
]

# Layout: 9 rows x 3 cols = 27 cells (>= 25 queries).
ROWS = 9
COLS = 3
assert len(QUERIES) <= ROWS * COLS


def render_one_model(pil, model_label, heats, save_path):
    fig = plt.figure(figsize=(COLS * 3.2, (ROWS + 1) * 3.0))
    gs = fig.add_gridspec(ROWS + 1, COLS)
    ax_in = fig.add_subplot(gs[0, :])
    ax_in.imshow(pil)
    ax_in.set_title(f"{model_label}", fontsize=12)
    ax_in.set_xticks([]); ax_in.set_yticks([])
    arr = np.array(pil)
    H, W = arr.shape[:2]
    for i, (q, heat, hmax, hmean) in enumerate(heats):
        r = 1 + (i // COLS); c = i % COLS
        ax = fig.add_subplot(gs[r, c])
        # heat is (G, G) in [0, 1]; nearest-upsample to image size
        G = heat.shape[0]
        heat_up = np.kron(heat, np.ones((H // G + 1, W // G + 1)))[:H, :W]
        ax.imshow(arr, alpha=0.55)
        ax.imshow(heat_up, alpha=0.55, cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"'{q}'  max={hmax:.2f} mean={hmean:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def render_grid(pil, all_models, queries, save_path):
    """Side-by-side: each row = 1 query, each col = 1 model.
    Top strip = input."""
    n_q = len(queries)
    n_m = len(all_models)
    fig = plt.figure(figsize=(n_m * 3.4, (n_q + 1) * 2.6))
    gs = fig.add_gridspec(n_q + 1, max(n_m, 3))
    ax_in = fig.add_subplot(gs[0, :])
    ax_in.imshow(pil)
    ax_in.set_title("input", fontsize=11)
    ax_in.set_xticks([]); ax_in.set_yticks([])
    arr = np.array(pil)
    H, W = arr.shape[:2]
    for q_i, q in enumerate(queries):
        for m_i, (label, heats_dict) in enumerate(all_models):
            ax = fig.add_subplot(gs[q_i + 1, m_i])
            heat, hmax, hmean = heats_dict[q]
            G = heat.shape[0]
            heat_up = np.kron(heat, np.ones((H // G + 1, W // G + 1)))[:H, :W]
            ax.imshow(arr, alpha=0.55)
            ax.imshow(heat_up, alpha=0.55, cmap="hot", vmin=0, vmax=1)
            title = f"'{q}'  max={hmax:.2f}" if q_i + m_i == 0 else f"max={hmax:.2f} mean={hmean:.2f}"
            ax.set_title(title, fontsize=8)
            if m_i == 0:
                ax.set_ylabel(q, fontsize=10, rotation=0, labelpad=60, va="center", ha="right")
            if q_i == 0:
                ax.text(0.5, 1.15, label, transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=11, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_internal_ckpt(ckpt_path, pil, queries, device):
    """Returns dict[query] = (heat14, hmax, hmean)."""
    bb_fn, head, sb, mean, std, model, kind, bb_module, obj_head = load_ckpt(ckpt_path, device)
    import open_clip
    text_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    text_model = text_model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-B-16")
    out = {}
    for q in queries:
        heat = heatmap_one(pil, q, bb_fn, head, sb, mean, std, text_model, tok, device, obj_head=obj_head)
        out[q] = (heat, float(heat.max()), float(heat.mean()))
    # Free
    del bb_fn, head, sb, bb_module, text_model
    if obj_head is not None:
        del obj_head
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def run_clipseg(pil, queries, device):
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device).eval()
    pil_352 = pil.resize((352, 352), _PIL.BICUBIC)
    out = {}
    for q in queries:
        inputs = proc(text=[q], images=[pil_352], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        if logits.dim() == 2: logits = logits.unsqueeze(0)
        h352 = torch.sigmoid(logits[0]).cpu().numpy().astype(np.float32)
        h14 = cv2.resize(h352, (GRID, GRID), interpolation=cv2.INTER_AREA)
        h14 = np.clip(h14, 0.0, 1.0)
        out[q] = (h14, float(h14.max()), float(h14.mean()))
    del model
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def run_maskclip(pil, queries, device):
    """MaskCLIP using the helper functions from baseline_maskclip.py."""
    import torch.nn.functional as F
    from scripts.baseline_maskclip import (
        maskclip_patch_features, preprocess_image, cosine_to_score,
    )
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai", force_quick_gelu=True
    )
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-B-16")
    visual = model.visual

    text_tokens = tok(list(queries)).to(device)
    text_emb = F.normalize(model.encode_text(text_tokens), dim=-1)  # (K, 512)
    img_t = preprocess_image(pil).to(device)
    patch_feats = maskclip_patch_features(visual, img_t, apply_ffn=True)
    patch_feats = F.normalize(patch_feats, dim=-1)
    cos = torch.einsum("bhwd,kd->bhwk", patch_feats, text_emb)
    scores = cosine_to_score(cos, "cliptemp").squeeze(0).cpu().numpy()  # (14, 14, K)
    out = {}
    for i, q in enumerate(queries):
        heat = scores[..., i]
        out[q] = (heat, float(heat.max()), float(heat.mean()))
    del model
    torch.cuda.empty_cache()
    return out


def main(args):
    device = torch.device(args.device)
    pil = _PIL.open(args.image).convert("RGB")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[image] {args.image} -> {pil.size}", flush=True)
    print(f"[queries] {len(QUERIES)}: {QUERIES}", flush=True)

    all_models = []
    for label, ckpt in [
        ("phase29q (atto)", "/home/ogata/semantic-autogaze/results/phase29q_atto_lambda1_no_existing_distill_10k/best_val.pt"),
        ("phase24d (v0.7.0 ship)", "/home/ogata/semantic-autogaze/results/phase24d_atto_mpp05_aggrAug_10k/best_val.pt"),
        ("phase32c (ConvNeXt-large 50K)", "/home/ogata/semantic-autogaze/results/phase32c_convnext_large_q_lr1e4_50k/best_val.pt"),
        ("phase32d (DINOv2-large 50K)", "/home/ogata/semantic-autogaze/results/phase32d_dinov2l_q_lr1e4_50k/best_val.pt"),
    ]:
        if not Path(ckpt).exists():
            print(f"[skip] {label} ckpt missing: {ckpt}", flush=True)
            continue
        print(f"[run] {label}", flush=True)
        heats = run_internal_ckpt(ckpt, pil, QUERIES, device)
        all_models.append((label, heats))
        for q in QUERIES:
            h, hmax, hmean = heats[q]
            print(f"  '{q:18s}' max={hmax:.3f} mean={hmean:.3f}", flush=True)

    print(f"[run] CLIPSeg (HF CIDAS/clipseg-rd64-refined)", flush=True)
    cs_heats = run_clipseg(pil, QUERIES, device)
    all_models.append(("CLIPSeg (HF rd64-refined)", cs_heats))
    for q in QUERIES:
        h, hmax, hmean = cs_heats[q]
        print(f"  '{q:18s}' max={hmax:.3f} mean={hmean:.3f}", flush=True)

    print(f"[run] MaskCLIP", flush=True)
    try:
        mc_heats = run_maskclip(pil, QUERIES, device)
        all_models.append(("MaskCLIP (CLIP-B/16)", mc_heats))
        for q in QUERIES:
            h, hmax, hmean = mc_heats[q]
            print(f"  '{q:18s}' max={hmax:.3f} mean={hmean:.3f}", flush=True)
    except Exception as e:
        print(f"[skip] MaskCLIP failed: {e}", flush=True)

    # Per-model panels
    for label, heats in all_models:
        slug = label.split()[0].lower().replace("(", "").replace(")", "")
        save = out_dir / f"panel_{slug}.png"
        ordered = [(q, *heats[q]) for q in QUERIES]
        render_one_model(pil, label, ordered, save)
        print(f"[saved] {save}", flush=True)

    # Cross-model grid skipped (matplotlib renders >150 image-cells too slowly).
    # The per-model panels + summary.csv are enough for cross-model comparison.

    # CSV
    csv_save = out_dir / "summary.csv"
    with open(csv_save, "w") as f:
        f.write("model,query,hmax,hmean\n")
        for label, heats in all_models:
            for q in QUERIES:
                h, hmax, hmean = heats[q]
                f.write(f'"{label}",{q},{hmax:.4f},{hmean:.4f}\n')
    print(f"[saved] {csv_save}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="/home/ogata/mac-brain/projects/semantic-autogaze/validation/20260501_215915.jpg")
    p.add_argument("--output_dir", default="/home/ogata/mac-brain/projects/semantic-autogaze/figures/user_validation_20260501_215915")
    p.add_argument("--device", default="cuda:0")
    main(p.parse_args())
