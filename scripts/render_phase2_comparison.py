"""Render the paper hero figure for Phase 2: side-by-side qualitative
comparison across (baseline CLIPSeg-distilled) vs (Phase 2 SigLIP-37k)
vs (Phase 2 SigLIP-FN-filtered, 95k).

For each (image, query) pair, renders heatmap overlays from each ckpt.

Two pages:
  page A: COCO qual-grid 8 categories (in-distribution)
  page B: Pi-captured demo frames with OOD/small-object queries
          (the user's failure cases: pen, pencil, hand, robot gripper, ...)

Output:
  figures/phase2-vs-baseline-coco.png
  figures/phase2-vs-baseline-demo.png
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
from pycocotools.coco import COCO
from train_siglip_dense_distill import build_backbone, SiglipBias, CLIP_MEAN, CLIP_STD
from train_independent_scorer_v2 import IM_MEAN, IM_STD

QUAL_PAIRS = [
    ("bird", 337987, "bird"),
    ("person", 32861, "people"),
    ("bicycle", 370208, "bicycle"),
    ("tv", 346638, "screen"),
    ("cat", 223747, "cat"),
    ("dog", 267300, "dog"),
    ("car", 151962, "car"),
    ("pizza", 232489, "pizza"),
]
COCO_ROOT = "/home/ogata/semantic-autogaze/data/coco_val2017"

DEMO_FRAMES = ["/tmp/pi_raw_frames/raw_2.jpg", "/tmp/pi_raw_frames/raw_4.jpg"]
DEMO_QUERIES = ["pen", "pencil", "hand", "robot gripper"]


def load_ckpt_inference(ckpt_path, device):
    """Returns (bb_fn, head, mean, std, kind, model_name) for inference.
    Handles SigLIP ckpts with sb calibration (baked into head._siglip_t/_bias)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ck.get("args", {})
    model = args.get("model", "v1")
    bb_fn, patch_dim, mean, std, kind, bb_module = build_backbone(
        model, device, finetune_blocks=0)
    head = TextScorerHead(
        patch_dim=patch_dim, text_dim=512,
        hidden_dim=args.get("head_hidden_dim", 384),
        n_attn_heads=args.get("head_attn_heads", 6),
        n_attn_layers=args.get("head_attn_layers", 2),
        grid_size=GRID, use_spatial=args.get("head_use_spatial", True),
    ).to(device).eval()
    head.load_state_dict(ck["head"])
    head._siglip_t = 1.0; head._siglip_bias = 0.0
    if "sb" in ck:
        sb = ck["sb"]
        log_t = sb["log_t"].item() if hasattr(sb["log_t"], "item") else float(sb["log_t"])
        bias = sb["bias"].item() if hasattr(sb["bias"], "item") else float(sb["bias"])
        head._siglip_t = float(np.exp(log_t))
        head._siglip_bias = float(bias)
    return bb_fn, head, mean, std, kind, model


def load_baseline_ckpt(model_name, device):
    """Load the original CLIPSeg-distilled ckpt for the given backbone."""
    if model_name == "v1":
        ckpt_path = "/home/ogata/semantic-autogaze/results/independent_scorer/best_v1.pt"
    elif model_name == "v2-tiny":
        ckpt_path = "/home/ogata/semantic-autogaze/results/independent_scorer_v2_tiny/best.pt"
    elif model_name == "d-mobile":
        ckpt_path = "/home/ogata/semantic-autogaze/results/sweep_v3/D_mobilenet_std/best.pt"
    else:
        raise ValueError(model_name)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cargs = ck.get("args", {}) or {}
    if model_name == "v1":
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        clip_model = clip_model.to(device).eval()
        clip_model.visual.output_tokens = True
        head = TextScorerHead(
            patch_dim=768, text_dim=512,
            hidden_dim=cargs.get("head_hidden_dim", 384),
            n_attn_heads=cargs.get("head_attn_heads", 6),
            n_attn_layers=cargs.get("head_attn_layers", 2),
            grid_size=GRID, use_spatial=cargs.get("head_use_spatial", True),
        ).to(device).eval()
        head.load_state_dict(ck["head"])
        head._siglip_t = 1.0; head._siglip_bias = 0.0
        def fn(x):
            _, p = clip_model.visual(x); return p
        return fn, head, CLIP_MEAN, CLIP_STD, "clip-visual"
    import timm
    bb = timm.create_model(ck["backbone"], pretrained=True, num_classes=0).to(device).eval()
    head = TextScorerHead(
        patch_dim=ck["embed_dim"], text_dim=512,
        hidden_dim=cargs.get("head_hidden_dim", 384),
        n_attn_heads=cargs.get("head_attn_heads", 6),
        n_attn_layers=cargs.get("head_attn_layers", 2),
        grid_size=GRID, use_spatial=cargs.get("head_use_spatial", True),
    ).to(device).eval()
    head.load_state_dict(ck["head"])
    head._siglip_t = 1.0; head._siglip_bias = 0.0
    if model_name == "v2-tiny":
        def fn(x):
            f = bb.forward_features(x)
            return f[:, 1:, :] if f.shape[1] == 197 else f
    else:
        def fn(x):
            f = bb.forward_features(x)
            f = F.interpolate(f, size=(GRID, GRID), mode="bilinear", align_corners=False)
            return f.permute(0, 2, 3, 1).reshape(f.shape[0], GRID * GRID, f.shape[1])
    return fn, head, IM_MEAN, IM_STD, "timm"


@torch.no_grad()
def heatmap(pil, query, bb_fn, head, mean, std, kind, clip_text, clip_tok, device):
    arr = np.array(pil.resize((224, 224), Image.BICUBIC))
    x = (arr.astype(np.float32) / 255.0 - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(device)
    patches = bb_fn(x)
    toks = clip_tok([query]).to(device)
    text = F.normalize(clip_text.encode_text(toks), dim=-1)
    logits = head(patches, text).reshape(GRID, GRID)
    if hasattr(head, "_siglip_t") and (head._siglip_t != 1.0 or head._siglip_bias != 0.0):
        logits = logits * head._siglip_t + head._siglip_bias
    return torch.sigmoid(logits).cpu().numpy()


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[clip-text] loading...", flush=True)
    import open_clip
    clip_text, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_text = clip_text.to(device).eval()

    print(f"[ckpts] loading 3 ckpts ...", flush=True)
    # v1 baseline (CLIPSeg-distilled)
    bb_b, head_b, mn_b, std_b, kind_b = load_baseline_ckpt(args.baseline_model, device)
    bb_p1, head_p1, mn_p1, std_p1, kind_p1, mn_p1_model = load_ckpt_inference(args.phase2_37k_ckpt, device)
    bb_p2, head_p2, mn_p2, std_p2, kind_p2, mn_p2_model = load_ckpt_inference(args.phase2_fn_ckpt, device)
    print(f"  baseline: {args.baseline_model}")
    print(f"  phase2-37k: {mn_p1_model}")
    print(f"  phase2-FN:  {mn_p2_model}")

    # ---- Page A: COCO qual grid ----
    coco = COCO(os.path.join(COCO_ROOT, "annotations", "instances_val2017.json"))
    rows_a = []
    for cat, img_id, query in QUAL_PAIRS:
        cat_id = coco.getCatIds(catNms=[cat])[0]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=False)
        anns = sorted(coco.loadAnns(ann_ids), key=lambda a: -a.get("area", 0))
        gt = coco.annToMask(anns[0]).astype(np.float32)
        info = coco.loadImgs([img_id])[0]
        pil = Image.open(os.path.join(COCO_ROOT, "val2017", info["file_name"])).convert("RGB")
        h_b = heatmap(pil, query, bb_b, head_b, mn_b, std_b, kind_b, clip_text, clip_tok, device)
        h_p1 = heatmap(pil, query, bb_p1, head_p1, mn_p1, std_p1, kind_p1, clip_text, clip_tok, device)
        h_p2 = heatmap(pil, query, bb_p2, head_p2, mn_p2, std_p2, kind_p2, clip_text, clip_tok, device)
        rows_a.append((cat, query, np.array(pil), gt, h_b, h_p1, h_p2))
        print(f"  {cat:8s} {query}: max scores -> b:{h_b.max():.2f} p1:{h_p1.max():.2f} p2:{h_p2.max():.2f}", flush=True)

    # ---- Page B: demo frames, OOD queries ----
    rows_b = []
    for fpath in DEMO_FRAMES:
        if not os.path.exists(fpath): continue
        pil = Image.open(fpath).convert("RGB").rotate(-90, expand=True)
        for q in DEMO_QUERIES:
            h_b = heatmap(pil, q, bb_b, head_b, mn_b, std_b, kind_b, clip_text, clip_tok, device)
            h_p1 = heatmap(pil, q, bb_p1, head_p1, mn_p1, std_p1, kind_p1, clip_text, clip_tok, device)
            h_p2 = heatmap(pil, q, bb_p2, head_p2, mn_p2, std_p2, kind_p2, clip_text, clip_tok, device)
            rows_b.append((os.path.basename(fpath), q, np.array(pil), h_b, h_p1, h_p2))
            print(f"  {os.path.basename(fpath)} '{q:20s}' max -> b:{h_b.max():.2f} p1:{h_p1.max():.2f} p2:{h_p2.max():.2f}", flush=True)

    # ---- Render page A: COCO ----
    cols = ["input", "GT mask", f"baseline\n({args.baseline_model})", "phase2 SigLIP\n(37k)", "phase2 SigLIP\n(95k+FN)"]
    n = len(rows_a); ncols = len(cols)
    fig, axes = plt.subplots(n, ncols, figsize=(2.0 * ncols + 0.5, 1.7 * n + 0.5))
    if n == 1: axes = axes[None, :]
    for i, (cat, query, arr, gt, h_b, h_p1, h_p2) in enumerate(rows_a):
        for col_i, label in enumerate(cols):
            ax = axes[i, col_i]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if col_i == 0:
                ax.imshow(arr)
                ax.set_ylabel(f"{cat}\n'{query}'", fontsize=8, rotation=0, ha="right", va="center", labelpad=22)
            elif col_i == 1:
                ax.imshow(arr); ax.imshow(gt, alpha=0.5, cmap="Greens")
            else:
                h = [h_b, h_p1, h_p2][col_i - 2]
                ax.imshow(arr, alpha=0.6)
                h_up = np.kron(h, np.ones((arr.shape[0] // GRID + 1, arr.shape[1] // GRID + 1)))[:arr.shape[0], :arr.shape[1]]
                ax.imshow(h_up, alpha=0.5, cmap="hot", vmin=0, vmax=1)
                ax.text(0.02, 0.95, f"max={h.max():.2f}", color="white", fontsize=7,
                        transform=ax.transAxes, va="top",
                        bbox=dict(facecolor="black", alpha=0.6, pad=2, edgecolor="none"))
            if i == 0:
                ax.set_title(label, fontsize=9)
    plt.suptitle("Phase 2: COCO qual grid — baseline vs SigLIP-37k vs SigLIP-FN", fontsize=10, y=1.005)
    plt.tight_layout()
    out_a = os.path.join(args.output_dir, "phase2-vs-baseline-coco.png")
    plt.savefig(out_a, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out_a}")

    # ---- Render page B: demo failure modes ----
    cols_b = ["input frame", f"baseline ({args.baseline_model})", "phase2 SigLIP-37k", "phase2 SigLIP-95k+FN"]
    n = len(rows_b); ncols = len(cols_b)
    fig, axes = plt.subplots(n, ncols, figsize=(2.0 * ncols + 0.5, 1.7 * n + 0.5))
    if n == 1: axes = axes[None, :]
    for i, (fname, q, arr, h_b, h_p1, h_p2) in enumerate(rows_b):
        for col_i, label in enumerate(cols_b):
            ax = axes[i, col_i]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if col_i == 0:
                ax.imshow(arr)
                ax.set_ylabel(f"{fname[:5]}\n'{q[:18]}'", fontsize=7, rotation=0, ha="right", va="center", labelpad=24)
            else:
                h = [h_b, h_p1, h_p2][col_i - 1]
                ax.imshow(arr, alpha=0.6)
                h_up = np.kron(h, np.ones((arr.shape[0] // GRID + 1, arr.shape[1] // GRID + 1)))[:arr.shape[0], :arr.shape[1]]
                ax.imshow(h_up, alpha=0.5, cmap="hot", vmin=0, vmax=1)
                ax.text(0.02, 0.95, f"max={h.max():.2f}", color="white", fontsize=7,
                        transform=ax.transAxes, va="top",
                        bbox=dict(facecolor="black", alpha=0.6, pad=2, edgecolor="none"))
            if i == 0:
                ax.set_title(label, fontsize=9)
    plt.suptitle("Phase 2: demo failure modes (pen / pencil OOD, hand / gripper present)\n"
                 "baseline blind on small/OOD; SigLIP variants discriminate", fontsize=10, y=1.005)
    plt.tight_layout()
    out_b = os.path.join(args.output_dir, "phase2-vs-baseline-demo.png")
    plt.savefig(out_b, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out_b}")

    if args.library_figures_dir:
        os.makedirs(args.library_figures_dir, exist_ok=True)
        import shutil
        for f in [out_a, out_b]:
            shutil.copy(f, os.path.join(args.library_figures_dir, os.path.basename(f)))
            print(f"  -> {args.library_figures_dir}/{os.path.basename(f)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--baseline_model", default="v2-tiny", choices=["v1", "v2-tiny", "d-mobile"])
    p.add_argument("--phase2_37k_ckpt", default="/home/ogata/semantic-autogaze/results/phase2_v2tiny_run2_37k/ckpt_step6000.pt")
    p.add_argument("--phase2_fn_ckpt", default="/home/ogata/semantic-autogaze/results/phase2_v2tiny_fn/best.pt")
    p.add_argument("--output_dir", default="/home/ogata/semantic-autogaze/results/phase2_comparison")
    p.add_argument("--library_figures_dir",
                   default="/home/ogata/mac-brain/projects/semantic-autogaze/figures")
    args = p.parse_args()
    main(args)
