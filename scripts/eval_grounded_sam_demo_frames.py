"""Phase 1b: run all 3 teachers on actual demo-captured webcam frames.

Tests on real demo failure modes (small objects, text-busy backgrounds,
OOD vocab) — not the COCO qual grid which is too easy.

For each (frame, query) pair, runs:
  - Ours v1 (CLIP-B/16 + std head, COCO-mIoU 0.79)
  - CLIPSeg (current teacher)
  - Grounded-SAM (gdino-base + sam-huge, top-1 box)

Renders a side-by-side figure with per-cell visualization.
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")
from train_independent_scorer import TextScorerHead, GRID
from scripts.eval_grounded_sam_ceiling import (
    heatmap_ours_v1, heatmap_clipseg, mask_grounded_sam,
    OURS_CKPT, CLIPSEG_NAME, GDINO_NAME, SAM_NAME,
)

FRAMES_DIR = "/tmp/pi_raw_frames"
DEFAULT_FRAMES = ["raw_0.jpg", "raw_2.jpg", "raw_4.jpg"]
DEFAULT_QUERIES = ["pen", "pencil", "Hugging Face logo", "text", "white object"]
ROTATE_90 = True  # rotate 90° CW so models see what the demo sees


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("loading CLIP ViT-B/16 ...", flush=True)
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_model = clip_model.to(device).eval()

    print(f"loading Ours v1 head from {OURS_CKPT}...", flush=True)
    ck = torch.load(OURS_CKPT, map_location=device)
    ca = ck.get("args", {}) or {}
    head = TextScorerHead(
        patch_dim=768, text_dim=512,
        hidden_dim=ca.get("head_hidden_dim", 384),
        n_attn_heads=ca.get("head_attn_heads", 6),
        n_attn_layers=ca.get("head_attn_layers", 2),
        grid_size=GRID, use_spatial=ca.get("head_use_spatial", True),
    ).to(device).eval()
    head.load_state_dict(ck["head"])

    print(f"loading CLIPSeg {CLIPSEG_NAME}...", flush=True)
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    clipseg_proc = CLIPSegProcessor.from_pretrained(CLIPSEG_NAME)
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_NAME).to(device).eval()

    print(f"loading GroundingDINO {GDINO_NAME}...", flush=True)
    from transformers import AutoProcessor as _AP, AutoModelForZeroShotObjectDetection
    gdino_proc = _AP.from_pretrained(GDINO_NAME)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_NAME).to(device).eval()

    print(f"loading SAM {SAM_NAME}...", flush=True)
    from transformers import SamModel, SamProcessor
    sam_proc = SamProcessor.from_pretrained(SAM_NAME)
    sam_model = SamModel.from_pretrained(SAM_NAME).to(device).eval()

    rows = []
    for fname in args.frames:
        fpath = os.path.join(FRAMES_DIR, fname)
        pil = Image.open(fpath).convert("RGB")
        if ROTATE_90:
            pil = pil.rotate(-90, expand=True)  # 90 CW
        arr = np.array(pil)
        for query in args.queries:
            print(f"\n=== {fname} / '{query}' ===", flush=True)
            h_v1 = heatmap_ours_v1(clip_model, head, pil, query, device)
            h_cs = heatmap_clipseg(clipseg_model, clipseg_proc, pil, query, device)
            gsam_mask, gsam_boxes, gsam_score = mask_grounded_sam(
                gdino_model, gdino_proc, sam_model, sam_proc, pil, query, device,
                box_threshold=0.30, text_threshold=0.25, top1_only=True)
            print(f"  v1 max={h_v1.max():.3f} min={h_v1.min():.3f} std={h_v1.std():.3f}")
            print(f"  cs max={h_cs.max():.3f} min={h_cs.min():.3f} std={h_cs.std():.3f}")
            print(f"  gsam: {len(gsam_boxes)} boxes (top score {gsam_score:.2f}), mask sum={int(gsam_mask.sum())}")
            rows.append((fname, query, arr, h_v1, h_cs, gsam_mask, gsam_score))

    # ---- Render figure: rows = (frame, query) pairs, cols = methods
    methods = ["frame", "CLIPSeg", "Ours v1", "Grounded-SAM"]
    n_rows = len(rows); n_cols = len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols, 1.8 * n_rows))
    if n_rows == 1: axes = axes[None, :]

    for i, (fname, query, arr, h_v1, h_cs, gsam_mask, gsam_score) in enumerate(rows):
        ax = axes[i, 0]; ax.imshow(arr); ax.set_xticks([]); ax.set_yticks([])
        if i == 0: ax.set_title(methods[0], fontsize=10)
        ax.set_ylabel(f"{fname[:5]}\n'{query[:18]}'", fontsize=8, rotation=0, ha="right", va="center", labelpad=24)

        for col_i, (label, h14) in enumerate([("CLIPSeg", h_cs), ("Ours v1", h_v1)]):
            ax = axes[i, 1 + col_i]
            ax.imshow(arr, alpha=0.6)
            # Per-frame max-normalize for fair display
            hn = (h14 - h14.min()) / max(h14.max() - h14.min(), 1e-6)
            h_up = np.kron(hn, np.ones((arr.shape[0] // GRID + 1, arr.shape[1] // GRID + 1)))[
                :arr.shape[0], :arr.shape[1]]
            ax.imshow(h_up, alpha=0.5, cmap="hot", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.02, 0.95, f"max={h14.max():.2f}", color="white", fontsize=8,
                    transform=ax.transAxes, va="top",
                    bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"))
            if i == 0: ax.set_title(label, fontsize=10)

        ax = axes[i, 3]
        ax.imshow(arr, alpha=0.6)
        ax.imshow(gsam_mask.astype(np.float32), alpha=0.55, cmap="hot", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.02, 0.95, f"score={gsam_score:.2f}\n{int(gsam_mask.sum())}px",
                color="white", fontsize=8, transform=ax.transAxes, va="top",
                bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"))
        if i == 0: ax.set_title(methods[3], fontsize=10)

    plt.suptitle("Phase 1b: teachers on demo-captured webcam frames", fontsize=11, y=1.001)
    plt.tight_layout()
    out_png = os.path.join(args.output_dir, "demo_frames_teacher_grid.png")
    plt.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"\nsaved {out_png}")

    if args.library_figures_dir:
        os.makedirs(args.library_figures_dir, exist_ok=True)
        import shutil
        lib_png = os.path.join(args.library_figures_dir, "phase1b-teachers-on-demo-frames.png")
        shutil.copy(out_png, lib_png)
        print(f"  -> {lib_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/phase1_ceiling_teacher")
    p.add_argument("--library_figures_dir",
                   default="/home/ogata/mac-brain/projects/semantic-autogaze/figures")
    p.add_argument("--frames", nargs="+", default=DEFAULT_FRAMES)
    p.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    args = p.parse_args()
    main(args)
