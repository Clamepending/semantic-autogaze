"""
Qualitative 4-frame comparison: AutoGaze-kept patches vs AutoGaze+semantic-kept.

For a set of validation videos and a text query, render a grid showing:
  row 0: the raw frame (4 representative frames per video)
  row 1: AutoGaze-only kept patches (adaptive) — green overlay
  row 2: AutoGaze ∩ semantic (Intersect 50%, query-conditioned) — blue overlay
  row 3: AutoGaze ∩ semantic (Intersect 10%, the "tight" selection) — red overlay

The point is to *show* that the semantic filter preserves coverage of the
queried subject while discarding patches that AutoGaze kept for purely
reconstruction-based reasons (texture, motion, contrast) but are off-subject.

Safety: AutoGaze's native `gazing_pos` can index a wider token space than
T*196 for multi-scale configs. We ALWAYS clamp to [0, TOTAL-1] before using
positions as indices.

Usage:
    CUDA_VISIBLE_DEVICES=5 python3 -m semantic_autogaze.qual_frame_comparison \\
        --device cuda:0 --n_videos 4 --query people \\
        --output_dir results/qual_frame_compare
"""

import os
import glob
import json
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open_clip
import av

from autogaze.models.autogaze import AutoGazeImageProcessor
from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper


T_FRAMES = 16
GRID = 14
N_PATCHES = GRID * GRID  # 196
TOTAL = T_FRAMES * N_PATCHES  # 3136


def clip_text_embed(text, model, tok, device):
    tokens = tok([text]).to(device)
    with torch.no_grad():
        f = model.encode_text(tokens)
        f = F.normalize(f, dim=-1)
    return f


def info_to_per_frame_mask(info):
    """Build a (T, GRID, GRID) boolean mask from a gazing_info dict.

    Uses clamp to keep positions in [0, TOTAL-1] so multi-scale token
    positions do not cause OOB indexing (canonical fix pattern).
    """
    gp = info["gazing_pos"]  # (B, K)
    pd = info["if_padded_gazing"]  # (B, K)
    valid = (~pd[0]).cpu()
    pos = gp[0].cpu()
    pos = pos[valid]
    pos = pos.clamp(0, TOTAL - 1).numpy().astype(np.int64)
    flat = np.zeros(TOTAL, dtype=bool)
    flat[pos] = True
    return flat.reshape(T_FRAMES, GRID, GRID)


def pick_4_frames(ag_mask_per_frame):
    """Choose 4 frame indices that best illustrate the AutoGaze selection:
    spread across the clip and prefer frames with moderate-to-high AutoGaze
    keep counts (so the overlay has something visible).
    """
    counts = ag_mask_per_frame.sum(axis=(1, 2))  # (T,)
    # Stratified: 4 quartiles across time, pick frame with highest count in each
    idx = []
    for lo, hi in [(0, 4), (4, 8), (8, 12), (12, 16)]:
        sub = counts[lo:hi]
        idx.append(int(lo + int(np.argmax(sub))))
    return idx


def overlay_mask(ax, frame_img, mask_2d, color_rgb=(0.2, 1.0, 0.2), alpha=0.45):
    """Show frame with semi-transparent per-patch mask overlay + grid lines."""
    ax.imshow(frame_img)
    H, W = frame_img.shape[:2]
    # Build per-pixel RGBA overlay at the frame resolution
    ph, pw = H // GRID, W // GRID
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    for r in range(GRID):
        for c in range(GRID):
            if mask_2d[r, c]:
                overlay[r*ph:(r+1)*ph, c*pw:(c+1)*pw, :3] = color_rgb
                overlay[r*ph:(r+1)*ph, c*pw:(c+1)*pw, 3] = alpha
    ax.imshow(overlay)
    # Light grid
    for i in range(1, GRID):
        ax.axhline(i * ph, color="white", lw=0.2, alpha=0.4)
        ax.axvline(i * pw, color="white", lw=0.2, alpha=0.4)
    ax.set_xticks([]); ax.set_yticks([])


def main(args):
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("[setup] Loading AutoGaze + head + CLIP ...")
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )
    ag_tf = AutoGazeImageProcessor.from_pretrained(args.autogaze_model)

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()

    # Use SigLIP processor only to get decent-looking frames for display (we
    # don't actually run SigLIP here — display only).
    from transformers import AutoImageProcessor
    siglip_tf = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")

    videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    random.shuffle(videos)
    videos = videos[: args.n_videos]
    print(f"[eval] n_videos = {len(videos)}, query = '{args.query}'")

    q_emb = clip_text_embed(args.query, clip_model, clip_tok, device)

    report_rows = []
    for vi, vp in enumerate(videos):
        name = os.path.basename(vp)
        try:
            c = av.open(vp)
            raw = read_video_pyav(container=c, indices=list(range(T_FRAMES)))
            c.close()
            if raw.shape[0] < T_FRAMES:
                print(f"  skip {name}: short clip")
                continue
            v_ag = transform_video_for_pytorch(raw, ag_tf)[None].to(device)
            v_sg = transform_video_for_pytorch(raw, siglip_tf)[None].to(device)
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue

        # Three runs against the SAME clip
        info_gaze = wrapper.forward(v_ag, q_emb, mode="gaze_only",
                                    gazing_ratio=args.gazing_ratio,
                                    task_loss_requirement=args.task_loss_req,
                                    semantic_keep_ratio=1.0)
        info_i50 = wrapper.forward(v_ag, q_emb, mode="intersect",
                                   gazing_ratio=args.gazing_ratio,
                                   task_loss_requirement=args.task_loss_req,
                                   semantic_keep_ratio=0.5)
        info_i10 = wrapper.forward(v_ag, q_emb, mode="intersect",
                                   gazing_ratio=args.gazing_ratio,
                                   task_loss_requirement=args.task_loss_req,
                                   semantic_keep_ratio=0.1)

        m_gaze = info_to_per_frame_mask(info_gaze)   # (T, 14, 14)
        m_i50 = info_to_per_frame_mask(info_i50)
        m_i10 = info_to_per_frame_mask(info_i10)

        n_gaze = int(m_gaze.sum())
        n_i50 = int(m_i50.sum())
        n_i10 = int(m_i10.sum())

        # Build 4-frame panel
        fidx = pick_4_frames(m_gaze)

        # Render from SigLIP-normalized tensor for a clean 224×224 frame
        frames_np = v_sg[0].cpu().numpy().transpose(0, 2, 3, 1)
        frames_np = (frames_np - frames_np.min()) / (frames_np.max() - frames_np.min() + 1e-9)

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for col, t in enumerate(fidx):
            frame = frames_np[t]

            ax = axes[0, col]
            ax.imshow(frame)
            ax.set_title(f"frame {t}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0: ax.set_ylabel("raw", fontsize=11)

            ax = axes[1, col]
            overlay_mask(ax, frame, m_gaze[t], color_rgb=(0.2, 0.9, 0.2), alpha=0.45)
            ax.set_title(f"AutoGaze: {int(m_gaze[t].sum())}/196", fontsize=9)
            if col == 0: ax.set_ylabel("AutoGaze\nonly", fontsize=11)

            ax = axes[2, col]
            overlay_mask(ax, frame, m_i50[t], color_rgb=(0.15, 0.5, 1.0), alpha=0.50)
            ax.set_title(f"AG∩sem 50%: {int(m_i50[t].sum())}/{int(m_gaze[t].sum()) if m_gaze[t].sum() else 1}",
                         fontsize=9)
            if col == 0: ax.set_ylabel(f"+ semantic\n'{args.query}' 50%", fontsize=11)

            ax = axes[3, col]
            overlay_mask(ax, frame, m_i10[t], color_rgb=(1.0, 0.2, 0.2), alpha=0.55)
            ax.set_title(f"AG∩sem 10%: {int(m_i10[t].sum())}/{int(m_gaze[t].sum()) if m_gaze[t].sum() else 1}",
                         fontsize=9)
            if col == 0: ax.set_ylabel(f"+ semantic\n'{args.query}' 10%", fontsize=11)

        fig.suptitle(
            f"{name}    query=\"{args.query}\"    "
            f"AutoGaze kept {n_gaze}/{TOTAL} ({100*n_gaze/TOTAL:.0f}%)   "
            f"→ ∩sem50%: {n_i50} ({100*n_i50/max(n_gaze,1):.0f}% of AG)   "
            f"→ ∩sem10%: {n_i10} ({100*n_i10/max(n_gaze,1):.0f}% of AG)",
            fontsize=11,
        )
        fig.tight_layout()
        safe = name.replace(".mp4", "").replace("/", "_")
        out = os.path.join(args.output_dir, f"qual_{safe}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{vi+1}/{len(videos)}] {name[:35]:<35}  AG={n_gaze}  i50={n_i50}  i10={n_i10}  -> {out}")

        report_rows.append({
            "video": name, "frames_shown": fidx,
            "n_gaze": n_gaze, "n_i50": n_i50, "n_i10": n_i10,
            "pct_of_total_gaze": 100 * n_gaze / TOTAL,
            "pct_of_gaze_i50": 100 * n_i50 / max(n_gaze, 1),
            "pct_of_gaze_i10": 100 * n_i10 / max(n_gaze, 1),
            "output_png": out,
        })

    # Dump json
    with open(os.path.join(args.output_dir, "qual_frame_compare.json"), "w") as f:
        json.dump({"query": args.query, "rows": report_rows}, f, indent=2)
    print(f"[save] {args.output_dir}/qual_frame_compare.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", default="data")
    p.add_argument("--n_videos", type=int, default=4)
    p.add_argument("--query", default="people")
    p.add_argument("--gazing_ratio", type=float, default=0.75)
    p.add_argument("--task_loss_req", type=float, default=0.7)
    p.add_argument("--ckpt", default="results/bighead_warmrestart/best_bighead_student.pt")
    p.add_argument("--head_type", default="bighead")
    p.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/qual_frame_compare")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
