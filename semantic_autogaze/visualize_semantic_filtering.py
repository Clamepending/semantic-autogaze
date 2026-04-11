"""
Visualize semantic filtering on real video frames.

Shows side-by-side:
  1. Original frame with all patches
  2. Semantic head prediction heatmap
  3. CLIPSeg ground truth heatmap
  4. Selected patches (top-k by semantic score) overlaid on frame

Usage:
  python3 -m semantic_autogaze.visualize_semantic_filtering \
    --hidden_dir results/distill/hidden_cache \
    --clipseg_dir results/distill/clipseg_cache \
    --ckpt results/distill_bighead/best_bighead_student.pt \
    --head_type bighead \
    --video_dir data \
    --output_dir results/qual_semantic \
    --n_samples 10 \
    --keep_ratio 0.2
"""

import os
import glob
import hashlib
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from PIL import Image
import av

from semantic_autogaze.eval_filtering import load_head
from semantic_autogaze.train_bighead import BigSimilarityHead
from semantic_autogaze.model import SimilarityHead


def read_middle_frame(video_path, frame_idx=0):
    """Read a single frame from a video."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    frames = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            frames.append(frame)
            if len(frames) > frame_idx:
                img = frames[frame_idx].to_image()
                container.close()
                return np.array(img)
    container.close()
    if frames:
        return np.array(frames[-1].to_image())
    return None


def create_heatmap_overlay(frame, heatmap, grid_size=14, alpha=0.5, cmap="jet"):
    """Overlay a heatmap on a video frame."""
    H, W = frame.shape[:2]

    # Upsample heatmap to frame size
    heatmap_2d = heatmap.reshape(grid_size, grid_size)
    heatmap_resized = np.array(Image.fromarray(heatmap_2d.astype(np.float32)).resize(
        (W, H), Image.BILINEAR))

    # Normalize
    vmin, vmax = heatmap_resized.min(), heatmap_resized.max()
    if vmax > vmin:
        heatmap_norm = (heatmap_resized - vmin) / (vmax - vmin)
    else:
        heatmap_norm = np.zeros_like(heatmap_resized)

    # Apply colormap
    cm = plt.get_cmap(cmap)
    colored = cm(heatmap_norm)[:, :, :3]  # (H, W, 3)

    # Blend
    overlay = (frame / 255.0) * (1 - alpha) + colored * alpha
    return (overlay * 255).clip(0, 255).astype(np.uint8)


def create_patch_selection_overlay(frame, selected_mask, grid_size=14):
    """Show which patches are selected (kept) vs. dropped."""
    H, W = frame.shape[:2]
    patch_h = H / grid_size
    patch_w = W / grid_size

    overlay = frame.copy().astype(np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            y0 = int(i * patch_h)
            y1 = int((i + 1) * patch_h)
            x0 = int(j * patch_w)
            x1 = int((j + 1) * patch_w)

            if not selected_mask[i * grid_size + j]:
                # Dropped patch: darken significantly
                overlay[y0:y1, x0:x1] *= 0.15
            else:
                # Kept patch: slight green tint border
                border = 2
                overlay[y0:y0 + border, x0:x1, 1] = 255
                overlay[y1 - border:y1, x0:x1, 1] = 255
                overlay[y0:y1, x0:x0 + border, 1] = 255
                overlay[y0:y1, x1 - border:x1, 1] = 255

    return overlay.clip(0, 255).astype(np.uint8)


def visualize_sample(frame, pred_scores, gt_scores, query_text,
                     keep_ratio, grid_size, output_path):
    """Generate a 4-panel visualization for one sample."""
    N = grid_size * grid_size
    k = max(1, int(keep_ratio * N))

    # Get top-k mask
    topk_idx = np.argsort(pred_scores)[-k:]
    selected = np.zeros(N, dtype=bool)
    selected[topk_idx] = True

    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.05)

    # Panel 1: Original
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(frame)
    ax1.set_title("Original Frame", fontsize=11)
    ax1.axis("off")

    # Panel 2: Prediction heatmap
    ax2 = fig.add_subplot(gs[1])
    pred_overlay = create_heatmap_overlay(frame, pred_scores, grid_size)
    ax2.imshow(pred_overlay)
    ax2.set_title(f"Predicted: \"{query_text}\"", fontsize=11)
    ax2.axis("off")

    # Panel 3: GT heatmap
    ax3 = fig.add_subplot(gs[2])
    gt_overlay = create_heatmap_overlay(frame, gt_scores, grid_size)
    ax3.imshow(gt_overlay)
    ax3.set_title("Ground Truth (CLIPSeg)", fontsize=11)
    ax3.axis("off")

    # Panel 4: Selected patches
    ax4 = fig.add_subplot(gs[3])
    selection_overlay = create_patch_selection_overlay(frame, selected, grid_size)
    ax4.imshow(selection_overlay)
    kept_pct = keep_ratio * 100
    recall = (selected & (gt_scores > 0.5)).sum() / max((gt_scores > 0.5).sum(), 1)
    ax4.set_title(f"Keep {kept_pct:.0f}% → Recall {recall:.1%}", fontsize=11)
    ax4.axis("off")

    fig.suptitle(f"Query: \"{query_text}\" | {k}/{N} patches kept",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(args):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load head
    head = load_head(args, device)

    # Load some CLIPSeg cache files
    clipseg_files = sorted(glob.glob(os.path.join(args.clipseg_dir, "*_clipseg_clip*.pt")))
    random.shuffle(clipseg_files)

    count = 0
    for cf in clipseg_files:
        if count >= args.n_samples:
            break

        data = torch.load(cf, map_location="cpu", weights_only=False)
        vp = data["video_path"]
        key = hashlib.md5(vp.encode()).hexdigest()
        hidden_path = os.path.join(args.hidden_dir, f"{key}_hidden.pt")

        if not os.path.exists(hidden_path):
            continue

        hidden = torch.load(hidden_path, map_location="cpu", weights_only=True)

        # Try to read a frame from the video
        if os.path.exists(vp):
            frame = read_middle_frame(vp, frame_idx=8)  # middle-ish frame
        else:
            # Try data/ directory
            alt_path = os.path.join(args.video_dir, os.path.basename(vp))
            if os.path.exists(alt_path):
                frame = read_middle_frame(alt_path, frame_idx=8)
            else:
                continue

        if frame is None:
            continue

        # Pick a query with interesting GT signal
        best_query = None
        best_signal = 0
        for q in data["queries"]:
            gt = torch.sigmoid(q["target_scores"]).numpy()
            # Pick queries with moderate positive area (not too sparse, not too dense)
            pos_frac = (gt > 0.5).mean()
            if 0.02 < pos_frac < 0.5:
                signal = pos_frac * gt.max()
                if signal > best_signal:
                    best_signal = signal
                    best_query = q

        if best_query is None:
            # Fall back to query with highest max GT
            for q in data["queries"]:
                gt = torch.sigmoid(q["target_scores"]).numpy()
                if gt.max() > best_signal:
                    best_signal = gt.max()
                    best_query = q

        if best_query is None:
            continue

        # Run inference
        with torch.no_grad():
            h = hidden.unsqueeze(0).to(device)
            q_emb = best_query["text_embedding"].unsqueeze(0).to(device)
            pred_logits = head(h, q_emb)
            pred_probs = torch.sigmoid(pred_logits).cpu().numpy().squeeze()

        gt_probs = torch.sigmoid(best_query["target_scores"]).numpy()
        query_text = best_query.get("text", best_query.get("query_text", "unknown"))

        # Use frame 8's patches (14x14 = 196 patches, frame 8 = patches 8*196 : 9*196)
        frame_idx = min(8, pred_probs.shape[0] // 196 - 1)
        start = frame_idx * 196
        end = start + 196
        pred_frame = pred_probs[start:end]
        gt_frame = gt_probs[start:end]

        output_path = os.path.join(args.output_dir, f"sample_{count:03d}_{query_text[:20].replace(' ', '_')}.png")
        visualize_sample(frame, pred_frame, gt_frame, query_text,
                         args.keep_ratio, 14, output_path)
        print(f"  [{count+1}/{args.n_samples}] {query_text}: {output_path}")
        count += 1

    print(f"\nGenerated {count} visualizations in {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--head_type", choices=["small", "bighead"], default="bighead")
    parser.add_argument("--video_dir", default="data")
    parser.add_argument("--output_dir", default="results/qual_semantic")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--keep_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--expanded_dim", type=int, default=384)
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--n_attn_layers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args)
