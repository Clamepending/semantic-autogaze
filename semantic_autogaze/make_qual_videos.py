"""
Generate qualitative heatmap images from validation set.

For each video:
- Pick a few patches from the first frame as queries
- Render image grid: rows = [original w/ query outlined | predicted heatmap | GT heatmap]
  columns = sampled frames across the video
- Heatmaps use a GLOBAL color scale (not per-frame normalized) so frames
  where the query has no match show uniformly cool colors.
"""

import os
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm

from autogaze.models.autogaze import AutoGaze
from semantic_autogaze.model import SemanticAutoGaze
from semantic_autogaze.data import SigLIPEmbedder, read_video_frames


def draw_patch_rect(frame, patch_idx, grid_size=14, color=(0, 255, 0), thickness=3):
    """Draw rectangle around a patch."""
    H, W = frame.shape[:2]
    ph, pw = H // grid_size, W // grid_size
    r, c = patch_idx // grid_size, patch_idx % grid_size
    y1, y2, x1, x2 = r * ph, (r + 1) * ph, c * pw, (c + 1) * pw
    out = frame.copy()
    t = thickness
    out[y1:y1+t, x1:x2] = color
    out[y2-t:y2, x1:x2] = color
    out[y1:y2, x1:x1+t] = color
    out[y1:y2, x2-t:x2] = color
    return out


def scores_to_heatmap(scores, grid_size, H, W, vmin, vmax):
    """
    Overlay heatmap on a frame-sized canvas using a GLOBAL color scale.

    vmin/vmax are the global min/max across all frames for this query,
    so the colormap is consistent and frames with low similarity
    appear uniformly cool.
    """
    hmap = scores.reshape(grid_size, grid_size).cpu().numpy()
    # Clip to global range and normalize
    rng = vmax - vmin
    if rng > 1e-8:
        hmap_norm = np.clip((hmap - vmin) / rng, 0, 1)
    else:
        hmap_norm = np.zeros_like(hmap)
    hmap_resized = np.array(
        Image.fromarray((hmap_norm * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    colored = (plt.cm.jet(hmap_resized)[:, :, :3] * 255).astype(np.uint8)
    return colored


def blend(frame, heatmap, alpha=0.55):
    return (frame.astype(np.float32) * (1 - alpha) + heatmap.astype(np.float32) * alpha).astype(np.uint8)


def process_video(video_path, model, siglip_embedder, output_dir, grid_size=14,
                  num_frames=16, device="cuda", n_display_frames=8):
    """Generate qualitative heatmap image for one input video."""
    basename = os.path.splitext(os.path.basename(video_path))[0]
    frames = read_video_frames(video_path, num_frames, 224)
    if frames is None:
        return

    T = frames.shape[0]
    N = grid_size * grid_size
    H, W = frames.shape[1], frames.shape[2]

    # Get SigLIP embeddings
    patch_emb = siglip_embedder.get_patch_embeddings(frames)  # (T, N, D)

    # Get model predictions for video
    video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    video_tensor = video_tensor.unsqueeze(0).to(device)

    # Find interesting patches: corners + center + high-variance
    first_emb = patch_emb[0]  # (N, D)
    mean_emb = first_emb.mean(dim=0)
    dists = torch.norm(first_emb - mean_emb, dim=-1)
    top_patches = dists.topk(3).indices.tolist()
    query_patches = list(set([0, N // 2, N - 1] + top_patches))[:5]

    # Sample frame indices to display
    n_cols = min(T, n_display_frames)
    sample_indices = np.linspace(0, T - 1, n_cols, dtype=int)

    for qp in query_patches:
        query_emb = patch_emb[0, qp]  # (D,)

        # GT similarities
        all_patches = patch_emb.reshape(T * N, -1)
        gt_sims = torch.mv(all_patches, query_emb)

        # Predicted similarities
        with torch.inference_mode():
            out = model(video_tensor, query_emb.unsqueeze(0).to(device))
        pred_sims = out["similarity_scores"][0].cpu()

        # Compute GLOBAL min/max across all frames for consistent color scale
        pred_vmin, pred_vmax = pred_sims.min().item(), pred_sims.max().item()
        gt_vmin, gt_vmax = gt_sims.min().item(), gt_sims.max().item()

        # Build figure: 3 rows x n_cols columns
        fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 9))
        if n_cols == 1:
            axes = axes[:, None]

        for col_idx, t in enumerate(sample_indices):
            frame = frames[t]

            # Row 0: Original with query patch outlined on frame 0
            display_frame = frame.copy()
            if t == 0:
                display_frame = draw_patch_rect(display_frame, qp, grid_size,
                                                color=(0, 255, 0), thickness=3)
            axes[0, col_idx].imshow(display_frame)
            title = f"t={t}"
            if t == 0:
                title += f" (query)"
            axes[0, col_idx].set_title(title, fontsize=9)
            axes[0, col_idx].axis('off')

            # Row 1: Predicted heatmap (global scale)
            pred_t = pred_sims[t * N:(t + 1) * N]
            pred_heat = scores_to_heatmap(pred_t, grid_size, H, W, pred_vmin, pred_vmax)
            axes[1, col_idx].imshow(blend(frame, pred_heat))
            axes[1, col_idx].set_title(f"[{pred_t.min():.2f}, {pred_t.max():.2f}]", fontsize=8)
            axes[1, col_idx].axis('off')

            # Row 2: GT heatmap (global scale)
            gt_t = gt_sims[t * N:(t + 1) * N]
            gt_heat = scores_to_heatmap(gt_t, grid_size, H, W, gt_vmin, gt_vmax)
            axes[2, col_idx].imshow(blend(frame, gt_heat))
            axes[2, col_idx].set_title(f"[{gt_t.min():.2f}, {gt_t.max():.2f}]", fontsize=8)
            axes[2, col_idx].axis('off')

        # Row labels
        row_labels = ["Original", "Predicted", "GT (SigLIP)"]
        for row, label in enumerate(row_labels):
            axes[row, 0].set_ylabel(label, fontsize=11, rotation=0,
                                    labelpad=65, ha='right', va='center')

        # Colorbars for global scale
        fig.subplots_adjust(bottom=0.08)
        # Predicted colorbar
        sm_pred = plt.cm.ScalarMappable(cmap='jet',
                                        norm=plt.Normalize(vmin=pred_vmin, vmax=pred_vmax))
        cb_pred_ax = fig.add_axes([0.15, 0.03, 0.3, 0.015])
        fig.colorbar(sm_pred, cax=cb_pred_ax, orientation='horizontal', label='Predicted sim.')

        # GT colorbar
        sm_gt = plt.cm.ScalarMappable(cmap='jet',
                                      norm=plt.Normalize(vmin=gt_vmin, vmax=gt_vmax))
        cb_gt_ax = fig.add_axes([0.55, 0.03, 0.3, 0.015])
        fig.colorbar(sm_gt, cax=cb_gt_ax, orientation='horizontal', label='GT sim.')

        qr, qc = qp // grid_size, qp % grid_size
        plt.suptitle(
            f"{basename}  |  query patch {qp} (row {qr}, col {qc})\n"
            f"Pred range: [{pred_vmin:.3f}, {pred_vmax:.3f}]   "
            f"GT range: [{gt_vmin:.3f}, {gt_vmax:.3f}]",
            fontsize=12, y=1.02,
        )
        plt.tight_layout(rect=[0.08, 0.06, 1, 0.97])

        out_path = os.path.join(output_dir, f"{basename}_patch{qp}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="data")
    parser.add_argument("--output_dir", default="results/qual_images")
    parser.add_argument("--num_videos", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--n_display_frames", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Find videos
    import glob
    all_vids = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    if not all_vids:
        print("No videos found")
        return

    # Pick evenly spaced videos for diversity
    step = max(1, len(all_vids) // args.num_videos)
    selected = [all_vids[i] for i in range(0, len(all_vids), step)][:args.num_videos]
    print(f"Selected {len(selected)} videos from {len(all_vids)} total")

    # Load models
    print("Loading SigLIP...")
    siglip = SigLIPEmbedder(device=device)

    print("Loading AutoGaze...")
    autogaze = AutoGaze.from_pretrained("nvidia/AutoGaze", use_flash_attn=False).to(device).eval()
    model = SemanticAutoGaze(autogaze, embedding_dim=siglip.embed_dim).to(device)

    head_path = os.path.join(os.path.dirname(args.output_dir), "best_similarity_head.pt")
    if os.path.exists(head_path):
        model.similarity_head.load_state_dict(torch.load(head_path, map_location=device))
        print(f"Loaded head from {head_path}")
    model.eval()

    grid_size = model.patch_grid_size

    # Process each video
    for vp in selected:
        print(f"\nProcessing: {os.path.basename(vp)}")
        process_video(vp, model, siglip, args.output_dir, grid_size,
                      args.num_frames, device, args.n_display_frames)

    print(f"\nAll images saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
