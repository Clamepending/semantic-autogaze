"""
Visualization script for Semantic AutoGaze.

Generates:
1. AutoGaze output on video (0.7 threshold for redundancy removal)
2. Similarity heatmap overlay for a text task embedding
3. Combined video with both semantic and redundancy filters
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import av
from PIL import Image
from einops import rearrange
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from autogaze.models.autogaze import AutoGaze
from autogaze.models.autogaze.processing_autogaze import AutoGazeImageProcessor
from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch

from semantic_autogaze.model import SemanticAutoGaze
from semantic_autogaze.data import SigLIPEmbedder, read_video_frames


def get_text_embedding(text, model_name="google/siglip2-base-patch16-224", device="cuda"):
    """Get SigLIP text embedding for a task description."""
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = F.normalize(text_features, dim=-1)

    result = text_features.squeeze(0).cpu()  # (embed_dim,)

    # Free memory
    del model
    torch.cuda.empty_cache()

    return result


def create_heatmap_overlay(frame, heatmap_values, patch_grid_size=14, alpha=0.5):
    """
    Overlay a heatmap on a frame.

    Args:
        frame: (H, W, 3) uint8
        heatmap_values: (N_patches,) similarity values
        patch_grid_size: sqrt of N_patches
        alpha: blending factor
    Returns:
        overlay: (H, W, 3) uint8
    """
    H, W = frame.shape[:2]
    heatmap = heatmap_values.reshape(patch_grid_size, patch_grid_size)
    heatmap = heatmap.cpu().numpy()

    # Normalize to [0, 1]
    vmin, vmax = heatmap.min(), heatmap.max()
    if vmax - vmin > 1e-6:
        heatmap_norm = (heatmap - vmin) / (vmax - vmin)
    else:
        heatmap_norm = np.zeros_like(heatmap)

    # Resize heatmap to frame size
    heatmap_resized = np.array(
        Image.fromarray((heatmap_norm * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    # Apply colormap
    cmap = plt.cm.jet
    heatmap_colored = (cmap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)

    # Blend
    overlay = (frame.astype(np.float32) * (1 - alpha) + heatmap_colored.astype(np.float32) * alpha).astype(np.uint8)
    return overlay


def create_gaze_mask_overlay(frame, gaze_mask, patch_grid_size=14, alpha=0.4):
    """
    Overlay AutoGaze mask on a frame.
    Gazed patches shown normally, non-gazed patches darkened.
    """
    H, W = frame.shape[:2]
    mask = gaze_mask.reshape(patch_grid_size, patch_grid_size).cpu().numpy()

    # Resize mask to frame size
    mask_resized = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
    ).astype(np.float32) / 255.0

    # Darken non-gazed patches
    overlay = frame.copy().astype(np.float32)
    overlay[mask_resized < 0.5] *= (1 - alpha)
    return overlay.astype(np.uint8)


def create_combined_overlay(frame, gaze_mask, similarity_scores, patch_grid_size=14,
                           similarity_threshold=0.3, alpha_heatmap=0.5, alpha_mask=0.6):
    """
    Combined overlay: AutoGaze mask + similarity heatmap.
    Patches that are both redundant AND semantically irrelevant are darkened most.
    """
    H, W = frame.shape[:2]

    mask = gaze_mask.reshape(patch_grid_size, patch_grid_size).cpu().numpy()
    sim = similarity_scores.reshape(patch_grid_size, patch_grid_size).cpu().numpy()

    # Normalize similarity
    vmin, vmax = sim.min(), sim.max()
    if vmax - vmin > 1e-6:
        sim_norm = (sim - vmin) / (vmax - vmin)
    else:
        sim_norm = np.zeros_like(sim)

    # Combined filter: keep patches that are either gazed OR semantically relevant
    semantic_mask = sim_norm > similarity_threshold
    combined_mask = np.logical_or(mask > 0.5, semantic_mask).astype(np.float32)

    # Resize
    combined_resized = np.array(
        Image.fromarray((combined_mask * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
    ).astype(np.float32) / 255.0

    sim_resized = np.array(
        Image.fromarray((sim_norm * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    # Create overlay
    overlay = frame.copy().astype(np.float32)
    # Darken filtered-out patches
    overlay[combined_resized < 0.5] *= 0.2
    # Add subtle heatmap tint on kept patches
    cmap = plt.cm.jet
    heatmap_colored = (cmap(sim_resized)[:, :, :3] * 255).astype(np.float32)
    kept = combined_resized > 0.5
    overlay[kept] = overlay[kept] * 0.7 + heatmap_colored[kept] * 0.3

    return overlay.astype(np.uint8)


def visualize(args):
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print("Loading AutoGaze...")
    autogaze_transform = AutoGazeImageProcessor.from_pretrained("nvidia/AutoGaze")
    autogaze = AutoGaze.from_pretrained("nvidia/AutoGaze", use_flash_attn=False).to(device).eval()

    # Get text embedding first, then free the full SigLIP model
    print(f"Getting text embedding for: '{args.task_text}'")
    task_embedding = get_text_embedding(args.task_text, model_name=args.siglip_model, device=device)
    task_embedding = task_embedding.unsqueeze(0).to(device)  # (1, embed_dim)

    # Determine embedding dim from the text embedding
    embedding_dim = task_embedding.shape[-1]

    print("Loading SemanticAutoGaze...")
    model = SemanticAutoGaze(autogaze, embedding_dim=embedding_dim)
    model.to(device)

    # Load trained similarity head
    head_path = os.path.join(args.output_dir, "best_similarity_head.pt")
    if os.path.exists(head_path):
        model.similarity_head.load_state_dict(torch.load(head_path, map_location=device))
        print(f"Loaded similarity head from {head_path}")
    else:
        print(f"WARNING: No trained similarity head found at {head_path}, using random weights")

    model.eval()

    # Load video
    print(f"Loading video: {args.video_path}")
    raw_frames = read_video_frames(args.video_path, num_frames=args.num_frames, size=224)  # (T, H, W, 3)

    # Prepare video tensor for AutoGaze
    video_tensor = torch.from_numpy(raw_frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    video_tensor = video_tensor.unsqueeze(0).to(device)  # (1, T, C, H, W)

    # Run inference
    print("Running inference...")
    with torch.inference_mode():
        outputs = model(
            video_tensor, task_embedding,
            return_gaze=True,
            gazing_ratio=0.75,
            task_loss_requirement=args.gaze_threshold,
        )

    similarity_scores = outputs["similarity_scores"]  # (1, T*N)
    gaze_outputs = outputs["gaze_outputs"]

    T = raw_frames.shape[0]
    N_per_frame = model.num_patches_per_frame  # 196 (encoder grid patches)
    patch_grid = model.patch_grid_size  # 14

    # Get per-frame gaze masks across all scales
    # gaze_outputs["gazing_mask"] is a list: scale_sizes = [4, 16, 49, 196]
    gaze_mask_scales = gaze_outputs["gazing_mask"]  # list of (1, T, N_scale)
    scales = gaze_outputs["scales"]  # [32, 64, 112, 224]

    # Reshape similarity scores to per-frame
    sim_per_frame = similarity_scores[0].reshape(T, N_per_frame)

    def composite_gaze_masks(gaze_mask_scales, scales, t, img_size=224):
        """Composite multi-scale gaze masks into a single 14x14 grid."""
        composite = torch.zeros(img_size, img_size)
        for scale_idx, scale in enumerate(scales):
            mask = gaze_mask_scales[scale_idx][0, t]  # (N_scale,)
            grid = int(mask.shape[0] ** 0.5)
            mask_2d = mask.reshape(grid, grid).cpu().float()
            # Upscale to img_size
            mask_up = torch.nn.functional.interpolate(
                mask_2d.unsqueeze(0).unsqueeze(0),
                size=(img_size, img_size),
                mode='nearest',
            ).squeeze()
            composite = torch.max(composite, mask_up)
        # Downscale to 14x14 grid for consistency
        composite_grid = torch.nn.functional.adaptive_max_pool2d(
            composite.unsqueeze(0).unsqueeze(0), (patch_grid, patch_grid)
        ).squeeze()
        return composite_grid  # (14, 14)

    # Generate visualization frames
    print("Generating visualizations...")
    fig_frames_gaze = []
    fig_frames_sim = []
    fig_frames_combined = []
    gaze_masks_composite = []

    for t in range(T):
        frame = raw_frames[t]
        gaze_mask_t = composite_gaze_masks(gaze_mask_scales, scales, t).flatten()  # (196,)
        gaze_masks_composite.append(gaze_mask_t)
        sim_t = sim_per_frame[t]  # (N_per_frame,)

        # 1. AutoGaze mask overlay
        gaze_overlay = create_gaze_mask_overlay(frame, gaze_mask_t, patch_grid)
        fig_frames_gaze.append(gaze_overlay)

        # 2. Similarity heatmap overlay
        sim_overlay = create_heatmap_overlay(frame, sim_t, patch_grid)
        fig_frames_sim.append(sim_overlay)

        # 3. Combined filter
        combined_overlay = create_combined_overlay(
            frame, gaze_mask_t, sim_t, patch_grid,
            similarity_threshold=args.similarity_threshold,
        )
        fig_frames_combined.append(combined_overlay)

    # Save video outputs
    def save_video(frames, path, fps=8):
        output = av.open(path, mode='w')
        stream = output.add_stream('libx264', rate=fps)
        stream.height = frames[0].shape[0]
        stream.width = frames[0].shape[1]
        stream.pix_fmt = 'yuv420p'
        for f in frames:
            av_frame = av.VideoFrame.from_ndarray(f, format='rgb24')
            for packet in stream.encode(av_frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
        output.close()

    save_video(fig_frames_gaze, os.path.join(args.output_dir, "autogaze_output.mp4"))
    save_video(fig_frames_sim, os.path.join(args.output_dir, "similarity_heatmap.mp4"))
    save_video(fig_frames_combined, os.path.join(args.output_dir, "combined_filter.mp4"))

    # Generate comparison figure (side-by-side frames)
    print("Generating comparison figures...")
    sample_frames = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    sample_frames = [min(t, T - 1) for t in sample_frames]

    fig, axes = plt.subplots(4, len(sample_frames), figsize=(4 * len(sample_frames), 16))

    for col, t in enumerate(sample_frames):
        axes[0, col].imshow(raw_frames[t])
        axes[0, col].set_title(f"Frame {t}", fontsize=12)
        axes[0, col].axis('off')

        axes[1, col].imshow(fig_frames_gaze[t])
        axes[1, col].set_title(f"AutoGaze (τ={args.gaze_threshold})", fontsize=10)
        axes[1, col].axis('off')

        axes[2, col].imshow(fig_frames_sim[t])
        axes[2, col].set_title(f"Similarity: '{args.task_text[:20]}...'", fontsize=10)
        axes[2, col].axis('off')

        axes[3, col].imshow(fig_frames_combined[t])
        axes[3, col].set_title("Combined Filter", fontsize=10)
        axes[3, col].axis('off')

    row_labels = ["Original", "AutoGaze\n(Redundancy)", "Semantic\nSimilarity", "Combined\n(Both Filters)"]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=14, rotation=0, labelpad=80, ha='right', va='center')

    plt.suptitle(
        f"Semantic AutoGaze: Redundancy + Semantic Filtering\nTask: \"{args.task_text}\"",
        fontsize=16, y=1.02,
    )
    plt.tight_layout()
    comparison_path = os.path.join(args.output_dir, "comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Stats figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Stack composite masks for stats
    gaze_masks_stacked = torch.stack(gaze_masks_composite)  # (T, 196)

    # Per-frame gaze counts
    gaze_counts = gaze_masks_stacked.sum(dim=-1).cpu().numpy()
    axes[0].bar(range(T), gaze_counts, color='steelblue')
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("# Gazed Patches")
    axes[0].set_title("AutoGaze: Patches Kept per Frame")
    axes[0].axhline(y=N_per_frame, color='r', linestyle='--', label=f"Total ({N_per_frame})")
    axes[0].legend()

    # Per-frame mean similarity
    mean_sim = sim_per_frame.mean(dim=-1).cpu().numpy()
    axes[1].plot(range(T), mean_sim, 'o-', color='darkorange')
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Mean Similarity")
    axes[1].set_title(f"Semantic Similarity to \"{args.task_text[:30]}...\"")

    # Combined filter efficiency
    # Summary bar chart
    total_patches = T * N_per_frame
    total_gaze_kept = gaze_masks_stacked.sum().item()
    sim_norm_all = (sim_per_frame - sim_per_frame.min()) / (sim_per_frame.max() - sim_per_frame.min() + 1e-6)
    total_semantic_kept = (sim_norm_all > args.similarity_threshold).sum().item()
    gaze_flat = gaze_masks_stacked.reshape(-1).bool().cpu()
    sem_flat = (sim_norm_all.reshape(-1) > args.similarity_threshold).cpu()
    total_combined = (gaze_flat & sem_flat).sum().item()

    categories = ['All\nPatches', 'AutoGaze\nOnly', 'Semantic\nOnly', 'Both\nFilters']
    counts = [total_patches, total_gaze_kept, total_semantic_kept, total_combined]
    colors = ['gray', 'steelblue', 'darkorange', 'green']
    bars = axes[2].bar(categories, counts, color=colors)
    for bar, count in zip(bars, counts):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f'{count}\n({100*count/total_patches:.0f}%)',
                    ha='center', va='bottom', fontsize=10)
    axes[2].set_ylabel("# Patches")
    axes[2].set_title("Filter Efficiency Comparison")

    plt.tight_layout()
    stats_path = os.path.join(args.output_dir, "filter_stats.png")
    plt.savefig(stats_path, dpi=150)
    plt.close()

    # Compute and print statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total patches: {total_patches}")
    print(f"AutoGaze kept: {total_gaze_kept} ({100*total_gaze_kept/total_patches:.1f}%)")
    print(f"Semantic filter kept: {total_semantic_kept} ({100*total_semantic_kept/total_patches:.1f}%)")
    print(f"Combined (intersection): {total_combined} ({100*total_combined/total_patches:.1f}%)")
    print(f"Combined reduction: {100*(1-total_combined/total_patches):.1f}%")
    print("=" * 60)
    print(f"\nOutputs saved to {args.output_dir}/:")
    print(f"  - autogaze_output.mp4")
    print(f"  - similarity_heatmap.mp4")
    print(f"  - combined_filter.mp4")
    print(f"  - comparison.png")
    print(f"  - filter_stats.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize Semantic AutoGaze results")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--task_text", type=str, default="a traffic light changing colors")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--siglip_model", type=str, default="google/siglip2-base-patch16-224")
    parser.add_argument("--gaze_threshold", type=float, default=0.7)
    parser.add_argument("--similarity_threshold", type=float, default=0.3)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.parse_args(namespace=(args := argparse.Namespace()))
    visualize(args)


if __name__ == "__main__":
    main()
