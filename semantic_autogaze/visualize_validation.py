"""
Validation visualization for Semantic AutoGaze.

Shows:
- The query patch outlined on the original frame
- Heatmap of predicted similarity scores across all patches
- Ground truth SigLIP similarity for comparison
- Text embedding query with heatmap

This is essentially a "validation loss visualization" where we can see
what the model predicts given a specific patch or text embedding as input.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import av
from PIL import Image
from einops import rearrange
from transformers import AutoModel, AutoProcessor

from autogaze.models.autogaze import AutoGaze

from semantic_autogaze.model import SemanticAutoGaze
from semantic_autogaze.data import SigLIPEmbedder, read_video_frames


def get_text_embedding_and_cleanup(text, model_name, device):
    """Get SigLIP text embedding, then free memory."""
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = F.normalize(text_features, dim=-1)
    result = text_features.squeeze(0).cpu()
    del model, processor
    torch.cuda.empty_cache()
    return result


def draw_patch_outline(frame, patch_idx, grid_size=14, color=(0, 255, 0), thickness=3):
    """Draw a colored outline around a specific patch on the frame."""
    H, W = frame.shape[:2]
    patch_h = H // grid_size
    patch_w = W // grid_size

    row = patch_idx // grid_size
    col = patch_idx % grid_size

    y1, y2 = row * patch_h, (row + 1) * patch_h
    x1, x2 = col * patch_w, (col + 1) * patch_w

    out = frame.copy()
    t = thickness
    out[y1:y1+t, x1:x2] = color  # top
    out[y2-t:y2, x1:x2] = color  # bottom
    out[y1:y2, x1:x1+t] = color  # left
    out[y1:y2, x2-t:x2] = color  # right
    return out


def similarity_to_heatmap(scores, grid_size=14, img_size=224):
    """Convert per-patch similarity scores to a colored heatmap image."""
    heatmap = scores.reshape(grid_size, grid_size).cpu().numpy()
    # Resize to image size
    heatmap_resized = np.array(
        Image.fromarray(((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) * 255).astype(np.uint8))
        .resize((img_size, img_size), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    cmap = plt.cm.jet
    heatmap_colored = (cmap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    return heatmap_colored


def blend_heatmap(frame, heatmap_colored, alpha=0.5):
    """Blend a heatmap onto a frame."""
    return (frame.astype(np.float32) * (1 - alpha) + heatmap_colored.astype(np.float32) * alpha).astype(np.uint8)


def visualize_patch_query(model, frames, siglip_embedder, query_frame_idx, query_patch_idx,
                          output_path, grid_size=14, device="cuda"):
    """
    Visualize what happens when we use a specific patch's embedding as the query.

    Shows: original frame with query outlined → predicted heatmap → GT heatmap
    """
    T = frames.shape[0]
    N = grid_size * grid_size

    # Get video tensor
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    video = video.unsqueeze(0).to(device)

    # Get SigLIP embeddings for ground truth
    patch_embeddings = siglip_embedder.get_patch_embeddings(frames)  # (T, N, D)
    query_emb = patch_embeddings[query_frame_idx, query_patch_idx]  # (D,)

    # Ground truth similarities
    all_patches = patch_embeddings.reshape(T * N, -1)
    gt_sims = torch.mv(all_patches, query_emb)  # (T*N,)

    # Model predictions
    with torch.inference_mode():
        outputs = model(video, query_emb.unsqueeze(0).to(device))
    pred_sims = outputs["similarity_scores"][0].cpu()  # (T*N,)

    # Create visualization for each frame
    H, W = frames.shape[1], frames.shape[2]
    n_cols = min(T, 8)
    sample_indices = np.linspace(0, T - 1, n_cols, dtype=int)

    fig, axes = plt.subplots(3, n_cols, figsize=(3.5 * n_cols, 10.5))
    if n_cols == 1:
        axes = axes[:, None]

    for col_idx, t in enumerate(sample_indices):
        frame = frames[t]
        gt_t = gt_sims[t * N:(t + 1) * N]
        pred_t = pred_sims[t * N:(t + 1) * N]

        # Row 0: Original frame with query patch outlined (if this is the query frame)
        display_frame = frame.copy()
        if t == query_frame_idx:
            display_frame = draw_patch_outline(display_frame, query_patch_idx, grid_size, color=(0, 255, 0), thickness=3)
        axes[0, col_idx].imshow(display_frame)
        title = f"Frame {t}"
        if t == query_frame_idx:
            title += f"\n(query patch {query_patch_idx})"
        axes[0, col_idx].set_title(title, fontsize=10)
        axes[0, col_idx].axis('off')

        # Row 1: Predicted heatmap
        pred_heatmap = similarity_to_heatmap(pred_t, grid_size, H)
        axes[1, col_idx].imshow(blend_heatmap(frame, pred_heatmap, alpha=0.55))
        axes[1, col_idx].set_title(f"Predicted\n[{pred_t.min():.2f}, {pred_t.max():.2f}]", fontsize=9)
        axes[1, col_idx].axis('off')

        # Row 2: Ground truth heatmap
        gt_heatmap = similarity_to_heatmap(gt_t, grid_size, H)
        axes[2, col_idx].imshow(blend_heatmap(frame, gt_heatmap, alpha=0.55))
        axes[2, col_idx].set_title(f"GT SigLIP\n[{gt_t.min():.2f}, {gt_t.max():.2f}]", fontsize=9)
        axes[2, col_idx].axis('off')

    row_labels = ["Original\n(query outlined)", "Predicted\nSimilarity", "Ground Truth\n(SigLIP)"]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=12, rotation=0, labelpad=80, ha='right', va='center')

    plt.suptitle(
        f"Patch Query Validation: frame={query_frame_idx}, patch={query_patch_idx}",
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_text_query(model, frames, text, text_embedding, output_path,
                         grid_size=14, device="cuda"):
    """
    Visualize predicted similarity heatmap for a text query.
    """
    T = frames.shape[0]
    N = grid_size * grid_size

    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    video = video.unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(video, text_embedding.unsqueeze(0).to(device))
    pred_sims = outputs["similarity_scores"][0].cpu()

    H, W = frames.shape[1], frames.shape[2]
    n_cols = min(T, 8)
    sample_indices = np.linspace(0, T - 1, n_cols, dtype=int)

    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))
    if n_cols == 1:
        axes = axes[:, None]

    for col_idx, t in enumerate(sample_indices):
        frame = frames[t]
        pred_t = pred_sims[t * N:(t + 1) * N]

        axes[0, col_idx].imshow(frame)
        axes[0, col_idx].set_title(f"Frame {t}", fontsize=10)
        axes[0, col_idx].axis('off')

        pred_heatmap = similarity_to_heatmap(pred_t, grid_size, H)
        axes[1, col_idx].imshow(blend_heatmap(frame, pred_heatmap, alpha=0.55))
        axes[1, col_idx].set_title(f"[{pred_t.min():.2f}, {pred_t.max():.2f}]", fontsize=9)
        axes[1, col_idx].axis('off')

    axes[0, 0].set_ylabel("Original", fontsize=12, rotation=0, labelpad=60, ha='right', va='center')
    axes[1, 0].set_ylabel("Predicted\nSimilarity", fontsize=12, rotation=0, labelpad=60, ha='right', va='center')

    plt.suptitle(f"Text Query: \"{text}\"", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def make_heatmap_video(frames, similarity_scores, grid_size, output_path, fps=8, alpha=0.55):
    """Save a video with heatmap overlay."""
    H, W = frames.shape[1], frames.shape[2]
    N = grid_size * grid_size
    T = frames.shape[0]

    output = av.open(output_path, mode='w')
    stream = output.add_stream('libx264', rate=fps)
    stream.height = H
    stream.width = W
    stream.pix_fmt = 'yuv420p'

    for t in range(T):
        sim_t = similarity_scores[t * N:(t + 1) * N]
        heatmap = similarity_to_heatmap(sim_t, grid_size, H)
        blended = blend_heatmap(frames[t], heatmap, alpha)
        av_frame = av.VideoFrame.from_ndarray(blended, format='rgb24')
        for packet in stream.encode(av_frame):
            output.mux(packet)
    for packet in stream.encode():
        output.mux(packet)
    output.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--siglip_model", type=str, default="google/siglip2-base-patch16-224")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Paths
    example_video = os.path.join(project_dir, "assets", "example_input.mp4")
    bair_video = os.path.join(project_dir, "results", "validation_data", "walk_around_bair.mp4")
    head_path = os.path.join(args.output_dir, "best_similarity_head.pt")

    # --- Get text embeddings before loading other models ---
    print("Getting text embeddings...")
    human_emb = get_text_embedding_and_cleanup("human", args.siglip_model, device)
    trash_emb = get_text_embedding_and_cleanup("trash cans", args.siglip_model, device)

    # --- Load SigLIP for patch embeddings ---
    print("Loading SigLIP embedder...")
    siglip_embedder = SigLIPEmbedder(model_name=args.siglip_model, device=device)

    # --- Load AutoGaze + SemanticAutoGaze ---
    print("Loading AutoGaze...")
    autogaze = AutoGaze.from_pretrained("nvidia/AutoGaze", use_flash_attn=False).to(device).eval()
    model = SemanticAutoGaze(autogaze, embedding_dim=siglip_embedder.embed_dim)
    model.to(device)

    if os.path.exists(head_path):
        model.similarity_head.load_state_dict(torch.load(head_path, map_location=device))
        print(f"Loaded trained head from {head_path}")
    else:
        print(f"WARNING: No trained head at {head_path}")
    model.eval()

    grid_size = model.patch_grid_size  # 14

    # ========== 1. Example video + "human" text query ==========
    print("\n--- Example video: text query 'human' ---")
    ex_frames = read_video_frames(example_video, args.num_frames, 224)
    visualize_text_query(
        model, ex_frames, "human", human_emb,
        os.path.join(args.output_dir, "example_text_human.png"),
        grid_size, device,
    )

    # ========== 2. Example video + patch query (pick diverse patches) ==========
    print("\n--- Example video: patch query validation ---")
    for patch_idx in [0, 50, 98, 150, 195]:
        visualize_patch_query(
            model, ex_frames, siglip_embedder,
            query_frame_idx=0, query_patch_idx=patch_idx,
            output_path=os.path.join(args.output_dir, f"example_patch_query_{patch_idx}.png"),
            grid_size=grid_size, device=device,
        )

    # ========== 3. BAIR video + "trash cans" text query ==========
    if os.path.exists(bair_video):
        print("\n--- BAIR video: text query 'trash cans' ---")
        bair_frames = read_video_frames(bair_video, args.num_frames, 224)

        visualize_text_query(
            model, bair_frames, "trash cans", trash_emb,
            os.path.join(args.output_dir, "bair_text_trash_cans.png"),
            grid_size, device,
        )

        # Also save heatmap video
        with torch.inference_mode():
            video_t = torch.from_numpy(bair_frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            out = model(video_t.unsqueeze(0).to(device), trash_emb.unsqueeze(0).to(device))
        make_heatmap_video(
            bair_frames, out["similarity_scores"][0].cpu(),
            grid_size, os.path.join(args.output_dir, "bair_trash_cans_heatmap.mp4"),
        )

        # Patch query on BAIR video
        print("\n--- BAIR video: patch query validation ---")
        for patch_idx in [0, 50, 98, 150, 195]:
            visualize_patch_query(
                model, bair_frames, siglip_embedder,
                query_frame_idx=0, query_patch_idx=patch_idx,
                output_path=os.path.join(args.output_dir, f"bair_patch_query_{patch_idx}.png"),
                grid_size=grid_size, device=device,
            )
    else:
        print(f"BAIR video not found at {bair_video}, skipping")

    print("\nAll validation visualizations complete!")


if __name__ == "__main__":
    main()
