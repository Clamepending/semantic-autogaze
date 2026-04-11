"""
Multi-query comparison: shows how different text queries select different patches.

For a single video, runs semantic filtering with multiple queries and
visualizes the spatial distribution of selected patches for each query.

Usage:
  python3 -m semantic_autogaze.multi_query_comparison \
    --video assets/example_input.mp4 \
    --ckpt results/distill_bighead/best_bighead_student.pt
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open_clip
import av
from PIL import Image
from einops import rearrange

from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper


def get_clip_text_embeddings(texts, device="cuda"):
    """Get CLIP embeddings for multiple queries at once."""
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model = model.to(device).eval()

    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)

    del model
    torch.cuda.empty_cache()
    return text_features  # (num_queries, 512)


def visualize_multi_query(raw_frames, all_scores, queries, keep_ratio,
                          output_path, grid_size=14):
    """Create a visualization showing different patch selections per query."""
    N = grid_size * grid_size
    k = max(1, int(keep_ratio * N))

    # Select a representative frame (middle)
    T = len(raw_frames)
    frame_idx = T // 2
    frame = raw_frames[frame_idx]
    H, W = frame.shape[:2]

    num_queries = len(queries)
    fig, axes = plt.subplots(2, num_queries, figsize=(4.5 * num_queries, 8))
    if num_queries == 1:
        axes = axes.reshape(2, 1)

    for col, (query, scores_np) in enumerate(zip(queries, all_scores)):
        frame_scores = scores_np[frame_idx * N:(frame_idx + 1) * N]

        # Top row: heatmap
        heatmap = frame_scores.reshape(grid_size, grid_size)
        heatmap_resized = np.array(Image.fromarray(heatmap.astype(np.float32)).resize(
            (W, H), Image.BILINEAR))
        vmax = max(heatmap_resized.max(), 0.01)
        heatmap_norm = heatmap_resized / vmax

        cm = plt.get_cmap("jet")
        colored = cm(heatmap_norm)[:, :, :3]
        overlay = (frame / 255.0) * 0.5 + colored * 0.5
        axes[0, col].imshow(overlay.clip(0, 1))
        axes[0, col].set_title(f'"{query}"', fontsize=11, fontweight="bold")
        axes[0, col].axis("off")

        # Bottom row: selected patches
        topk_idx = np.argsort(frame_scores)[-k:]
        kept = np.zeros(N, dtype=bool)
        kept[topk_idx] = True

        patch_h = H / grid_size
        patch_w = W / grid_size
        filtered = frame.copy().astype(np.float32)
        for i in range(grid_size):
            for j in range(grid_size):
                y0, y1 = int(i * patch_h), int((i + 1) * patch_h)
                x0, x1 = int(j * patch_w), int((j + 1) * patch_w)
                if not kept[i * grid_size + j]:
                    filtered[y0:y1, x0:x1] *= 0.15

        axes[1, col].imshow(filtered.astype(np.uint8))
        axes[1, col].set_title(f"Top {k}/{N} patches", fontsize=10)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Semantic\nHeatmap", fontsize=11, rotation=0, labelpad=60)
    axes[1, 0].set_ylabel(f"Selected\nPatches\n({keep_ratio*100:.0f}%)",
                          fontsize=11, rotation=0, labelpad=60)

    fig.suptitle(f"Query-Conditioned Patch Selection (Frame {frame_idx})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main(args):
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    os.makedirs(args.output_dir, exist_ok=True)

    queries = args.queries or [
        "person",
        "background",
        "sky or ceiling",
        "text or writing",
    ]

    print(f"Video: {args.video}")
    print(f"Queries: {queries}")
    print(f"Keep ratio: {args.keep_ratio}")

    # Load video
    container = av.open(args.video)
    indices = list(range(16))
    raw_video = read_video_pyav(container=container, indices=indices)
    container.close()

    # Keep raw frames for visualization
    raw_frames = [np.array(Image.fromarray(f).resize((224, 224))) for f in raw_video]

    # Preprocess
    autogaze_transform = AutoGazeImageProcessor.from_pretrained(args.autogaze_model)
    video_autogaze = transform_video_for_pytorch(raw_video, autogaze_transform)[None].to(device)

    # Get embeddings for all queries
    print("Computing CLIP text embeddings...")
    text_embs = get_clip_text_embeddings(queries, device=device)

    # Load wrapper
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )

    # Extract hidden states once
    print("Extracting hidden states...")
    hidden = wrapper.extract_hidden_states(video_autogaze)

    # Get scores for each query
    print("Computing semantic scores for each query...")
    all_scores = []
    for i, query in enumerate(queries):
        scores = wrapper.semantic_filter.get_scores(
            hidden, text_embs[i:i+1]
        )
        all_scores.append(scores[0].cpu().numpy())

        # Print per-query stats
        s = scores[0].cpu().numpy()
        print(f"  '{query}': mean={s.mean():.4f}, max={s.max():.4f}, "
              f">0.5: {(s > 0.5).sum()}/{len(s)}")

    # Compute pairwise overlap between queries
    print("\nPatch selection overlap analysis:")
    N = 14 * 14
    k = max(1, int(args.keep_ratio * N))
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            # Check overlap for middle frame
            mid = len(raw_frames) // 2
            si = all_scores[i][mid * N:(mid + 1) * N]
            sj = all_scores[j][mid * N:(mid + 1) * N]
            idx_i = set(np.argsort(si)[-k:])
            idx_j = set(np.argsort(sj)[-k:])
            overlap = len(idx_i & idx_j)
            print(f"  '{queries[i]}' vs '{queries[j]}': "
                  f"{overlap}/{k} overlap ({overlap/k*100:.0f}%)")

    # Visualize
    visualize_multi_query(
        raw_frames, all_scores, queries, args.keep_ratio,
        os.path.join(args.output_dir, "multi_query_comparison.png"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="assets/example_input.mp4")
    parser.add_argument("--queries", nargs="+", default=None,
                        help="List of text queries")
    parser.add_argument("--keep_ratio", type=float, default=0.2)
    parser.add_argument("--ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--head_type", default="bighead")
    parser.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/multi_query")
    args = parser.parse_args()
    main(args)
