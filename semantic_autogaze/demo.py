"""
End-to-end demo of Semantic AutoGaze filtering.

Given a video and text query, produces:
  1. Per-patch semantic scores
  2. Filtered gazing_info (compatible with AutoGaze VLM pipeline)
  3. Visualization showing kept vs dropped patches
  4. Statistics on token reduction

Usage:
  python3 -m semantic_autogaze.demo \
    --video data/example.mp4 \
    --query "person holding a cup" \
    --keep_ratio 0.2 \
    --ckpt results/distill_bighead/best_bighead_student.pt
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange
import av
import open_clip

from autogaze.models.autogaze import AutoGaze
from semantic_autogaze.semantic_filter import SemanticFilter


def load_video_frames(video_path, num_frames=16, size=224):
    """Load and preprocess video frames."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames or 1000
    stride = max(1, total // num_frames)

    frames = []
    raw_frames = []
    for i, frame in enumerate(container.decode(stream)):
        if len(frames) >= num_frames:
            break
        if i % stride == 0:
            img = frame.to_image().resize((size, size))
            raw_frames.append(np.array(frame.to_image()))
            arr = np.array(img).astype(np.float32) / 255.0
            arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            frames.append(arr)

    container.close()

    # Pad if needed
    while len(frames) < num_frames:
        frames.append(frames[-1])
        raw_frames.append(raw_frames[-1])

    video = torch.tensor(np.stack(frames[:num_frames])).permute(0, 3, 1, 2).float()
    return video.unsqueeze(0), raw_frames[:num_frames]  # (1, T, 3, H, W), list of HWC arrays


def get_clip_text_embedding(text, device="cuda"):
    """Get CLIP text embedding for a query."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model = model.to(device).eval()

    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)

    embedding = text_features.squeeze(0)  # (512,)
    del model
    torch.cuda.empty_cache()
    return embedding


def extract_hidden_states(autogaze, video, device):
    """Extract AutoGaze hidden states from video."""
    gaze_model = autogaze.gazing_model
    B, T = video.shape[:2]

    video_resized = rearrange(video, 'b t c h w -> (b t) c h w')
    video_resized = F.interpolate(
        video_resized,
        size=(gaze_model.input_img_size, gaze_model.input_img_size),
        mode="bicubic", align_corners=False,
    )
    video_resized = rearrange(video_resized, '(b t) c h w -> b t c h w', b=B)

    with torch.no_grad():
        vision_features, _ = gaze_model.vision_model(video_resized)
        vision_features = vision_features.transpose(1, 2)
        vision_features = rearrange(vision_features, 'b t c h w -> b t (h w) c')
        vision_features = gaze_model.connector(vision_features)

        B2, T2, N, C = vision_features.shape
        inputs_embeds = vision_features.reshape(B2, T2 * N, C)
        attention_mask = torch.ones(B2, T2 * N, device=device, dtype=torch.long)
        decoder_outputs = gaze_model.gaze_decoder.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=attention_mask.cumsum(dim=-1) - 1,
        )

    return decoder_outputs.last_hidden_state  # (B, T*196, hidden_dim)


def visualize_demo(raw_frames, scores, gazing_info, query, keep_ratio,
                   output_path, grid_size=14):
    """Create a multi-frame visualization of the filtering result."""
    T = len(raw_frames)
    N = grid_size * grid_size

    # Show 4 evenly-spaced frames
    frame_indices = [0, T // 4, T // 2, 3 * T // 4]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    scores_np = scores[0].cpu().numpy()

    for col, fi in enumerate(frame_indices):
        frame = raw_frames[fi]
        frame_scores = scores_np[fi * N:(fi + 1) * N]

        # Top row: heatmap overlay
        H, W = frame.shape[:2]
        heatmap = frame_scores.reshape(grid_size, grid_size)
        from PIL import Image
        heatmap_resized = np.array(Image.fromarray(heatmap.astype(np.float32)).resize(
            (W, H), Image.BILINEAR))
        vmax = max(heatmap_resized.max(), 0.01)
        heatmap_norm = heatmap_resized / vmax

        cm = plt.get_cmap("jet")
        colored = cm(heatmap_norm)[:, :, :3]
        overlay = (frame / 255.0) * 0.5 + colored * 0.5
        axes[0, col].imshow(overlay.clip(0, 1))
        axes[0, col].set_title(f"Frame {fi}", fontsize=10)
        axes[0, col].axis("off")

        # Bottom row: kept patches
        k = max(1, int(keep_ratio * N))
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
        axes[1, col].set_title(f"Keep {k}/{N}", fontsize=10)
        axes[1, col].axis("off")

    fig.suptitle(f'Query: "{query}" | Keep ratio: {keep_ratio*100:.0f}%',
                 fontsize=14, fontweight="bold")
    axes[0, 0].set_ylabel("Semantic\nHeatmap", fontsize=11, rotation=0, labelpad=50)
    axes[1, 0].set_ylabel("Filtered\nPatches", fontsize=11, rotation=0, labelpad=50)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


def main(args):
    device = torch.device(args.device)

    print(f"Video: {args.video}")
    print(f"Query: {args.query}")
    print(f"Keep ratio: {args.keep_ratio*100:.0f}%")

    # Load video
    print("\n1. Loading video...")
    video, raw_frames = load_video_frames(args.video, num_frames=16)
    video = video.to(device)
    print(f"   Video shape: {video.shape}")

    # Get text embedding
    print("2. Computing text embedding...")
    text_emb = get_clip_text_embedding(args.query, device=device)
    print(f"   Embedding shape: {text_emb.shape}")

    # Extract hidden states
    print("3. Extracting AutoGaze hidden states...")
    autogaze = AutoGaze.from_pretrained(args.autogaze_model, use_flash_attn=False).to(device).eval()
    hidden = extract_hidden_states(autogaze, video, device)
    print(f"   Hidden states shape: {hidden.shape}")

    # Run semantic filtering
    print("4. Running semantic filtering...")
    sf = SemanticFilter(
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )

    scores = sf.get_scores(hidden, text_emb.unsqueeze(0))
    gazing_info = sf.scores_to_gazing_info(scores, keep_ratio=args.keep_ratio)
    stats = sf.get_filtering_stats(scores, keep_ratio=args.keep_ratio)

    # Print results
    print(f"\n{'='*50}")
    print(f"SEMANTIC FILTERING RESULTS")
    print(f"{'='*50}")
    print(f"  Patches per frame: {stats['total_patches_per_frame']}")
    print(f"  Kept per frame: {stats['patches_kept_per_frame']}")
    print(f"  Token savings: {stats['tokens_saved_pct']:.1f}%")
    print(f"  Mean semantic score: {stats['mean_score']:.4f}")
    print(f"  Mean kept score: {stats['mean_kept_score']:.4f}")
    print(f"  gazing_pos shape: {gazing_info['gazing_pos'].shape}")
    print(f"  Padded positions: {gazing_info['if_padded_gazing'].sum().item()}")

    # Visualize
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    visualize_demo(raw_frames, scores, gazing_info, args.query,
                   args.keep_ratio, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--query", required=True, help="Text query for filtering")
    parser.add_argument("--keep_ratio", type=float, default=0.2)
    parser.add_argument("--ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--head_type", default="bighead")
    parser.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results/demo_output.png")
    args = parser.parse_args()
    main(args)
