"""
NVILA integration: SemanticAutoGaze + SigLIP → VLM features.

Demonstrates the full pipeline from video + text query to filtered
SigLIP features ready for the NVILA language model.

Pipeline:
  1. Load video → preprocess for AutoGaze and SigLIP
  2. AutoGaze → gazing_info (reconstruction-based patch selection)
  3. Semantic head scores patches against text query
  4. Filter gazing_info → only semantically relevant patches
  5. SigLIP encodes only the filtered patches
  6. Output: (B, K, 768) features for the LLM, where K << T*196

Usage:
  python3 -m semantic_autogaze.nvila_integration \
    --video data/example.mp4 \
    --query "person holding a cup" \
    --gazing_ratio 0.75 \
    --semantic_keep_ratio 0.5 \
    --ckpt results/distill_bighead/best_bighead_student.pt

Requires: autogaze, open_clip, transformers (for SigLIP)
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
import av
from einops import rearrange

from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper


def load_video(video_path, num_frames=16):
    """Load video frames for both AutoGaze and SigLIP preprocessing."""
    container = av.open(video_path)
    indices = list(range(num_frames))
    raw_video = read_video_pyav(container=container, indices=indices)
    container.close()
    return raw_video


def get_clip_text_embedding(text, device="cuda"):
    """Get CLIP text embedding for query-conditioned filtering."""
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model = model.to(device).eval()

    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)

    embedding = text_features  # (1, 512)
    del model
    torch.cuda.empty_cache()
    return embedding


def run_pipeline(args):
    """Run the full SemanticAutoGaze → SigLIP pipeline."""
    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    print(f"=== SemanticAutoGaze + NVILA Integration ===")
    print(f"Video: {args.video}")
    print(f"Query: {args.query}")
    print(f"Mode: {args.mode}")
    print(f"Gaze ratio: {args.gazing_ratio}")
    print(f"Semantic keep: {args.semantic_keep_ratio}")
    print()

    # 1. Load video
    print("1. Loading video...")
    raw_video = load_video(args.video, num_frames=16)
    print(f"   Loaded {len(raw_video)} frames")

    # 2. Preprocess for AutoGaze
    print("2. Preprocessing...")
    autogaze_transform = AutoGazeImageProcessor.from_pretrained(args.autogaze_model)
    video_autogaze = transform_video_for_pytorch(raw_video, autogaze_transform)
    video_autogaze = video_autogaze[None].to(device)  # (1, T, C, H, W)
    print(f"   AutoGaze input: {video_autogaze.shape}")

    # 3. Get text embedding
    print("3. Computing CLIP text embedding...")
    text_emb = get_clip_text_embedding(args.query, device=device)
    print(f"   Embedding: {text_emb.shape}")

    # 4. Run SemanticAutoGaze
    print("4. Running SemanticAutoGaze wrapper...")
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )

    # Baseline: gaze only
    t0 = time.perf_counter()
    gaze_only = wrapper.forward(
        video_autogaze, text_emb,
        mode="gaze_only",
        gazing_ratio=args.gazing_ratio,
        task_loss_requirement=args.task_loss_requirement,
    )
    torch.cuda.synchronize()
    t_gaze = time.perf_counter() - t0

    # Semantic filtered
    t0 = time.perf_counter()
    filtered = wrapper.forward(
        video_autogaze, text_emb,
        mode=args.mode,
        gazing_ratio=args.gazing_ratio,
        task_loss_requirement=args.task_loss_requirement,
        semantic_keep_ratio=args.semantic_keep_ratio,
    )
    torch.cuda.synchronize()
    t_filtered = time.perf_counter() - t0

    # Stats
    gaze_tokens = (~gaze_only["if_padded_gazing"]).sum().item()
    filtered_tokens = (~filtered["if_padded_gazing"]).sum().item()

    print(f"\n{'='*50}")
    print(f"TOKEN REDUCTION RESULTS")
    print(f"{'='*50}")
    print(f"  AutoGaze only:      {gaze_tokens} tokens ({t_gaze*1000:.1f}ms)")
    print(f"  + Semantic filter:   {filtered_tokens} tokens ({t_filtered*1000:.1f}ms)")
    print(f"  Reduction:           {gaze_tokens - filtered_tokens} tokens "
          f"({(1 - filtered_tokens/max(gaze_tokens,1))*100:.1f}%)")
    print(f"  gazing_pos shape:    {filtered['gazing_pos'].shape}")
    print(f"  Padded positions:    {filtered['if_padded_gazing'].sum().item()}")

    # 5. Encode with SigLIP (if available)
    try:
        from autogaze.vision_encoders.siglip import SiglipVisionModel
        from transformers import AutoImageProcessor

        print("\n5. Encoding with SigLIP...")
        siglip_transform = AutoImageProcessor.from_pretrained(
            "google/siglip2-base-patch16-224"
        )
        siglip_model = SiglipVisionModel.from_pretrained(
            "google/siglip2-base-patch16-224",
            scales=wrapper.autogaze.config.scales,
            attn_implementation="sdpa",
        ).to(device).eval()

        video_siglip = transform_video_for_pytorch(raw_video, siglip_transform)
        video_siglip = video_siglip[None].to(device)

        # Encode with gaze-only
        t0 = time.perf_counter()
        out_gaze = siglip_model(video_siglip, gazing_info=gaze_only)
        torch.cuda.synchronize()
        t_enc_gaze = time.perf_counter() - t0

        # Encode with semantic filter
        t0 = time.perf_counter()
        out_filtered = siglip_model(video_siglip, gazing_info=filtered)
        torch.cuda.synchronize()
        t_enc_filtered = time.perf_counter() - t0

        # Remove padded features
        gaze_features = [
            f[~pad] for f, pad in
            zip(out_gaze.last_hidden_state, gaze_only["if_padded_gazing"])
        ]
        filtered_features = [
            f[~pad] for f, pad in
            zip(out_filtered.last_hidden_state, filtered["if_padded_gazing"])
        ]

        print(f"\n{'='*50}")
        print(f"SIGLIP ENCODING RESULTS")
        print(f"{'='*50}")
        print(f"  Gaze-only features:    {gaze_features[0].shape} ({t_enc_gaze*1000:.1f}ms)")
        print(f"  Filtered features:     {filtered_features[0].shape} ({t_enc_filtered*1000:.1f}ms)")
        print(f"  SigLIP encoding speedup: {t_enc_gaze/max(t_enc_filtered, 1e-6):.2f}x")
        print(f"\n  These features are ready to be fed into the NVILA LLM.")

    except ImportError:
        print("\n5. SigLIP encoding skipped (siglip model not available)")
        print("   The gazing_info dict is ready for SigLIP integration.")

    # 6. Summary
    total_patches = 16 * 196  # 16 frames * 14*14 patches
    print(f"\n{'='*50}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*50}")
    print(f"  Total patches (all frames):  {total_patches}")
    print(f"  After AutoGaze:              {gaze_tokens} ({gaze_tokens/total_patches*100:.1f}%)")
    print(f"  After semantic filter:       {filtered_tokens} ({filtered_tokens/total_patches*100:.1f}%)")
    print(f"  Final token savings:         {(1 - filtered_tokens/total_patches)*100:.1f}%")

    return filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", default="intersect",
                        choices=["intersect", "semantic_only", "gaze_only"])
    parser.add_argument("--gazing_ratio", type=float, default=0.75)
    parser.add_argument("--task_loss_requirement", type=float, default=0.7)
    parser.add_argument("--semantic_keep_ratio", type=float, default=0.5)
    parser.add_argument("--ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--head_type", default="bighead")
    parser.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_pipeline(args)
