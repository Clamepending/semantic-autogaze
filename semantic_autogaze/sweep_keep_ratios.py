"""
Sweep different semantic keep ratios through the full pipeline.

For each keep ratio, measures:
  - Number of tokens kept
  - SigLIP encoding time
  - Token distribution across frames

Usage:
  python3 -m semantic_autogaze.sweep_keep_ratios \
    --video assets/example_input.mp4 \
    --query "person holding a cup" \
    --ckpt results/distill_bighead/best_bighead_student.pt
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open_clip
import av
from einops import rearrange

from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper


def get_clip_text_embedding(text, device="cuda"):
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model = model.to(device).eval()
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
    embedding = text_features
    del model
    torch.cuda.empty_cache()
    return embedding


def main(args):
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load video
    container = av.open(args.video)
    indices = list(range(16))
    raw_video = read_video_pyav(container=container, indices=indices)
    container.close()

    autogaze_transform = AutoGazeImageProcessor.from_pretrained(args.autogaze_model)
    video_autogaze = transform_video_for_pytorch(raw_video, autogaze_transform)[None].to(device)

    # Text embedding
    text_emb = get_clip_text_embedding(args.query, device=device)

    # Load wrapper
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )

    # Load SigLIP
    try:
        from autogaze.vision_encoders.siglip import SiglipVisionModel
        from transformers import AutoImageProcessor

        siglip_transform = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
        siglip_model = SiglipVisionModel.from_pretrained(
            "google/siglip2-base-patch16-224",
            scales=wrapper.autogaze.config.scales,
            attn_implementation="sdpa",
        ).to(device).eval()
        video_siglip = transform_video_for_pytorch(raw_video, siglip_transform)[None].to(device)
        has_siglip = True
    except ImportError:
        has_siglip = False

    # Baseline: gaze only
    gaze_only = wrapper.forward(video_autogaze, text_emb, mode="gaze_only",
                                gazing_ratio=args.gazing_ratio,
                                task_loss_requirement=args.task_loss_requirement)
    gaze_tokens = (~gaze_only["if_padded_gazing"]).sum().item()

    if has_siglip:
        # Warmup
        _ = siglip_model(video_siglip, gazing_info=gaze_only)
        torch.cuda.synchronize()

    # Sweep keep ratios
    keep_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []

    # Also extract hidden states once
    hidden = wrapper.extract_hidden_states(video_autogaze)
    scores = wrapper.semantic_filter.get_scores(hidden, text_emb)

    for kr in keep_ratios:
        # Intersect mode
        filtered = wrapper.semantic_filter.intersect_with_gaze(
            scores, gaze_only, semantic_keep_ratio=kr,
        )
        n_tokens = (~filtered["if_padded_gazing"]).sum().item()

        # Semantic-only mode
        sem_only = wrapper.semantic_filter.scores_to_gazing_info(
            scores, keep_ratio=kr, num_frames=16,
        )
        n_tokens_sem = (~sem_only["if_padded_gazing"]).sum().item()

        enc_time = None
        if has_siglip:
            # Time SigLIP encoding (average of 3 runs)
            times = []
            for _ in range(3):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = siglip_model(video_siglip, gazing_info=filtered)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            enc_time = np.mean(times) * 1000  # ms

        results.append({
            "keep_ratio": kr,
            "intersect_tokens": n_tokens,
            "semantic_only_tokens": n_tokens_sem,
            "siglip_time_ms": enc_time,
        })
        print(f"  keep_ratio={kr:.1f}: intersect={n_tokens}, "
              f"sem_only={n_tokens_sem}"
              + (f", siglip={enc_time:.1f}ms" if enc_time else ""))

    # Also measure gaze-only SigLIP time
    gaze_enc_time = None
    if has_siglip:
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = siglip_model(video_siglip, gazing_info=gaze_only)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        gaze_enc_time = np.mean(times) * 1000

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Token count vs keep ratio
    ax = axes[0]
    krs = [r["keep_ratio"] for r in results]
    int_tokens = [r["intersect_tokens"] for r in results]
    sem_tokens = [r["semantic_only_tokens"] for r in results]

    ax.plot(krs, int_tokens, 'o-', color="#2196F3", lw=2, label="Intersect (Gaze + Semantic)")
    ax.plot(krs, sem_tokens, 's--', color="#FF9800", lw=2, label="Semantic only")
    ax.axhline(y=gaze_tokens, color="gray", linestyle=":", lw=1.5,
               label=f"Gaze only ({gaze_tokens})")
    ax.axhline(y=16*196, color="red", linestyle=":", lw=1, alpha=0.5,
               label=f"All patches ({16*196})")

    ax.set_xlabel("Semantic Keep Ratio")
    ax.set_ylabel("Tokens Kept")
    ax.set_title(f'Token Budget vs Keep Ratio\nQuery: "{args.query}"')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: SigLIP encoding time
    ax = axes[1]
    if has_siglip:
        enc_times = [r["siglip_time_ms"] for r in results]
        ax.plot(int_tokens, enc_times, 'o-', color="#4CAF50", lw=2,
                label="Intersect mode")
        ax.plot(gaze_tokens, gaze_enc_time, '*', color="gray", markersize=15,
                label=f"Gaze only ({gaze_enc_time:.1f}ms)")
        ax.set_xlabel("Tokens")
        ax.set_ylabel("SigLIP Encoding Time (ms)")
        ax.set_title("SigLIP Latency vs Token Count")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "SigLIP not available", ha="center", va="center",
                transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "keep_ratio_sweep.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved: {args.output_dir}/keep_ratio_sweep.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="assets/example_input.mp4")
    parser.add_argument("--query", default="person")
    parser.add_argument("--gazing_ratio", type=float, default=0.75)
    parser.add_argument("--task_loss_requirement", type=float, default=0.7)
    parser.add_argument("--ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--head_type", default="bighead")
    parser.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/sweep")
    args = parser.parse_args()
    main(args)
