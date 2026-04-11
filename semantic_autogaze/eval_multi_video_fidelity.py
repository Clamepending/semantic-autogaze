"""
Multi-video feature fidelity evaluation.

Runs the feature fidelity test on multiple videos to get
statistically robust numbers for the cos similarity metric.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -m semantic_autogaze.eval_multi_video_fidelity \
    --video_dir data --n_videos 20 --device cuda:0
"""

import os
import glob
import random
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open_clip
import av

from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper


def get_clip_text_embedding(text, model, tokenizer, device="cuda"):
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
    return text_features


def main(args):
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load CLIP for text embeddings
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()

    # Load AutoGaze wrapper
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )

    autogaze_transform = AutoGazeImageProcessor.from_pretrained(args.autogaze_model)

    # Load SigLIP
    from autogaze.vision_encoders.siglip import SiglipVisionModel
    from transformers import AutoImageProcessor

    siglip_transform = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
    siglip_model = SiglipVisionModel.from_pretrained(
        "google/siglip2-base-patch16-224",
        scales=wrapper.autogaze.config.scales,
        attn_implementation="sdpa",
    ).to(device).eval()

    # Find videos
    videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    random.shuffle(videos)
    videos = videos[:args.n_videos]
    print(f"Testing {len(videos)} videos")

    queries = args.queries.split(",")
    configs = [
        ("Gaze only (75%)", "gaze_only", 0.75, 0.7, 1.0),
        ("Intersect (50%)", "intersect", 0.75, 0.7, 0.5),
        ("Intersect (10%)", "intersect", 0.75, 0.7, 0.1),
        ("Semantic only (20%)", "semantic_only", 0.75, 0.7, 0.2),
        ("Semantic only (10%)", "semantic_only", 0.75, 0.7, 0.1),
    ]

    # Collect results across all videos and queries
    config_sims = {name: [] for name, *_ in configs}
    config_tokens = {name: [] for name, *_ in configs}

    for vi, video_path in enumerate(videos):
        print(f"\n  [{vi+1}/{len(videos)}] {os.path.basename(video_path)}")
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            n_frames = stream.frames
            if n_frames < 16:
                # Try to read 16 frames anyway
                indices = list(range(min(16, n_frames or 16)))
            else:
                indices = list(range(16))
            raw_video = read_video_pyav(container=container, indices=indices)
            container.close()

            if raw_video.shape[0] < 16:
                print(f"    Skipping — only {raw_video.shape[0]} frames")
                continue

            video_autogaze = transform_video_for_pytorch(raw_video, autogaze_transform)[None].to(device)
            video_siglip = transform_video_for_pytorch(raw_video, siglip_transform)[None].to(device)
        except Exception as e:
            print(f"    Error loading video: {e}")
            continue

        for query in queries:
            query = query.strip()
            text_emb = get_clip_text_embedding(query, clip_model, clip_tokenizer, device=device)

            # Reference: gaze-only
            ref_info = wrapper.forward(
                video_autogaze, text_emb, mode="gaze_only",
                gazing_ratio=0.75, task_loss_requirement=0.7, semantic_keep_ratio=1.0,
            )
            ref_out = siglip_model(video_siglip, gazing_info=ref_info)
            ref_hidden = ref_out.last_hidden_state
            ref_mask = (~ref_info["if_padded_gazing"].bool())[:, :ref_hidden.shape[1]].unsqueeze(-1).float()
            ref_feat = F.normalize((ref_hidden * ref_mask).sum(1) / ref_mask.sum(1).clamp(min=1), dim=-1)

            for name, mode, gaze_r, task_r, sem_r in configs:
                gazing_info = wrapper.forward(
                    video_autogaze, text_emb, mode=mode,
                    gazing_ratio=gaze_r, task_loss_requirement=task_r,
                    semantic_keep_ratio=sem_r,
                )
                out = siglip_model(video_siglip, gazing_info=gazing_info)
                hidden = out.last_hidden_state
                mask = (~gazing_info["if_padded_gazing"].bool())[:, :hidden.shape[1]].unsqueeze(-1).float()
                feat = F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1), dim=-1)

                cos_sim = F.cosine_similarity(ref_feat, feat).item()
                n_tokens = (~gazing_info["if_padded_gazing"]).sum().item()

                config_sims[name].append(cos_sim)
                config_tokens[name].append(n_tokens)

    # Cleanup CLIP model
    del clip_model
    torch.cuda.empty_cache()

    # Aggregate results
    print(f"\n{'='*70}")
    print(f"MULTI-VIDEO FEATURE FIDELITY ({len(videos)} videos, {len(queries)} queries)")
    print(f"{'='*70}")
    summary = {}
    for name, _, _, _, _ in configs:
        sims = config_sims[name]
        tokens = config_tokens[name]
        if sims:
            summary[name] = {
                "mean_cos_sim": float(np.mean(sims)),
                "std_cos_sim": float(np.std(sims)),
                "mean_tokens": float(np.mean(tokens)),
                "n_samples": len(sims),
            }
            print(f"  {name:<25}: cos_sim={np.mean(sims):.4f} ± {np.std(sims):.4f}, "
                  f"tokens={np.mean(tokens):.0f} (n={len(sims)})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    mode_colors = {
        "Gaze only": "#2196F3", "Intersect": "#4CAF50", "Semantic only": "#FF9800"
    }

    for name, data in summary.items():
        mode = name.split("(")[0].strip()
        color = mode_colors.get(mode, "#9E9E9E")
        ax.errorbar(data["mean_tokens"], data["mean_cos_sim"],
                     yerr=data["std_cos_sim"],
                     fmt="o", color=color, markersize=10, capsize=5,
                     elinewidth=1.5, markeredgecolor="black", markeredgewidth=0.5)
        ax.annotate(name, (data["mean_tokens"], data["mean_cos_sim"]),
                    textcoords="offset points", xytext=(8, -12), fontsize=8)

    ax.set_xlabel("Mean Tokens")
    ax.set_ylabel("Mean Cosine Similarity to Reference")
    ax.set_title(f"Feature Fidelity Across {len(videos)} Videos ({len(queries)} queries)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.05])

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "multi_video_fidelity.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved: {args.output_dir}/multi_video_fidelity.png")

    with open(os.path.join(args.output_dir, "multi_video_fidelity.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {args.output_dir}/multi_video_fidelity.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="data")
    parser.add_argument("--n_videos", type=int, default=20)
    parser.add_argument("--queries", default="person,face,background,animal")
    parser.add_argument("--ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--head_type", default="bighead")
    parser.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/multi_video_fidelity")
    args = parser.parse_args()
    main(args)
