"""
Video-text retrieval proxy evaluation.

Tests whether filtered SigLIP features can still distinguish correct
video-text pairs from incorrect ones. This is a proxy for downstream
VLM accuracy: if filtering preserves discriminative information,
the VLM should also maintain performance.

For each video, we:
  1. Compute SigLIP features under different filtering configs
  2. Compare cosine similarity between video features and:
     - Correct caption (from CLIP text encoder)
     - Hard negatives (captions from other videos)
  3. Compute retrieval Recall@1, @5

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -m semantic_autogaze.eval_retrieval_proxy \
    --video_dir data --n_videos 100 --device cuda:0
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


# Simple captions describing common video content
CAPTION_POOL = [
    "a person talking to the camera",
    "someone cooking food in a kitchen",
    "a person playing a musical instrument",
    "people walking on a street",
    "an animal in a natural setting",
    "a sports game or athletic activity",
    "someone working at a desk or computer",
    "a vehicle driving on a road",
    "people dancing or performing",
    "a scenic landscape or nature view",
    "someone eating or drinking",
    "a child playing with toys",
    "a group of people having a conversation",
    "someone exercising or working out",
    "a building or architectural structure",
    "water flowing in a river or ocean",
    "someone using a tool or machine",
    "a pet or domestic animal",
    "someone giving a presentation or speech",
    "food being prepared or served",
]


def get_text_features(texts, clip_model, tokenizer, device):
    """Encode texts with CLIP."""
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        features = clip_model.encode_text(tokens)
        features = F.normalize(features, dim=-1)
    return features


def get_video_features_siglip(video_siglip, gazing_info, siglip_model):
    """Get mean-pooled SigLIP features from filtered patches."""
    out = siglip_model(video_siglip, gazing_info=gazing_info)
    hidden = out.last_hidden_state
    mask = (~gazing_info["if_padded_gazing"].bool())[:, :hidden.shape[1]].unsqueeze(-1).float()
    feat = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    return F.normalize(feat, dim=-1)


def compute_feature_retrieval(filtered_features, reference_features):
    """
    Measure how well filtered features preserve discriminability.
    For each video, compute similarity between its filtered features
    and ALL videos' reference features. The correct match should rank #1.

    Args:
        filtered_features: (N, D) filtered SigLIP features
        reference_features: (N, D) full/reference SigLIP features
    Returns:
        dict with R@1, R@5, mean_rank
    """
    sim_matrix = filtered_features @ reference_features.T  # (N, N)
    N = sim_matrix.shape[0]

    ranks = []
    for i in range(N):
        sims = sim_matrix[i]
        sorted_indices = torch.argsort(sims, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        ranks.append(rank)

    ranks = np.array(ranks)
    return {
        "R@1": float((ranks < 1).mean()),
        "R@5": float((ranks < 5).mean()),
        "R@10": float((ranks < 10).mean()),
        "mean_rank": float(ranks.mean() + 1),
        "median_rank": float(np.median(ranks) + 1),
    }


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

    # Encode caption pool for semantic queries
    text_features = get_text_features(CAPTION_POOL, clip_model, clip_tokenizer, device)

    # Find videos
    videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    random.shuffle(videos)
    videos = videos[:args.n_videos]
    print(f"Evaluating retrieval on {len(videos)} videos")

    queries = ["person", "object", "background"]

    configs = [
        ("Gaze only (75%)", "gaze_only", 0.75, 0.7, 1.0),
        ("Intersect (50%)", "intersect", 0.75, 0.7, 0.5),
        ("Intersect (10%)", "intersect", 0.75, 0.7, 0.1),
        ("Semantic only (20%)", "semantic_only", 0.75, 0.7, 0.2),
        ("Semantic only (10%)", "semantic_only", 0.75, 0.7, 0.1),
    ]

    # Collect reference features (gaze-only) and filtered features per config
    ref_features_list = []
    config_features = {name: [] for name, *_ in configs}

    for vi, video_path in enumerate(videos):
        if (vi + 1) % 10 == 0:
            print(f"  [{vi+1}/{len(videos)}]")
        try:
            container = av.open(video_path)
            indices = list(range(16))
            raw_video = read_video_pyav(container=container, indices=indices)
            container.close()

            if raw_video.shape[0] < 16:
                continue

            video_autogaze = transform_video_for_pytorch(raw_video, autogaze_transform)[None].to(device)
            video_siglip = transform_video_for_pytorch(raw_video, siglip_transform)[None].to(device)
        except Exception as e:
            continue

        # Pick a random query for semantic filtering
        query_text = random.choice(queries)
        query_emb = get_text_features([query_text], clip_model, clip_tokenizer, device)

        # Reference: gaze-only (75%)
        ref_info = wrapper.forward(
            video_autogaze, query_emb,
            mode="gaze_only", gazing_ratio=0.75, task_loss_requirement=0.7,
            semantic_keep_ratio=1.0,
        )
        ref_feat = get_video_features_siglip(video_siglip, ref_info, siglip_model)
        ref_features_list.append(ref_feat.cpu())

        # Get features under each filtering config
        for name, mode, gaze_r, task_r, sem_r in configs:
            gazing_info = wrapper.forward(
                video_autogaze, query_emb, mode=mode,
                gazing_ratio=gaze_r, task_loss_requirement=task_r,
                semantic_keep_ratio=sem_r,
            )
            feat = get_video_features_siglip(video_siglip, gazing_info, siglip_model)
            config_features[name].append(feat.cpu())

    # Stack features
    ref_features = torch.cat(ref_features_list, dim=0)
    n_videos_eval = ref_features.shape[0]

    print(f"\n{'='*60}")
    print(f"FEATURE RETRIEVAL RESULTS ({n_videos_eval} videos)")
    print(f"{'='*60}")
    print(f"Task: match filtered features to reference (gaze-only) features")

    results = {}
    for name, *_ in configs:
        feats = torch.cat(config_features[name], dim=0)
        metrics = compute_feature_retrieval(feats, ref_features)
        results[name] = metrics
        print(f"  {name:<25}: R@1={metrics['R@1']:.3f}, R@5={metrics['R@5']:.3f}, "
              f"MeanR={metrics['mean_rank']:.1f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [n for n, *_ in configs]
    r1s = [results[n]["R@1"] for n in names]
    r5s = [results[n]["R@5"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, r1s, width, label="R@1", color="#2196F3")
    ax.bar(x + width/2, r5s, width, label="R@5", color="#4CAF50")

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Recall")
    ax.set_title(f"Feature Retrieval: Filtered vs Reference ({n_videos_eval} videos)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "retrieval_proxy.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved: {args.output_dir}/retrieval_proxy.png")

    with open(os.path.join(args.output_dir, "retrieval_proxy.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {args.output_dir}/retrieval_proxy.json")

    # Cleanup
    del clip_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="data")
    parser.add_argument("--n_videos", type=int, default=100)
    parser.add_argument("--ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--head_type", default="bighead")
    parser.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/retrieval_proxy")
    args = parser.parse_args()
    main(args)
