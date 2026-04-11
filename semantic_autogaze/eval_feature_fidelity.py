"""
Evaluate feature fidelity: how well do filtered patches preserve
the downstream SigLIP features compared to using all patches?

This is a proxy for VLM quality impact without needing the full VLM.
Measures cosine similarity between:
  - SigLIP features from ALL patches (reference)
  - SigLIP features from FILTERED patches (various strategies)

Higher cosine similarity = better feature preservation = likely better VLM performance.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -m semantic_autogaze.eval_feature_fidelity \
    --video assets/example_input.mp4 \
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
    del model
    torch.cuda.empty_cache()
    return text_features


def get_siglip_features(siglip_model, video, gazing_info=None):
    """Extract SigLIP features, returning pooled feature vector."""
    with torch.no_grad():
        out = siglip_model(video, gazing_info=gazing_info)
    # Pool across all tokens: mean of non-padded tokens
    if gazing_info is not None and "if_padded_gazing" in gazing_info:
        # out.last_hidden_state shape: (B, N_tokens, D)
        hidden = out.last_hidden_state
        mask = ~gazing_info["if_padded_gazing"]  # (B, K)
        # Expand mask for broadcasting
        mask = mask[:, :hidden.shape[1]].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        hidden = out.last_hidden_state
        pooled = hidden.mean(dim=1)
    return F.normalize(pooled, dim=-1)


def main(args):
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load video
    container = av.open(args.video)
    raw_video = read_video_pyav(container=container, indices=list(range(16)))
    container.close()

    autogaze_transform = AutoGazeImageProcessor.from_pretrained(args.autogaze_model)
    video_autogaze = transform_video_for_pytorch(raw_video, autogaze_transform)[None].to(device)

    # Load models
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )

    # Load SigLIP
    from autogaze.vision_encoders.siglip import SiglipVisionModel
    from transformers import AutoImageProcessor

    siglip_transform = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
    siglip_model = SiglipVisionModel.from_pretrained(
        "google/siglip2-base-patch16-224",
        scales=wrapper.autogaze.config.scales,
        attn_implementation="sdpa",
    ).to(device).eval()
    video_siglip = transform_video_for_pytorch(raw_video, siglip_transform)[None].to(device)

    # Reference: full SigLIP features (gaze-only with high ratio = all patches)
    print("Computing reference features (gaze-only, all patches)...")
    # Use gaze_only mode with 100% ratio to get all gazed patches as reference
    ref_text_emb = get_clip_text_embedding("person", device=device)
    ref_gazing_info = wrapper.forward(
        video_autogaze, ref_text_emb, mode="gaze_only",
        gazing_ratio=0.75, task_loss_requirement=0.7,
        semantic_keep_ratio=1.0,
    )
    ref_features = get_siglip_features(siglip_model, video_siglip, gazing_info=ref_gazing_info)
    ref_tokens = (~ref_gazing_info["if_padded_gazing"]).sum().item()
    print(f"  Reference: {ref_tokens} tokens")

    # Test multiple queries
    queries = args.queries.split(",")
    all_results = {}

    for query in queries:
        query = query.strip()
        print(f"\nQuery: '{query}'")
        text_emb = get_clip_text_embedding(query, device=device)

        configs = [
            ("Gaze only (75%)", "gaze_only", 0.75, 0.7, 1.0),
            ("Intersect (75%→50%)", "intersect", 0.75, 0.7, 0.5),
            ("Intersect (75%→30%)", "intersect", 0.75, 0.7, 0.3),
            ("Intersect (75%→10%)", "intersect", 0.75, 0.7, 0.1),
            ("Semantic only (50%)", "semantic_only", 0.75, 0.7, 0.5),
            ("Semantic only (20%)", "semantic_only", 0.75, 0.7, 0.2),
            ("Semantic only (10%)", "semantic_only", 0.75, 0.7, 0.1),
        ]

        query_results = []
        for name, mode, gaze_r, task_r, sem_r in configs:
            gazing_info = wrapper.forward(
                video_autogaze, text_emb, mode=mode,
                gazing_ratio=gaze_r, task_loss_requirement=task_r,
                semantic_keep_ratio=sem_r,
            )

            filtered_features = get_siglip_features(siglip_model, video_siglip,
                                                     gazing_info=gazing_info)
            cosine_sim = F.cosine_similarity(ref_features, filtered_features).item()
            n_tokens = (~gazing_info["if_padded_gazing"]).sum().item()

            query_results.append({
                "name": name,
                "mode": mode,
                "tokens": n_tokens,
                "cosine_sim": cosine_sim,
            })
            print(f"  {name:<25}: tokens={n_tokens:>5}, cos_sim={cosine_sim:.4f}")

        all_results[query] = query_results

    # Plot: feature fidelity vs tokens for each query
    fig, ax = plt.subplots(figsize=(10, 6))

    query_colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
    for i, (query, results) in enumerate(all_results.items()):
        color = query_colors[i % len(query_colors)]
        tokens = [r["tokens"] for r in results]
        sims = [r["cosine_sim"] for r in results]

        ax.scatter(tokens, sims, s=80, color=color, edgecolors="black",
                   linewidth=0.5, zorder=5, label=f'"{query}"')

        # Connect points by mode
        for mode, marker in [("gaze_only", "o"), ("intersect", "s"), ("semantic_only", "^")]:
            mode_results = [r for r in results if r["mode"] == mode]
            if mode_results:
                t = [r["tokens"] for r in mode_results]
                s = [r["cosine_sim"] for r in mode_results]
                ax.plot(t, s, f'{marker}--', color=color, alpha=0.5, markersize=6)

    ax.set_xlabel("Tokens Kept")
    ax.set_ylabel("Cosine Similarity to Full-Patch Features")
    ax.set_title("Feature Fidelity: How Well Do Filtered Patches\nPreserve SigLIP Representations?")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.02])

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "feature_fidelity.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved: {args.output_dir}/feature_fidelity.png")

    # Summary
    print(f"\n{'='*70}")
    print("FEATURE FIDELITY SUMMARY")
    print(f"{'='*70}")
    for query, results in all_results.items():
        print(f"\n  Query: '{query}'")
        for r in results:
            print(f"    {r['name']:<25}: {r['tokens']:>5} tokens, "
                  f"cos_sim={r['cosine_sim']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="assets/example_input.mp4")
    parser.add_argument("--queries", default="person,background,face,hand")
    parser.add_argument("--ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--head_type", default="bighead")
    parser.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/feature_fidelity")
    args = parser.parse_args()
    main(args)
