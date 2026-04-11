"""
End-to-end latency benchmark: AutoGaze + semantic filter + SigLIP.

Measures total pipeline latency for different configurations:
  1. AutoGaze → SigLIP (baseline)
  2. AutoGaze → Semantic Filter → SigLIP (intersect mode)
  3. AutoGaze → Semantic Filter → SigLIP (semantic-only mode)

Each is run multiple times with GPU warmup for reliable timing.

Usage:
  python3 -m semantic_autogaze.benchmark_e2e \
    --video assets/example_input.mp4 \
    --query "person" \
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
    del model
    torch.cuda.empty_cache()
    return text_features


def benchmark_config(wrapper, siglip_model, video_autogaze, video_siglip,
                     text_emb, mode, gazing_ratio, task_loss_req,
                     semantic_keep_ratio, n_warmup=3, n_runs=10):
    """Benchmark a single configuration."""
    device = video_autogaze.device

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Warmup
    for _ in range(n_warmup):
        gazing_info = wrapper.forward(
            video_autogaze, text_emb, mode=mode,
            gazing_ratio=gazing_ratio, task_loss_requirement=task_loss_req,
            semantic_keep_ratio=semantic_keep_ratio,
        )
        if siglip_model is not None:
            _ = siglip_model(video_siglip, gazing_info=gazing_info)
        sync()

    # Timed runs
    total_times = []
    autogaze_times = []
    semantic_times = []
    siglip_times = []
    token_counts = []

    for _ in range(n_runs):
        sync()

        # AutoGaze + semantic filter
        t0 = time.perf_counter()
        gazing_info = wrapper.forward(
            video_autogaze, text_emb, mode=mode,
            gazing_ratio=gazing_ratio, task_loss_requirement=task_loss_req,
            semantic_keep_ratio=semantic_keep_ratio,
        )
        sync()
        t_filter = time.perf_counter() - t0

        # SigLIP encoding
        t0 = time.perf_counter()
        if siglip_model is not None:
            out = siglip_model(video_siglip, gazing_info=gazing_info)
        sync()
        t_enc = time.perf_counter() - t0

        n_tokens = (~gazing_info["if_padded_gazing"]).sum().item()

        total_times.append(t_filter + t_enc)
        autogaze_times.append(t_filter)
        siglip_times.append(t_enc)
        token_counts.append(n_tokens)

    return {
        "total_ms": np.mean(total_times) * 1000,
        "total_std_ms": np.std(total_times) * 1000,
        "filter_ms": np.mean(autogaze_times) * 1000,
        "siglip_ms": np.mean(siglip_times) * 1000,
        "tokens": int(np.mean(token_counts)),
        "mode": mode,
        "gazing_ratio": gazing_ratio,
        "semantic_keep_ratio": semantic_keep_ratio,
    }


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
    text_emb = get_clip_text_embedding(args.query, device=device)

    # Load models
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )

    siglip_model = None
    video_siglip = None
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
    except ImportError:
        print("SigLIP not available, benchmarking filter only")

    # Configurations to benchmark
    configs = [
        ("Gaze only (75%)", "gaze_only", 0.75, 0.7, 1.0),
        ("Gaze only (50%)", "gaze_only", 0.50, None, 1.0),
        ("Intersect (75%→50%)", "intersect", 0.75, 0.7, 0.5),
        ("Intersect (75%→30%)", "intersect", 0.75, 0.7, 0.3),
        ("Intersect (75%→10%)", "intersect", 0.75, 0.7, 0.1),
        ("Semantic only (20%)", "semantic_only", 0.75, 0.7, 0.2),
        ("Semantic only (10%)", "semantic_only", 0.75, 0.7, 0.1),
    ]

    print(f"Benchmarking {len(configs)} configurations...")
    print(f"  Video: {args.video}")
    print(f"  Query: {args.query}")
    print(f"  Runs: {args.n_runs}")
    print()

    results = []
    for name, mode, gaze_r, task_r, sem_r in configs:
        print(f"  {name}...", end=" ", flush=True)
        r = benchmark_config(
            wrapper, siglip_model, video_autogaze, video_siglip, text_emb,
            mode=mode, gazing_ratio=gaze_r, task_loss_req=task_r,
            semantic_keep_ratio=sem_r, n_warmup=args.n_warmup, n_runs=args.n_runs,
        )
        r["name"] = name
        results.append(r)
        print(f"tokens={r['tokens']}, total={r['total_ms']:.1f}ms "
              f"(filter={r['filter_ms']:.1f}, siglip={r['siglip_ms']:.1f})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: total latency bar chart
    ax = axes[0]
    names = [r["name"] for r in results]
    totals = [r["total_ms"] for r in results]
    filter_times = [r["filter_ms"] for r in results]
    siglip_times = [r["siglip_ms"] for r in results]

    x = range(len(results))
    ax.barh(x, filter_times, color="#42A5F5", label="AutoGaze + Semantic")
    ax.barh(x, siglip_times, left=filter_times, color="#66BB6A", label="SigLIP Encoding")

    for i, total in enumerate(totals):
        ax.text(total + 1, i, f"{total:.1f}ms", va="center", fontsize=9)

    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Latency (ms)")
    ax.set_title("End-to-End Pipeline Latency")
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    # Right: latency vs tokens
    ax = axes[1]
    tokens = [r["tokens"] for r in results]
    for r in results:
        color = "#2196F3" if "Gaze only" in r["name"] else \
                "#4CAF50" if "Intersect" in r["name"] else "#FF9800"
        ax.scatter(r["tokens"], r["total_ms"], s=100, color=color, zorder=5)
        ax.annotate(r["name"].split("(")[0].strip(), (r["tokens"], r["total_ms"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Tokens")
    ax.set_ylabel("Total Latency (ms)")
    ax.set_title("Latency vs Token Count")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "e2e_benchmark.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved: {args.output_dir}/e2e_benchmark.png")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Config':<25} {'Tokens':>8} {'Filter':>10} {'SigLIP':>10} {'Total':>10} {'Speedup':>8}")
    print(f"{'='*80}")
    baseline_total = results[0]["total_ms"]
    for r in results:
        speedup = baseline_total / r["total_ms"] if r["total_ms"] > 0 else 0
        print(f"  {r['name']:<23} {r['tokens']:>6} {r['filter_ms']:>8.1f}ms "
              f"{r['siglip_ms']:>8.1f}ms {r['total_ms']:>8.1f}ms {speedup:>6.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="assets/example_input.mp4")
    parser.add_argument("--query", default="person")
    parser.add_argument("--ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--head_type", default="bighead")
    parser.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_warmup", type=int, default=3)
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--output_dir", default="results/e2e_benchmark")
    args = parser.parse_args()
    main(args)
