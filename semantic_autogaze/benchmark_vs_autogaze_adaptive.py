"""
Honest comparison: AutoGaze in its ADAPTIVE mode vs AutoGaze + semantic filter.

The earlier `benchmark_speedup_regimes.py` used fixed `gazing_ratio=0.75`,
which is not how AutoGaze is normally deployed. In practice AutoGaze uses
`task_loss_requirement=0.7` to *adaptively* drop tokens — on easy/static
videos it may already keep <20%, in which case semantic filtering adds
little (or no) extra speedup.

This script measures, on real videos:
  1. How many tokens AutoGaze adaptively keeps, per video.
  2. Latency of (a) AutoGaze-adaptive + SigLIP vs (b) AutoGaze-adaptive +
     semantic head + SigLIP with Intersect at 50%/10%, and Semantic-only.
  3. Distribution of incremental speedup by video complexity.

Bins videos by AutoGaze-adaptive token count (low/mid/high keep) and reports
speedup per bin — the real answer to "when does semantic filtering actually
help the most".

Usage:
  CUDA_VISIBLE_DEVICES=5 python3 -m semantic_autogaze.benchmark_vs_autogaze_adaptive \
    --device cuda:0 --n_videos 40
"""

import os, glob, time, json, random, argparse
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


def cuda_time(fn, n_warmup=2, n_runs=5):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record(); fn(); t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))
    return float(np.mean(times))


def get_clip_text_embedding(text, clip_model, tokenizer, device):
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        feat = clip_model.encode_text(tokens)
        feat = F.normalize(feat, dim=-1)
    return feat


def main(args):
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading AutoGaze + head + SigLIP + CLIP ...")
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )
    autogaze_transform = AutoGazeImageProcessor.from_pretrained(args.autogaze_model)
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()

    from autogaze.vision_encoders.siglip import SiglipVisionModel
    siglip_model = SiglipVisionModel.from_pretrained(
        "google/siglip2-base-patch16-224",
        scales=wrapper.autogaze.config.scales,
        attn_implementation="sdpa",
    ).to(device).eval()
    from transformers import AutoImageProcessor
    siglip_tf = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")

    videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    random.shuffle(videos)
    videos = videos[: args.n_videos]
    print(f"Evaluating on {len(videos)} videos")

    # For each video, use a generic query — we care about latency/tokens, not semantic fit here
    query_emb = get_clip_text_embedding(args.query, clip_model, clip_tok, device)

    per_video = []
    for vi, vp in enumerate(videos):
        try:
            c = av.open(vp)
            raw = read_video_pyav(container=c, indices=list(range(16)))
            c.close()
            if raw.shape[0] < 16: continue
            v_ag = transform_video_for_pytorch(raw, autogaze_transform)[None].to(device)
            v_sg = transform_video_for_pytorch(raw, siglip_tf)[None].to(device)
        except Exception as e:
            print(f"  skip {os.path.basename(vp)}: {e}")
            continue

        row = {"video": os.path.basename(vp)}

        # (A) AutoGaze ADAPTIVE (realistic baseline) — gazing_ratio caps, task_loss_requirement adapts
        def run_gaze_only():
            return wrapper.forward(v_ag, query_emb, mode="gaze_only",
                                   gazing_ratio=args.gazing_ratio,
                                   task_loss_requirement=args.task_loss_req,
                                   semantic_keep_ratio=1.0)
        info_gaze = run_gaze_only()
        n_gaze = int((~info_gaze["if_padded_gazing"]).sum().item())
        row["tokens_autogaze_adaptive"] = n_gaze
        row["tokens_total"] = int(info_gaze["if_padded_gazing"].numel())

        # Component A timings: AutoGaze forward + SigLIP at n_gaze tokens
        t_ag = cuda_time(run_gaze_only)
        t_sg_baseline = cuda_time(lambda: siglip_model(v_sg, gazing_info=info_gaze))
        row["lat_autogaze_ms"] = t_ag
        row["lat_siglip_baseline_ms"] = t_sg_baseline
        row["lat_total_baseline_ms"] = t_ag + t_sg_baseline

        # (B) Intersect 50% of gazed patches
        def run_intersect(kr):
            return wrapper.forward(v_ag, query_emb, mode="intersect",
                                   gazing_ratio=args.gazing_ratio,
                                   task_loss_requirement=args.task_loss_req,
                                   semantic_keep_ratio=kr)
        info_int50 = run_intersect(0.5)
        n_int50 = int((~info_int50["if_padded_gazing"]).sum().item())
        row["tokens_intersect_50"] = n_int50
        t_int50 = cuda_time(lambda: run_intersect(0.5))
        t_sg_int50 = cuda_time(lambda: siglip_model(v_sg, gazing_info=info_int50))
        row["lat_intersect_50_ms"] = t_int50
        row["lat_siglip_int50_ms"] = t_sg_int50
        row["lat_total_int50_ms"] = t_int50 + t_sg_int50

        # (C) Intersect 10%
        info_int10 = run_intersect(0.1)
        n_int10 = int((~info_int10["if_padded_gazing"]).sum().item())
        row["tokens_intersect_10"] = n_int10
        t_int10 = cuda_time(lambda: run_intersect(0.1))
        t_sg_int10 = cuda_time(lambda: siglip_model(v_sg, gazing_info=info_int10))
        row["lat_intersect_10_ms"] = t_int10
        row["lat_siglip_int10_ms"] = t_sg_int10
        row["lat_total_int10_ms"] = t_int10 + t_sg_int10

        per_video.append(row)
        if (vi + 1) % 5 == 0 or vi == 0:
            print(f"  [{vi+1:>3}/{len(videos)}] {row['video'][:25]:<25} "
                  f"gaze={n_gaze:>4} ({100*n_gaze/row['tokens_total']:.0f}%), "
                  f"int50={n_int50:>4}, int10={n_int10:>4}, "
                  f"baseline={row['lat_total_baseline_ms']:.0f}ms, "
                  f"int50={row['lat_total_int50_ms']:.0f}ms "
                  f"(speedup {row['lat_total_baseline_ms']/row['lat_total_int50_ms']:.2f}x)")

    if not per_video:
        print("No videos processed.")
        return

    # ========== Aggregate by AutoGaze keep ratio bins ==========
    bins = [(0, 0.25, "easy (AG<25%)"), (0.25, 0.50, "mid (25-50%)"),
            (0.50, 0.75, "dense (50-75%)"), (0.75, 1.01, "near-full (≥75%)")]

    def frac(r): return r["tokens_autogaze_adaptive"] / r["tokens_total"]
    def speedup(r, key): return r["lat_total_baseline_ms"] / max(r[key], 1e-9)

    print("\n" + "=" * 90)
    print("SPEEDUP vs AutoGaze-adaptive baseline, grouped by AutoGaze keep ratio")
    print("=" * 90)
    print(f"{'bin':<22}{'n':>4}{'ag_keep':>10}{'int50_sp':>12}{'int10_sp':>12}{'baseline_ms':>14}{'int50_ms':>12}")
    bin_stats = []
    for lo, hi, name in bins:
        rs = [r for r in per_video if lo <= frac(r) < hi]
        if not rs:
            print(f"{name:<22}{0:>4}")
            continue
        ag = np.mean([frac(r) for r in rs])
        sp50 = np.mean([speedup(r, "lat_total_int50_ms") for r in rs])
        sp10 = np.mean([speedup(r, "lat_total_int10_ms") for r in rs])
        base = np.mean([r["lat_total_baseline_ms"] for r in rs])
        i50 = np.mean([r["lat_total_int50_ms"] for r in rs])
        print(f"{name:<22}{len(rs):>4}{ag:>10.2f}{sp50:>12.2f}{sp10:>12.2f}{base:>14.1f}{i50:>12.1f}")
        bin_stats.append({
            "bin": name, "n": len(rs), "mean_autogaze_keep": float(ag),
            "mean_speedup_int50": float(sp50), "mean_speedup_int10": float(sp10),
            "mean_baseline_ms": float(base), "mean_int50_ms": float(i50),
        })

    # Save full results
    with open(os.path.join(args.output_dir, "vs_autogaze_adaptive.json"), "w") as f:
        json.dump({"per_video": per_video, "bins": bin_stats,
                   "config": {"gazing_ratio": args.gazing_ratio,
                              "task_loss_req": args.task_loss_req,
                              "query": args.query, "n_videos": len(per_video)}}, f, indent=2)
    print(f"\nSaved: {args.output_dir}/vs_autogaze_adaptive.json")

    # Histogram + scatter
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    keeps = [frac(r) for r in per_video]
    ax.hist(keeps, bins=20, color="tab:gray", alpha=0.8)
    ax.set_xlabel("AutoGaze keep ratio (adaptive)")
    ax.set_ylabel("# videos")
    ax.set_title(f"AutoGaze adaptive token keep (n={len(per_video)})")
    ax.axvline(np.mean(keeps), color="red", ls="--",
               label=f"mean={np.mean(keeps):.2f}")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    sp50s = [speedup(r, "lat_total_int50_ms") for r in per_video]
    sp10s = [speedup(r, "lat_total_int10_ms") for r in per_video]
    ax.scatter(keeps, sp50s, color="tab:blue", alpha=0.7, label="Intersect 50%")
    ax.scatter(keeps, sp10s, color="tab:green", alpha=0.7, label="Intersect 10%")
    ax.axhline(1.0, color="black", ls=":")
    ax.set_xlabel("AutoGaze keep ratio")
    ax.set_ylabel("Speedup (baseline / semantic)")
    ax.set_title("Speedup vs video complexity")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    labels = [b["bin"] for b in bin_stats]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, [b["mean_speedup_int50"] for b in bin_stats], w,
           color="tab:blue", label="Intersect 50%")
    ax.bar(x + w/2, [b["mean_speedup_int10"] for b in bin_stats], w,
           color="tab:green", label="Intersect 10%")
    ax.axhline(1.0, color="black", ls=":")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Mean speedup")
    ax.set_title("Speedup by video complexity bin")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "vs_autogaze_adaptive.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {args.output_dir}/vs_autogaze_adaptive.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", default="data")
    p.add_argument("--n_videos", type=int, default=40)
    p.add_argument("--gazing_ratio", type=float, default=0.75)
    p.add_argument("--task_loss_req", type=float, default=0.7)
    p.add_argument("--query", default="the main subject")
    p.add_argument("--ckpt", default="results/bighead_warmrestart/best_bighead_student.pt")
    p.add_argument("--head_type", default="bighead")
    p.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/vs_autogaze_adaptive")
    args = p.parse_args()
    main(args)
