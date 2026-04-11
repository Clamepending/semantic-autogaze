"""
Temporal analysis of semantic filtering across video frames.

Analyzes how semantic relevance scores change across frames:
  1. Score trajectory: how does mean/max score evolve per frame?
  2. Spatial stability: how much do selected patches shift between frames?
  3. Per-frame budget: does uniform keep_ratio waste budget on empty frames?

This informs whether adaptive per-frame budgets could improve efficiency.

Usage:
  python3 -m semantic_autogaze.temporal_analysis \
    --hidden_dir results/distill/hidden_cache \
    --clipseg_dir results/distill/clipseg_cache \
    --ckpt results/distill_bighead/best_bighead_student.pt \
    --output_dir results/temporal_analysis
"""

import os
import glob
import hashlib
import random
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from semantic_autogaze.eval_filtering import EvalDataset, load_head


def analyze_temporal_patterns(all_preds, all_targets, grid_size=14, num_frames=16):
    """Analyze temporal patterns in semantic scores."""
    B, TN = all_preds.shape
    N = grid_size * grid_size  # 196
    T = TN // N
    assert T == num_frames

    preds_per_frame = all_preds.reshape(B, T, N)
    targets_per_frame = all_targets.reshape(B, T, N)

    results = {
        "per_frame_mean_score": [],
        "per_frame_max_score": [],
        "per_frame_positive_fraction": [],
        "consecutive_iou": [],
        "consecutive_correlation": [],
        "score_variance_across_frames": [],
    }

    # Per-frame statistics
    for t in range(T):
        frame_preds = preds_per_frame[:, t]  # (B, N)
        frame_targets = targets_per_frame[:, t]

        results["per_frame_mean_score"].append(frame_preds.mean())
        results["per_frame_max_score"].append(frame_preds.max(axis=1).mean())
        results["per_frame_positive_fraction"].append(
            (frame_targets > 0.5).mean()
        )

    # Consecutive frame analysis
    for keep_ratio in [0.1, 0.2, 0.3, 0.5]:
        k = max(1, int(keep_ratio * N))
        ious = []
        corrs = []

        for b in range(min(B, 200)):  # limit for speed
            for t in range(T - 1):
                scores_t = preds_per_frame[b, t]
                scores_t1 = preds_per_frame[b, t + 1]

                # Top-k overlap (IoU)
                topk_t = set(np.argsort(scores_t)[-k:])
                topk_t1 = set(np.argsort(scores_t1)[-k:])
                inter = len(topk_t & topk_t1)
                union = len(topk_t | topk_t1)
                ious.append(inter / union if union > 0 else 1.0)

                # Score correlation
                corr = np.corrcoef(scores_t, scores_t1)[0, 1]
                corrs.append(corr if not np.isnan(corr) else 0)

        results["consecutive_iou"].append({
            "keep_ratio": keep_ratio,
            "mean_iou": np.mean(ious),
            "std_iou": np.std(ious),
        })
        results["consecutive_correlation"].append({
            "keep_ratio": keep_ratio,
            "mean_corr": np.mean(corrs),
            "std_corr": np.std(corrs),
        })

    # Score variance across frames (per sample)
    frame_means = preds_per_frame.mean(axis=2)  # (B, T)
    results["score_variance_across_frames"] = {
        "mean_std": frame_means.std(axis=1).mean(),
        "max_std": frame_means.std(axis=1).max(),
    }

    # Adaptive budget analysis: would some frames benefit from more/fewer tokens?
    pos_fracs = (targets_per_frame > 0.5).mean(axis=2)  # (B, T)
    results["adaptive_budget"] = {
        "mean_pos_frac_per_frame": pos_fracs.mean(axis=0).tolist(),
        "std_pos_frac_per_frame": pos_fracs.std(axis=0).tolist(),
        "empty_frames_pct": (pos_fracs == 0).mean() * 100,
        "sparse_frames_pct": (pos_fracs < 0.01).mean() * 100,
        "dense_frames_pct": (pos_fracs > 0.1).mean() * 100,
    }

    return results


def plot_temporal_analysis(results, output_dir):
    """Generate temporal analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    T = len(results["per_frame_mean_score"])
    frames = list(range(T))

    # 1. Score trajectory
    ax = axes[0, 0]
    ax.plot(frames, results["per_frame_mean_score"], 'o-', color="#2196F3",
            lw=2, label="Mean score")
    ax.plot(frames, results["per_frame_max_score"], 's-', color="#FF9800",
            lw=2, label="Mean max score")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Semantic Score")
    ax.set_title("Score Trajectory Across Frames")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Positive fraction per frame
    ax = axes[0, 1]
    ax.plot(frames, results["per_frame_positive_fraction"], 'o-',
            color="#4CAF50", lw=2)
    ax.fill_between(frames, results["per_frame_positive_fraction"],
                    alpha=0.2, color="#4CAF50")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Fraction of Positive Patches")
    ax.set_title("GT Positive Patch Density per Frame")
    ax.grid(True, alpha=0.3)

    # 3. Consecutive frame IoU at different keep ratios
    ax = axes[1, 0]
    krs = [r["keep_ratio"] for r in results["consecutive_iou"]]
    ious = [r["mean_iou"] for r in results["consecutive_iou"]]
    iou_stds = [r["std_iou"] for r in results["consecutive_iou"]]
    corrs = [r["mean_corr"] for r in results["consecutive_correlation"]]

    x = range(len(krs))
    bars = ax.bar(x, ious, yerr=iou_stds, capsize=4, color="#42A5F5",
                  alpha=0.8, edgecolor="white", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{kr*100:.0f}%" for kr in krs])
    ax.set_xlabel("Keep Ratio")
    ax.set_ylabel("Mean Consecutive Frame IoU")
    ax.set_title("Temporal Stability of Selected Patches")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate with correlation
    for i, (bar, corr) in enumerate(zip(bars, corrs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + iou_stds[i] + 0.02,
                f"r={corr:.2f}", ha="center", fontsize=9)

    # 4. Adaptive budget analysis
    ax = axes[1, 1]
    budget = results["adaptive_budget"]
    pos_fracs = budget["mean_pos_frac_per_frame"]
    std_fracs = budget["std_pos_frac_per_frame"]

    ax.bar(frames, pos_fracs, yerr=std_fracs, capsize=2, color="#66BB6A",
           alpha=0.8, edgecolor="white", linewidth=1)
    ax.axhline(y=np.mean(pos_fracs), color="#D32F2F", linestyle="--",
               lw=1.5, label=f"Mean: {np.mean(pos_fracs):.3f}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean Positive Fraction")
    ax.set_title(f"Per-Frame Relevance Density\n"
                 f"(Empty: {budget['empty_frames_pct']:.1f}%, "
                 f"Sparse: {budget['sparse_frames_pct']:.1f}%, "
                 f"Dense: {budget['dense_frames_pct']:.1f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Temporal Analysis of Semantic Filtering", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "temporal_analysis.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir}/temporal_analysis.png")


def main(args):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load data
    clipseg_files = sorted(glob.glob(os.path.join(args.clipseg_dir, "*_clipseg_clip*.pt")))
    random.shuffle(clipseg_files)
    split = int(0.9 * len(clipseg_files))
    val_files = clipseg_files[split:]

    dataset = EvalDataset(val_files, args.hidden_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    head = load_head(args, device)

    # Run inference
    print("Running inference...")
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            hidden = batch["hidden_states"].to(device)
            query = batch["text_embedding"].to(device)
            target = batch["target_scores"].to(device)

            pred_logits = head(hidden, query)
            pred_probs = torch.sigmoid(pred_logits).cpu().numpy()
            gt_probs = torch.sigmoid(target).cpu().numpy()

            all_preds.append(pred_probs)
            all_targets.append(gt_probs)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    print(f"  Predictions: {all_preds.shape}")

    # Analyze
    print("\nAnalyzing temporal patterns...")
    results = analyze_temporal_patterns(all_preds, all_targets)

    # Print summary
    print(f"\n{'='*50}")
    print("TEMPORAL ANALYSIS SUMMARY")
    print(f"{'='*50}")

    print("\nConsecutive frame stability:")
    for r in results["consecutive_iou"]:
        print(f"  Keep ratio {r['keep_ratio']*100:.0f}%: "
              f"IoU={r['mean_iou']:.3f} +/- {r['std_iou']:.3f}")

    print(f"\nScore variance across frames: "
          f"mean_std={results['score_variance_across_frames']['mean_std']:.4f}")

    budget = results["adaptive_budget"]
    print(f"\nAdaptive budget insights:")
    print(f"  Empty frames: {budget['empty_frames_pct']:.1f}%")
    print(f"  Sparse (<1% positive): {budget['sparse_frames_pct']:.1f}%")
    print(f"  Dense (>10% positive): {budget['dense_frames_pct']:.1f}%")

    # Plot
    print("\nGenerating plots...")
    plot_temporal_analysis(results, args.output_dir)

    # Save
    # Convert numpy types for JSON
    save_results = {
        "per_frame_mean_score": [float(x) for x in results["per_frame_mean_score"]],
        "per_frame_max_score": [float(x) for x in results["per_frame_max_score"]],
        "per_frame_positive_fraction": [float(x) for x in results["per_frame_positive_fraction"]],
        "consecutive_iou": results["consecutive_iou"],
        "consecutive_correlation": results["consecutive_correlation"],
        "score_variance_across_frames": {
            k: float(v) for k, v in results["score_variance_across_frames"].items()
        },
        "adaptive_budget": {
            k: [float(x) for x in v] if isinstance(v, list) else float(v)
            for k, v in budget.items()
        },
    }
    with open(os.path.join(args.output_dir, "temporal_results.json"), "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"  Saved: {args.output_dir}/temporal_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--head_type", choices=["small", "bighead"], default="bighead")
    parser.add_argument("--output_dir", default="results/temporal_analysis")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--expanded_dim", type=int, default=384)
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--n_attn_layers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args)
