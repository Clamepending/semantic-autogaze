"""
Analyze the interaction between AutoGaze gaze selection and semantic filtering.

Simulates different operating points:
  1. AutoGaze only: keep X% via gaze (reconstruction-based)
  2. Semantic only: keep X% via semantic head (text-query-based)
  3. Combined: AutoGaze keeps Y%, then semantic keeps Z% of those
  4. Oracle: what's the theoretical best?

Uses cached predictions to compute how much information is retained
at various combined budgets.

Usage:
  python3 -m semantic_autogaze.analyze_combined_filtering \
    --hidden_dir results/distill/hidden_cache \
    --clipseg_dir results/distill/clipseg_cache \
    --ckpt results/distill_bighead/best_bighead_student.pt \
    --head_type bighead \
    --output_dir results/combined_analysis
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


def simulate_gaze_selection(n_patches, gaze_ratio):
    """
    Simulate AutoGaze gaze selection.
    Since we don't have actual gaze outputs cached, we approximate:
    AutoGaze tends to select patches that are "interesting" (high variance,
    edges, motion). We simulate this as random selection with slight bias
    toward patches that CLIPSeg also considers positive.

    For a fair analysis, we model gaze as RANDOM (worst case) since
    we can't run AutoGaze without the full model.
    """
    k = max(1, int(gaze_ratio * n_patches))
    idx = np.random.choice(n_patches, k, replace=False)
    mask = np.zeros(n_patches, dtype=bool)
    mask[idx] = True
    return mask


def analyze_combined(all_preds, all_targets, output_dir):
    """Analyze combined gaze + semantic filtering at various budgets."""
    B, N = all_preds.shape
    gt_binary = all_targets > 0.5

    # Operating points
    gaze_ratios = [0.3, 0.5, 0.7, 0.9, 1.0]  # AutoGaze keeps this fraction
    semantic_ratios = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # Semantic keeps this fraction of gazed
    n_trials = 10  # Average over random gaze simulations

    results = {}

    for gaze_r in gaze_ratios:
        for sem_r in semantic_ratios:
            effective_ratio = gaze_r * sem_r
            trial_recalls = []
            trial_precisions = []

            for trial in range(n_trials):
                batch_recalls = []
                batch_precisions = []

                for b in range(B):
                    # Step 1: Simulate gaze selection
                    if gaze_r >= 1.0:
                        gaze_mask = np.ones(N, dtype=bool)
                    else:
                        gaze_mask = simulate_gaze_selection(N, gaze_r)

                    # Step 2: From gazed patches, keep top-k by semantic score
                    gazed_indices = np.where(gaze_mask)[0]
                    gazed_scores = all_preds[b, gazed_indices]
                    k = max(1, int(sem_r * len(gazed_indices)))
                    topk_local = np.argsort(gazed_scores)[-k:]
                    kept_indices = gazed_indices[topk_local]

                    kept_mask = np.zeros(N, dtype=bool)
                    kept_mask[kept_indices] = True

                    # Measure recall and precision
                    tp = (kept_mask & gt_binary[b]).sum()
                    fn = (~kept_mask & gt_binary[b]).sum()
                    fp = (kept_mask & ~gt_binary[b]).sum()

                    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    batch_recalls.append(recall)
                    batch_precisions.append(precision)

                trial_recalls.append(np.mean(batch_recalls))
                trial_precisions.append(np.mean(batch_precisions))

            key = f"gaze_{gaze_r}_sem_{sem_r}"
            results[key] = {
                "gaze_ratio": gaze_r,
                "semantic_ratio": sem_r,
                "effective_ratio": effective_ratio,
                "effective_pct": effective_ratio * 100,
                "recall_mean": np.mean(trial_recalls),
                "recall_std": np.std(trial_recalls),
                "precision_mean": np.mean(trial_precisions),
                "precision_std": np.std(trial_precisions),
                "patches_kept": int(effective_ratio * N),
            }

    # Also compute semantic-only baseline (gaze_ratio=1.0)
    semantic_only = {k: v for k, v in results.items() if v["gaze_ratio"] == 1.0}

    return results, semantic_only


def plot_combined_analysis(results, output_dir):
    """Plot combined filtering analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Group by gaze ratio
    gaze_ratios = sorted(set(r["gaze_ratio"] for r in results.values()))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(gaze_ratios)))

    # Plot 1: Effective ratio vs recall
    ax = axes[0]
    for gaze_r, color in zip(gaze_ratios, colors):
        entries = sorted(
            [r for r in results.values() if r["gaze_ratio"] == gaze_r],
            key=lambda x: x["effective_ratio"]
        )
        x = [e["effective_pct"] for e in entries]
        y = [e["recall_mean"] for e in entries]
        yerr = [e["recall_std"] for e in entries]

        label = f"Gaze={gaze_r*100:.0f}%" if gaze_r < 1.0 else "Semantic only"
        ls = "-" if gaze_r < 1.0 else "--"
        ax.errorbar(x, y, yerr=yerr, fmt="o-", color=color, lw=2, label=label,
                    capsize=3, markersize=5, linestyle=ls)

    ax.set_xlabel("Effective % Patches Kept")
    ax.set_ylabel("Recall of Relevant Patches")
    ax.set_title("Combined Gaze + Semantic Filtering")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 105])
    ax.set_ylim([0, 1.05])

    # Plot 2: Heatmap of recall at different gaze/semantic combinations
    ax = axes[1]
    gaze_vals = sorted(set(r["gaze_ratio"] for r in results.values()))
    sem_vals = sorted(set(r["semantic_ratio"] for r in results.values()))

    recall_grid = np.zeros((len(gaze_vals), len(sem_vals)))
    for i, gaze_r in enumerate(gaze_vals):
        for j, sem_r in enumerate(sem_vals):
            key = f"gaze_{gaze_r}_sem_{sem_r}"
            if key in results:
                recall_grid[i, j] = results[key]["recall_mean"]

    im = ax.imshow(recall_grid, cmap="YlGn", aspect="auto", vmin=0, vmax=1,
                   origin="lower")
    ax.set_xticks(range(len(sem_vals)))
    ax.set_xticklabels([f"{s*100:.0f}%" for s in sem_vals])
    ax.set_yticks(range(len(gaze_vals)))
    ax.set_yticklabels([f"{g*100:.0f}%" for g in gaze_vals])
    ax.set_xlabel("Semantic Keep Ratio")
    ax.set_ylabel("Gaze Keep Ratio")
    ax.set_title("Recall Heatmap")

    # Annotate cells
    for i in range(len(gaze_vals)):
        for j in range(len(sem_vals)):
            val = recall_grid[i, j]
            eff = gaze_vals[i] * sem_vals[j] * 100
            text_color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}\n({eff:.0f}%)",
                    ha="center", va="center", fontsize=8, color=text_color)

    plt.colorbar(im, ax=ax, label="Recall")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "combined_filtering.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/combined_filtering.png")


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
    print("\nAnalyzing combined gaze + semantic filtering...")
    results, semantic_only = analyze_combined(all_preds, all_targets, args.output_dir)

    # Print summary
    print(f"\n{'Gaze%':<8} {'Sem%':<8} {'Eff%':<8} {'Recall':<10} {'Precision':<10}")
    print("-" * 50)
    for key in sorted(results.keys()):
        r = results[key]
        print(f"  {r['gaze_ratio']*100:<6.0f} {r['semantic_ratio']*100:<6.0f} "
              f"{r['effective_pct']:<6.1f} {r['recall_mean']:<10.4f} {r['precision_mean']:<10.4f}")

    # Key insight: compare gaze(50%)+semantic(40%) vs semantic-only(20%)
    # Both keep ~20% of patches — which one retains more semantic info?
    print("\n=== KEY COMPARISON: 20% effective budget ===")
    for key, r in sorted(results.items()):
        if abs(r["effective_pct"] - 20) < 5:
            print(f"  Gaze={r['gaze_ratio']*100:.0f}% → Sem={r['semantic_ratio']*100:.0f}%: "
                  f"Recall={r['recall_mean']:.4f} (eff={r['effective_pct']:.1f}%)")

    # Plot
    print("\nGenerating plots...")
    plot_combined_analysis(results, args.output_dir)

    # Save
    with open(os.path.join(args.output_dir, "combined_results.json"), "w") as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2)
    print(f"Saved results to {args.output_dir}/combined_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--head_type", choices=["small", "bighead"], default="bighead")
    parser.add_argument("--output_dir", default="results/combined_analysis")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--expanded_dim", type=int, default=384)
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--n_attn_layers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args)
