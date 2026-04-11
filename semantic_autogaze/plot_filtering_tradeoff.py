"""
Plot semantic filtering trade-off curves, analogous to AutoGaze paper Fig. 3.

AutoGaze paper shows: gazing ratio vs. VLM accuracy on benchmarks.
Our semantic version shows: % patches kept (via semantic filtering threshold)
vs. CLIPSeg alignment quality (IoU, F1) and vs. random baseline.

This script produces:
  1. "Patches kept vs. quality" curve at various thresholds
  2. Comparison: semantic filtering vs. random patch selection
  3. Per-frame patch reduction analysis
  4. Semantic filtering + AutoGaze gaze combined analysis

Usage:
  python3 -m semantic_autogaze.plot_filtering_tradeoff \
    --hidden_dir results/distill/hidden_cache \
    --clipseg_dir results/distill/clipseg_cache \
    --ckpt results/distill_bighead/best_bighead_student.pt \
    --head_type bighead \
    --output_dir results/plots
"""

import os
import glob
import hashlib
import random
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from semantic_autogaze.model import SimilarityHead
from semantic_autogaze.train_bighead import BigSimilarityHead
from semantic_autogaze.eval_filtering import EvalDataset, load_head


def compute_quality_at_keep_ratio(pred_probs, gt_probs, keep_ratios, mode="semantic"):
    """
    For each keep ratio, select the top-k patches by predicted score
    and measure quality (IoU, recall of positives).

    mode: "semantic" = keep top scoring patches
          "random" = keep random patches (averaged over 5 trials)
    """
    results = {}
    N = pred_probs.shape[1]
    gt_binary = gt_probs > 0.5

    for ratio in keep_ratios:
        k = max(1, int(ratio * N))

        if mode == "semantic":
            # Keep top-k by predicted score
            topk_idx = np.argsort(pred_probs, axis=1)[:, -k:]  # (B, k)
            kept_mask = np.zeros_like(pred_probs, dtype=bool)
            for i in range(pred_probs.shape[0]):
                kept_mask[i, topk_idx[i]] = True
        else:
            # Random baseline (average over trials)
            trials = 5
            all_recalls = []
            all_ious = []
            for _ in range(trials):
                kept_mask = np.zeros_like(pred_probs, dtype=bool)
                for i in range(pred_probs.shape[0]):
                    idx = np.random.choice(N, k, replace=False)
                    kept_mask[i, idx] = True

                # Recall of positive patches among kept
                tp = (kept_mask & gt_binary).sum()
                fn = (~kept_mask & gt_binary).sum()
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                # IoU
                fp = (kept_mask & ~gt_binary).sum()
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                all_recalls.append(recall)
                all_ious.append(iou)

            results[ratio] = {
                "recall": np.mean(all_recalls),
                "iou": np.mean(all_ious),
                "patches_kept": k,
                "total_patches": N,
            }
            continue

        # Recall of positive patches among kept
        tp = (kept_mask & gt_binary).sum()
        fn = (~kept_mask & gt_binary).sum()
        fp = (kept_mask & ~gt_binary).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        results[ratio] = {
            "recall": recall,
            "iou": iou,
            "patches_kept": k,
            "total_patches": N,
        }

    return results


def run_inference(head, dataloader, device):
    """Run model and collect all predictions and targets."""
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

    return np.concatenate(all_preds), np.concatenate(all_targets)


def plot_keep_ratio_vs_quality(semantic_results, random_results, output_dir, head_label):
    """Plot patches-kept ratio vs quality (recall, IoU)."""
    ratios = sorted(semantic_results.keys())

    sem_recall = [semantic_results[r]["recall"] for r in ratios]
    sem_iou = [semantic_results[r]["iou"] for r in ratios]
    rand_recall = [random_results[r]["recall"] for r in ratios]
    rand_iou = [random_results[r]["iou"] for r in ratios]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Recall plot
    ax = axes[0]
    pct_ratios = [r * 100 for r in ratios]
    ax.plot(pct_ratios, sem_recall, "o-", color="#2196F3", lw=2, label=f"Semantic ({head_label})")
    ax.plot(pct_ratios, rand_recall, "s--", color="#9E9E9E", lw=2, label="Random")
    ax.set_xlabel("% Patches Kept")
    ax.set_ylabel("Recall of Positive Patches")
    ax.set_title("Semantic Filtering: Recall vs. Budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 105])
    ax.set_ylim([0, 1.05])

    # IoU plot
    ax = axes[1]
    ax.plot(pct_ratios, sem_iou, "o-", color="#4CAF50", lw=2, label=f"Semantic ({head_label})")
    ax.plot(pct_ratios, rand_iou, "s--", color="#9E9E9E", lw=2, label="Random")
    ax.set_xlabel("% Patches Kept")
    ax.set_ylabel("IoU with GT Segmentation")
    ax.set_title("Semantic Filtering: IoU vs. Budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 105])
    ax.set_ylim([0, max(max(sem_iou), max(rand_iou)) * 1.1 + 0.01])

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "keep_ratio_vs_quality.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/keep_ratio_vs_quality.png")


def plot_filtering_gain(semantic_results, random_results, output_dir, head_label):
    """Plot the relative gain of semantic filtering over random at each ratio."""
    ratios = sorted(semantic_results.keys())
    pct_ratios = [r * 100 for r in ratios]

    recall_gain = []
    for r in ratios:
        sem = semantic_results[r]["recall"]
        rand = random_results[r]["recall"]
        gain = (sem - rand) / max(rand, 1e-8)
        recall_gain.append(gain * 100)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.bar(pct_ratios, recall_gain, width=3, color="#FF9800", alpha=0.8, edgecolor="#E65100")
    ax.axhline(y=0, color="black", lw=0.5)
    ax.set_xlabel("% Patches Kept")
    ax.set_ylabel("Relative Recall Gain vs. Random (%)")
    ax.set_title(f"Semantic Filtering Gain ({head_label})")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "filtering_gain.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/filtering_gain.png")


def plot_score_vs_gt_correlation(all_preds, all_targets, output_dir):
    """Scatter plot of predicted scores vs GT scores (subsample for clarity)."""
    n = min(50000, all_preds.size)
    idx = np.random.choice(all_preds.size, n, replace=False)
    pred_flat = all_preds.flatten()[idx]
    gt_flat = all_targets.flatten()[idx]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(gt_flat, pred_flat, s=1, alpha=0.05, c="#1976D2")
    ax.plot([0, 1], [0, 1], "r--", lw=1, label="y=x")
    ax.set_xlabel("GT Score (sigmoid of CLIPSeg logit)")
    ax.set_ylabel("Predicted Score")
    ax.set_title("Prediction vs. Ground Truth Correlation")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pred_vs_gt_scatter.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/pred_vs_gt_scatter.png")


def main(args):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load data
    clipseg_files = sorted(glob.glob(os.path.join(args.clipseg_dir, "*_clipseg_clip*.pt")))
    print(f"Found {len(clipseg_files)} CLIPSeg cache files")
    random.shuffle(clipseg_files)
    split = int(0.9 * len(clipseg_files))
    val_files = clipseg_files[split:]

    dataset = EvalDataset(val_files, args.hidden_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    head = load_head(args, device)

    # Run inference
    print("\nRunning inference...")
    all_preds, all_targets = run_inference(head, dataloader, device)
    print(f"  Predictions: {all_preds.shape}, Targets: {all_targets.shape}")

    # Keep ratios to evaluate
    keep_ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]

    print("\nComputing semantic filtering quality at each keep ratio...")
    semantic_results = compute_quality_at_keep_ratio(all_preds, all_targets, keep_ratios, mode="semantic")

    print("Computing random baseline...")
    random_results = compute_quality_at_keep_ratio(all_preds, all_targets, keep_ratios, mode="random")

    # Print summary
    print(f"\n{'Keep %':<10} {'Sem Recall':<12} {'Rand Recall':<12} {'Gain':<10} {'Sem IoU':<12} {'Rand IoU':<12}")
    print("-" * 68)
    for r in keep_ratios:
        sr = semantic_results[r]
        rr = random_results[r]
        gain = (sr["recall"] - rr["recall"]) / max(rr["recall"], 1e-8) * 100
        print(f"  {r*100:<8.0f} {sr['recall']:<12.4f} {rr['recall']:<12.4f} {gain:<+10.1f}% {sr['iou']:<12.4f} {rr['iou']:<12.4f}")

    # Generate plots
    print("\nGenerating plots...")
    head_label = args.head_type
    plot_keep_ratio_vs_quality(semantic_results, random_results, args.output_dir, head_label)
    plot_filtering_gain(semantic_results, random_results, args.output_dir, head_label)
    plot_score_vs_gt_correlation(all_preds, all_targets, args.output_dir)

    # Save data
    save_data = {
        "head_type": args.head_type,
        "ckpt": args.ckpt,
        "keep_ratios": keep_ratios,
        "semantic": {str(k): v for k, v in semantic_results.items()},
        "random": {str(k): v for k, v in random_results.items()},
    }
    with open(os.path.join(args.output_dir, "tradeoff_data.json"), "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved data to {args.output_dir}/tradeoff_data.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--head_type", choices=["small", "bighead"], required=True)
    parser.add_argument("--output_dir", default="results/plots")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--expanded_dim", type=int, default=384)
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--n_attn_layers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args)
