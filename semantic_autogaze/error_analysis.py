"""
Error analysis for the semantic filtering head.

Analyzes the gap between semantic head and oracle selection:
  1. False negatives: GT-positive patches the head misses (ranked low)
  2. False positives: GT-negative patches the head selects (ranked high)
  3. Spatial error patterns: where on the grid do errors concentrate?
  4. Confidence calibration: does the head's score correlate with GT quality?

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -m semantic_autogaze.error_analysis \
    --hidden_dir results/distill/hidden_cache \
    --clipseg_dir results/distill/clipseg_cache \
    --ckpt results/distill_bighead/best_bighead_student.pt
"""

import os
import glob
import random
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from semantic_autogaze.eval_filtering import EvalDataset, load_head


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

    # Inference
    print("Running inference...")
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            hidden = batch["hidden_states"].to(device)
            query = batch["text_embedding"].to(device)
            target = batch["target_scores"].to(device)
            pred_logits = head(hidden, query)
            all_preds.append(torch.sigmoid(pred_logits).cpu().numpy())
            all_targets.append(torch.sigmoid(target).cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    B, TN = all_preds.shape
    T, N = 16, 196
    G = 14  # grid size
    print(f"  Shape: {all_preds.shape}, {B} samples")

    budget_frac = 0.10
    budget = max(T, int(budget_frac * TN))

    # Aggregate spatial error maps across all samples
    # For each sample: which spatial positions are false negatives / false positives?
    fn_heatmap = np.zeros((T, G, G))  # false negative spatial map
    fp_heatmap = np.zeros((T, G, G))  # false positive spatial map
    tp_heatmap = np.zeros((T, G, G))  # true positive spatial map
    gt_heatmap = np.zeros((T, G, G))  # ground truth spatial map

    pred_scores_when_fn = []  # pred scores for GT-positive patches that were missed
    pred_scores_when_tp = []  # pred scores for GT-positive patches that were selected
    gt_scores_when_fp = []    # GT scores for patches selected despite being negative

    n_analyzed = 0

    print("Analyzing errors...")
    for b in tqdm(range(B), desc="Samples"):
        p = all_preds[b]
        t = all_targets[b]

        gt_binary = t > 0.5
        if gt_binary.sum() < 1:
            continue  # skip samples with no positive patches

        # Semantic top-k selection
        topk = np.argsort(p)[-budget:]
        selected = np.zeros(TN, dtype=bool)
        selected[topk] = True

        # Error types
        tp = selected & gt_binary           # correctly selected
        fn = ~selected & gt_binary          # missed GT positives
        fp = selected & ~gt_binary          # incorrectly selected negatives

        # Accumulate spatial maps
        for idx in np.where(tp)[0]:
            frame = idx // N
            pos = idx % N
            r, c = pos // G, pos % G
            tp_heatmap[frame, r, c] += 1

        for idx in np.where(fn)[0]:
            frame = idx // N
            pos = idx % N
            r, c = pos // G, pos % G
            fn_heatmap[frame, r, c] += 1

        for idx in np.where(fp)[0]:
            frame = idx // N
            pos = idx % N
            r, c = pos // G, pos % G
            fp_heatmap[frame, r, c] += 1

        for idx in np.where(gt_binary)[0]:
            frame = idx // N
            pos = idx % N
            r, c = pos // G, pos % G
            gt_heatmap[frame, r, c] += 1

        # Collect score distributions
        pred_scores_when_fn.extend(p[fn].tolist())
        pred_scores_when_tp.extend(p[tp].tolist())
        gt_scores_when_fp.extend(t[fp].tolist())

        n_analyzed += 1

    print(f"  Analyzed {n_analyzed} samples with positive patches")

    # Aggregate spatial maps across frames (sum over frames)
    fn_spatial = fn_heatmap.sum(axis=0)
    fp_spatial = fp_heatmap.sum(axis=0)
    tp_spatial = tp_heatmap.sum(axis=0)
    gt_spatial = gt_heatmap.sum(axis=0)

    # Normalize
    total = fn_spatial + tp_spatial + 1e-8
    fn_rate_spatial = fn_spatial / (gt_spatial + 1e-8)  # FN rate per position
    fp_rate_spatial = fp_spatial / (fp_spatial + tp_spatial + 1e-8)  # FP rate per position

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. GT density spatial map
    ax = axes[0, 0]
    im = ax.imshow(gt_spatial / gt_spatial.max(), cmap="Blues", vmin=0, vmax=1)
    ax.set_title("GT Positive Density")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks([])
    ax.set_yticks([])

    # 2. False Negative rate spatial map
    ax = axes[0, 1]
    im = ax.imshow(fn_rate_spatial, cmap="Reds", vmin=0, vmax=1)
    ax.set_title("False Negative Rate\n(missed GT positives)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks([])
    ax.set_yticks([])

    # 3. False Positive rate spatial map
    ax = axes[0, 2]
    im = ax.imshow(fp_rate_spatial, cmap="Oranges", vmin=0, vmax=1)
    ax.set_title("False Positive Rate\n(selected GT negatives)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks([])
    ax.set_yticks([])

    # 4. Pred score distribution for TP vs FN
    ax = axes[1, 0]
    if pred_scores_when_tp:
        ax.hist(pred_scores_when_tp, bins=50, alpha=0.7, color="#4CAF50",
                label=f"TP (n={len(pred_scores_when_tp)})", density=True)
    if pred_scores_when_fn:
        ax.hist(pred_scores_when_fn, bins=50, alpha=0.7, color="#EF5350",
                label=f"FN (n={len(pred_scores_when_fn)})", density=True)
    ax.set_xlabel("Predicted Score")
    ax.set_ylabel("Density")
    ax.set_title("Pred Scores: TP vs FN")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # 5. GT score distribution for FP
    ax = axes[1, 1]
    if gt_scores_when_fp:
        ax.hist(gt_scores_when_fp, bins=50, alpha=0.7, color="#FF9800",
                label=f"FP (n={len(gt_scores_when_fp)})")
    ax.set_xlabel("GT Score of False Positives")
    ax.set_ylabel("Count")
    ax.set_title("GT Scores of Incorrectly Selected Patches")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # 6. Calibration: pred vs actual positive rate
    ax = axes[1, 2]
    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    actual_pos_rates = []
    bin_counts = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (all_preds >= lo) & (all_preds < hi)
        if mask.sum() > 0:
            actual_rate = (all_targets[mask] > 0.5).mean()
            actual_pos_rates.append(actual_rate)
            bin_counts.append(mask.sum())
        else:
            actual_pos_rates.append(0)
            bin_counts.append(0)

    ax.bar(bin_centers, actual_pos_rates, width=0.04, alpha=0.7, color="#42A5F5",
           edgecolor="white")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Predicted Score")
    ax.set_ylabel("Actual Positive Rate (GT > 0.5)")
    ax.set_title("Calibration Plot")
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, max(actual_pos_rates) * 1.2 + 0.01])

    fig.suptitle(f"Error Analysis: Semantic Head at {budget_frac*100:.0f}% Budget ({budget} tokens)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "error_analysis.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {args.output_dir}/error_analysis.png")

    # Summary stats
    print(f"\n{'='*60}")
    print("ERROR ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total samples analyzed: {n_analyzed}")
    print(f"  Budget: {budget} tokens ({budget_frac*100:.0f}%)")
    print(f"  True positives: {len(pred_scores_when_tp)}")
    print(f"  False negatives: {len(pred_scores_when_fn)}")
    print(f"  False positives: {len(gt_scores_when_fp)}")

    if pred_scores_when_tp:
        print(f"\n  TP pred score: mean={np.mean(pred_scores_when_tp):.4f}, "
              f"median={np.median(pred_scores_when_tp):.4f}")
    if pred_scores_when_fn:
        print(f"  FN pred score: mean={np.mean(pred_scores_when_fn):.4f}, "
              f"median={np.median(pred_scores_when_fn):.4f}")
    if gt_scores_when_fp:
        print(f"  FP GT score:   mean={np.mean(gt_scores_when_fp):.4f}, "
              f"median={np.median(gt_scores_when_fp):.4f}")

    # Spatial analysis
    print(f"\n  Spatial FN concentration:")
    fn_flat = fn_rate_spatial.flatten()
    print(f"    Center (7x7): {fn_rate_spatial[3:11, 3:11].mean():.4f}")
    print(f"    Edges:        {(fn_rate_spatial.sum() - fn_rate_spatial[3:11, 3:11].sum()) / max(1, (fn_rate_spatial.size - 64)):.4f}")
    print(f"    Max FN rate position: {np.unravel_index(fn_rate_spatial.argmax(), fn_rate_spatial.shape)}")

    # Save summary
    summary = {
        "n_analyzed": n_analyzed,
        "budget": budget,
        "n_tp": len(pred_scores_when_tp),
        "n_fn": len(pred_scores_when_fn),
        "n_fp": len(gt_scores_when_fp),
        "tp_pred_mean": float(np.mean(pred_scores_when_tp)) if pred_scores_when_tp else 0,
        "fn_pred_mean": float(np.mean(pred_scores_when_fn)) if pred_scores_when_fn else 0,
        "fp_gt_mean": float(np.mean(gt_scores_when_fp)) if gt_scores_when_fp else 0,
    }
    with open(os.path.join(args.output_dir, "error_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.output_dir}/error_analysis.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--head_type", choices=["small", "bighead"], default="bighead")
    parser.add_argument("--output_dir", default="results/error_analysis")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--expanded_dim", type=int, default=384)
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--n_attn_layers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args)
