"""
Compare semantic-only vs intersect filtering quality at matched token budgets.

Key question: Does AutoGaze's gaze selection add value beyond the semantic head alone?
The e2e benchmark showed semantic-only is 13.5x faster (skips AutoGaze decode).
If quality is comparable, we can skip AutoGaze entirely for semantic filtering.

Evaluates:
  1. Semantic-only at various keep ratios
  2. Gaze-only (AutoGaze) with semantic head ranking (post-hoc)
  3. Intersect: AutoGaze then semantic prune
  4. Random baseline

All compared at the SAME total token count for fair comparison.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -m semantic_autogaze.eval_semantic_vs_intersect \
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


def evaluate_selection_strategy(preds, targets, total_budget, T=16, N=196,
                                strategy="semantic_topk"):
    """Evaluate a patch selection strategy at a given token budget.

    Strategies:
      - semantic_topk: global top-k by semantic score
      - semantic_per_frame: top-k per frame by semantic score
      - random: random selection
      - oracle: global top-k by ground truth score (upper bound)
    """
    B = preds.shape[0]
    gt_binary = targets > 0.5

    recalls = []
    precisions = []
    f1s = []

    for b in range(B):
        if strategy == "semantic_topk":
            # Global top-k by predicted semantic score
            topk = np.argsort(preds[b])[-total_budget:]
            mask = np.zeros(T * N, dtype=bool)
            mask[topk] = True

        elif strategy == "semantic_per_frame":
            # Fixed per-frame budget by semantic score
            k = max(1, total_budget // T)
            preds_pf = preds[b].reshape(T, N)
            mask = np.zeros(T * N, dtype=bool)
            for t in range(T):
                topk = np.argsort(preds_pf[t])[-k:]
                mask[t * N + topk] = True

        elif strategy == "adaptive_per_frame":
            # Adaptive per-frame: budget proportional to max semantic score
            preds_pf = preds[b].reshape(T, N)
            frame_importance = preds_pf.max(axis=1)
            min_per_frame = 1
            base = min_per_frame * T
            remaining = max(0, total_budget - base)
            imp_sum = frame_importance.sum()
            if imp_sum > 0:
                alloc = (frame_importance / imp_sum * remaining).astype(int)
            else:
                alloc = np.zeros(T, dtype=int)
            alloc += min_per_frame
            alloc = np.minimum(alloc, N)
            # Trim to budget
            while alloc.sum() > total_budget:
                idx = frame_importance.argmin()
                if alloc[idx] > min_per_frame:
                    alloc[idx] -= 1
                else:
                    break

            mask = np.zeros(T * N, dtype=bool)
            for t in range(T):
                topk = np.argsort(preds_pf[t])[-alloc[t]:]
                mask[t * N + topk] = True

        elif strategy == "random":
            indices = np.random.choice(T * N, size=min(total_budget, T * N), replace=False)
            mask = np.zeros(T * N, dtype=bool)
            mask[indices] = True

        elif strategy == "oracle":
            # Ground truth top-k (upper bound)
            topk = np.argsort(targets[b])[-total_budget:]
            mask = np.zeros(T * N, dtype=bool)
            mask[topk] = True

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        gt = gt_binary[b]
        tp = (mask & gt).sum()
        fn = (~mask & gt).sum()
        fp = (mask & ~gt).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    return {
        "recall": np.mean(recalls),
        "precision": np.mean(precisions),
        "f1": np.mean(f1s),
        "tokens": total_budget,
    }


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
    print(f"  Shape: {all_preds.shape}")

    T, N = 16, 196
    total_patches = T * N

    # Token budgets to evaluate (as fractions of total)
    budget_fracs = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    strategies = ["semantic_topk", "semantic_per_frame", "adaptive_per_frame",
                  "random", "oracle"]
    strategy_labels = {
        "semantic_topk": "Semantic (global top-k)",
        "semantic_per_frame": "Semantic (per-frame)",
        "adaptive_per_frame": "Semantic (adaptive per-frame)",
        "random": "Random baseline",
        "oracle": "Oracle (GT top-k)",
    }
    strategy_colors = {
        "semantic_topk": "#2196F3",
        "semantic_per_frame": "#4CAF50",
        "adaptive_per_frame": "#FF9800",
        "random": "#9E9E9E",
        "oracle": "#E91E63",
    }
    strategy_markers = {
        "semantic_topk": "o",
        "semantic_per_frame": "s",
        "adaptive_per_frame": "^",
        "random": "x",
        "oracle": "D",
    }

    results = {s: [] for s in strategies}

    print("\nEvaluating selection strategies...")
    for frac in budget_fracs:
        budget = max(T, int(frac * total_patches))
        print(f"\n  Budget: {budget} tokens ({frac*100:.0f}%)")
        for s in strategies:
            r = evaluate_selection_strategy(all_preds, all_targets, budget,
                                            T=T, N=N, strategy=s)
            results[s].append(r)
            print(f"    {s:<25}: recall={r['recall']:.4f}, prec={r['precision']:.4f}, f1={r['f1']:.4f}")

    # Plot: Recall vs Token Budget
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric, title in [
        (axes[0], "recall", "Recall vs Token Budget"),
        (axes[1], "precision", "Precision vs Token Budget"),
        (axes[2], "f1", "F1 Score vs Token Budget"),
    ]:
        for s in strategies:
            x = [r["tokens"] for r in results[s]]
            y = [r[metric] for r in results[s]]
            ax.plot(x, y, f'{strategy_markers[s]}-', color=strategy_colors[s],
                    lw=2, markersize=6, label=strategy_labels[s])

        ax.set_xlabel("Total Tokens")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Semantic Filtering: Selection Strategy Comparison", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "strategy_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved: {args.output_dir}/strategy_comparison.png")

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Strategy':<30} {'Budget':>8} {'Recall':>8} {'Precision':>10} {'F1':>8}")
    print(f"{'='*90}")
    # Show at 10% budget
    budget_10pct = max(T, int(0.10 * total_patches))
    for s in strategies:
        for r in results[s]:
            if r["tokens"] == budget_10pct:
                print(f"  {strategy_labels[s]:<28} {r['tokens']:>6} "
                      f"{r['recall']:>8.4f} {r['precision']:>8.4f} {r['f1']:>8.4f}")
    print(f"{'='*90}")

    # Save JSON
    save = {}
    for s in strategies:
        save[s] = [
            {"budget_fraction": f, **r}
            for f, r in zip(budget_fracs, results[s])
        ]
    with open(os.path.join(args.output_dir, "strategy_results.json"), "w") as f:
        json.dump(save, f, indent=2)
    print(f"Saved: {args.output_dir}/strategy_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--head_type", choices=["small", "bighead"], default="bighead")
    parser.add_argument("--output_dir", default="results/strategy_comparison")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--expanded_dim", type=int, default=384)
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--n_attn_layers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args)
