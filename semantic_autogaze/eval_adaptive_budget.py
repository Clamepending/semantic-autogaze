"""
Evaluate adaptive per-frame budget vs fixed budget for semantic filtering.

Compares three strategies at the same total token budget:
  1. Fixed: same k patches per frame
  2. Adaptive: allocate more tokens to relevant frames
  3. Global top-k: top k across all frames (no per-frame constraint)

Usage:
  python3 -m semantic_autogaze.eval_adaptive_budget \
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


def evaluate_budget_strategy(preds, targets, total_budget, T=16, N=196,
                             strategy="fixed", min_per_frame=1):
    """Evaluate a budget allocation strategy."""
    B = preds.shape[0]
    preds_per_frame = preds.reshape(B, T, N)
    targets_per_frame = targets.reshape(B, T, N)
    gt_binary = targets > 0.5

    recalls = []
    precisions = []

    for b in range(B):
        if strategy == "fixed":
            # Same budget per frame
            k = max(1, total_budget // T)
            mask = np.zeros(T * N, dtype=bool)
            for t in range(T):
                topk = np.argsort(preds_per_frame[b, t])[-k:]
                mask[t * N + topk] = True

        elif strategy == "adaptive":
            # Proportional to max score per frame
            frame_importance = preds_per_frame[b].max(axis=1)
            base = min_per_frame * T
            remaining = max(0, total_budget - base)
            imp_sum = frame_importance.sum()
            if imp_sum > 0:
                alloc = (frame_importance / imp_sum * remaining).astype(int)
            else:
                alloc = np.zeros(T, dtype=int)
            alloc += min_per_frame
            alloc = np.minimum(alloc, N)

            # Adjust to match budget
            while alloc.sum() > total_budget:
                idx = frame_importance.argmin()
                if alloc[idx] > min_per_frame:
                    alloc[idx] -= 1
                else:
                    break

            mask = np.zeros(T * N, dtype=bool)
            for t in range(T):
                k_t = alloc[t]
                topk = np.argsort(preds_per_frame[b, t])[-k_t:]
                mask[t * N + topk] = True

        elif strategy == "global_topk":
            # Global top-k across all frames
            flat_preds = preds[b]
            topk = np.argsort(flat_preds)[-total_budget:]
            mask = np.zeros(T * N, dtype=bool)
            mask[topk] = True

        # Compute recall
        gt = gt_binary[b]
        tp = (mask & gt).sum()
        fn = (~mask & gt).sum()
        fp = (mask & ~gt).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recalls.append(recall)
        precisions.append(precision)

    return {
        "recall": np.mean(recalls),
        "precision": np.mean(precisions),
        "tokens_used": total_budget,
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

    # Evaluate at different budgets
    budget_fractions = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    strategies = ["fixed", "adaptive", "global_topk"]
    strategy_labels = {
        "fixed": "Fixed (uniform per-frame)",
        "adaptive": "Adaptive (proportional to relevance)",
        "global_topk": "Global top-k (no per-frame constraint)",
    }

    results = {s: [] for s in strategies}

    print("\nEvaluating budget strategies...")
    for frac in budget_fractions:
        budget = max(T, int(frac * total_patches))
        print(f"\n  Budget: {budget} tokens ({frac*100:.0f}%)")
        for s in strategies:
            r = evaluate_budget_strategy(all_preds, all_targets, budget,
                                         T=T, N=N, strategy=s)
            results[s].append(r)
            print(f"    {s:<15}: recall={r['recall']:.4f}, precision={r['precision']:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"fixed": "#2196F3", "adaptive": "#4CAF50", "global_topk": "#FF9800"}
    markers = {"fixed": "o", "adaptive": "s", "global_topk": "^"}

    for ax, metric, title in [
        (axes[0], "recall", "Recall vs Token Budget"),
        (axes[1], "precision", "Precision vs Token Budget"),
    ]:
        for s in strategies:
            x = [r["tokens_used"] for r in results[s]]
            y = [r[metric] for r in results[s]]
            ax.plot(x, y, f'{markers[s]}-', color=colors[s], lw=2,
                    markersize=6, label=strategy_labels[s])

        ax.set_xlabel("Total Tokens")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Budget Allocation Strategy Comparison", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "adaptive_budget_comparison.png"),
                dpi=150)
    plt.close(fig)
    print(f"\nSaved: {args.output_dir}/adaptive_budget_comparison.png")

    # Save
    save = {}
    for s in strategies:
        save[s] = [
            {"budget_fraction": f, **r}
            for f, r in zip(budget_fractions, results[s])
        ]
    with open(os.path.join(args.output_dir, "adaptive_budget_results.json"), "w") as f:
        json.dump(save, f, indent=2)
    print(f"Saved: {args.output_dir}/adaptive_budget_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--head_type", choices=["small", "bighead"], default="bighead")
    parser.add_argument("--output_dir", default="results/adaptive_budget")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--expanded_dim", type=int, default=384)
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--n_attn_layers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args)
