"""
Evaluate semantic score retention under different token budgets.

Instead of binary recall (which is inflated by sparse ground truth),
measures what fraction of the TOTAL CLIPSeg similarity score mass
is retained by the selected patches. This better reflects the
quality of preserved visual information.

Metrics:
  - Score Mass Retention: sum(scores[selected]) / sum(scores[all])
  - Weighted Recall: sum(gt_scores[selected]) / sum(gt_scores[all])
  - Top-k GT Overlap: how many of the k highest GT patches are selected

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -m semantic_autogaze.eval_score_retention \
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


def evaluate_score_retention(preds, targets, total_budget, T=16, N=196,
                             strategy="semantic_topk"):
    """Evaluate score mass retention for a given strategy and budget."""
    B = preds.shape[0]

    score_retentions = []
    weighted_recalls = []
    topk_gt_overlaps = []

    for b in range(B):
        # Select patches
        if strategy == "semantic_topk":
            topk = np.argsort(preds[b])[-total_budget:]
            mask = np.zeros(T * N, dtype=bool)
            mask[topk] = True
        elif strategy == "semantic_per_frame":
            k = max(1, total_budget // T)
            preds_pf = preds[b].reshape(T, N)
            mask = np.zeros(T * N, dtype=bool)
            for t in range(T):
                topk = np.argsort(preds_pf[t])[-k:]
                mask[t * N + topk] = True
        elif strategy == "random":
            indices = np.random.choice(T * N, size=min(total_budget, T * N), replace=False)
            mask = np.zeros(T * N, dtype=bool)
            mask[indices] = True
        elif strategy == "oracle":
            topk = np.argsort(targets[b])[-total_budget:]
            mask = np.zeros(T * N, dtype=bool)
            mask[topk] = True
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        gt = targets[b]
        gt_sum = gt.sum()
        if gt_sum < 1e-6:
            continue  # skip samples with no positive patches

        # Score mass retention: what fraction of GT score is captured
        score_retention = gt[mask].sum() / gt_sum
        score_retentions.append(score_retention)

        # Weighted recall: same as score retention but more intuitive name
        weighted_recalls.append(score_retention)

        # Top-k GT overlap: how many of the budget-sized GT top patches are selected
        gt_topk = set(np.argsort(gt)[-total_budget:])
        selected = set(np.where(mask)[0])
        overlap = len(gt_topk & selected) / len(gt_topk) if len(gt_topk) > 0 else 0
        topk_gt_overlaps.append(overlap)

    return {
        "score_retention": np.mean(score_retentions) if score_retentions else 0,
        "score_retention_std": np.std(score_retentions) if score_retentions else 0,
        "topk_gt_overlap": np.mean(topk_gt_overlaps) if topk_gt_overlaps else 0,
        "n_valid_samples": len(score_retentions),
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

    budget_fracs = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75]
    strategies = ["semantic_topk", "semantic_per_frame", "random", "oracle"]
    strategy_labels = {
        "semantic_topk": "Semantic (global top-k)",
        "semantic_per_frame": "Semantic (per-frame)",
        "random": "Random baseline",
        "oracle": "Oracle (GT top-k)",
    }
    strategy_colors = {
        "semantic_topk": "#2196F3",
        "semantic_per_frame": "#4CAF50",
        "random": "#9E9E9E",
        "oracle": "#E91E63",
    }

    results = {s: [] for s in strategies}

    print("\nEvaluating score retention...")
    for frac in budget_fracs:
        budget = max(T, int(frac * total_patches))
        print(f"\n  Budget: {budget} tokens ({frac*100:.0f}%)")
        for s in strategies:
            r = evaluate_score_retention(all_preds, all_targets, budget,
                                          T=T, N=N, strategy=s)
            results[s].append(r)
            print(f"    {s:<25}: retention={r['score_retention']:.4f}, "
                  f"gt_overlap={r['topk_gt_overlap']:.4f} "
                  f"(n={r['n_valid_samples']})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title, ylabel in [
        (axes[0], "score_retention", "CLIPSeg Score Mass Retention", "Score Retention"),
        (axes[1], "topk_gt_overlap", "Top-k GT Patch Overlap", "Overlap with GT Top-k"),
    ]:
        for s in strategies:
            x = [r["tokens"] / total_patches * 100 for r in results[s]]
            y = [r[metric] for r in results[s]]
            marker = "o" if "topk" in s else "s" if "frame" in s else "x" if "random" in s else "D"
            ax.plot(x, y, f'{marker}-', color=strategy_colors[s],
                    lw=2, markersize=6, label=strategy_labels[s])

        ax.set_xlabel("Token Budget (% of total)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 80])
        ax.set_ylim([0, 1.05])

    fig.suptitle("Semantic Filtering: Information Retention Quality", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "score_retention.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved: {args.output_dir}/score_retention.png")

    # Summary at key budgets
    print(f"\n{'='*80}")
    print("SCORE RETENTION SUMMARY (at 10% budget = 313 tokens)")
    print(f"{'='*80}")
    budget_10 = max(T, int(0.10 * total_patches))
    for s in strategies:
        for r in results[s]:
            if r["tokens"] == budget_10:
                print(f"  {strategy_labels[s]:<30}: "
                      f"retention={r['score_retention']:.4f}, "
                      f"gt_overlap={r['topk_gt_overlap']:.4f}")
    print(f"{'='*80}")

    # Save
    save = {}
    for s in strategies:
        save[s] = [
            {"budget_fraction": f,
             "score_retention": float(r["score_retention"]),
             "score_retention_std": float(r["score_retention_std"]),
             "topk_gt_overlap": float(r["topk_gt_overlap"]),
             "n_valid_samples": r["n_valid_samples"],
             "tokens": r["tokens"]}
            for f, r in zip(budget_fracs, results[s])
        ]
    with open(os.path.join(args.output_dir, "score_retention_results.json"), "w") as f:
        json.dump(save, f, indent=2)
    print(f"Saved: {args.output_dir}/score_retention_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--head_type", choices=["small", "bighead"], default="bighead")
    parser.add_argument("--output_dir", default="results/score_retention")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--expanded_dim", type=int, default=384)
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--n_attn_layers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args)
