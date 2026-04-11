"""
Per-category score retention analysis.

Breaks down semantic filtering quality by LVIS category to identify:
  1. Which object categories the semantic head handles well
  2. Which categories have poor score retention (failure modes)
  3. Correlation between category frequency and performance

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -m semantic_autogaze.eval_per_category_retention \
    --hidden_dir results/distill/hidden_cache \
    --clipseg_dir results/distill/clipseg_cache \
    --ckpt results/distill_bighead/best_bighead_student.pt
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
from collections import defaultdict
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

    # We need per-query metadata, so load samples individually
    print("Loading samples with category info...")
    hidden_cache = {}
    samples = []
    skipped = 0

    for cf in tqdm(val_files, desc="Loading"):
        data = torch.load(cf, map_location="cpu", weights_only=False)
        vp = data["video_path"]
        key = hashlib.md5(vp.encode()).hexdigest()
        hidden_path = os.path.join(args.hidden_dir, f"{key}_hidden.pt")

        if not os.path.exists(hidden_path):
            skipped += 1
            continue

        if hidden_path not in hidden_cache:
            hidden_cache[hidden_path] = torch.load(hidden_path, map_location="cpu", weights_only=True)

        for q in data["queries"]:
            query_text = q.get("text", q.get("query_text", "unknown"))
            samples.append({
                "hidden_states": hidden_cache[hidden_path],
                "text_embedding": q["text_embedding"],
                "target_scores": q["target_scores"],
                "query_text": query_text,
            })

    print(f"  Loaded {len(samples)} samples, skipped {skipped}")

    # Load head
    head = load_head(args, device)

    # Run inference in batches
    print("Running inference...")
    T, N = 16, 196
    budget_frac = 0.10
    total_patches = T * N
    budget = max(T, int(budget_frac * total_patches))

    category_metrics = defaultdict(lambda: {
        "score_retentions": [], "topk_overlaps": [], "gt_densities": [],
        "pred_max_scores": [],
    })

    batch_size = args.batch_size
    for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating"):
        batch = samples[i:i+batch_size]
        hidden = torch.stack([s["hidden_states"] for s in batch]).to(device)
        query = torch.stack([s["text_embedding"] for s in batch]).to(device)
        target = torch.stack([s["target_scores"] for s in batch]).to(device)

        with torch.no_grad():
            pred_logits = head(hidden, query)
            preds = torch.sigmoid(pred_logits).cpu().numpy()
            targets = torch.sigmoid(target).cpu().numpy()

        for j in range(len(batch)):
            cat = batch[j]["query_text"]
            p = preds[j]
            t = targets[j]

            # Score retention with global top-k
            topk = np.argsort(p)[-budget:]
            mask = np.zeros(total_patches, dtype=bool)
            mask[topk] = True

            gt_sum = t.sum()
            if gt_sum < 1e-6:
                continue  # skip empty samples

            score_retention = t[mask].sum() / gt_sum

            # GT top-k overlap
            gt_topk = set(np.argsort(t)[-budget:])
            overlap = len(set(topk) & gt_topk) / len(gt_topk)

            # GT density (fraction of patches that are positive)
            gt_density = (t > 0.5).mean()

            category_metrics[cat]["score_retentions"].append(float(score_retention))
            category_metrics[cat]["topk_overlaps"].append(float(overlap))
            category_metrics[cat]["gt_densities"].append(float(gt_density))
            category_metrics[cat]["pred_max_scores"].append(float(p.max()))

    # Aggregate
    print(f"\n{len(category_metrics)} unique categories")
    cat_summary = {}
    for cat, metrics in category_metrics.items():
        n = len(metrics["score_retentions"])
        if n < 3:
            continue
        cat_summary[cat] = {
            "count": n,
            "mean_retention": np.mean(metrics["score_retentions"]),
            "std_retention": np.std(metrics["score_retentions"]),
            "mean_overlap": np.mean(metrics["topk_overlaps"]),
            "mean_gt_density": np.mean(metrics["gt_densities"]),
            "mean_pred_max": np.mean(metrics["pred_max_scores"]),
        }

    # Sort by retention
    sorted_cats = sorted(cat_summary.items(), key=lambda x: x[1]["mean_retention"], reverse=True)
    top_cats = sorted_cats[:20]
    bottom_cats = sorted_cats[-20:]

    # Print
    print(f"\n{'='*80}")
    print(f"TOP 20 CATEGORIES (best score retention at {budget_frac*100:.0f}% budget)")
    print(f"{'='*80}")
    print(f"  {'Category':<25} {'N':>5} {'Retention':>10} {'GT Overlap':>12} {'GT Density':>12}")
    for cat, m in top_cats:
        print(f"  {cat:<25} {m['count']:>5} {m['mean_retention']:>10.4f} "
              f"{m['mean_overlap']:>12.4f} {m['mean_gt_density']:>12.4f}")

    print(f"\n{'='*80}")
    print(f"BOTTOM 20 CATEGORIES (worst score retention)")
    print(f"{'='*80}")
    for cat, m in bottom_cats:
        print(f"  {cat:<25} {m['count']:>5} {m['mean_retention']:>10.4f} "
              f"{m['mean_overlap']:>12.4f} {m['mean_gt_density']:>12.4f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Top/bottom categories bar chart
    ax = axes[0, 0]
    n_show = 15
    top_names = [c[0][:20] for c in top_cats[:n_show]]
    top_vals = [c[1]["mean_retention"] for c in top_cats[:n_show]]
    ax.barh(range(n_show), top_vals, color="#4CAF50", alpha=0.8, edgecolor="#2E7D32")
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Score Retention")
    ax.set_title(f"Top {n_show} Categories (10% budget)")
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis="x")

    ax = axes[0, 1]
    bot_names = [c[0][:20] for c in bottom_cats[-n_show:]]
    bot_vals = [c[1]["mean_retention"] for c in bottom_cats[-n_show:]]
    ax.barh(range(n_show), bot_vals, color="#EF9A9A", alpha=0.8, edgecolor="#C62828")
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(bot_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Score Retention")
    ax.set_title(f"Bottom {n_show} Categories")
    ax.set_xlim([0, max(bot_vals) * 1.3 + 0.01])
    ax.grid(True, alpha=0.3, axis="x")

    # 2. Retention vs GT density scatter
    ax = axes[1, 0]
    all_retentions = [m["mean_retention"] for _, m in cat_summary.items()]
    all_densities = [m["mean_gt_density"] for _, m in cat_summary.items()]
    all_counts = [m["count"] for _, m in cat_summary.items()]
    sizes = [max(20, min(200, c * 3)) for c in all_counts]
    ax.scatter(all_densities, all_retentions, s=sizes, alpha=0.5, color="#2196F3",
               edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Mean GT Density (fraction of positive patches)")
    ax.set_ylabel("Mean Score Retention")
    ax.set_title("Score Retention vs GT Sparsity")
    ax.grid(True, alpha=0.3)

    # 3. Distribution of category retentions
    ax = axes[1, 1]
    ax.hist(all_retentions, bins=30, color="#42A5F5", edgecolor="white",
            alpha=0.8, linewidth=1)
    ax.axvline(np.mean(all_retentions), color="#D32F2F", linestyle="--",
               lw=2, label=f"Mean: {np.mean(all_retentions):.3f}")
    ax.axvline(np.median(all_retentions), color="#FF9800", linestyle="--",
               lw=2, label=f"Median: {np.median(all_retentions):.3f}")
    ax.set_xlabel("Score Retention at 10% Budget")
    ax.set_ylabel("Number of Categories")
    ax.set_title("Distribution of Per-Category Score Retention")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Per-Category Semantic Filtering Analysis", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "per_category_retention.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {args.output_dir}/per_category_retention.png")

    # Save JSON
    with open(os.path.join(args.output_dir, "per_category_results.json"), "w") as f:
        json.dump(cat_summary, f, indent=2)
    print(f"Saved: {args.output_dir}/per_category_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--head_type", choices=["small", "bighead"], default="bighead")
    parser.add_argument("--output_dir", default="results/per_category_retention")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--expanded_dim", type=int, default=384)
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--n_attn_layers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args)
