"""
Evaluate pre-decoder BigHead and compare with post-decoder BigHead.

Runs the same filtering quality evaluation on pre-decoder features,
then generates comparison plots.

Usage:
  python3 -m semantic_autogaze.eval_predecoder \
    --predecoder_ckpt results/distill_predecoder/best_predecoder_bighead.pt \
    --postdecoder_ckpt results/distill_bighead/best_bighead_student.pt \
    --predecoder_dir results/distill/predecoder_cache \
    --hidden_dir results/distill/hidden_cache \
    --clipseg_dir results/distill/clipseg_cache \
    --output_dir results/eval_predecoder_comparison
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
from sklearn.metrics import average_precision_score, precision_recall_curve

from semantic_autogaze.train_bighead import BigSimilarityHead
from semantic_autogaze.eval_filtering import (
    EvalDataset, load_head, evaluate, compute_metrics_at_threshold,
)
from semantic_autogaze.plot_filtering_tradeoff import (
    compute_quality_at_keep_ratio, run_inference,
)


class PredecoderEvalDataset(Dataset):
    """Dataset using pre-decoder features instead of post-decoder hidden states."""

    def __init__(self, clipseg_files, predecoder_dir):
        self.samples = []
        cache = {}
        skipped = 0

        for cf in clipseg_files:
            data = torch.load(cf, map_location="cpu", weights_only=False)
            vp = data["video_path"]
            key = hashlib.md5(vp.encode()).hexdigest()
            feat_path = os.path.join(predecoder_dir, f"{key}_predecoder.pt")

            if not os.path.exists(feat_path):
                skipped += 1
                continue

            if feat_path not in cache:
                cache[feat_path] = torch.load(feat_path, map_location="cpu", weights_only=True)

            for q in data["queries"]:
                self.samples.append({
                    "hidden_states": cache[feat_path],  # pre-decoder features
                    "text_embedding": q["text_embedding"],
                    "target_scores": q["target_scores"],
                    "query_text": q.get("text", q.get("query_text", "unknown")),
                    "video_path": vp,
                })

        if skipped:
            print(f"  Skipped {skipped} files missing predecoder cache")
        print(f"  Loaded {len(self.samples)} samples from {len(cache)} videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "hidden_states": s["hidden_states"],
            "text_embedding": s["text_embedding"],
            "target_scores": s["target_scores"],
            "query_text": s["query_text"],
            "video_path": s["video_path"],
        }


def plot_comparison(post_results, pre_results, output_dir):
    """Generate comparison plots between post-decoder and pre-decoder."""

    # 1. Threshold sweep comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    thresholds = sorted([float(t) for t in post_results["aggregate"].keys()])

    for ax, metric, title in [
        (axes[0], "f1", "F1 Score vs Threshold"),
        (axes[1], "iou", "IoU vs Threshold"),
    ]:
        post_vals = [post_results["aggregate"][str(t)][metric] for t in thresholds]
        pre_vals = [pre_results["aggregate"][str(t)][metric] for t in thresholds]

        ax.plot(thresholds, post_vals, "o-", color="#2196F3", lw=2, label="Post-decoder")
        ax.plot(thresholds, pre_vals, "s-", color="#FF9800", lw=2, label="Pre-decoder")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "threshold_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/threshold_comparison.png")

    # 2. PR curve comparison
    fig, ax = plt.subplots(figsize=(6, 5))

    for label, results, color in [
        ("Post-decoder", post_results, "#2196F3"),
        ("Pre-decoder", pre_results, "#FF9800"),
    ]:
        gt_bin = (results["all_targets"] > 0.5).astype(np.float32).flatten()
        pred_flat = results["all_preds"].flatten()
        if gt_bin.sum() > 0 and gt_bin.sum() < len(gt_bin):
            precision, recall, _ = precision_recall_curve(gt_bin, pred_flat)
            ap = average_precision_score(gt_bin, pred_flat)
            ax.plot(recall, precision, lw=2, color=color, label=f"{label} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve: Post-decoder vs Pre-decoder")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pr_curve_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/pr_curve_comparison.png")

    # 3. Keep-ratio trade-off comparison
    keep_ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]

    post_sem = compute_quality_at_keep_ratio(
        post_results["all_preds"], post_results["all_targets"], keep_ratios, mode="semantic")
    pre_sem = compute_quality_at_keep_ratio(
        pre_results["all_preds"], pre_results["all_targets"], keep_ratios, mode="semantic")
    rand = compute_quality_at_keep_ratio(
        post_results["all_preds"], post_results["all_targets"], keep_ratios, mode="random")

    fig, ax = plt.subplots(figsize=(8, 5))
    pct = [r * 100 for r in keep_ratios]

    post_recall = [post_sem[r]["recall"] for r in keep_ratios]
    pre_recall = [pre_sem[r]["recall"] for r in keep_ratios]
    rand_recall = [rand[r]["recall"] for r in keep_ratios]

    ax.plot(pct, post_recall, "o-", color="#2196F3", lw=2, label="Post-decoder BigHead")
    ax.plot(pct, pre_recall, "s-", color="#FF9800", lw=2, label="Pre-decoder BigHead")
    ax.plot(pct, rand_recall, "^--", color="#9E9E9E", lw=1.5, label="Random")

    ax.set_xlabel("% Patches Kept")
    ax.set_ylabel("Recall of Positive Patches")
    ax.set_title("Filtering Trade-off: Post-decoder vs Pre-decoder")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 105])
    ax.set_ylim([0, 1.05])

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "tradeoff_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/tradeoff_comparison.png")

    # 4. Summary table
    summary = {
        "post_decoder": {
            "mAP": float(post_results["global_mAP"]),
            "best_f1": float(max(post_results["aggregate"][str(t)]["f1"]
                                 for t in thresholds)),
            "best_iou": float(max(post_results["aggregate"][str(t)]["iou"]
                                  for t in thresholds)),
            "latency_ms": 14.91,
        },
        "pre_decoder": {
            "mAP": float(pre_results["global_mAP"]),
            "best_f1": float(max(pre_results["aggregate"][str(t)]["f1"]
                                 for t in thresholds)),
            "best_iou": float(max(pre_results["aggregate"][str(t)]["iou"]
                                  for t in thresholds)),
            "latency_ms": 3.77,
        },
        "keep_ratio_recall": {
            str(r): {
                "post": post_sem[r]["recall"],
                "pre": pre_sem[r]["recall"],
                "random": rand[r]["recall"],
            }
            for r in keep_ratios
        },
    }

    with open(os.path.join(output_dir, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {output_dir}/comparison_summary.json")

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'Post-decoder':<15} {'Pre-decoder':<15} {'Delta':<15}")
    print("-" * 70)
    for metric in ["mAP", "best_f1", "best_iou", "latency_ms"]:
        post_v = summary["post_decoder"][metric]
        pre_v = summary["pre_decoder"][metric]
        delta = pre_v - post_v
        sign = "+" if delta >= 0 else ""
        if metric == "latency_ms":
            pct = (delta / post_v) * 100
            print(f"  {metric:<23} {post_v:<15.2f} {pre_v:<15.2f} {sign}{delta:.2f}ms ({sign}{pct:.1f}%)")
        else:
            pct = (delta / max(post_v, 1e-8)) * 100
            print(f"  {metric:<23} {post_v:<15.4f} {pre_v:<15.4f} {sign}{delta:.4f} ({sign}{pct:.1f}%)")


def main(args):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load data — same val split as training
    clipseg_files = sorted(glob.glob(os.path.join(args.clipseg_dir, "*_clipseg_clip*.pt")))
    print(f"Found {len(clipseg_files)} CLIPSeg cache files")
    random.shuffle(clipseg_files)
    split = int(0.9 * len(clipseg_files))
    val_files = clipseg_files[split:]
    print(f"Using {len(val_files)} validation files")

    # Post-decoder evaluation
    print("\n--- Post-decoder BigHead ---")
    post_dataset = EvalDataset(val_files, args.hidden_dir)
    post_loader = DataLoader(post_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    post_head = BigSimilarityHead(
        hidden_dim=192, embedding_dim=512, expanded_dim=args.expanded_dim,
        n_attn_heads=args.n_attn_heads, n_attn_layers=args.n_attn_layers, grid_size=14,
    ).to(device)
    post_head.load_state_dict(torch.load(args.postdecoder_ckpt, map_location=device))
    post_head.eval()
    print(f"Loaded post-decoder head from {args.postdecoder_ckpt}")

    print("Running post-decoder evaluation...")
    post_results = evaluate(post_head, post_loader, device)

    # Pre-decoder evaluation
    print("\n--- Pre-decoder BigHead ---")
    pre_dataset = PredecoderEvalDataset(val_files, args.predecoder_dir)
    pre_loader = DataLoader(pre_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    pre_head = BigSimilarityHead(
        hidden_dim=192, embedding_dim=512, expanded_dim=args.expanded_dim,
        n_attn_heads=args.n_attn_heads, n_attn_layers=args.n_attn_layers, grid_size=14,
    ).to(device)
    pre_head.load_state_dict(torch.load(args.predecoder_ckpt, map_location=device))
    pre_head.eval()
    print(f"Loaded pre-decoder head from {args.predecoder_ckpt}")

    print("Running pre-decoder evaluation...")
    pre_results = evaluate(pre_head, pre_loader, device)

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(post_results, pre_results, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecoder_ckpt", required=True)
    parser.add_argument("--postdecoder_ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--predecoder_dir", default="results/distill/predecoder_cache")
    parser.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--output_dir", default="results/eval_predecoder_comparison")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--expanded_dim", type=int, default=384)
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--n_attn_layers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args)
