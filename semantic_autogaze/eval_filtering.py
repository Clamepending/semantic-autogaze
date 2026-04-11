"""
Filtering quality evaluation for Semantic AutoGaze.

Measures how well each trained head identifies semantically relevant patches
compared to CLIPSeg ground truth. Computes:
  - Precision/Recall/F1 at multiple thresholds
  - IoU (intersection over union) at multiple thresholds
  - mAP (mean average precision)
  - Per-category breakdown (top/bottom categories)
  - Visualization of PR curves and score distributions

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -m semantic_autogaze.eval_filtering \
    --hidden_dir results/distill/hidden_cache \
    --clipseg_dir results/distill/clipseg_cache \
    --ckpt results/distill_bighead/best_bighead_student.pt \
    --head_type bighead \
    --output_dir results/eval_filtering
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

from semantic_autogaze.model import SimilarityHead
from semantic_autogaze.train_bighead import BigSimilarityHead


class EvalDataset(Dataset):
    """Dataset for evaluation — loads hidden states and CLIPSeg targets."""

    def __init__(self, clipseg_files, hidden_dir):
        self.samples = []
        hidden_cache = {}
        skipped = 0

        for cf in clipseg_files:
            data = torch.load(cf, map_location="cpu", weights_only=False)
            vp = data["video_path"]
            key = hashlib.md5(vp.encode()).hexdigest()
            hidden_path = os.path.join(hidden_dir, f"{key}_hidden.pt")

            if not os.path.exists(hidden_path):
                skipped += 1
                continue

            if hidden_path not in hidden_cache:
                hidden_cache[hidden_path] = torch.load(hidden_path, map_location="cpu", weights_only=True)

            for q in data["queries"]:
                self.samples.append({
                    "hidden_states": hidden_cache[hidden_path],
                    "text_embedding": q["text_embedding"],
                    "target_scores": q["target_scores"],
                    "query_text": q.get("text", q.get("query_text", "unknown")),
                    "video_path": vp,
                })

        if skipped:
            print(f"  Skipped {skipped} files missing hidden cache")
        print(f"  Loaded {len(self.samples)} samples from {len(hidden_cache)} videos")

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


def load_head(args, device):
    """Load the appropriate head architecture and checkpoint."""
    if args.head_type == "small":
        head = SimilarityHead(
            hidden_dim=192, embedding_dim=512,
            grid_size=14, num_frames=16, use_spatial=True,
        ).to(device)
    elif args.head_type == "bighead":
        head = BigSimilarityHead(
            hidden_dim=192, embedding_dim=512,
            expanded_dim=args.expanded_dim,
            n_attn_heads=args.n_attn_heads,
            n_attn_layers=args.n_attn_layers,
            grid_size=14,
        ).to(device)
    else:
        raise ValueError(f"Unknown head type: {args.head_type}")

    state_dict = torch.load(args.ckpt, map_location=device)
    head.load_state_dict(state_dict)
    head.eval()
    n_params = sum(p.numel() for p in head.parameters())
    print(f"Loaded {args.head_type} head ({n_params/1e3:.1f}K params) from {args.ckpt}")
    return head


def compute_metrics_at_threshold(pred_binary, gt_binary):
    """Compute precision, recall, F1, IoU for binary masks."""
    tp = (pred_binary & gt_binary).sum().item()
    fp = (pred_binary & ~gt_binary).sum().item()
    fn = (~pred_binary & gt_binary).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def evaluate(head, dataloader, device, thresholds=None):
    """Run full evaluation and return per-sample and aggregate metrics."""
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    all_preds = []
    all_targets = []
    per_query_metrics = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            hidden = batch["hidden_states"].to(device)
            query = batch["text_embedding"].to(device)
            target = batch["target_scores"].to(device)

            pred_logits = head(hidden, query)
            pred_probs = torch.sigmoid(pred_logits)
            gt_probs = torch.sigmoid(target)

            for i in range(pred_probs.shape[0]):
                pred_np = pred_probs[i].cpu().numpy()
                gt_np = gt_probs[i].cpu().numpy()
                query_text = batch["query_text"][i]

                all_preds.append(pred_np)
                all_targets.append(gt_np)

                # Per-threshold metrics
                sample_metrics = {}
                for t in thresholds:
                    pred_bin = pred_np > t
                    gt_bin = gt_np > t
                    m = compute_metrics_at_threshold(
                        torch.tensor(pred_bin), torch.tensor(gt_bin)
                    )
                    sample_metrics[t] = m

                # Average precision (soft)
                gt_bin_05 = (gt_np > 0.5).astype(np.float32)
                if gt_bin_05.sum() > 0 and gt_bin_05.sum() < len(gt_bin_05):
                    ap = average_precision_score(gt_bin_05, pred_np)
                else:
                    ap = float("nan")

                per_query_metrics[query_text].append({
                    "thresholds": sample_metrics,
                    "ap": ap,
                })

    # Aggregate metrics across all samples
    all_preds = np.stack(all_preds)
    all_targets = np.stack(all_targets)

    aggregate = {}
    for t in thresholds:
        pred_bin = all_preds > t
        gt_bin = all_targets > t
        m = compute_metrics_at_threshold(
            torch.tensor(pred_bin), torch.tensor(gt_bin)
        )
        aggregate[str(t)] = m

    # Global mAP
    gt_bin_global = (all_targets > 0.5).astype(np.float32)
    valid_mask = (gt_bin_global.sum(axis=1) > 0) & (gt_bin_global.sum(axis=1) < gt_bin_global.shape[1])
    if valid_mask.sum() > 0:
        aps = []
        for i in range(len(all_preds)):
            if valid_mask[i]:
                aps.append(average_precision_score(gt_bin_global[i], all_preds[i]))
        global_map = np.mean(aps)
    else:
        global_map = float("nan")

    # Per-category summary
    category_summary = {}
    for query_text, metrics_list in per_query_metrics.items():
        aps = [m["ap"] for m in metrics_list if not np.isnan(m["ap"])]
        f1s = [m["thresholds"][0.5]["f1"] for m in metrics_list]
        ious = [m["thresholds"][0.5]["iou"] for m in metrics_list]
        category_summary[query_text] = {
            "count": len(metrics_list),
            "mAP": np.mean(aps) if aps else float("nan"),
            "mean_f1_05": np.mean(f1s),
            "mean_iou_05": np.mean(ious),
        }

    return {
        "aggregate": aggregate,
        "global_mAP": global_map,
        "category_summary": category_summary,
        "all_preds": all_preds,
        "all_targets": all_targets,
        "thresholds": thresholds,
    }


def plot_pr_curve(all_preds, all_targets, output_dir, label=""):
    """Plot precision-recall curve using all samples."""
    gt_bin = (all_targets > 0.5).astype(np.float32).flatten()
    pred_flat = all_preds.flatten()

    if gt_bin.sum() == 0 or gt_bin.sum() == len(gt_bin):
        print("  Skipping PR curve — no valid positive/negative split")
        return

    precision, recall, _ = precision_recall_curve(gt_bin, pred_flat)
    ap = average_precision_score(gt_bin, pred_flat)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved PR curve to {output_dir}/pr_curve.png")


def plot_threshold_sweep(aggregate, output_dir):
    """Plot F1, IoU, precision, recall as a function of threshold."""
    thresholds = sorted([float(t) for t in aggregate.keys()])
    f1s = [aggregate[str(t)]["f1"] for t in thresholds]
    ious = [aggregate[str(t)]["iou"] for t in thresholds]
    precs = [aggregate[str(t)]["precision"] for t in thresholds]
    recs = [aggregate[str(t)]["recall"] for t in thresholds]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(thresholds, f1s, "o-", label="F1", lw=2)
    ax.plot(thresholds, ious, "s-", label="IoU", lw=2)
    ax.plot(thresholds, precs, "^--", label="Precision", alpha=0.7)
    ax.plot(thresholds, recs, "v--", label="Recall", alpha=0.7)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs. Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "threshold_sweep.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved threshold sweep to {output_dir}/threshold_sweep.png")


def plot_score_distributions(all_preds, all_targets, output_dir):
    """Plot histogram of predicted scores for positive vs negative patches."""
    gt_bin = all_targets.flatten() > 0.5
    pred_flat = all_preds.flatten()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.hist(pred_flat[gt_bin], bins=50, alpha=0.6, label=f"Positive (n={gt_bin.sum()})", density=True)
    ax.hist(pred_flat[~gt_bin], bins=50, alpha=0.6, label=f"Negative (n={(~gt_bin).sum()})", density=True)
    ax.set_xlabel("Predicted Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution: Positive vs Negative Patches")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "score_distributions.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved score distributions to {output_dir}/score_distributions.png")


def main(args):
    torch.manual_seed(42)
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load data — use all cached files, split into same val set as training
    clipseg_files = sorted(glob.glob(os.path.join(args.clipseg_dir, "*_clipseg_clip*.pt")))
    print(f"Found {len(clipseg_files)} CLIPSeg cache files")

    random.shuffle(clipseg_files)
    split = int(0.9 * len(clipseg_files))
    val_files = clipseg_files[split:]
    print(f"Using {len(val_files)} validation files")

    dataset = EvalDataset(val_files, args.hidden_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Load head
    head = load_head(args, device)

    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate(head, dataloader, device)

    # Print summary
    print("\n" + "=" * 60)
    print(f"FILTERING QUALITY — {args.head_type} ({os.path.basename(args.ckpt)})")
    print("=" * 60)
    print(f"\nGlobal mAP: {results['global_mAP']:.4f}")

    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'IoU':<12}")
    print("-" * 60)
    for t in results["thresholds"]:
        m = results["aggregate"][str(t)]
        print(f"  {t:<10.1f} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['iou']:<12.4f}")

    # Best F1 threshold
    best_t = max(results["thresholds"], key=lambda t: results["aggregate"][str(t)]["f1"])
    best_m = results["aggregate"][str(best_t)]
    print(f"\nBest F1 threshold: {best_t:.1f} → F1={best_m['f1']:.4f}, IoU={best_m['iou']:.4f}")

    # Top/bottom categories
    cat_sum = results["category_summary"]
    valid_cats = {k: v for k, v in cat_sum.items() if not np.isnan(v["mAP"]) and v["count"] >= 3}
    if valid_cats:
        sorted_cats = sorted(valid_cats.items(), key=lambda x: x[1]["mAP"], reverse=True)
        print(f"\nTop 10 categories (by mAP):")
        for name, v in sorted_cats[:10]:
            print(f"  {name:<30s} mAP={v['mAP']:.3f}  F1={v['mean_f1_05']:.3f}  (n={v['count']})")
        print(f"\nBottom 10 categories (by mAP):")
        for name, v in sorted_cats[-10:]:
            print(f"  {name:<30s} mAP={v['mAP']:.3f}  F1={v['mean_f1_05']:.3f}  (n={v['count']})")

    # Plots
    print("\nGenerating plots...")
    plot_pr_curve(results["all_preds"], results["all_targets"], args.output_dir, label=args.head_type)
    plot_threshold_sweep(results["aggregate"], args.output_dir)
    plot_score_distributions(results["all_preds"], results["all_targets"], args.output_dir)

    # Save results (without large arrays)
    # Per-category results for plotting
    per_category = {}
    for name, v in valid_cats.items():
        per_category[name] = {
            "ap": float(v["mAP"]),
            "f1": float(v["mean_f1_05"]),
            "iou": float(v["mean_iou_05"]),
            "count": int(v["count"]),
        }

    save_results = {
        "head_type": args.head_type,
        "ckpt": args.ckpt,
        "global_mAP": float(results["global_mAP"]),
        "best_f1_threshold": float(best_t),
        "best_f1": float(best_m["f1"]),
        "best_iou_at_best_f1": float(best_m["iou"]),
        "aggregate": {str(k): v for k, v in results["aggregate"].items()},
        "per_category": per_category,
        "n_samples": len(dataset),
    }
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--ckpt", required=True, help="Path to trained head checkpoint")
    parser.add_argument("--head_type", choices=["small", "bighead"], required=True)
    parser.add_argument("--output_dir", default="results/eval_filtering")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--expanded_dim", type=int, default=384)
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--n_attn_layers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args)
