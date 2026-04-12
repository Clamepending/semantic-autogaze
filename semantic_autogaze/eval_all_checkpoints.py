"""
Evaluate all available checkpoints and generate comparison plots.

Automatically finds and evaluates:
  1. BigHead distill (baseline): results/distill_bighead/best_bighead_student.pt
  2. Pre-decoder BigHead: results/distill_predecoder/best_predecoder_bighead.pt
  3. Temporal BigHead: results/temporal_bighead/best_temporal_bighead.pt
  4. Ranking BigHead: results/ranking_bighead/best_ranking_bighead.pt

Computes score retention, binary recall, and ranking quality for each.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -m semantic_autogaze.eval_all_checkpoints \
    --device cuda:0
"""

import os
import glob
import random
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from semantic_autogaze.eval_filtering import EvalDataset
from semantic_autogaze.train_bighead import BigSimilarityHead


def load_model(ckpt_path, model_class, device, **kwargs):
    """Load a model checkpoint."""
    model = model_class(**kwargs).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


def evaluate_model(model, dataloader, device, T=16, N=196):
    """Run inference and compute metrics at multiple budgets."""
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Inference", leave=False):
            hidden = batch["hidden_states"].to(device)
            query = batch["text_embedding"].to(device)
            target = batch["target_scores"].to(device)
            pred_logits = model(hidden, query)
            all_preds.append(torch.sigmoid(pred_logits).cpu().numpy())
            all_targets.append(torch.sigmoid(target).cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    total_patches = T * N
    budget_fracs = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    results = []

    for frac in budget_fracs:
        budget = max(T, int(frac * total_patches))
        retentions = []
        recalls = []
        gt_overlaps = []

        for b in range(all_preds.shape[0]):
            p = all_preds[b]
            t = all_targets[b]
            gt_binary = t > 0.5

            # Global top-k selection
            topk = np.argsort(p)[-budget:]
            mask = np.zeros(total_patches, dtype=bool)
            mask[topk] = True

            # Score retention
            gt_sum = t.sum()
            if gt_sum > 1e-6:
                retentions.append(t[mask].sum() / gt_sum)

            # Binary recall
            tp = (mask & gt_binary).sum()
            fn = (~mask & gt_binary).sum()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            recalls.append(recall)

            # GT top-k overlap
            gt_topk = set(np.argsort(t)[-budget:])
            overlap = len(set(topk) & gt_topk) / len(gt_topk) if len(gt_topk) > 0 else 0
            gt_overlaps.append(overlap)

        results.append({
            "budget_frac": frac,
            "budget": budget,
            "score_retention": float(np.mean(retentions)) if retentions else 0,
            "recall": float(np.mean(recalls)),
            "gt_overlap": float(np.mean(gt_overlaps)),
        })

    return results


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

    # Define checkpoints to evaluate
    checkpoints = [
        {
            "name": "BigHead Distill (baseline)",
            "path": "results/distill_bighead/best_bighead_student.pt",
            "class": BigSimilarityHead,
            "kwargs": {"hidden_dim": 192, "embedding_dim": 512, "expanded_dim": 384,
                       "n_attn_heads": 6, "n_attn_layers": 2, "grid_size": 14},
            "color": "#4CAF50",
            "marker": "o",
        },
        {
            "name": "BigHead Warm Restart",
            "path": "results/bighead_warmrestart/best_bighead_student.pt",
            "class": BigSimilarityHead,
            "kwargs": {"hidden_dim": 192, "embedding_dim": 512, "expanded_dim": 384,
                       "n_attn_heads": 6, "n_attn_layers": 2, "grid_size": 14},
            "color": "#D32F2F",
            "marker": "*",
        },
        {
            "name": "Temporal BigHead",
            "path": "results/temporal_bighead/best_temporal_bighead.pt",
            "class": None,  # Will import if available
            "color": "#2196F3",
            "marker": "s",
        },
        {
            "name": "Ranking BigHead",
            "path": "results/ranking_bighead/best_ranking_bighead.pt",
            "class": BigSimilarityHead,
            "kwargs": {"hidden_dim": 192, "embedding_dim": 512, "expanded_dim": 384,
                       "n_attn_heads": 6, "n_attn_layers": 2, "grid_size": 14},
            "color": "#FF9800",
            "marker": "^",
        },
        {
            "name": "Pre-decoder BigHead",
            "path": "results/distill_predecoder/best_predecoder_bighead.pt",
            "class": None,  # Different hidden_dir needed
            "color": "#E91E63",
            "marker": "D",
        },
        {
            "name": "Temporal+Ranking",
            "path": "results/temporal_ranking/best_temporal_ranking.pt",
            "class": None,  # Will import TemporalBigSimilarityHead
            "color": "#9C27B0",
            "marker": "P",
        },
    ]

    all_results = {}

    for ckpt in checkpoints:
        if not os.path.exists(ckpt["path"]):
            print(f"\nSkipping {ckpt['name']} — checkpoint not found: {ckpt['path']}")
            continue

        print(f"\nEvaluating: {ckpt['name']}")

        if ckpt["class"] is None and "Temporal" in ckpt["name"]:
            from semantic_autogaze.train_temporal_bighead import TemporalBigSimilarityHead
            model, n_params = load_model(
                ckpt["path"], TemporalBigSimilarityHead, device,
                hidden_dim=192, embedding_dim=512, expanded_dim=384,
                n_attn_heads=6, n_spatial_layers=2, n_temporal_layers=1,
                grid_size=14, num_frames=16,
            )
        elif ckpt["class"] is None and "Pre-decoder" in ckpt["name"]:
            print(f"  Skipping — pre-decoder uses different hidden_dir")
            continue
        else:
            model, n_params = load_model(ckpt["path"], ckpt["class"], device, **ckpt["kwargs"])

        print(f"  Parameters: {n_params/1e3:.1f}K")
        results = evaluate_model(model, dataloader, device)
        all_results[ckpt["name"]] = {
            "params_k": n_params / 1e3,
            "results": results,
            "color": ckpt["color"],
            "marker": ckpt["marker"],
        }

        # Print summary
        for r in results:
            print(f"  {r['budget_frac']*100:>4.0f}% ({r['budget']:>4d} tokens): "
                  f"retention={r['score_retention']:.4f}, recall={r['recall']:.4f}")

        del model
        torch.cuda.empty_cache()

    if len(all_results) < 2:
        print("\nNeed at least 2 checkpoints for comparison. Run again when more models finish.")
        # Still save what we have
        with open(os.path.join(args.output_dir, "checkpoint_comparison.json"), "w") as f:
            save = {name: {"params_k": d["params_k"], "results": d["results"]}
                    for name, d in all_results.items()}
            json.dump(save, f, indent=2)
        print(f"Saved: {args.output_dir}/checkpoint_comparison.json")
        return

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric, title in [
        (axes[0], "score_retention", "Score Mass Retention"),
        (axes[1], "recall", "Binary Recall"),
        (axes[2], "gt_overlap", "GT Top-k Overlap"),
    ]:
        for name, data in all_results.items():
            x = [r["budget_frac"] * 100 for r in data["results"]]
            y = [r[metric] for r in data["results"]]
            label = f"{name} ({data['params_k']:.0f}K)"
            ax.plot(x, y, f'{data["marker"]}-', color=data["color"],
                    lw=2, markersize=6, label=label)

        ax.set_xlabel("Token Budget (%)")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 55])

    fig.suptitle("Checkpoint Comparison: Score Retention Quality",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "checkpoint_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved: {args.output_dir}/checkpoint_comparison.png")

    # Save JSON
    with open(os.path.join(args.output_dir, "checkpoint_comparison.json"), "w") as f:
        save = {name: {"params_k": d["params_k"], "results": d["results"]}
                for name, d in all_results.items()}
        json.dump(save, f, indent=2)
    print(f"Saved: {args.output_dir}/checkpoint_comparison.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--output_dir", default="results/checkpoint_comparison")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args)
