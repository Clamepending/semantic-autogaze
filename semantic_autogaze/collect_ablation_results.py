"""
Collect ablation results from W&B and generate comparison plots.

Pulls val_bce curves for all ablation runs and creates:
  1. Training curve comparison (all ablations)
  2. Final val_bce bar chart
  3. Summary table

Usage:
  python3 -m semantic_autogaze.collect_ablation_results \
    --wandb_run 839/semantic-autogaze/6unhnux3 \
    --output_dir results/ablation_analysis
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


ABLATION_CONFIGS = {
    "A1_bighead_distill": {
        "label": "BigHead + Distill",
        "params": "3,438K",
        "description": "Reference: BigSimilarityHead with KD from teacher",
        "ref_val_bce": 0.0668,  # Known from prior training
    },
    "A2_small_distill": {
        "label": "Small + Distill",
        "params": "201K",
        "description": "SimilarityHead with KD from teacher",
    },
    "A3_bighead_nodistill": {
        "label": "BigHead, No Distill",
        "params": "3,438K",
        "description": "BigSimilarityHead trained on GT only (no KD)",
    },
    "A4_nospatial": {
        "label": "BigHead, No Spatial",
        "params": "~3.4M",
        "description": "BigSimilarityHead without spatial conv refinement",
    },
    "A5_mlp_only": {
        "label": "BigHead, MLP Only",
        "params": "~1.5M",
        "description": "BigSimilarityHead without self/cross attention",
    },
}


def collect_from_wandb(wandb_path, max_samples=5000):
    """Collect all ablation metrics from a W&B run."""
    api = wandb.Api()
    run = api.run(wandb_path)
    history = run.history(samples=max_samples)

    results = {}
    for abl_name in ABLATION_CONFIGS:
        val_col = f"{abl_name}/val_bce"
        epoch_col = f"{abl_name}/epoch"

        if val_col in history.columns:
            valid = history[history[val_col].notna()]
            if len(valid) > 0:
                epochs = valid[epoch_col].values
                val_bces = valid[val_col].values
                best_val = val_bces.min()
                best_epoch = epochs[val_bces.argmin()]

                results[abl_name] = {
                    "epochs": epochs.tolist(),
                    "val_bces": val_bces.tolist(),
                    "best_val": float(best_val),
                    "best_epoch": int(best_epoch),
                    "num_epochs_completed": int(epochs.max()),
                }
                ABLATION_CONFIGS[abl_name]["best_val"] = float(best_val)

    return results


def plot_training_curves(results, output_dir):
    """Plot validation BCE curves for all ablations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0']
    for i, (abl_name, config) in enumerate(ABLATION_CONFIGS.items()):
        if abl_name in results:
            r = results[abl_name]
            color = colors[i % len(colors)]
            ax.plot(r["epochs"], r["val_bces"], '-o', color=color,
                    lw=2, markersize=4, label=f'{config["label"]} ({config["params"]})')
            # Mark best
            ax.plot(r["best_epoch"], r["best_val"], '*', color=color,
                    markersize=12, zorder=5)
        elif "ref_val_bce" in config:
            # Plot reference as horizontal line
            color = colors[i % len(colors)]
            ax.axhline(y=config["ref_val_bce"], color=color, linestyle='--',
                       lw=1.5, label=f'{config["label"]} (ref: {config["ref_val_bce"]:.4f})')

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation BCE", fontsize=12)
    ax.set_title("Ablation Study: Training Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ablation_training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/ablation_training_curves.png")


def plot_final_comparison(output_dir):
    """Plot bar chart of best val_bce for each ablation."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = []
    vals = []
    colors_list = []
    base_colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0']

    for i, (abl_name, config) in enumerate(ABLATION_CONFIGS.items()):
        if "best_val" in config or "ref_val_bce" in config:
            val = config.get("best_val", config.get("ref_val_bce"))
            names.append(config["label"])
            vals.append(val)
            colors_list.append(base_colors[i % len(base_colors)])

    if not vals:
        print("  No results to plot yet.")
        return

    bars = ax.barh(range(len(vals)), vals, color=colors_list, alpha=0.8,
                   edgecolor='white', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Best Validation BCE (lower is better)", fontsize=12)
    ax.set_title("Ablation Study: Final Results", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ablation_final_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/ablation_final_comparison.png")


def print_summary():
    """Print a markdown table of ablation results."""
    print("\n| Ablation | Params | Best Val BCE | Description |")
    print("|----------|--------|-------------|-------------|")
    for abl_name, config in ABLATION_CONFIGS.items():
        val = config.get("best_val", config.get("ref_val_bce", "—"))
        if isinstance(val, float):
            val = f"{val:.4f}"
        print(f"| {config['label']} | {config['params']} | {val} | {config['description']} |")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    if HAS_WANDB and args.wandb_run:
        print("Collecting results from W&B...")
        results = collect_from_wandb(args.wandb_run)
        print(f"  Found data for: {list(results.keys())}")

    print("\nGenerating plots...")
    plot_training_curves(results, args.output_dir)
    plot_final_comparison(args.output_dir)

    # Save results
    save_data = {}
    for abl_name, config in ABLATION_CONFIGS.items():
        save_data[abl_name] = {
            "label": config["label"],
            "params": config["params"],
            "best_val": config.get("best_val", config.get("ref_val_bce")),
        }
        if abl_name in results:
            save_data[abl_name].update(results[abl_name])

    with open(os.path.join(args.output_dir, "ablation_results.json"), "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved: {args.output_dir}/ablation_results.json")

    print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_run", default="839/semantic-autogaze/6unhnux3")
    parser.add_argument("--output_dir", default="results/ablation_analysis")
    args = parser.parse_args()
    main(args)
