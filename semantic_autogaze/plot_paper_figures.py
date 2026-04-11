"""
Generate publication-quality figures for Semantic AutoGaze paper/report.

Produces:
  1. Architecture comparison: Val BCE across all experiment versions (v1-v6)
  2. Ablation study bar chart: contribution of each component
  3. Latency vs quality Pareto plot
  4. Per-category AP analysis (top/bottom categories)
  5. Combined: gaze ratio × semantic filtering grid analysis

Usage:
  python3 -m semantic_autogaze.plot_paper_figures \
    --eval_dir results/eval_filtering/bighead_distill \
    --output_dir results/paper_figures
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter

# Style defaults
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def plot_experiment_timeline(output_dir):
    """Bar chart: Val BCE across all experiment versions."""
    experiments = [
        ("v2\nCLIPSeg+MSE", 4.70, "#BDBDBD", "MSE loss"),
        ("v3\nBCE+spatial", 0.0792, "#64B5F6", "BCE baseline"),
        ("v4a\nFocal small", 0.1298, "#EF9A9A", "Focal loss"),
        ("v4b\nFocal big", 0.1215, "#EF9A9A", "Focal loss"),
        ("v5a\nDistill small", 0.0771, "#81C784", "Distillation"),
        ("v5b\nDistill big", 0.0668, "#4CAF50", "Distillation"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    names = [e[0] for e in experiments]
    values = [e[1] for e in experiments]
    colors = [e[2] for e in experiments]

    # Use log scale for y since v2 is ~70x larger
    bars = ax.bar(range(len(experiments)), values, color=colors,
                  edgecolor="white", linewidth=1.5, width=0.7)

    # Annotate
    for i, (bar, val) in enumerate(zip(bars, values)):
        if val > 1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Reference line for best
    ax.axhline(y=0.0668, color="#2E7D32", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(experiments) - 0.5, 0.0668 + 0.003, "Best: 0.0668",
            color="#2E7D32", fontsize=9, ha="right")

    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Validation BCE Loss")
    ax.set_title("Semantic AutoGaze: Experiment Progression")
    ax.set_yscale("log")
    ax.set_ylim([0.04, 10])
    ax.grid(True, alpha=0.2, axis="y")

    # Legend
    patches = [
        mpatches.Patch(color="#BDBDBD", label="MSE loss"),
        mpatches.Patch(color="#64B5F6", label="BCE baseline"),
        mpatches.Patch(color="#EF9A9A", label="Focal loss (failed)"),
        mpatches.Patch(color="#4CAF50", label="Knowledge distillation"),
    ]
    ax.legend(handles=patches, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "experiment_timeline.png"))
    plt.close(fig)
    print(f"  Saved: {output_dir}/experiment_timeline.png")


def plot_latency_vs_quality(output_dir):
    """Pareto plot: latency (ms) vs quality (val BCE)."""
    variants = [
        # (name, latency_ms, val_bce, params_k, color, marker)
        ("v3 baseline\n(small, no distill)", 6.59, 0.0792, 201, "#64B5F6", "o"),
        ("Small distill", 6.59, 0.0771, 201, "#81C784", "s"),
        ("BigHead distill\n(post-decoder)", 14.91, 0.0668, 3438, "#4CAF50", "D"),
        ("BigHead no distill", 14.91, 0.1215, 3438, "#EF9A9A", "x"),
        ("Pre-decoder BigHead\n(projected)", 3.77, None, 3438, "#FF9800", "^"),
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, lat, bce, params, color, marker in variants:
        if bce is None:
            # Pre-decoder: plot at projected position with "?" marker
            ax.scatter([lat], [0.07], s=150, c=color, marker=marker,
                       edgecolors="black", linewidth=1, zorder=5)
            ax.annotate(name + "\n(training...)", (lat, 0.07),
                        textcoords="offset points", xytext=(10, -15),
                        fontsize=8, color=color)
            continue

        size = 80 + params / 20
        ax.scatter([lat], [bce], s=size, c=color, marker=marker,
                   edgecolors="black", linewidth=1, zorder=5)
        ax.annotate(name, (lat, bce), textcoords="offset points",
                    xytext=(10, 5), fontsize=8, color="black")

    # Mark vanilla AutoGaze latency
    ax.axvline(x=6.18, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(6.18, ax.get_ylim()[1] if ax.get_ylim()[1] > 0.1 else 0.13,
            "Vanilla\nAutoGaze", ha="center", va="bottom", fontsize=8, color="gray")

    ax.set_xlabel("Total Latency (ms)")
    ax.set_ylabel("Validation BCE (lower = better)")
    ax.set_title("Latency vs. Quality Trade-off")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # lower BCE = better → higher on plot

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "latency_vs_quality.png"))
    plt.close(fig)
    print(f"  Saved: {output_dir}/latency_vs_quality.png")


def plot_ablation_chart(output_dir, ablation_results=None):
    """Bar chart showing ablation results (contribution of each component)."""
    if ablation_results is None:
        # Use placeholder values — will be updated when ablations finish
        ablation_results = {
            "A1_bighead_distill": 0.0668,
            "A2_small_distill": 0.0771,  # from v5 results
            "A3_bighead_nodistill": 0.1215,  # from v4
        }

    labels = {
        "A1_bighead_distill": "Full model\n(BigHead+distill)",
        "A2_small_distill": "Small head\n+distill",
        "A3_bighead_nodistill": "BigHead\nno distill",
        "A4_bighead_nospatial": "BigHead+distill\nno spatial conv",
        "A5_mlp_only": "BigHead+distill\nMLP only",
    }

    colors = {
        "A1_bighead_distill": "#4CAF50",
        "A2_small_distill": "#81C784",
        "A3_bighead_nodistill": "#EF9A9A",
        "A4_bighead_nospatial": "#FFB74D",
        "A5_mlp_only": "#FFB74D",
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    sorted_keys = sorted(ablation_results.keys())
    x_labels = [labels.get(k, k) for k in sorted_keys]
    values = [ablation_results[k] for k in sorted_keys]
    bar_colors = [colors.get(k, "#90A4AE") for k in sorted_keys]

    bars = ax.bar(range(len(sorted_keys)), values, color=bar_colors,
                  edgecolor="white", linewidth=1.5, width=0.6)

    ref = ablation_results.get("A1_bighead_distill", 0.0668)
    ax.axhline(y=ref, color="#2E7D32", linestyle="--", linewidth=1, alpha=0.7)

    for i, (bar, val) in enumerate(zip(bars, values)):
        delta_pct = (val - ref) / ref * 100
        sign = "+" if delta_pct >= 0 else ""
        color = "#C62828" if delta_pct > 10 else "#2E7D32" if delta_pct <= 0 else "#E65100"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}\n({sign}{delta_pct:.1f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(len(sorted_keys)))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Validation BCE (lower = better)")
    ax.set_title("Ablation Study: Component Contributions")
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ablation_chart.png"))
    plt.close(fig)
    print(f"  Saved: {output_dir}/ablation_chart.png")
    return ablation_results


def plot_per_category_ap(eval_results_path, output_dir, top_n=15):
    """Horizontal bar chart showing per-category AP (top and bottom)."""
    if not os.path.exists(eval_results_path):
        print(f"  Skipping per-category plot — {eval_results_path} not found")
        return

    with open(eval_results_path) as f:
        data = json.load(f)

    if "per_category" not in data:
        print("  Skipping per-category plot — no per_category data")
        return

    categories = data["per_category"]
    # Filter to categories with enough samples
    valid = {k: v for k, v in categories.items()
             if v.get("count", 0) >= 5 and v.get("ap", 0) > 0}

    if len(valid) < 5:
        print(f"  Skipping per-category plot — only {len(valid)} valid categories")
        return

    sorted_cats = sorted(valid.items(), key=lambda x: x[1]["ap"], reverse=True)

    # Top and bottom
    top = sorted_cats[:top_n]
    bottom = sorted_cats[-top_n:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Top categories
    ax = axes[0]
    names = [c[0][:25] for c in top]
    aps = [c[1]["ap"] for c in top]
    y_pos = range(len(top))
    ax.barh(y_pos, aps, color="#4CAF50", alpha=0.8, edgecolor="#2E7D32")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Average Precision")
    ax.set_title(f"Top {top_n} Categories by AP")
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis="x")

    # Bottom categories
    ax = axes[1]
    names = [c[0][:25] for c in bottom]
    aps = [c[1]["ap"] for c in bottom]
    y_pos = range(len(bottom))
    ax.barh(y_pos, aps, color="#EF9A9A", alpha=0.8, edgecolor="#C62828")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Average Precision")
    ax.set_title(f"Bottom {top_n} Categories by AP")
    ax.set_xlim([0, max(aps) * 1.3 + 0.01])
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Semantic Filtering: Per-Category Average Precision", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_category_ap.png"))
    plt.close(fig)
    print(f"  Saved: {output_dir}/per_category_ap.png")


def plot_token_reduction_summary(output_dir):
    """Summary figure: token reduction at various operating points."""
    # From trade-off analysis results
    keep_ratios = [0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00]
    sem_recall = [0.280, 0.441, 0.637, 0.756, 0.895, 0.980, 1.000]
    rand_recall = [0.049, 0.100, 0.200, 0.301, 0.501, 0.800, 1.000]
    token_savings = [(1 - r) * 100 for r in keep_ratios]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot as function of token savings (inverted x)
    ax.plot(token_savings, sem_recall, "o-", color="#2196F3", lw=2.5,
            markersize=8, label="Semantic Filtering (BigHead)")
    ax.plot(token_savings, rand_recall, "s--", color="#9E9E9E", lw=2,
            markersize=6, label="Random Selection")

    # Fill the gap
    ax.fill_between(token_savings, rand_recall, sem_recall,
                     alpha=0.15, color="#2196F3", label="Semantic advantage")

    # Annotate key operating point
    idx_20 = keep_ratios.index(0.20)
    ax.annotate(f"80% savings\n64% recall",
                xy=(token_savings[idx_20], sem_recall[idx_20]),
                xytext=(token_savings[idx_20] + 8, sem_recall[idx_20] - 0.12),
                fontsize=10, fontweight="bold", color="#1565C0",
                arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5))

    ax.set_xlabel("Token Reduction (%)")
    ax.set_ylabel("Recall of Relevant Patches")
    ax.set_title("Semantic Filtering: Token Savings vs. Information Retention")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 1.05])
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "token_reduction_summary.png"))
    plt.close(fig)
    print(f"  Saved: {output_dir}/token_reduction_summary.png")


def plot_distillation_effect(output_dir):
    """Side-by-side comparison showing distillation's impact."""
    data = {
        "Small head": {
            "No distill (v3)": 0.0792,
            "With distill (v5a)": 0.0771,
        },
        "BigHead": {
            "No distill (v4b)": 0.1215,
            "With distill (v5b)": 0.0668,
        },
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(data))
    width = 0.35

    no_distill = [data[k][list(data[k].keys())[0]] for k in data]
    with_distill = [data[k][list(data[k].keys())[1]] for k in data]

    bars1 = ax.bar(x - width / 2, no_distill, width, label="Without Distillation",
                   color="#EF9A9A", edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + width / 2, with_distill, width, label="With Distillation",
                   color="#4CAF50", edgecolor="white", linewidth=1.5)

    # Annotate improvement
    for i in range(len(x)):
        improvement = (no_distill[i] - with_distill[i]) / no_distill[i] * 100
        mid_x = x[i]
        mid_y = max(no_distill[i], with_distill[i]) + 0.005
        ax.annotate(f"-{improvement:.1f}%", xy=(mid_x, mid_y),
                    ha="center", fontsize=11, fontweight="bold", color="#2E7D32")

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(list(data.keys()), fontsize=11)
    ax.set_ylabel("Validation BCE (lower = better)")
    ax.set_title("Effect of Knowledge Distillation")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim([0, max(no_distill) * 1.2])

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "distillation_effect.png"))
    plt.close(fig)
    print(f"  Saved: {output_dir}/distillation_effect.png")


def plot_latency_breakdown(output_dir):
    """Stacked bar chart showing latency breakdown by component."""
    # Measured values from benchmark_latency.py (actual end-to-end timings)
    # Note: component times don't exactly sum to total due to overhead
    variants = [
        ("Vanilla\nAutoGaze", 0.92, 5.27, 0),        # total=6.18
        ("+ Small\nHead", 0.92, 5.27, 0.40),          # total=6.59
        ("+ BigHead\nPost-Dec", 0.92, 5.27, 8.72),    # total=14.91
        ("Pre-Dec\n+ BigHead", 1.31, 0, 2.46),        # total=3.77
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    names = [v[0] for v in variants]
    enc = [v[1] for v in variants]
    dec = [v[2] for v in variants]
    head = [v[3] for v in variants]

    x = range(len(variants))
    ax.bar(x, enc, label="CNN Encoder", color="#42A5F5", edgecolor="white", linewidth=1.5)
    ax.bar(x, dec, bottom=enc, label="LLaMA Decoder", color="#EF5350", edgecolor="white", linewidth=1.5)
    ax.bar(x, head, bottom=[e + d for e, d in zip(enc, dec)],
           label="Semantic Head", color="#66BB6A", edgecolor="white", linewidth=1.5)

    # Total labels
    totals = [e + d + h for e, d, h in zip(enc, dec, head)]
    for i, total in enumerate(totals):
        ax.text(i, total + 0.2, f"{total:.2f}ms",
                ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency Breakdown (B=1, T=16)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "latency_breakdown.png"))
    plt.close(fig)
    print(f"  Saved: {output_dir}/latency_breakdown.png")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating paper figures...\n")

    print("1. Experiment timeline")
    plot_experiment_timeline(args.output_dir)

    print("2. Latency vs quality")
    plot_latency_vs_quality(args.output_dir)

    print("3. Ablation chart")
    # Try to load ablation results from W&B or file
    ablation_results = None
    ablation_file = os.path.join(args.eval_dir, "..", "ablation_results.json") if args.eval_dir else None
    if ablation_file and os.path.exists(ablation_file):
        with open(ablation_file) as f:
            ablation_results = json.load(f)
    plot_ablation_chart(args.output_dir, ablation_results)

    print("4. Per-category AP")
    eval_results = os.path.join(args.eval_dir, "eval_results.json") if args.eval_dir else None
    if eval_results:
        plot_per_category_ap(eval_results, args.output_dir)

    print("5. Token reduction summary")
    plot_token_reduction_summary(args.output_dir)

    print("6. Distillation effect")
    plot_distillation_effect(args.output_dir)

    print("7. Latency breakdown")
    plot_latency_breakdown(args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="results/eval_filtering/bighead_distill",
                        help="Directory with eval_results.json")
    parser.add_argument("--output_dir", default="results/paper_figures",
                        help="Output directory for figures")
    args = parser.parse_args()
    main(args)
