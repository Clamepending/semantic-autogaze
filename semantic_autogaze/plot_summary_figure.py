"""
Generate a comprehensive summary figure combining all key results.

6-panel figure:
  1. Token reduction cascade (AutoGaze → Semantic → Final)
  2. Score retention vs budget (semantic vs random vs oracle)
  3. Feature fidelity (cosine sim vs tokens)
  4. E2E latency comparison
  5. Per-category retention distribution
  6. Error analysis: spatial FN pattern

Usage:
  python3 -m semantic_autogaze.plot_summary_figure \
    --output_dir results/paper_figures
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # ========== Panel 1: Token Reduction Cascade ==========
    ax = fig.add_subplot(gs[0, 0])
    stages = ["All\nPatches", "After\nAutoGaze", "After\nSemantic"]
    tokens = [3136, 213, 106]
    pcts = ["100%", "6.8%", "3.4%"]
    colors = ["#BDBDBD", "#42A5F5", "#4CAF50"]

    bars = ax.bar(range(len(stages)), tokens, color=colors, edgecolor="white",
                  linewidth=2, width=0.55)
    for bar, tok, pct in zip(bars, tokens, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                f"{tok}\n({pct})", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for i in range(len(stages) - 1):
        reduction = (1 - tokens[i + 1] / tokens[i]) * 100
        ax.annotate(f"-{reduction:.0f}%", xy=(i + 0.5, (tokens[i] + tokens[i + 1]) / 2),
                    fontsize=9, ha="center", color="#D32F2F", fontweight="bold")

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages)
    ax.set_ylabel("Visual Tokens")
    ax.set_title("(a) Cascaded Token Reduction")
    ax.set_ylim([0, 3800])
    ax.grid(True, alpha=0.15, axis="y")

    # ========== Panel 2: Score Retention vs Budget ==========
    ax = fig.add_subplot(gs[0, 1])

    retention_file = "results/score_retention/score_retention_results.json"
    if os.path.exists(retention_file):
        with open(retention_file) as f:
            ret_data = json.load(f)

        for s, color, marker, label in [
            ("oracle", "#E91E63", "D", "Oracle"),
            ("semantic_topk", "#2196F3", "o", "Semantic (global)"),
            ("semantic_per_frame", "#4CAF50", "s", "Semantic (per-frame)"),
            ("random", "#9E9E9E", "x", "Random"),
        ]:
            if s in ret_data:
                x = [r["budget_fraction"] * 100 for r in ret_data[s]]
                y = [r["score_retention"] for r in ret_data[s]]
                ax.plot(x, y, f'{marker}-', color=color, lw=2, markersize=5, label=label)

        ax.fill_between(
            [r["budget_fraction"] * 100 for r in ret_data["random"]],
            [r["score_retention"] for r in ret_data["random"]],
            [r["score_retention"] for r in ret_data["semantic_topk"]],
            alpha=0.1, color="#2196F3",
        )
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Token Budget (%)")
    ax.set_ylabel("Score Mass Retention")
    ax.set_title("(b) Information Retention vs Budget")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 80])
    ax.set_ylim([0, 1.05])

    # ========== Panel 3: Feature Fidelity ==========
    ax = fig.add_subplot(gs[0, 2])

    # Hardcoded from feature fidelity results (person query)
    fidelity_data = [
        ("Gaze 75%", 213, 1.0, "gaze_only"),
        ("Int 50%", 106, 0.82, "intersect"),
        ("Int 30%", 63, 0.69, "intersect"),
        ("Int 10%", 21, 0.63, "intersect"),
        ("Sem 50%", 1568, 0.95, "semantic"),
        ("Sem 20%", 624, 0.88, "semantic"),
        ("Sem 10%", 304, 0.85, "semantic"),
    ]

    mode_colors = {"gaze_only": "#2196F3", "intersect": "#4CAF50", "semantic": "#FF9800"}
    mode_markers = {"gaze_only": "o", "intersect": "s", "semantic": "^"}

    for name, tokens, sim, mode in fidelity_data:
        ax.scatter(tokens, sim, s=80, color=mode_colors[mode], marker=mode_markers[mode],
                   edgecolors="black", linewidth=0.5, zorder=5)
        if tokens < 200 or "50%" in name:
            ax.annotate(name, (tokens, sim), textcoords="offset points",
                        xytext=(5, -10 if sim < 0.9 else 5), fontsize=7)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=6, label='Gaze only'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#4CAF50', markersize=6, label='Intersect'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#FF9800', markersize=6, label='Semantic only'),
    ]
    ax.legend(handles=legend_elements, fontsize=7)
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Cosine Similarity to Reference")
    ax.set_title("(c) SigLIP Feature Fidelity")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.55, 1.05])

    # ========== Panel 4: E2E Latency ==========
    ax = fig.add_subplot(gs[1, 0])

    configs = [
        ("Gaze 75%", 335.9, 17.2),
        ("Int 50%", 316.2, 16.8),
        ("Int 10%", 324.6, 16.8),
        ("Sem 20%", 9.5, 16.4),
        ("Sem 10%", 9.4, 16.7),
    ]
    names = [c[0] for c in configs]
    filter_t = [c[1] for c in configs]
    siglip_t = [c[2] for c in configs]
    totals = [c[1] + c[2] for c in configs]

    y = range(len(configs))
    ax.barh(y, filter_t, color="#42A5F5", label="Filter", height=0.6)
    ax.barh(y, siglip_t, left=filter_t, color="#66BB6A", label="SigLIP", height=0.6)

    for i, total in enumerate(totals):
        ax.text(total + 5, i, f"{total:.0f}ms", va="center", fontsize=8, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Latency (ms)")
    ax.set_title("(d) End-to-End Pipeline Latency")
    ax.legend(fontsize=7, loc="lower right")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.15, axis="x")

    # ========== Panel 5: Per-Category Retention Distribution ==========
    ax = fig.add_subplot(gs[1, 1])

    cat_file = "results/per_category_retention/per_category_results.json"
    if os.path.exists(cat_file):
        with open(cat_file) as f:
            cat_data = json.load(f)
        retentions = [v["mean_retention"] for v in cat_data.values() if v["count"] >= 3]
        ax.hist(retentions, bins=30, color="#42A5F5", edgecolor="white",
                alpha=0.8, linewidth=1)
        ax.axvline(np.mean(retentions), color="#D32F2F", linestyle="--",
                   lw=2, label=f"Mean: {np.mean(retentions):.3f}")
        ax.axvline(np.median(retentions), color="#FF9800", linestyle="--",
                   lw=2, label=f"Median: {np.median(retentions):.3f}")
        ax.legend(fontsize=7)
    ax.set_xlabel("Score Retention at 10% Budget")
    ax.set_ylabel("Number of Categories")
    ax.set_title(f"(e) Per-Category Retention ({len(retentions)} categories)")
    ax.grid(True, alpha=0.15, axis="y")

    # ========== Panel 6: Training Progression ==========
    ax = fig.add_subplot(gs[1, 2])

    experiments = [
        ("v2\nMSE", 4.70, "#BDBDBD"),
        ("v3\nBCE", 0.0792, "#64B5F6"),
        ("v4\nFocal", 0.1215, "#EF9A9A"),
        ("v5a\nSmall\nDistill", 0.0771, "#81C784"),
        ("v5b\nBigHead\nDistill", 0.0668, "#4CAF50"),
        ("Pre-Dec\n(in prog)", 0.0711, "#FF9800"),
    ]

    names = [e[0] for e in experiments]
    vals = [e[1] for e in experiments]
    colors = [e[2] for e in experiments]

    bars = ax.bar(range(len(experiments)), vals, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.6)

    for bar, val in zip(bars, vals):
        if val > 1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.1f}", ha="center", fontsize=8, fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", fontsize=8, fontweight="bold")

    ax.axhline(y=0.0668, color="#2E7D32", linestyle="--", lw=1, alpha=0.5)
    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Val BCE")
    ax.set_title("(f) Training Progression")
    ax.set_yscale("log")
    ax.set_ylim([0.04, 10])
    ax.grid(True, alpha=0.15, axis="y")

    fig.suptitle("Semantic AutoGaze: Comprehensive Results Summary",
                 fontsize=15, fontweight="bold", y=0.98)

    fig.savefig(os.path.join(args.output_dir, "summary_figure.png"), dpi=200)
    plt.close(fig)
    print(f"Saved: {args.output_dir}/summary_figure.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/paper_figures")
    args = parser.parse_args()
    main(args)
