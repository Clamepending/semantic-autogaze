"""One-shot: re-render figures/scorer-latency-bar.png with the Ours v1 entry added.

Reads results/qual_method_grid/bench.json and merges the Ours v1 latency
(44.2 +- 0.71 ms; measured in scripts/qual_method_grid_coco.py).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


PRIOR_BENCH = "/home/ogata/semantic-autogaze/results/qual_method_grid/bench.json"
# From the COCO grid run [bench] log:  Ours v1   44.20 ± 0.71 ms
OURS_V1_BENCH = {"mean_ms": 44.20, "std_ms": 0.71}


def main():
    with open(PRIOR_BENCH) as f:
        bench = json.load(f)
    bench["Ours v1"] = OURS_V1_BENCH

    names = ["AutoGaze fwd", "AutoGaze (deployed)",
             "BigHead", "Ours v1",
             "CLIPSeg",
             "raw CLIP", "raw SigLIP-2", "OWL-ViT"]
    means = [bench[n]["mean_ms"] for n in names]
    stds = [bench[n]["std_ms"] for n in names]
    palette = {
        "AutoGaze fwd":         "#2ca02c",  # speed bar (forward only — what BigHead shares)
        "AutoGaze (deployed)":  "#9ec39e",  # full deployed cost
        "BigHead":              "#1f77b4",  # AutoGaze-features student
        "Ours v1":              "#9467bd",  # CLIP-features independent student (NEW)
        "CLIPSeg":              "#d62728",  # quality target
        "raw CLIP":             "#7f7f7f",
        "raw SigLIP-2":         "#7f7f7f",
        "OWL-ViT":              "#7f7f7f",
    }
    colors = [palette[n] for n in names]

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    bars = ax.bar(names, means, yerr=stds, color=colors, capsize=4,
                  edgecolor="black", linewidth=0.5)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.08,
                f"{m:.1f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(bench["AutoGaze fwd"]["mean_ms"], color="#2ca02c",
               linestyle="--", linewidth=1.0,
               label=f"AutoGaze fwd ({bench['AutoGaze fwd']['mean_ms']:.1f} ms) — speed target")

    ax.set_yscale("log")
    ax.set_ylabel("Scorer wall time per 16-frame video (ms, log scale)")
    ax.set_title("Scorer-only latency on RTX 4090 (no NVILA ViT/LLM)")
    ax.legend(loc="upper left", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    outs = [
        Path("/home/ogata/semantic-autogaze/results/qual_method_grid/scorer-latency-bar.png"),
        Path("/home/ogata/mac-brain/projects/semantic-autogaze/figures/scorer-latency-bar.png"),
    ]
    for p in outs:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=130, bbox_inches="tight")
        print(f"[fig] saved {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
