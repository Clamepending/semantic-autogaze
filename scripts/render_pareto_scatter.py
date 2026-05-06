"""Render the v3-sweep Pareto-frontier scatter: COCO qual-grid mIoU vs deployment-realistic latency."""
import numpy as np
import matplotlib.pyplot as plt

# (label, mIoU, latency_ms, marker_color, params_M)
ROWS = [
    ("CLIPSeg (target)",         0.806, 561.0, "#d62728", "1.5 GB"),
    ("v1 (CLIP-B / std head)",   0.786,  19.0, "#1f77b4", "91.6 M"),
    ("v2-Small / std head",      0.646,   7.5, "#9467bd", "24.5 M"),
    ("v2-Tiny / std head",       0.613,   6.6, "#9467bd", "8.9 M"),
    ("C: ViT-S + h192/a1",       0.593,   7.0, "#9467bd", "22.4 M"),
    ("D: MobileNet-V3-S / std",  0.587,  17.8, "#ff7f0e", "5.05 M"),
    ("A: ViT-T + h192/a1",       0.541,   5.9, "#9467bd", "6.1 M"),
    ("B: ViT-T + h96/a1",        0.527,   5.3, "#9467bd", "5.7 M"),
    ("BigHead",                  0.450,   7.6, "#7f7f7f", "3.6 M+AutoGaze"),
    ("OWL-ViT",                  0.398, 259.0, "#7f7f7f", "92 M"),
    ("AutoGaze (text-blind)",    0.228,   5.9, "#2ca02c", "1 GB"),
    ("raw CLIP",                 0.222, 171.0, "#7f7f7f", "88 M"),
    ("raw SigLIP-2",             0.195, 213.0, "#7f7f7f", "88 M"),
]

fig, ax = plt.subplots(figsize=(8.5, 5.5))

# Plot each method
for (label, miou, lat, color, params) in ROWS:
    ax.scatter(lat, miou, c=color, s=80, edgecolor="black", linewidth=0.5, zorder=3)
    # Custom label offset to avoid overlap
    ax.annotate(label, (lat, miou), xytext=(6, 4), textcoords="offset points",
                fontsize=8, ha="left", va="bottom")

# AutoGaze fwd target line
ax.axvline(5.9, color="#2ca02c", linestyle="--", linewidth=1.0,
           label="AutoGaze fwd-only target (5.9 ms)")
# v1 mIoU horizontal line
ax.axhline(0.786, color="#1f77b4", linestyle=":", linewidth=1.0,
           label="v1 mIoU (0.786)")
# CLIPSeg mIoU horizontal line
ax.axhline(0.806, color="#d62728", linestyle=":", linewidth=1.0,
           label="CLIPSeg mIoU (0.806)")

ax.set_xscale("log")
ax.set_xlabel("Deployment-realistic scorer latency, ms (16-frame video, RTX 4090, log scale; pre-cached text emb)")
ax.set_ylabel("COCO qual-grid mIoU (n=8)")
ax.set_title("Pareto frontier: scorer latency vs COCO mask quality\n"
             "(purple = ours v1/v2/v3 sweep; green = AutoGaze speed target; red = CLIPSeg quality target)",
             fontsize=11)
ax.set_ylim(0.15, 0.85)
ax.set_xlim(3, 1500)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()

import sys
out = sys.argv[1] if len(sys.argv) > 1 else "/home/ogata/mac-brain/projects/semantic-autogaze/figures/pareto-scorer-latency-vs-miou.png"
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}")
plt.close(fig)
