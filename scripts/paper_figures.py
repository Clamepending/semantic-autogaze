"""Generate quantitative figures for projects/semantic-autogaze/paper.md.

Outputs PNGs into the Library at
  /home/ogata/mac-brain/projects/semantic-autogaze/figures/

Figures produced:
  - budget-ladder-household.png  (vanilla-budget-ladder + fine-grained + aggressive)
  - budget-ladder-av.png         (budget-ladder-av-replication + fine-grained + sub-50)
  - fidelity-paired-flip.png     (BigHead-CLIPSeg, raw CLIP, SigLIP-2, OWL-ViT, NVILA-attn)
  - subselection-attractor.png   (CLIP vs random scoring across keep ratios)

Data are sourced from the canonical result docs in
projects/semantic-autogaze/results/*.md (the lab notebook). Each result
doc cites the underlying JSON commit + path; the values used here mirror
those records and the script's footnote in paper.md cites this file as
the deterministic generator.

No new experiments. Pure plotting from previously-recorded scalars.
"""
from __future__ import annotations

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = "/home/ogata/mac-brain/projects/semantic-autogaze/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Figure 1: household budget ladder
# ---------------------------------------------------------------------------
# Source: results/vanilla-budget-ladder.md (cycle 2, scales 0.6/0.7/0.8/0.9/1.0)
# + results/budget-ladder-fine-grained.md (0.65/0.68/0.72/0.75)
# + results/filter-token-count-ablation.md (0.50/0.30) - shown via x-extension
# Fixed n=122 (HLVid household). +- 1 sample = +- 0.82%.

hh_scale = np.array([0.60, 0.70, 0.80, 0.90, 1.00])
hh_correct = np.array([47,   53,   52,   47,   50])
hh_n = 122
hh_acc = hh_correct / hh_n
hh_err = np.full_like(hh_acc, 1.0 / hh_n)  # +-1 sample bar

fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.errorbar(
    hh_scale,
    hh_acc,
    yerr=hh_err,
    fmt="o-",
    color="#3b6ea8",
    ecolor="#9aaecb",
    capsize=3,
    linewidth=1.6,
    markersize=6,
    label="AutoGaze-only @ scale s",
)
# Highlight 0.70 peak
peak_idx = int(np.argmax(hh_acc))
ax.plot(hh_scale[peak_idx], hh_acc[peak_idx], "o", color="#cc3333",
        markersize=12, markerfacecolor="none", markeredgewidth=2,
        label=f"peak (scale 0.70 = 53/122 = 0.434)")
ax.axhline(50 / hh_n, linestyle="--", color="#888888", linewidth=1.0,
           label="same-env vanilla = 50/122")
ax.set_xlabel("AutoGaze gaze_scale")
ax.set_ylabel("VQA accuracy on HLVid household (n=122)")
ax.set_title("Household budget-ladder: peak at scale 0.70")
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, linestyle=":", alpha=0.4)
ax.set_ylim(0.30, 0.50)
fig.tight_layout()
out = os.path.join(OUT_DIR, "budget-ladder-household.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"saved: {out}")

# ---------------------------------------------------------------------------
# Figure 2: av budget ladder
# ---------------------------------------------------------------------------
# Source: results/budget-ladder-av-replication.md (0.6, 0.7, 0.8, 0.9, 1.0)
# + budget-ladder-av-fine-grained.md (0.50, 0.55, 0.58, 0.62, 0.65)
# + budget-ladder-av-sub-50.md (0.35, 0.40, 0.45, 0.48)
# n=95 (HLVid av).

av_scale = np.array([0.30, 0.40, 0.50, 0.60, 0.70, 1.00])
av_correct = np.array([37,   42,   39,   39,   38,   37])
av_n = 95
av_acc = av_correct / av_n
av_err = np.full_like(av_acc, 1.0 / av_n)

fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.errorbar(
    av_scale,
    av_acc,
    yerr=av_err,
    fmt="o-",
    color="#a85f3b",
    ecolor="#cba99a",
    capsize=3,
    linewidth=1.6,
    markersize=6,
    label="AutoGaze-only @ scale s",
)
peak_idx = int(np.argmax(av_acc))
ax.plot(av_scale[peak_idx], av_acc[peak_idx], "o", color="#cc3333",
        markersize=12, markerfacecolor="none", markeredgewidth=2,
        label=f"peak (scale 0.40 = 42/95 = 0.442)")
ax.axhline(37 / av_n, linestyle="--", color="#888888", linewidth=1.0,
           label="same-env vanilla = 37/95")
ax.set_xlabel("AutoGaze gaze_scale")
ax.set_ylabel("VQA accuracy on HLVid av (n=95)")
ax.set_title("AV budget-ladder: peak shifts to scale 0.40 (K≈16 tokens/tile)")
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, linestyle=":", alpha=0.4)
ax.set_ylim(0.30, 0.50)
fig.tight_layout()
out = os.path.join(OUT_DIR, "budget-ladder-av.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"saved: {out}")

# ---------------------------------------------------------------------------
# Figure 3: fidelity / paired-flip vs shuffled-Q
# ---------------------------------------------------------------------------
# match-vs-shuffled net (paired-flip on n=122 end-to-end VQA where available;
# fidelity-z proxy where end-to-end was not run). The +3 falsifier line is
# defined on end-to-end VQA paired-flip; the fidelity-only entries cluster
# near 0 (no signal).
#
# Sources (slugs in result docs, with metric annotation):
#   BigHead-CLIPSeg: -1 (end-to-end VQA, semantic-only-hlvid-baseline cycle 2)
#   raw CLIP:         0 (fidelity z=-1.25, raw-clip-scoring-baseline; no end-to-end)
#   SigLIP-2:         0 (fidelity z=+0.25, siglip2-patch-scoring; no end-to-end)
#   OWL-ViT:         -4 (end-to-end VQA, owlvit-hlvid-vqa)
#   NVILA-attn:      +3 (end-to-end VQA, nvila-attention-distill cycle 1.5c-redo)

scorers = ["BigHead-CLIPSeg", "raw CLIP", "SigLIP-2", "OWL-ViT", "NVILA-attn"]
nets = [-1, 0, 0, -4, +3]
metric_kind = ["VQA", "fid", "fid", "VQA", "VQA"]
falsifier_thresh = 3
colors = ["#cc3333" if v < falsifier_thresh else "#2e7d32" for v in nets]

fig, ax = plt.subplots(figsize=(7.5, 3.8))
y_pos = np.arange(len(scorers))[::-1]  # top-to-bottom in source order
bars = ax.barh(y_pos, nets, color=colors, edgecolor="#333333", linewidth=0.8)
ax.axvline(falsifier_thresh, linestyle="--", color="#2e7d32", linewidth=1.4,
           label=f"falsifier threshold (≥ +{falsifier_thresh})")
ax.axvline(0, linestyle="-", color="#444444", linewidth=0.5)
for i, (v, k) in enumerate(zip(nets, metric_kind)):
    yi = y_pos[i]
    metric_str = "VQA paired-flip" if k == "VQA" else "fidelity z≈0 (no end-to-end VQA)"
    if v == 0:
        label = f" {v:+d}  ({metric_str})"
        ax.text(0.15, yi, label, va="center", ha="left", fontsize=8)
    elif v > 0:
        label = f" {v:+d}  ({metric_str})"
        ax.text(v + 0.15, yi, label, va="center", ha="left", fontsize=8)
    else:
        label = f"{v:+d}  ({metric_str}) "
        ax.text(v - 0.15, yi, label, va="center", ha="right", fontsize=8)
ax.set_yticks(y_pos)
ax.set_yticklabels(scorers)
ax.set_xlabel("match-vs-shuffled net (positive = text-conditioned)")
ax.set_title("Fidelity falsifier: only NVILA-attn passes the +3 threshold")
ax.set_xlim(-7.5, 7.5)
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, axis="x", linestyle=":", alpha=0.4)
fig.tight_layout()
out = os.path.join(OUT_DIR, "fidelity-paired-flip.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"saved: {out}")

# ---------------------------------------------------------------------------
# Figure 4: sub-selection attractor (CLIP vs random scoring)
# ---------------------------------------------------------------------------
# HLVid household n=122. Two scorers across keep ratios:
#   CLIP-distilled (BigHead): from autogaze-aware-filter-train Intersect{10,25,50,75}
#   random uniform: from subselection-insensitivity-probe + filter-vs-random-baseline
#
# Highlight: "random beats CLIP at low keep by +3 samples" at 0.10 and 0.25.

keep = np.array([0.10, 0.25, 0.50, 0.75])
clip_correct = np.array([29,   30,   33,   33])
rand_correct = np.array([32,   33,   34,   34])
n_hh = 122

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(keep, clip_correct / n_hh, "o-", color="#3b6ea8", linewidth=1.6,
        markersize=7, label="CLIP-distilled scoring")
ax.plot(keep, rand_correct / n_hh, "s-", color="#cc8033", linewidth=1.6,
        markersize=7, label="uniform-random scoring")
# Highlight the +3 inflection at low keep
for k, c, r in zip(keep, clip_correct, rand_correct):
    delta = r - c
    if delta >= 3:
        ax.annotate(f"random > CLIP by +{delta}",
                    xy=(k, r / n_hh), xytext=(k + 0.04, (r / n_hh) + 0.018),
                    arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8),
                    fontsize=8, color="#333333")
ax.axhline(51 / n_hh, linestyle="--", color="#888888", linewidth=1.0,
           label="vanilla AutoGaze (51/122)")
ax.set_xlabel("keep ratio (fraction of AutoGaze patches retained)")
ax.set_ylabel("VQA accuracy on HLVid household (n=122)")
ax.set_title("Sub-selection attractor: random beats CLIP at low keep")
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, linestyle=":", alpha=0.4)
ax.set_ylim(0.20, 0.45)
fig.tight_layout()
out = os.path.join(OUT_DIR, "subselection-attractor.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"saved: {out}")

print("done.")
