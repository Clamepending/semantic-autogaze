"""Generate quantitative figures for projects/semantic-autogaze/paper.md.

Outputs PNGs into the Library at
  /home/ogata/mac-brain/projects/semantic-autogaze/figures/

Figures produced:
  - budget-ladder-household.png    (vanilla-budget-ladder + fine-grained + aggressive)
  - budget-ladder-av.png           (budget-ladder-av-replication + fine-grained + sub-50)
  - fidelity-paired-flip.png       (HLVid + EgoSchema + Phase 6/10/15 m-vs-shuf, n=122 / n=500)
  - subselection-attractor.png     (CLIP vs random scoring across keep ratios)
  - pareto-pi-deployment.png       (COCO mIoU vs Pi 5 ms/frame, with §1 floors and v0.5.0 candidate)

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
hh_scale = np.array([0.60, 0.70, 0.80, 0.90, 1.00])
hh_correct = np.array([47,   53,   52,   47,   50])
hh_n = 122
hh_acc = hh_correct / hh_n
hh_err = np.full_like(hh_acc, 1.0 / hh_n)

fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.errorbar(
    hh_scale, hh_acc, yerr=hh_err, fmt="o-",
    color="#3b6ea8", ecolor="#9aaecb", capsize=3,
    linewidth=1.6, markersize=6, label="AutoGaze-only @ scale s",
)
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
av_scale = np.array([0.30, 0.40, 0.50, 0.60, 0.70, 1.00])
av_correct = np.array([37,   42,   39,   39,   38,   37])
av_n = 95
av_acc = av_correct / av_n
av_err = np.full_like(av_acc, 1.0 / av_n)

fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.errorbar(
    av_scale, av_acc, yerr=av_err, fmt="o-",
    color="#a85f3b", ecolor="#cba99a", capsize=3,
    linewidth=1.6, markersize=6, label="AutoGaze-only @ scale s",
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
# Figure 3: fidelity / paired-flip vs shuffled-Q  — REFRESHED 2026-04-30
# ---------------------------------------------------------------------------
# match-vs-shuffled net per scorer. Two benchmarks now:
#   (a) HLVid household n=122 — the original Reading-A/B closure cohort
#   (b) EgoSchema Subset n=500 — the sparse-relevance retry cohort
#
# Sources:
#   HLVid n=122:
#     BigHead-CLIPSeg: -1   (semantic-only-hlvid-baseline)
#     raw CLIP:         0   (raw-clip-scoring-baseline; fidelity-only)
#     SigLIP-2:         0   (siglip2-patch-scoring; fidelity-only)
#     OWL-ViT:         -4   (owlvit-hlvid-vqa)
#     NVILA-attn:      +3   (nvila-attention-distill cycle 1.5c-redo, threshold)
#     Ours v1 (CLIPSeg-recipe): 0  (independent-text-scorer-v1 cycle 2)
#   EgoSchema n=500 (sparse relevance):
#     Ours v1 (CLIPSeg-recipe): +12 (egoschema-needle-haystack-pilot cycle 2; binom p≈0.033)
#     Phase 10 atto:            -2  (phase10-atto-egoschema-vqa)
#     Phase 6 DINOv2-s:         -1  (dinov2s-egoschema-vqa)

scorers = [
    ("BigHead (HLVid)",           -1, "VQA",      "HLVid"),
    ("raw CLIP (HLVid)",           0, "fid",      "HLVid"),
    ("SigLIP-2 (HLVid)",           0, "fid",      "HLVid"),
    ("OWL-ViT (HLVid)",           -4, "VQA",      "HLVid"),
    ("NVILA-attn (HLVid)",        +3, "VQA",      "HLVid"),
    ("Ours v1 (HLVid)",            0, "VQA",      "HLVid"),
    ("Ours v1 (EgoSchema, n=500)",+12, "VQA",     "EgoSchema"),
    ("Phase 10 atto (EgoSchema)",  -2, "VQA",     "EgoSchema"),
    ("Phase 6 DINOv2-s (EgoSchema)",-1,"VQA",     "EgoSchema"),
]
falsifier_thresh = 3

fig, ax = plt.subplots(figsize=(10.5, 5.0))
y_pos = np.arange(len(scorers))[::-1]
for i, (name, v, k, cohort) in enumerate(scorers):
    yi = y_pos[i]
    if cohort == "EgoSchema":
        edge = "#1565c0"
    else:
        edge = "#333333"
    color = "#2e7d32" if v >= falsifier_thresh else "#cc3333"
    ax.barh(yi, v, color=color, edgecolor=edge, linewidth=1.2)
    metric_str = "VQA paired-flip" if k == "VQA" else "fidelity z≈0"
    if v == 0:
        ax.text(0.25, yi, f" {v:+d}  ({metric_str})", va="center", ha="left", fontsize=8)
    elif v > 0:
        ax.text(v + 0.25, yi, f" {v:+d}  ({metric_str})", va="center", ha="left", fontsize=8)
    else:
        ax.text(v - 0.25, yi, f"{v:+d}  ({metric_str}) ", va="center", ha="right", fontsize=8)
ax.axvline(falsifier_thresh, linestyle="--", color="#2e7d32", linewidth=1.4,
           label=f"falsifier threshold (≥ +{falsifier_thresh})")
ax.axvline(0, linestyle="-", color="#444444", linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([s[0] for s in scorers], fontsize=8.5)
ax.set_xlabel("match-vs-shuffled net (positive = text-conditioned)")
ax.set_title("Fidelity falsifier: match-vs-shuffled across HLVid + EgoSchema cohorts\n"
             "Only Ours-v1 on EgoSchema crosses +3 (+12, p≈0.033); Phase 10/Phase 6 falsify the scaling claim",
             fontsize=10)
ax.set_xlim(-7.5, 14.5)
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
keep = np.array([0.10, 0.25, 0.50, 0.75])
clip_correct = np.array([29,   30,   33,   33])
rand_correct = np.array([32,   33,   34,   34])
n_hh = 122

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(keep, clip_correct / n_hh, "o-", color="#3b6ea8", linewidth=1.6,
        markersize=7, label="CLIP-distilled scoring")
ax.plot(keep, rand_correct / n_hh, "s-", color="#cc8033", linewidth=1.6,
        markersize=7, label="uniform-random scoring")
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

# ---------------------------------------------------------------------------
# Figure 5: Pi-deployment Pareto — NEW 2026-04-30
# ---------------------------------------------------------------------------
# x-axis: Pi 5 ms/frame (Cortex-A76, 4 threads, deployment-realistic, log scale)
# y-axis: COCO 8-image qual-grid mIoU
# §1 deployment floors: Pi 5 ≤ 100 ms (≥10 fps), mIoU ≥ 0.71 (CLIPSeg − 0.10)
#
# Sources:
#   Phase 15 atto step35000 (v0.5.0):    37 ms (~27 fps Pi 5),  mIoU 0.722  ⭐  results/phase15-mIoU-eval.md
#   Phase 10 atto best_val (v0.4.0):     36 ms (~28 fps Pi 5),  mIoU 0.696      results/phase15-mIoU-eval.md cycle 0 baseline
#   v2-Tiny FN-12k (Phase 2d):           63 ms (15.9 fps),       mIoU 0.563      results/phase2-fn-filter.md
#   D-Mobile FN-12k (Phase 2d):          38 ms (26.6 fps),       mIoU 0.522      results/phase2-fn-filter.md
#   v1 (CLIP-B/16, Phase 2-FN step12k): 545 ms (1.8 fps),        mIoU 0.737      results/egoschema-pi-class-scorer-test.md
#   Phase 6 DINOv2-s optionB:            ~Pi-infeasible,         mIoU 0.799      RTX-class only (results/phase3-10-siglip-recipe.md)
#   AutoGaze (text-blind):                                       mIoU 0.228      reference baseline (n=8 grid)
#   CLIPSeg target:                      Pi-infeasible,          mIoU 0.806      red dotted target line

PARETO_ROWS = [
    # (label, pi5_ms, miou, color, marker, params, callout)
    ("Phase 15 atto step35000 (v0.5.0)", 37,   0.722, "#cc3333", "*", "3.7M+3.6M", True),
    ("Phase 10 atto (v0.4.0)",           36,   0.696, "#9467bd", "o", "3.7M+3.6M", False),
    ("v2-Tiny FN-12k",                   63,   0.563, "#1f77b4", "o", "5.7M+3.6M", False),
    ("D-Mobile FN-12k",                  38,   0.522, "#ff7f0e", "o", "1.5M+3.6M", False),
    ("v1 (CLIP-B/16) FN",                545,  0.737, "#7f7f7f", "o", "91.6M",     False),
    ("Phase 6 DINOv2-s (RTX-only)",      None, 0.799, "#7f7f7f", "D", "22M",       False),
    ("AutoGaze (text-blind)",            36,   0.228, "#2ca02c", "s", "1 GB",      False),
    ("CLIPSeg (target)",                 None, 0.806, "#d62728", "X", "1.5 GB",    False),
]

fig, ax = plt.subplots(figsize=(9.0, 5.6))

# §1 deployment floor box (lower-right corner = ≤100 ms AND ≥0.71 mIoU)
ax.axvspan(0, 100, alpha=0.07, color="#2e7d32", zorder=0)
ax.axhspan(0.71, 1.0, alpha=0.07, color="#2e7d32", zorder=0)
ax.axvline(100, linestyle="--", color="#2e7d32", linewidth=1.2, alpha=0.6,
           label="§1 floor: Pi 5 ≤ 100 ms (≥10 fps)")
ax.axhline(0.71, linestyle="--", color="#cc3333", linewidth=1.2, alpha=0.6,
           label="§1 floor: COCO mIoU ≥ 0.71 (CLIPSeg − 0.10)")
ax.axhline(0.806, linestyle=":", color="#d62728", linewidth=0.9, alpha=0.5,
           label="CLIPSeg target (0.806)")

# Plot points; entries with pi5_ms = None are pinned at the right edge with an arrow
right_edge_ms = 1500
for (label, ms, miou, color, marker, params, callout) in PARETO_ROWS:
    if ms is None:
        # Pi-infeasible — draw at right edge with leftward marker
        ax.scatter(right_edge_ms, miou, c=color, s=140, marker=marker,
                   edgecolor="black", linewidth=0.6, zorder=4)
        ax.annotate(f"{label}  →", xy=(right_edge_ms, miou),
                    xytext=(-6, 4), textcoords="offset points",
                    fontsize=8.5, ha="right", va="bottom")
    else:
        size = 220 if callout else 110
        edge_w = 1.6 if callout else 0.6
        ax.scatter(ms, miou, c=color, s=size, marker=marker,
                   edgecolor="black", linewidth=edge_w, zorder=5 if callout else 4)
        if callout:
            ax.annotate(f"[v0.5.0] {label}\n  ({params})",
                        xy=(ms, miou), xytext=(14, 14),
                        textcoords="offset points",
                        fontsize=9, fontweight="bold", color="#cc3333",
                        ha="left", va="bottom",
                        arrowprops=dict(arrowstyle="->", color="#cc3333", lw=1.0))
        else:
            offset_y = -10 if "Phase 10" in label else 4
            offset_x = -8 if "Phase 10" in label else 6
            ha = "right" if "Phase 10" in label else "left"
            va = "top" if "Phase 10" in label else "bottom"
            ax.annotate(label, (ms, miou), xytext=(offset_x, offset_y),
                        textcoords="offset points",
                        fontsize=8, ha=ha, va=va)

ax.set_xscale("log")
ax.set_xlim(20, right_edge_ms + 200)
ax.set_ylim(0.18, 0.86)
ax.set_xlabel("Pi 5 latency, ms / single frame (Cortex-A76, 4 threads, deployment-realistic; log scale)")
ax.set_ylabel("COCO 8-image qual-grid mIoU")
ax.set_title("Pi-deployment Pareto: COCO mIoU vs Pi 5 ms/frame\n"
             "Phase 15 atto step 35000 (v0.5.0, red star) is the first ckpt inside both §1 deployment floors",
             fontsize=11)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower right", fontsize=8.5)
fig.tight_layout()
out = os.path.join(OUT_DIR, "pareto-pi-deployment.png")
fig.savefig(out, dpi=140)
plt.close(fig)
print(f"saved: {out}")

print("done.")
