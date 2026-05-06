"""Figure 1 for v060-simplification-ablation result doc.
Two-panel comparison:
  (left) 50-img mIoU bar chart for v0.5.0, v0.6.0, A1-A5 + TTA/PE rows
         on v0.5.0 + v0.6.0
  (right) Per-failure-cat IoU heatmap: v0.5.0 / v0.6.0 / A3 / A4 / A1
         (the simplification candidates) across the 8 user-surfaced
         failure categories.
"""
from __future__ import annotations
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path("/home/ogata/mac-brain/projects/semantic-autogaze/figures/v060-simplification-ablation-fig1.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

# (label, path, color group)
ROWS = [
    ("v0.5.0 (phase15)", "results/eval_phase15/atto_long_50k_step35000/coco50.json", "ref"),
    ("v0.6.0 (phase19b)", "results/eval_phase19/phase19b_atto_perquery_best/coco50.json", "ref"),
    ("A1 no-distill", "results/eval_phase23/phase23a_atto_nodistill_best/coco50.json", "abl"),
    ("A2 linear-bias", "results/eval_phase23/phase23b_atto_perquery_linear_best/coco50.json", "abl"),
    ("A3 no-perquery", "results/eval_phase23/phase23c_atto_noperquery_best/coco50.json", "abl"),
    ("A4 COCO-only", "results/eval_phase23/phase23d_atto_cocoonly_best/coco50.json", "abl"),
    ("A5 no-class-bal", "results/eval_phase23/phase23e_atto_vanillaclassbalance_best/coco50.json", "abl"),
    ("v0.5.0 + TTA", "results/eval_phase23/v050_phase15_step35000_tta/coco50.json", "trick"),
    ("v0.5.0 + PE", "results/eval_phase23/v050_phase15_step35000_pe/coco50.json", "trick"),
    ("v0.6.0 + TTA", "results/eval_phase23/v060_phase19b_best_tta/coco50.json", "trick"),
    ("v0.6.0 + PE", "results/eval_phase23/v060_phase19b_best_pe/coco50.json", "trick"),
]
ROOT = pathlib.Path("/home/ogata/semantic-autogaze")
data = []
for label, p, group in ROWS:
    fp = ROOT / p
    if not fp.exists():
        print(f"missing: {fp}"); continue
    d = json.load(open(fp))
    data.append((label, d["miou"], group, d.get("per_cat", {})))

V050 = next(m for l, m, g, _ in data if l.startswith("v0.5.0 (phase"))
V060 = next(m for l, m, g, _ in data if l.startswith("v0.6.0 (phase"))

# ---------- LEFT: bar chart ----------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ax = axes[0]
labels = [d[0] for d in data]
mious = [d[1] for d in data]
groups = [d[2] for d in data]
colors = {
    "ref": "#2563eb",   # blue (v0.5.0, v0.6.0 references)
    "abl": "#dc2626",   # red (ablations)
    "trick": "#16a34a", # green (TTA / PE inference tricks)
}
bar_colors = [colors[g] for g in groups]
y = np.arange(len(labels))
bars = ax.barh(y, mious, color=bar_colors, edgecolor="black", linewidth=0.5)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.axvline(V050, color="#2563eb", ls=":", alpha=0.7, label=f"v0.5.0 = {V050:.3f}")
ax.axvline(V060, color="#1e3a8a", ls="--", alpha=0.7, label=f"v0.6.0 = {V060:.3f}")
ax.set_xlim(0.62, 0.71)
ax.set_xlabel("50-image COCO val mIoU", fontsize=11)
ax.set_title("v0.6.0 simplification ablation — 50-img mIoU\nblue = reference, red = ablation, green = inference trick",
             fontsize=11)
ax.legend(loc="lower right", fontsize=9)
for b, m in zip(bars, mious):
    ax.text(m + 0.001, b.get_y() + b.get_height()/2, f"{m:.4f}",
            va="center", fontsize=8.5)
ax.grid(axis="x", alpha=0.3)

# ---------- RIGHT: per-failure-cat heatmap ----------
FAIL = ["knife", "skis", "tie", "baseball bat", "snowboard", "sports ball", "skateboard", "spoon"]
hm_keys = ["v0.5.0 (phase15)", "v0.6.0 (phase19b)", "A1 no-distill", "A3 no-perquery", "A4 COCO-only"]
M = np.full((len(hm_keys), len(FAIL)), np.nan)
for i, key in enumerate(hm_keys):
    pc = next((d[3] for d in data if d[0] == key), {})
    for j, c in enumerate(FAIL):
        if c in pc: M[i, j] = pc[c]

ax = axes[1]
im = ax.imshow(M, cmap="RdYlGn", vmin=0.0, vmax=0.7, aspect="auto")
ax.set_xticks(np.arange(len(FAIL)))
ax.set_xticklabels(FAIL, rotation=35, ha="right", fontsize=9)
ax.set_yticks(np.arange(len(hm_keys)))
ax.set_yticklabels(hm_keys, fontsize=10)
ax.set_title("Per-failure-cat IoU (50-img)\nuser-surfaced failure cats only", fontsize=11)
for i in range(len(hm_keys)):
    for j in range(len(FAIL)):
        v = M[i, j]
        if not np.isnan(v):
            txt = f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color="white" if v < 0.3 else "black")
plt.colorbar(im, ax=ax, fraction=0.04)

plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"saved {OUT}")
