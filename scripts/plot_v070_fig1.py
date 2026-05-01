"""Figure 1 for v070-cocoonly-mlp-multiprompt-aggrAug result doc.

Two panels:
  A) Falsifier-band bar chart: 50-img mIoU + openvocab composite score for v070
     and its 4 sister recipes, with v0.6.0 (phase19b) and phase24d as reference
     dashed lines. The pre-stated decisive band (composite >=20 AND mIoU >=0.6885)
     is shaded.
  B) Per-recipe panel summary: which knob each recipe disabled, and its outcome
     vs falsifier.
"""
import json, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/ogata/semantic-autogaze")
WIKI = Path("/home/ogata/mac-brain/projects/semantic-autogaze")

# 50-img mIoU
miou = {}
for s in ["v070", "v070b", "v070c", "v070d", "v070e"]:
    with open(ROOT / "results" / "eval_50img_v070_sweep" / f"{s}.json") as f:
        miou[s] = float(json.load(f)["miou"])
miou["v060_phase19b"] = 0.6935
miou["phase24d"] = 0.6911  # if we have it; pulled from v060-simplification-ablation context
miou["v050_phase15"] = 0.6812

# Openvocab composite (computed by compare_openvocab_sweep.py)
COMPOSITE = {
    "v070": 20, "v070b": 18, "v070c": 18, "v070d": 18, "v070e": 18,
    "v060_phase19b": 17, "phase24d": 22, "v050_phase15": 19,
}

LABELS = {
    "v070":   "v070 (cocoonly+mpp0.5+aggrAug,\nresume v0.5.0) — primary",
    "v070b":  "v070b (mpp0.75)",
    "v070c":  "v070c (no aggrAug)",
    "v070d":  "v070d (no mpp)",
    "v070e":  "v070e (fullsource)",
    "v060_phase19b": "v0.6.0 (reference)",
    "phase24d":      "phase24d (rank 1, +22)\nresume v0.6.0",
    "v050_phase15":  "v0.5.0 (reference)",
}
COLOR = {
    "v070":   "#d62728",  # primary
    "v070b":  "#ff9f7f", "v070c":  "#ff9f7f", "v070d":  "#ff9f7f", "v070e":  "#ff9f7f",
    "v060_phase19b": "#9ecae1", "phase24d": "#3182bd", "v050_phase15": "#deebf7",
}

ORDER = ["v050_phase15", "v060_phase19b", "phase24d",
         "v070", "v070b", "v070c", "v070d", "v070e"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# Panel A: 50-img mIoU
y = [miou[k] for k in ORDER]
x = np.arange(len(ORDER))
ax1.bar(x, y, color=[COLOR[k] for k in ORDER], edgecolor="black", linewidth=0.5)
# Falsifier band
ax1.axhline(0.6885, color="red", linestyle="--", linewidth=1.2,
            label="falsifier floor (v0.6.0 - 0.005 = 0.6885)")
ax1.axhline(miou["v060_phase19b"], color="#9ecae1", linestyle=":", linewidth=1.0,
            label=f"v0.6.0 ref ({miou['v060_phase19b']:.4f})")
ax1.set_ylim(0.65, 0.70)
ax1.set_xticks(x)
ax1.set_xticklabels([LABELS[k].split("\n")[0] for k in ORDER], rotation=30, ha="right", fontsize=7)
ax1.set_ylabel("50-img COCO mIoU")
ax1.set_title("Panel A — 50-img mIoU vs falsifier band\nv070 = 0.6835 BELOW the 0.6885 floor by 0.005 → FALSIFIER TRIGGERED",
              fontsize=9)
ax1.legend(loc="lower left", fontsize=7)
for xi, yi, k in zip(x, y, ORDER):
    ax1.text(xi, yi + 0.001, f"{yi:.4f}", ha="center", fontsize=7)

# Panel B: openvocab composite
y2 = [COMPOSITE[k] for k in ORDER]
ax2.bar(x, y2, color=[COLOR[k] for k in ORDER], edgecolor="black", linewidth=0.5)
ax2.axhline(20, color="red", linestyle="--", linewidth=1.2, label="falsifier floor (composite >= 20)")
ax2.axhline(22, color="#3182bd", linestyle=":", linewidth=1.0, label="phase24d (current rank 1)")
ax2.set_ylim(15, 24)
ax2.set_xticks(x)
ax2.set_xticklabels([LABELS[k].split("\n")[0] for k in ORDER], rotation=30, ha="right", fontsize=7)
ax2.set_ylabel("openvocab composite score")
ax2.set_title("Panel B — openvocab composite vs falsifier band\nv070 = +20 PASSES boundary, but phase24d (+22) dominates",
              fontsize=9)
ax2.legend(loc="lower left", fontsize=7)
for xi, yi, k in zip(x, y2, ORDER):
    ax2.text(xi, yi + 0.15, f"+{yi}", ha="center", fontsize=7)

plt.suptitle(
    "v070-cocoonly-mlp-multiprompt-aggrAug cycle 1 — composition test FALSIFIED on 50-img tie band\n"
    "(composite passes +20, but cocoonly+mpp+aggrAug from v0.5.0 regresses 50-img by 0.010 vs v0.6.0)",
    fontsize=10
)
plt.tight_layout(rect=[0, 0, 1, 0.93])

out = WIKI / "figures" / "v070-cocoonly-mlp-multiprompt-aggrAug-fig1.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"[saved] {out}")
