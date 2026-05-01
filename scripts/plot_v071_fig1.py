"""Figure 1 for v071-vshift-fromphase24d result doc.

Two panels:
  A) Openvocab composite score: phase24d (rank 1, +22) vs v071/v071b/v071d
     vs v0.6.0 reference. Shows vshift is NOT competitive with phase24d.
  B) Side-by-side comparison of phase24d vs v071 frame 01 panels — the
     decisive qualitative test for horizon-band prior. (Composite hides
     this.)
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

WIKI = Path("/home/ogata/mac-brain/projects/semantic-autogaze")

COMPOSITE = {
    "phase24d (rank 1)": 22,
    "v071\nvshift 0.25\n(PRIMARY)": 20,
    "v071b\nvshift 0.5": 17,
    "v071d\naux-obj": 9,
    "v0.6.0 (ref)": 17,
    "v0.5.0 (ref)": 19,
}
COLORS = {
    "phase24d (rank 1)": "#3182bd",
    "v071\nvshift 0.25\n(PRIMARY)": "#d62728",
    "v071b\nvshift 0.5": "#ff9f7f",
    "v071d\naux-obj": "#ff9f7f",
    "v0.6.0 (ref)": "#9ecae1",
    "v0.5.0 (ref)": "#deebf7",
}

ORDER = ["v0.5.0 (ref)", "v0.6.0 (ref)", "phase24d (rank 1)",
         "v071\nvshift 0.25\n(PRIMARY)", "v071b\nvshift 0.5", "v071d\naux-obj"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel A: composite score
y = [COMPOSITE[k] for k in ORDER]
x = np.arange(len(ORDER))
bars = ax1.bar(x, y, color=[COLORS[k] for k in ORDER], edgecolor="black", linewidth=0.5)
ax1.axhline(22, color="#3182bd", linestyle=":", linewidth=1.0, label="phase24d (rank 1, must beat)")
ax1.axhline(17, color="#9ecae1", linestyle=":", linewidth=1.0, label="v0.6.0 reference")
ax1.set_ylim(7, 24)
ax1.set_xticks(x)
ax1.set_xticklabels(ORDER, rotation=0, ha="center", fontsize=8)
ax1.set_ylabel("openvocab composite score")
ax1.set_title("Panel A — openvocab composite\nv071 (vshift 0.25) +20 BELOW phase24d's +22; vshift 0.5 regresses; aux-obj catastrophic",
              fontsize=10)
ax1.legend(loc="lower right", fontsize=8)
for xi, yi in zip(x, y):
    ax1.text(xi, yi + 0.3, f"+{yi}", ha="center", fontsize=9, fontweight="bold")

# Panel B: frame 01 side-by-side qualitative
ax2.axis("off")
ax2.set_title("Panel B — frame 01 qualitative inspection\nphase24d (left) vs v071 vshift 0.25 (right) — same horizon-band failure",
              fontsize=10)
phase24d_panel = WIKI / "figures" / "envvideo_eval" / "phase24d" / "frame_01.png"
v071_panel = WIKI / "figures" / "envvideo_eval" / "v071_vshift025" / "frame_01.png"
combined = WIKI / "figures" / "v071-frame01-side-by-side.png"
# Stitch the two panels horizontally
img1 = Image.open(phase24d_panel)
img2 = Image.open(v071_panel)
h = max(img1.height, img2.height)
new = Image.new("RGB", (img1.width + img2.width + 30, h), (255, 255, 255))
new.paste(img1, (0, 0))
new.paste(img2, (img1.width + 30, 0))
new.save(combined)
ax2.imshow(np.array(new))

plt.suptitle(
    "v071-vshift-fromphase24d cycle 1 — vshift is insufficient to break the horizon-band prior\n"
    "Composite +20 < phase24d +22; qualitative panels (right of right panel) still show every query firing on upper-mid band.",
    fontsize=11, y=0.99
)
plt.tight_layout(rect=[0, 0, 1, 0.96])

out = WIKI / "figures" / "v071-vshift-fromphase24d-fig1.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"[saved] {out}")
