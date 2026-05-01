"""Figure 1 for phase25 result doc.

Two panels:
  A) Cross-backbone composite score bar chart (phase24d ConvNeXt-atto ref +
     5 phase25-cycle-1 variants + v0.6.0).
  B) Frame_05 side-by-side: phase24d (ConvNeXt-atto, horizon-band fail) vs
     phase25 (DINOv2-s, horizon-band BROKEN). The decisive qualitative test.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

WIKI = Path("/home/ogata/mac-brain/projects/semantic-autogaze")

COMPOSITE = {
    "v0.6.0\n(ref)": 17,
    "phase24d\n(ConvNeXt-atto,\nrank 1)": 22,
    "phase25\nDINOv2-s\n+ recipe": 18,
    "phase25b\nDINOv2-s\nbaseline": 17,
    "phase26\nMobileCLIP-S2\n+ recipe": 7,
    "phase26b\nMobileCLIP-S2\nbaseline": 18,
    "phase27\nFastViT-T8\n+ recipe": 10,
}
COLORS = {
    "v0.6.0\n(ref)": "#9ecae1",
    "phase24d\n(ConvNeXt-atto,\nrank 1)": "#3182bd",
    "phase25\nDINOv2-s\n+ recipe": "#d62728",
    "phase25b\nDINOv2-s\nbaseline": "#ff7f7f",
    "phase26\nMobileCLIP-S2\n+ recipe": "#bcbd22",
    "phase26b\nMobileCLIP-S2\nbaseline": "#dbdb73",
    "phase27\nFastViT-T8\n+ recipe": "#7f7f7f",
}
ORDER = list(COMPOSITE.keys())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Panel A: composite bar chart
y = [COMPOSITE[k] for k in ORDER]
x = np.arange(len(ORDER))
ax1.bar(x, y, color=[COLORS[k] for k in ORDER], edgecolor="black", linewidth=0.5)
ax1.axhline(22, color="#3182bd", linestyle=":", linewidth=1.0, label="phase24d (rank 1, must beat)")
ax1.axhline(17, color="#9ecae1", linestyle=":", linewidth=1.0, label="v0.6.0 reference")
ax1.set_ylim(5, 25)
ax1.set_xticks(x)
ax1.set_xticklabels(ORDER, rotation=0, ha="center", fontsize=8)
ax1.set_ylabel("openvocab composite score (street validation, higher is better)")
ax1.set_title("Panel A — backbone sweep on openvocab composite\nNo backbone beats phase24d (ConvNeXt-atto) on the bench v1 outdoor metric",
              fontsize=10)
ax1.legend(loc="upper right", fontsize=8)
for xi, yi in zip(x, y):
    ax1.text(xi, yi + 0.4, f"+{yi}", ha="center", fontsize=9, fontweight="bold")

# Panel B: frame_05 side-by-side
ax2.axis("off")
ax2.set_title("Panel B — envvideo frame_05 (Pi-on-chest down-view at carpet+legs)\n"
              "phase24d ConvNeXt-atto: horizon-band failure (left) vs phase25 DINOv2-s: BROKEN (right)",
              fontsize=10)

phase24d_panel = WIKI / "figures" / "envvideo_eval" / "phase24d" / "frame_05.png"
phase25_panel  = WIKI / "figures" / "envvideo_eval" / "phase25_dinov2s_recipe" / "frame_05.png"
combined = WIKI / "figures" / "phase25-frame05-side-by-side.png"
img1 = Image.open(phase24d_panel)
img2 = Image.open(phase25_panel)
h = max(img1.height, img2.height)
new = Image.new("RGB", (img1.width + img2.width + 30, h), (255, 255, 255))
new.paste(img1, (0, 0))
new.paste(img2, (img1.width + 30, 0))
new.save(combined)
ax2.imshow(np.array(new))

plt.suptitle(
    "phase25-dinov2s-pi-class cycle 1 — DINOv2-s breaks the horizon-band spatial prior on indoor scenes\n"
    "Composite metric (Panel A) misses this: only outdoor street panels are in bench v1. Frame_05 (Panel B) is the qualitative test.",
    fontsize=11, y=0.99
)
plt.tight_layout(rect=[0, 0, 1, 0.95])

out = WIKI / "figures" / "phase25-dinov2s-pi-class-fig1.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"[saved] {out}")
