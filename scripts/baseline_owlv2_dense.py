"""OWLv2 dense per-patch text-image similarity baseline.

OWLv2 (Minderer et al., ICCV 2023; `google/owlv2-base-patch16-ensemble`) is
trained for open-vocabulary OBJECT DETECTION. Internally it computes a
per-patch (60x60 at native 960x960 input) class logit between each image
patch's class embedding and each text-query embedding BEFORE non-max
suppression. That dense per-patch tensor is essentially an open-vocab
heatmap. We bench it as a head-to-head baseline for our 14x14 sigmoid
heatmap models. OWLv2 is hypothesized to win on compositional adjective
queries (red sign / black car / white chair / wooden table) thanks to
web-scale pre-training.

Output: per (image, query) a 14x14 numpy array in [0, 1], rendered into
4-row x 4-col panels matching scripts/qual_openvocab_eval.py:render_panel
and a summary.csv with columns image,query,hmax,hmean.

Run:
  CUDA_VISIBLE_DEVICES=3 /home/ogata/miniconda3/envs/hunter/bin/python \\
      -m scripts.baseline_owlv2_dense \\
      --image_dir /home/ogata/semantic-autogaze/data/eval_openvocab_streets \\
      --output_dir /home/ogata/mac-brain/projects/semantic-autogaze/figures/openvocab_eval/baseline_owlv2_dense \\
      --keyword_set street
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image as _PIL

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")

# Import KEYWORDS from the existing qual scripts so cross-comparison is
# guaranteed identical.
from scripts.qual_openvocab_eval import KEYWORDS as STREET_KEYWORDS
from scripts.qual_envvideo_eval import KEYWORDS as ENV_KEYWORDS

from transformers import Owlv2Processor, Owlv2ForObjectDetection

GRID = 14  # output heatmap resolution (matches our scorer)


def render_panel(pil, image_label, heats, save_path):
    """Mirror of scripts/qual_openvocab_eval.py:render_panel.

    heats: list of (query, heat_14x14, hmax, hmean).
    """
    n = len(heats)
    cols = 4
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 3.0, (rows + 1) * 3.0))
    gs = fig.add_gridspec(rows + 1, cols)

    ax_in = fig.add_subplot(gs[0, :])
    ax_in.imshow(pil)
    ax_in.set_title(f"input: {image_label}", fontsize=10)
    ax_in.set_xticks([]); ax_in.set_yticks([])

    arr = np.array(pil)
    H, W = arr.shape[:2]
    for i, (q, heat, hmax, hmean) in enumerate(heats):
        r = 1 + (i // cols); c = i % cols
        ax = fig.add_subplot(gs[r, c])
        heat_up = np.kron(heat, np.ones((H // GRID + 1, W // GRID + 1)))[:H, :W]
        ax.imshow(arr, alpha=0.55)
        ax.imshow(heat_up, alpha=0.55, cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"'{q}'  max={hmax:.2f} mean={hmean:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def owlv2_dense_heatmaps(pil_image, queries, processor, model, device):
    """Compute dense 14x14 [0,1] heatmaps for all queries on one image.

    Strategy:
      1. Use Owlv2Processor to resize/pad to the model's native 960x960
         and tokenize all query strings as a single multi-query batch.
      2. Run the full forward; pull `outputs.logits` of shape
         (1, 3600, num_queries). These are per-patch class logits AFTER
         the model's learned shift/scale (so they are properly calibrated
         for sigmoid -> probability, exactly as the detector itself uses
         them pre-NMS).
      3. Reshape (60, 60, num_queries), apply sigmoid, then per-query
         INTER_AREA downsample to 14x14.

    Returns: list of 14x14 float32 arrays in [0, 1], one per query.
    """
    # The processor expects a list of lists of queries (one inner list per
    # image in the batch). Single image, all queries jointly.
    inputs = processor(text=[queries], images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    # outputs.logits: (batch=1, num_patches=3600, num_queries)
    logits = outputs.logits[0]  # (3600, Q)
    num_patches, num_queries = logits.shape

    # 60x60 grid for ViT-B/16 at 960x960
    H = W = int(num_patches ** 0.5)
    assert H * W == num_patches, f"unexpected patch count {num_patches}"

    probs = torch.sigmoid(logits).reshape(H, W, num_queries).cpu().numpy()

    # Downsample 60x60 -> 14x14. Detection-style logits are SPARSE: a single
    # strong patch dominates a region. INTER_AREA (mean-pool) destroys those
    # peak signals (e.g. jeep raw max=0.64 -> INTER_AREA max=0.03). Use
    # max-pool over each output cell instead so an "any patch fires" signal
    # in a region survives. This is the standard recipe for binning a fine
    # detection grid to a coarse heatmap.
    probs_t = torch.from_numpy(probs).permute(2, 0, 1).unsqueeze(0)  # (1,Q,60,60)
    pooled = torch.nn.functional.adaptive_max_pool2d(probs_t, (GRID, GRID))[0].numpy()  # (Q,14,14)
    heats_14 = [np.clip(pooled[q].astype(np.float32), 0.0, 1.0) for q in range(num_queries)]
    return heats_14


def main(args):
    device = torch.device(args.device)

    if args.keyword_set == "street":
        KEYWORDS = STREET_KEYWORDS
    elif args.keyword_set == "envvideo":
        KEYWORDS = ENV_KEYWORDS
    else:
        raise ValueError(args.keyword_set)

    print(f"[owlv2] loading google/owlv2-base-patch16-ensemble on {device}", flush=True)
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval()

    img_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if args.keyword_set == "envvideo":
        image_paths = sorted(img_dir.glob("frame_*.png"))
    else:
        image_paths = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))

    aggregate = []
    for img_p in image_paths:
        pil = _PIL.open(img_p).convert("RGB")
        heats_arrays = owlv2_dense_heatmaps(pil, list(KEYWORDS), processor, model, device)

        heats = []
        for q, h14 in zip(KEYWORDS, heats_arrays):
            hmax = float(h14.max()); hmean = float(h14.mean())
            heats.append((q, h14, hmax, hmean))
            aggregate.append((img_p.stem, q, hmax, hmean))
            print(f"  [{img_p.stem:35s}] '{q:14s}' max={hmax:.3f} mean={hmean:.3f}", flush=True)
        save_path = out_dir / f"{img_p.stem}.png"
        render_panel(pil, img_p.name, heats, save_path)
        print(f"  [saved] {save_path}", flush=True)

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("image,query,hmax,hmean\n")
        for img, q, hmax, hmean in aggregate:
            f.write(f"{img},{q},{hmax:.4f},{hmean:.4f}\n")
    print(f"[saved] {csv_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--keyword_set", choices=["street", "envvideo"], required=True)
    main(p.parse_args())
