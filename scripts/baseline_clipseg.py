"""CLIPSeg open-vocab segmentation baseline (Lüddecke + Ecker, CVPR 2022,
arxiv 2112.10003). HF checkpoint `CIDAS/clipseg-rd64-refined`, no fine-tuning.

We evaluate CLIPSeg on the same eval panels we use for our internal
text-conditioned scorers, so it can be cross-compared via
`scripts/compare_openvocab_sweep.py`. CLIPSeg natively outputs a 352x352
sigmoid map; we downsample (cv2 INTER_AREA) to our 14x14 patch grid so
each cell is a true mean-pool over the corresponding image region.

Outputs match `qual_openvocab_eval.py:render_panel` exactly, including
`summary.csv` with the same `image,query,hmax,hmean` schema.

Usage:
  CUDA_VISIBLE_DEVICES=2 python -m scripts.baseline_clipseg \
      --eval streets --output_dir /home/.../baseline_clipseg_rd64
  CUDA_VISIBLE_DEVICES=2 python -m scripts.baseline_clipseg \
      --eval envvideo --output_dir /home/.../baseline_clipseg_rd64
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image as _PIL

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")
from scripts.qual_openvocab_eval import KEYWORDS as STREET_KEYWORDS
from scripts.qual_envvideo_eval import KEYWORDS as ENVVIDEO_KEYWORDS

GRID = 14
CLIPSEG_NATIVE = 352
HF_CKPT = "CIDAS/clipseg-rd64-refined"


@torch.no_grad()
def clipseg_heatmap(pil, query, processor, model, device):
    """Return a 14x14 numpy array in [0, 1]. Resizes input to 352x352
    (CLIPSeg native), runs the model, applies sigmoid, then INTER_AREA
    downsamples to 14x14 (true mean-pool over patch regions)."""
    pil_352 = pil.resize((CLIPSEG_NATIVE, CLIPSEG_NATIVE), _PIL.BICUBIC)
    inputs = processor(text=[query], images=[pil_352], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits  # CLIPSeg returns (352, 352) when batch=1, sometimes (1, 352, 352)
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    heatmap_352 = torch.sigmoid(logits[0]).cpu().numpy().astype(np.float32)  # (352, 352)
    heat_14 = cv2.resize(heatmap_352, (GRID, GRID), interpolation=cv2.INTER_AREA)
    heat_14 = np.clip(heat_14, 0.0, 1.0)
    return heat_14


def render_panel(pil, image_label, heats, save_path):
    """Identical layout to scripts/qual_openvocab_eval.py:render_panel."""
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


def discover_images(image_dir, eval_kind):
    img_dir = Path(image_dir)
    if eval_kind == "streets":
        return sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    elif eval_kind == "envvideo":
        return sorted(img_dir.glob("frame_*.png"))
    else:
        raise ValueError(f"unknown eval kind: {eval_kind}")


def main(args):
    if args.eval == "streets":
        keywords = STREET_KEYWORDS
        default_image_dir = "/home/ogata/semantic-autogaze/data/eval_openvocab_streets"
    elif args.eval == "envvideo":
        keywords = ENVVIDEO_KEYWORDS
        default_image_dir = "/home/ogata/mac-brain/projects/semantic-autogaze/envvideo_frames"
    else:
        raise ValueError(args.eval)

    image_dir = args.image_dir or default_image_dir
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[clipseg] loading {HF_CKPT} -> {device}", flush=True)
    processor = CLIPSegProcessor.from_pretrained(HF_CKPT)
    model = CLIPSegForImageSegmentation.from_pretrained(HF_CKPT)
    model = model.to(device).eval()
    print(f"[clipseg] ready. eval={args.eval} keywords={len(keywords)} image_dir={image_dir}", flush=True)

    image_paths = discover_images(image_dir, args.eval)
    print(f"[clipseg] found {len(image_paths)} images", flush=True)

    aggregate = []
    for img_p in image_paths:
        pil = _PIL.open(img_p).convert("RGB")
        heats = []
        for q in keywords:
            heat = clipseg_heatmap(pil, q, processor, model, device)
            hmax = float(heat.max()); hmean = float(heat.mean())
            heats.append((q, heat, hmax, hmean))
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
    p.add_argument("--eval", choices=["streets", "envvideo"], required=True,
                   help="Which keyword set + default image dir to use.")
    p.add_argument("--device", default="cuda:0",
                   help="Use 'cuda:0' under CUDA_VISIBLE_DEVICES=<gpu> to pin a GPU.")
    p.add_argument("--image_dir", default=None,
                   help="Override default image directory for the chosen --eval.")
    p.add_argument("--output_dir", required=True)
    main(p.parse_args())
