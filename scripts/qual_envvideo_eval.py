"""Render per-frame heatmap panels from envvideo on the current best ckpt
to diagnose real-world (indoor) deployment failure modes.

Picks 4 representative frames spanning the video's content variety:
  frame_01 — close-up indoor wall + chair (cubicle/partition test)
  frame_05 — down-view at feet on carpet (Pi-on-chest viewpoint)
  frame_10 — cafe/study area with people and chairs
  frame_16 — open office with people at desks

Keyword set (16) split across in-distribution / OOD / abstention bands
chosen for indoor-Pi deployment, NOT the street scene:

  in-distribution things : person, chair, table, laptop, book
  in-distribution stuff  : floor, wall, ceiling, window
  OOD vocab gap          : partition, cubicle wall, green wall, blue carpet
  adjective+object       : white chair, wooden table
  abstention             : elephant, mountain

NEVER train on these images. The video is pure validation signal.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image as _PIL

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")
from scripts.eval_phase2_ckpt import load_ckpt, heatmap_one, GRID

KEYWORDS = [
    # In-distribution things
    "person", "chair", "table", "laptop", "book",
    # In-distribution stuff
    "floor", "wall", "ceiling", "window",
    # OOD vocab gap (the "wall on cubicle" failure mode)
    "partition", "cubicle wall", "green wall", "blue carpet",
    # Adjective + object
    "white chair", "wooden table",
    # Abstention test
    "elephant",
]
# 16 keywords -> 4-col x 4-row grid


def render_panel(pil, image_label, heats, save_path):
    n = len(heats)
    cols = 4
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 3.2, (rows + 1) * 3.2))
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


def main(args):
    device = torch.device(args.device)
    bb_fn, head, sb, mean, std, model, kind, bb_module, obj_head = load_ckpt(args.ckpt, device)
    if obj_head is not None and args.no_objectness:
        obj_head = None
    print(f"[ckpt] {args.ckpt}  obj_gate={'on' if obj_head is not None else 'off'}", flush=True)

    import open_clip
    text_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    text_model = text_model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-B-16")

    img_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(img_dir.glob("frame_*.png"))
    if args.frames:
        keep = set(int(f) for f in args.frames.split(","))
        image_paths = [p for p in image_paths if int(p.stem.split("_")[1]) in keep]

    aggregate = []
    for img_p in image_paths:
        pil = _PIL.open(img_p).convert("RGB")
        heats = []
        for q in KEYWORDS:
            heat = heatmap_one(pil, q, bb_fn, head, sb, mean, std, text_model, tok, device, obj_head=obj_head)
            hmax = float(heat.max()); hmean = float(heat.mean())
            heats.append((q, heat, hmax, hmean))
            aggregate.append((img_p.stem, q, hmax, hmean))
            print(f"  [{img_p.stem:12s}] '{q:14s}' max={hmax:.3f} mean={hmean:.3f}", flush=True)
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
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--image_dir", default="/home/ogata/mac-brain/projects/semantic-autogaze/envvideo_frames")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--no_objectness", action="store_true")
    p.add_argument("--frames", default="1,5,10,16",
                   help="Comma-separated frame numbers to render. Default '1,5,10,16'.")
    main(p.parse_args())
