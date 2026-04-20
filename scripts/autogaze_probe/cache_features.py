"""Cycle 1: cache AutoGaze gaze-decoder hidden states for COCO val2017.

Per image: a (196, 192) tensor of post-LLaMA-decoder patch features
(14x14 grid at 224x224 input, 192-dim each, frozen `bfshi/AutoGaze`).

This is the strongest tap available — it's `SemanticAutoGaze.get_patch_hidden_states`,
which is what the prior 2026-04-16 linear/MLP/BigHead probes also used.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from semantic_autogaze.inference import load_autogaze
from semantic_autogaze.model import SemanticAutoGaze


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/coco")
    p.add_argument("--ann", default="annotations/instances_val2017.json")
    p.add_argument("--img-subdir", default="val2017")
    p.add_argument("--cache-dir", default="results/autogaze_probe/features_gaze_val")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-images", type=int, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(os.path.join(args.data_dir, args.ann))
    img_dir = os.path.join(args.data_dir, args.img_subdir)
    img_ids = sorted(coco.getImgIds())
    if args.max_images:
        img_ids = img_ids[: args.max_images]

    already = sum(1 for i in img_ids if (cache_dir / f"{i}.pt").exists())
    print(f"[diag] {already}/{len(img_ids)} already cached")
    if already == len(img_ids):
        return

    print(f"[diag] loading AutoGaze on {device}...")
    ag = load_autogaze(device=device)
    w = SemanticAutoGaze(ag).to(device).eval()

    t0 = time.time()
    for i, img_id in enumerate(tqdm(img_ids, desc="cache autogaze gaze-hidden")):
        out = cache_dir / f"{img_id}.pt"
        if out.exists():
            continue
        info = coco.imgs[img_id]
        try:
            img = Image.open(os.path.join(img_dir, info["file_name"])).convert("RGB")
        except FileNotFoundError:
            continue
        t = TF.to_tensor(img).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,C,H,W)
        with torch.inference_mode():
            h = w.get_patch_hidden_states(t)  # (1, 196, 192)
        torch.save(h[0].cpu().contiguous(), out)
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(img_ids) - (i + 1)) / max(rate, 1e-9)
            print(f"  [{i+1}/{len(img_ids)}] {rate:.1f} img/s, eta {eta/60:.1f} min")

    n_done = sum(1 for i in img_ids if (cache_dir / f"{i}.pt").exists())
    print(f"[diag] cached {n_done} files in {cache_dir} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
