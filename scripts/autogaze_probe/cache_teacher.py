"""Cycle 1: cache IconStudent-118K teacher heatmaps on val2017, mean-pooled to 14x14.

For each image, for each category present (positive cat list from COCO ann),
runs the IconStudent-118K head over the (already-cached native-aspect DINOv2
features) and mean-pools the per-pixel heatmap to a 14x14 grid matching
AutoGaze's input resolution.

Output per image: dict {cat_id: (14,14) float tensor in [0,1]}.

Also writes a sidecar dict mapping image_id -> sorted list of positive cat_ids
for the probe trainer's iteration order.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from semantic_autogaze.icon_student import IconStudent
from semantic_autogaze.train_coco_seg import build_category_mask, get_image_categories  # noqa: F401


def load_student(ckpt_path: str, device: torch.device) -> IconStudent:
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = raw["config"]
    model = IconStudent(
        patch_dim=cfg["patch_dim"], query_dim=cfg["query_dim"],
        decoder_dim=cfg["decoder_dim"],
        in_grid=cfg["in_grid"], out_grid=cfg["out_grid"],
        n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        dropout=0.0,
    ).to(device)
    sd = raw["state_dict"]
    pe = sd.get("pos_embed")
    if pe is not None and pe.dim() == 3:
        ig = cfg["in_grid"]; D = cfg["decoder_dim"]
        sd["pos_embed"] = pe.transpose(1, 2).reshape(1, D, ig, ig).contiguous()
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.inference_mode()
def predict_native_to_14x14(model, native_obj, query, device) -> np.ndarray:
    """Run IconStudent at native aspect, project back to a square 224x224
    canvas, mean-pool to 14x14."""
    patches = native_obj["patches"].float()
    n_h, n_w = native_obj["grid"]
    logits = model(patches.unsqueeze(0).to(device), query.unsqueeze(0).to(device),
                   grid_hw=(n_h, n_w))
    probs = torch.sigmoid(logits)[0].cpu()  # (out_h, out_w) at native grid

    # AutoGaze processes the image at 224x224. The native cache stores
    # encode_hw / resize_hw / crop_top_left so we know which sub-region of
    # the 224x224 frame the student's heatmap actually covers.
    enc_h, enc_w = native_obj["encode_hw"]
    new_h, new_w = native_obj["resize_hw"]
    top_pad, left_pad = native_obj["crop_top_left"]

    # Native cache was made by:
    #   1. resize image to (new_h, new_w) such that shortest side = 224
    #   2. center-crop to (enc_h, enc_w) where (enc_h, enc_w) = mult of 14
    #   3. that crop is what DINOv2 sees -> what the student's heatmap covers.
    # AutoGaze instead bicubically resizes the raw image to 224x224.
    # We map the student's heatmap to AutoGaze's 224x224 frame by:
    #   1. resize student logits to (enc_h, enc_w) — the actual encoded area
    #   2. paste into a (new_h, new_w) canvas at (top_pad, left_pad) [black elsewhere]
    #   3. resize that canvas to 224x224 — same operation AutoGaze applies to the raw image
    # Then mean-pool to 14x14.
    canvas = torch.zeros((new_h, new_w), dtype=torch.float32)
    region = F.interpolate(probs[None, None], size=(enc_h, enc_w),
                           mode="bilinear", align_corners=False)[0, 0]
    canvas[top_pad:top_pad + enc_h, left_pad:left_pad + enc_w] = region.clamp(0, 1)
    sq = F.interpolate(canvas[None, None], size=(224, 224),
                       mode="bilinear", align_corners=False)[0, 0]
    pooled = F.avg_pool2d(sq[None, None], kernel_size=16, stride=16)[0, 0]
    return pooled.numpy()  # (14, 14)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/icon_student_B_native_train/best.pt")
    p.add_argument("--clip-text-embeddings", default="results/icon_student_B_native_train/clip_text_embeddings.pt")
    p.add_argument("--dinov2-cache-native", default="results/dinov2_val_native")
    p.add_argument("--data-dir", default="data/coco")
    p.add_argument("--ann", default="annotations/instances_val2017.json")
    p.add_argument("--img-subdir", default="val2017")
    p.add_argument("--out-dir", default="results/autogaze_probe/teacher_14x14_val")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-images", type=int, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.dinov2_cache_native)

    coco = COCO(os.path.join(args.data_dir, args.ann))
    img_ids = sorted(coco.getImgIds())
    if args.max_images:
        img_ids = img_ids[: args.max_images]

    print(f"[diag] loading 118K student: {args.ckpt}")
    model = load_student(args.ckpt, device)
    clip_text = torch.load(args.clip_text_embeddings, map_location="cpu", weights_only=True)

    sidecar = {}
    t0 = time.time()
    n_pred = 0
    for i, img_id in enumerate(tqdm(img_ids, desc="cache teacher")):
        out = out_dir / f"{img_id}.pt"
        nv = cache_dir / f"{img_id}.pt"
        if not nv.exists():
            continue
        if out.exists():
            heatmaps = torch.load(out, weights_only=True)
            sidecar[img_id] = sorted(heatmaps.keys())
            continue
        cats = sorted(get_image_categories(coco, img_id).keys())
        if not cats:
            continue
        native_obj = torch.load(nv, map_location="cpu", weights_only=True)
        heatmaps = {}
        for cat_id in cats:
            query = clip_text[str(cat_id)].float()
            h = predict_native_to_14x14(model, native_obj, query, device)
            heatmaps[int(cat_id)] = torch.from_numpy(h).contiguous()
            n_pred += 1
        torch.save(heatmaps, out)
        sidecar[img_id] = cats
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = n_pred / max(elapsed, 1e-9)
            print(f"  [{i+1}/{len(img_ids)}] preds={n_pred} ({rate:.0f}/s)")

    with open(out_dir / "_sidecar.json", "w") as f:
        json.dump({str(k): v for k, v in sidecar.items()}, f)
    print(f"[diag] cached {len(sidecar)} images, {n_pred} per-cat heatmaps in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
