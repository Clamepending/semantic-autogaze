"""Larger-N COCO val mIoU eval — confirms whether the 8-image qual-grid mIoU
generalizes. Samples N COCO val2017 images (one query per image, balanced
across categories) and computes mean per-image IoU at the GRID resolution.

Usage:
  CUDA_VISIBLE_DEVICES="" python -m scripts.eval_coco_val_miou \
      --device cpu --ckpt results/phase15_atto_long_50k/ckpt_step35000.pt \
      --n_images 50 --seed 2026
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")
from pycocotools.coco import COCO

# Reuse the same loader + helpers as eval_phase2_ckpt.py
from scripts.eval_phase2_ckpt import (
    load_ckpt, heatmap_one, iou_topk, GRID,
)

COCO_ROOT = "/home/ogata/semantic-autogaze/data/coco_val2017"


def main(args):
    device = torch.device(args.device)
    bb_fn, head, sb, mean, std, model, kind, bb_module = load_ckpt(args.ckpt, device)
    print(f"[ckpt] loaded {args.ckpt}", flush=True)

    import open_clip
    clip_model_text, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai")
    clip_model_text = clip_model_text.to(device).eval()
    clip_tok = open_clip.get_tokenizer("ViT-B-16")

    coco = COCO(os.path.join(COCO_ROOT, "annotations", "instances_val2017.json"))
    cat_ids = sorted(coco.getCatIds())
    rng = np.random.default_rng(args.seed)

    # Sample N (image, category, query) tuples — one per image, balanced across cats.
    samples = []
    img_ids_used = set()
    n_per_cat = max(1, args.n_images // len(cat_ids) + 1)
    for cat_id in cat_ids:
        cat_name = coco.loadCats([cat_id])[0]["name"]
        ann_ids = coco.getAnnIds(catIds=[cat_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        # Pick the largest mask per image, dedup by image
        per_img = {}
        for a in anns:
            if a["image_id"] in img_ids_used: continue
            if a["image_id"] not in per_img or a.get("area", 0) > per_img[a["image_id"]].get("area", 0):
                per_img[a["image_id"]] = a
        candidates = sorted(per_img.values(), key=lambda a: -a.get("area", 0))
        if not candidates: continue
        # Pick the top n_per_cat by area, randomized within
        chosen = candidates[:n_per_cat]
        rng.shuffle(chosen)
        for ann in chosen[:n_per_cat]:
            samples.append((cat_name, ann["image_id"], ann))
            img_ids_used.add(ann["image_id"])
        if len(samples) >= args.n_images: break

    samples = samples[:args.n_images]
    print(f"[sampled] {len(samples)} (image, category) pairs across {len(set(s[0] for s in samples))} categories")

    ious = []
    per_cat_ious = {}
    for i, (cat_name, img_id, ann) in enumerate(samples):
        gt = coco.annToMask(ann).astype(np.float32)
        info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(COCO_ROOT, "val2017", info["file_name"])
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [skip {i}] {img_path}: {e}")
            continue
        h = heatmap_one(pil, cat_name, bb_fn, head, sb, mean, std, clip_model_text, clip_tok, device)
        iou = iou_topk(h, gt)
        ious.append(iou)
        per_cat_ious.setdefault(cat_name, []).append(iou)
        if i % 10 == 0 or i == len(samples) - 1:
            print(f"  {i+1:3d}/{len(samples)} {cat_name:18s} img={img_id:6d} IoU={iou:.3f}  running_mean={np.mean(ious):.3f}", flush=True)

    miou = float(np.mean(ious))
    print(f"\n[final] N={len(ious)} mIoU={miou:.4f}  (std={np.std(ious):.3f})")
    cat_means = {c: float(np.mean(vs)) for c, vs in per_cat_ious.items()}
    print(f"[per-cat] mean per cat (top-10):")
    for c, v in sorted(cat_means.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {c:20s} {v:.3f}  (n={len(per_cat_ious[c])})")
    print(f"[per-cat] worst-10:")
    for c, v in sorted(cat_means.items(), key=lambda kv: kv[1])[:10]:
        print(f"  {c:20s} {v:.3f}  (n={len(per_cat_ious[c])})")

    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump({"ckpt": args.ckpt, "n": len(ious), "miou": miou, "std": float(np.std(ious)),
                       "per_cat": cat_means, "samples": [{"cat": s[0], "img_id": s[1]} for s in samples],
                       "ious": ious},
                       f, indent=2)
        print(f"[saved] {args.output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_images", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output_path", default=None)
    args = p.parse_args()
    main(args)
