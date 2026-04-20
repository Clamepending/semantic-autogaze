"""Per-area-bucket eval: native-B-5K vs native-B-118K on val2017.

Quantifies whether the audit-level small-object regression is area-driven.
For each (img_id, cat_id) pair, computes:
  - GT mask area (fraction of image pixels)
  - IoU at thresholds 0.3 and 0.1 for each checkpoint
  - mean foreground prob, mean background prob
Then buckets by area: tiny <1%, small 1-5%, medium 5-15%, large >15%.

CPU-only. Reuses the predict_native path from compare_native_5k_vs_118k.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO

from semantic_autogaze.icon_student import IconStudent
from semantic_autogaze.train_coco_seg import build_category_mask, get_image_categories


AREA_BUCKETS = [
    ("tiny", 0.0, 0.01),
    ("small", 0.01, 0.05),
    ("medium", 0.05, 0.15),
    ("large", 0.15, 1.01),
]


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
def predict_native(model, native_obj, query, device, h: int, w: int) -> np.ndarray:
    patches = native_obj["patches"].float()
    n_h, n_w = native_obj["grid"]
    logits = model(patches.unsqueeze(0).to(device), query.unsqueeze(0).to(device),
                   grid_hw=(n_h, n_w))
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    enc_h, enc_w = native_obj["encode_hw"]
    new_h, new_w = native_obj["resize_hw"]
    top_pad, left_pad = native_obj["crop_top_left"]
    sy = h / float(new_h); sx = w / float(new_w)
    og_top = max(0, int(round(top_pad * sy)))
    og_left = max(0, int(round(left_pad * sx)))
    og_bot = min(h, int(round((top_pad + enc_h) * sy)))
    og_right = min(w, int(round((left_pad + enc_w) * sx)))
    region_h = og_bot - og_top
    region_w = og_right - og_left
    if region_h <= 0 or region_w <= 0:
        return np.zeros((h, w), dtype=np.float32)
    t = torch.from_numpy(probs).float().unsqueeze(0).unsqueeze(0)
    region = F.interpolate(t, size=(region_h, region_w), mode="bilinear",
                            align_corners=False)[0, 0].numpy()
    out = np.zeros((h, w), dtype=np.float32)
    out[og_top:og_bot, og_left:og_right] = np.clip(region, 0.0, 1.0)
    return out


def iou_at(pred: np.ndarray, gt: np.ndarray, thresh: float) -> float:
    p = (pred >= thresh).astype(np.float32)
    g = (gt >= 0.5).astype(np.float32)
    inter = float((p * g).sum())
    union = float((p + g - p * g).sum())
    return inter / max(union, 1e-9)


def sample_pairs(coco, img_ids, n_per_cat=2, max_pairs=200, seed=42):
    """Stratified-ish sample: up to n_per_cat positive (img, cat) pairs per category."""
    rng = random.Random(seed)
    cats = coco.loadCats(coco.getCatIds())
    pairs = []
    for c in cats:
        cat_id = c["id"]
        cat_imgs = coco.getImgIds(catIds=[cat_id])
        cat_imgs = [i for i in cat_imgs if i in img_ids]
        rng.shuffle(cat_imgs)
        for img_id in cat_imgs[:n_per_cat]:
            pairs.append((img_id, cat_id))
        if len(pairs) >= max_pairs:
            break
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def bucket_label(area_frac: float) -> str:
    for name, lo, hi in AREA_BUCKETS:
        if lo <= area_frac < hi:
            return name
    return "large"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-5k", required=True)
    p.add_argument("--ckpt-118k", required=True)
    p.add_argument("--dinov2-cache-native", required=True)
    p.add_argument("--clip-text-embeddings", required=True)
    p.add_argument("--data-dir", default="data/coco")
    p.add_argument("--ann", default="annotations/instances_val2017.json")
    p.add_argument("--img-subdir", default="val2017")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-pairs", type=int, default=200)
    p.add_argument("--n-per-cat", type=int, default=3)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(os.path.join(args.data_dir, args.ann))
    img_dir = os.path.join(args.data_dir, args.img_subdir)
    cache_dir = Path(args.dinov2_cache_native)

    cached_ids = set()
    for f in cache_dir.iterdir():
        if f.suffix == ".pt":
            try:
                cached_ids.add(int(f.stem))
            except ValueError:
                continue
    print(f"[diag] cache has {len(cached_ids)} images")

    pairs = sample_pairs(coco, cached_ids, n_per_cat=args.n_per_cat, max_pairs=args.max_pairs)
    print(f"[diag] sampled {len(pairs)} (img, cat) pairs")

    clip_text = torch.load(args.clip_text_embeddings, map_location="cpu", weights_only=True)
    print(f"[diag] loading native B 5K: {args.ckpt_5k}")
    model_5k = load_student(args.ckpt_5k, device)
    print(f"[diag] loading native B 118K: {args.ckpt_118k}")
    model_118k = load_student(args.ckpt_118k, device)

    rows = []
    for i, (img_id, cat_id) in enumerate(pairs):
        nv_path = cache_dir / f"{img_id}.pt"
        if not nv_path.exists():
            continue

        img_info = coco.imgs[img_id]
        try:
            img = Image.open(os.path.join(img_dir, img_info["file_name"])).convert("RGB")
        except FileNotFoundError:
            continue
        img_np = np.array(img); h, w = img_np.shape[:2]
        if h * w == 0:
            continue

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=0))
        if not anns:
            continue
        gt = build_category_mask(coco, img_id, cat_id, anns).astype(np.float32)
        gt = np.clip(gt, 0, 1)
        area_frac = float(gt.sum() / (h * w))
        if area_frac < 1e-5:  # mask was emptied
            continue

        query = clip_text[str(cat_id)].float()
        native_obj = torch.load(nv_path, map_location="cpu", weights_only=True)
        p5 = predict_native(model_5k, native_obj, query, device, h, w)
        p118 = predict_native(model_118k, native_obj, query, device, h, w)

        gt_bool = gt >= 0.5
        nbg = (~gt_bool).sum() + 1e-9
        nfg = gt_bool.sum() + 1e-9
        rows.append({
            "img_id": img_id, "cat_id": cat_id,
            "cat_name": coco.cats[cat_id]["name"],
            "area_frac": area_frac,
            "bucket": bucket_label(area_frac),
            "iou30_5k": iou_at(p5, gt, 0.3),
            "iou30_118k": iou_at(p118, gt, 0.3),
            "iou10_5k": iou_at(p5, gt, 0.1),
            "iou10_118k": iou_at(p118, gt, 0.1),
            "fg_prob_5k": float((p5 * gt_bool).sum() / nfg),
            "fg_prob_118k": float((p118 * gt_bool).sum() / nfg),
            "bg_prob_5k": float((p5 * (~gt_bool)).sum() / nbg),
            "bg_prob_118k": float((p118 * (~gt_bool)).sum() / nbg),
        })
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(pairs)}] processed")

    print(f"[diag] {len(rows)} pairs evaluated")
    with open(out_dir / "per_pair.json", "w") as f:
        json.dump(rows, f, indent=2)

    summary = {}
    for name, lo, hi in AREA_BUCKETS:
        bucket_rows = [r for r in rows if r["bucket"] == name]
        if not bucket_rows:
            summary[name] = {"n": 0}
            continue
        agg = {"n": len(bucket_rows)}
        for k in ("iou30_5k", "iou30_118k", "iou10_5k", "iou10_118k",
                  "fg_prob_5k", "fg_prob_118k", "bg_prob_5k", "bg_prob_118k"):
            agg[k] = float(np.mean([r[k] for r in bucket_rows]))
        agg["delta_iou30"] = agg["iou30_118k"] - agg["iou30_5k"]
        agg["delta_iou10"] = agg["iou10_118k"] - agg["iou10_5k"]
        agg["delta_fg_prob"] = agg["fg_prob_118k"] - agg["fg_prob_5k"]
        summary[name] = agg

    overall = {"n": len(rows)}
    for k in ("iou30_5k", "iou30_118k", "iou10_5k", "iou10_118k",
              "fg_prob_5k", "fg_prob_118k", "bg_prob_5k", "bg_prob_118k"):
        overall[k] = float(np.mean([r[k] for r in rows]))
    overall["delta_iou30"] = overall["iou30_118k"] - overall["iou30_5k"]
    overall["delta_iou10"] = overall["iou10_118k"] - overall["iou10_5k"]
    overall["delta_fg_prob"] = overall["fg_prob_118k"] - overall["fg_prob_5k"]
    summary["overall"] = overall

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[summary] per area-bucket (mean across pairs):")
    print(f"{'bucket':10s} {'N':>4s} {'iou30_5k':>10s} {'iou30_118k':>11s} {'Δ':>8s} "
          f"{'fg_5k':>7s} {'fg_118k':>8s} {'Δfg':>7s}")
    for name, _, _ in AREA_BUCKETS:
        s = summary[name]
        if s["n"] == 0:
            print(f"{name:10s} {0:>4d}  (no pairs)")
            continue
        print(f"{name:10s} {s['n']:>4d} {s['iou30_5k']:>10.3f} {s['iou30_118k']:>11.3f} "
              f"{s['delta_iou30']:>+8.3f} {s['fg_prob_5k']:>7.3f} {s['fg_prob_118k']:>8.3f} "
              f"{s['delta_fg_prob']:>+7.3f}")
    s = summary["overall"]
    print(f"{'OVERALL':10s} {s['n']:>4d} {s['iou30_5k']:>10.3f} {s['iou30_118k']:>11.3f} "
          f"{s['delta_iou30']:>+8.3f} {s['fg_prob_5k']:>7.3f} {s['fg_prob_118k']:>8.3f} "
          f"{s['delta_fg_prob']:>+7.3f}")
    print(f"\n[diag] artifacts: {out_dir / 'per_pair.json'}, {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
