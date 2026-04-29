"""Generate clean (image, query, mask) training targets directly from
COCO instance segmentations and / or LVIS instance segmentations — replaces
the noisy Grounded-SAM teacher with human-labeled masks.

Each (image, category) pair produces one .npz with the same schema as the
Phase 2 Grounded-SAM outputs, so `train_siglip_dense_distill.py` can consume
them unchanged.

Schema (per .npz, named `<image_id>__<query_slug>.npz`):
  img_id        str
  query         str       (the human-readable category name)
  presence      bool      (True if any instance of category in image)
  mask14        (14, 14)  uint8 (0/255), max-pooled from full mask
  mask_full     (H, W)    uint8 (0/255), union of all instance masks for this category
  H, W          int       (original image dimensions)
  clip_sim      float     (1.0 — no gate, gold supervision)
  gate_passed   bool      (True — gold supervision)
  top_box_score float     (1.0 — gold supervision)
  n_boxes       int       (number of instances of category in image)
  source        str       ("coco" or "lvis")

LVIS-specific extras:
  lvis_neg_verified bool  (True iff category is in `neg_category_ids`)
                           — for these, presence=False is gold-verified, not
                           just missing-annotation. Train absent-class loss on
                           these.
  lvis_not_exhaustive bool (True iff category is in `not_exhaustive_category_ids`)
                           — for these, presence=True but the mask may be
                           incomplete; usable but with caveat.

Negative-pair handling:
  - For COCO: emit a negative npz (presence=False, mask=zeros) for every
    category in --neg_categories list that is NOT present (since COCO does
    not enumerate guaranteed absents). With 80 categories this is ~80 neg
    pairs per image, all weakly-verified ("not in the 80 instance
    annotations" — usually but not always equal to "not present"). The
    presence_lookup-based FN filter handles the residual noise.
  - For LVIS: emit a positive npz when the category has instances; emit a
    GOLD-verified negative npz for every category in `neg_category_ids`;
    skip categories in `not_exhaustive_category_ids` UNLESS they have
    instances (in which case emit positive with `lvis_not_exhaustive=True`).

Usage:
  # Option A: COCO val2017 (already on disk)
  python -m scripts.generate_clean_targets \\
      --coco_ann data/coco_val2017/annotations/instances_val2017.json \\
      --image_dir data/coco_val2017/val2017 \\
      --output_dir results/clean_targets_val2017

  # Option B: COCO train2017 + LVIS train
  python -m scripts.generate_clean_targets \\
      --coco_ann data/coco_train2017/annotations/instances_train2017.json \\
      --lvis_ann data/coco_train2017/lvis_v1_train.json \\
      --image_dir data/coco_train2017/train2017 \\
      --output_dir results/clean_targets_train2017
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


def slugify(s: str) -> str:
    return s.lower().strip().replace(" ", "_").replace("/", "_")


def union_masks(masks: list[np.ndarray]) -> np.ndarray:
    if not masks: return None
    m = masks[0].astype(bool)
    for x in masks[1:]:
        m |= x.astype(bool)
    return m


def pool_to_14(mask_full: np.ndarray) -> np.ndarray:
    """Max-pool a (H, W) binary mask to (14, 14)."""
    t = torch.from_numpy(mask_full.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_max_pool2d(t, (14, 14)).squeeze().numpy()
    return (pooled > 0.5).astype(np.uint8) * 255


def write_npz(path: Path, *, img_id: str, query: str, presence: bool,
              mask_full: np.ndarray, mask14: np.ndarray, H: int, W: int,
              n_boxes: int, source: str,
              lvis_neg_verified: bool = False,
              lvis_not_exhaustive: bool = False):
    np.savez_compressed(
        path,
        img_id=img_id,
        query=query,
        presence=bool(presence),
        mask_full=mask_full.astype(np.uint8),
        mask14=mask14.astype(np.uint8),
        H=int(H), W=int(W),
        clip_sim=np.float32(1.0),
        gate_passed=bool(True),
        top_box_score=np.float32(1.0),
        n_boxes=int(n_boxes),
        source=source,
        lvis_neg_verified=bool(lvis_neg_verified),
        lvis_not_exhaustive=bool(lvis_not_exhaustive),
    )


def process_coco(args, ann_path: str, output_dir: Path):
    print(f"[coco] loading {ann_path} ...", flush=True)
    coco = COCO(ann_path)
    cat_ids = coco.getCatIds()
    cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(cat_ids)}
    cat_names = [cat_id_to_name[c] for c in cat_ids]
    print(f"  {len(cat_names)} categories: {cat_names[:6]} ...", flush=True)

    img_ids = sorted(coco.getImgIds())
    if args.image_limit:
        img_ids = img_ids[:args.image_limit]
    print(f"  {len(img_ids)} images", flush=True)

    n_pos = 0; n_neg = 0; n_skip = 0; t0 = time.time()
    for ix, img_id in enumerate(img_ids):
        info = coco.loadImgs(img_id)[0]
        H, W = info["height"], info["width"]
        img_id_str = f"{img_id:012d}"

        # Group annotations by category
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        by_cat: dict[int, list[np.ndarray]] = {}
        for a in anns:
            m = coco.annToMask(a)
            if m.sum() == 0: continue
            by_cat.setdefault(a["category_id"], []).append(m)

        for cat_id in cat_ids:
            cat_name = cat_id_to_name[cat_id]
            slug = slugify(cat_name)
            out_path = output_dir / f"{img_id_str}__{slug}.npz"
            if out_path.exists() and not args.overwrite:
                n_skip += 1
                continue

            if cat_id in by_cat:
                mask_full = union_masks(by_cat[cat_id])
                mask14 = pool_to_14(mask_full)
                write_npz(out_path,
                          img_id=img_id_str, query=cat_name, presence=True,
                          mask_full=(mask_full * 255), mask14=mask14,
                          H=H, W=W, n_boxes=len(by_cat[cat_id]),
                          source="coco")
                n_pos += 1
            else:
                # COCO doesn't verify absents, but writing a zero target gives
                # the trainer a balanced negative pool. The FN-filter handles
                # any residual noise from "missing annotation" rather than
                # "actually absent".
                if not args.skip_negatives:
                    mask_full = np.zeros((H, W), dtype=np.uint8)
                    mask14 = np.zeros((14, 14), dtype=np.uint8)
                    write_npz(out_path,
                              img_id=img_id_str, query=cat_name, presence=False,
                              mask_full=mask_full, mask14=mask14,
                              H=H, W=W, n_boxes=0, source="coco")
                    n_neg += 1

        if (ix + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (ix + 1) / elapsed
            eta = (len(img_ids) - ix - 1) / rate
            print(f"  [{ix+1}/{len(img_ids)}] pos={n_pos} neg={n_neg} skip={n_skip} "
                  f"rate={rate:.1f} img/s eta={eta/60:.1f} min", flush=True)

    print(f"[coco] done. pos={n_pos} neg={n_neg} skip={n_skip} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


def process_lvis(args, ann_path: str, output_dir: Path):
    print(f"[lvis] loading {ann_path} (~1 GB; takes ~30s) ...", flush=True)
    with open(ann_path) as f:
        lvis = json.load(f)
    print(f"  loaded: {len(lvis['images'])} images, {len(lvis['categories'])} cats, "
          f"{len(lvis['annotations'])} instance anns", flush=True)

    cats = {c["id"]: c for c in lvis["categories"]}
    images = {i["id"]: i for i in lvis["images"]}

    # IMPORTANT: LVIS val/train can contain images from BOTH COCO val2017 AND
    # train2017. Filter to only images that actually exist in args.image_dir
    # so we don't write npz files referencing missing images.
    image_dir_path = Path(args.image_dir)
    n_before = len(images)
    images = {k: v for k, v in images.items()
              if (image_dir_path / f"{k:012d}.jpg").exists()}
    n_filtered = n_before - len(images)
    print(f"  filtered out {n_filtered} images not present in {args.image_dir}; "
          f"keeping {len(images)}", flush=True)

    if args.image_limit:
        keep_ids = set(sorted(images.keys())[:args.image_limit])
        images = {k: v for k, v in images.items() if k in keep_ids}
        lvis["annotations"] = [a for a in lvis["annotations"] if a["image_id"] in keep_ids]

    # Index annotations by (image, category)
    by_img_cat: dict[tuple, list[dict]] = {}
    for a in lvis["annotations"]:
        by_img_cat.setdefault((a["image_id"], a["category_id"]), []).append(a)

    img_ids = sorted(images.keys())
    print(f"  processing {len(img_ids)} images", flush=True)

    n_pos = 0; n_neg = 0; n_skip = 0; n_drop = 0; t0 = time.time()
    for ix, img_id in enumerate(img_ids):
        info = images[img_id]
        H, W = info["height"], info["width"]
        # LVIS uses COCO file_name format (e.g. "000000397133.jpg"); we keep
        # the image_id as 12-digit zero-padded
        img_id_str = f"{img_id:012d}"
        neg_set = set(info.get("neg_category_ids", []))
        not_exh_set = set(info.get("not_exhaustive_category_ids", []))

        for cat_id, cat_meta in cats.items():
            cat_name = cat_meta["name"].replace("_(", " (")  # human-readable
            slug = slugify(cat_meta["name"])
            out_path = output_dir / f"{img_id_str}__lvis_{slug}.npz"
            if out_path.exists() and not args.overwrite:
                n_skip += 1
                continue

            anns_here = by_img_cat.get((img_id, cat_id), [])
            if anns_here:
                # Positive: union the instance masks
                masks = []
                for a in anns_here:
                    seg = a["segmentation"]
                    if isinstance(seg, list):
                        # polygon → RLE → mask
                        rles = mask_utils.frPyObjects(seg, H, W)
                        rle = mask_utils.merge(rles)
                        m = mask_utils.decode(rle)
                    elif isinstance(seg, dict):
                        # already RLE
                        if isinstance(seg.get("counts"), list):
                            rle = mask_utils.frPyObjects(seg, H, W)
                        else:
                            rle = seg
                        m = mask_utils.decode(rle)
                    else:
                        continue
                    if m.sum() == 0: continue
                    masks.append(m)
                if not masks: continue
                mask_full = union_masks(masks)
                mask14 = pool_to_14(mask_full)
                write_npz(out_path,
                          img_id=img_id_str, query=cat_name, presence=True,
                          mask_full=(mask_full * 255), mask14=mask14,
                          H=H, W=W, n_boxes=len(masks),
                          source="lvis",
                          lvis_not_exhaustive=(cat_id in not_exh_set))
                n_pos += 1
            elif cat_id in neg_set:
                # Gold-verified absent
                if not args.skip_negatives:
                    mask_full = np.zeros((H, W), dtype=np.uint8)
                    mask14 = np.zeros((14, 14), dtype=np.uint8)
                    write_npz(out_path,
                              img_id=img_id_str, query=cat_name, presence=False,
                              mask_full=mask_full, mask14=mask14,
                              H=H, W=W, n_boxes=0, source="lvis",
                              lvis_neg_verified=True)
                    n_neg += 1
            else:
                # Not annotated, not in neg list, not in not_exhaustive list
                # → ambiguous; LVIS' federated annotation does NOT certify
                # this. Skip (don't write the npz). The presence_lookup-
                # based FN filter at training time ensures off-diagonal
                # supervision uses only confirmed pairs.
                n_drop += 1

        if (ix + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (ix + 1) / elapsed
            eta = (len(img_ids) - ix - 1) / rate
            print(f"  [{ix+1}/{len(img_ids)}] pos={n_pos} neg={n_neg} drop_unverified={n_drop} "
                  f"skip={n_skip} rate={rate:.1f} img/s eta={eta/60:.1f} min", flush=True)

    print(f"[lvis] done. pos={n_pos} neg={n_neg} drop_unverified={n_drop} "
          f"skip={n_skip} in {(time.time()-t0)/60:.1f} min", flush=True)


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.coco_ann:
        process_coco(args, args.coco_ann, output_dir)
    if args.lvis_ann:
        process_lvis(args, args.lvis_ann, output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coco_ann", default=None,
                   help="Path to COCO instances JSON (val or train).")
    p.add_argument("--lvis_ann", default=None,
                   help="Path to LVIS v1 annotations JSON.")
    p.add_argument("--image_dir", required=True,
                   help="Directory of source JPEG images (used implicitly via train script).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--image_limit", type=int, default=None,
                   help="If set, process only the first N images (debug).")
    p.add_argument("--skip_negatives", action="store_true",
                   help="If set, only emit positive npz files. The trainer's BalancedSampler will draw negatives from the in-batch off-diagonal pairs only.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    main(args)
