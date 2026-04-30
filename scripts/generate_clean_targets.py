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


def process_coco(args, ann_path: str, output_dir: Path,
                 source_label: str = "coco", filename_prefix: str = ""):
    """Process a COCO-format JSON (instances OR stuff). Both share schema.
    `source_label` is written into npz['source']; `filename_prefix` is added
    in front of the slug to namespace stuff files (e.g., 'stuff_').
    """
    print(f"[{source_label}] loading {ann_path} ...", flush=True)
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
            slug = filename_prefix + slugify(cat_name)
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
                          source=source_label)
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
                              H=H, W=W, n_boxes=0, source=source_label)
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


def process_pascal_part(args, ann_dir: str, image_dir: str, output_dir: Path):
    """Process Pascal-Part .mat files. Pascal-Part is on VOC2010 images.

    Emits a unified `<voc_id>__pp_<query>.npz` per (image, body-part query).
    Body-part queries are aggregated from L/R variants:
      - 'hand'   = lhand U rhand (person only)
      - 'arm'    = luarm U ruarm U llarm U rlarm (person)
      - 'head'   = head (person, dog, cat, cow, sheep, horse, bird)
      - 'foot'   = lfoot U rfoot (person)
      - 'leg'    = luleg U ruleg U llleg U rlleg (person)
      - 'paw'    = lfpa U rfpa U lbpa U rbpa (dog, cat)
      - 'wing'   = lwing U rwing (bird, aeroplane)
      - 'wheel'  = wheel_* (car) or fwheel U bwheel (bicycle, motorbike)
      - 'screen' = screen (tvmonitor)
    Negative-pair handling: skip; the trainer's BalancedSampler will get
    negatives from cross-pair off-diagonals at training time.
    """
    import scipy.io as sio
    print(f"[pascal-part] scanning {ann_dir} ...", flush=True)
    ann_dir_path = Path(ann_dir)
    img_dir_path = Path(image_dir)
    files = sorted(ann_dir_path.glob("*.mat"))
    print(f"  {len(files)} .mat files", flush=True)
    if args.image_limit:
        files = files[:args.image_limit]

    QUERY_RULES = [
        ("hand",   "person", lambda n: n in ("lhand", "rhand")),
        ("arm",    "person", lambda n: n in ("luarm", "ruarm", "llarm", "rlarm")),
        ("head",   "person", lambda n: n == "head"),
        ("face",   "person", lambda n: n in ("leye", "reye", "lear", "rear",
                                              "nose", "mouth", "lebrow", "rebrow")),
        ("nose",   "person", lambda n: n == "nose"),
        ("mouth",  "person", lambda n: n == "mouth"),
        ("hair",   "person", lambda n: n == "hair"),
        ("torso",  "person", lambda n: n == "torso"),
        ("leg",    "person", lambda n: n in ("luleg", "ruleg", "llleg", "rlleg")),
        ("foot",   "person", lambda n: n in ("lfoot", "rfoot")),
        ("dog_head",  "dog", lambda n: n == "head"),
        ("dog_paw",   "dog", lambda n: n in ("lfpa", "rfpa", "lbpa", "rbpa")),
        ("cat_head",  "cat", lambda n: n == "head"),
        ("cat_paw",   "cat", lambda n: n in ("lfpa", "rfpa", "lbpa", "rbpa")),
        ("bird_wing", "bird", lambda n: n in ("lwing", "rwing")),
        ("bird_beak", "bird", lambda n: n == "beak"),
        ("car_wheel", "car",  lambda n: n.startswith("wheel")),
        ("bicycle_wheel", "bicycle", lambda n: n in ("fwheel", "bwheel")),
        ("aeroplane_wing", "aeroplane", lambda n: n in ("lwing", "rwing")),
        ("tv_screen", "tvmonitor", lambda n: n == "screen"),
    ]

    n_pos = 0; n_skip = 0; n_no_img = 0; t0 = time.time()
    for ix, f in enumerate(files):
        img_id = f.stem  # like "2008_000002"
        img_path = img_dir_path / f"{img_id}.jpg"
        if not img_path.exists():
            n_no_img += 1
            continue
        try:
            mat = sio.loadmat(f)
            anno = mat.get('anno')
            if anno is None: continue
            objects = anno[0,0]['objects']
        except Exception:
            continue

        # Get image dims from any mask we'll process
        H = None; W = None
        # Build per-query mask
        for query, target_cls, predicate in QUERY_RULES:
            slug = f"pp_{query}"
            out_path = output_dir / f"{img_id}__{slug}.npz"
            if out_path.exists() and not args.overwrite:
                n_skip += 1
                continue

            union = None
            for obj in objects[0]:
                cls = str(obj['class'][0]) if 'class' in obj.dtype.names else ''
                if cls != target_cls: continue
                parts = obj['parts'] if 'parts' in obj.dtype.names else None
                if parts is None or parts.size == 0: continue
                for p in parts[0]:
                    pname = str(p['part_name'][0])
                    if not predicate(pname): continue
                    pmask = p['mask']
                    if H is None: H, W = pmask.shape
                    if union is None:
                        union = pmask.astype(bool).copy()
                    else:
                        union |= pmask.astype(bool)

            if union is None: continue
            if int(union.sum()) < 50: continue
            mask14 = pool_to_14(union.astype(np.uint8) * 255)
            write_npz(out_path,
                      img_id=img_id, query=query.replace("_", " "), presence=True,
                      mask_full=(union.astype(np.uint8) * 255), mask14=mask14,
                      H=int(H), W=int(W), n_boxes=1, source="pascal_part")
            n_pos += 1

        if (ix + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (ix + 1) / elapsed
            eta = (len(files) - ix - 1) / rate
            print(f"  [{ix+1}/{len(files)}] pos={n_pos} no_img={n_no_img} "
                  f"skip={n_skip} rate={rate:.1f} img/s eta={eta/60:.1f} min", flush=True)

    print(f"[pascal-part] done. pos={n_pos} no_img={n_no_img} skip={n_skip} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


def process_panoptic(args, ann_path: str, png_dir: str, output_dir: Path):
    """Process COCO Panoptic annotations. Each segment becomes a contribution
    to a (image, category) mask. Multiple segments of the same category in
    the same image are unioned. Stuff classes are kept; thing classes are
    skipped here (they're already covered by COCO instances + LVIS).

    Since COCO Panoptic is on val2017 images that we already process via
    `process_coco(instances)`, we add ONLY stuff classes here to avoid
    duplicate npz writes.
    """
    print(f"[panoptic] loading {ann_path} ...", flush=True)
    with open(ann_path) as fh:
        pan = json.load(fh)
    cats_by_id = {c['id']: c for c in pan['categories']}
    img_by_id = {i['id']: i for i in pan['images']}
    print(f"  {len(pan['images'])} images, {len(pan['categories'])} cats "
          f"({sum(c['isthing']==1 for c in pan['categories'])} thing, "
          f"{sum(c['isthing']==0 for c in pan['categories'])} stuff)", flush=True)

    image_dir_path = Path(args.image_dir)
    png_dir_path = Path(png_dir)

    n_pos = 0; n_neg = 0; n_skip = 0; n_no_img = 0; t0 = time.time()
    annotations = pan['annotations']
    if args.image_limit:
        annotations = annotations[:args.image_limit]

    for ix, ann in enumerate(annotations):
        img_id = ann['image_id']
        info = img_by_id[img_id]
        H, W = info['height'], info['width']
        img_id_str = f"{img_id:012d}"
        if not (image_dir_path / f"{img_id_str}.jpg").exists():
            n_no_img += 1
            continue
        png_path = png_dir_path / ann['file_name']
        if not png_path.exists():
            n_no_img += 1
            continue
        png = np.array(Image.open(png_path).convert("RGB"))
        ids = png[..., 0].astype(np.int64) + (png[..., 1].astype(np.int64) << 8) + (png[..., 2].astype(np.int64) << 16)

        # Group segments by category — keep stuff only
        present_cats: dict[int, list[int]] = {}
        for seg in ann['segments_info']:
            cid = seg['category_id']
            if cats_by_id[cid].get('isthing', 0) == 1:
                continue  # things are covered by COCO instances
            present_cats.setdefault(cid, []).append(seg['id'])

        # Emit positives + negatives for stuff cats
        stuff_cat_ids = [c['id'] for c in pan['categories'] if c['isthing'] == 0]
        for cid in stuff_cat_ids:
            cat_meta = cats_by_id[cid]
            cat_name = cat_meta['name']
            slug = f"stuff_{slugify(cat_name)}"
            out_path = output_dir / f"{img_id_str}__{slug}.npz"
            if out_path.exists() and not args.overwrite:
                n_skip += 1
                continue
            if cid in present_cats:
                # Union all segments of this stuff class
                seg_ids = present_cats[cid]
                mask = np.zeros((H, W), dtype=bool)
                for sid in seg_ids:
                    mask |= (ids == sid)
                if not mask.any(): continue
                mask14 = pool_to_14(mask.astype(np.uint8) * 255)
                write_npz(out_path,
                          img_id=img_id_str, query=cat_name, presence=True,
                          mask_full=(mask.astype(np.uint8) * 255), mask14=mask14,
                          H=H, W=W, n_boxes=len(seg_ids), source="panoptic")
                n_pos += 1
            else:
                if not args.skip_negatives:
                    mask_full = np.zeros((H, W), dtype=np.uint8)
                    mask14 = np.zeros((14, 14), dtype=np.uint8)
                    write_npz(out_path,
                              img_id=img_id_str, query=cat_name, presence=False,
                              mask_full=mask_full, mask14=mask14,
                              H=H, W=W, n_boxes=0, source="panoptic")
                    n_neg += 1

        if (ix + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (ix + 1) / elapsed
            eta = (len(annotations) - ix - 1) / rate
            print(f"  [{ix+1}/{len(annotations)}] pos={n_pos} neg={n_neg} "
                  f"skip={n_skip} no_img={n_no_img} rate={rate:.1f} img/s eta={eta/60:.1f} min",
                  flush=True)

    print(f"[panoptic] done. pos={n_pos} neg={n_neg} skip={n_skip} no_img={n_no_img} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


def process_ade20k(args, ade_root: str, output_dir: Path):
    """Process ADE20K Challenge 2016 — 150 categories, pixel-perfect labels.

    Layout (after unzip):
      <ade_root>/objectInfo150.txt       — class list
      <ade_root>/images/{training,validation}/ADE_*.jpg
      <ade_root>/annotations/{training,validation}/ADE_*.png   pixel value = class id (0..150)

    Emits npz with prefix `ade_<slug>` so it doesn't collide with COCO/LVIS.
    Image filename is ADE_train_00000001 (not 12-digit COCO id), so we'll
    keep the bare stem as img_id and require the trainer's --image_dir to
    be a unified directory containing these jpgs.
    """
    print(f"[ade20k] loading {ade_root} ...", flush=True)
    ade_path = Path(ade_root)
    classes = []
    with open(ade_path / "objectInfo150.txt") as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5:
                idx = int(parts[0])
                name = parts[4].split(",")[0].strip()
                classes.append((idx, name))
    print(f"  {len(classes)} categories", flush=True)
    cls_id_to_name = {idx: n for idx, n in classes}

    splits = ["training", "validation"]
    n_pos = 0; n_skip = 0; t0 = time.time()
    for split in splits:
        ann_dir = ade_path / "annotations" / split
        img_dir = ade_path / "images" / split
        ann_files = sorted(ann_dir.glob("*.png"))
        if args.image_limit:
            ann_files = ann_files[:args.image_limit]
        print(f"[ade20k:{split}] {len(ann_files)} images", flush=True)
        for ix, ann_f in enumerate(ann_files):
            img_id = ann_f.stem  # ADE_train_00000001
            img_f = img_dir / f"{img_id}.jpg"
            if not img_f.exists(): continue
            ann = np.array(Image.open(ann_f))
            H, W = ann.shape
            present_ids = np.unique(ann)
            present_ids = present_ids[present_ids > 0]
            for cls_id in present_ids:
                cls_id = int(cls_id)
                if cls_id not in cls_id_to_name: continue
                name = cls_id_to_name[cls_id]
                slug = f"ade_{slugify(name)}"
                out_path = output_dir / f"{img_id}__{slug}.npz"
                if out_path.exists() and not args.overwrite:
                    n_skip += 1
                    continue
                mask_full = (ann == cls_id).astype(np.uint8) * 255
                if mask_full.sum() == 0: continue
                mask14 = pool_to_14(mask_full)
                write_npz(out_path,
                          img_id=img_id, query=name, presence=True,
                          mask_full=mask_full, mask14=mask14,
                          H=H, W=W, n_boxes=1, source="ade20k")
                n_pos += 1
            if (ix + 1) % 1000 == 0:
                el = time.time() - t0
                print(f"  [ade:{split} {ix+1}/{len(ann_files)}] pos={n_pos} skip={n_skip} "
                      f"rate={(ix+1)/max(1,el):.1f} img/s", flush=True)
    print(f"[ade20k] done. pos={n_pos} skip={n_skip} in {(time.time()-t0)/60:.1f} min", flush=True)


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.ade20k_root:
        process_ade20k(args, args.ade20k_root, output_dir)
    if args.coco_ann:
        process_coco(args, args.coco_ann, output_dir,
                     source_label="coco", filename_prefix="")
    if args.stuff_ann:
        process_coco(args, args.stuff_ann, output_dir,
                     source_label="stuff", filename_prefix="stuff_")
    if args.lvis_ann:
        process_lvis(args, args.lvis_ann, output_dir)
    if args.panoptic_ann and args.panoptic_png_dir:
        process_panoptic(args, args.panoptic_ann, args.panoptic_png_dir, output_dir)
    if args.pascal_part_ann_dir and args.pascal_part_image_dir:
        process_pascal_part(args, args.pascal_part_ann_dir,
                            args.pascal_part_image_dir, output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coco_ann", default=None,
                   help="Path to COCO instances JSON (val or train).")
    p.add_argument("--stuff_ann", default=None,
                   help="Path to COCO-Stuff annotations JSON (val or train).")
    p.add_argument("--lvis_ann", default=None,
                   help="Path to LVIS v1 annotations JSON.")
    p.add_argument("--panoptic_ann", default=None,
                   help="Path to COCO Panoptic JSON.")
    p.add_argument("--panoptic_png_dir", default=None,
                   help="Directory of Panoptic PNG masks (e.g. annotations/panoptic_val2017/).")
    p.add_argument("--pascal_part_ann_dir", default=None,
                   help="Pascal-Part Annotations_Part directory.")
    p.add_argument("--pascal_part_image_dir", default=None,
                   help="VOC2010 JPEGImages directory.")
    p.add_argument("--ade20k_root", default=None,
                   help="ADEChallengeData2016 root (containing objectInfo150.txt + images/ + annotations/).")
    p.add_argument("--image_dir", required=True,
                   help="Directory of source JPEG images for COCO/LVIS/Panoptic processing.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--image_limit", type=int, default=None,
                   help="If set, process only the first N images (debug).")
    p.add_argument("--skip_negatives", action="store_true",
                   help="If set, only emit positive npz files. The trainer's BalancedSampler will draw negatives from the in-batch off-diagonal pairs only.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    main(args)
