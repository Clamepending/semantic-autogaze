"""Measure IconStudent-118K end-to-end ms/query on M3 Pro CPU.

Counts the full deployable path: image -> DINOv2 native-aspect features
-> 9.5M cross-attn decoder -> heatmap. CLIP text query is cached
per-category and assumed already-resident. Uses the same 10 audit images
as bench_and_audit.py so the numbers are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools.coco import COCO

from semantic_autogaze.icon_student import IconStudent


FILENAME_RE = re.compile(r"^\d{3}_(\d+)_(.+)\.png$")


def load_iconstudent(ckpt: str, device: torch.device) -> IconStudent:
    raw = torch.load(ckpt, map_location=device, weights_only=False)
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


@torch.no_grad()
def native_dinov2(dinov2_model, image_pil, device):
    """Reproduce native-aspect cache pipeline at inference time."""
    w0, h0 = image_pil.size
    short = min(w0, h0)
    scale = 224.0 / short
    new_w = int(round(w0 * scale))
    new_h = int(round(h0 * scale))
    img = image_pil.resize((new_w, new_h), Image.BICUBIC)
    enc_w = (new_w // 14) * 14
    enc_h = (new_h // 14) * 14
    left = (new_w - enc_w) // 2
    top = (new_h - enc_h) // 2
    img = img.crop((left, top, left + enc_w, top + enc_h))
    t = TF.to_tensor(img)
    t = TF.normalize(t, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    t = t.unsqueeze(0).to(device)
    n_h = enc_h // 14; n_w = enc_w // 14
    out = dinov2_model.forward_features(t)
    patches = out["x_norm_patchtokens"][0].contiguous()  # (n_h*n_w, 384)
    return patches, (n_h, n_w)


def parse_qual_dir(qual_dir: Path):
    pairs = []
    for p in sorted(qual_dir.iterdir()):
        m = FILENAME_RE.match(p.name)
        if m:
            pairs.append((int(m.group(1)), m.group(2)))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/icon_student_B_native_train/best.pt")
    p.add_argument("--clip-text", default="results/icon_student_B_native_train/clip_text_embeddings.pt")
    p.add_argument("--qual-dir", default="results/compare_native_5k_vs_118k")
    p.add_argument("--data-dir", default="data/coco")
    p.add_argument("--ann", default="annotations/instances_val2017.json")
    p.add_argument("--img-subdir", default="val2017")
    p.add_argument("--out", default="results/autogaze_frozen_head/cycle1/bench_iconstudent.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--n-runs", type=int, default=50)
    p.add_argument("--n-images", type=int, default=10)
    args = p.parse_args()

    device = torch.device(args.device)
    coco = COCO(os.path.join(args.data_dir, args.ann))
    img_dir = os.path.join(args.data_dir, args.img_subdir)
    name_to_id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    clip_text = torch.load(args.clip_text, map_location="cpu", weights_only=True)

    print("[diag] loading DINOv2-small...")
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                            verbose=False).to(device).eval()

    print(f"[diag] loading IconStudent-118K: {args.ckpt}")
    icon = load_iconstudent(args.ckpt, device)

    pairs = parse_qual_dir(Path(args.qual_dir))[: args.n_images]
    print(f"[diag] benchmarking {len(pairs)} images x ({args.n_warmup} warmup + {args.n_runs} runs)")

    per_image = []
    for img_id, cat_slug in pairs:
        info = coco.imgs[img_id]
        img = Image.open(os.path.join(img_dir, info["file_name"])).convert("RGB")
        cat_name = cat_slug.replace("_", " ")
        if cat_name not in name_to_id:
            continue
        cat_id = name_to_id[cat_name]
        query = clip_text[str(cat_id)].float().unsqueeze(0).to(device)

        for _ in range(args.n_warmup):
            patches, grid = native_dinov2(dinov2, img, device)
            _ = icon(patches.unsqueeze(0).to(device), query, grid_hw=grid)

        d_times, h_times, t_times = [], [], []
        for _ in range(args.n_runs):
            t0 = time.perf_counter()
            patches, grid = native_dinov2(dinov2, img, device)
            t1 = time.perf_counter()
            logits = icon(patches.unsqueeze(0).to(device), query, grid_hw=grid)
            _ = torch.sigmoid(logits)
            t2 = time.perf_counter()
            d_times.append((t1 - t0) * 1000)
            h_times.append((t2 - t1) * 1000)
            t_times.append((t2 - t0) * 1000)

        def stats(xs):
            a = np.array(xs)
            return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
                    "p95": float(np.percentile(a, 95))}
        rec = {"img_id": img_id, "cat": cat_name,
               "dinov2_ms": stats(d_times), "decoder_ms": stats(h_times),
               "total_ms": stats(t_times)}
        per_image.append(rec)
        print(f"  img {img_id} cat={cat_name:14s} total={rec['total_ms']['mean']:.1f}ms "
              f"(dinov2={rec['dinov2_ms']['mean']:.1f} decoder={rec['decoder_ms']['mean']:.1f})")

    agg = {"dinov2_ms_mean": float(np.mean([r["dinov2_ms"]["mean"] for r in per_image])),
           "decoder_ms_mean": float(np.mean([r["decoder_ms"]["mean"] for r in per_image])),
           "total_ms_mean": float(np.mean([r["total_ms"]["mean"] for r in per_image])),
           "total_ms_p50": float(np.mean([r["total_ms"]["p50"] for r in per_image])),
           "total_ms_p95": float(np.mean([r["total_ms"]["p95"] for r in per_image])),
           "per_image": per_image}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[diag] iconstudent-118k: total={agg['total_ms_mean']:.1f}ms "
          f"(dinov2={agg['dinov2_ms_mean']:.1f}, decoder={agg['decoder_ms_mean']:.1f})")
    print(f"[diag] wrote {args.out}")


if __name__ == "__main__":
    main()
