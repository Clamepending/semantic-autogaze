"""r/autogaze-frozen-head cycle 1: end-to-end ms/query benchmark + 20-pair
qual audit panels (frozen-head vs IconStudent-118K).

Benchmark:
    100 cold + 100 warm runs on M3 Pro CPU. Reports mean / p50 / p95 ms
    per (image, query), broken down into:
        - autogaze (vision_model + connector + gaze_decoder)
        - head (bilinear cosine + scale/bias)
        - upsample (14x14 -> 224 -> native HxW)

Qual audit:
    Re-uses the 20-pair set from `results/compare_native_5k_vs_118k/`.
    Renders 4-panel figures: [image | COCO GT | AutoGaze+frozen-head | IconStudent-118K].
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools.coco import COCO

from semantic_autogaze.icon_student import IconStudent
from semantic_autogaze.inference import load_autogaze
from semantic_autogaze.model import SemanticAutoGaze
from semantic_autogaze.train_coco_seg import build_category_mask, get_image_categories


FILENAME_RE = re.compile(r"^\d{3}_(\d+)_(.+)\.png$")


# Re-declare the head class (matches training script).
class BilinearCosineHead(nn.Module):
    def __init__(self, patch_dim=192, query_dim=512, proj_dim=128):
        super().__init__()
        self.proj_p = nn.Linear(patch_dim, proj_dim, bias=False)
        self.proj_q = nn.Linear(query_dim, proj_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, patches, query):
        zp = F.normalize(self.proj_p(patches), dim=-1)
        zq = F.normalize(self.proj_q(query), dim=-1).unsqueeze(1)
        return (zp * zq).sum(-1) * self.scale + self.bias


def load_head(ckpt: str, device: torch.device) -> BilinearCosineHead:
    raw = torch.load(ckpt, map_location=device, weights_only=False)
    head = BilinearCosineHead(**raw["config"]).to(device)
    head.load_state_dict(raw["state_dict"])
    head.eval()
    return head


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


@torch.inference_mode()
def predict_iconstudent_native(model, native_obj, query, device, h, w):
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
    if og_bot - og_top <= 0 or og_right - og_left <= 0:
        return np.zeros((h, w), dtype=np.float32)
    region = F.interpolate(
        torch.from_numpy(probs).float()[None, None],
        size=(og_bot - og_top, og_right - og_left),
        mode="bilinear", align_corners=False)[0, 0].numpy()
    out = np.zeros((h, w), dtype=np.float32)
    out[og_top:og_bot, og_left:og_right] = np.clip(region, 0, 1)
    return out


@torch.inference_mode()
def predict_frozen_head(image_pil, query, autogaze_wrapper, head, device, h, w):
    """Full forward path being timed in benchmark."""
    t = TF.to_tensor(image_pil.resize((224, 224))).unsqueeze(0).unsqueeze(0).to(device)
    hidden = autogaze_wrapper.get_patch_hidden_states(t)  # (1, 196, 192)
    logits = head(hidden, query.unsqueeze(0).to(device))   # (1, 196)
    probs = torch.sigmoid(logits)[0].view(14, 14).cpu().numpy()
    sq = F.interpolate(
        torch.from_numpy(probs).float()[None, None], size=(224, 224),
        mode="bilinear", align_corners=False)[0, 0].numpy()
    out = np.array(Image.fromarray((sq * 255).astype(np.uint8)).resize((w, h),
                                                                       Image.BILINEAR)) / 255.0
    return out.astype(np.float32)


def parse_qual_dir(qual_dir: Path):
    pairs = []
    for p in sorted(qual_dir.iterdir()):
        m = FILENAME_RE.match(p.name)
        if m:
            pairs.append((int(m.group(1)), m.group(2)))
    return pairs


def overlay(img_np, gray, cmap):
    rgb = (cmap(gray)[..., :3] * 255).astype(np.uint8)
    return (0.5 * img_np.astype(np.float32) + 0.5 * rgb).clip(0, 255).astype(np.uint8)


@torch.inference_mode()
def benchmark(autogaze_wrapper, head, image_pil, query, device, n_warmup=20, n_runs=100):
    """Time the frozen-head forward path with sub-step breakdown."""
    pre_t = TF.to_tensor(image_pil.resize((224, 224))).unsqueeze(0).unsqueeze(0).to(device)
    q = query.unsqueeze(0).to(device)
    # Warmup
    for _ in range(n_warmup):
        h_ = autogaze_wrapper.get_patch_hidden_states(pre_t)
        _ = head(h_, q)

    ag_times, head_times, total_times = [], [], []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        hidden = autogaze_wrapper.get_patch_hidden_states(pre_t)
        t1 = time.perf_counter()
        logits = head(hidden, q)
        _ = torch.sigmoid(logits)
        t2 = time.perf_counter()
        ag_times.append((t1 - t0) * 1000)
        head_times.append((t2 - t1) * 1000)
        total_times.append((t2 - t0) * 1000)

    def stats(xs):
        a = np.array(xs)
        return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95))}
    return {"autogaze_ms": stats(ag_times), "head_ms": stats(head_times),
            "total_ms": stats(total_times), "n_runs": n_runs, "n_warmup": n_warmup}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head-ckpt", default="results/autogaze_frozen_head/cycle1/best.pt")
    p.add_argument("--iconstudent-ckpt", default="results/icon_student_B_native_train/best.pt")
    p.add_argument("--qual-dir", default="results/compare_native_5k_vs_118k")
    p.add_argument("--dinov2-cache-native", default="results/dinov2_val_native")
    p.add_argument("--clip-text", default="results/icon_student_B_native_train/clip_text_embeddings.pt")
    p.add_argument("--data-dir", default="data/coco")
    p.add_argument("--ann", default="annotations/instances_val2017.json")
    p.add_argument("--img-subdir", default="val2017")
    p.add_argument("--output-dir", default="results/autogaze_frozen_head/cycle1/audit")
    p.add_argument("--bench-out", default="results/autogaze_frozen_head/cycle1/bench.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--bench-runs", type=int, default=100)
    p.add_argument("--bench-warmup", type=int, default=20)
    p.add_argument("--bench-images", type=int, default=10)
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(os.path.join(args.data_dir, args.ann))
    img_dir = os.path.join(args.data_dir, args.img_subdir)
    name_to_id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    clip_text = torch.load(args.clip_text, map_location="cpu", weights_only=True)

    print("[diag] loading frozen head...")
    head = load_head(args.head_ckpt, device)

    print("[diag] loading AutoGaze...")
    ag = load_autogaze(device=device)
    w = SemanticAutoGaze(ag).to(device).eval()

    print("[diag] loading IconStudent-118K...")
    icon = load_iconstudent(args.iconstudent_ckpt, device)

    pairs = parse_qual_dir(Path(args.qual_dir))
    print(f"[diag] {len(pairs)} audit pairs")

    # ----- benchmark over first N audit images -----
    bench_pairs = pairs[: args.bench_images]
    bench_results = []
    print(f"[diag] benchmarking on {len(bench_pairs)} images "
          f"({args.bench_warmup} warmup + {args.bench_runs} runs each)...")
    for img_id, cat_slug in bench_pairs:
        info = coco.imgs[img_id]
        img = Image.open(os.path.join(img_dir, info["file_name"])).convert("RGB")
        cat_name = cat_slug.replace("_", " ")
        if cat_name not in name_to_id:
            continue
        cat_id = name_to_id[cat_name]
        query = clip_text[str(cat_id)].float()
        b = benchmark(w, head, img, query, device,
                      n_warmup=args.bench_warmup, n_runs=args.bench_runs)
        b["img_id"] = img_id
        b["cat"] = cat_name
        bench_results.append(b)
        print(f"  img {img_id} cat={cat_name:14s} total={b['total_ms']['mean']:.1f}ms "
              f"(autogaze={b['autogaze_ms']['mean']:.1f} head={b['head_ms']['mean']:.2f})")

    agg = {"autogaze_ms_mean": float(np.mean([r["autogaze_ms"]["mean"] for r in bench_results])),
           "head_ms_mean": float(np.mean([r["head_ms"]["mean"] for r in bench_results])),
           "total_ms_mean": float(np.mean([r["total_ms"]["mean"] for r in bench_results])),
           "total_ms_p50": float(np.mean([r["total_ms"]["p50"] for r in bench_results])),
           "total_ms_p95": float(np.mean([r["total_ms"]["p95"] for r in bench_results])),
           "per_image": bench_results}
    Path(args.bench_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.bench_out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[diag] benchmark agg: total={agg['total_ms_mean']:.1f}ms "
          f"(autogaze={agg['autogaze_ms_mean']:.1f}, head={agg['head_ms_mean']:.3f})")
    print(f"[diag] wrote {args.bench_out}")

    # ----- qual audit panels -----
    cmap = matplotlib.colormaps["jet"]
    summary = []
    for idx, (img_id, cat_slug) in enumerate(pairs):
        cat_name = cat_slug.replace("_", " ")
        if cat_name not in name_to_id:
            continue
        cat_id = name_to_id[cat_name]
        info = coco.imgs[img_id]
        img = Image.open(os.path.join(img_dir, info["file_name"])).convert("RGB")
        img_np = np.array(img); h, w_im = img_np.shape[:2]

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=0))
        if not anns:
            continue
        gt = build_category_mask(coco, img_id, cat_id, anns).astype(np.float32)
        gt = np.clip(gt, 0, 1)
        query = clip_text[str(cat_id)].float()

        p_frozen = predict_frozen_head(img, query, w, head, device, h, w_im)
        nv = Path(args.dinov2_cache_native) / f"{img_id}.pt"
        if nv.exists():
            native_obj = torch.load(nv, map_location="cpu", weights_only=True)
            p_icon = predict_iconstudent_native(icon, native_obj, query, device, h, w_im)
        else:
            p_icon = np.zeros_like(p_frozen)

        other = [coco.cats[c]["name"] for c in get_image_categories(coco, img_id).keys() if c != cat_id]
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        axes[0].imshow(img_np); axes[0].set_title(f"image {img_id}"); axes[0].axis("off")
        axes[1].imshow(overlay(img_np, gt, cmap)); axes[1].set_title(f"COCO GT \"{cat_name}\""); axes[1].axis("off")
        axes[2].imshow(overlay(img_np, p_frozen, cmap)); axes[2].set_title(f"AutoGaze+head (max={p_frozen.max():.2f})"); axes[2].axis("off")
        axes[3].imshow(overlay(img_np, p_icon, cmap)); axes[3].set_title(f"IconStudent-118K (max={p_icon.max():.2f})"); axes[3].axis("off")
        others = ", ".join(other[:8]) + ("…" if len(other) > 8 else "")
        plt.suptitle(f"query: \"{cat_name}\" | also present: {others}", fontsize=12)
        plt.tight_layout()
        out_path = out_dir / f"{idx:03d}_{img_id}_{cat_slug}.png"
        plt.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close()
        summary.append({"idx": idx, "img_id": img_id, "cat": cat_name,
                        "out": str(out_path),
                        "max_frozen": float(p_frozen.max()),
                        "max_118k": float(p_icon.max())})
        print(f"  {idx:03d}: {cat_name:14s} on {img_id} -> {out_path.name}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[diag] wrote {len(summary)} audit panels to {out_dir}")


if __name__ == "__main__":
    main()
