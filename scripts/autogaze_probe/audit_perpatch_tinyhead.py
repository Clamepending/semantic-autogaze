"""r/per-patch-supervised-tinyhead: 20-pair qual audit + bench for PerPatchTinyHead's
per-patch heatmap, vs IconStudent-118K.

Architecture: AutoGaze L3 patches -> patch_proj (192->128) -> 1x TransformerEncoderLayer
-> cosine with text -> per-patch logits. Trained per-patch BCE against IconStudent-118K
teacher heatmaps on val2017 cache.

Produces 4-panel figures: [image | COCO GT | AutoGaze+tinyhead per-patch | IconStudent-118K].
Reports per-pair IoU (heatmap > 0.5 vs GT) for both methods, W/L/T tally with tie band 0.05.
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


class PerPatchTinyHead(nn.Module):
    def __init__(self, patch_dim=192, text_dim=512, proj_dim=128,
                 n_heads=4, ff_dim=256, n_layers=1, dropout=0.0):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, proj_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.text_proj = nn.Linear(text_dim, proj_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def patch_logits(self, patches, text):
        # patches: (B, 196, patch_dim) ; text: (Q, text_dim) -> (B, Q, 196)
        zp = self.transformer(self.patch_proj(patches))
        zp = F.normalize(zp, dim=-1)
        zt = F.normalize(self.text_proj(text), dim=-1)
        return torch.einsum("bnd,qd->bqn", zp, zt) * self.scale + self.bias


def load_tinyhead(ckpt: str, device):
    raw = torch.load(ckpt, map_location=device, weights_only=False)
    head = PerPatchTinyHead(**raw["config"]).to(device)
    head.load_state_dict(raw["state_dict"])
    head.eval()
    return head


def load_iconstudent(ckpt: str, device):
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
def predict_tinyhead(image_pil, query, autogaze_wrapper, head, device, h, w):
    """End-to-end: 224x224 squash -> AutoGaze L3 -> tinyhead transformer -> per-patch sigmoid -> upsample."""
    t = TF.to_tensor(image_pil.resize((224, 224))).unsqueeze(0).unsqueeze(0).to(device)
    hidden = autogaze_wrapper.get_patch_hidden_states(t)  # (1, 196, 192)
    logits = head.patch_logits(hidden, query.unsqueeze(0).to(device))  # (1, 1, 196)
    probs = torch.sigmoid(logits)[0, 0].view(14, 14).cpu().numpy()
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


def iou(pred: np.ndarray, gt: np.ndarray, thresh: float = 0.5) -> float:
    p = (pred >= thresh)
    g = (gt >= 0.5)
    inter = float((p & g).sum())
    union = float((p | g).sum())
    return inter / union if union > 0 else 0.0


@torch.inference_mode()
def benchmark(autogaze_wrapper, head, image_pil, query, device, n_warmup=20, n_runs=100):
    pre_t = TF.to_tensor(image_pil.resize((224, 224))).unsqueeze(0).unsqueeze(0).to(device)
    q = query.unsqueeze(0).to(device)
    for _ in range(n_warmup):
        h_ = autogaze_wrapper.get_patch_hidden_states(pre_t)
        _ = head.patch_logits(h_, q)
    ag_times, head_times, total_times = [], [], []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        hidden = autogaze_wrapper.get_patch_hidden_states(pre_t)
        t1 = time.perf_counter()
        logits = head.patch_logits(hidden, q)
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
    p.add_argument("--head-ckpt", default="results/perpatch_tinyhead/best.pt")
    p.add_argument("--iconstudent-ckpt", default="results/icon_student_B_native_train/best.pt")
    p.add_argument("--qual-dir", default="results/compare_native_5k_vs_118k")
    p.add_argument("--dinov2-cache-native", default="results/dinov2_val_native")
    p.add_argument("--clip-text", default="results/icon_student_B_native_train/clip_text_embeddings.pt")
    p.add_argument("--data-dir", default="data/coco")
    p.add_argument("--ann", default="annotations/instances_val2017.json")
    p.add_argument("--img-subdir", default="val2017")
    p.add_argument("--output-dir", default="results/perpatch_tinyhead/audit")
    p.add_argument("--bench-out", default="results/perpatch_tinyhead/bench.json")
    p.add_argument("--summary-out", default="results/perpatch_tinyhead/summary.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--bench-runs", type=int, default=100)
    p.add_argument("--bench-warmup", type=int, default=20)
    p.add_argument("--bench-images", type=int, default=10)
    p.add_argument("--tie-band", type=float, default=0.05)
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.bench_out).parent.mkdir(parents=True, exist_ok=True)

    coco = COCO(os.path.join(args.data_dir, args.ann))
    img_dir = os.path.join(args.data_dir, args.img_subdir)
    name_to_id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    clip_text = torch.load(args.clip_text, map_location="cpu", weights_only=True)

    print("[diag] loading filter head + AutoGaze + IconStudent-118K...")
    head = load_tinyhead(args.head_ckpt, device)
    ag = load_autogaze(device=device)
    w_model = SemanticAutoGaze(ag).to(device).eval()
    icon = load_iconstudent(args.iconstudent_ckpt, device)

    pairs = parse_qual_dir(Path(args.qual_dir))
    print(f"[diag] {len(pairs)} audit pairs")

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
        b = benchmark(w_model, head, img, query, device,
                      n_warmup=args.bench_warmup, n_runs=args.bench_runs)
        b["img_id"] = img_id; b["cat"] = cat_name
        bench_results.append(b)
        print(f"  img {img_id} cat={cat_name:14s} total={b['total_ms']['mean']:.1f}ms "
              f"(autogaze={b['autogaze_ms']['mean']:.1f} head={b['head_ms']['mean']:.2f})")

    agg = {"autogaze_ms_mean": float(np.mean([r["autogaze_ms"]["mean"] for r in bench_results])),
           "head_ms_mean": float(np.mean([r["head_ms"]["mean"] for r in bench_results])),
           "total_ms_mean": float(np.mean([r["total_ms"]["mean"] for r in bench_results])),
           "total_ms_p50": float(np.mean([r["total_ms"]["p50"] for r in bench_results])),
           "total_ms_p95": float(np.mean([r["total_ms"]["p95"] for r in bench_results])),
           "per_image": bench_results}
    with open(args.bench_out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[diag] benchmark agg: total={agg['total_ms_mean']:.1f}ms "
          f"(autogaze={agg['autogaze_ms_mean']:.1f}, head={agg['head_ms_mean']:.3f})")

    # ----- qual audit panels + IoU -----
    cmap = matplotlib.colormaps["jet"]
    summary = []
    wins = losses = ties = skipped = 0
    for idx, (img_id, cat_slug) in enumerate(pairs):
        cat_name = cat_slug.replace("_", " ")
        if cat_name not in name_to_id:
            skipped += 1; continue
        cat_id = name_to_id[cat_name]
        info = coco.imgs[img_id]
        img = Image.open(os.path.join(img_dir, info["file_name"])).convert("RGB")
        img_np = np.array(img); h, w_im = img_np.shape[:2]

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=0))
        if not anns:
            skipped += 1; continue
        gt = build_category_mask(coco, img_id, cat_id, anns).astype(np.float32)
        gt = np.clip(gt, 0, 1)
        query = clip_text[str(cat_id)].float()

        p_filter = predict_tinyhead(img, query, w_model, head, device, h, w_im)
        nv = Path(args.dinov2_cache_native) / f"{img_id}.pt"
        if nv.exists():
            native_obj = torch.load(nv, map_location="cpu", weights_only=True)
            p_icon = predict_iconstudent_native(icon, native_obj, query, device, h, w_im)
        else:
            p_icon = np.zeros_like(p_filter)

        iou_filter = iou(p_filter, gt)
        iou_icon = iou(p_icon, gt)
        delta = iou_filter - iou_icon
        if abs(delta) < args.tie_band:
            verdict = "T"; ties += 1
        elif delta > 0:
            verdict = "W"; wins += 1
        else:
            verdict = "L"; losses += 1

        other = [coco.cats[c]["name"] for c in get_image_categories(coco, img_id).keys() if c != cat_id]
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        axes[0].imshow(img_np); axes[0].set_title(f"image {img_id}"); axes[0].axis("off")
        axes[1].imshow(overlay(img_np, gt, cmap)); axes[1].set_title(f"COCO GT \"{cat_name}\""); axes[1].axis("off")
        axes[2].imshow(overlay(img_np, p_filter, cmap))
        axes[2].set_title(f"AutoGaze+filter (IoU={iou_filter:.3f}, max={p_filter.max():.2f})"); axes[2].axis("off")
        axes[3].imshow(overlay(img_np, p_icon, cmap))
        axes[3].set_title(f"IconStudent-118K (IoU={iou_icon:.3f}, max={p_icon.max():.2f})"); axes[3].axis("off")
        others = ", ".join(other[:8]) + ("…" if len(other) > 8 else "")
        plt.suptitle(f"query: \"{cat_name}\" | also present: {others} | verdict: {verdict} (Δ={delta:+.3f})", fontsize=12)
        plt.tight_layout()
        out_path = out_dir / f"{idx:03d}_{img_id}_{cat_slug}_{verdict}.png"
        plt.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close()
        summary.append({"idx": idx, "img_id": img_id, "cat": cat_name,
                        "out": str(out_path),
                        "iou_filter": float(iou_filter), "iou_icon": float(iou_icon),
                        "delta": float(delta), "verdict": verdict})
        print(f"  {idx:03d}: {cat_name:14s} on {img_id}: filter={iou_filter:.3f} icon={iou_icon:.3f} -> {verdict}")

    summary_obj = {"head_ckpt": args.head_ckpt,
                   "tie_band": args.tie_band,
                   "wins": wins, "losses": losses, "ties": ties, "skipped": skipped,
                   "summary_line": f"{wins}W/{losses}L/{ties}T (skipped {skipped})",
                   "per_pair": summary,
                   "bench_summary_ms": {k: agg[k] for k in ("total_ms_mean", "total_ms_p50",
                                                             "total_ms_p95",
                                                             "autogaze_ms_mean", "head_ms_mean")}}
    with open(args.summary_out, "w") as f:
        json.dump(summary_obj, f, indent=2)

    print(f"\n[summary] {wins}W / {losses}L / {ties}T (skipped {skipped})")
    print(f"  bench: total={agg['total_ms_mean']:.1f}ms  autogaze={agg['autogaze_ms_mean']:.1f}ms  "
          f"head={agg['head_ms_mean']:.3f}ms")
    print(f"  panels: {out_dir}")


if __name__ == "__main__":
    main()
