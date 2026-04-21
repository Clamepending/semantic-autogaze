"""r/criterion-bootstrap-gt cycle 1: unified re-audit of every AutoGaze+head ckpt
under the new RANKING CRITERION.

For each head spec:
  - Load head ckpt
  - For each of the 20 v11 audit pairs: compute IoU(pred>=0.5, GT>=0.5) and the
    same for the IconStudent-118K reference. Assign W/L/T (tie band 0.05).
  - Bench head_only_ms on M3 Pro CPU: 20 warmup + 100 timed runs on a cached
    feature tensor (L0 or L3 depending on the head). Excludes AutoGaze backbone.

Writes a single JSON with:
  { "heads": [
      { "name": ..., "ckpt": ..., "wins": .., "losses": .., "ties": ..,
        "head_ms_mean": .., "head_ms_p95": .. },
      ...
    ], "tie_band": 0.05, "n_pairs": 20, "device": "cpu" }

Run:
  .venv/bin/python scripts/criterion_bootstrap/audit_all.py
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from pathlib import Path

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
from semantic_autogaze.train_coco_seg import build_category_mask


FILENAME_RE = re.compile(r"^\d{3}_(\d+)_(.+)\.png$")


# ---------------------------------------------------------------------------
# Head classes (copied from their respective branches; self-contained here)
# ---------------------------------------------------------------------------

class BilinearCosineHead(nn.Module):
    """r/autogaze-frozen-head cycle 1/2 — 90,114 params on L3 patches."""
    def __init__(self, patch_dim=192, query_dim=512, proj_dim=128):
        super().__init__()
        self.proj_p = nn.Linear(patch_dim, proj_dim, bias=False)
        self.proj_q = nn.Linear(query_dim, proj_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def patch_logits(self, patches, query):
        # patches: (B, N, D) ; query: (Q, qd) -> (B, Q, N)
        zp = F.normalize(self.proj_p(patches), dim=-1)
        zq = F.normalize(self.proj_q(query), dim=-1)
        return torch.einsum("bnd,qd->bqn", zp, zq) * self.scale + self.bias


class ImageLevelFilterHead(nn.Module):
    """r/filter-head-retrain cycle 2 / r/grid-upscale-frozen / r/autogaze-deep-finetune."""
    def __init__(self, patch_dim=192, text_dim=512, proj_dim=128, hidden=256,
                 aggregator="attn", attn_temp=4.0):
        super().__init__()
        self.aggregator = aggregator
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, proj_dim),
        )
        self.text_proj = nn.Linear(text_dim, proj_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        if aggregator == "attn":
            self.attn_log_temp = nn.Parameter(torch.log(torch.tensor(attn_temp)))

    def patch_logits(self, patches, text):
        zp = F.normalize(self.patch_proj(patches), dim=-1)
        zt = F.normalize(self.text_proj(text), dim=-1)
        return torch.einsum("bnd,qd->bqn", zp, zt) * self.scale + self.bias


class PerPatchTinyHead(nn.Module):
    """r/per-patch-supervised-tinyhead — 1× TransformerEncoderLayer + cosine."""
    def __init__(self, patch_dim=192, text_dim=512, proj_dim=128,
                 n_heads=4, ff_dim=256, n_layers=1, dropout=0.0):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, proj_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.text_proj = nn.Linear(text_dim, proj_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def patch_logits(self, patches, text):
        zp = self.transformer(self.patch_proj(patches))
        zp = F.normalize(zp, dim=-1)
        zt = F.normalize(self.text_proj(text), dim=-1)
        return torch.einsum("bnd,qd->bqn", zp, zt) * self.scale + self.bias


class DeepUnfreezeBackbone(nn.Module):
    """r/autogaze-deep-finetune cycle 1 — 3 trainable LLaMA layers + frozen norm over L0."""
    def __init__(self, autogaze, n_patches=196):
        super().__init__()
        gm = autogaze.gazing_model
        self.layer1 = copy.deepcopy(gm.gaze_decoder.model.layers[1])
        self.layer2 = copy.deepcopy(gm.gaze_decoder.model.layers[2])
        self.layer3 = copy.deepcopy(gm.gaze_decoder.model.layers[3])
        for layer in (self.layer1, self.layer2, self.layer3):
            layer.self_attn.config._attn_implementation = "eager"
        self.norm = copy.deepcopy(gm.gaze_decoder.model.norm)
        self.rotary_emb = copy.deepcopy(gm.gaze_decoder.model.rotary_emb)
        cm = torch.full((n_patches, n_patches), float("-inf"))
        cm = torch.triu(cm, diagonal=1)
        self.register_buffer("causal_mask", cm.unsqueeze(0).unsqueeze(0))
        pos = torch.arange(n_patches).unsqueeze(0)
        self.register_buffer("position_ids", pos)
        self.n_patches = n_patches

    def forward(self, l0_patches: torch.Tensor) -> torch.Tensor:
        B = l0_patches.shape[0]
        position_ids = self.position_ids.expand(B, -1)
        cos, sin = self.rotary_emb(l0_patches, position_ids)
        attn_mask = self.causal_mask.expand(B, -1, -1, -1)
        x = l0_patches
        for layer in (self.layer1, self.layer2, self.layer3):
            out = layer(x, attention_mask=attn_mask, position_ids=position_ids,
                        position_embeddings=(cos, sin))
            x = out[0] if isinstance(out, tuple) else out
        return self.norm(x)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_bilinear_cosine(ckpt, device):
    raw = torch.load(ckpt, map_location=device, weights_only=False)
    h = BilinearCosineHead(**raw["config"]).to(device).eval()
    h.load_state_dict(raw["state_dict"])
    return h


def load_filter_head(ckpt, device):
    raw = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = raw["config"]
    # Some ckpts may store just the aggregator or all args; be permissive.
    h = ImageLevelFilterHead(**cfg).to(device).eval()
    h.load_state_dict(raw["state_dict"])
    target_hw = int(raw.get("target_hw", 14))
    return h, target_hw


def load_perpatch_tinyhead(ckpt, device):
    raw = torch.load(ckpt, map_location=device, weights_only=False)
    h = PerPatchTinyHead(**raw["config"]).to(device).eval()
    h.load_state_dict(raw["state_dict"])
    return h


def load_deep_finetune(ckpt, device, autogaze):
    raw = torch.load(ckpt, map_location=device, weights_only=False)
    backbone = DeepUnfreezeBackbone(autogaze).to(device).eval()
    head = ImageLevelFilterHead(**raw["head_config"]).to(device).eval()
    backbone.load_state_dict(raw["backbone_state_dict"])
    head.load_state_dict(raw["head_state_dict"])
    return backbone, head


def load_iconstudent(ckpt, device):
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


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def upscale_patches(patches_196: torch.Tensor, target_hw: int) -> torch.Tensor:
    B, N, D = patches_196.shape
    assert N == 196
    grid = patches_196.reshape(B, 14, 14, D).permute(0, 3, 1, 2)
    up = F.interpolate(grid, size=(target_hw, target_hw),
                       mode="bilinear", align_corners=False)
    return up.permute(0, 2, 3, 1).reshape(B, target_hw * target_hw, D)


def logits_to_heatmap_native(logits_flat, grid_hw, h, w):
    """logits_flat: (grid*grid,) -> prob map at native HxW."""
    g = grid_hw
    probs = torch.sigmoid(logits_flat).view(g, g).cpu().numpy()
    sq = F.interpolate(torch.from_numpy(probs).float()[None, None],
                       size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
    out = np.array(Image.fromarray((sq * 255).astype(np.uint8)).resize((w, h),
                                                                        Image.BILINEAR)) / 255.0
    return out.astype(np.float32)


def binarize_with_fallback(prob_map: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Criterion: threshold at 0.5 prob (== logit 0); if empty, fall back to per-image
    median so the IoU comparison is always well-defined."""
    mask = (prob_map >= thresh)
    if mask.sum() == 0:
        med = float(np.median(prob_map))
        mask = (prob_map >= med)
        if mask.sum() == 0:  # everything equal
            mask = np.zeros_like(prob_map, dtype=bool)
    return mask


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


def iou(pred, gt, thresh=0.5):
    p = binarize_with_fallback(pred, thresh)
    g = (gt >= 0.5)
    inter = float((p & g).sum())
    union = float((p | g).sum())
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/criterion_bootstrap/audit_all.json")
    p.add_argument("--data-dir", default="data/coco")
    p.add_argument("--ann", default="annotations/instances_val2017.json")
    p.add_argument("--img-subdir", default="val2017")
    p.add_argument("--qual-dir", default="results/compare_native_5k_vs_118k")
    p.add_argument("--layer-cache-dir", default="results/autogaze_probe/features_gaze_layers_val")
    p.add_argument("--dinov2-cache-native", default="results/dinov2_val_native")
    p.add_argument("--clip-text", default="results/icon_student_B_native_train/clip_text_embeddings.pt")
    p.add_argument("--iconstudent-ckpt", default="results/icon_student_B_native_train/best.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tie-band", type=float, default=0.05)
    p.add_argument("--bench-warmup", type=int, default=20)
    p.add_argument("--bench-runs", type=int, default=100)
    args = p.parse_args()

    device = torch.device(args.device)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Shared setup ---
    print("[diag] loading COCO + text...", flush=True)
    coco = COCO(os.path.join(args.data_dir, args.ann))
    img_dir = os.path.join(args.data_dir, args.img_subdir)
    name_to_id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    clip_text = torch.load(args.clip_text, map_location="cpu", weights_only=True)

    print("[diag] loading AutoGaze + IconStudent-118K...", flush=True)
    ag = load_autogaze(device=device)
    w_model = SemanticAutoGaze(ag).to(device).eval()
    icon = load_iconstudent(args.iconstudent_ckpt, device)

    # --- Parse the 20 audit pairs ---
    pairs = []
    for p_ in sorted(Path(args.qual_dir).iterdir()):
        m = FILENAME_RE.match(p_.name)
        if m:
            pairs.append((int(m.group(1)), m.group(2)))
    print(f"[diag] {len(pairs)} audit pairs", flush=True)

    # --- Preload L0/L3 cached features + GT masks + queries ---
    bundles = []
    for img_id, cat_slug in pairs:
        cat_name = cat_slug.replace("_", " ")
        if cat_name not in name_to_id:
            continue
        cat_id = name_to_id[cat_name]
        info = coco.imgs[img_id]
        img = Image.open(os.path.join(img_dir, info["file_name"])).convert("RGB")
        img_np = np.array(img); h, w = img_np.shape[:2]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=0))
        if not anns:
            continue
        gt = build_category_mask(coco, img_id, cat_id, anns).astype(np.float32)
        gt = np.clip(gt, 0, 1)
        query = clip_text[str(cat_id)].float()
        stack = torch.load(Path(args.layer_cache_dir) / f"{img_id}.pt", weights_only=True)
        l0 = stack[0].float()   # (196, 192)
        l3 = stack[3].float()   # (196, 192)
        nv = Path(args.dinov2_cache_native) / f"{img_id}.pt"
        if nv.exists():
            native_obj = torch.load(nv, map_location="cpu", weights_only=True)
            p_icon = predict_iconstudent_native(icon, native_obj, query, device, h, w)
        else:
            p_icon = np.zeros((h, w), dtype=np.float32)
        iou_icon = iou(p_icon, gt)
        bundles.append({
            "img_id": img_id, "cat": cat_name, "h": h, "w": w,
            "l0": l0, "l3": l3, "query": query, "gt": gt,
            "p_icon": p_icon, "iou_icon": iou_icon,
        })
    print(f"[diag] {len(bundles)} usable audit bundles", flush=True)

    # --- Head specs ---
    head_specs = [
        {"name": "frozen_cycle1", "kind": "bilinear_cosine",
         "ckpt": "results/autogaze_frozen_head/cycle1/best.pt"},
        {"name": "frozen_cycle2", "kind": "bilinear_cosine",
         "ckpt": "results/autogaze_frozen_head/cycle2/best.pt"},
        {"name": "filter_attn", "kind": "filter_head",
         "ckpt": "results/filter_head_retrain/best_attn.pt"},
        {"name": "perpatch_tinyhead", "kind": "perpatch_tinyhead",
         "ckpt": "results/perpatch_tinyhead/best.pt"},
        {"name": "upscale_28", "kind": "filter_head_upscale",
         "ckpt": "results/filter_head_upscale/best.pt"},
        {"name": "deep_finetune", "kind": "deep_finetune",
         "ckpt": "results/autogaze_deep_finetune/cycle1/best.pt"},
        {"name": "teacher_mse", "kind": "filter_head",
         "ckpt": "results/teacher_mse_distill/best.pt"},
        {"name": "coco_mask_distill", "kind": "filter_head",
         "ckpt": "results/coco_mask_distill/best.pt"},
        {"name": "multitask_bce_mse", "kind": "filter_head",
         "ckpt": "results/multitask_bce_mse/best.pt"},
    ]

    results = []

    for spec in head_specs:
        name = spec["name"]; kind = spec["kind"]; ckpt = spec["ckpt"]
        if not Path(ckpt).exists():
            print(f"[WARN] skipping {name} — missing ckpt {ckpt}", flush=True)
            continue
        print(f"\n=== {name} ({kind}) ===", flush=True)
        backbone = None
        if kind == "bilinear_cosine":
            head = load_bilinear_cosine(ckpt, device)
            feat_key = "l3"; target_hw = 14
        elif kind == "filter_head":
            head, target_hw = load_filter_head(ckpt, device)
            feat_key = "l3"
        elif kind == "perpatch_tinyhead":
            head = load_perpatch_tinyhead(ckpt, device)
            feat_key = "l3"; target_hw = 14
        elif kind == "filter_head_upscale":
            head, target_hw = load_filter_head(ckpt, device)
            feat_key = "l3"  # upscale applied inside
        elif kind == "deep_finetune":
            backbone, head = load_deep_finetune(ckpt, device, ag)
            feat_key = "l0"; target_hw = 14
        else:
            raise ValueError(kind)

        # --- per-pair IoU vs GT + verdict vs 118K ---
        wins = losses = ties = 0
        per_pair = []
        for b in bundles:
            feats = b[feat_key].unsqueeze(0).to(device)  # (1, 196, 192)
            with torch.inference_mode():
                if kind == "deep_finetune":
                    feats = backbone(feats)
                if kind == "filter_head_upscale":
                    feats_up = upscale_patches(feats, target_hw)
                    logits = head.patch_logits(feats_up, b["query"].unsqueeze(0).to(device))
                else:
                    logits = head.patch_logits(feats, b["query"].unsqueeze(0).to(device))
            p_head = logits_to_heatmap_native(logits[0, 0], target_hw, b["h"], b["w"])
            iou_head = iou(p_head, b["gt"])
            delta = iou_head - b["iou_icon"]
            if abs(delta) < args.tie_band:
                verdict = "T"; ties += 1
            elif delta > 0:
                verdict = "W"; wins += 1
            else:
                verdict = "L"; losses += 1
            per_pair.append({
                "img_id": b["img_id"], "cat": b["cat"],
                "iou_head": iou_head, "iou_icon": b["iou_icon"],
                "delta": delta, "verdict": verdict,
            })

        # --- head_only_ms bench on a representative cached feature ---
        sample_feat = bundles[0][feat_key].unsqueeze(0).to(device)
        sample_q = bundles[0]["query"].unsqueeze(0).to(device)
        for _ in range(args.bench_warmup):
            with torch.inference_mode():
                x = sample_feat
                if kind == "deep_finetune":
                    x = backbone(x)
                if kind == "filter_head_upscale":
                    x = upscale_patches(x, target_hw)
                _ = head.patch_logits(x, sample_q)
        head_times = []
        for _ in range(args.bench_runs):
            with torch.inference_mode():
                x = sample_feat
                t0 = time.perf_counter()
                if kind == "deep_finetune":
                    x = backbone(x)
                if kind == "filter_head_upscale":
                    x = upscale_patches(x, target_hw)
                _ = head.patch_logits(x, sample_q)
                t1 = time.perf_counter()
            head_times.append((t1 - t0) * 1000)
        ht = np.array(head_times)

        res = {
            "name": name, "kind": kind, "ckpt": ckpt,
            "wins": wins, "losses": losses, "ties": ties,
            "wins_plus_ties": wins + ties, "n": len(per_pair),
            "head_ms_mean": float(ht.mean()),
            "head_ms_p50": float(np.percentile(ht, 50)),
            "head_ms_p95": float(np.percentile(ht, 95)),
            "head_ms_std": float(ht.std()),
            "per_pair": per_pair,
        }
        results.append(res)
        print(f"  {name}: {wins}W/{losses}L/{ties}T = (W+T)/n = {wins+ties}/{len(per_pair)} | "
              f"head_ms mean={ht.mean():.2f} p95={np.percentile(ht,95):.2f}", flush=True)

    out = {
        "tie_band": args.tie_band, "n_pairs": len(bundles),
        "device": args.device, "heads": results,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {out_path}", flush=True)

    # --- compact summary table ---
    print("\n=== SUMMARY ===", flush=True)
    print(f"{'head':<22s} {'W+T/n':>8s} {'head_ms':>10s} {'clears ≥6/20':>14s} {'clears ≤5ms':>14s}", flush=True)
    for r in results:
        qual_pass = r["wins_plus_ties"] >= 6
        lat_pass = r["head_ms_mean"] <= 5.0
        print(f"{r['name']:<22s} {r['wins_plus_ties']:>3d}/{r['n']:<4d} "
              f"{r['head_ms_mean']:>8.2f}ms  {'YES' if qual_pass else 'no':>12s}  "
              f"{'YES' if lat_pass else 'no':>12s}", flush=True)


if __name__ == "__main__":
    main()
