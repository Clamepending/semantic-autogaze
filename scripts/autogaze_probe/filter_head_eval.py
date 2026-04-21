"""Evaluate the new ImageLevelFilterHead as a test-time filter.

Same framework as filter_use_eval.py (per-reduction ROC-AUC, AP, threshold sweep,
recall-target latency arithmetic) but uses the new head class. Adds a 7th
reduction `head_image` = the head's own trained image_logits aggregator (max-pool).

Eval is restricted to the val-holdout subset (1000 images, seed-0 split) so train
images aren't scored — keeps the metric honest.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools.coco import COCO
from tqdm import tqdm


class ImageLevelFilterHead(nn.Module):
    def __init__(self, patch_dim=192, text_dim=512, proj_dim=128, hidden=256,
                 aggregator="max", attn_temp=4.0):
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

    def head_image_score(self, log):
        if self.aggregator == "max":
            return log.max(dim=-1).values
        if self.aggregator == "mean":
            return log.mean(dim=-1)
        if self.aggregator == "attn":
            t = torch.exp(self.attn_log_temp)
            w = F.softmax(log / t, dim=-1)
            return (w * log).sum(dim=-1)
        if self.aggregator == "topk":
            k = max(1, log.shape[-1] // 20)
            top, _ = log.topk(k, dim=-1)
            return top.mean(dim=-1)
        raise ValueError(self.aggregator)


def load_head(ckpt: str, device):
    raw = torch.load(ckpt, map_location=device, weights_only=False)
    head = ImageLevelFilterHead(**raw["config"]).to(device)
    head.load_state_dict(raw["state_dict"])
    head.eval()
    layers = raw.get("layers", [3])
    return head, raw, layers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head-ckpt", default="results/filter_head_retrain/best.pt")
    p.add_argument("--layer-cache-dir", default="results/autogaze_probe/features_gaze_layers_val")
    p.add_argument("--clip-text", default="results/icon_student_B_native_train/clip_text_embeddings.pt")
    p.add_argument("--data-dir", default="data/coco")
    p.add_argument("--ann", default="annotations/instances_val2017.json")
    p.add_argument("--out-dir", default="results/filter_head_retrain")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all-images", action="store_true",
                   help="Eval on every cached image (train+val); reports a contamination-aware number")
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    layer_cache = Path(args.layer_cache_dir)

    print("[diag] loading head + clip_text + COCO ann...")
    head, raw, layer_idx = load_head(args.head_ckpt, device)
    print(f"[diag] head layers={layer_idx}")
    clip_text = torch.load(args.clip_text, map_location="cpu", weights_only=True)
    coco = COCO(os.path.join(args.data_dir, args.ann))

    cat_ids = sorted(int(k) for k in clip_text.keys())
    cat_id_to_col = {c: i for i, c in enumerate(cat_ids)}
    Q = len(cat_ids)
    text_mat = torch.stack([clip_text[str(c)].float() for c in cat_ids]).to(device)

    cached_imgs = sorted(int(p.stem) for p in layer_cache.glob("*.pt"))
    if args.all_images:
        eval_imgs = cached_imgs
        print(f"[diag] all-images mode: {len(eval_imgs)} (TRAIN-CONTAMINATED)")
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(cached_imgs))
        val_idx = perm[: args.n_val]
        eval_imgs = [cached_imgs[i] for i in sorted(val_idx)]
        print(f"[diag] val-holdout: {len(eval_imgs)} images (seed-{args.seed} split, no overlap with train)")

    REDUCTIONS = ["max", "mean", "top5_mean", "top10_mean", "logit_max", "logit_mean", "head_image"]
    all_scores = {r: np.zeros((len(eval_imgs), Q), dtype=np.float32) for r in REDUCTIONS}
    all_labels = np.zeros((len(eval_imgs), Q), dtype=np.bool_)

    t0 = time.time()
    with torch.inference_mode():
        for i, img_id in enumerate(tqdm(eval_imgs, desc="filter eval")):
            stack = torch.load(layer_cache / f"{img_id}.pt", weights_only=True)
            if len(layer_idx) == 1:
                patches = stack[layer_idx[0]].float().unsqueeze(0).to(device)
            else:
                patches = torch.cat([stack[k].float() for k in layer_idx], dim=-1).unsqueeze(0).to(device)
            logits = head.patch_logits(patches, text_mat)  # (1, Q, 196)
            probs = torch.sigmoid(logits)[0]              # (Q, 196)
            logits1 = logits[0]                            # (Q, 196)
            sorted_p, _ = probs.sort(dim=1, descending=True)
            all_scores["max"][i] = sorted_p[:, 0].cpu().numpy()
            all_scores["mean"][i] = probs.mean(dim=1).cpu().numpy()
            all_scores["top5_mean"][i] = sorted_p[:, :5].mean(dim=1).cpu().numpy()
            all_scores["top10_mean"][i] = sorted_p[:, :10].mean(dim=1).cpu().numpy()
            all_scores["logit_max"][i] = logits1.max(dim=1).values.cpu().numpy()
            all_scores["logit_mean"][i] = logits1.mean(dim=1).cpu().numpy()
            all_scores["head_image"][i] = head.head_image_score(logits1.unsqueeze(0))[0].cpu().numpy()
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
            present = {coco.anns[a]["category_id"] for a in ann_ids}
            for c in present:
                if c in cat_id_to_col:
                    all_labels[i, cat_id_to_col[c]] = True

    elapsed = time.time() - t0
    print(f"[diag] eval took {elapsed:.1f}s "
          f"({len(eval_imgs) * Q / elapsed:.0f} pairs/s)")

    flat_y = all_labels.flatten().astype(np.int32)
    n_pos = int(flat_y.sum()); n_neg = int(len(flat_y) - n_pos)
    pos_rate = n_pos / len(flat_y)
    print(f"[diag] pairs: {len(flat_y)}, pos: {n_pos} ({100*pos_rate:.2f}%), neg: {n_neg}")

    ms_filter = 12.4
    ms_118k = 93.6

    per_reduction = {}
    for red_name in REDUCTIONS:
        flat_s = all_scores[red_name].flatten()
        order = np.argsort(-flat_s)
        sorted_y = flat_y[order]
        cum_tp = np.cumsum(sorted_y)
        cum_fp = np.cumsum(1 - sorted_y)
        tpr = cum_tp / max(1, n_pos)
        fpr = cum_fp / max(1, n_neg)
        roc_auc = float(np.trapezoid(tpr, fpr))
        precision_at_k = cum_tp / np.arange(1, len(sorted_y) + 1)
        recall_at_k = tpr
        ap = float(np.sum((recall_at_k[1:] - recall_at_k[:-1]) * precision_at_k[1:]))

        s_min, s_max = float(flat_s.min()), float(flat_s.max())
        taus = np.unique(np.concatenate([
            np.linspace(s_min, s_max, 50),
            np.percentile(flat_s, np.linspace(0, 100, 50)),
        ]))
        rows = []
        for tau in taus:
            passed = flat_s >= tau
            tp = int((passed & (flat_y == 1)).sum())
            fp = int((passed & (flat_y == 0)).sum())
            recall = tp / max(1, n_pos)
            precision = tp / max(1, tp + fp)
            pass_rate = (tp + fp) / len(flat_y)
            filter_rate = (~passed & (flat_y == 0)).sum() / max(1, n_neg)
            rows.append({"tau": float(tau), "recall": float(recall),
                         "precision": float(precision),
                         "filter_rate_neg": float(filter_rate),
                         "pass_rate": float(pass_rate)})

        targets = []
        for target_recall in [0.90, 0.95, 0.99]:
            candidates = [r for r in rows if r["recall"] >= target_recall]
            if not candidates:
                targets.append({"target_recall": target_recall, "achievable": False})
                continue
            best = max(candidates, key=lambda r: r["tau"])
            ms_stacked = ms_filter + best["pass_rate"] * ms_118k
            speedup = ms_118k / ms_stacked
            targets.append({"target_recall": target_recall,
                            "tau": best["tau"], "recall": best["recall"],
                            "precision": best["precision"],
                            "pass_rate": best["pass_rate"],
                            "filter_rate_neg": best["filter_rate_neg"],
                            "ms_stacked": ms_stacked,
                            "speedup_vs_118k": speedup})
        per_reduction[red_name] = {"roc_auc": roc_auc, "average_precision": ap,
                                   "targets": targets,
                                   "score_min": s_min, "score_max": s_max}

    summary = {
        "n_images": len(eval_imgs), "n_cats": Q, "n_pairs": int(len(flat_y)),
        "n_pos": n_pos, "n_neg": n_neg, "pos_rate": float(pos_rate),
        "latency_ms": {"filter_only": ms_filter, "icon_118k": ms_118k},
        "head_ckpt": args.head_ckpt,
        "ckpt_meta": {k: raw[k] for k in ("epoch", "val_auc", "val_ap", "reduction")
                      if k in raw},
        "eval_mode": "all_images" if args.all_images else "val_holdout",
        "per_reduction": per_reduction,
    }
    out_name = "filter_eval_all.json" if args.all_images else "filter_eval_val.json"
    with open(out_dir / out_name, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[summary]")
    print(f"  baseline (always run 118K) = {ms_118k:.1f} ms/query")
    print(f"  filter-only (no 118K)     = {ms_filter:.1f} ms/query")
    print(f"  random baseline ROC-AUC = 0.5  AP = {pos_rate:.4f}")
    print(f"\n  {'reduction':<13}  {'ROC':>6}  {'AP':>6}  "
          f"{'r=90%':>14}  {'r=95%':>14}  {'r=99%':>14}")
    for red_name, r in per_reduction.items():
        targets_str = []
        for t in r["targets"]:
            if t.get("achievable", True):
                targets_str.append(f"{t['speedup_vs_118k']:.2f}x@{100*t['pass_rate']:.0f}%")
            else:
                targets_str.append("--")
        print(f"  {red_name:<13}  {r['roc_auc']:.4f}  {r['average_precision']:.4f}  "
              f"{targets_str[0]:>14}  {targets_str[1]:>14}  {targets_str[2]:>14}")

    print(f"\n[diag] wrote summary to {out_dir / out_name}")


if __name__ == "__main__":
    main()
