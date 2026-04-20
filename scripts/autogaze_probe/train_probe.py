"""Cycle 1: train 3 probe variants on cached AutoGaze gaze-decoder features
and report per-patch FG-vs-BG ranking quality.

Probes (capacity ladder):
    - BilinearCosine: cos(W_p patch, W_q text), both projected to dim 128.
    - ConcatMLP:      (patch || text) -> 256 -> 256 -> 1.
    - CrossAttn:      1-block IconStudent-style cross-attn at decoder_dim=128.

Inputs (read from disk):
    - results/autogaze_probe/features_gaze_val/{img_id}.pt -> (196, 192) float
    - results/autogaze_probe/teacher_14x14_val/{img_id}.pt -> dict {cat_id: (14,14)}
    - results/icon_student_B_native_train/clip_text_embeddings.pt -> dict {str(cat_id): (512,)}

Metrics (per (image, query) pair, then averaged):
    - AUROC: per-patch FG-vs-BG (FG = teacher cell prob >= 0.3).
    - Recall@top-25%: of patches ranked by probe score, fraction of FG patches
      captured by the top quartile.
    - IoU@0.3: thresholded probe heatmap vs thresholded teacher heatmap.
    - FG/BG mean prob and gap.

Train/val: 5000 image_ids -> shuffle(seed=42) -> 4000 train / 1000 val.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# -------------------------- probe definitions --------------------------

class BilinearCosineProbe(nn.Module):
    def __init__(self, patch_dim=192, query_dim=512, proj_dim=128):
        super().__init__()
        self.proj_p = nn.Linear(patch_dim, proj_dim, bias=False)
        self.proj_q = nn.Linear(query_dim, proj_dim, bias=False)
        # Logit calibration so cosine similarity (range [-1,1]) maps onto a
        # usable BCE logit range.
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, patches, query):
        # patches: (B, N, P), query: (B, Q)
        zp = F.normalize(self.proj_p(patches), dim=-1)
        zq = F.normalize(self.proj_q(query), dim=-1).unsqueeze(1)  # (B, 1, D)
        cos = (zp * zq).sum(-1)  # (B, N)
        return cos * self.scale + self.bias


class ConcatMLPProbe(nn.Module):
    def __init__(self, patch_dim=192, query_dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(patch_dim + query_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, patches, query):
        B, N, _ = patches.shape
        q = query.unsqueeze(1).expand(-1, N, -1)
        return self.net(torch.cat([patches, q], dim=-1)).squeeze(-1)


class _CrossAttnBlock(nn.Module):
    def __init__(self, dim, n_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, patches, query):
        h = self.norm_self(patches)
        a, _ = self.self_attn(h, h, h, need_weights=False)
        patches = patches + a
        hp = self.norm_cross(patches)
        hq = self.norm_q(query)
        c, _ = self.cross_attn(hp, hq, hq, need_weights=False)
        patches = patches + c
        return patches + self.mlp(self.norm_mlp(patches))


class CrossAttnProbe(nn.Module):
    def __init__(self, patch_dim=192, query_dim=512, dim=128, n_heads=4):
        super().__init__()
        self.in_p = nn.Linear(patch_dim, dim)
        self.in_q = nn.Linear(query_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, 196, dim) * 0.02)
        self.block = _CrossAttnBlock(dim, n_heads=n_heads)
        self.out = nn.Linear(dim, 1)

    def forward(self, patches, query):
        x = self.in_p(patches) + self.pos
        q = self.in_q(query).unsqueeze(1)
        x = self.block(x, q)
        return self.out(x).squeeze(-1)


# -------------------------- dataset --------------------------

class ProbeDataset(Dataset):
    """One sample = (image_id, cat_id) pair. Returns (patches, query, target).

    `target` is the teacher 14x14 heatmap flattened to 196.
    """

    def __init__(self, image_ids, sidecar, feature_dir: Path, teacher_dir: Path,
                 clip_text: dict):
        self.feature_dir = feature_dir
        self.teacher_dir = teacher_dir
        self.clip_text = clip_text
        self.pairs = []
        for img_id in image_ids:
            cats = sidecar.get(str(img_id), [])
            for c in cats:
                self.pairs.append((int(img_id), int(c)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_id, cat_id = self.pairs[idx]
        patches = torch.load(self.feature_dir / f"{img_id}.pt", weights_only=True).float()
        teacher = torch.load(self.teacher_dir / f"{img_id}.pt", weights_only=True)
        target = teacher[int(cat_id)].float().reshape(-1)  # (196,)
        query = self.clip_text[str(cat_id)].float()
        return {"patches": patches, "query": query, "target": target,
                "img_id": img_id, "cat_id": cat_id}


# -------------------------- metrics --------------------------

def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney rank AUROC. y_true in {0,1}, scores any real."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks over ties (rare with continuous logits, cheap to skip).
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = float(ranks[y_true == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _per_pair_metrics(probe_logits: torch.Tensor, target: torch.Tensor,
                      thr: float = 0.3, top_k_frac: float = 0.25):
    """Compute AUROC, recall@top-K%, IoU@thr, FG/BG mean prob, per pair.

    probe_logits, target: (B, 196) on CPU.
    Returns dict of lists (one entry per pair, NaN if undefined).
    """
    probs = torch.sigmoid(probe_logits).numpy()
    target_np = target.numpy()
    fg = target_np >= thr  # (B, 196) bool
    pred_bin = probs >= 0.5

    out = {"auroc": [], "recall_topK": [], "iou30": [], "fg_prob": [], "bg_prob": []}
    B, N = probs.shape
    k = max(1, int(round(top_k_frac * N)))
    for i in range(B):
        y = fg[i]
        n_fg = int(y.sum()); n_bg = int((~y).sum())
        if n_fg == 0 or n_bg == 0:
            out["auroc"].append(float("nan"))
            out["recall_topK"].append(float("nan"))
        else:
            out["auroc"].append(_auroc(y.astype(np.int32), probs[i]))
            top_idx = np.argpartition(-probs[i], k - 1)[:k]
            out["recall_topK"].append(float(y[top_idx].sum() / n_fg))
        # IoU@thr against the teacher-binarized mask
        inter = float((pred_bin[i] & y).sum())
        union = float((pred_bin[i] | y).sum())
        out["iou30"].append(inter / union if union > 0 else float("nan"))
        out["fg_prob"].append(float(probs[i][y].mean()) if n_fg > 0 else float("nan"))
        out["bg_prob"].append(float(probs[i][~y].mean()) if n_bg > 0 else float("nan"))
    return out


def _agg(d: dict) -> dict:
    return {k: float(np.nanmean(v)) if len(v) else float("nan") for k, v in d.items()}


# -------------------------- train loop --------------------------

def train_one(probe, train_loader, val_loader, device, n_epochs=5, lr=1e-3,
              log_prefix="probe"):
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    best_val = float("inf")
    best_metrics = None
    for ep in range(n_epochs):
        probe.train()
        t0 = time.time()
        losses = []
        for batch in tqdm(train_loader, desc=f"[{log_prefix}] ep{ep+1}", leave=False):
            patches = batch["patches"].to(device)
            query = batch["query"].to(device)
            target = batch["target"].to(device)
            logits = probe(patches, query)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        probe.eval()
        agg = {"auroc": [], "recall_topK": [], "iou30": [], "fg_prob": [], "bg_prob": []}
        val_losses = []
        with torch.inference_mode():
            for batch in val_loader:
                patches = batch["patches"].to(device)
                query = batch["query"].to(device)
                target = batch["target"].to(device)
                logits = probe(patches, query)
                val_losses.append(F.binary_cross_entropy_with_logits(logits, target).item())
                m = _per_pair_metrics(logits.cpu(), target.cpu())
                for k, v in m.items():
                    agg[k].extend(v)
        val_bce = float(np.mean(val_losses))
        metrics = _agg(agg)
        print(f"  [{log_prefix}] ep{ep+1} train_bce={np.mean(losses):.4f} "
              f"val_bce={val_bce:.4f} AUROC={metrics['auroc']:.4f} "
              f"R@25={metrics['recall_topK']:.4f} IoU30={metrics['iou30']:.4f} "
              f"FG={metrics['fg_prob']:.3f} BG={metrics['bg_prob']:.3f} "
              f"gap={metrics['fg_prob']-metrics['bg_prob']:.3f} "
              f"({time.time()-t0:.0f}s)")
        if val_bce < best_val:
            best_val = val_bce
            best_metrics = {"epoch": ep + 1, "val_bce": val_bce, **metrics}
    return best_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-dir", default="results/autogaze_probe/features_gaze_val")
    p.add_argument("--teacher-dir", default="results/autogaze_probe/teacher_14x14_val")
    p.add_argument("--clip-text", default="results/icon_student_B_native_train/clip_text_embeddings.pt")
    p.add_argument("--out", default="results/autogaze_probe/cycle1_metrics.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    feature_dir = Path(args.feature_dir)
    teacher_dir = Path(args.teacher_dir)
    sidecar_path = teacher_dir / "_sidecar.json"
    print(f"[diag] reading sidecar {sidecar_path}")
    with open(sidecar_path) as f:
        sidecar = json.load(f)

    clip_text = torch.load(args.clip_text, map_location="cpu", weights_only=True)

    image_ids = [int(k) for k in sidecar.keys()]
    image_ids = [i for i in image_ids if (feature_dir / f"{i}.pt").exists()]
    rng.shuffle(image_ids)
    n_train = int(0.8 * len(image_ids))
    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:]
    print(f"[diag] {len(image_ids)} images -> {len(train_ids)} train / {len(val_ids)} val")

    train_ds = ProbeDataset(train_ids, sidecar, feature_dir, teacher_dir, clip_text)
    val_ds = ProbeDataset(val_ids, sidecar, feature_dir, teacher_dir, clip_text)
    print(f"[diag] {len(train_ds)} train pairs / {len(val_ds)} val pairs")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            persistent_workers=args.num_workers > 0)

    results = {}
    for name, probe_cls in [
        ("bilinear_cosine", BilinearCosineProbe),
        ("concat_mlp", ConcatMLPProbe),
        ("cross_attn", CrossAttnProbe),
    ]:
        probe = probe_cls().to(device)
        n_params = sum(p.numel() for p in probe.parameters())
        print(f"\n=== {name} ({n_params:,} params) ===")
        best = train_one(probe, train_loader, val_loader, device,
                         n_epochs=args.n_epochs, log_prefix=name)
        best["n_params"] = n_params
        results[name] = best

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[diag] wrote {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
