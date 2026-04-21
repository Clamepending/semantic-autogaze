"""Cycle 1 (light-finetune): train one bilinear_cosine head per gaze_decoder layer.

Reads the (5, 196, 192) per-image cache from `cache_features_layers.py`,
trains a bilinear_cosine head on each layer K∈{0..4} independently, and
writes per-layer metrics + a summary.

Identical to the cycle-1 head/lr/seed of `r/autogaze-frozen-head` so the
layer-K=4 result here should reproduce that move's val-AUROC=0.7581
(within seed noise) — sanity check.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class BilinearCosineHead(nn.Module):
    def __init__(self, patch_dim=192, query_dim=512, proj_dim=128):
        super().__init__()
        self.patch_dim = patch_dim
        self.query_dim = query_dim
        self.proj_dim = proj_dim
        self.proj_p = nn.Linear(patch_dim, proj_dim, bias=False)
        self.proj_q = nn.Linear(query_dim, proj_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, patches, query):
        zp = F.normalize(self.proj_p(patches), dim=-1)
        zq = F.normalize(self.proj_q(query), dim=-1).unsqueeze(1)
        return (zp * zq).sum(-1) * self.scale + self.bias


class LayerDataset(Dataset):
    """Returns the K-th layer's patch features per (img, cat) pair."""
    def __init__(self, image_ids, sidecar, layer_cache_dir: Path, teacher_dir: Path,
                 clip_text: dict, layer_idx: int, preload_cache: dict | None = None):
        self.layer_cache_dir = layer_cache_dir
        self.teacher_dir = teacher_dir
        self.clip_text = clip_text
        self.layer_idx = layer_idx
        self.preload = preload_cache  # dict {img_id: (patches[K,N,D], teacher_dict)}
        self.pairs = []
        for img_id in image_ids:
            for c in sidecar.get(str(img_id), []):
                self.pairs.append((int(img_id), int(c)))
        if preload_cache is not None:
            self.pairs = [(i, c) for (i, c) in self.pairs if i in preload_cache]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_id, cat_id = self.pairs[idx]
        if self.preload is not None:
            patches_all, teacher = self.preload[img_id]
            patches = patches_all[self.layer_idx].float()
        else:
            patches_all = torch.load(self.layer_cache_dir / f"{img_id}.pt", weights_only=True)
            teacher = torch.load(self.teacher_dir / f"{img_id}.pt", weights_only=True)
            patches = patches_all[self.layer_idx].float()
        target = teacher[int(cat_id)].float().reshape(-1)
        query = self.clip_text[str(cat_id)].float()
        return {"patches": patches, "query": query, "target": target}


def _auroc(y, s):
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (float(ranks[y == 1].sum()) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def evaluate(head, loader, device):
    head.eval()
    aurocs, recall25, iou30, fg, bg, bces = [], [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            patches = batch["patches"].to(device)
            query = batch["query"].to(device)
            target = batch["target"].to(device)
            logits = head(patches, query)
            bces.append(F.binary_cross_entropy_with_logits(logits, target).item())
            probs = torch.sigmoid(logits).cpu().numpy()
            t = target.cpu().numpy()
            yfg = t >= 0.3
            pred = probs >= 0.5
            B, N = probs.shape
            k = max(1, int(round(0.25 * N)))
            for i in range(B):
                y = yfg[i]
                n_pos = int(y.sum()); n_neg = int((~y).sum())
                if n_pos and n_neg:
                    aurocs.append(_auroc(y.astype(np.int32), probs[i]))
                    top = np.argpartition(-probs[i], k - 1)[:k]
                    recall25.append(float(y[top].sum() / n_pos))
                inter = float((pred[i] & y).sum()); union = float((pred[i] | y).sum())
                iou30.append(inter / union if union > 0 else float("nan"))
                if n_pos: fg.append(float(probs[i][y].mean()))
                if n_neg: bg.append(float(probs[i][~y].mean()))
    return {
        "val_bce": float(np.mean(bces)),
        "auroc": float(np.nanmean(aurocs)),
        "recall_topK": float(np.nanmean(recall25)),
        "iou30": float(np.nanmean(iou30)),
        "fg_prob": float(np.nanmean(fg)),
        "bg_prob": float(np.nanmean(bg)),
    }


def preload_all(image_ids, layer_cache_dir, teacher_dir, max_workers=16):
    """Load all per-image (layer-stack, teacher) into RAM. fp16 to save space."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _load(img_id):
        fp = layer_cache_dir / f"{img_id}.pt"
        tp = teacher_dir / f"{img_id}.pt"
        if not fp.exists() or not tp.exists():
            return img_id, None
        patches = torch.load(fp, weights_only=True).half()
        t = torch.load(tp, weights_only=True)
        teach = {int(k): v.half() for k, v in t.items()}
        return img_id, (patches, teach)

    cache = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_load, int(i)) for i in image_ids]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="preload"):
            img_id, val = fut.result()
            if val is not None:
                cache[img_id] = val
    return cache


def train_one_layer(layer_idx: int, train_ids, val_ids, sidecar, layer_cache_dir,
                    teacher_dir, clip_text, preload_cache, args, device, out_dir: Path):
    print(f"\n========== layer {layer_idx} ==========")
    train_ds = LayerDataset(train_ids, sidecar, layer_cache_dir, teacher_dir, clip_text,
                            layer_idx, preload_cache)
    val_ds = LayerDataset(val_ids, sidecar, layer_cache_dir, teacher_dir, clip_text,
                          layer_idx, preload_cache)
    print(f"[diag] L{layer_idx}: {len(train_ds)} train pairs / {len(val_ds)} val pairs")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    torch.manual_seed(args.seed + layer_idx)  # vary seed per layer to reduce overlap
    head = BilinearCosineHead().to(device)
    n_params = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)

    history = []
    best_auroc = -1.0
    best_path = out_dir / f"best_layer{layer_idx}.pt"
    for ep in range(args.n_epochs):
        head.train()
        t0 = time.time()
        losses = []
        for batch in train_loader:
            patches = batch["patches"].to(device)
            query = batch["query"].to(device)
            target = batch["target"].to(device)
            logits = head(patches, query)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        m = evaluate(head, val_loader, device)
        m["train_bce"] = float(np.mean(losses)); m["epoch"] = ep + 1
        m["t_sec"] = time.time() - t0
        history.append(m)
        print(f"  L{layer_idx} ep{ep+1} train_bce={m['train_bce']:.4f} val_bce={m['val_bce']:.4f} "
              f"AUROC={m['auroc']:.4f} R@25={m['recall_topK']:.4f} "
              f"IoU30={m['iou30']:.4f} FG={m['fg_prob']:.3f} BG={m['bg_prob']:.3f} "
              f"({m['t_sec']:.0f}s)")
        if m["auroc"] > best_auroc:
            best_auroc = m["auroc"]
            torch.save({
                "state_dict": head.state_dict(),
                "config": {"patch_dim": head.patch_dim, "query_dim": head.query_dim,
                           "proj_dim": head.proj_dim},
                "layer_idx": layer_idx,
                "metrics": m,
            }, best_path)

    metrics_path = out_dir / f"metrics_layer{layer_idx}.json"
    with open(metrics_path, "w") as f:
        json.dump({"layer_idx": layer_idx, "history": history,
                   "best_auroc": best_auroc, "n_params": n_params}, f, indent=2)
    print(f"[diag] L{layer_idx}: best AUROC={best_auroc:.4f}; ckpt={best_path}")
    return {"layer_idx": layer_idx, "best_auroc": best_auroc,
            "best_iou30": max(h["iou30"] for h in history),
            "best_recall_topK": max(h["recall_topK"] for h in history)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer-cache-dir", default="results/autogaze_probe/features_gaze_layers_val")
    p.add_argument("--teacher-dir", default="results/autogaze_probe/teacher_14x14_val")
    p.add_argument("--clip-text", default="results/icon_student_B_native_train/clip_text_embeddings.pt")
    p.add_argument("--out-dir", default="results/autogaze_light_finetune/cycle1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--layers", default="0,1,2,3",
                   help="Comma-separated layer indices to train (0=after layer 0, 3=after layer 3 == last layer)")
    args = p.parse_args()

    rng = random.Random(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    layer_cache_dir = Path(args.layer_cache_dir)
    teacher_dir = Path(args.teacher_dir)

    clip_text = torch.load(args.clip_text, map_location="cpu", weights_only=True)
    with open(teacher_dir / "_sidecar.json") as f:
        sidecar = json.load(f)
    image_ids = [int(k) for k in sidecar.keys() if (layer_cache_dir / f"{k}.pt").exists()]
    rng.shuffle(image_ids)
    n_train = int(0.8 * len(image_ids))
    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:]
    print(f"[diag] {len(train_ids)} train images / {len(val_ids)} val images")

    print("[diag] preloading all (layer-stack, teacher) into RAM...")
    preload_cache = preload_all(image_ids, layer_cache_dir, teacher_dir)
    print(f"[diag] preloaded {len(preload_cache)} images")

    layers = [int(x) for x in args.layers.split(",")]
    summary = []
    for layer_idx in layers:
        s = train_one_layer(layer_idx, train_ids, val_ids, sidecar, layer_cache_dir,
                            teacher_dir, clip_text, preload_cache, args, device, out_dir)
        summary.append(s)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n========== summary ==========")
    print(f"{'layer':>5} {'auroc':>7} {'iou30':>7} {'R@25':>7}")
    for s in summary:
        print(f"{s['layer_idx']:>5} {s['best_auroc']:>7.4f} {s['best_iou30']:>7.4f} {s['best_recall_topK']:>7.4f}")
    best = max(summary, key=lambda s: s["best_auroc"])
    print(f"\n[winner] layer {best['layer_idx']} AUROC={best['best_auroc']:.4f}")


if __name__ == "__main__":
    main()
