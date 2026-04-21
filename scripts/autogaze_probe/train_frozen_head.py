"""r/autogaze-frozen-head cycle 1: train bilinear_cosine head on val2017
8715 (img, cat) pairs (same as feasibility probe), save checkpoint, write
val metrics JSON.

Inputs:
    results/autogaze_probe/features_gaze_val/{img_id}.pt -> (196, 192) float
    results/autogaze_probe/teacher_14x14_val/{img_id}.pt -> dict {cat_id: (14,14)}
    results/icon_student_B_native_train/clip_text_embeddings.pt

Outputs:
    results/autogaze_frozen_head/cycle1/best.pt
    results/autogaze_frozen_head/cycle1/metrics.json
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


class ProbeDataset(Dataset):
    def __init__(self, image_ids, sidecar, feature_dir: Path, teacher_dir: Path,
                 clip_text: dict, preload: bool = False):
        self.feature_dir = feature_dir
        self.teacher_dir = teacher_dir
        self.clip_text = clip_text
        self.pairs = []
        for img_id in image_ids:
            for c in sidecar.get(str(img_id), []):
                self.pairs.append((int(img_id), int(c)))
        self.preload = preload
        self.feat_cache = None
        self.teach_cache = None
        if preload:
            self._do_preload(image_ids)

    def _do_preload(self, image_ids):
        # Load every (feature, teacher) file once at startup. Modal Volume's
        # ~250ms/file random-access latency dominates batched training; once
        # loaded into a process-local dict, __getitem__ is pure RAM.
        # Stored as fp16 to fit 118K x 196 x 192 features in ~8.7GB RAM.
        # Parallelized: serial 118K * 250ms = ~8 hr; 64 threads ~ 8 min.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm as _tqdm
        self.feat_cache = {}
        self.teach_cache = {}

        def _load_one(img_id: int):
            fp = self.feature_dir / f"{img_id}.pt"
            tp = self.teacher_dir / f"{img_id}.pt"
            if not fp.exists() or not tp.exists():
                return img_id, None, None
            patches = torch.load(fp, weights_only=True).half()
            t = torch.load(tp, weights_only=True)
            teach = {int(k): v.half() for k, v in t.items()}
            return img_id, patches, teach

        with ThreadPoolExecutor(max_workers=64) as ex:
            futures = [ex.submit(_load_one, int(i)) for i in image_ids]
            for fut in _tqdm(as_completed(futures), total=len(futures), desc="preload"):
                img_id, patches, teach = fut.result()
                if patches is None:
                    continue
                self.feat_cache[img_id] = patches
                self.teach_cache[img_id] = teach
        # Drop pairs whose images failed to preload.
        self.pairs = [(i, c) for (i, c) in self.pairs if i in self.feat_cache and i in self.teach_cache]
        print(f"[preload] {len(self.feat_cache)} images, {len(self.pairs)} pairs in cache", flush=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_id, cat_id = self.pairs[idx]
        if self.preload:
            patches = self.feat_cache[img_id].float()
            teacher = self.teach_cache[img_id]
        else:
            patches = torch.load(self.feature_dir / f"{img_id}.pt", weights_only=True).float()
            teacher = torch.load(self.teacher_dir / f"{img_id}.pt", weights_only=True)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-dir", default="results/autogaze_probe/features_gaze_val")
    p.add_argument("--teacher-dir", default="results/autogaze_probe/teacher_14x14_val")
    p.add_argument("--val-feature-dir", default=None,
                   help="If set, train on (feature-dir, teacher-dir) and val on these.")
    p.add_argument("--val-teacher-dir", default=None)
    p.add_argument("--clip-text", default="results/icon_student_B_native_train/clip_text_embeddings.pt")
    p.add_argument("--out-dir", default="results/autogaze_frozen_head/cycle1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--preload", action="store_true",
                   help="Load all features+teachers into RAM at startup. "
                        "Use on Modal Volumes where per-file latency dominates.")
    args = p.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    feature_dir = Path(args.feature_dir)
    teacher_dir = Path(args.teacher_dir)
    with open(teacher_dir / "_sidecar.json") as f:
        sidecar = json.load(f)
    clip_text = torch.load(args.clip_text, map_location="cpu", weights_only=True)

    image_ids = [int(k) for k in sidecar.keys() if (feature_dir / f"{k}.pt").exists()]
    rng.shuffle(image_ids)

    if args.val_feature_dir and args.val_teacher_dir:
        val_feature_dir = Path(args.val_feature_dir)
        val_teacher_dir = Path(args.val_teacher_dir)
        with open(val_teacher_dir / "_sidecar.json") as f:
            val_sidecar = json.load(f)
        val_ids = [int(k) for k in val_sidecar.keys()
                   if (val_feature_dir / f"{k}.pt").exists()]
        train_ids = image_ids
        train_ds = ProbeDataset(train_ids, sidecar, feature_dir, teacher_dir, clip_text,
                                preload=args.preload)
        val_ds = ProbeDataset(val_ids, val_sidecar, val_feature_dir, val_teacher_dir, clip_text,
                              preload=args.preload)
    else:
        n_train = int(0.8 * len(image_ids))
        train_ids, val_ids = image_ids[:n_train], image_ids[n_train:]
        train_ds = ProbeDataset(train_ids, sidecar, feature_dir, teacher_dir, clip_text,
                                preload=args.preload)
        val_ds = ProbeDataset(val_ids, sidecar, feature_dir, teacher_dir, clip_text,
                              preload=args.preload)
    print(f"[diag] {len(train_ds)} train pairs / {len(val_ds)} val pairs "
          f"(over {len(train_ids)}/{len(val_ids)} images)")

    # When preloaded, __getitem__ is pure-RAM dict lookup; multi-worker forking
    # would copy the giant cache (CoW saves us in theory but glibc fragments it).
    # 0 workers is fastest in practice once I/O is removed.
    nw = 0 if args.preload else args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, drop_last=True,
                              persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw,
                            persistent_workers=nw > 0)

    head = BilinearCosineHead().to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[diag] bilinear_cosine head: {n_params:,} params")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    history = []
    best_auroc = -1.0
    best_path = out_dir / "best.pt"
    for ep in range(args.n_epochs):
        head.train()
        t0 = time.time()
        losses = []
        for batch in tqdm(train_loader, desc=f"ep{ep+1}", leave=False):
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
        print(f"  ep{ep+1} train_bce={m['train_bce']:.4f} val_bce={m['val_bce']:.4f} "
              f"AUROC={m['auroc']:.4f} R@25={m['recall_topK']:.4f} "
              f"IoU30={m['iou30']:.4f} FG={m['fg_prob']:.3f} BG={m['bg_prob']:.3f} "
              f"({m['t_sec']:.0f}s)")
        if m["auroc"] > best_auroc:
            best_auroc = m["auroc"]
            torch.save({
                "state_dict": head.state_dict(),
                "config": {"patch_dim": head.patch_dim, "query_dim": head.query_dim,
                           "proj_dim": head.proj_dim},
                "metrics": m,
            }, best_path)
            print(f"    [best] saved to {best_path} (AUROC={best_auroc:.4f})")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"history": history, "best_auroc": best_auroc, "n_params": n_params}, f, indent=2)
    print(f"[diag] best AUROC={best_auroc:.4f}; ckpt={best_path}")


if __name__ == "__main__":
    main()
