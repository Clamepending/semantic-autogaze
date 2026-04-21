"""Cycle 2 (light-finetune): unfreeze the last gaze_decoder LLaMA layer.

Trains a copy of `gm.gaze_decoder.model.layers[3]` (initialized from
pretrained AutoGaze weights) end-to-end with the bilinear_cosine head, on
top of the cached layer-2 features (= input to layer 3) on the same
val2017 80/20 split used in cycle 1. Compares to L3-frozen baseline (cycle 1
AUROC=0.7620).

The trainable surface is ~520K LLaMA-layer params + 90K head params = ~610K
total trainable. Frozen baseline used 90K. If the frozen baseline is already
extracting what the bilinear head can use, this should not improve much —
falsifying the "feature side is the bottleneck" diagnosis. If it does
improve, the diagnosis was right and cycle 3 (deeper unfreeze) is justified.
"""
from __future__ import annotations

import argparse
import copy
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

from semantic_autogaze.inference import load_autogaze


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


class L3UnfreezeModel(nn.Module):
    """Trainable last LlamaDecoderLayer + frozen final RMSNorm + trainable bilinear head.

    Inputs are L2-cached features (output of frozen layers 0..2). Mimics the
    forward path inside `LlamaModel` for the last layer + final norm so that
    a fresh frozen run with K=0 unfrozen layers should reproduce L3-frozen.
    """

    def __init__(self, autogaze, head: BilinearCosineHead, n_patches: int = 196):
        super().__init__()
        gm = autogaze.gazing_model
        # deepcopy so backward through the trainable layer doesn't mutate the original
        self.layer = copy.deepcopy(gm.gaze_decoder.model.layers[-1])
        # AutoGaze is loaded fully frozen; re-enable grad on the trainable layer copy
        for p in self.layer.parameters():
            p.requires_grad = True
        # frozen final RMSNorm (matches the frozen-baseline output convention:
        # outputs.last_hidden_state is post-final-norm)
        self.norm = copy.deepcopy(gm.gaze_decoder.model.norm)
        for p in self.norm.parameters():
            p.requires_grad = False
        # frozen rotary embeddings (compute cos/sin from position_ids)
        self.rotary_emb = copy.deepcopy(gm.gaze_decoder.model.rotary_emb)
        for p in self.rotary_emb.parameters():
            p.requires_grad = False

        # force eager attention (SDPA path is fussy about None masks)
        self.layer.self_attn.config._attn_implementation = "eager"

        self.head = head
        self.n_patches = n_patches
        # build a static causal mask for n_patches; LLaMA expects (B, 1, T, T)
        # additive float mask (-inf above diagonal)
        cm = torch.full((n_patches, n_patches), float("-inf"))
        cm = torch.triu(cm, diagonal=1)
        self.register_buffer("causal_mask", cm.unsqueeze(0).unsqueeze(0))  # (1,1,T,T)

        pos = torch.arange(n_patches).unsqueeze(0)  # (1, T)
        self.register_buffer("position_ids", pos)

    def forward(self, l2_patches: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """l2_patches: (B, T, D)  query: (B, query_dim)."""
        B, T, D = l2_patches.shape
        position_ids = self.position_ids.expand(B, -1)  # (B, T)
        cos, sin = self.rotary_emb(l2_patches, position_ids)
        out = self.layer(
            l2_patches,
            attention_mask=self.causal_mask.expand(B, -1, -1, -1),
            position_ids=position_ids,
            position_embeddings=(cos, sin),
        )
        # transformers >=4.45: layer returns a single Tensor (not tuple)
        if isinstance(out, tuple):
            out = out[0]
        out = self.norm(out)
        return self.head(out, query)


class L2Dataset(Dataset):
    """Returns L2 patch features per (img, cat) pair."""

    def __init__(self, image_ids, sidecar, layer_cache_dir: Path, teacher_dir: Path,
                 clip_text: dict, preload_cache: dict | None = None):
        self.layer_cache_dir = layer_cache_dir
        self.teacher_dir = teacher_dir
        self.clip_text = clip_text
        self.preload = preload_cache
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
            patches = patches_all[2].float()  # L2 = output of layer 2 = input to layer 3
        else:
            patches_all = torch.load(self.layer_cache_dir / f"{img_id}.pt", weights_only=True)
            teacher = torch.load(self.teacher_dir / f"{img_id}.pt", weights_only=True)
            patches = patches_all[2].float()
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


def evaluate(model, loader, device):
    model.eval()
    aurocs, recall25, iou30, fg, bg, bces = [], [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            patches = batch["patches"].to(device)
            query = batch["query"].to(device)
            target = batch["target"].to(device)
            logits = model(patches, query)
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
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _load(img_id):
        fp = layer_cache_dir / f"{img_id}.pt"
        tp = teacher_dir / f"{img_id}.pt"
        if not fp.exists() or not tp.exists():
            return img_id, None
        # we only need L2, so slice now to save memory
        patches = torch.load(fp, weights_only=True).half()  # (4, T, D)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer-cache-dir", default="results/autogaze_probe/features_gaze_layers_val")
    p.add_argument("--teacher-dir", default="results/autogaze_probe/teacher_14x14_val")
    p.add_argument("--clip-text", default="results/icon_student_B_native_train/clip_text_embeddings.pt")
    p.add_argument("--out-dir", default="results/autogaze_light_finetune/cycle2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-layer", type=float, default=1e-4,
                   help="Conservative LR for unfrozen LLaMA layer (10x smaller than head)")
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

    print("[diag] loading AutoGaze (to extract layer 3 weights + RMSNorm + rotary)...")
    ag = load_autogaze(device=device)

    print("[diag] preloading L2 cache + teacher into RAM...")
    preload_cache = preload_all(image_ids, layer_cache_dir, teacher_dir)
    print(f"[diag] preloaded {len(preload_cache)} images")

    train_ds = L2Dataset(train_ids, sidecar, layer_cache_dir, teacher_dir, clip_text, preload_cache)
    val_ds = L2Dataset(val_ids, sidecar, layer_cache_dir, teacher_dir, clip_text, preload_cache)
    print(f"[diag] {len(train_ds)} train pairs / {len(val_ds)} val pairs")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    torch.manual_seed(args.seed)
    head = BilinearCosineHead().to(device)
    model = L3UnfreezeModel(ag, head).to(device)

    n_layer_params = sum(p.numel() for p in model.layer.parameters() if p.requires_grad)
    n_head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    n_total = n_layer_params + n_head_params
    print(f"[diag] trainable: layer={n_layer_params}, head={n_head_params}, total={n_total}")

    opt = torch.optim.AdamW([
        {"params": model.layer.parameters(), "lr": args.lr_layer},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=0.01)

    history = []
    best_auroc = -1.0
    best_path = out_dir / "best.pt"
    for ep in range(args.n_epochs):
        model.train()
        # keep frozen modules in eval mode
        model.norm.eval()
        model.rotary_emb.eval()
        t0 = time.time()
        losses = []
        for batch in train_loader:
            patches = batch["patches"].to(device)
            query = batch["query"].to(device)
            target = batch["target"].to(device)
            logits = model(patches, query)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        m = evaluate(model, val_loader, device)
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
                "layer_state_dict": model.layer.state_dict(),
                "head_state_dict": model.head.state_dict(),
                "head_config": {"patch_dim": head.patch_dim, "query_dim": head.query_dim,
                                "proj_dim": head.proj_dim},
                "metrics": m,
            }, best_path)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"history": history, "best_auroc": best_auroc,
                   "n_trainable_params": n_total,
                   "n_layer_params": n_layer_params,
                   "n_head_params": n_head_params}, f, indent=2)
    print(f"\n[diag] best AUROC={best_auroc:.4f}; ckpt={best_path}")
    print(f"[diag] vs cycle-1 frozen L3 baseline (AUROC=0.7620): "
          f"Δ={best_auroc - 0.7620:+.4f}")


if __name__ == "__main__":
    main()
