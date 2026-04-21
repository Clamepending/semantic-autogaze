"""r/distill-118k-tiny cycle 1: latency feasibility bench.

DINOv2-small (ViT-S/14) + minimal cross-attn head -> end-to-end ms/query
on M3 Pro CPU. Falsifier: > 31 ms/query (3x faster than IconStudent-118K
93.6 ms/query). No training; this is a scope check before committing
training compute on a non-AutoGaze base.

Stages (printed individually):
  preprocess  : PIL load + resize + normalize (224x224)
  vision      : DINOv2 forward -> patch tokens (B, 256, 384)
  head        : 1-layer cross-attn (text query 512 -> 384 -> attend over patches) -> per-patch logits
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class TinyCrossAttnHead(nn.Module):
    """Minimal: text query attends over DINOv2 patch tokens, returns per-patch logits.

    1 cross-attention head, no FFN, no layer norm beyond q/k/v projections.
    text (Q, 512) -> q_proj (Q, 384)
    patches (B, N, 384) -> k_proj, v_proj (B, N, 384)
    cosine_logits = q . k / sqrt(384) shape (B, Q, N)
    """

    def __init__(self, patch_dim=384, text_dim=512, head_dim=384):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(text_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(patch_dim, head_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, patches: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        # patches: (B, N, patch_dim) ; text: (Q, text_dim) -> (B, Q, N)
        q = F.normalize(self.q_proj(text), dim=-1)            # (Q, head_dim)
        k = F.normalize(self.k_proj(patches), dim=-1)         # (B, N, head_dim)
        return torch.einsum("bnd,qd->bqn", k, q) * self.scale + self.bias


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results/distill_118k_tiny/cycle1")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-queries", type=int, default=1, help="categories per query (text rows)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=None,
                   help="torch.set_num_threads override; default uses pytorch default")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    print(f"[diag] torch threads = {torch.get_num_threads()}", flush=True)

    device = torch.device(args.device)

    print("[diag] loading DINOv2-small from HF cache...", flush=True)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    backbone = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    n_back = sum(p.numel() for p in backbone.parameters())
    print(f"[diag] DINOv2-small params: {n_back:,}", flush=True)

    head = TinyCrossAttnHead(patch_dim=384, text_dim=512, head_dim=384).to(device).eval()
    n_head = sum(p.numel() for p in head.parameters())
    print(f"[diag] head params: {n_head:,} (total: {n_back + n_head:,})", flush=True)

    # Synthetic 224x224 image batch — covers preprocess + forward latency without I/O variance
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
    text = torch.randn(args.n_queries, 512, device=device)

    # Warmup
    print(f"[diag] warmup ({args.n_warmup} trials)...", flush=True)
    with torch.inference_mode():
        for _ in range(args.n_warmup):
            inp = processor(images=img, return_tensors="pt").to(device)
            out = backbone(**inp)
            patches = out.last_hidden_state[:, 1:, :]  # drop CLS
            _ = head(patches, text)

    # Bench
    print(f"[diag] benching {args.n_trials} trials...", flush=True)
    t_pre, t_vis, t_head, t_total = [], [], [], []
    with torch.inference_mode():
        for i in range(args.n_trials):
            t0 = time.perf_counter()
            inp = processor(images=img, return_tensors="pt").to(device)
            t1 = time.perf_counter()
            out = backbone(**inp)
            patches = out.last_hidden_state[:, 1:, :]
            t2 = time.perf_counter()
            _ = head(patches, text)
            t3 = time.perf_counter()
            t_pre.append((t1 - t0) * 1000)
            t_vis.append((t2 - t1) * 1000)
            t_head.append((t3 - t2) * 1000)
            t_total.append((t3 - t0) * 1000)

    def stats(xs):
        return {"mean": float(np.mean(xs)), "std": float(np.std(xs)),
                "min": float(np.min(xs)), "max": float(np.max(xs))}

    res = {
        "n_trials": args.n_trials, "n_warmup": args.n_warmup,
        "n_queries": args.n_queries,
        "torch_threads": torch.get_num_threads(),
        "n_backbone_params": n_back, "n_head_params": n_head,
        "preprocess_ms": stats(t_pre),
        "vision_ms": stats(t_vis),
        "head_ms": stats(t_head),
        "total_ms": stats(t_total),
        "iconstudent_118k_baseline_ms": 93.6,
        "speedup_target_ms": 93.6 / 3.0,
        "passes_3x_target": float(np.mean(t_total)) <= 93.6 / 3.0,
    }
    print(json.dumps(res, indent=2), flush=True)

    out_path = out_dir / "bench.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
