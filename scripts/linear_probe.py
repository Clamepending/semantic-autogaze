"""Quick linear probe: can a simple linear layer on AutoGaze features + CLIP query
discriminate categories at all? This tests the information content of the features.

Usage:
    cd /Users/mark/code/semantic-autogaze
    source .venv/bin/activate
    python scripts/linear_probe.py --device mps
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from semantic_autogaze.train_coco_seg import (
    PATCH_GRID,
    CocoSegDataset,
    cache_autogaze_hidden_states,
    cache_clip_text_embeddings,
    download_coco_val,
    get_image_categories,
)
from pycocotools.coco import COCO


class LinearProbe(nn.Module):
    """Simplest possible head: project (patch || query) → 1."""
    def __init__(self, hidden_dim=192, embedding_dim=512):
        super().__init__()
        self.linear = nn.Linear(hidden_dim + embedding_dim, 1)

    def forward(self, patch_hidden, query_embed):
        B, N, _ = patch_hidden.shape
        q = query_embed.unsqueeze(1).expand(B, N, -1)
        h = torch.cat([patch_hidden, q], dim=-1)
        return self.linear(h).squeeze(-1)


class MLPProbe(nn.Module):
    """2-layer MLP: (patch || query) → hidden → 1. No spatial attention."""
    def __init__(self, hidden_dim=192, embedding_dim=512, mlp_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 1),
        )

    def forward(self, patch_hidden, query_embed):
        B, N, _ = patch_hidden.shape
        q = query_embed.unsqueeze(1).expand(B, N, -1)
        h = torch.cat([patch_hidden, q], dim=-1)
        return self.net(h).squeeze(-1)


def train_and_eval(head, train_loader, val_loader, device, num_epochs=10, lr=1e-3):
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(num_epochs):
        head.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            hidden = batch["hidden"].to(device)
            query = batch["query"].to(device)
            target = batch["target"].to(device)

            logits = head(hidden, query)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Validate
        head.eval()
        val_losses, fg_probs, bg_probs = [], [], []
        with torch.inference_mode():
            for batch in val_loader:
                hidden = batch["hidden"].to(device)
                query = batch["query"].to(device)
                target = batch["target"].to(device)

                logits = head(hidden, query)
                val_loss = F.binary_cross_entropy_with_logits(logits, target)
                val_losses.append(val_loss.item())

                probs = torch.sigmoid(logits)
                fg_mask = target > 0.5
                bg_mask = target < 0.5
                if fg_mask.any():
                    fg_probs.append(probs[fg_mask].mean().item())
                if bg_mask.any():
                    bg_probs.append(probs[bg_mask].mean().item())

        mean_val = sum(val_losses) / len(val_losses)
        mean_fg = sum(fg_probs) / max(len(fg_probs), 1)
        mean_bg = sum(bg_probs) / max(len(bg_probs), 1)
        print(f"  Epoch {epoch+1}: train_bce={sum(losses)/len(losses):.4f} "
              f"val_bce={mean_val:.4f} fg_prob={mean_fg:.4f} bg_prob={mean_bg:.4f} "
              f"gap={mean_fg-mean_bg:.4f}")

    return mean_val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/coco")
    p.add_argument("--device", default="mps")
    p.add_argument("--hidden_cache_dir", default="results/coco_seg_v7/hidden_cache")
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    img_dir, ann_file = download_coco_val(args.data_dir)
    coco = COCO(ann_file)
    cat_info = coco.loadCats(coco.getCatIds())
    categories = {c["id"]: c["name"] for c in cat_info}

    clip_path = "results/coco_seg_v7/clip_text_embeddings.pt"
    clip_embeddings = cache_clip_text_embeddings(categories, clip_path, device)

    all_ids = sorted(coco.getImgIds())
    random.shuffle(all_ids)
    split = int(len(all_ids) * 0.8)
    train_ids, val_ids = all_ids[:split], all_ids[split:]

    train_ds = CocoSegDataset(coco, train_ids, args.hidden_cache_dir, clip_embeddings, categories)
    val_ds = CocoSegDataset(coco, val_ids, args.hidden_cache_dir, clip_embeddings, categories)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\n{'='*60}")
    print(f"LINEAR PROBE (192+512 → 1)")
    print(f"{'='*60}")
    linear = LinearProbe().to(device)
    n_params = sum(p.numel() for p in linear.parameters())
    print(f"  Params: {n_params}")
    train_and_eval(linear, train_loader, val_loader, device, args.num_epochs)

    print(f"\n{'='*60}")
    print(f"MLP PROBE (192+512 → 256 → 256 → 1)")
    print(f"{'='*60}")
    mlp = MLPProbe().to(device)
    n_params = sum(p.numel() for p in mlp.parameters())
    print(f"  Params: {n_params}")
    train_and_eval(mlp, train_loader, val_loader, device, args.num_epochs)

    print(f"\nFor reference: BigHead (v7) val_bce=0.2639, FG prob=0.245, BG prob=0.098, gap=0.147")


if __name__ == "__main__":
    main()
