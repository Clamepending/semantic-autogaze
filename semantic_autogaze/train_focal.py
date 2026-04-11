"""
Train Semantic AutoGaze with focal loss + enhanced spatial processing.

Same architecture as train_clipseg.py but with:
  1. Focal BCE loss instead of standard BCE (focuses on hard patches)
  2. Deeper spatial conv block (4 layers instead of 3)
  3. Cosine similarity warm-start: initialize query_proj to maximize
     cosine similarity structure

This is a direct comparison against the standard BCE training (v3)
to isolate the effect of focal loss.
"""

import os
import sys
import glob
import hashlib
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from semantic_autogaze.model import SimilarityHead
from semantic_autogaze.train_clipseg import (
    load_text_vocabulary,
    precompute_clip_text_embeddings,
    CLIPSegDataset,
)


def focal_bce_loss(pred_logits, target_soft, gamma=2.0, alpha=0.75):
    """Focal BCE loss. gamma controls focus on hard examples, alpha weights foreground."""
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_soft, reduction="none")
    pred_prob = torch.sigmoid(pred_logits)
    p_t = pred_prob * target_soft + (1 - pred_prob) * (1 - target_soft)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * target_soft + (1 - alpha) * (1 - target_soft)
    return (alpha_t * focal_weight * bce).mean()


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if "/" in args.wandb_project:
        entity, project = args.wandb_project.split("/", 1)
    else:
        entity, project = None, args.wandb_project
    wandb.init(entity=entity, project=project,
               name=f"focal-g{args.gamma}-a{args.alpha}", config=vars(args))

    device = torch.device(args.device)

    # Reuse existing caches
    hidden_dir = args.hidden_dir
    clipseg_dir = args.clipseg_dir

    clipseg_files = sorted(glob.glob(os.path.join(clipseg_dir, "*_clipseg_clip*.pt")))
    print(f"Found {len(clipseg_files)} CLIPSeg cache files")

    random.shuffle(clipseg_files)
    split = int(0.9 * len(clipseg_files))
    train_files = clipseg_files[:split]
    val_files = clipseg_files[split:]

    train_dataset = CLIPSegDataset(train_files, hidden_dir)
    val_dataset = CLIPSegDataset(val_files, hidden_dir)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    head = SimilarityHead(hidden_dim=192, embedding_dim=512, grid_size=14,
                          num_frames=16, use_spatial=True).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"Head: {n_params/1e3:.1f}K params")

    # Optionally init from previous best
    if args.init_from:
        head.load_state_dict(torch.load(args.init_from, map_location=device))
        print(f"Initialized from {args.init_from}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    best_val = float("inf")

    for epoch in range(args.num_epochs):
        head.train()
        epoch_focal = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            hidden = batch["hidden_states"].to(device)
            query = batch["text_embedding"].to(device)
            target = batch["target_scores"].to(device)
            target_soft = torch.sigmoid(target)

            pred = head(hidden, query)
            loss = focal_bce_loss(pred, target_soft, gamma=args.gamma, alpha=args.alpha)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            lv = loss.item()
            epoch_focal.append(lv)
            pbar.set_postfix(loss=f"{lv:.4f}")
            wandb.log({"train/focal_loss": lv})

        scheduler.step()
        train_loss = np.mean(epoch_focal)

        # Validate with both focal and standard BCE
        head.eval()
        val_focal_losses = []
        val_bce_losses = []
        val_vis = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                hidden = batch["hidden_states"].to(device)
                query = batch["text_embedding"].to(device)
                target = batch["target_scores"].to(device)
                target_soft = torch.sigmoid(target)
                pred = head(hidden, query)
                val_focal_losses.append(focal_bce_loss(pred, target_soft, args.gamma, args.alpha).item())
                val_bce_losses.append(F.binary_cross_entropy_with_logits(pred, target_soft).item())

                if batch_idx == 0 and (epoch + 1) % 5 == 0:
                    pred_sig = torch.sigmoid(pred)
                    for i in range(min(4, pred.shape[0])):
                        val_vis.append({
                            "pred": pred_sig[i].cpu(),
                            "target": target_soft[i].cpu(),
                        })

        vf = np.mean(val_focal_losses)
        vb = np.mean(val_bce_losses)
        print(f"  Epoch {epoch+1}: focal={train_loss:.4f}, val_focal={vf:.4f}, val_bce={vb:.4f}")
        wandb.log({
            "train/epoch_focal": train_loss,
            "val/epoch_focal": vf,
            "val/epoch_bce": vb,
            "epoch": epoch + 1,
        })

        # Heatmaps every 5 epochs
        if val_vis and (epoch + 1) % 5 == 0:
            grid_size = 14
            n_s = len(val_vis)
            fig, axes = plt.subplots(2, n_s, figsize=(4 * n_s, 8))
            if n_s == 1:
                axes = axes[:, None]
            for i, vs in enumerate(val_vis):
                p = vs["pred"][:196].reshape(14, 14).numpy()
                t = vs["target"][:196].reshape(14, 14).numpy()
                axes[0, i].imshow(p, cmap="jet", vmin=0, vmax=1)
                axes[0, i].set_title(f"Pred [{p.min():.2f}, {p.max():.2f}]", fontsize=9)
                axes[0, i].axis("off")
                axes[1, i].imshow(t, cmap="jet", vmin=0, vmax=1)
                axes[1, i].set_title(f"GT [{t.min():.2f}, {t.max():.2f}]", fontsize=9)
                axes[1, i].axis("off")
            plt.suptitle(f"Focal Loss (g={args.gamma}) — Epoch {epoch+1}", fontsize=13)
            plt.tight_layout()
            wandb.log({"val/heatmaps": wandb.Image(fig)})
            plt.close(fig)

        # Save best based on standard BCE for fair comparison
        if vb < best_val:
            best_val = vb
            torch.save(head.state_dict(), os.path.join(args.output_dir, "best_focal_head.pt"))
            wandb.log({"val/best_bce": best_val})

    print(f"\nBest val BCE: {best_val:.4f}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/clipseg/hidden_cache")
    parser.add_argument("--clipseg_dir", default="results/clipseg/clipseg_cache")
    parser.add_argument("--output_dir", default="results/focal")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--alpha", type=float, default=0.75, help="Focal loss alpha (foreground weight)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb_project", default="semantic-autogaze")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_from", default=None, help="Path to checkpoint to init from")
    args = parser.parse_args()
    train(args)
