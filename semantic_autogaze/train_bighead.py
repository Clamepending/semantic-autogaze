"""
Train a larger similarity head with cross-patch attention.

Key insight: the current head processes each patch independently (MLP) then
applies local spatial conv. But semantic segmentation benefits from global
context — "is this patch part of a person?" depends on what the neighboring
patches look like.

This variant adds:
  1. Feature expansion: 192 → 384 via learned projection
  2. Cross-patch self-attention within each frame (14x14 = 196 tokens)
  3. Query-conditioned cross-attention (text query attends to patches)
  4. Deeper spatial conv refinement
  5. Focal loss

The head is larger (~1-2M params) but still tiny compared to AutoGaze (3.3M)
and runs on pre-computed features, so latency increase is minimal.
"""

import os
import glob
import hashlib
import random
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from semantic_autogaze.train_clipseg import CLIPSegDataset


def focal_bce_loss(pred_logits, target_soft, gamma=2.0, alpha=0.75):
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_soft, reduction="none")
    pred_prob = torch.sigmoid(pred_logits)
    p_t = pred_prob * target_soft + (1 - pred_prob) * (1 - target_soft)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * target_soft + (1 - alpha) * (1 - target_soft)
    return (alpha_t * focal_weight * bce).mean()


class BigSimilarityHead(nn.Module):
    """Larger head with feature expansion, self-attention, and cross-attention."""

    def __init__(self, hidden_dim=192, embedding_dim=512, expanded_dim=384,
                 n_attn_heads=6, n_attn_layers=2, grid_size=14):
        super().__init__()
        self.grid_size = grid_size
        self.expanded_dim = expanded_dim

        # Feature expansion
        self.feature_expand = nn.Sequential(
            nn.Linear(hidden_dim, expanded_dim),
            nn.GELU(),
            nn.LayerNorm(expanded_dim),
        )

        # Learnable spatial position embedding for 14x14 grid
        self.pos_embed = nn.Parameter(torch.randn(1, grid_size * grid_size, expanded_dim) * 0.02)

        # Self-attention layers (patches attend to each other within a frame)
        self.self_attn_layers = nn.ModuleList()
        for _ in range(n_attn_layers):
            self.self_attn_layers.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(expanded_dim, n_attn_heads, batch_first=True),
                "norm1": nn.LayerNorm(expanded_dim),
                "ffn": nn.Sequential(
                    nn.Linear(expanded_dim, expanded_dim * 2),
                    nn.GELU(),
                    nn.Linear(expanded_dim * 2, expanded_dim),
                ),
                "norm2": nn.LayerNorm(expanded_dim),
            }))

        # Query projection + cross-attention
        self.query_proj = nn.Sequential(
            nn.Linear(embedding_dim, expanded_dim),
            nn.GELU(),
            nn.LayerNorm(expanded_dim),
        )
        self.cross_attn = nn.MultiheadAttention(expanded_dim, n_attn_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(expanded_dim)

        # Score prediction
        self.score_mlp = nn.Sequential(
            nn.Linear(expanded_dim, expanded_dim // 2),
            nn.GELU(),
            nn.Linear(expanded_dim // 2, 1),
        )

        # Spatial refinement
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, patch_hidden_states, query_embedding):
        """
        Args:
            patch_hidden_states: (B, T*196, 192)
            query_embedding: (B, 512)
        Returns:
            scores: (B, T*196)
        """
        B = patch_hidden_states.shape[0]
        N = patch_hidden_states.shape[1]
        G = self.grid_size
        T = N // (G * G)

        # Expand features
        x = self.feature_expand(patch_hidden_states)  # (B, T*196, 384)

        # Process per-frame with self-attention
        x = x.reshape(B * T, G * G, self.expanded_dim)
        x = x + self.pos_embed  # add spatial position info

        for layer in self.self_attn_layers:
            # Self-attention
            residual = x
            x = layer["norm1"](x)
            x_attn, _ = layer["attn"](x, x, x)
            x = residual + x_attn
            # FFN
            residual = x
            x = layer["norm2"](x)
            x = residual + layer["ffn"](x)

        x = x.reshape(B, T * G * G, self.expanded_dim)

        # Cross-attention with text query
        query_proj = self.query_proj(query_embedding).unsqueeze(1)  # (B, 1, 384)
        # Query attends to patches → modulate patch features
        cross_out, attn_weights = self.cross_attn(
            query_proj.expand(-1, T * G * G, -1),  # queries: each patch position
            query_proj.expand(-1, 1, -1),           # keys: single text query
            query_proj.expand(-1, 1, -1),           # values: text query
        )
        x = self.cross_norm(x + cross_out)

        # Score prediction
        scores = self.score_mlp(x).squeeze(-1)  # (B, T*196)

        # Spatial refinement per frame
        grids = scores.reshape(B * T, 1, G, G)
        refined = grids + self.spatial(grids)
        scores = refined.reshape(B, T * G * G)

        return scores


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if "/" in args.wandb_project:
        entity, project = args.wandb_project.split("/", 1)
    else:
        entity, project = None, args.wandb_project
    wandb.init(entity=entity, project=project, name="bighead-attn", config=vars(args))

    device = torch.device(args.device)

    clipseg_files = sorted(glob.glob(os.path.join(args.clipseg_dir, "*_clipseg_clip*.pt")))
    print(f"Found {len(clipseg_files)} CLIPSeg cache files")

    random.shuffle(clipseg_files)
    split = int(0.9 * len(clipseg_files))
    train_dataset = CLIPSegDataset(clipseg_files[:split], args.hidden_dir)
    val_dataset = CLIPSegDataset(clipseg_files[split:], args.hidden_dir)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    head = BigSimilarityHead(
        hidden_dim=192, embedding_dim=512, expanded_dim=args.expanded_dim,
        n_attn_heads=args.n_attn_heads, n_attn_layers=args.n_attn_layers,
        grid_size=14,
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"BigHead: {n_params/1e3:.1f}K params (expanded_dim={args.expanded_dim}, "
          f"attn_layers={args.n_attn_layers}, attn_heads={args.n_attn_heads})")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    best_val = float("inf")

    for epoch in range(args.num_epochs):
        head.train()
        epoch_losses = []
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
            epoch_losses.append(lv)
            pbar.set_postfix(loss=f"{lv:.4f}")
            wandb.log({"train/focal_loss": lv})

        scheduler.step()
        train_loss = np.mean(epoch_losses)

        head.eval()
        val_focal = []
        val_bce = []
        val_vis = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                hidden = batch["hidden_states"].to(device)
                query = batch["text_embedding"].to(device)
                target = batch["target_scores"].to(device)
                target_soft = torch.sigmoid(target)
                pred = head(hidden, query)
                val_focal.append(focal_bce_loss(pred, target_soft, args.gamma, args.alpha).item())
                val_bce.append(F.binary_cross_entropy_with_logits(pred, target_soft).item())

                if batch_idx == 0 and (epoch + 1) % 5 == 0:
                    pred_sig = torch.sigmoid(pred)
                    for i in range(min(4, pred.shape[0])):
                        val_vis.append({
                            "pred": pred_sig[i].cpu(),
                            "target": target_soft[i].cpu(),
                        })

        vf = np.mean(val_focal)
        vb = np.mean(val_bce)
        print(f"  Epoch {epoch+1}: train_focal={train_loss:.4f}, val_focal={vf:.4f}, val_bce={vb:.4f}")
        wandb.log({
            "train/epoch_focal": train_loss,
            "val/epoch_focal": vf,
            "val/epoch_bce": vb,
            "epoch": epoch + 1,
        })

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
            plt.suptitle(f"BigHead Attn — Epoch {epoch+1}", fontsize=13)
            plt.tight_layout()
            wandb.log({"val/heatmaps": wandb.Image(fig)})
            plt.close(fig)

        if vb < best_val:
            best_val = vb
            torch.save(head.state_dict(), os.path.join(args.output_dir, "best_bighead.pt"))
            wandb.log({"val/best_bce": best_val})

    print(f"\nBest val BCE: {best_val:.4f}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/clipseg/hidden_cache")
    parser.add_argument("--clipseg_dir", default="results/clipseg/clipseg_cache")
    parser.add_argument("--output_dir", default="results/bighead")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--expanded_dim", type=int, default=384)
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--n_attn_layers", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb_project", default="semantic-autogaze")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
