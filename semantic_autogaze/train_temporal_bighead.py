"""
Train BigHead with temporal cross-attention.

Extends the BigHead architecture with cross-frame attention,
allowing the model to leverage inter-frame context for better
patch selection. Key idea: if frame t has a person, frame t+1
likely has a person too, even if the per-frame features are ambiguous.

Architecture additions:
  1. Temporal position embedding (per-frame)
  2. Cross-frame attention after per-frame self-attention
  3. Frame-level pooling for adaptive budget signals

Uses distillation from CLIP visual features teacher (same as train_distill.py).

Usage:
  CUDA_VISIBLE_DEVICES=2 python3 -m semantic_autogaze.train_temporal_bighead \
    --hidden_dir results/distill/hidden_cache \
    --clip_visual_dir results/distill/clip_visual_cache \
    --clipseg_dir results/distill/clipseg_cache \
    --teacher_ckpt results/distill/best_teacher.pt \
    --device cuda:0
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

from semantic_autogaze.train_bighead import BigSimilarityHead, focal_bce_loss
from semantic_autogaze.train_distill import DistillDataset
from semantic_autogaze.model import TeacherHead


class TemporalBigSimilarityHead(nn.Module):
    """BigHead with temporal cross-attention between frames."""

    def __init__(self, hidden_dim=192, embedding_dim=512, expanded_dim=384,
                 n_attn_heads=6, n_spatial_layers=2, n_temporal_layers=1,
                 grid_size=14, num_frames=16):
        super().__init__()
        self.grid_size = grid_size
        self.expanded_dim = expanded_dim
        self.num_frames = num_frames
        N = grid_size * grid_size

        # Feature expansion
        self.feature_expand = nn.Sequential(
            nn.Linear(hidden_dim, expanded_dim),
            nn.GELU(),
            nn.LayerNorm(expanded_dim),
        )

        # Spatial position embedding (14x14)
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, N, expanded_dim) * 0.02)

        # Temporal position embedding (per frame)
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, num_frames, expanded_dim) * 0.02)

        # Per-frame self-attention layers
        self.spatial_attn_layers = nn.ModuleList()
        for _ in range(n_spatial_layers):
            self.spatial_attn_layers.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(expanded_dim, n_attn_heads, batch_first=True),
                "norm1": nn.LayerNorm(expanded_dim),
                "ffn": nn.Sequential(
                    nn.Linear(expanded_dim, expanded_dim * 2),
                    nn.GELU(),
                    nn.Linear(expanded_dim * 2, expanded_dim),
                ),
                "norm2": nn.LayerNorm(expanded_dim),
            }))

        # Temporal cross-attention: frame tokens attend to other frames
        # We use frame-pooled representations as keys/values
        self.temporal_attn_layers = nn.ModuleList()
        for _ in range(n_temporal_layers):
            self.temporal_attn_layers.append(nn.ModuleDict({
                "pool": nn.Linear(expanded_dim, expanded_dim),  # frame pooling projection
                "attn": nn.MultiheadAttention(expanded_dim, n_attn_heads, batch_first=True),
                "norm1": nn.LayerNorm(expanded_dim),
                "ffn": nn.Sequential(
                    nn.Linear(expanded_dim, expanded_dim * 2),
                    nn.GELU(),
                    nn.Linear(expanded_dim * 2, expanded_dim),
                ),
                "norm2": nn.LayerNorm(expanded_dim),
                "broadcast": nn.Linear(expanded_dim, expanded_dim),  # project temporal info back to patches
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
        G = self.grid_size
        N = G * G
        T = patch_hidden_states.shape[1] // N

        # Expand features
        x = self.feature_expand(patch_hidden_states)  # (B, T*N, expanded_dim)

        # Per-frame self-attention with spatial position embedding
        x = x.reshape(B * T, N, self.expanded_dim)
        x = x + self.spatial_pos_embed

        for layer in self.spatial_attn_layers:
            residual = x
            x = layer["norm1"](x)
            x_attn, _ = layer["attn"](x, x, x)
            x = residual + x_attn
            residual = x
            x = layer["norm2"](x)
            x = residual + layer["ffn"](x)

        x = x.reshape(B, T, N, self.expanded_dim)

        # Temporal cross-attention
        for layer in self.temporal_attn_layers:
            # Pool each frame to a single vector
            frame_pooled = x.mean(dim=2)  # (B, T, expanded_dim)
            frame_pooled = frame_pooled + self.temporal_pos_embed[:, :T]
            frame_pooled = layer["pool"](frame_pooled)

            # Cross-frame attention: each frame attends to all frames
            residual = frame_pooled
            frame_pooled = layer["norm1"](frame_pooled)
            frame_attn, _ = layer["attn"](frame_pooled, frame_pooled, frame_pooled)
            frame_pooled = residual + frame_attn

            residual = frame_pooled
            frame_pooled = layer["norm2"](frame_pooled)
            frame_pooled = residual + layer["ffn"](frame_pooled)

            # Broadcast temporal info back to patches
            temporal_signal = layer["broadcast"](frame_pooled)  # (B, T, expanded_dim)
            x = x + temporal_signal.unsqueeze(2)  # (B, T, N, expanded_dim)

        x = x.reshape(B, T * N, self.expanded_dim)

        # Cross-attention with text query
        query_proj = self.query_proj(query_embedding).unsqueeze(1)
        cross_out, _ = self.cross_attn(
            query_proj.expand(-1, T * N, -1),
            query_proj.expand(-1, 1, -1),
            query_proj.expand(-1, 1, -1),
        )
        x = self.cross_norm(x + cross_out)

        # Score prediction
        scores = self.score_mlp(x).squeeze(-1)  # (B, T*N)

        # Spatial refinement per frame
        grids = scores.reshape(B * T, 1, G, G)
        refined = grids + self.spatial(grids)
        scores = refined.reshape(B, T * N)

        return scores


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if "/" in args.wandb_project:
        entity, project = args.wandb_project.split("/", 1)
    else:
        entity, project = None, args.wandb_project
    wandb.init(entity=entity, project=project, name="temporal-bighead-distill",
               config=vars(args))

    device = torch.device(args.device)

    # Load data
    clipseg_files = sorted(glob.glob(os.path.join(args.clipseg_dir, "*_clipseg_clip*.pt")))
    print(f"Found {len(clipseg_files)} CLIPSeg cache files")
    random.shuffle(clipseg_files)
    split = int(0.9 * len(clipseg_files))

    train_dataset = DistillDataset(clipseg_files[:split], args.hidden_dir, args.clip_visual_dir)
    val_dataset = DistillDataset(clipseg_files[split:], args.hidden_dir, args.clip_visual_dir)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Teacher
    teacher = TeacherHead(autogaze_dim=192, clip_visual_dim=768, text_dim=512, grid_size=14).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    teacher.eval()
    print("Loaded teacher")

    # Student: temporal bighead
    student = TemporalBigSimilarityHead(
        hidden_dim=192, embedding_dim=512, expanded_dim=args.expanded_dim,
        n_attn_heads=args.n_attn_heads, n_spatial_layers=args.n_spatial_layers,
        n_temporal_layers=args.n_temporal_layers, grid_size=14, num_frames=16,
    ).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"TemporalBigHead: {n_params/1e3:.1f}K params")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    best_val = float("inf")

    for epoch in range(args.num_epochs):
        student.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"E{epoch+1}/{args.num_epochs}")
        for batch in pbar:
            hidden = batch["hidden_states"].to(device)
            clip_vis = batch["clip_visual"].to(device)
            query = batch["text_embedding"].to(device)
            target = batch["target_scores"].to(device)
            target_soft = torch.sigmoid(target)

            # Teacher soft targets
            with torch.no_grad():
                teacher_logits = teacher(hidden, clip_vis, query)
                teacher_soft = torch.sigmoid(teacher_logits / args.distill_temp)

            student_logits = student(hidden, query)
            student_soft = torch.sigmoid(student_logits / args.distill_temp)

            # Distillation loss
            distill_loss = F.binary_cross_entropy(student_soft, teacher_soft)
            # Hard target loss
            hard_loss = F.binary_cross_entropy_with_logits(student_logits, target_soft)
            # Combined
            loss = args.distill_alpha * distill_loss + (1 - args.distill_alpha) * hard_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            lv = loss.item()
            epoch_losses.append(lv)
            pbar.set_postfix(loss=f"{lv:.4f}")
            wandb.log({"train/loss": lv, "train/distill_loss": distill_loss.item(),
                        "train/hard_loss": hard_loss.item()})

        scheduler.step()

        # Validation
        student.eval()
        val_bces = []
        with torch.no_grad():
            for batch in val_loader:
                hidden = batch["hidden_states"].to(device)
                query = batch["text_embedding"].to(device)
                target = batch["target_scores"].to(device)
                target_soft = torch.sigmoid(target)
                pred = student(hidden, query)
                val_bces.append(F.binary_cross_entropy_with_logits(pred, target_soft).item())

        vb = np.mean(val_bces)
        train_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch+1}: train={train_loss:.4f}, val_bce={vb:.4f}")
        wandb.log({"train/epoch_loss": train_loss, "val/epoch_bce": vb, "epoch": epoch + 1})

        if vb < best_val:
            best_val = vb
            torch.save(student.state_dict(),
                       os.path.join(args.output_dir, "best_temporal_bighead.pt"))
            wandb.log({"val/best_bce": best_val})
            print(f"    New best: {best_val:.4f}")

    print(f"\nBest val BCE: {best_val:.4f}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    parser.add_argument("--clip_visual_dir", default="results/distill/clip_visual_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--teacher_ckpt", default="results/distill/best_teacher.pt")
    parser.add_argument("--output_dir", default="results/temporal_bighead")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--expanded_dim", type=int, default=384)
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--n_spatial_layers", type=int, default=2)
    parser.add_argument("--n_temporal_layers", type=int, default=1)
    parser.add_argument("--distill_alpha", type=float, default=0.5)
    parser.add_argument("--distill_temp", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb_project", default="semantic-autogaze")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
