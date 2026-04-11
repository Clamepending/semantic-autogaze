"""
Feature-Aligned Semantic AutoGaze Training

Two-stage approach:
  Stage 1: Train a feature projector AutoGaze (192-dim) → CLIP visual (768-dim)
            using MSE regression on cached patch features. This learns to "translate"
            AutoGaze's gaze-oriented features into CLIP's semantic space.
  Stage 2: Train a similarity head that operates on the projected 768-dim features
            (now in CLIP's semantic space) to predict CLIPSeg heatmaps.
            Uses focal loss to focus on hard patches.

The intuition: rather than asking a small head to bridge the gap between 192-dim gaze
features and semantic heatmaps, first align the features to CLIP space where semantic
similarity is already meaningful, then do the easier task of predicting heatmaps from
semantically-meaningful features.
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
)


# ============================================================
# Feature Projector
# ============================================================

class FeatureProjector(nn.Module):
    """Projects AutoGaze 192-dim features to CLIP 768-dim space."""

    def __init__(self, input_dim=192, output_dim=768, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class AlignedSimilarityHead(nn.Module):
    """Similarity head that operates on projected (CLIP-aligned) features.

    Uses focal loss weighting internally and deeper spatial processing.
    """

    def __init__(self, feature_dim=768, text_dim=512, grid_size=14):
        super().__init__()
        self.grid_size = grid_size

        # Text query projection to feature space
        self.query_proj = nn.Linear(text_dim, feature_dim)

        # Cross-attention-style scoring: dot product + learned bias
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.score_bias = nn.Linear(feature_dim, 1)

        # Spatial refinement with more capacity
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, projected_features, query_embedding):
        """
        Args:
            projected_features: (B, T*196, 768) CLIP-aligned patch features
            query_embedding: (B, 512) CLIP text embedding
        Returns:
            scores: (B, T*196) logits
        """
        query_proj = self.query_proj(query_embedding)  # (B, 768)
        query_proj = F.normalize(query_proj, dim=-1)
        feat_norm = F.normalize(projected_features, dim=-1)

        # Cosine similarity + learned bias
        cos_sim = (feat_norm * query_proj.unsqueeze(1)).sum(dim=-1)  # (B, N)
        bias = self.score_bias(projected_features).squeeze(-1)  # (B, N)
        scores = self.score_scale * cos_sim + bias

        # Spatial refinement
        B = scores.shape[0]
        T = scores.shape[1] // (self.grid_size ** 2)
        G = self.grid_size
        grids = scores.reshape(B * T, 1, G, G)
        refined = grids + self.spatial(grids)
        scores = refined.reshape(B, T * G * G)

        return scores


# ============================================================
# Focal loss
# ============================================================

def focal_bce_loss(pred_logits, target_soft, gamma=2.0, alpha=0.75):
    """Focal BCE loss for per-patch prediction.

    Focuses training on hard patches (where the model is wrong/uncertain).
    alpha > 0.5 upweights positive patches (foreground).
    """
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_soft, reduction="none")
    pred_prob = torch.sigmoid(pred_logits)
    # p_t = prob of correct class
    p_t = pred_prob * target_soft + (1 - pred_prob) * (1 - target_soft)
    focal_weight = (1 - p_t) ** gamma
    # alpha weighting
    alpha_t = alpha * target_soft + (1 - alpha) * (1 - target_soft)
    loss = (alpha_t * focal_weight * bce).mean()
    return loss


# ============================================================
# Datasets
# ============================================================

class AlignmentDataset(Dataset):
    """Paired AutoGaze and CLIP visual features for projector training."""

    def __init__(self, hidden_dir, clip_visual_dir):
        self.pairs = []
        hidden_files = sorted(glob.glob(os.path.join(hidden_dir, "*_hidden.pt")))
        loaded = 0
        for hf in hidden_files:
            key = os.path.basename(hf).replace("_hidden.pt", "")
            vf = os.path.join(clip_visual_dir, f"{key}_clip_visual.pt")
            if not os.path.exists(vf):
                continue
            h = torch.load(hf, map_location="cpu", weights_only=True)
            v = torch.load(vf, map_location="cpu", weights_only=True)
            # h: (3136, 192), v: (3136, 768) — store all patches
            self.pairs.append((h, v))
            loaded += 1

        print(f"  Loaded {loaded} video pairs for alignment")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        h, v = self.pairs[idx]
        # Random subset of patches for efficiency
        n = h.shape[0]
        indices = torch.randperm(n)[:512]
        return h[indices], v[indices]


class AlignedSegDataset(Dataset):
    """Dataset using projected features for similarity head training."""

    def __init__(self, clipseg_files, hidden_dir, projector, device="cpu"):
        self.samples = []
        hidden_cache = {}

        projector.eval()
        for cf in clipseg_files:
            data = torch.load(cf, map_location="cpu", weights_only=False)
            vp = data["video_path"]
            key = hashlib.md5(vp.encode()).hexdigest()
            hidden_path = os.path.join(hidden_dir, f"{key}_hidden.pt")
            if not os.path.exists(hidden_path):
                continue

            if hidden_path not in hidden_cache:
                h = torch.load(hidden_path, map_location="cpu", weights_only=True)
                with torch.no_grad():
                    # Project to CLIP space
                    projected = projector(h.to(device)).cpu()
                hidden_cache[hidden_path] = projected

            for q in data["queries"]:
                self.samples.append({
                    "projected_features": hidden_cache[hidden_path],
                    "text_embedding": q["text_embedding"],
                    "target_scores": q["target_scores"],
                })

        print(f"  Projected {len(hidden_cache)} videos, {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "projected_features": s["projected_features"],
            "text_embedding": s["text_embedding"],
            "target_scores": s["target_scores"],
        }


# ============================================================
# Training
# ============================================================

def train_projector(hidden_dir, clip_visual_dir, output_dir, device, args):
    """Stage 1: Train AutoGaze → CLIP feature projector."""
    print("\n" + "=" * 60)
    print("STAGE 1: Training feature projector (AutoGaze → CLIP)")
    print("=" * 60)

    dataset = AlignmentDataset(hidden_dir, clip_visual_dir)
    # Split
    n = len(dataset)
    train_n = int(0.9 * n)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_n, n - train_n])
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    projector = FeatureProjector(input_dim=192, output_dim=768, hidden_dim=512).to(device)
    n_params = sum(p.numel() for p in projector.parameters())
    print(f"  Projector: {n_params/1e3:.1f}K params")

    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.proj_epochs)
    best_val = float("inf")

    for epoch in range(args.proj_epochs):
        projector.train()
        epoch_losses = []
        for h_batch, v_batch in tqdm(train_loader, desc=f"Proj {epoch+1}/{args.proj_epochs}"):
            h = h_batch.to(device)  # (B, 512, 192)
            v = v_batch.to(device)  # (B, 512, 768)
            B, N, _ = h.shape
            h_flat = h.reshape(B * N, -1)
            v_flat = v.reshape(B * N, -1)

            pred = projector(h_flat)
            # Cosine similarity loss + MSE
            loss_mse = F.mse_loss(pred, v_flat)
            loss_cos = 1 - F.cosine_similarity(pred, v_flat, dim=-1).mean()
            loss = loss_mse + 0.5 * loss_cos

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            wandb.log({"proj/train_loss": loss.item(), "proj/mse": loss_mse.item(),
                        "proj/cos": loss_cos.item()})

        scheduler.step()
        train_loss = np.mean(epoch_losses)

        # Validate
        projector.eval()
        val_losses = []
        val_cos = []
        with torch.no_grad():
            for h_batch, v_batch in val_loader:
                h = h_batch.to(device)
                v = v_batch.to(device)
                B, N, _ = h.shape
                pred = projector(h.reshape(B * N, -1))
                v_flat = v.reshape(B * N, -1)
                val_losses.append(F.mse_loss(pred, v_flat).item())
                val_cos.append(F.cosine_similarity(pred, v_flat, dim=-1).mean().item())

        val_loss = np.mean(val_losses)
        val_cosine = np.mean(val_cos)
        print(f"  Proj epoch {epoch+1}: train={train_loss:.4f}, val_mse={val_loss:.4f}, "
              f"val_cos={val_cosine:.4f}")
        wandb.log({
            "proj/epoch_train": train_loss, "proj/epoch_val": val_loss,
            "proj/val_cosine": val_cosine, "proj/epoch": epoch + 1,
        })

        if val_loss < best_val:
            best_val = val_loss
            torch.save(projector.state_dict(), os.path.join(output_dir, "best_projector.pt"))

    print(f"  Best projector val MSE: {best_val:.4f}, cosine: {val_cosine:.4f}")
    projector.load_state_dict(torch.load(os.path.join(output_dir, "best_projector.pt"), map_location=device))
    return projector


def train_aligned_head(projector, clipseg_files, hidden_dir, output_dir, device, args):
    """Stage 2: Train similarity head on projected features with focal loss."""
    print("\n" + "=" * 60)
    print("STAGE 2: Training aligned similarity head (focal loss)")
    print("=" * 60)

    projector.eval()
    for p in projector.parameters():
        p.requires_grad = False

    random.shuffle(clipseg_files)
    split = int(0.9 * len(clipseg_files))
    train_files = clipseg_files[:split]
    val_files = clipseg_files[split:]

    train_dataset = AlignedSegDataset(train_files, hidden_dir, projector, device)
    val_dataset = AlignedSegDataset(val_files, hidden_dir, projector, device)
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    head = AlignedSimilarityHead(feature_dim=768, text_dim=512, grid_size=14).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"  Aligned head: {n_params/1e3:.1f}K params")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.head_epochs)
    best_val = float("inf")

    for epoch in range(args.head_epochs):
        head.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Head {epoch+1}/{args.head_epochs}")
        for batch in pbar:
            feat = batch["projected_features"].to(device)
            query = batch["text_embedding"].to(device)
            target = batch["target_scores"].to(device)
            target_soft = torch.sigmoid(target)

            pred = head(feat, query)
            loss = focal_bce_loss(pred, target_soft, gamma=2.0, alpha=0.75)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            lv = loss.item()
            epoch_losses.append(lv)
            pbar.set_postfix(loss=f"{lv:.4f}")
            wandb.log({"head/train_loss": lv})

        scheduler.step()
        train_loss = np.mean(epoch_losses)

        # Validate (use both focal and standard BCE for comparability)
        head.eval()
        val_focal = []
        val_bce = []
        val_vis = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                feat = batch["projected_features"].to(device)
                query = batch["text_embedding"].to(device)
                target = batch["target_scores"].to(device)
                target_soft = torch.sigmoid(target)
                pred = head(feat, query)
                val_focal.append(focal_bce_loss(pred, target_soft).item())
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
        print(f"  Head epoch {epoch+1}: train_focal={train_loss:.4f}, val_focal={vf:.4f}, val_bce={vb:.4f}")
        wandb.log({
            "head/epoch_train": train_loss, "head/epoch_val_focal": vf,
            "head/epoch_val_bce": vb, "head/epoch": epoch + 1,
        })

        # Heatmap visualization
        if val_vis and (epoch + 1) % 5 == 0:
            grid_size = 14
            n_samples = len(val_vis)
            fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))
            if n_samples == 1:
                axes = axes[:, None]
            for i, vs in enumerate(val_vis):
                p_grid = vs["pred"][:grid_size**2].reshape(grid_size, grid_size).numpy()
                t_grid = vs["target"][:grid_size**2].reshape(grid_size, grid_size).numpy()
                axes[0, i].imshow(p_grid, cmap="jet", vmin=0, vmax=1)
                axes[0, i].set_title(f"Pred [{p_grid.min():.2f}, {p_grid.max():.2f}]", fontsize=9)
                axes[0, i].axis("off")
                axes[1, i].imshow(t_grid, cmap="jet", vmin=0, vmax=1)
                axes[1, i].set_title(f"GT [{t_grid.min():.2f}, {t_grid.max():.2f}]", fontsize=9)
                axes[1, i].axis("off")
            axes[0, 0].set_ylabel("Predicted", fontsize=11)
            axes[1, 0].set_ylabel("GT (CLIPSeg)", fontsize=11)
            plt.suptitle(f"Aligned Head — Epoch {epoch+1}", fontsize=13)
            plt.tight_layout()
            wandb.log({"head/heatmaps": wandb.Image(fig)})
            plt.close(fig)

        if vb < best_val:
            best_val = vb
            torch.save({
                "projector": projector.state_dict(),
                "head": head.state_dict(),
            }, os.path.join(output_dir, "best_aligned_model.pt"))
            wandb.log({"head/best_val_bce": best_val})

    print(f"  Best aligned head val BCE: {best_val:.4f}")
    return head


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if "/" in args.wandb_project:
        entity, project = args.wandb_project.split("/", 1)
    else:
        entity, project = None, args.wandb_project
    wandb.init(entity=entity, project=project, name="aligned-focal", config=vars(args))

    device = torch.device(args.device)

    # Reuse existing caches
    hidden_dir = args.hidden_dir
    clip_visual_dir = args.clip_visual_dir
    clipseg_dir = args.clipseg_dir

    print(f"Using cached hidden states: {hidden_dir}")
    print(f"Using cached CLIP visual: {clip_visual_dir}")
    print(f"Using cached CLIPSeg targets: {clipseg_dir}")

    clipseg_files = sorted(glob.glob(os.path.join(clipseg_dir, "*_clipseg_clip*.pt")))
    print(f"Found {len(clipseg_files)} CLIPSeg cache files")

    # Stage 1: Train projector
    projector = train_projector(hidden_dir, clip_visual_dir, args.output_dir, device, args)

    # Stage 2: Train aligned head with focal loss
    head = train_aligned_head(projector, clipseg_files, hidden_dir, args.output_dir, device, args)

    wandb.finish()
    print(f"\nDone! Model saved to {args.output_dir}/best_aligned_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    parser.add_argument("--clip_visual_dir", default="results/distill/clip_visual_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--output_dir", default="results/aligned")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--proj_epochs", type=int, default=30)
    parser.add_argument("--head_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb_project", default="semantic-autogaze")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
