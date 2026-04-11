"""
Train TemporalBigHead with combined distillation + ranking loss.

Combines the two most promising extensions:
  1. Temporal cross-attention (inter-frame context)
  2. Pairwise ranking loss (correct relative patch ordering)

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 -m semantic_autogaze.train_temporal_ranking \
    --hidden_dir results/distill/hidden_cache \
    --clip_visual_dir results/distill/clip_visual_cache \
    --clipseg_dir results/distill/clipseg_cache \
    --teacher_ckpt results/distill/best_teacher.pt \
    --device cuda:0
"""

import os
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from semantic_autogaze.train_temporal_bighead import TemporalBigSimilarityHead
from semantic_autogaze.train_distill import DistillDataset
from semantic_autogaze.model import TeacherHead


def pairwise_ranking_loss(pred_logits, targets, margin=0.2, n_pairs=50):
    """Pairwise ranking loss: score(positive) > score(negative) + margin."""
    B, N = pred_logits.shape
    gt_binary = (targets > 0.5).float()
    losses = []

    for b in range(B):
        pos_idx = torch.where(gt_binary[b] > 0.5)[0]
        neg_idx = torch.where(gt_binary[b] <= 0.5)[0]

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        n = min(n_pairs, len(pos_idx), len(neg_idx))
        pos_sample = pos_idx[torch.randint(len(pos_idx), (n,))]
        neg_sample = neg_idx[torch.randint(len(neg_idx), (n,))]

        pos_scores = pred_logits[b, pos_sample]
        neg_scores = pred_logits[b, neg_sample]

        loss = F.relu(margin - (pos_scores - neg_scores)).mean()
        losses.append(loss)

    if losses:
        return torch.stack(losses).mean()
    return pred_logits.sum() * 0.0


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if "/" in args.wandb_project:
        entity, project = args.wandb_project.split("/", 1)
    else:
        entity, project = None, args.wandb_project
    wandb.init(entity=entity, project=project, name="temporal-ranking-bighead",
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
    print(f"Temporal+Ranking BigHead: {n_params/1e3:.1f}K params")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    best_val = float("inf")

    for epoch in range(args.num_epochs):
        student.train()
        epoch_losses = []
        epoch_rank_losses = []
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
            # Ranking loss
            rank_loss = pairwise_ranking_loss(student_logits, target_soft,
                                              margin=args.rank_margin,
                                              n_pairs=args.rank_pairs)

            # Combined loss
            loss = (args.distill_alpha * distill_loss +
                    args.hard_alpha * hard_loss +
                    args.rank_alpha * rank_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            lv = loss.item()
            rlv = rank_loss.item()
            epoch_losses.append(lv)
            epoch_rank_losses.append(rlv)
            pbar.set_postfix(loss=f"{lv:.4f}", rank=f"{rlv:.4f}")
            wandb.log({"train/loss": lv, "train/distill_loss": distill_loss.item(),
                        "train/hard_loss": hard_loss.item(), "train/rank_loss": rlv})

        scheduler.step()

        # Validation
        student.eval()
        val_bces = []
        val_ranks = []
        with torch.no_grad():
            for batch in val_loader:
                hidden = batch["hidden_states"].to(device)
                query = batch["text_embedding"].to(device)
                target = batch["target_scores"].to(device)
                target_soft = torch.sigmoid(target)
                pred = student(hidden, query)
                val_bces.append(F.binary_cross_entropy_with_logits(pred, target_soft).item())
                val_ranks.append(pairwise_ranking_loss(pred, target_soft,
                                                        margin=args.rank_margin).item())

        vb = np.mean(val_bces)
        vr = np.mean(val_ranks)
        train_loss = np.mean(epoch_losses)
        train_rank = np.mean(epoch_rank_losses)
        print(f"  Epoch {epoch+1}: train={train_loss:.4f} (rank={train_rank:.4f}), "
              f"val_bce={vb:.4f}, val_rank={vr:.4f}")
        wandb.log({"train/epoch_loss": train_loss, "val/epoch_bce": vb,
                    "val/epoch_rank": vr, "epoch": epoch + 1})

        if vb < best_val:
            best_val = vb
            torch.save(student.state_dict(),
                       os.path.join(args.output_dir, "best_temporal_ranking.pt"))
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
    parser.add_argument("--output_dir", default="results/temporal_ranking")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--expanded_dim", type=int, default=384)
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--n_spatial_layers", type=int, default=2)
    parser.add_argument("--n_temporal_layers", type=int, default=1)
    parser.add_argument("--distill_alpha", type=float, default=0.4)
    parser.add_argument("--hard_alpha", type=float, default=0.3)
    parser.add_argument("--rank_alpha", type=float, default=0.3)
    parser.add_argument("--rank_margin", type=float, default=0.2)
    parser.add_argument("--rank_pairs", type=int, default=50)
    parser.add_argument("--distill_temp", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb_project", default="semantic-autogaze")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
