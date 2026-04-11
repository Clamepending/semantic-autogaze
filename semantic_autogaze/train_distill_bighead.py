"""
Knowledge distillation with BigSimilarityHead as student.

Combines the two best ideas:
  1. Teacher distillation (the winning approach — teacher has CLIP visual features)
  2. BigHead with cross-patch attention (more capacity for the student)

The small SimilarityHead student (200K params) reached val=0.0797, nearly matching
v3 baseline (0.0792). This variant uses BigSimilarityHead (~1-3M params) as the
student, which should have enough capacity to fully capture the teacher's knowledge
while still being tiny compared to AutoGaze's 3.3M params.

Key: more params don't increase latency since we're operating on pre-computed
hidden states, not running additional vision encoders.
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

from semantic_autogaze.model import TeacherHead
from semantic_autogaze.train_bighead import BigSimilarityHead
from semantic_autogaze.train_distill import DistillDataset, log_heatmaps


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if "/" in args.wandb_project:
        entity, project = args.wandb_project.split("/", 1)
    else:
        entity, project = None, args.wandb_project
    wandb.init(entity=entity, project=project,
               name=f"distill-bighead-e{args.expanded_dim}-L{args.n_attn_layers}",
               config=vars(args))

    device = torch.device(args.device)

    # Load data from existing caches
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

    # Load pre-trained teacher
    teacher = TeacherHead(
        autogaze_dim=192, clip_visual_dim=768, text_dim=512,
        grid_size=14, num_frames=16,
    ).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher: {t_params/1e3:.1f}K params (frozen)")

    # Create BigHead student
    student = BigSimilarityHead(
        hidden_dim=192, embedding_dim=512,
        expanded_dim=args.expanded_dim,
        n_attn_heads=args.n_attn_heads,
        n_attn_layers=args.n_attn_layers,
        grid_size=14,
    ).to(device)
    s_params = sum(p.numel() for p in student.parameters())
    print(f"BigHead student: {s_params/1e3:.1f}K params "
          f"(expanded_dim={args.expanded_dim}, layers={args.n_attn_layers})")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    best_val = float("inf")
    alpha = args.distill_alpha

    for epoch in range(args.num_epochs):
        student.train()
        epoch_losses, epoch_gt, epoch_kd = [], [], []
        pbar = tqdm(train_loader, desc=f"BigStudent {epoch+1}/{args.num_epochs}")

        for batch in pbar:
            hidden = batch["hidden_states"].to(device)
            clip_vis = batch["clip_visual"].to(device)
            query = batch["text_embedding"].to(device)
            target = batch["target_scores"].to(device)
            target_soft = torch.sigmoid(target)

            student_logits = student(hidden, query)

            # GT loss
            loss_gt = F.binary_cross_entropy_with_logits(student_logits, target_soft)

            # Distillation loss
            with torch.no_grad():
                teacher_logits = teacher(hidden, clip_vis, query)

            T_temp = args.distill_temp
            student_soft = torch.sigmoid(student_logits / T_temp)
            teacher_soft = torch.sigmoid(teacher_logits / T_temp)
            eps = 1e-7
            teacher_soft = teacher_soft.clamp(eps, 1 - eps)
            student_soft = student_soft.clamp(eps, 1 - eps)
            loss_kd = (
                teacher_soft * torch.log(teacher_soft / student_soft)
                + (1 - teacher_soft) * torch.log((1 - teacher_soft) / (1 - student_soft))
            ).mean()

            loss = alpha * loss_gt + (1 - alpha) * loss_kd

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            lv = loss.item()
            epoch_losses.append(lv)
            epoch_gt.append(loss_gt.item())
            epoch_kd.append(loss_kd.item())
            pbar.set_postfix(loss=f"{lv:.4f}", gt=f"{loss_gt.item():.4f}", kd=f"{loss_kd.item():.4f}")
            wandb.log({
                "train/total_loss": lv,
                "train/gt_loss": loss_gt.item(),
                "train/kd_loss": loss_kd.item(),
            })

        scheduler.step()
        train_loss = np.mean(epoch_losses)
        gt_mean = np.mean(epoch_gt)
        kd_mean = np.mean(epoch_kd)

        # Validate (BCE against GT for comparability)
        student.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                hidden = batch["hidden_states"].to(device)
                query = batch["text_embedding"].to(device)
                target = batch["target_scores"].to(device)
                target_soft = torch.sigmoid(target)
                pred = student(hidden, query)
                val_losses.append(F.binary_cross_entropy_with_logits(pred, target_soft).item())

        val_loss = np.mean(val_losses)
        print(f"  Epoch {epoch+1}: total={train_loss:.4f} gt={gt_mean:.4f} kd={kd_mean:.4f} val={val_loss:.4f}")
        wandb.log({
            "train/epoch_total": train_loss,
            "train/epoch_gt": gt_mean,
            "train/epoch_kd": kd_mean,
            "val/epoch_bce": val_loss,
            "epoch": epoch + 1,
        })

        # Heatmap visualization every 5 epochs
        if (epoch + 1) % 5 == 0:
            # Wrap student to match log_heatmaps interface (expects student(hidden, query))
            log_heatmaps(teacher, student, val_loader, device, epoch + 1, tag="val")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(student.state_dict(), os.path.join(args.output_dir, "best_bighead_student.pt"))
            wandb.log({"val/best_bce": best_val})

        torch.save(student.state_dict(), os.path.join(args.output_dir, "latest_bighead_student.pt"))

    print(f"\nBest val BCE: {best_val:.4f}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    parser.add_argument("--clip_visual_dir", default="results/distill/clip_visual_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--teacher_ckpt", default="results/distill/best_teacher.pt")
    parser.add_argument("--output_dir", default="results/distill_bighead")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--expanded_dim", type=int, default=384)
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--n_attn_layers", type=int, default=2)
    parser.add_argument("--distill_alpha", type=float, default=0.5)
    parser.add_argument("--distill_temp", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb_project", default="semantic-autogaze")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
