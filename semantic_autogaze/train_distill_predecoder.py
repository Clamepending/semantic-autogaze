"""
Knowledge distillation with BigSimilarityHead using PRE-DECODER features.

Same as train_distill_bighead.py but uses CNN+connector output (before LLaMA
decoder) instead of post-decoder hidden states. The hypothesis: pre-decoder
features are "purer" visual features not yet distorted by the gaze objective,
which may be more naturally alignable with CLIP/text semantics.

The teacher still uses post-decoder features + CLIP visual (since it was
already trained that way), but we also train a variant with a new teacher
on pre-decoder features.
"""

import os
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


class PredecoderDistillDataset(Dataset):
    """Dataset with pre-decoder features, CLIP visual features, and CLIPSeg targets."""

    def __init__(self, clipseg_files, predecoder_dir, clip_visual_dir):
        self.samples = []
        predecoder_cache = {}
        visual_cache = {}
        skipped = 0

        for cf in clipseg_files:
            data = torch.load(cf, map_location="cpu", weights_only=False)
            vp = data["video_path"]
            key = hashlib.md5(vp.encode()).hexdigest()

            predecoder_path = os.path.join(predecoder_dir, f"{key}_predecoder.pt")
            visual_path = os.path.join(clip_visual_dir, f"{key}_clip_visual.pt")

            if not os.path.exists(predecoder_path) or not os.path.exists(visual_path):
                skipped += 1
                continue

            if predecoder_path not in predecoder_cache:
                predecoder_cache[predecoder_path] = torch.load(predecoder_path, map_location="cpu", weights_only=True)
            if visual_path not in visual_cache:
                visual_cache[visual_path] = torch.load(visual_path, map_location="cpu", weights_only=True)

            for q in data["queries"]:
                self.samples.append({
                    "hidden_states": predecoder_cache[predecoder_path],
                    "clip_visual": visual_cache[visual_path],
                    "text_embedding": q["text_embedding"],
                    "target_scores": q["target_scores"],
                })

        if skipped:
            print(f"  Skipped {skipped} files missing predecoder or visual cache")
        print(f"  Preloaded {len(predecoder_cache)} predecoder + {len(visual_cache)} visual files into RAM")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "hidden_states": s["hidden_states"],
            "clip_visual": s["clip_visual"],
            "text_embedding": s["text_embedding"],
            "target_scores": s["target_scores"],
        }


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if "/" in args.wandb_project:
        entity, project = args.wandb_project.split("/", 1)
    else:
        entity, project = None, args.wandb_project
    wandb.init(entity=entity, project=project,
               name=f"distill-predecoder-bighead-e{args.expanded_dim}",
               config=vars(args))

    device = torch.device(args.device)

    clipseg_files = sorted(glob.glob(os.path.join(args.clipseg_dir, "*_clipseg_clip*.pt")))
    print(f"Found {len(clipseg_files)} CLIPSeg cache files")

    random.shuffle(clipseg_files)
    split = int(0.9 * len(clipseg_files))
    train_dataset = PredecoderDistillDataset(clipseg_files[:split], args.predecoder_dir, args.clip_visual_dir)
    val_dataset = PredecoderDistillDataset(clipseg_files[split:], args.predecoder_dir, args.clip_visual_dir)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Train a NEW teacher on pre-decoder features + CLIP visual
    teacher = TeacherHead(
        autogaze_dim=192, clip_visual_dim=768, text_dim=512,
        grid_size=14, num_frames=16,
    ).to(device)
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher: {t_params/1e3:.1f}K params")

    # Phase 1: Train teacher on pre-decoder features
    if args.skip_teacher and os.path.exists(os.path.join(args.output_dir, "best_teacher_predecoder.pt")):
        teacher.load_state_dict(torch.load(os.path.join(args.output_dir, "best_teacher_predecoder.pt"), map_location=device))
        print("Loaded pre-trained teacher")
    else:
        print("\n=== Phase 1: Training teacher on pre-decoder features ===")
        teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=args.lr, weight_decay=0.01)
        teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(teacher_optimizer, T_max=args.teacher_epochs)
        best_teacher_val = float("inf")

        for epoch in range(args.teacher_epochs):
            teacher.train()
            epoch_losses = []
            pbar = tqdm(train_loader, desc=f"Teacher {epoch+1}/{args.teacher_epochs}")
            for batch in pbar:
                hidden = batch["hidden_states"].to(device)
                clip_vis = batch["clip_visual"].to(device)
                query = batch["text_embedding"].to(device)
                target = batch["target_scores"].to(device)
                target_soft = torch.sigmoid(target)

                pred = teacher(hidden, clip_vis, query)
                loss = F.binary_cross_entropy_with_logits(pred, target_soft)

                teacher_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
                teacher_optimizer.step()

                lv = loss.item()
                epoch_losses.append(lv)
                pbar.set_postfix(loss=f"{lv:.4f}")
                wandb.log({"teacher/train_loss": lv})

            teacher_scheduler.step()
            train_loss = np.mean(epoch_losses)

            teacher.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    hidden = batch["hidden_states"].to(device)
                    clip_vis = batch["clip_visual"].to(device)
                    query = batch["text_embedding"].to(device)
                    target = batch["target_scores"].to(device)
                    target_soft = torch.sigmoid(target)
                    pred = teacher(hidden, clip_vis, query)
                    val_losses.append(F.binary_cross_entropy_with_logits(pred, target_soft).item())

            val_loss = np.mean(val_losses)
            print(f"  Teacher epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")
            wandb.log({"teacher/epoch_train": train_loss, "teacher/epoch_val": val_loss, "teacher/epoch": epoch + 1})

            if val_loss < best_teacher_val:
                best_teacher_val = val_loss
                torch.save(teacher.state_dict(), os.path.join(args.output_dir, "best_teacher_predecoder.pt"))

        teacher.load_state_dict(torch.load(os.path.join(args.output_dir, "best_teacher_predecoder.pt"), map_location=device))
        print(f"  Teacher best val: {best_teacher_val:.4f}")

    # Phase 2: Distill to BigHead student
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = BigSimilarityHead(
        hidden_dim=192, embedding_dim=512,
        expanded_dim=args.expanded_dim,
        n_attn_heads=args.n_attn_heads,
        n_attn_layers=args.n_attn_layers,
        grid_size=14,
    ).to(device)
    s_params = sum(p.numel() for p in student.parameters())
    print(f"\n=== Phase 2: Distilling to BigHead student ===")
    print(f"BigHead student: {s_params/1e3:.1f}K params")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    best_val = float("inf")
    alpha = args.distill_alpha

    for epoch in range(args.num_epochs):
        student.train()
        epoch_losses, epoch_gt, epoch_kd = [], [], []
        pbar = tqdm(train_loader, desc=f"Student {epoch+1}/{args.num_epochs}")

        for batch in pbar:
            hidden = batch["hidden_states"].to(device)
            clip_vis = batch["clip_visual"].to(device)
            query = batch["text_embedding"].to(device)
            target = batch["target_scores"].to(device)
            target_soft = torch.sigmoid(target)

            student_logits = student(hidden, query)

            loss_gt = F.binary_cross_entropy_with_logits(student_logits, target_soft)

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
            wandb.log({"train/total_loss": lv, "train/gt_loss": loss_gt.item(), "train/kd_loss": loss_kd.item()})

        scheduler.step()

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
        print(f"  Student epoch {epoch+1}: total={np.mean(epoch_losses):.4f} gt={np.mean(epoch_gt):.4f} kd={np.mean(epoch_kd):.4f} val={val_loss:.4f}")
        wandb.log({"val/epoch_bce": val_loss, "epoch": epoch + 1})

        if val_loss < best_val:
            best_val = val_loss
            torch.save(student.state_dict(), os.path.join(args.output_dir, "best_predecoder_bighead.pt"))
            wandb.log({"val/best_bce": best_val})

        torch.save(student.state_dict(), os.path.join(args.output_dir, "latest_predecoder_bighead.pt"))

    print(f"\nBest val BCE: {best_val:.4f}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecoder_dir", default="results/distill/predecoder_cache")
    parser.add_argument("--clip_visual_dir", default="results/distill/clip_visual_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--output_dir", default="results/distill_predecoder")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--teacher_epochs", type=int, default=50)
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
    parser.add_argument("--skip_teacher", action="store_true")
    args = parser.parse_args()
    train(args)
