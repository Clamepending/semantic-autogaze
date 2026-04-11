"""
Knowledge distillation training for Semantic AutoGaze.

Two-phase approach:
  Phase 1: Train a teacher head that uses BOTH AutoGaze (192-dim) + CLIP visual
            (768-dim) features to predict CLIPSeg heatmaps. The teacher learns a
            much better mapping because it has rich visual features.
  Phase 2: Train a student head (AutoGaze-only, 192-dim) using:
            - L_gt: BCE against CLIPSeg ground truth
            - L_distill: KL divergence against teacher's soft predictions
            The student inherits the teacher's knowledge without needing CLIP at inference.

Deploy the student head only — no CLIP visual encoder at inference.
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
from PIL import Image
import open_clip

from autogaze.models.autogaze import AutoGaze
from semantic_autogaze.model import SemanticAutoGaze, SimilarityHead, TeacherHead
from semantic_autogaze.data import read_video_frames
from semantic_autogaze.train_clipseg import (
    load_text_vocabulary,
    precompute_clip_text_embeddings,
    precompute_clipseg_targets,
    precompute_hidden_states,
)


def precompute_clip_visual_features(video_paths, cache_dir, num_frames=16,
                                     grid_size=14, device="cuda"):
    """Extract CLIP ViT-B/16 patch tokens (768-dim, 14x14 grid) for each video frame."""
    os.makedirs(cache_dir, exist_ok=True)

    print("Loading CLIP ViT-B/16 visual encoder...")
    model, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai", device=device
    )
    model.eval()

    # CLIP ViT-B/16 image preprocessing normalization
    from torchvision import transforms
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    valid = {}
    for vp in tqdm(video_paths, desc="Precomputing CLIP visual features"):
        key = hashlib.md5(vp.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{key}_clip_visual.pt")

        if os.path.exists(cache_path):
            valid[vp] = cache_path
            continue

        frames = read_video_frames(vp, num_frames, 224)
        if frames is None:
            continue

        try:
            T = frames.shape[0]
            # frames: (T, 224, 224, 3) uint8
            imgs = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (T, 3, 224, 224)
            imgs = normalize(imgs).to(device)

            all_patch_tokens = []
            with torch.no_grad():
                # Process frames in batches to avoid OOM
                for t in range(T):
                    img = imgs[t:t+1]  # (1, 3, 224, 224)
                    out = model.visual.forward_intermediates(img, intermediates_only=True)
                    # Last layer's patch tokens: (1, 768, 14, 14)
                    patch_feats = out["image_intermediates"][-1]
                    # Reshape to (196, 768)
                    patch_feats = patch_feats[0].permute(1, 2, 0).reshape(grid_size * grid_size, -1)
                    all_patch_tokens.append(patch_feats.cpu())

            # Stack: (T*196, 768)
            clip_visual = torch.cat(all_patch_tokens, dim=0)
            torch.save(clip_visual, cache_path)
            valid[vp] = cache_path

        except Exception as e:
            print(f"  Skipping {os.path.basename(vp)}: {e}")

    del model
    torch.cuda.empty_cache()
    print(f"  {len(valid)}/{len(video_paths)} videos cached")
    return valid


class DistillDataset(Dataset):
    """Dataset with AutoGaze hidden states, CLIP visual features, and CLIPSeg targets.

    Preloads everything into RAM for fast training.
    """

    def __init__(self, clipseg_files, hidden_dir, clip_visual_dir):
        self.samples = []
        hidden_cache = {}
        visual_cache = {}
        skipped = 0

        for cf in clipseg_files:
            data = torch.load(cf, map_location="cpu", weights_only=False)
            vp = data["video_path"]
            key = hashlib.md5(vp.encode()).hexdigest()

            hidden_path = os.path.join(hidden_dir, f"{key}_hidden.pt")
            visual_path = os.path.join(clip_visual_dir, f"{key}_clip_visual.pt")

            if not os.path.exists(hidden_path) or not os.path.exists(visual_path):
                skipped += 1
                continue

            if hidden_path not in hidden_cache:
                hidden_cache[hidden_path] = torch.load(hidden_path, map_location="cpu", weights_only=True)
            if visual_path not in visual_cache:
                visual_cache[visual_path] = torch.load(visual_path, map_location="cpu", weights_only=True)

            for q in data["queries"]:
                self.samples.append({
                    "hidden_states": hidden_cache[hidden_path],
                    "clip_visual": visual_cache[visual_path],
                    "text_embedding": q["text_embedding"],
                    "target_scores": q["target_scores"],
                })

        if skipped:
            print(f"  Skipped {skipped} files missing visual or hidden cache")
        print(f"  Preloaded {len(hidden_cache)} hidden + {len(visual_cache)} visual files into RAM")

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


def log_heatmaps(teacher, student, val_loader, device, epoch, tag="val"):
    """Log comparison heatmaps: student vs teacher vs GT."""
    teacher.eval()
    student.eval()
    grid_size = 14

    with torch.no_grad():
        for batch in val_loader:
            hidden = batch["hidden_states"].to(device)
            clip_vis = batch["clip_visual"].to(device)
            query = batch["text_embedding"].to(device)
            target = batch["target_scores"].to(device)
            target_soft = torch.sigmoid(target)

            teacher_logits = teacher(hidden, clip_vis, query)
            student_logits = student(hidden, query)

            teacher_sig = torch.sigmoid(teacher_logits)
            student_sig = torch.sigmoid(student_logits)

            n_samples = min(4, hidden.shape[0])
            fig, axes = plt.subplots(3, n_samples, figsize=(4 * n_samples, 12))
            if n_samples == 1:
                axes = axes[:, None]

            for i in range(n_samples):
                s_grid = student_sig[i][:grid_size**2].reshape(grid_size, grid_size).cpu().numpy()
                t_grid = teacher_sig[i][:grid_size**2].reshape(grid_size, grid_size).cpu().numpy()
                gt_grid = target_soft[i][:grid_size**2].reshape(grid_size, grid_size).cpu().numpy()

                axes[0, i].imshow(s_grid, cmap="jet", vmin=0, vmax=1)
                axes[0, i].set_title(f"Student [{s_grid.min():.2f}, {s_grid.max():.2f}]", fontsize=9)
                axes[0, i].axis("off")
                axes[1, i].imshow(t_grid, cmap="jet", vmin=0, vmax=1)
                axes[1, i].set_title(f"Teacher [{t_grid.min():.2f}, {t_grid.max():.2f}]", fontsize=9)
                axes[1, i].axis("off")
                axes[2, i].imshow(gt_grid, cmap="jet", vmin=0, vmax=1)
                axes[2, i].set_title(f"GT [{gt_grid.min():.2f}, {gt_grid.max():.2f}]", fontsize=9)
                axes[2, i].axis("off")

            axes[0, 0].set_ylabel("Student", fontsize=11)
            axes[1, 0].set_ylabel("Teacher", fontsize=11)
            axes[2, 0].set_ylabel("GT (CLIPSeg)", fontsize=11)
            plt.suptitle(f"Epoch {epoch}", fontsize=13)
            plt.tight_layout()
            wandb.log({f"{tag}/heatmaps": wandb.Image(fig)})
            plt.close(fig)
            break


def train_teacher(teacher, train_loader, val_loader, args, device):
    """Phase 1: Train teacher with AutoGaze + CLIP visual features."""
    print("\n" + "="*60)
    print("PHASE 1: Training teacher head")
    print("="*60)

    optimizer = torch.optim.AdamW(teacher.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.teacher_epochs)
    best_val = float("inf")

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

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            optimizer.step()

            lv = loss.item()
            epoch_losses.append(lv)
            pbar.set_postfix(loss=f"{lv:.4f}")
            wandb.log({"teacher/train_loss": lv, "teacher/lr": optimizer.param_groups[0]["lr"]})

        scheduler.step()
        train_loss = sum(epoch_losses) / len(epoch_losses)

        # Validate
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

        val_loss = sum(val_losses) / max(len(val_losses), 1)
        print(f"  Teacher epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")
        wandb.log({
            "teacher/epoch_train": train_loss,
            "teacher/epoch_val": val_loss,
            "teacher/epoch": epoch + 1,
        })

        if val_loss < best_val:
            best_val = val_loss
            torch.save(teacher.state_dict(), os.path.join(args.output_dir, "best_teacher.pt"))
            wandb.log({"teacher/best_val": best_val})

    print(f"  Teacher best val: {best_val:.4f}")
    # Reload best
    teacher.load_state_dict(torch.load(os.path.join(args.output_dir, "best_teacher.pt"), map_location=device))
    return teacher


def train_student(teacher, student, train_loader, val_loader, args, device):
    """Phase 2: Distill teacher knowledge into student (AutoGaze-only)."""
    print("\n" + "="*60)
    print("PHASE 2: Distilling to student head")
    print("="*60)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.student_epochs)
    best_val = float("inf")
    alpha = args.distill_alpha  # weight for GT loss vs distillation loss

    for epoch in range(args.student_epochs):
        student.train()
        epoch_losses = []
        epoch_gt_losses = []
        epoch_kd_losses = []
        pbar = tqdm(train_loader, desc=f"Student {epoch+1}/{args.student_epochs}")

        for batch in pbar:
            hidden = batch["hidden_states"].to(device)
            clip_vis = batch["clip_visual"].to(device)
            query = batch["text_embedding"].to(device)
            target = batch["target_scores"].to(device)
            target_soft = torch.sigmoid(target)

            student_logits = student(hidden, query)

            # GT loss: BCE against CLIPSeg ground truth
            loss_gt = F.binary_cross_entropy_with_logits(student_logits, target_soft)

            # Distillation loss: match teacher's soft predictions
            with torch.no_grad():
                teacher_logits = teacher(hidden, clip_vis, query)

            # KL divergence on sigmoid outputs (treat each patch as independent Bernoulli)
            T_temp = args.distill_temp
            student_soft = torch.sigmoid(student_logits / T_temp)
            teacher_soft = torch.sigmoid(teacher_logits / T_temp)
            # Binary KL: t*log(t/s) + (1-t)*log((1-t)/(1-s))
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
            epoch_gt_losses.append(loss_gt.item())
            epoch_kd_losses.append(loss_kd.item())
            pbar.set_postfix(loss=f"{lv:.4f}", gt=f"{loss_gt.item():.4f}", kd=f"{loss_kd.item():.4f}")
            wandb.log({
                "student/train_loss": lv,
                "student/gt_loss": loss_gt.item(),
                "student/kd_loss": loss_kd.item(),
                "student/lr": optimizer.param_groups[0]["lr"],
            })

        scheduler.step()
        train_loss = sum(epoch_losses) / len(epoch_losses)
        gt_loss = sum(epoch_gt_losses) / len(epoch_gt_losses)
        kd_loss = sum(epoch_kd_losses) / len(epoch_kd_losses)

        # Validate (BCE against GT only, for comparability)
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

        val_loss = sum(val_losses) / max(len(val_losses), 1)
        print(f"  Student epoch {epoch+1}: total={train_loss:.4f} gt={gt_loss:.4f} kd={kd_loss:.4f} val={val_loss:.4f}")
        wandb.log({
            "student/epoch_train": train_loss,
            "student/epoch_gt": gt_loss,
            "student/epoch_kd": kd_loss,
            "student/epoch_val": val_loss,
            "student/epoch": epoch + 1,
        })

        # Heatmap visualization every 5 epochs
        if (epoch + 1) % 5 == 0:
            log_heatmaps(teacher, student, val_loader, device, epoch + 1, tag="student")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(student.state_dict(), os.path.join(args.output_dir, "best_student.pt"))
            wandb.log({"student/best_val": best_val})

        torch.save(student.state_dict(), os.path.join(args.output_dir, "latest_student.pt"))

    print(f"  Student best val: {best_val:.4f}")
    return student


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb
    if "/" in args.wandb_project:
        entity, project = args.wandb_project.split("/", 1)
    else:
        entity, project = None, args.wandb_project
    wandb.init(entity=entity, project=project, name="distill-clip-teacher", config=vars(args))

    device = torch.device(args.device)

    # Find videos
    video_paths = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    if not video_paths:
        video_paths = sorted(glob.glob(os.path.join(args.video_dir, "**/*.mp4"), recursive=True))
    if not video_paths:
        print("ERROR: no videos found")
        sys.exit(1)
    print(f"Found {len(video_paths)} videos")

    # Load text vocabulary
    lvis_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "lvis_categories.txt")
    text_queries = load_text_vocabulary(lvis_path)

    # 1. Precompute CLIP text embeddings
    print("\n=== Precomputing CLIP text embeddings ===")
    text_embeddings, embed_dim = precompute_clip_text_embeddings(text_queries, device)

    # 2. Precompute AutoGaze hidden states (reuse existing cache)
    print("\n=== Precomputing AutoGaze hidden states ===")
    autogaze = AutoGaze.from_pretrained(args.autogaze_model, use_flash_attn=False).to(device).eval()
    hidden_dir = os.path.join(args.output_dir, "hidden_cache")
    hidden_map = precompute_hidden_states(
        video_paths, autogaze, hidden_dir, args.num_frames, device,
    )
    hidden_dim = autogaze.config.gaze_model_config.gaze_decoder_config.hidden_size
    del autogaze
    torch.cuda.empty_cache()
    print(f"  {len(hidden_map)}/{len(video_paths)} videos cached")

    # 3. Precompute CLIP visual features
    print("\n=== Precomputing CLIP visual features ===")
    clip_visual_dir = os.path.join(args.output_dir, "clip_visual_cache")
    clip_visual_map = precompute_clip_visual_features(
        video_paths, clip_visual_dir, args.num_frames, grid_size=14, device=device,
    )

    # 4. Precompute CLIPSeg targets (reuse existing cache)
    print("\n=== Precomputing CLIPSeg targets ===")
    clipseg_dir = os.path.join(args.output_dir, "clipseg_cache")
    valid_video_paths = [vp for vp in video_paths if vp in hidden_map and vp in clip_visual_map]
    clipseg_files = precompute_clipseg_targets(
        valid_video_paths, text_queries, text_embeddings,
        clipseg_dir, args.num_frames, grid_size=14,
        device=device, queries_per_video=args.queries_per_video,
        rounds=args.rounds,
    )
    print(f"  {len(clipseg_files)} CLIPSeg cache files")

    # 5. Create dataset
    print("\n=== Setting up datasets ===")
    random.shuffle(clipseg_files)
    split = int(0.9 * len(clipseg_files))
    train_files = clipseg_files[:split]
    val_files = clipseg_files[split:]

    train_dataset = DistillDataset(train_files, hidden_dir, clip_visual_dir)
    val_dataset = DistillDataset(val_files, hidden_dir, clip_visual_dir)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # 6. Create models
    clip_visual_dim = 768  # CLIP ViT-B/16

    teacher = TeacherHead(
        autogaze_dim=hidden_dim,
        clip_visual_dim=clip_visual_dim,
        text_dim=embed_dim,
        grid_size=14,
        num_frames=args.num_frames,
    ).to(device)
    print(f"Teacher head: {sum(p.numel() for p in teacher.parameters())/1e3:.1f}K params")

    student = SimilarityHead(
        hidden_dim=hidden_dim,
        embedding_dim=embed_dim,
        grid_size=14,
        num_frames=args.num_frames,
        use_spatial=True,
    ).to(device)
    print(f"Student head: {sum(p.numel() for p in student.parameters())/1e3:.1f}K params")

    # Optionally initialize student from previous best checkpoint
    prev_best = os.path.join(os.path.dirname(args.output_dir), "clipseg", "best_similarity_head.pt")
    if os.path.exists(prev_best) and args.init_student:
        student.load_state_dict(torch.load(prev_best, map_location=device))
        print(f"Initialized student from {prev_best}")

    # Phase 1: Train teacher (or load from checkpoint)
    teacher_ckpt = os.path.join(args.output_dir, "best_teacher.pt")
    if args.skip_teacher and os.path.exists(teacher_ckpt):
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
        print(f"Loaded teacher from {teacher_ckpt}, skipping teacher training")
    else:
        teacher = train_teacher(teacher, train_loader, val_loader, args, device)

    # Phase 2: Distill to student
    student = train_student(teacher, student, train_loader, val_loader, args, device)

    # Final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    teacher.eval()
    student.eval()
    teacher_vals, student_vals = [], []
    with torch.no_grad():
        for batch in val_loader:
            hidden = batch["hidden_states"].to(device)
            clip_vis = batch["clip_visual"].to(device)
            query = batch["text_embedding"].to(device)
            target = batch["target_scores"].to(device)
            target_soft = torch.sigmoid(target)

            t_pred = teacher(hidden, clip_vis, query)
            s_pred = student(hidden, query)
            teacher_vals.append(F.binary_cross_entropy_with_logits(t_pred, target_soft).item())
            student_vals.append(F.binary_cross_entropy_with_logits(s_pred, target_soft).item())

    t_val = sum(teacher_vals) / len(teacher_vals)
    s_val = sum(student_vals) / len(student_vals)
    print(f"  Teacher val BCE: {t_val:.4f}")
    print(f"  Student val BCE: {s_val:.4f}")
    print(f"  Gap: {s_val - t_val:.4f}")
    wandb.log({"final/teacher_val": t_val, "final/student_val": s_val, "final/gap": s_val - t_val})

    # Log final heatmap comparison
    log_heatmaps(teacher, student, val_loader, device, epoch=999, tag="final")

    wandb.finish()
    print("\nDone! Deploy student from:", os.path.join(args.output_dir, "best_student.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_dir", default="results/distill")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--teacher_epochs", type=int, default=50)
    parser.add_argument("--student_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--queries_per_video", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb_project", default="semantic-autogaze")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--distill_alpha", type=float, default=0.5,
                        help="Weight for GT loss (1-alpha for distillation loss)")
    parser.add_argument("--distill_temp", type=float, default=2.0,
                        help="Temperature for distillation softening")
    parser.add_argument("--init_student", action="store_true",
                        help="Initialize student from previous best checkpoint")
    parser.add_argument("--skip_teacher", action="store_true",
                        help="Skip teacher training, load from best_teacher.pt")
    args = parser.parse_args()
    train(args)
