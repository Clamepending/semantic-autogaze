"""
Ablation studies for Semantic AutoGaze.

Evaluates which innovations are responsible for the gains by training
controlled variants with one component changed at a time.

Ablations:
  A1. BigHead distill (full model) — reference: val=0.0668
  A2. Small head distill — isolate: BigHead vs small head
  A3. BigHead direct (no distillation) — isolate: distillation effect
  A4. BigHead distill, no spatial conv — isolate: spatial refinement
  A5. BigHead distill, no self-attention (MLP only) — isolate: cross-patch attention
  A6. BigHead distill, pre-decoder features — isolate: feature extraction point

All use same data split, same seed, same training hyperparameters.
Runs for 50 epochs (enough to see convergence from prior experiments).
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

from semantic_autogaze.model import SimilarityHead, TeacherHead
from semantic_autogaze.train_bighead import BigSimilarityHead
from semantic_autogaze.train_distill import DistillDataset
from semantic_autogaze.train_clipseg import CLIPSegDataset


class BigHeadNoSpatial(nn.Module):
    """BigHead variant without spatial conv refinement."""

    def __init__(self, hidden_dim=192, embedding_dim=512, expanded_dim=384,
                 n_attn_heads=6, n_attn_layers=2, grid_size=14):
        super().__init__()
        self.grid_size = grid_size
        self.expanded_dim = expanded_dim

        self.feature_expand = nn.Sequential(
            nn.Linear(hidden_dim, expanded_dim),
            nn.GELU(),
            nn.LayerNorm(expanded_dim),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, grid_size * grid_size, expanded_dim) * 0.02)

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

        self.query_proj = nn.Sequential(
            nn.Linear(embedding_dim, expanded_dim),
            nn.GELU(),
            nn.LayerNorm(expanded_dim),
        )
        self.cross_attn = nn.MultiheadAttention(expanded_dim, n_attn_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(expanded_dim)

        self.score_mlp = nn.Sequential(
            nn.Linear(expanded_dim, expanded_dim // 2),
            nn.GELU(),
            nn.Linear(expanded_dim // 2, 1),
        )
        # NO spatial conv

    def forward(self, patch_hidden_states, query_embedding):
        B = patch_hidden_states.shape[0]
        N = patch_hidden_states.shape[1]
        G = self.grid_size
        T = N // (G * G)

        x = self.feature_expand(patch_hidden_states)
        x = x.reshape(B * T, G * G, self.expanded_dim)
        x = x + self.pos_embed

        for layer in self.self_attn_layers:
            residual = x
            x = layer["norm1"](x)
            x_attn, _ = layer["attn"](x, x, x)
            x = residual + x_attn
            residual = x
            x = layer["norm2"](x)
            x = residual + layer["ffn"](x)

        x = x.reshape(B, T * G * G, self.expanded_dim)

        query_proj = self.query_proj(query_embedding).unsqueeze(1)
        cross_out, _ = self.cross_attn(
            query_proj.expand(-1, T * G * G, -1),
            query_proj.expand(-1, 1, -1),
            query_proj.expand(-1, 1, -1),
        )
        x = self.cross_norm(x + cross_out)

        scores = self.score_mlp(x).squeeze(-1)
        return scores


class BigHeadMLPOnly(nn.Module):
    """BigHead variant: expanded MLP but NO self-attention (isolates attention contribution)."""

    def __init__(self, hidden_dim=192, embedding_dim=512, expanded_dim=384, grid_size=14):
        super().__init__()
        self.grid_size = grid_size
        self.expanded_dim = expanded_dim

        self.feature_expand = nn.Sequential(
            nn.Linear(hidden_dim, expanded_dim),
            nn.GELU(),
            nn.LayerNorm(expanded_dim),
        )

        self.query_proj = nn.Sequential(
            nn.Linear(embedding_dim, expanded_dim),
            nn.GELU(),
            nn.LayerNorm(expanded_dim),
        )

        # Deeper MLP to roughly match BigHead param count
        self.patch_mlp = nn.Sequential(
            nn.Linear(expanded_dim * 2, expanded_dim),
            nn.GELU(),
            nn.Linear(expanded_dim, expanded_dim // 2),
            nn.GELU(),
            nn.Linear(expanded_dim // 2, 1),
        )

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
        B = patch_hidden_states.shape[0]
        N = patch_hidden_states.shape[1]
        G = self.grid_size
        T = N // (G * G)

        x = self.feature_expand(patch_hidden_states)
        query_proj = self.query_proj(query_embedding)
        query_expanded = query_proj.unsqueeze(1).expand_as(x)
        combined = torch.cat([x, query_expanded], dim=-1)
        scores = self.patch_mlp(combined).squeeze(-1)

        grids = scores.reshape(B * T, 1, G, G)
        refined = grids + self.spatial(grids)
        scores = refined.reshape(B, T * G * G)
        return scores


def train_ablation(name, head, train_loader, val_loader, teacher, args, device,
                   use_distill=True):
    """Train one ablation variant and return best val BCE."""
    print(f"\n{'='*60}")
    print(f"ABLATION: {name}")
    print(f"{'='*60}")

    n_params = sum(p.numel() for p in head.parameters())
    print(f"  Params: {n_params/1e3:.1f}K")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    best_val = float("inf")
    alpha = args.distill_alpha

    for epoch in range(args.num_epochs):
        head.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"{name} E{epoch+1}/{args.num_epochs}", leave=False)

        for batch in pbar:
            hidden = batch["hidden_states"].to(device)
            query = batch["text_embedding"].to(device)
            target = batch["target_scores"].to(device)
            target_soft = torch.sigmoid(target)

            student_logits = head(hidden, query)
            loss_gt = F.binary_cross_entropy_with_logits(student_logits, target_soft)

            if use_distill and teacher is not None:
                clip_vis = batch["clip_visual"].to(device)
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
            else:
                loss = loss_gt

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            wandb.log({f"{name}/train_loss": loss.item()})

        scheduler.step()

        head.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                hidden = batch["hidden_states"].to(device)
                query = batch["text_embedding"].to(device)
                target = batch["target_scores"].to(device)
                target_soft = torch.sigmoid(target)
                pred = head(hidden, query)
                val_losses.append(F.binary_cross_entropy_with_logits(pred, target_soft).item())

        val_loss = np.mean(val_losses)
        if (epoch + 1) % 10 == 0:
            print(f"  E{epoch+1}: train={np.mean(epoch_losses):.4f}, val={val_loss:.4f}")
        wandb.log({f"{name}/val_bce": val_loss, f"{name}/epoch": epoch + 1})

        if val_loss < best_val:
            best_val = val_loss
            wandb.log({f"{name}/best_val": best_val})

    print(f"  Best val: {best_val:.4f}")
    return best_val


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if "/" in args.wandb_project:
        entity, project = args.wandb_project.split("/", 1)
    else:
        entity, project = None, args.wandb_project
    wandb.init(entity=entity, project=project, name="ablations", config=vars(args))

    device = torch.device(args.device)

    # Load data
    clipseg_files = sorted(glob.glob(os.path.join(args.clipseg_dir, "*_clipseg_clip*.pt")))
    print(f"Found {len(clipseg_files)} CLIPSeg cache files")
    random.shuffle(clipseg_files)
    split = int(0.9 * len(clipseg_files))

    # For distill ablations (need CLIP visual features)
    distill_train = DistillDataset(clipseg_files[:split], args.hidden_dir, args.clip_visual_dir)
    distill_val = DistillDataset(clipseg_files[split:], args.hidden_dir, args.clip_visual_dir)

    distill_train_loader = DataLoader(distill_train, batch_size=args.batch_size, shuffle=True,
                                       num_workers=4, pin_memory=True, drop_last=True)
    distill_val_loader = DataLoader(distill_val, batch_size=args.batch_size, shuffle=False,
                                     num_workers=4, pin_memory=True)

    # For non-distill ablation (no CLIP visual needed)
    nodistill_train = CLIPSegDataset(clipseg_files[:split], args.hidden_dir)
    nodistill_val = CLIPSegDataset(clipseg_files[split:], args.hidden_dir)

    nodistill_train_loader = DataLoader(nodistill_train, batch_size=args.batch_size, shuffle=True,
                                         num_workers=4, pin_memory=True, drop_last=True)
    nodistill_val_loader = DataLoader(nodistill_val, batch_size=args.batch_size, shuffle=False,
                                       num_workers=4, pin_memory=True)

    # Load pre-trained teacher
    teacher = TeacherHead(autogaze_dim=192, clip_visual_dim=768, text_dim=512,
                          grid_size=14, num_frames=16).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Loaded teacher from {args.teacher_ckpt}")

    results = {}

    # A1: BigHead + distill (reference — already trained, just report)
    results["A1_bighead_distill"] = 0.0668
    print(f"\nA1. BigHead + distill (reference): 0.0668")

    # A2: Small head + distill
    head_a2 = SimilarityHead(hidden_dim=192, embedding_dim=512, grid_size=14,
                              num_frames=16, use_spatial=True).to(device)
    results["A2_small_distill"] = train_ablation(
        "A2_small_distill", head_a2, distill_train_loader, distill_val_loader,
        teacher, args, device, use_distill=True)

    # A3: BigHead direct (no distillation)
    head_a3 = BigSimilarityHead(hidden_dim=192, embedding_dim=512, expanded_dim=384,
                                 n_attn_heads=6, n_attn_layers=2, grid_size=14).to(device)
    results["A3_bighead_nodistill"] = train_ablation(
        "A3_bighead_nodistill", head_a3, nodistill_train_loader, nodistill_val_loader,
        None, args, device, use_distill=False)

    # A4: BigHead distill, no spatial conv
    head_a4 = BigHeadNoSpatial(hidden_dim=192, embedding_dim=512, expanded_dim=384,
                                n_attn_heads=6, n_attn_layers=2, grid_size=14).to(device)
    results["A4_bighead_nospatial"] = train_ablation(
        "A4_nospatial", head_a4, distill_train_loader, distill_val_loader,
        teacher, args, device, use_distill=True)

    # A5: BigHead distill, MLP only (no self-attention)
    head_a5 = BigHeadMLPOnly(hidden_dim=192, embedding_dim=512, expanded_dim=384,
                              grid_size=14).to(device)
    results["A5_mlp_only"] = train_ablation(
        "A5_mlp_only", head_a5, distill_train_loader, distill_val_loader,
        teacher, args, device, use_distill=True)

    # Summary
    print("\n" + "="*60)
    print("ABLATION RESULTS")
    print("="*60)
    print(f"{'Ablation':<35} {'Val BCE':<12} {'vs Full (A1)':<15}")
    print("-" * 62)
    for key, val in sorted(results.items()):
        delta = val - results["A1_bighead_distill"]
        pct = delta / results["A1_bighead_distill"] * 100
        sign = "+" if delta >= 0 else ""
        print(f"  {key:<33} {val:<12.4f} {sign}{delta:.4f} ({sign}{pct:.1f}%)")

    wandb.log({"ablation_results": results})
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    parser.add_argument("--clip_visual_dir", default="results/distill/clip_visual_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--teacher_ckpt", default="results/distill/best_teacher.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--distill_alpha", type=float, default=0.5)
    parser.add_argument("--distill_temp", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb_project", default="semantic-autogaze")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
