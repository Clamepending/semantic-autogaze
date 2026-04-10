"""
Training script for Semantic AutoGaze similarity head.

Freezes AutoGaze backbone and trains only the similarity head
using L2 loss against SigLIP patch-level cosine similarities.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm

from autogaze.models.autogaze import AutoGaze

from semantic_autogaze.model import SemanticAutoGaze
from semantic_autogaze.data import (
    SigLIPEmbedder,
    download_sample_videos,
    create_dataloader,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Semantic AutoGaze similarity head")
    parser.add_argument("--autogaze_model", type=str, default="nvidia/AutoGaze")
    parser.add_argument("--siglip_model", type=str, default="google/siglip2-base-patch16-224")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory with training videos")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--wandb_project", type=str, default="claude/semantic-autogaze")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train(args):
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb
    # Parse entity/project format
    if "/" in args.wandb_project:
        entity, project = args.wandb_project.split("/", 1)
    else:
        entity, project = None, args.wandb_project
    wandb.init(
        entity=entity,
        project=project,
        name="similarity-head-training",
        config=vars(args),
    )

    device = torch.device(args.device)

    # Load SigLIP embedder
    print("Loading SigLIP embedder...")
    siglip_embedder = SigLIPEmbedder(model_name=args.siglip_model, device=device)
    embedding_dim = siglip_embedder.embed_dim

    # Load AutoGaze
    print("Loading AutoGaze model...")
    autogaze = AutoGaze.from_pretrained(args.autogaze_model, use_flash_attn=False)
    autogaze.to(device)
    autogaze.eval()

    # Create Semantic AutoGaze
    print("Creating SemanticAutoGaze model...")
    model = SemanticAutoGaze(autogaze, embedding_dim=embedding_dim)
    model.to(device)

    # Prepare training data
    print("Preparing training data...")
    if args.video_dir and os.path.isdir(args.video_dir):
        import glob
        video_paths = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
        if not video_paths:
            video_paths = sorted(glob.glob(os.path.join(args.video_dir, "**/*.mp4"), recursive=True))
    else:
        # Generate clips from the example video
        clips_dir = os.path.join(args.output_dir, "training_clips")
        video_paths = download_sample_videos(clips_dir, num_videos=30)

    print(f"Found {len(video_paths)} training videos")
    if len(video_paths) == 0:
        print("ERROR: No training videos found!")
        sys.exit(1)

    dataloader = create_dataloader(
        video_paths, siglip_embedder,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
    )

    # Optimizer - only train similarity head
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Training loop
    print(f"Starting training for {args.num_epochs} epochs...")
    train_losses = []
    best_loss = float('inf')

    for epoch in range(args.num_epochs):
        model.similarity_head.train()
        epoch_losses = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            video = batch["video"].to(device)  # (B, T, C, H, W)
            query_embedding = batch["query_embedding"].to(device)  # (B, embed_dim)
            gt_similarities = batch["gt_similarities"].to(device)  # (B, T*N)

            # Forward pass
            outputs = model(video, query_embedding)
            pred_similarities = outputs["similarity_scores"]  # (B, T*N)

            # L2 loss
            loss = F.mse_loss(pred_similarities, gt_similarities)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.4f}")

            wandb.log({
                "train/loss": loss_val,
                "train/lr": optimizer.param_groups[0]["lr"],
            })

        scheduler.step()

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}: avg loss = {epoch_loss:.4f}")

        wandb.log({
            "train/epoch_loss": epoch_loss,
            "train/epoch": epoch + 1,
        })

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                model.similarity_head.state_dict(),
                os.path.join(args.output_dir, "best_similarity_head.pt"),
            )
            wandb.log({"train/best_loss": best_loss})

        # Save latest
        torch.save(
            model.similarity_head.state_dict(),
            os.path.join(args.output_dir, "latest_similarity_head.pt"),
        )

    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("MSE Loss", fontsize=14)
    plt.title("Semantic AutoGaze: Similarity Head Training", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "training_loss.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    wandb.log({"training_curve": wandb.Image(plot_path)})

    print(f"Training complete! Best loss: {best_loss:.4f}")
    print(f"Results saved to {args.output_dir}/")

    wandb.finish()
    return model


if __name__ == "__main__":
    args = parse_args()
    train(args)
