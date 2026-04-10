"""
Fast training for Semantic AutoGaze similarity head.

Precomputes both SigLIP patch embeddings AND AutoGaze hidden states to disk,
then trains the lightweight MLP head purely on cached tensors — no model
forward passes during training.
"""

import os
import sys
import glob
import hashlib
import random
import argparse
import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from autogaze.models.autogaze import AutoGaze
from semantic_autogaze.model import SemanticAutoGaze, SimilarityHead
from semantic_autogaze.data import SigLIPEmbedder, read_video_frames


class PrecomputedDataset(Dataset):
    """Dataset of precomputed (hidden_states, siglip_embeddings) pairs."""

    def __init__(self, cache_files):
        self.cache_files = cache_files

    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, idx):
        data = torch.load(self.cache_files[idx], map_location="cpu", weights_only=True)
        hidden_states = data["hidden_states"]  # (T*N, hidden_dim)
        patch_embeddings = data["patch_embeddings"]  # (T*N, embed_dim)

        T_N, D = patch_embeddings.shape

        # Sample random patch as query
        query_idx = random.randint(0, T_N - 1)
        query_embedding = patch_embeddings[query_idx]  # (embed_dim,)

        # Ground truth: cosine similarity to all patches
        gt_similarities = torch.mv(patch_embeddings, query_embedding)  # (T*N,)

        return {
            "hidden_states": hidden_states,
            "query_embedding": query_embedding,
            "gt_similarities": gt_similarities,
        }


def precompute_all(video_paths, siglip_embedder, autogaze_model, cache_dir,
                   num_frames=16, device="cuda"):
    """Precompute both SigLIP embeddings and AutoGaze hidden states."""
    os.makedirs(cache_dir, exist_ok=True)

    semantic_model = SemanticAutoGaze(autogaze_model, embedding_dim=siglip_embedder.embed_dim)
    semantic_model.to(device)
    semantic_model.eval()

    valid_files = []
    for vp in tqdm(video_paths, desc="Precomputing features"):
        key = hashlib.md5(vp.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{key}.pt")

        if os.path.exists(cache_path):
            valid_files.append(cache_path)
            continue

        frames = read_video_frames(vp, num_frames)
        if frames is None:
            continue

        try:
            # SigLIP patch embeddings
            patch_emb = siglip_embedder.get_patch_embeddings(frames)  # (T, N, D)
            T, N, D = patch_emb.shape
            patch_emb_flat = patch_emb.reshape(T * N, D).cpu()

            # AutoGaze hidden states
            video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            video = video.unsqueeze(0).to(device)

            with torch.no_grad():
                hidden = semantic_model.get_patch_hidden_states(video)  # (1, T*N, hidden_dim)
            hidden_flat = hidden[0].cpu()  # (T*N, hidden_dim)

            torch.save({
                "hidden_states": hidden_flat,
                "patch_embeddings": patch_emb_flat,
            }, cache_path)
            valid_files.append(cache_path)

        except Exception as e:
            print(f"  Skipping {os.path.basename(vp)}: {e}")

    return valid_files


def train_fast(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb
    if "/" in args.wandb_project:
        entity, project = args.wandb_project.split("/", 1)
    else:
        entity, project = None, args.wandb_project
    wandb.init(
        entity=entity,
        project=project,
        name="kinetics-fast-training",
        config=vars(args),
    )

    device = torch.device(args.device)

    # Find videos
    if args.video_dir and os.path.isdir(args.video_dir):
        video_paths = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
        if not video_paths:
            video_paths = sorted(glob.glob(os.path.join(args.video_dir, "**/*.mp4"), recursive=True))
    else:
        print("ERROR: --video_dir required")
        sys.exit(1)

    print(f"Found {len(video_paths)} videos")

    # Load models for precomputation
    print("Loading SigLIP...")
    siglip_embedder = SigLIPEmbedder(model_name=args.siglip_model, device=device)

    print("Loading AutoGaze...")
    autogaze = AutoGaze.from_pretrained(args.autogaze_model, use_flash_attn=False)
    autogaze.to(device).eval()

    # Precompute everything
    cache_dir = os.path.join(args.output_dir, "precomputed_cache")
    valid_files = precompute_all(
        video_paths, siglip_embedder, autogaze, cache_dir,
        num_frames=args.num_frames, device=device,
    )
    print(f"{len(valid_files)}/{len(video_paths)} videos successfully processed")

    # Free GPU memory from the big models
    hidden_dim = autogaze.config.gaze_model_config.gaze_decoder_config.hidden_size
    embedding_dim = siglip_embedder.embed_dim
    del siglip_embedder, autogaze
    torch.cuda.empty_cache()

    # Create just the similarity head (no need for full model during training)
    head = SimilarityHead(hidden_dim, embedding_dim).to(device)
    print(f"Similarity head: {sum(p.numel() for p in head.parameters())/1e3:.1f}K params")

    # Split train/val
    random.shuffle(valid_files)
    split = int(0.9 * len(valid_files))
    train_files = valid_files[:split]
    val_files = valid_files[split:]
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    train_dataset = PrecomputedDataset(train_files)
    val_dataset = PrecomputedDataset(val_files)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        # Train
        head.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [train]")
        for batch in pbar:
            hidden = batch["hidden_states"].to(device)
            query = batch["query_embedding"].to(device)
            gt = batch["gt_similarities"].to(device)

            pred = head(hidden, query)
            loss = F.mse_loss(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.4f}")
            wandb.log({"train/loss": loss_val, "train/lr": optimizer.param_groups[0]["lr"]})

        scheduler.step()
        train_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(train_loss)

        # Validate
        head.eval()
        val_epoch_losses = []
        with torch.no_grad():
            for batch in val_loader:
                hidden = batch["hidden_states"].to(device)
                query = batch["query_embedding"].to(device)
                gt = batch["gt_similarities"].to(device)
                pred = head(hidden, query)
                val_epoch_losses.append(F.mse_loss(pred, gt).item())

        val_loss = sum(val_epoch_losses) / max(len(val_epoch_losses), 1)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        wandb.log({
            "train/epoch_loss": train_loss,
            "val/epoch_loss": val_loss,
            "train/epoch": epoch + 1,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(head.state_dict(), os.path.join(args.output_dir, "best_similarity_head.pt"))
            wandb.log({"val/best_loss": best_val_loss})

        torch.save(head.state_dict(), os.path.join(args.output_dir, "latest_similarity_head.pt"))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2, label='Train')
    ax.plot(range(1, len(val_losses) + 1), val_losses, 'r--', linewidth=2, label='Val')
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("MSE Loss", fontsize=14)
    ax.set_title("Semantic AutoGaze: Training on Kinetics-400", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "training_loss.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    wandb.log({"training_curve": wandb.Image(plot_path)})

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    parser.add_argument("--siglip_model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb_project", default="semantic-autogaze")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_fast(args)
