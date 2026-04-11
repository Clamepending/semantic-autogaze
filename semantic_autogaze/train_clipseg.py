"""
Train Semantic AutoGaze similarity head using CLIPSeg as ground truth.

Uses CLIPSeg's pixel-level segmentation as the training signal, with
CLIP ViT-B/16 text encoder for query embeddings (matching CLIPSeg's
internal text encoder for consistency).

Text vocabulary: LVIS (1203 categories) + custom additions.

Pipeline:
1. Precompute AutoGaze hidden states for each video
2. Precompute CLIPSeg heatmaps for sampled text queries per frame
3. Precompute CLIP text embeddings for each query
4. Train: hidden_states + clip_text_embedding → predict CLIPSeg heatmap
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
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import open_clip

from autogaze.models.autogaze import AutoGaze
from semantic_autogaze.model import SemanticAutoGaze, SimilarityHead
from semantic_autogaze.data import read_video_frames


def load_text_vocabulary(lvis_path=None):
    """Load LVIS categories + custom additions."""
    queries = []

    # Load LVIS categories
    if lvis_path and os.path.exists(lvis_path):
        with open(lvis_path) as f:
            queries = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(queries)} LVIS categories")

    # Add custom body parts / spatial concepts not in LVIS
    custom = [
        "right hand", "left hand", "fingers", "feet", "arm", "leg",
        "face", "eyes", "mouth", "hair", "head", "body", "torso",
        "sky", "ground", "floor", "wall", "grass", "water", "road",
        "shadow", "light", "background", "foreground",
        "person", "people", "human", "clothing",
    ]
    existing = set(q.lower() for q in queries)
    for c in custom:
        if c.lower() not in existing:
            queries.append(c)

    print(f"Total vocabulary: {len(queries)} queries")
    return queries


def precompute_clip_text_embeddings(queries, device="cuda"):
    """Precompute CLIP ViT-B/16 text embeddings for all queries."""
    print("Loading CLIP ViT-B/16 text encoder...")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model.eval()

    # CLIP ViT-B/16 always has 512-dim text embeddings
    embed_dim = 512

    text_embeddings = {}
    # Process in batches for speed
    batch_size = 64
    for i in tqdm(range(0, len(queries), batch_size), desc="Computing CLIP text embeddings"):
        batch = queries[i:i + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = F.normalize(feats, dim=-1)
        for j, text in enumerate(batch):
            text_embeddings[text] = feats[j].cpu()

    del model
    torch.cuda.empty_cache()
    print(f"Computed {len(text_embeddings)} text embeddings (dim={feats.shape[-1]})")
    return text_embeddings, feats.shape[-1]


def precompute_clipseg_targets(video_paths, text_queries, text_embeddings,
                                cache_dir, num_frames=16, grid_size=14,
                                device="cuda", queries_per_video=12, rounds=1):
    """Precompute CLIPSeg heatmaps for training. Multiple rounds = more data per video."""
    os.makedirs(cache_dir, exist_ok=True)

    print("Loading CLIPSeg...")
    clipseg_proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined"
    ).to(device).eval()

    valid_files = []
    for vp in tqdm(video_paths, desc="Precomputing CLIPSeg targets"):
        key = hashlib.md5(vp.encode()).hexdigest()

        for rnd in range(rounds):
            suffix = f"_clipseg_clip.pt" if rnd == 0 else f"_clipseg_clip_r{rnd}.pt"
            cache_path = os.path.join(cache_dir, f"{key}{suffix}")

            if os.path.exists(cache_path):
                valid_files.append(cache_path)
                continue

            frames = read_video_frames(vp, num_frames, 224)
            if frames is None:
                break

            try:
                T = frames.shape[0]
                queries = random.sample(text_queries, min(queries_per_video, len(text_queries)))

                all_query_data = []
                for text in queries:
                    frame_heatmaps = []
                    for t in range(T):
                        pil_img = Image.fromarray(frames[t])
                        inputs = clipseg_proc(
                            text=[text], images=[pil_img],
                            return_tensors="pt", padding=True,
                        ).to(device)
                        with torch.no_grad():
                            logits = clipseg_model(**inputs).logits[0]
                        logits_grid = F.adaptive_avg_pool2d(
                            logits.unsqueeze(0).unsqueeze(0),
                            (grid_size, grid_size),
                        )[0, 0]
                        frame_heatmaps.append(logits_grid.cpu())

                    heatmap_stack = torch.stack(frame_heatmaps)
                    heatmap_flat = heatmap_stack.reshape(T * grid_size * grid_size)

                    all_query_data.append({
                        "text": text,
                        "text_embedding": text_embeddings[text],
                        "target_scores": heatmap_flat,
                    })

                torch.save({
                    "video_path": vp,
                    "num_frames": T,
                    "queries": all_query_data,
                }, cache_path)
                valid_files.append(cache_path)

            except Exception as e:
                print(f"  Skipping {os.path.basename(vp)}: {e}")

    del clipseg_model, clipseg_proc
    torch.cuda.empty_cache()
    return valid_files


def precompute_hidden_states(video_paths, autogaze_model, cache_dir,
                              num_frames=16, device="cuda"):
    """Precompute AutoGaze hidden states."""
    os.makedirs(cache_dir, exist_ok=True)

    model = SemanticAutoGaze(autogaze_model, embedding_dim=768)
    model.to(device).eval()

    valid = {}
    for vp in tqdm(video_paths, desc="Precomputing AutoGaze hidden states"):
        key = hashlib.md5(vp.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{key}_hidden.pt")

        if os.path.exists(cache_path):
            valid[vp] = cache_path
            continue

        frames = read_video_frames(vp, num_frames, 224)
        if frames is None:
            continue

        try:
            video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            video = video.unsqueeze(0).to(device)

            with torch.no_grad():
                hidden = model.get_patch_hidden_states(video)

            torch.save(hidden[0].cpu(), cache_path)
            valid[vp] = cache_path
        except Exception as e:
            print(f"  Skipping {os.path.basename(vp)}: {e}")

    return valid


class CLIPSegDataset(Dataset):
    """Dataset of (hidden_states, clip_text_embedding, clipseg_target) triples.

    Preloads all hidden states into RAM to avoid repeated disk reads.
    """

    def __init__(self, clipseg_files, hidden_dir):
        self.samples = []
        # Cache hidden states in RAM (one per video, ~3GB total)
        hidden_cache = {}
        for cf in clipseg_files:
            data = torch.load(cf, map_location="cpu", weights_only=False)
            vp = data["video_path"]
            key = hashlib.md5(vp.encode()).hexdigest()
            hidden_path = os.path.join(hidden_dir, f"{key}_hidden.pt")
            if not os.path.exists(hidden_path):
                continue
            if hidden_path not in hidden_cache:
                hidden_cache[hidden_path] = torch.load(hidden_path, map_location="cpu", weights_only=True)
            for q in data["queries"]:
                self.samples.append({
                    "hidden_states": hidden_cache[hidden_path],
                    "text_embedding": q["text_embedding"],
                    "target_scores": q["target_scores"],
                })
        print(f"  Preloaded {len(hidden_cache)} hidden state files into RAM")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "hidden_states": s["hidden_states"],
            "text_embedding": s["text_embedding"],
            "target_scores": s["target_scores"],
        }


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb
    if "/" in args.wandb_project:
        entity, project = args.wandb_project.split("/", 1)
    else:
        entity, project = None, args.wandb_project
    wandb.init(entity=entity, project=project, name="clipseg-clip-lvis", config=vars(args))

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

    # 2. Precompute AutoGaze hidden states
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

    # 3. Precompute CLIPSeg targets
    print("\n=== Precomputing CLIPSeg targets ===")
    clipseg_dir = os.path.join(args.output_dir, "clipseg_cache")
    valid_video_paths = [vp for vp in video_paths if vp in hidden_map]
    clipseg_files = precompute_clipseg_targets(
        valid_video_paths, text_queries, text_embeddings,
        clipseg_dir, args.num_frames, grid_size=14,
        device=device, queries_per_video=args.queries_per_video,
        rounds=args.rounds,
    )
    print(f"  {len(clipseg_files)} CLIPSeg cache files")

    # 4. Create dataset and train
    print("\n=== Setting up training ===")
    random.shuffle(clipseg_files)
    split = int(0.9 * len(clipseg_files))
    train_files = clipseg_files[:split]
    val_files = clipseg_files[split:]

    train_dataset = CLIPSegDataset(train_files, hidden_dir)
    val_dataset = CLIPSegDataset(val_files, hidden_dir)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Create similarity head with CLIP embedding dim (512) + spatial conv refinement
    head = SimilarityHead(hidden_dim, embed_dim, grid_size=14, num_frames=args.num_frames,
                          use_spatial=True).to(device)
    print(f"Similarity head: {sum(p.numel() for p in head.parameters())/1e3:.1f}K params")
    print(f"  hidden_dim={hidden_dim}, embed_dim={embed_dim}, spatial=True")

    # Resume from checkpoint if requested
    if args.resume:
        resume_path = os.path.join(args.output_dir, "best_similarity_head.pt")
        if os.path.exists(resume_path):
            head.load_state_dict(torch.load(resume_path, map_location=device))
            print(f"Resumed from {resume_path}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(args.num_epochs):
        head.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [train]")
        for batch in pbar:
            hidden = batch["hidden_states"].to(device)
            query = batch["text_embedding"].to(device)
            target = batch["target_scores"].to(device)

            pred = head(hidden, query)
            # Normalize CLIPSeg logits to [0,1] soft targets via sigmoid
            target_soft = torch.sigmoid(target)
            loss = F.binary_cross_entropy_with_logits(pred, target_soft)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            lv = loss.item()
            epoch_losses.append(lv)
            pbar.set_postfix(loss=f"{lv:.4f}")
            wandb.log({"train/loss": lv, "train/lr": optimizer.param_groups[0]["lr"]})

        scheduler.step()
        train_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(train_loss)

        # Validate
        head.eval()
        val_epoch = []
        val_vis_samples = []  # collect a few samples for wandb visualization
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                hidden = batch["hidden_states"].to(device)
                query = batch["text_embedding"].to(device)
                target = batch["target_scores"].to(device)
                target_soft = torch.sigmoid(target)
                pred = head(hidden, query)
                val_epoch.append(F.binary_cross_entropy_with_logits(pred, target_soft).item())
                # Collect first few samples for visualization
                if batch_idx == 0 and (epoch + 1) % 5 == 0:
                    pred_sig = torch.sigmoid(pred)
                    for i in range(min(4, pred.shape[0])):
                        val_vis_samples.append({
                            "pred": pred_sig[i].cpu(),
                            "target": target_soft[i].cpu(),
                        })

        val_loss = sum(val_epoch) / max(len(val_epoch), 1)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")
        wandb.log({
            "train/epoch_loss": train_loss,
            "val/epoch_loss": val_loss,
            "train/epoch": epoch + 1,
        })

        # Log validation heatmap visualizations every 5 epochs
        if val_vis_samples and (epoch + 1) % 5 == 0:
            grid_size = 14
            n_samples = len(val_vis_samples)
            fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))
            if n_samples == 1:
                axes = axes[:, None]
            for i, vs in enumerate(val_vis_samples):
                # Extract first frame's grid from the flattened T*G*G scores
                pred_grid = vs["pred"][:grid_size * grid_size].reshape(grid_size, grid_size).numpy()
                tgt_grid = vs["target"][:grid_size * grid_size].reshape(grid_size, grid_size).numpy()
                axes[0, i].imshow(pred_grid, cmap="jet", vmin=0, vmax=1)
                axes[0, i].set_title(f"Pred [{pred_grid.min():.2f}, {pred_grid.max():.2f}]", fontsize=9)
                axes[0, i].axis("off")
                axes[1, i].imshow(tgt_grid, cmap="jet", vmin=0, vmax=1)
                axes[1, i].set_title(f"GT [{tgt_grid.min():.2f}, {tgt_grid.max():.2f}]", fontsize=9)
                axes[1, i].axis("off")
            axes[0, 0].set_ylabel("Predicted", fontsize=11)
            axes[1, 0].set_ylabel("GT (CLIPSeg)", fontsize=11)
            plt.suptitle(f"Validation Samples — Epoch {epoch+1}", fontsize=13)
            plt.tight_layout()
            wandb.log({"val/heatmaps": wandb.Image(fig)})
            plt.close(fig)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(head.state_dict(), os.path.join(args.output_dir, "best_similarity_head.pt"))
            wandb.log({"val/best_loss": best_val_loss})

        torch.save(head.state_dict(), os.path.join(args.output_dir, "latest_similarity_head.pt"))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(train_losses)+1), train_losses, "b-", lw=2, label="Train")
    ax.plot(range(1, len(val_losses)+1), val_losses, "r--", lw=2, label="Val")
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("MSE Loss", fontsize=14)
    ax.set_title("Semantic AutoGaze: CLIPSeg + CLIP + LVIS Training", fontsize=16)
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
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_dir", default="results/clipseg")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--queries_per_video", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb_project", default="semantic-autogaze")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint")
    parser.add_argument("--rounds", type=int, default=1, help="Number of query rounds per video (more = more data)")
    args = parser.parse_args()
    train(args)
