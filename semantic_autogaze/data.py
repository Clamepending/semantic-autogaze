"""
Data loading for Semantic AutoGaze training.

Uses SigLIP to compute per-patch embeddings as ground truth targets.
During training, a random patch embedding is sampled as the query,
and the model predicts similarity of all patches to that query.
"""

import os
import glob
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import av
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor


def read_video_frames(video_path, num_frames=16, size=224):
    """Read and sample frames from a video file."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames
    if total_frames == 0:
        # Estimate from duration
        total_frames = int(stream.duration * stream.average_rate) if stream.duration else 1000

    # Sample frame indices evenly
    if total_frames <= num_frames:
        indices = list(range(total_frames))
        # Pad by repeating last frame
        while len(indices) < num_frames:
            indices.append(indices[-1])
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            img = frame.to_image().resize((size, size), Image.BILINEAR)
            frames.append(np.array(img))
        if len(frames) == num_frames:
            break
    container.close()

    # Pad if we didn't get enough frames
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((size, size, 3), dtype=np.uint8))

    return np.stack(frames)  # (T, H, W, 3)


class SigLIPEmbedder:
    """Extracts per-patch SigLIP embeddings from images."""

    def __init__(self, model_name="google/siglip2-base-patch16-224", device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).vision_model.to(device).eval()
        self.embed_dim = self.model.config.hidden_size  # 768 for base

    @torch.no_grad()
    def get_patch_embeddings(self, frames):
        """
        Get per-patch embeddings for a batch of frames.

        Args:
            frames: numpy array (T, H, W, 3) uint8
        Returns:
            patch_embeddings: (T, N_patches, embed_dim)
        """
        # Process frames through SigLIP
        images = [Image.fromarray(f) for f in frames]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        outputs = self.model(
            pixel_values=inputs.pixel_values,
            output_hidden_states=True,
        )

        # Use the last hidden state which has per-patch embeddings
        # Shape: (T, N_patches+1, embed_dim) - first token is CLS
        hidden = outputs.last_hidden_state
        # SigLIP doesn't have a CLS token - all tokens are patch tokens
        patch_embeddings = hidden  # (T, N_patches, embed_dim)

        # Normalize embeddings for cosine similarity
        patch_embeddings = F.normalize(patch_embeddings, dim=-1)

        return patch_embeddings


class VideoSigLIPDataset(Dataset):
    """
    Dataset that loads videos and computes SigLIP patch embeddings.

    For each sample:
    - Loads a video clip
    - Computes SigLIP patch embeddings for all patches
    - Samples a random patch as the query
    - Computes cosine similarities as ground truth
    """

    def __init__(
        self,
        video_paths,
        siglip_embedder,
        num_frames=16,
        img_size=224,
        cache_embeddings=True,
    ):
        self.video_paths = video_paths
        self.siglip_embedder = siglip_embedder
        self.num_frames = num_frames
        self.img_size = img_size
        self.cache_embeddings = cache_embeddings
        self._embedding_cache = {}

    def __len__(self):
        return len(self.video_paths)

    def _get_embeddings(self, idx):
        """Get or compute SigLIP embeddings for a video."""
        if self.cache_embeddings and idx in self._embedding_cache:
            return self._embedding_cache[idx]

        video_path = self.video_paths[idx]
        frames = read_video_frames(video_path, self.num_frames, self.img_size)
        patch_embeddings = self.siglip_embedder.get_patch_embeddings(frames)

        if self.cache_embeddings:
            self._embedding_cache[idx] = (frames, patch_embeddings.cpu())
            return self._embedding_cache[idx]

        return frames, patch_embeddings.cpu()

    def __getitem__(self, idx):
        frames, patch_embeddings = self._get_embeddings(idx)
        # frames: (T, H, W, 3), patch_embeddings: (T, N_patches, embed_dim)

        T, N, D = patch_embeddings.shape

        # Sample a random patch as query
        query_frame = random.randint(0, T - 1)
        query_patch = random.randint(0, N - 1)
        query_embedding = patch_embeddings[query_frame, query_patch]  # (embed_dim,)

        # Compute ground truth similarities (cosine sim since embeddings are normalized)
        all_patches = patch_embeddings.reshape(T * N, D)  # (T*N, embed_dim)
        gt_similarities = torch.mv(all_patches, query_embedding)  # (T*N,)

        # Convert frames to tensor: (T, C, H, W), normalized to [-1, 1]
        video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0

        return {
            "video": video,  # (T, C, H, W)
            "query_embedding": query_embedding,  # (embed_dim,)
            "gt_similarities": gt_similarities,  # (T*N,)
            "query_frame": query_frame,
            "query_patch": query_patch,
        }


def download_sample_videos(output_dir, num_videos=20):
    """
    Generate training data by creating augmented clips from the example video,
    and optionally downloading additional videos.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Look for the example video relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example_video = os.path.join(project_root, "assets", "example_input.mp4")

    video_paths = []

    if os.path.exists(example_video):
        video_paths.append(example_video)

        # Create augmented versions by reading at different temporal offsets
        container = av.open(example_video)
        stream = container.streams.video[0]
        total_frames = 0
        frames_list = []
        for frame in container.decode(video=0):
            frames_list.append(frame.to_ndarray(format='rgb24'))
            total_frames += 1
        container.close()

        if total_frames > 16:
            for i in range(min(num_videos - 1, total_frames - 16)):
                offset = i * max(1, (total_frames - 16) // num_videos)
                if offset + 16 > total_frames:
                    break
                clip_frames = frames_list[offset:offset + 16]

                out_path = os.path.join(output_dir, f"clip_{i:03d}.mp4")
                output = av.open(out_path, mode='w')
                out_stream = output.add_stream('libx264', rate=8)
                out_stream.height = clip_frames[0].shape[0]
                out_stream.width = clip_frames[0].shape[1]
                out_stream.pix_fmt = 'yuv420p'

                for f in clip_frames:
                    av_frame = av.VideoFrame.from_ndarray(f, format='rgb24')
                    for packet in out_stream.encode(av_frame):
                        output.mux(packet)
                for packet in out_stream.encode():
                    output.mux(packet)
                output.close()
                video_paths.append(out_path)

    return video_paths


def create_dataloader(video_paths, siglip_embedder, batch_size=4, num_frames=16, num_workers=0):
    """Create a DataLoader for training."""
    dataset = VideoSigLIPDataset(
        video_paths=video_paths,
        siglip_embedder=siglip_embedder,
        num_frames=num_frames,
        cache_embeddings=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
