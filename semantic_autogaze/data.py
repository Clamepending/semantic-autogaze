"""
Data loading for Semantic AutoGaze training.

Uses SigLIP to compute per-patch embeddings as ground truth targets.
During training, a random patch embedding is sampled as the query,
and the model predicts similarity of all patches to that query.
"""

import os
import glob
import random
import hashlib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import av
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm


def read_video_frames(video_path, num_frames=16, size=224):
    """Read and sample frames from a video file."""
    try:
        container = av.open(video_path)
    except Exception:
        return None

    stream = container.streams.video[0]
    total_frames = stream.frames
    if total_frames == 0:
        total_frames = int(stream.duration * stream.average_rate) if stream.duration else 1000

    if total_frames <= num_frames:
        indices = list(range(total_frames))
        while len(indices) < num_frames:
            indices.append(indices[-1])
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    frames = []
    try:
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                img = frame.to_image().resize((size, size), Image.BILINEAR)
                frames.append(np.array(img))
            if len(frames) == num_frames:
                break
    except Exception:
        pass
    container.close()

    if len(frames) == 0:
        return None

    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.stack(frames)  # (T, H, W, 3)


class SigLIPEmbedder:
    """Extracts per-patch SigLIP embeddings from images."""

    def __init__(self, model_name="google/siglip2-base-patch16-224", device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).vision_model.to(device).eval()
        self.embed_dim = self.model.config.hidden_size

    @torch.no_grad()
    def get_patch_embeddings(self, frames):
        """
        Get per-patch embeddings for a batch of frames.

        Args:
            frames: numpy array (T, H, W, 3) uint8
        Returns:
            patch_embeddings: (T, N_patches, embed_dim)
        """
        images = [Image.fromarray(f) for f in frames]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        outputs = self.model(
            pixel_values=inputs.pixel_values,
            output_hidden_states=True,
        )

        hidden = outputs.last_hidden_state
        patch_embeddings = F.normalize(hidden, dim=-1)

        return patch_embeddings

    def precompute_and_save(self, video_paths, cache_dir, num_frames=16, batch_size_frames=64):
        """Precompute SigLIP embeddings for all videos and save to disk."""
        os.makedirs(cache_dir, exist_ok=True)

        valid_paths = []
        for vp in tqdm(video_paths, desc="Precomputing SigLIP embeddings"):
            # Use hash of path as cache key
            key = hashlib.md5(vp.encode()).hexdigest()
            cache_path = os.path.join(cache_dir, f"{key}.pt")

            if os.path.exists(cache_path):
                valid_paths.append(vp)
                continue

            frames = read_video_frames(vp, num_frames)
            if frames is None:
                continue

            try:
                embeddings = self.get_patch_embeddings(frames).cpu()
                torch.save({"embeddings": embeddings, "path": vp}, cache_path)
                valid_paths.append(vp)
            except Exception as e:
                print(f"  Skipping {os.path.basename(vp)}: {e}")

        return valid_paths


class VideoSigLIPDataset(Dataset):
    """
    Dataset that loads videos and precomputed SigLIP patch embeddings.

    For each sample:
    - Loads a video clip
    - Loads precomputed SigLIP patch embeddings
    - Samples a random patch as the query
    - Computes cosine similarities as ground truth
    """

    def __init__(self, video_paths, cache_dir, num_frames=16, img_size=224):
        self.video_paths = video_paths
        self.cache_dir = cache_dir
        self.num_frames = num_frames
        self.img_size = img_size

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        vp = self.video_paths[idx]
        key = hashlib.md5(vp.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{key}.pt")

        # Load cached embeddings
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        patch_embeddings = cached["embeddings"]  # (T, N, D)

        # Load video frames
        frames = read_video_frames(vp, self.num_frames, self.img_size)
        if frames is None:
            # Return a dummy sample that will be filtered
            frames = np.zeros((self.num_frames, self.img_size, self.img_size, 3), dtype=np.uint8)

        T, N, D = patch_embeddings.shape

        # Sample a random patch as query
        query_frame = random.randint(0, T - 1)
        query_patch = random.randint(0, N - 1)
        query_embedding = patch_embeddings[query_frame, query_patch]

        # Ground truth similarities
        all_patches = patch_embeddings.reshape(T * N, D)
        gt_similarities = torch.mv(all_patches, query_embedding)

        # Convert frames to tensor
        video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0

        return {
            "video": video,
            "query_embedding": query_embedding,
            "gt_similarities": gt_similarities,
            "query_frame": query_frame,
            "query_patch": query_patch,
        }


def download_sample_videos(output_dir, num_videos=20):
    """Generate training clips from the example video."""
    os.makedirs(output_dir, exist_ok=True)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example_video = os.path.join(project_root, "assets", "example_input.mp4")

    video_paths = []
    if os.path.exists(example_video):
        video_paths.append(example_video)
        container = av.open(example_video)
        frames_list = []
        for frame in container.decode(video=0):
            frames_list.append(frame.to_ndarray(format='rgb24'))
        container.close()
        total_frames = len(frames_list)

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


def create_dataloader(video_paths, siglip_embedder, batch_size=4, num_frames=16,
                      num_workers=0, cache_dir=None):
    """Create a DataLoader for training with precomputed embeddings."""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "siglip_cache")

    # Precompute embeddings
    valid_paths = siglip_embedder.precompute_and_save(video_paths, cache_dir, num_frames)
    print(f"  {len(valid_paths)}/{len(video_paths)} videos have valid embeddings")

    dataset = VideoSigLIPDataset(
        video_paths=valid_paths,
        cache_dir=cache_dir,
        num_frames=num_frames,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
