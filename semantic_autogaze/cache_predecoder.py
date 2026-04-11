"""
Cache pre-decoder (CNN+connector) features for all videos.

These features are extracted BEFORE the LLaMA decoder, so they are
raw visual patch embeddings with positional info but without the
4-layer transformer contextualization used for gaze decisions.

Same shape as post-decoder: (T*196, 192) per video.
"""

import os
import glob
import hashlib
import argparse
import torch
from einops import rearrange
import torch.nn.functional as F
from tqdm import tqdm

from autogaze.models.autogaze import AutoGaze
from semantic_autogaze.data import read_video_frames


def precompute_predecoder_features(video_paths, autogaze_model, cache_dir,
                                    num_frames=16, device="cuda"):
    """Extract features after CNN+connector, before LLaMA decoder."""
    os.makedirs(cache_dir, exist_ok=True)

    autogaze_model.to(device).eval()
    gaze_model = autogaze_model.gazing_model

    valid = {}
    for vp in tqdm(video_paths, desc="Precomputing pre-decoder features"):
        key = hashlib.md5(vp.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{key}_predecoder.pt")

        if os.path.exists(cache_path):
            valid[vp] = cache_path
            continue

        frames = read_video_frames(vp, num_frames, 224)
        if frames is None:
            continue

        try:
            video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            video = video.unsqueeze(0).to(device)  # (1, T, C, H, W)

            B, T = video.shape[:2]

            with torch.no_grad():
                # Resize to AutoGaze input size
                video_resized = rearrange(video, 'b t c h w -> (b t) c h w')
                video_resized = F.interpolate(
                    video_resized,
                    size=(gaze_model.input_img_size, gaze_model.input_img_size),
                    mode="bicubic", align_corners=False,
                )
                video_resized = rearrange(video_resized, '(b t) c h w -> b t c h w', b=B)

                # Vision encoder
                vision_features, _ = gaze_model.vision_model(video_resized)
                vision_features = vision_features.transpose(1, 2)
                vision_features = rearrange(vision_features, 'b t c h w -> b t (h w) c')

                # Connector (adds positional embeddings)
                vision_features = gaze_model.connector(vision_features)
                # vision_features: (1, T, 196, 192)

                # Flatten to (T*196, 192) — same shape as post-decoder cache
                predecoder = vision_features[0].reshape(T * 196, 192)

            torch.save(predecoder.cpu(), cache_path)
            valid[vp] = cache_path

        except Exception as e:
            print(f"  Skipping {os.path.basename(vp)}: {e}")

    print(f"  {len(valid)}/{len(video_paths)} videos cached")
    return valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="data")
    parser.add_argument("--cache_dir", default="results/distill/predecoder_cache")
    parser.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    video_paths = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    if not video_paths:
        video_paths = sorted(glob.glob(os.path.join(args.video_dir, "**/*.mp4"), recursive=True))
    print(f"Found {len(video_paths)} videos")

    autogaze = AutoGaze.from_pretrained(args.autogaze_model, use_flash_attn=False)
    precompute_predecoder_features(video_paths, autogaze, args.cache_dir,
                                    args.num_frames, args.device)
