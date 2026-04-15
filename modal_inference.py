"""
Modal inference entrypoint for semantic-autogaze.

Runs a (video, text_query) → per-patch similarity heatmap on a Modal GPU.
Designed to be idempotent and resumable: HF model + any Modal volume
attached for checkpoints are re-used across invocations.

Usage (once Modal keys are set up):

    modal run modal_inference.py::run_demo --query "person running" \
        --video-path https://example.com/test.mp4

    # Or with a local file uploaded to a Modal volume:
    modal run modal_inference.py::run_demo --query "trash cans" \
        --video-path /ckpts/bair_sample.mp4 --head-ckpt /ckpts/best_bighead.pt

Status: SCAFFOLD. Checkpoint loading path requires a trained `.pt` that we
don't yet have (blocked on cluster return). The pipeline itself runs
end-to-end on a freshly initialized head as a plumbing test.
"""

from __future__ import annotations

import modal

APP_NAME = "semantic-autogaze-infer"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.11.0",
        "torchvision==0.26.0",
        "transformers==4.57.6",
        "timm==1.0.26",
        "open_clip_torch==3.3.0",
        "einops",
        "av",
        "imageio",
        "imageio-ffmpeg",
        "hydra-core>=1.3.2",
        "loguru",
        "omegaconf",
    )
    .add_local_dir("semantic_autogaze", remote_path="/root/semantic_autogaze", copy=True)
    .add_local_dir("autogaze", remote_path="/root/autogaze", copy=True)
)

# Optional: attach a volume at /ckpts for trained head checkpoints + sample videos.
ckpts = modal.Volume.from_name("semantic-autogaze-ckpts", create_if_missing=True)

app = modal.App(APP_NAME, image=image)


@app.cls(gpu="A10G", volumes={"/ckpts": ckpts}, scaledown_window=300, timeout=1200)
class Infer:
    autogaze_model_name: str = modal.parameter(default="nvidia/AutoGaze")

    @modal.enter()
    def load(self):
        import sys, torch
        sys.path.insert(0, "/root")
        from semantic_autogaze.inference import load_autogaze

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autogaze = load_autogaze(self.autogaze_model_name, self.device)
        print(f"Loaded AutoGaze on {self.device}.")

    @modal.method()
    def similarity(self, video_bytes: bytes, query: str, head_ckpt_path: str | None = None):
        """Return per-patch similarity for a (video, query) pair.

        head_ckpt_path : path INSIDE the Modal container (e.g. '/ckpts/best_bighead.pt').
            If None, a freshly initialized BigHead is used — plumbing test only.
        """
        import io, tempfile, torch
        import imageio.v3 as iio
        from semantic_autogaze.inference import (
            encode_text_clip,
            extract_patch_hidden,
            load_head,
        )
        from semantic_autogaze.bighead import BigHead

        # Decode video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            video_path = f.name
        frames = iio.imread(video_path, plugin="pyav")  # (T, H, W, C) uint8
        video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
        video = video.unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        hidden = extract_patch_hidden(self.autogaze, video)
        query_embed = encode_text_clip(query, self.device)

        if head_ckpt_path is None:
            head = BigHead().to(self.device).eval()
            head_type = "BigHead (FRESH INIT — plumbing test only)"
        else:
            head, head_type = load_head(head_ckpt_path, self.device)

        with torch.no_grad():
            logits = head(hidden, query_embed)
            probs = torch.sigmoid(logits)

        T = video.shape[1]
        grid = 14
        return {
            "query": query,
            "head_type": head_type,
            "probs_grid": probs.reshape(T, grid, grid).cpu().numpy().tolist(),
            "num_frames": T,
        }


@app.local_entrypoint()
def run_demo(video_path: str, query: str = "person", head_ckpt: str | None = None):
    """Local entrypoint. Reads a local MP4 and prints the first-frame heatmap."""
    from pathlib import Path
    vb = Path(video_path).read_bytes()
    result = Infer().similarity.remote(vb, query, head_ckpt)
    print(f"Query: {result['query']!r}  head: {result['head_type']}  frames: {result['num_frames']}")
    heatmap = result["probs_grid"][0]
    for row in heatmap:
        print("  " + " ".join(f"{x:.2f}" for x in row))
