"""Cycle 1 (light-finetune): cache hidden states from EVERY gaze-decoder layer.

For each image, runs AutoGaze with `output_hidden_states=True` on the
4-layer LLaMA decoder and saves a (4, 196, 192) fp16 tensor:
    [0..3] = output of decoder layers 0..3 (skips input embed to save disk)

Stored as fp16 -- per-patch hidden states are in roughly [-10, 10] so 3-4
sig figs of fp16 is plenty. fp16 keeps the 5K-image val cache near 1.5GB
instead of ~3.7GB; 5-layer fp32 doesn't fit on the dev box.

The current `cache_features.py` only saves layer 4 (last_hidden_state).
This lets the layer-ablation trainer pick which layer is the best
semantic-feature tap without re-running AutoGaze.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from einops import rearrange
from pycocotools.coco import COCO
from tqdm import tqdm

from semantic_autogaze.inference import load_autogaze
from semantic_autogaze.model import SemanticAutoGaze


@torch.inference_mode()
def cache_image(model: SemanticAutoGaze, video: torch.Tensor) -> torch.Tensor:
    """Returns (n_layers+1, N_patches, hidden_dim) stacked layer outputs."""
    B, T = video.shape[:2]
    gm = model.autogaze.gazing_model

    video_resized = rearrange(video, "b t c h w -> (b t) c h w")
    video_resized = F.interpolate(
        video_resized,
        size=(gm.input_img_size, gm.input_img_size),
        mode="bicubic", align_corners=False,
    )
    video_resized = rearrange(video_resized, "(b t) c h w -> b t c h w", b=B)

    vision_features, _ = gm.vision_model(video_resized)
    vision_features = vision_features.transpose(1, 2)
    vision_features = rearrange(vision_features, "b t c h w -> b t (h w) c")
    vision_features = gm.connector(vision_features)
    B_, T_sub, N, C = vision_features.shape
    inputs_embeds = vision_features.reshape(B_, T_sub * N, C)
    attention_mask = torch.ones(B_, T_sub * N, device=video.device, dtype=torch.long)

    outputs = gm.gaze_decoder.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=attention_mask.cumsum(dim=-1) - 1,
        output_hidden_states=True,
    )
    # outputs.hidden_states is a tuple of length n_layers+1 (input embed + 4 layer outputs).
    # We skip [0] (input embed) and keep [1..4] (each decoder layer's output).
    hs = torch.stack([h[0] for h in outputs.hidden_states[1:]], dim=0)  # (n_layers, T*N, hidden_dim)
    return hs.cpu().contiguous().half()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/coco")
    p.add_argument("--ann", default="annotations/instances_val2017.json")
    p.add_argument("--img-subdir", default="val2017")
    p.add_argument("--cache-dir", default="results/autogaze_probe/features_gaze_layers_val")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-images", type=int, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(os.path.join(args.data_dir, args.ann))
    img_dir = os.path.join(args.data_dir, args.img_subdir)
    img_ids = sorted(coco.getImgIds())
    if args.max_images:
        img_ids = img_ids[: args.max_images]

    already = sum(1 for i in img_ids if (cache_dir / f"{i}.pt").exists())
    print(f"[diag] {already}/{len(img_ids)} already cached")
    if already == len(img_ids):
        return

    print(f"[diag] loading AutoGaze on {device}...")
    ag = load_autogaze(device=device)
    w = SemanticAutoGaze(ag).to(device).eval()

    t0 = time.time()
    n_done_this = 0
    for i, img_id in enumerate(tqdm(img_ids, desc="cache layers")):
        out = cache_dir / f"{img_id}.pt"
        if out.exists():
            continue
        info = coco.imgs[img_id]
        try:
            img = Image.open(os.path.join(img_dir, info["file_name"])).convert("RGB")
        except FileNotFoundError:
            continue
        t = TF.to_tensor(img).unsqueeze(0).unsqueeze(0).to(device)
        hs = cache_image(w, t)  # (n_layers+1, 196, 192)
        torch.save(hs, out)
        n_done_this += 1
        if n_done_this % 200 == 0:
            elapsed = time.time() - t0
            rate = n_done_this / elapsed
            remaining = sum(1 for j in img_ids if not (cache_dir / f"{j}.pt").exists())
            eta = remaining / max(rate, 1e-9)
            print(f"  [{i+1}/{len(img_ids)}] new={n_done_this} {rate:.2f} img/s eta {eta/60:.1f}m")

    n_done = sum(1 for i in img_ids if (cache_dir / f"{i}.pt").exists())
    print(f"[diag] cached {n_done} files in {cache_dir} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
