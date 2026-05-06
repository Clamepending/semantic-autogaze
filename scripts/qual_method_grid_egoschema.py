"""Qualitative method grid on EgoSchema clips — visualize sparse-relevance signal.

For each chosen (video, query) pair:
  rows = methods (AutoGaze, CLIPSeg, Ours v1, Ours v2-Tiny, D-MobileNet, raw CLIP, OWL-ViT, random)
  cols = 8 frames evenly spaced across the 3-min ego clip

Each cell = heatmap overlay (the method's per-patch score at that frame, upsampled
to image resolution). Below each row, the per-frame mean-of-top-K-patch-scores curve
shows whether the method *concentrates* its score on a subset of frames — the
"needle-in-haystack" signal.

The figure tests the §1 Question's load-bearing claim: a text-conditioned scorer
should drop irrelevant patches (and frames) on long-form video, while AutoGaze's
text-blind selection should be uniform across frames.

Usage:
  CUDA_VISIBLE_DEVICES=N python -m scripts.qual_method_grid_egoschema \
      --device cuda:0 --output_dir results/qual_method_grid_egoschema \
      --library_figures_dir /home/ogata/mac-brain/projects/semantic-autogaze/figures
"""
from __future__ import annotations
import os, sys, json, argparse, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import av
import timm

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")
from train_independent_scorer import TextScorerHead, GRID
from train_independent_scorer_v2 import IM_MEAN as V2_IM_MEAN, IM_STD as V2_IM_STD
from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor

# (video_idx_prefix, scoring_query, label) — picked from EgoSchema Subset gt-answers.
# Prefix-resolution: we glob video_dir at runtime to find the full UUID since
# only an 8-char prefix is known here.
DEFAULT_VIDEOS = [
    ("01a144a5", "playing dominoes",       "q=00005 — domino game"),
    ("049249dc", "dog mat",                 "q=00010 — washing dog mat"),
    ("0c51c89d", "grating ginger",          "q=00033 — grating ginger"),
    ("083c5e8e", "laptop motherboard",      "q=00024 — laptop interior"),
    ("0aadf5ce", "building a shelf",        "q=00029 — shelf construction"),
    ("13b86f2e", "rubbing hands",           "q=00044 — hand rubbing"),
]

CLIP_NAME = "ViT-B-16"; CLIP_PRETRAINED = "openai"
OURS_CKPT = "/home/ogata/semantic-autogaze/results/independent_scorer/best_v1.pt"
OURS_V2_TINY_CKPT = "/home/ogata/semantic-autogaze/results/independent_scorer_v2_tiny/best.pt"
OURS_D_MOBILE_CKPT = "/home/ogata/semantic-autogaze/results/sweep_v3/D_mobilenet_std/best.pt"
AUTOGAZE_NAME = "nvidia/AutoGaze"
CLIPSEG_NAME = "CIDAS/clipseg-rd64-refined"

T_FRAMES = 8  # frames per row in the grid


def load_video_frames(mp4_path, n_frames=T_FRAMES):
    """Sample n_frames evenly from a video. Returns (n_frames, H, W, 3) uint8."""
    container = av.open(mp4_path)
    stream = container.streams.video[0]
    n = stream.frames or n_frames
    indices = np.linspace(0, n - 1, n_frames).round().astype(int).tolist()
    frames = read_video_pyav(container=container, indices=indices)
    container.close()
    return frames  # ndarray (T, H, W, 3) uint8


def to_clip_tensor(frames_np, device, mean, std, size=224):
    """(T, H, W, 3) uint8 -> (T, 3, size, size) normalized for CLIP/timm."""
    x = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float().to(device) / 255.0
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    x = (x - mean[None, :, None, None]) / std[None, :, None, None]
    return x


def heatmap_clip_visual_with_head(clip_model, head, x_clip, text_emb_512):
    """Run frozen CLIP visual + Ours head per frame -> (T, 14, 14) sigmoid scores."""
    clip_model.visual.output_tokens = True
    _, patches = clip_model.visual(x_clip)  # (T, 196, 768)
    clip_model.visual.output_tokens = False
    text = text_emb_512.expand(patches.shape[0], -1)
    scores = head(patches, text)  # (T, 196)
    return torch.sigmoid(scores).reshape(-1, GRID, GRID).detach().cpu().numpy()


def heatmap_timm_with_head(backbone, head, x, clip_model, text_token):
    """Run frozen timm backbone + head per frame -> (T, 14, 14) sigmoid scores."""
    feats = backbone.forward_features(x)
    if feats.dim() == 4:
        feats = F.interpolate(feats, size=(GRID, GRID), mode="bilinear", align_corners=False)
        feats = feats.permute(0, 2, 3, 1).reshape(feats.shape[0], GRID * GRID, feats.shape[1])
    elif feats.shape[1] == 197:
        feats = feats[:, 1:, :]
    text_emb = F.normalize(clip_model.encode_text(text_token), dim=-1)
    text = text_emb.expand(feats.shape[0], -1)
    scores = head(feats, text)
    return torch.sigmoid(scores).reshape(-1, GRID, GRID).detach().cpu().numpy()


def heatmap_clipseg(clipseg_model, clipseg_proc, frames_np, query, device):
    """Run CLIPSeg per frame -> (T, 14, 14) sigmoid scores."""
    from PIL import Image as PIL_Image
    out = []
    for f in frames_np:
        pil = PIL_Image.fromarray(f)
        inputs = clipseg_proc(text=[query], images=[pil], return_tensors="pt").to(device)
        with torch.no_grad():
            preds = clipseg_model(**inputs).logits  # (1, 352, 352)
        preds = torch.sigmoid(preds.unsqueeze(1))  # (1, 1, 352, 352)
        h = F.interpolate(preds, size=(GRID, GRID), mode="bilinear", align_corners=False)
        out.append(h.squeeze().cpu().numpy())
    return np.stack(out)


def heatmap_autogaze(autogaze, frames_np, autogaze_processor, device):
    """AutoGaze gives a saliency map (text-blind) per frame -> (T, 14, 14)."""
    # AutoGaze expects a video tensor.
    video = transform_video_for_pytorch(frames_np, autogaze_processor.image_processor).to(device)
    # AutoGaze is text-blind; we use only its visual saliency.
    with torch.no_grad():
        out = autogaze(video.unsqueeze(0))
    # Take per-frame patch scores, reshape to (T, 14, 14); pick the head's
    # spatial map.  Different AutoGaze configs return different things; do
    # a robust softmax over the patches dim.
    if hasattr(out, "patch_scores"):
        s = out.patch_scores
    elif hasattr(out, "scores"):
        s = out.scores
    else:
        s = out
    s = s.squeeze(0)  # (T, 196)?
    if s.dim() == 2 and s.shape[-1] == GRID * GRID:
        s = s.reshape(-1, GRID, GRID)
    return torch.sigmoid(s).cpu().numpy()


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.library_figures_dir:
        os.makedirs(args.library_figures_dir, exist_ok=True)

    # ---- Load all scorers once ----
    print("loading CLIP ViT-B/16 ...", flush=True)
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms(CLIP_NAME, pretrained=CLIP_PRETRAINED)
    clip_tok = open_clip.get_tokenizer(CLIP_NAME)
    clip_model = clip_model.to(device).eval()
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
    V2_MEAN = torch.tensor(V2_IM_MEAN, device=device)
    V2_STD = torch.tensor(V2_IM_STD, device=device)

    print("loading Ours v1 head ...", flush=True)
    ck1 = torch.load(OURS_CKPT, map_location=device)
    ca = ck1.get("args", {}) or {}
    head_v1 = TextScorerHead(
        patch_dim=768, text_dim=512,
        hidden_dim=ca.get("head_hidden_dim", 384),
        n_attn_heads=ca.get("head_attn_heads", 6),
        n_attn_layers=ca.get("head_attn_layers", 2),
        grid_size=GRID, use_spatial=ca.get("head_use_spatial", True),
    ).to(device).eval()
    head_v1.load_state_dict(ck1["head"])

    print("loading Ours v2-Tiny ...", flush=True)
    ck_t = torch.load(OURS_V2_TINY_CKPT, map_location=device)
    bb_t = timm.create_model(ck_t["backbone"], pretrained=True, num_classes=0).to(device).eval()
    ca = ck_t.get("args", {}) or {}
    head_v2t = TextScorerHead(
        patch_dim=ck_t["embed_dim"], text_dim=512,
        hidden_dim=ca.get("head_hidden_dim", 384),
        n_attn_heads=ca.get("head_attn_heads", 6),
        n_attn_layers=ca.get("head_attn_layers", 2),
        grid_size=GRID, use_spatial=ca.get("head_use_spatial", True),
    ).to(device).eval()
    head_v2t.load_state_dict(ck_t["head"])

    print("loading D MobileNet ...", flush=True)
    ck_d = torch.load(OURS_D_MOBILE_CKPT, map_location=device)
    bb_d = timm.create_model(ck_d["backbone"], pretrained=True, num_classes=0).to(device).eval()
    ca = ck_d.get("args", {}) or {}
    head_d = TextScorerHead(
        patch_dim=ck_d["embed_dim"], text_dim=512,
        hidden_dim=ca.get("head_hidden_dim", 384),
        n_attn_heads=ca.get("head_attn_heads", 6),
        n_attn_layers=ca.get("head_attn_layers", 2),
        grid_size=GRID, use_spatial=ca.get("head_use_spatial", True),
    ).to(device).eval()
    head_d.load_state_dict(ck_d["head"])

    print("loading CLIPSeg ...", flush=True)
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    clipseg_proc = CLIPSegProcessor.from_pretrained(CLIPSEG_NAME)
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_NAME).to(device).eval()

    methods = ["frame", "CLIPSeg", "Ours v1", "v2-Tiny", "D-Mobile", "random"]

    for video_prefix, query, label in args.videos:
        # Resolve 8-char prefix to full UUID
        cands = glob.glob(os.path.join(args.video_dir, f"{video_prefix}*.mp4"))
        if not cands:
            print(f"[skip] prefix {video_prefix}: no mp4 found", flush=True)
            continue
        mp4_path = cands[0]
        video_idx = Path(mp4_path).stem
        print(f"\n=== {label} ({video_idx}, query='{query}') ===", flush=True)
        frames_np = load_video_frames(mp4_path, n_frames=T_FRAMES)

        x_clip = to_clip_tensor(frames_np, device, CLIP_MEAN, CLIP_STD)
        x_v2 = to_clip_tensor(frames_np, device, V2_MEAN, V2_STD)
        text_token = clip_tok([query]).to(device)
        with torch.no_grad():
            text_emb = F.normalize(clip_model.encode_text(text_token), dim=-1)

        with torch.no_grad():
            hv1 = heatmap_clip_visual_with_head(clip_model, head_v1, x_clip, text_emb)
            hv2t = heatmap_timm_with_head(bb_t, head_v2t, x_v2, clip_model, text_token)
            hd = heatmap_timm_with_head(bb_d, head_d, x_v2, clip_model, text_token)
            hclipseg = heatmap_clipseg(clipseg_model, clipseg_proc, frames_np, query, device)
        hrand = np.random.RandomState(42).random((T_FRAMES, GRID, GRID))

        heatmaps = {
            "CLIPSeg": hclipseg,
            "Ours v1": hv1,
            "v2-Tiny": hv2t,
            "D-Mobile": hd,
            "random": hrand,
        }
        # Per-method per-frame top-K mean (K=27) → temporal selectivity track
        tracks = {}
        for m, h in heatmaps.items():
            flat = h.reshape(T_FRAMES, -1)
            topk_means = np.array([np.partition(flat[t], -27)[-27:].mean() for t in range(T_FRAMES)])
            tracks[m] = topk_means

        # ---- Plot ----
        n_methods = len(methods)
        fig = plt.figure(figsize=(2.0 * T_FRAMES, 1.7 * n_methods + 1.5))
        gs = gridspec.GridSpec(n_methods + 1, T_FRAMES, hspace=0.05, wspace=0.05,
                               height_ratios=[1.0] * n_methods + [0.5])
        for i, m in enumerate(methods):
            for t in range(T_FRAMES):
                ax = fig.add_subplot(gs[i, t])
                if m == "frame":
                    ax.imshow(frames_np[t])
                else:
                    h = heatmaps[m][t]
                    ax.imshow(frames_np[t], alpha=0.6)
                    h_up = np.kron(h, np.ones((frames_np.shape[1] // GRID + 1,
                                               frames_np.shape[2] // GRID + 1)))[
                                              :frames_np.shape[1], :frames_np.shape[2]]
                    ax.imshow(h_up, alpha=0.5, cmap="hot", vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values(): spine.set_visible(False)
                if t == 0:
                    ax.set_ylabel(m, fontsize=11, rotation=0, ha="right", va="center", labelpad=24)
                if i == 0:
                    ax.set_title(f"t={t+1}/{T_FRAMES}", fontsize=8)
        # Bottom row: per-method top-K-mean across time
        ax = fig.add_subplot(gs[-1, :])
        for m in methods:
            if m == "frame": continue
            ax.plot(range(1, T_FRAMES + 1), tracks[m], marker="o", label=m, alpha=0.8)
        ax.set_xlabel("frame index across 3-min clip"); ax.set_ylabel("top-27 mean score")
        ax.set_title("temporal selectivity (per-frame top-K-patch mean): "
                     "high = scorer concentrates on this frame")
        ax.legend(loc="upper right", fontsize=8, ncol=4)
        ax.grid(alpha=0.3)

        fig.suptitle(f"{label}\nquery: \"{query}\"", fontsize=12, y=1.005)
        out_png = os.path.join(args.output_dir, f"qual_egoschema_{video_idx[:8]}.png")
        plt.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"saved {out_png}", flush=True)
        if args.library_figures_dir:
            lib_png = os.path.join(args.library_figures_dir,
                                    f"qual-egoschema-{video_idx[:8]}.png")
            import shutil; shutil.copy(out_png, lib_png)
            print(f"  -> {lib_png}", flush=True)

    print("\ndone.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--video_dir", default="/home/ogata/semantic-autogaze/data/egoschema/videos")
    p.add_argument("--output_dir", default="results/qual_method_grid_egoschema")
    p.add_argument("--library_figures_dir",
                   default="/home/ogata/mac-brain/projects/semantic-autogaze/figures")
    p.add_argument("--videos", default=None, help="optional override; else uses DEFAULT_VIDEOS")
    args = p.parse_args()
    if args.videos is None:
        args.videos = DEFAULT_VIDEOS
    main(args)
