"""Compact paper-ready figure: temporal-selectivity tracks for 4 EgoSchema videos.

Each subplot = one (video, query) pair. X axis = frame index 1-8 across the
3-min clip. Y axis = per-frame top-K-mean score for each method. The story:
trained scorers concentrate on relevant frames (high) and dim on irrelevant
frames (low), making the gradient observable. CLIPSeg is the teacher;
Ours v1/v2-Tiny/D-Mobile are the distilled students; random is the floor.

Reuses heatmap computations from qual_method_grid_egoschema by importing it.
"""
from __future__ import annotations
import os, sys, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import timm
import av

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")
from train_independent_scorer import TextScorerHead, GRID
from train_independent_scorer_v2 import IM_MEAN as V2_IM_MEAN, IM_STD as V2_IM_STD
from autogaze.datasets.video_utils import read_video_pyav

# Same picks as qual_method_grid_egoschema; using all 6 to give a 2x3 grid
PICKS = [
    ("01a144a5", "playing dominoes",       "q=00005 — domino game"),
    ("049249dc", "dog mat",                 "q=00010 — washing dog mat"),
    ("0c51c89d", "grating ginger",          "q=00033 — grating ginger"),
    ("083c5e8e", "laptop motherboard",      "q=00024 — laptop interior"),
    ("0aadf5ce", "building a shelf",        "q=00029 — shelf construction"),
    ("13b86f2e", "rubbing hands",           "q=00044 — hand rubbing"),
]
T = 8
VIDEO_DIR = "/home/ogata/semantic-autogaze/data/egoschema/videos"
OUT_PATH = "/home/ogata/mac-brain/projects/semantic-autogaze/figures/egoschema-temporal-selectivity.png"


def load_video_frames(mp4_path, n_frames=T):
    container = av.open(mp4_path)
    n = container.streams.video[0].frames or n_frames
    indices = np.linspace(0, n - 1, n_frames).round().astype(int).tolist()
    frames = read_video_pyav(container=container, indices=indices)
    container.close()
    return frames


def main():
    device = torch.device("cuda:0")
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()
    clip_model.visual.output_tokens = True
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
    V2_MEAN = torch.tensor(V2_IM_MEAN, device=device)
    V2_STD = torch.tensor(V2_IM_STD, device=device)

    def load_head(ckpt_path, fixed_patch_dim=None):
        ck = torch.load(ckpt_path, map_location=device)
        ca = ck.get("args", {}) or {}
        h = TextScorerHead(
            patch_dim=fixed_patch_dim or ck.get("embed_dim", 768), text_dim=512,
            hidden_dim=ca.get("head_hidden_dim", 384),
            n_attn_heads=ca.get("head_attn_heads", 6),
            n_attn_layers=ca.get("head_attn_layers", 2),
            grid_size=GRID, use_spatial=ca.get("head_use_spatial", True),
        ).to(device).eval()
        h.load_state_dict(ck["head"])
        return ck, h

    print("loading heads...", flush=True)
    ck1, head_v1 = load_head("/home/ogata/semantic-autogaze/results/independent_scorer/best_v1.pt", 768)
    ck_t, head_v2t = load_head("/home/ogata/semantic-autogaze/results/independent_scorer_v2_tiny/best.pt")
    bb_t = timm.create_model(ck_t["backbone"], pretrained=True, num_classes=0).to(device).eval()
    ck_d, head_d = load_head("/home/ogata/semantic-autogaze/results/sweep_v3/D_mobilenet_std/best.pt")
    bb_d = timm.create_model(ck_d["backbone"], pretrained=True, num_classes=0).to(device).eval()

    print("loading CLIPSeg...", flush=True)
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    clipseg_proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device).eval()

    def to_tensor(frames_np, mean, std):
        x = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float().to(device) / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        return (x - mean[None, :, None, None]) / std[None, :, None, None]

    def _adapt(feats, grid=GRID):
        if feats.dim() == 4:
            feats = F.interpolate(feats, size=(grid, grid), mode="bilinear", align_corners=False)
            return feats.permute(0, 2, 3, 1).reshape(feats.shape[0], grid * grid, feats.shape[1])
        if feats.shape[1] == 197:
            return feats[:, 1:, :]
        return feats

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
    axes = axes.flatten()
    method_colors = {"CLIPSeg": "#d62728", "Ours v1": "#1f77b4",
                     "v2-Tiny": "#2ca02c", "D-Mobile": "#ff7f0e", "random": "#7f7f7f"}

    for idx, (prefix, query, label) in enumerate(PICKS):
        cands = glob.glob(f"{VIDEO_DIR}/{prefix}*.mp4")
        if not cands:
            print(f"[skip] {prefix}: no mp4")
            continue
        frames_np = load_video_frames(cands[0])
        x_clip = to_tensor(frames_np, CLIP_MEAN, CLIP_STD)
        x_v2 = to_tensor(frames_np, V2_MEAN, V2_STD)
        toks = clip_tok([query]).to(device)
        with torch.no_grad():
            text_emb = F.normalize(clip_model.encode_text(toks), dim=-1)
            _, patches_v1 = clip_model.visual(x_clip)
            text_v1 = text_emb.expand(T, -1)
            hv1 = torch.sigmoid(head_v1(patches_v1, text_v1)).reshape(T, GRID, GRID).cpu().numpy()
            feats_t = _adapt(bb_t.forward_features(x_v2))
            hv2t = torch.sigmoid(head_v2t(feats_t, text_v1)).reshape(T, GRID, GRID).cpu().numpy()
            feats_d = _adapt(bb_d.forward_features(x_v2))
            hd = torch.sigmoid(head_d(feats_d, text_v1)).reshape(T, GRID, GRID).cpu().numpy()
            from PIL import Image as PIL_Image
            hclipseg = []
            for f in frames_np:
                pil = PIL_Image.fromarray(f)
                inp = clipseg_proc(text=[query], images=[pil], return_tensors="pt").to(device)
                preds = clipseg_model(**inp).logits.unsqueeze(1)
                h = F.interpolate(torch.sigmoid(preds), size=(GRID, GRID), mode="bilinear",
                                  align_corners=False)
                hclipseg.append(h.squeeze().cpu().numpy())
            hclipseg = np.stack(hclipseg)
        rng = np.random.RandomState(42)
        hrand = rng.random((T, GRID, GRID))

        heatmaps = {"CLIPSeg": hclipseg, "Ours v1": hv1,
                    "v2-Tiny": hv2t, "D-Mobile": hd, "random": hrand}
        ax = axes[idx]
        for m, h in heatmaps.items():
            flat = h.reshape(T, -1)
            top = np.array([np.partition(flat[t], -27)[-27:].mean() for t in range(T)])
            ax.plot(range(1, T + 1), top, marker="o", color=method_colors[m], label=m, alpha=0.85, lw=1.5)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("frame index"); ax.grid(alpha=0.3)
        if idx == 0:
            ax.set_ylabel("top-27 mean score (per frame)")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=1)

    fig.suptitle("EgoSchema temporal selectivity: per-frame top-K-patch mean across methods and videos\n"
                 "(higher = scorer concentrates on this frame; flat ≈ uniform = no temporal signal)",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
