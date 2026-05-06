"""Qualitative grid + scorer-latency bar for paper.md Section 4.

Renders, for each of N (video, query) pairs:
  cols = input frame | AutoGaze (text-blind) | CLIPSeg (target) |
         BigHead-CLIPSeg (student) | raw CLIP | raw SigLIP-2 | OWL-ViT
  rows = N pairs

Each cell is a [14, 14] patch heatmap on the middle frame, jet-overlayed on
the original frame. Also benchmarks per-video scorer wall time and writes a
bar chart of average ms.

Usage:
  CUDA_VISIBLE_DEVICES=N python -m scripts.qual_method_grid \
      --device cuda:0 --output_dir results/qual_method_grid
"""
from __future__ import annotations
import os, sys, glob, time, json, argparse, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import av
from PIL import Image

# ---- repo imports ----
sys.path.insert(0, "/home/ogata/semantic-autogaze")
from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper
from semantic_autogaze.train_bighead import BigSimilarityHead


GRID = 14
N_PATCH = GRID * GRID  # 196
T_FRAMES = 16
MID = T_FRAMES // 2  # frame 8

CLIP_NAME = "ViT-B-16"
CLIP_PRETRAINED = "openai"
CLIPSEG_NAME = "CIDAS/clipseg-rd64-refined"
SIGLIP2_NAME = "google/siglip2-base-patch16-224"
OWLVIT_NAME = "google/owlvit-base-patch32"
BIGHEAD_CKPT = "/home/ogata/semantic-autogaze/results/bighead/best_bighead.pt"
AUTOGAZE_NAME = "nvidia/AutoGaze"

# ---- Pair selection ----
# Pick (video_basename, query) with diverse, visually-distinct dominant subjects.
PAIRS = [
    ("0HN9Iz8MoYU_000006_000016.mp4", "bird"),
    ("-DUyMp0reKs_000147_000157.mp4", "people"),
    ("-myPMnDFa4Q_000003_000013.mp4", "bicycle"),
    ("0huAN2gC6Pc_000143_000153.mp4", "screen"),
]


# ---- Helpers ----

def load_video_and_frame(video_path, autogaze_transform, device):
    """Read 16 frames; return (video_tensor_for_autogaze [1,T,3,H,W], raw_frame_HWC_middle, all_raw_frames_THWC)."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    n = stream.frames or T_FRAMES
    indices = list(range(min(T_FRAMES, n)))
    raw_video = read_video_pyav(container=container, indices=indices)
    container.close()
    if raw_video.shape[0] < T_FRAMES:
        pad = np.repeat(raw_video[-1:], T_FRAMES - raw_video.shape[0], axis=0)
        raw_video = np.concatenate([raw_video, pad], axis=0)
    raw_video = raw_video[:T_FRAMES]
    video_autogaze = transform_video_for_pytorch(raw_video, autogaze_transform)[None].to(device)
    return video_autogaze, raw_video[MID], raw_video


# ---- Method heatmap functions: each returns a [14, 14] np.float32 heatmap of the middle frame ----

@torch.no_grad()
def heatmap_clipseg(clipseg_model, clipseg_proc, raw_frames_THWC, query, device):
    """Run CLIPSeg per-frame, return middle frame's heatmap downsampled to 14x14."""
    pil_frame = Image.fromarray(raw_frames_THWC[MID].astype(np.uint8))
    inputs = clipseg_proc(text=[query], images=[pil_frame], return_tensors="pt").to(device)
    out = clipseg_model(**inputs)
    logits = out.logits  # (1, H, W) — for clipseg-rd64 default 352
    probs = torch.sigmoid(logits)
    if probs.dim() == 2:
        probs = probs.unsqueeze(0)
    if probs.dim() == 3:
        probs = probs.unsqueeze(1)  # (1,1,H,W)
    hm = F.adaptive_avg_pool2d(probs.float(), (GRID, GRID)).squeeze().cpu().numpy()
    return hm


@torch.no_grad()
def heatmap_autogaze(autogaze, video_autogaze):
    """AutoGaze multi-scale gazing mask composited to a 14x14 heatmap on the middle frame.

    AutoGaze emits binary gazing masks at 4 scales: [4, 16, 49, 196] tokens/frame
    (corresponding to grid sizes [2, 4, 7, 14]). For a single 14x14 visualization
    we upsample each scale to 14x14 (nearest), then take the max — same pattern
    as semantic_autogaze.visualize.composite_gaze_masks. The output is in [0, 1].
    """
    out = autogaze({"video": video_autogaze},
                   gazing_ratio=0.5, task_loss_requirement=0.7, generate_only=True)
    masks = out["gazing_mask"]  # list of (1, T, N_scale)
    composite_T = torch.zeros(T_FRAMES, GRID, GRID, device=video_autogaze.device)
    for mask in masks:
        m = mask[0]  # (T, N)
        n = m.shape[1]
        g = int(round(n ** 0.5))
        m = m.float().reshape(T_FRAMES, 1, g, g)
        m_up = F.interpolate(m, size=(GRID, GRID), mode="nearest").squeeze(1)  # (T, 14, 14)
        composite_T = torch.maximum(composite_T, m_up)
    return composite_T[MID].cpu().numpy()


@torch.no_grad()
def heatmap_bighead(wrapper, bighead, video_autogaze, clip_text_emb_512):
    """BigHead-CLIPSeg student: AutoGaze hidden states -> BigHead head."""
    hidden = wrapper.extract_hidden_states(video_autogaze)  # (1, T*196, 192)
    scores = bighead(hidden, clip_text_emb_512)  # (1, T*196)
    scores = torch.sigmoid(scores)  # logits -> [0,1]
    sm = scores[0].reshape(T_FRAMES, GRID, GRID).cpu().numpy()
    return sm[MID]


@torch.no_grad()
def heatmap_raw_clip(clip_model, clip_text_emb_512, raw_frames_THWC, device,
                    clip_mean, clip_std):
    """raw CLIP ViT-B/16 per-patch cosine on middle frame."""
    frame = raw_frames_THWC[MID]  # (H, W, 3)
    img = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    img = F.interpolate(img, size=(224, 224), mode="bicubic", align_corners=False)
    img = (img - clip_mean[None, :, None, None]) / clip_std[None, :, None, None]

    clip_model.visual.output_tokens = True
    pooled, patch_tokens = clip_model.visual(img)  # (1,512), (1,196,768)
    if clip_model.visual.proj is not None:
        patch_proj = patch_tokens @ clip_model.visual.proj  # (1, 196, 512)
    else:
        patch_proj = patch_tokens
    patch_proj = F.normalize(patch_proj, dim=-1)
    text_norm = F.normalize(clip_text_emb_512, dim=-1)
    cos = (patch_proj.squeeze(0) * text_norm).sum(-1)  # (196,)
    clip_model.visual.output_tokens = False
    return cos.reshape(GRID, GRID).cpu().numpy()


@torch.no_grad()
def heatmap_siglip2(siglip2_model, siglip2_tok, raw_frames_THWC, query, device,
                   siglip2_size, siglip2_mean, siglip2_std):
    frame = raw_frames_THWC[MID]
    img = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    img = F.interpolate(img, size=siglip2_size, mode="bicubic", align_corners=False)
    img = (img - siglip2_mean[None, :, None, None]) / siglip2_std[None, :, None, None]
    out = siglip2_model.vision_model(pixel_values=img, interpolate_pos_encoding=False)
    patch = F.normalize(out.last_hidden_state, dim=-1)  # (1, 196, 768)

    enc = siglip2_tok([query], padding="max_length", return_tensors="pt").to(device)
    text_out = siglip2_model.text_model(input_ids=enc["input_ids"])
    text = F.normalize(text_out.pooler_output, dim=-1)  # (1, 768)
    cos = (patch.squeeze(0) * text).sum(-1)  # (196,)
    return cos.reshape(GRID, GRID).cpu().numpy()


@torch.no_grad()
def heatmap_owlvit(owlvit_det, owlvit_tok, raw_frames_THWC, query, device,
                  owlvit_size, owlvit_mean, owlvit_std):
    frame = raw_frames_THWC[MID]
    img = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    img = F.interpolate(img, size=(owlvit_size, owlvit_size), mode="bicubic", align_corners=False)
    img = (img - owlvit_mean[None, :, None, None]) / owlvit_std[None, :, None, None]
    image_embeds, _ = owlvit_det.image_embedder(pixel_values=img)
    image_class_embeds = owlvit_det.class_head.dense0(image_embeds)
    x = image_class_embeds.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(GRID, GRID), mode="bilinear", align_corners=False)
    x = x.permute(0, 2, 3, 1).contiguous().view(1, GRID * GRID, -1)
    x = F.normalize(x, dim=-1)

    enc = owlvit_tok([query], padding="max_length", return_tensors="pt").to(device)
    text_out = owlvit_det.owlvit.text_model(input_ids=enc["input_ids"])
    text = F.normalize(owlvit_det.owlvit.text_projection(text_out.pooler_output), dim=-1)
    cos = (x.squeeze(0) * text).sum(-1)
    return cos.reshape(GRID, GRID).cpu().numpy()


# ---- Latency benchmark wrappers (per video, frames in -> per-patch scores out) ----

def _bench(fn, n_warmup=5, n_trials=30):
    for _ in range(n_warmup):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


def main(args):
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load all models ----
    print("[setup] AutoGaze ...", flush=True)
    autogaze_transform = AutoGazeImageProcessor.from_pretrained(AUTOGAZE_NAME)
    autogaze = AutoGaze.from_pretrained(AUTOGAZE_NAME, use_flash_attn=False).to(device).eval()

    print("[setup] SemanticAutoGazeWrapper + BigHead ...", flush=True)
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=AUTOGAZE_NAME,
        head_ckpt=BIGHEAD_CKPT,
        head_type="bighead",
        device=str(device),
    )
    bighead = BigSimilarityHead(
        hidden_dim=192, embedding_dim=512, expanded_dim=384,
        n_attn_heads=6, n_attn_layers=2, grid_size=GRID,
    ).to(device).eval()
    bighead.load_state_dict(torch.load(BIGHEAD_CKPT, map_location=device))

    print("[setup] CLIPSeg ...", flush=True)
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    clipseg_proc = CLIPSegProcessor.from_pretrained(CLIPSEG_NAME)
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_NAME).to(device).eval()

    print("[setup] open_clip ViT-B-16 ...", flush=True)
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms(CLIP_NAME, pretrained=CLIP_PRETRAINED)
    clip_tok = open_clip.get_tokenizer(CLIP_NAME)
    clip_model = clip_model.to(device).eval()
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)

    print("[setup] SigLIP-2 base patch16-224 ...", flush=True)
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
    siglip2_model = AutoModel.from_pretrained(SIGLIP2_NAME).to(device).eval()
    siglip2_tok = AutoTokenizer.from_pretrained(SIGLIP2_NAME)
    siglip2_imgproc = AutoImageProcessor.from_pretrained(SIGLIP2_NAME)
    siglip2_size = (siglip2_imgproc.size["height"], siglip2_imgproc.size["width"]) \
        if isinstance(siglip2_imgproc.size, dict) else (224, 224)
    siglip2_mean = torch.tensor(siglip2_imgproc.image_mean, device=device)
    siglip2_std = torch.tensor(siglip2_imgproc.image_std, device=device)

    print("[setup] OWL-ViT base patch32 ...", flush=True)
    from transformers import OwlViTForObjectDetection
    owlvit_det = OwlViTForObjectDetection.from_pretrained(OWLVIT_NAME).to(device).eval()
    owlvit_tok = AutoTokenizer.from_pretrained(OWLVIT_NAME)
    owlvit_imgproc = AutoImageProcessor.from_pretrained(OWLVIT_NAME)
    owlvit_size = owlvit_imgproc.size["height"] if isinstance(owlvit_imgproc.size, dict) else 768
    owlvit_mean = torch.tensor(owlvit_imgproc.image_mean, device=device)
    owlvit_std = torch.tensor(owlvit_imgproc.image_std, device=device)

    # ---- Per-pair: extract heatmaps + middle frame ----
    pairs_data = []
    for vid_basename, query in PAIRS:
        vid_path = f"/home/ogata/semantic-autogaze/data/{vid_basename}"
        if not os.path.isfile(vid_path):
            print(f"[skip] missing video: {vid_path}", flush=True); continue
        print(f"[pair] {vid_basename}  query={query!r}", flush=True)
        video_autogaze, mid_frame, raw_frames = load_video_and_frame(vid_path, autogaze_transform, device)

        # CLIP text emb (512) for BigHead and raw CLIP
        toks = clip_tok([query]).to(device)
        clip_text_emb = F.normalize(clip_model.encode_text(toks), dim=-1)  # (1, 512)

        # Heatmaps (middle frame only, [14,14] each)
        hm_clipseg = heatmap_clipseg(clipseg_model, clipseg_proc, raw_frames, query, device)
        hm_autogaze = heatmap_autogaze(autogaze, video_autogaze)
        hm_bighead = heatmap_bighead(wrapper, bighead, video_autogaze, clip_text_emb)
        hm_clip = heatmap_raw_clip(clip_model, clip_text_emb, raw_frames, device, CLIP_MEAN, CLIP_STD)
        hm_siglip2 = heatmap_siglip2(siglip2_model, siglip2_tok, raw_frames, query, device,
                                     siglip2_size, siglip2_mean, siglip2_std)
        hm_owlvit = heatmap_owlvit(owlvit_det, owlvit_tok, raw_frames, query, device,
                                   owlvit_size, owlvit_mean, owlvit_std)

        pairs_data.append({
            "video": vid_basename,
            "query": query,
            "frame": mid_frame,
            "AutoGaze": hm_autogaze,
            "CLIPSeg": hm_clipseg,
            "BigHead": hm_bighead,
            "raw CLIP": hm_clip,
            "raw SigLIP-2": hm_siglip2,
            "OWL-ViT": hm_owlvit,
        })

    # ---- Latency benchmark (use the LAST loaded video as the bench input) ----
    bench_video = video_autogaze
    bench_query = query
    bench_clip_emb = clip_text_emb
    bench_raw_frames = raw_frames

    # AutoGaze, two costs:
    #   (a) "AutoGaze forward" = encoder + connector + decoder forward only (the
    #       deterministic feature stack BigHead's distillation also uses; matches the
    #       project's prior ~9.71ms pre-ViT reference);
    #   (b) "AutoGaze (deployed)" = forward + autoregressive generate at gazing_ratio=0.5
    #       (the actual deployed gater cost when AutoGaze is run end-to-end).
    def f_autogaze_forward():
        wrapper.extract_hidden_states(bench_video)

    def f_autogaze():
        autogaze({"video": bench_video}, gazing_ratio=0.5,
                 task_loss_requirement=0.7, generate_only=True)

    def f_bighead():
        hidden = wrapper.extract_hidden_states(bench_video)
        bighead(hidden, bench_clip_emb)

    def f_clipseg():
        # 16 frames, sequential
        for t in range(T_FRAMES):
            pil = Image.fromarray(bench_raw_frames[t].astype(np.uint8))
            inp = clipseg_proc(text=[bench_query], images=[pil], return_tensors="pt").to(device)
            clipseg_model(**inp)

    def f_clip():
        # raw CLIP per frame, projected patch features × text
        toks = clip_tok([bench_query]).to(device)
        text_emb = F.normalize(clip_model.encode_text(toks), dim=-1)
        clip_model.visual.output_tokens = True
        for t in range(T_FRAMES):
            frame = bench_raw_frames[t]
            img = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            img = F.interpolate(img, size=(224, 224), mode="bicubic", align_corners=False)
            img = (img - CLIP_MEAN[None, :, None, None]) / CLIP_STD[None, :, None, None]
            pooled, patch_tokens = clip_model.visual(img)
            if clip_model.visual.proj is not None:
                patch_proj = patch_tokens @ clip_model.visual.proj
            else:
                patch_proj = patch_tokens
            patch_proj = F.normalize(patch_proj, dim=-1)
            (patch_proj.squeeze(0) * text_emb).sum(-1)
        clip_model.visual.output_tokens = False

    def f_siglip2():
        enc = siglip2_tok([bench_query], padding="max_length", return_tensors="pt").to(device)
        text = F.normalize(siglip2_model.text_model(input_ids=enc["input_ids"]).pooler_output, dim=-1)
        for t in range(T_FRAMES):
            frame = bench_raw_frames[t]
            img = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            img = F.interpolate(img, size=siglip2_size, mode="bicubic", align_corners=False)
            img = (img - siglip2_mean[None, :, None, None]) / siglip2_std[None, :, None, None]
            out = siglip2_model.vision_model(pixel_values=img, interpolate_pos_encoding=False)
            p = F.normalize(out.last_hidden_state, dim=-1)
            (p.squeeze(0) * text).sum(-1)

    def f_owlvit():
        enc = owlvit_tok([bench_query], padding="max_length", return_tensors="pt").to(device)
        text = F.normalize(owlvit_det.owlvit.text_projection(
            owlvit_det.owlvit.text_model(input_ids=enc["input_ids"]).pooler_output), dim=-1)
        for t in range(T_FRAMES):
            frame = bench_raw_frames[t]
            img = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            img = F.interpolate(img, size=(owlvit_size, owlvit_size), mode="bicubic", align_corners=False)
            img = (img - owlvit_mean[None, :, None, None]) / owlvit_std[None, :, None, None]
            image_embeds, _ = owlvit_det.image_embedder(pixel_values=img)
            ic = owlvit_det.class_head.dense0(image_embeds)
            x = ic.permute(0, 3, 1, 2)
            x = F.interpolate(x, size=(GRID, GRID), mode="bilinear", align_corners=False)
            x = x.permute(0, 2, 3, 1).contiguous().view(1, GRID * GRID, -1)
            x = F.normalize(x, dim=-1)
            (x.squeeze(0) * text).sum(-1)

    print("[bench] timing each method on a 16-frame video (mean of 30 trials, 5 warmup) ...", flush=True)
    t0 = time.perf_counter()
    bench = {}
    for name, fn in [
        ("AutoGaze fwd",        f_autogaze_forward),
        ("AutoGaze (deployed)", f_autogaze),
        ("BigHead",             f_bighead),
        ("CLIPSeg",             f_clipseg),
        ("raw CLIP",            f_clip),
        ("raw SigLIP-2",        f_siglip2),
        ("OWL-ViT",             f_owlvit),
    ]:
        m, s = _bench(fn, n_warmup=5, n_trials=30)
        bench[name] = {"mean_ms": m, "std_ms": s}
        print(f"  {name:22s}  {m:8.2f} ± {s:5.2f} ms", flush=True)
    print(f"[bench] total wall {time.perf_counter()-t0:.1f}s", flush=True)

    # ---- Save data + render figures ----
    np.savez(out_dir / "qual_method_grid_data.npz",
             pairs=np.array(pairs_data, dtype=object),
             bench=np.array([bench], dtype=object))

    with open(out_dir / "bench.json", "w") as f:
        json.dump(bench, f, indent=2)

    # ---- Figure 1: qualitative grid ----
    # Heatmap-method names match the keys saved per pair (single AutoGaze
    # column: the deployed-gater binary mask). Latency annotation in the
    # column header uses the deployed cost.
    methods = ["AutoGaze", "CLIPSeg", "BigHead", "raw CLIP", "raw SigLIP-2", "OWL-ViT"]
    bench_for_grid = {
        "AutoGaze": bench["AutoGaze (deployed)"],
        "CLIPSeg": bench["CLIPSeg"],
        "BigHead": bench["BigHead"],
        "raw CLIP": bench["raw CLIP"],
        "raw SigLIP-2": bench["raw SigLIP-2"],
        "OWL-ViT": bench["OWL-ViT"],
    }
    cols = ["input"] + methods
    n_rows = len(pairs_data)
    n_cols = len(cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.6 * n_rows))
    if n_rows == 1: axes = np.array([axes])

    for r, pd in enumerate(pairs_data):
        frame = pd["frame"]
        H, W = frame.shape[:2]
        # input
        axes[r, 0].imshow(frame)
        axes[r, 0].set_ylabel(f'"{pd["query"]}"', fontsize=11, rotation=0,
                              ha="right", va="center", labelpad=18)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("input frame", fontsize=11, fontweight="bold")
        # method heatmaps
        for c, m in enumerate(methods, start=1):
            hm = pd[m]
            # normalize to [0, 1] for display
            lo, hi = float(np.min(hm)), float(np.max(hm))
            hm_n = (hm - lo) / max(1e-8, hi - lo)
            hm_up = np.array(Image.fromarray(hm_n.astype(np.float32)).resize((W, H), Image.BILINEAR))
            cmap = plt.get_cmap("jet")
            colored = cmap(hm_up)[:, :, :3]
            overlay = (frame / 255.0) * 0.45 + colored * 0.55
            axes[r, c].imshow(overlay.clip(0, 1))
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                ms = bench_for_grid[m]["mean_ms"]
                lab = f"{m}\n{ms:.1f} ms"
                axes[r, c].set_title(lab, fontsize=10, fontweight="bold")

    fig.suptitle("Per-patch heatmaps for each candidate scorer (middle frame, query in row label).",
                 fontsize=12, y=0.995)
    plt.tight_layout()
    grid_path = out_dir / "qualitative-method-grid.png"
    plt.savefig(grid_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved {grid_path}", flush=True)

    # ---- Figure 2: latency bar (log scale; both AutoGaze costs shown) ----
    fig2, ax = plt.subplots(figsize=(8.5, 4.2))
    names = ["AutoGaze fwd", "AutoGaze (deployed)", "BigHead", "CLIPSeg",
             "raw CLIP", "raw SigLIP-2", "OWL-ViT"]
    means = [bench[n]["mean_ms"] for n in names]
    stds = [bench[n]["std_ms"] for n in names]
    palette = {
        "AutoGaze fwd":         "#2ca02c",   # speed bar (forward only — what BigHead also runs)
        "AutoGaze (deployed)":  "#9ec39e",   # full deployed cost (forward + autoreg generate)
        "CLIPSeg":              "#d62728",   # quality target
        "BigHead":              "#1f77b4",   # our student
        "raw CLIP":             "#7f7f7f",
        "raw SigLIP-2":         "#7f7f7f",
        "OWL-ViT":              "#7f7f7f",
    }
    colors = [palette[n] for n in names]
    bars = ax.bar(names, means, yerr=stds, color=colors, capsize=4,
                  edgecolor="black", linewidth=0.5)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.08,
                f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(bench["AutoGaze fwd"]["mean_ms"], color="#2ca02c",
               linestyle="--", linewidth=1.0,
               label=f"AutoGaze fwd ({bench['AutoGaze fwd']['mean_ms']:.1f} ms) — speed target")
    ax.set_yscale("log")
    ax.set_ylabel("Scorer wall time per 16-frame video (ms, log scale)")
    ax.set_title("Scorer-only latency on RTX 4090 (no NVILA ViT/LLM)")
    ax.legend(loc="upper left", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    bar_path = out_dir / "scorer-latency-bar.png"
    plt.savefig(bar_path, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"[fig] saved {bar_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="/home/ogata/semantic-autogaze/results/qual_method_grid")
    args = p.parse_args()
    main(args)
