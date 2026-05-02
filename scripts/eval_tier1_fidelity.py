"""Tier-1 cheap-screen evaluation for the open-vocab semantic patch-filter.

Compares three patch-importance signals on (frame, question) triples sampled
from the HLVid household subset:

  1. phase24d  — ConvNeXt-atto + per-query SigLIP-bias MLP (the candidate)
  2. AutoGaze  — text-agnostic scene-saliency (vanilla baseline)
  3. CLIPSeg   — HF CIDAS/clipseg-rd64-refined, soft-target oracle

Per-triple metrics:
  - patch_iou(phase24d, autogaze) over top-K masks at K = 20% of 196 patches
  - clipseg_mass_retention = sum(clipseg_14x14 * top_k_phase24d) / sum(clipseg_14x14)
  - is_ocr_heavy: question contains text/say/word/label/sign/number

Stratified summary: overall, ocr-heavy, non-ocr.

Latency: phase24d single-frame forward, 100 trials w/ 10 warm-up, GPU wall time.

Fallback notes (documented in summary.json):
  - vanilla AutoGaze requires a 16-frame video tensor. We feed AutoGaze a
    16-frame window around the sampled frame_idx and read the per-frame mask
    for our target frame. If AutoGaze import or load fails, we fall back to a
    text-agnostic CLIP-ViT-B/16 visual self-attention map (MaskCLIP-style
    value-only attention, pre-text-norm) — same idea: pure scene saliency.
  - Question source priority:
      (a) results/hlvid_household_expand/sweep.log — only if it parses cleanly
          and the referenced clip_household_video_*.mp4 files are reachable.
      (b) HLVid parquet (loaded via eval_hlvid_subset.load_subset) — used when
          videos exist in either data/hlvid_videos/ or extracted_household/.
      (c) random frames + synthetic queries from data/ mp4s.

Usage:
  python3 scripts/eval_tier1_fidelity.py \
    --ckpt results/phase24d_atto_mpp05_aggrAug_10k/best_val.pt \
    --output_dir results/tier1_fidelity \
    --device cuda:0 --n_samples 50

This script is a CHEAP SCREEN — keep dependencies, models, and the main loop
straightforward. Do not over-engineer.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = "/home/ogata/semantic-autogaze"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

GRID = 14
N_PATCH = GRID * GRID  # 196
T_FRAMES = 16  # AutoGaze expects 16 frames

OCR_PATTERN = re.compile(
    r"\btext\b|\bsay(s)?\b|\bword(s)?\b|\blabel\b|\bsign\b|\bnumber(s)?\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Frame & video utilities
# ---------------------------------------------------------------------------

def open_container(video_path: str):
    """Open an mp4 container with PyAV; return (container, n_frames)."""
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    n = stream.frames or 0
    return container, n


def extract_pil_frame(video_path: str, frame_idx: int) -> Image.Image:
    """Extract a single frame at frame_idx (clamped) and return as a PIL RGB image."""
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames or 0
    if total > 0:
        frame_idx = max(0, min(frame_idx, total - 1))

    out = None
    for i, frame in enumerate(container.decode(video=0)):
        if i == frame_idx:
            out = frame.to_ndarray(format="rgb24")
            break
    container.close()

    if out is None:
        # fallback: re-open and grab first frame
        container = av.open(video_path)
        for frame in container.decode(video=0):
            out = frame.to_ndarray(format="rgb24")
            break
        container.close()
    if out is None:
        raise RuntimeError(f"Could not decode any frame from {video_path}")
    return Image.fromarray(out)


def extract_window_around(video_path: str, frame_idx: int, n: int = T_FRAMES) -> np.ndarray:
    """Extract n consecutive frames as (n, H, W, 3) uint8 array centered around frame_idx
    (right-padded by repeating the last frame if the clip ends early)."""
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames or 0
    if total <= 0:
        # Decode all to count
        all_frames = []
        for f in container.decode(video=0):
            all_frames.append(f.to_ndarray(format="rgb24"))
        container.close()
        total = len(all_frames)
        start = max(0, min(frame_idx - n // 2, max(0, total - n)))
        sel = all_frames[start:start + n]
        if len(sel) < n:
            pad = [sel[-1]] * (n - len(sel)) if sel else []
            sel = sel + pad
        return np.stack(sel, axis=0).astype(np.uint8)

    start = max(0, min(frame_idx - n // 2, max(0, total - n)))
    end = start + n
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i < start:
            continue
        if i >= end:
            break
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")
    while len(frames) < n:
        frames.append(frames[-1])
    return np.stack(frames, axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Question / triple sampling
# ---------------------------------------------------------------------------

SYNTH_QUERIES = [
    "person", "hand", "table", "chair", "screen", "book", "cup",
    "bottle", "laptop", "phone", "text on the box", "the green sign",
    "a label", "a number", "the kitchen counter", "a piece of paper",
]


def parse_sweep_log(path: str):
    """Parse the legacy sweep.log format: '  q[8] clip_household_video_0_000.mp4: <stem>'."""
    triples = []
    if not os.path.isfile(path):
        return triples
    pat = re.compile(r"^\s*q\[(\d+)\]\s+(\S+\.mp4):\s+(.+)$")
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = pat.match(line)
            if not m:
                continue
            qid = int(m.group(1))
            video = m.group(2)
            question = m.group(3).strip()
            triples.append({"qid": qid, "video_basename": video, "question": question})
    return triples


def find_video(basename: str, search_dirs):
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        cand = os.path.join(d, basename)
        if os.path.isfile(cand):
            return cand
    return None


def load_question_pool(args):
    """Return a list of dicts {video_path, question}. Tries sources in order."""
    pool = []
    notes = []

    # ---- (a) sweep.log ----
    sweep_log = os.path.join(REPO_ROOT, "results/hlvid_household_expand/sweep.log")
    sweep_triples = parse_sweep_log(sweep_log)
    sweep_search = [
        os.path.join(REPO_ROOT, "data/hlvid_videos"),
        os.path.join(REPO_ROOT, "data/hlvid_subset_v3_kshrink"),
        os.path.join(REPO_ROOT, "hlvid_videos/extracted_household/videos"),
        os.path.join(REPO_ROOT, "hlvid_videos/extracted/videos"),
    ]
    matched_sweep = []
    for t in sweep_triples:
        vp = find_video(t["video_basename"], sweep_search)
        if vp:
            matched_sweep.append({"video_path": vp, "question": t["question"]})
    if matched_sweep:
        notes.append(
            f"sweep.log: parsed {len(sweep_triples)} entries, matched {len(matched_sweep)} mp4s")
        pool = matched_sweep

    # ---- (b) HLVid parquet via eval_hlvid_subset.load_subset ----
    if not pool:
        try:
            from semantic_autogaze.eval_hlvid_subset import load_subset, PARQUET_PATH
            for video_dir in [
                os.path.join(REPO_ROOT, "data/hlvid_videos"),
                os.path.join(REPO_ROOT, "hlvid_videos/extracted_household/videos"),
                os.path.join(REPO_ROOT, "hlvid_videos/extracted/videos"),
            ]:
                if not os.path.isdir(video_dir):
                    continue
                if not os.path.isfile(PARQUET_PATH):
                    continue
                samples = load_subset(video_dir, PARQUET_PATH, query_mode="stem")
                if samples:
                    pool = [
                        {"video_path": s["video_path"],
                         "question": s["question_stem"]}
                        for s in samples
                    ]
                    notes.append(
                        f"hlvid parquet: {len(pool)} samples from {video_dir}")
                    break
        except Exception as e:
            notes.append(f"hlvid parquet load failed: {type(e).__name__}: {e}")

    # ---- (c) synth: random frames + canned queries from any mp4s on disk ----
    if not pool:
        notes.append("falling back to synthetic queries on data/ mp4s")
        cand_dirs = [
            args.hlvid_dir,
            os.path.join(REPO_ROOT, "data"),
            os.path.join(REPO_ROOT, "hlvid_videos/extracted/videos"),
            os.path.join(REPO_ROOT, "hlvid_videos/extracted_household/videos"),
        ]
        videos = []
        for d in cand_dirs:
            if not d or not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(".mp4"):
                    videos.append(os.path.join(d, f))
            if videos:
                break
        if not videos:
            raise RuntimeError(
                "Could not locate any mp4 file for question fallback. Tried: "
                + ", ".join(cand_dirs)
            )
        for v in videos:
            for q in SYNTH_QUERIES:
                pool.append({"video_path": v, "question": q})

    return pool, notes


def sample_triples(pool, n_samples: int, seed: int):
    """Sample n triples (video, frame_idx, question) deterministically.

    For each pool entry we read the clip's frame count once and pick a uniformly
    random frame_idx. Sampling proceeds without replacement on pool entries,
    falling back to with-replacement if the pool is small.
    """
    rng = random.Random(seed)
    triples = []
    if len(pool) >= n_samples:
        chosen = rng.sample(pool, n_samples)
    else:
        chosen = [rng.choice(pool) for _ in range(n_samples)]

    for entry in chosen:
        vp = entry["video_path"]
        try:
            container, n = open_container(vp)
            container.close()
        except Exception:
            n = 0
        if n <= 0:
            n = 240  # generous guess; extract_pil_frame will clamp
        frame_idx = rng.randrange(0, max(1, n))
        triples.append({
            "video_path": vp,
            "video": os.path.basename(vp),
            "frame_idx": int(frame_idx),
            "question": entry["question"],
        })
    return triples


# ---------------------------------------------------------------------------
# Model: phase24d (ConvNeXt-atto + SigLIP-bias-MLP) — uses eval_phase2_ckpt
# ---------------------------------------------------------------------------

def load_phase24d(ckpt_path, device):
    from eval_phase2_ckpt import load_ckpt
    return load_ckpt(ckpt_path, device)


def load_clip_text(device):
    import open_clip

    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()
    return clip_model, clip_tok


@torch.no_grad()
def heatmap_phase24d(pil, query, ph24, clip_model, clip_tok, device):
    from eval_phase2_ckpt import heatmap_one
    bb_fn, head, sb, mean, std, _model, _kind, _bb_module, obj_head = ph24
    return heatmap_one(
        pil, query, bb_fn, head, sb, mean, std,
        clip_model, clip_tok, device, obj_head=obj_head,
    )


# ---------------------------------------------------------------------------
# Model: vanilla AutoGaze (with MaskCLIP-style fallback)
# ---------------------------------------------------------------------------

class AutoGazeRunner:
    def __init__(self, device):
        self.device = device
        self.kind = None
        self.autogaze = None
        self.autogaze_transform = None
        self.clip_visual = None
        self.clip_mean = None
        self.clip_std = None
        self.notes = []
        self._try_load(device)

    def _try_load(self, device):
        try:
            from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
            from autogaze.datasets.video_utils import (
                read_video_pyav, transform_video_for_pytorch,
            )  # noqa: F401  (consumed via heatmap call)
            self._read_video_pyav = read_video_pyav
            self._transform_video_for_pytorch = transform_video_for_pytorch
            self.autogaze_transform = AutoGazeImageProcessor.from_pretrained(
                "nvidia/AutoGaze")
            self.autogaze = AutoGaze.from_pretrained(
                "nvidia/AutoGaze", use_flash_attn=False).to(device).eval()
            self.kind = "autogaze"
            self.notes.append("AutoGaze loaded (nvidia/AutoGaze).")
            return
        except Exception as e:
            self.notes.append(
                f"AutoGaze unavailable ({type(e).__name__}: {e}); "
                f"falling back to MaskCLIP-style CLIP self-attention."
            )

        # ---- fallback ----
        try:
            import open_clip
            cm, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained="openai")
            cm = cm.to(device).eval()
            self.clip_visual = cm.visual
            self.clip_mean = torch.tensor(
                [0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
            self.clip_std = torch.tensor(
                [0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
            self.kind = "maskclip"
            self.notes.append("Loaded CLIP ViT-B/16 visual for fallback saliency.")
        except Exception as e:
            self.notes.append(
                f"CLIP fallback also failed: {type(e).__name__}: {e}; "
                f"will return uniform 14x14 mask.")
            self.kind = "uniform"

    @torch.no_grad()
    def heatmap(self, video_path: str, frame_idx: int, pil_frame: Image.Image) -> np.ndarray:
        if self.kind == "autogaze":
            return self._heatmap_autogaze(video_path, frame_idx)
        if self.kind == "maskclip":
            return self._heatmap_maskclip(pil_frame)
        return np.ones((GRID, GRID), dtype=np.float32) / N_PATCH

    @torch.no_grad()
    def _heatmap_autogaze(self, video_path, frame_idx):
        raw_video = extract_window_around(video_path, frame_idx, n=T_FRAMES)
        # Find which slot the target frame ended up in (it's centered, so T/2,
        # but for short clips it could be different — recompute).
        import av
        with av.open(video_path) as c:
            total = c.streams.video[0].frames or 0
        start = max(0, min(frame_idx - T_FRAMES // 2, max(0, total - T_FRAMES)))
        slot = max(0, min(frame_idx - start, T_FRAMES - 1))

        video_t = self._transform_video_for_pytorch(raw_video, self.autogaze_transform)
        video_t = video_t[None].to(self.device)
        out = self.autogaze(
            {"video": video_t},
            gazing_ratio=0.5, task_loss_requirement=0.7, generate_only=True,
        )
        masks = out["gazing_mask"]  # list of (1, T, N_scale)
        composite = torch.zeros(T_FRAMES, GRID, GRID, device=self.device)
        for mask in masks:
            m = mask[0]  # (T, N)
            n = m.shape[1]
            g = int(round(n ** 0.5))
            m = m.float().reshape(T_FRAMES, 1, g, g)
            m_up = F.interpolate(m, size=(GRID, GRID), mode="nearest").squeeze(1)
            composite = torch.maximum(composite, m_up)
        return composite[slot].cpu().numpy()

    @torch.no_grad()
    def _heatmap_maskclip(self, pil_frame):
        """Text-agnostic CLIP visual saliency (MaskCLIP-style value-only attn).

        We approximate by taking the cosine norm of each patch's projected
        embedding to the CLIP CLS embedding — captures object-likeness without
        any text input. Output is the per-patch CLS-similarity, min-max
        normalized to [0,1].
        """
        arr = np.array(pil_frame.resize((224, 224), Image.BICUBIC))
        x = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        x = (x - self.clip_mean) / self.clip_std
        self.clip_visual.output_tokens = True
        pooled, patch_tokens = self.clip_visual(x)  # (1,512), (1,196,768)
        if self.clip_visual.proj is not None:
            patch_proj = patch_tokens @ self.clip_visual.proj  # (1, 196, 512)
        else:
            patch_proj = patch_tokens
        patch_proj = F.normalize(patch_proj, dim=-1)
        cls = F.normalize(pooled, dim=-1)
        cos = (patch_proj.squeeze(0) * cls).sum(-1)  # (196,)
        self.clip_visual.output_tokens = False
        cos = cos.reshape(GRID, GRID)
        # min-max normalize to [0, 1] for stable top-K + visualization
        c_min = cos.min()
        c_max = cos.max()
        if (c_max - c_min).item() > 1e-8:
            cos = (cos - c_min) / (c_max - c_min)
        else:
            cos = cos * 0 + 0.5
        return cos.cpu().numpy()


# ---------------------------------------------------------------------------
# Model: CLIPSeg
# ---------------------------------------------------------------------------

class CLIPSegRunner:
    def __init__(self, device):
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

        self.device = device
        self.proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined").to(device).eval()

    @torch.no_grad()
    def heatmap(self, pil_frame: Image.Image, query: str) -> np.ndarray:
        # Per spec: resize to 352x352, run, sigmoid, cv2.resize INTER_AREA to 14x14.
        import cv2

        pil352 = pil_frame.resize((352, 352), Image.BICUBIC)
        inputs = self.proc(text=[query], images=[pil352], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        logits = out.logits  # (1, H, W) typically (1, 352, 352)
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        probs = torch.sigmoid(logits.float()).squeeze().cpu().numpy()
        if probs.ndim == 3:
            probs = probs[0]
        hm = cv2.resize(probs.astype(np.float32), (GRID, GRID),
                        interpolation=cv2.INTER_AREA)
        return hm


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    flat = scores.flatten()
    if k >= flat.size:
        return np.ones_like(flat, dtype=bool).reshape(scores.shape)
    idx = np.argpartition(-flat, k - 1)[:k]
    m = np.zeros(flat.size, dtype=bool)
    m[idx] = True
    return m.reshape(scores.shape)


def patch_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def time_phase24d_forward(ph24, clip_model, clip_tok, device,
                          n_warmup=10, n_trials=100):
    # Use a fixed canonical input.
    pil = Image.new("RGB", (224, 224), color=(127, 127, 127))
    query = "a photo of an object"
    for _ in range(n_warmup):
        _ = heatmap_phase24d(pil, query, ph24, clip_model, clip_tok, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_trials):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = heatmap_phase24d(pil, query, ph24, clip_model, clip_tok, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


# ---------------------------------------------------------------------------
# Qualitative panel
# ---------------------------------------------------------------------------

def render_panel(records, out_path):
    """records: list of dicts with keys
        frame (HxWx3 uint8), question, hm_phase, hm_auto, hm_clipseg,
        topk_phase (14x14 bool).
    """
    import matplotlib.pyplot as plt

    n = len(records)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 5, figsize=(15, 3 * n))
    if n == 1:
        axes = axes[None, :]

    def overlay(ax, frame, hm, title):
        ax.imshow(frame)
        H, W = frame.shape[:2]
        # upsample hm to frame size (nearest is fine for cheap visual)
        hm_up = np.kron(
            hm,
            np.ones((H // GRID + 1, W // GRID + 1)),
        )[:H, :W]
        # min-max stretch for visibility
        hm_min, hm_max = float(hm_up.min()), float(hm_up.max())
        if hm_max - hm_min > 1e-8:
            hm_norm = (hm_up - hm_min) / (hm_max - hm_min)
        else:
            hm_norm = np.zeros_like(hm_up)
        ax.imshow(hm_norm, alpha=0.5, cmap="hot", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9)

    for i, r in enumerate(records):
        axes[i, 0].imshow(r["frame"])
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        axes[i, 0].set_title("input", fontsize=9)
        axes[i, 0].set_ylabel(
            r["question"][:40] + ("..." if len(r["question"]) > 40 else ""),
            fontsize=7, rotation=0, ha="right", va="center", labelpad=40,
        )
        overlay(axes[i, 1], r["frame"], r["hm_phase"], "phase24d")
        overlay(axes[i, 2], r["frame"], r["hm_auto"], "autogaze")
        overlay(axes[i, 3], r["frame"], r["hm_clipseg"], "clipseg")
        # top-K phase24d mask
        axes[i, 4].imshow(r["frame"])
        H, W = r["frame"].shape[:2]
        m_up = np.kron(
            r["topk_phase"].astype(np.float32),
            np.ones((H // GRID + 1, W // GRID + 1)),
        )[:H, :W]
        axes[i, 4].imshow(m_up, alpha=0.55, cmap="cool", vmin=0, vmax=1)
        axes[i, 4].set_xticks([]); axes[i, 4].set_yticks([])
        axes[i, 4].set_title("top-K phase24d", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    notes = {"sources": [], "models": []}

    # ---- Build question pool ----
    print("[data] building question pool ...", flush=True)
    pool, src_notes = load_question_pool(args)
    notes["sources"].extend(src_notes)
    print(f"[data] pool size = {len(pool)}", flush=True)

    triples = sample_triples(pool, args.n_samples, args.seed)
    print(f"[data] sampled {len(triples)} (video, frame, question) triples", flush=True)

    # ---- Load models ----
    print(f"[model] phase24d <- {args.ckpt}", flush=True)
    ph24 = load_phase24d(args.ckpt, device)
    clip_model, clip_tok = load_clip_text(device)
    notes["models"].append(f"phase24d ckpt: {args.ckpt}")

    print("[model] vanilla AutoGaze (or fallback) ...", flush=True)
    autogaze_runner = AutoGazeRunner(device)
    notes["models"].append(f"autogaze_kind: {autogaze_runner.kind}")
    for n in autogaze_runner.notes:
        notes["models"].append(n)

    print("[model] CLIPSeg CIDAS/clipseg-rd64-refined ...", flush=True)
    clipseg_runner = CLIPSegRunner(device)
    notes["models"].append("clipseg: CIDAS/clipseg-rd64-refined")

    # ---- Latency benchmark (phase24d only) ----
    print("[bench] timing phase24d forward (warm 10, trials 100) ...", flush=True)
    mean_ms, std_ms = time_phase24d_forward(ph24, clip_model, clip_tok, device,
                                            n_warmup=10, n_trials=100)
    print(f"[bench] phase24d mean={mean_ms:.2f} ms  std={std_ms:.2f} ms",
          flush=True)

    # ---- Per-triple loop ----
    K = max(1, int(args.keep_ratio * N_PATCH))
    print(f"[eval] K = max(1, int({args.keep_ratio} * {N_PATCH})) = {K}",
          flush=True)

    rows = []
    panel_records = []
    rng_panel = random.Random(args.seed + 1)
    panel_idxs = set(rng_panel.sample(range(len(triples)), min(6, len(triples))))

    for i, tr in enumerate(triples):
        try:
            pil = extract_pil_frame(tr["video_path"], tr["frame_idx"])
            pil_rgb = pil.convert("RGB")

            hm_phase = heatmap_phase24d(
                pil_rgb, tr["question"], ph24, clip_model, clip_tok, device)
            hm_auto = autogaze_runner.heatmap(
                tr["video_path"], tr["frame_idx"], pil_rgb)
            hm_clipseg = clipseg_runner.heatmap(pil_rgb, tr["question"])

            top_phase = topk_mask(hm_phase, K)
            top_auto = topk_mask(hm_auto, K)

            iou = patch_iou(top_phase, top_auto)
            cs_total = float(hm_clipseg.sum())
            if cs_total > 1e-8:
                retention = float((hm_clipseg * top_phase).sum() / cs_total)
            else:
                retention = 0.0
            is_ocr = bool(OCR_PATTERN.search(tr["question"]))

            row = {
                "video": tr["video"],
                "frame_idx": tr["frame_idx"],
                "question": tr["question"],
                "is_ocr": int(is_ocr),
                "patch_iou": iou,
                "clipseg_retention": retention,
            }
            rows.append(row)
            print(f"  [{i+1:3d}/{len(triples)}] iou={iou:.3f} ret={retention:.3f} "
                  f"ocr={int(is_ocr)} | {tr['video']} f={tr['frame_idx']} | "
                  f"{tr['question'][:60]}",
                  flush=True)

            if i in panel_idxs:
                panel_records.append({
                    "frame": np.array(pil_rgb.resize((224, 224), Image.BICUBIC)),
                    "question": tr["question"],
                    "hm_phase": hm_phase,
                    "hm_auto": hm_auto,
                    "hm_clipseg": hm_clipseg,
                    "topk_phase": top_phase,
                })
        except Exception as e:
            print(f"  [{i+1}/{len(triples)}] FAILED on {tr['video']} "
                  f"f={tr['frame_idx']}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue

    if not rows:
        raise RuntimeError("No successful triples — see traceback above.")

    # ---- Aggregate ----
    def stats(subset):
        if not subset:
            return {"n": 0, "patch_iou_mean": None, "clipseg_retention_mean": None}
        return {
            "n": len(subset),
            "patch_iou_mean": float(np.mean([r["patch_iou"] for r in subset])),
            "patch_iou_std": float(np.std([r["patch_iou"] for r in subset])),
            "clipseg_retention_mean": float(
                np.mean([r["clipseg_retention"] for r in subset])),
            "clipseg_retention_std": float(
                np.std([r["clipseg_retention"] for r in subset])),
        }

    ocr_rows = [r for r in rows if r["is_ocr"]]
    rest_rows = [r for r in rows if not r["is_ocr"]]

    summary = {
        "n_triples_attempted": len(triples),
        "n_triples_succeeded": len(rows),
        "keep_ratio": args.keep_ratio,
        "K": K,
        "seed": args.seed,
        "ckpt": args.ckpt,
        "phase24d_latency_ms": {
            "mean": mean_ms,
            "std": std_ms,
            "warmup": 10,
            "trials": 100,
        },
        "overall": stats(rows),
        "ocr_heavy": stats(ocr_rows),
        "non_ocr": stats(rest_rows),
        "notes": notes,
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {summary_path}", flush=True)

    # ---- per-triple csv ----
    csv_path = out_dir / "per_triple.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["video", "frame_idx", "question", "is_ocr",
                        "patch_iou", "clipseg_retention"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[save] {csv_path}", flush=True)

    # ---- qual panel ----
    panel_path = out_dir / "qual_panel.png"
    try:
        render_panel(panel_records, panel_path)
        print(f"[save] {panel_path}", flush=True)
    except Exception as e:
        print(f"[warn] qual_panel failed: {type(e).__name__}: {e}", flush=True)

    print("\n=== TIER-1 SUMMARY ===")
    print(f"  n succeeded: {summary['n_triples_succeeded']}")
    print(f"  phase24d latency: {mean_ms:.2f} +/- {std_ms:.2f} ms")
    print(f"  overall   patch_iou={summary['overall']['patch_iou_mean']:.3f}  "
          f"clipseg_retention={summary['overall']['clipseg_retention_mean']:.3f}")
    if summary["ocr_heavy"]["n"]:
        print(f"  ocr-heavy patch_iou={summary['ocr_heavy']['patch_iou_mean']:.3f}  "
              f"clipseg_retention={summary['ocr_heavy']['clipseg_retention_mean']:.3f}  "
              f"(n={summary['ocr_heavy']['n']})")
    if summary["non_ocr"]["n"]:
        print(f"  non-ocr   patch_iou={summary['non_ocr']['patch_iou_mean']:.3f}  "
              f"clipseg_retention={summary['non_ocr']['clipseg_retention_mean']:.3f}  "
              f"(n={summary['non_ocr']['n']})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default="/home/ogata/semantic-autogaze/results/phase24d_atto_mpp05_aggrAug_10k/best_val.pt",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--keep_ratio", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--hlvid_dir",
        default="/home/ogata/semantic-autogaze/data/hlvid_videos",
    )
    p.add_argument(
        "--question_source",
        default="/home/ogata/semantic-autogaze/results/hlvid_household_expand/tier3_full_sweep/hlvid_subset.json",
    )
    args = p.parse_args()
    main(args)
