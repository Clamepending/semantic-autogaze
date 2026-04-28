"""Real-time webcam heatmap demo for the semantic-autogaze scorers.

Streams the local webcam, runs a frozen backbone + the trained head
with one OR multiple text queries (late fusion of text emb), and renders
the reduced heatmap overlay live.

Multi-query: pass comma-separated queries (e.g. `--query "hand,coffee,laptop"`)
and a reduction (`--reduce max|min|mean|sum|softmax`). The text-INdependent
prefix of the head (patch_proj + pos_embed + 2× self_attn over patches,
~80% of head compute) runs ONCE per frame and is shared across all queries;
each extra query adds only ~5-15% wall time.

Designed for Apple Silicon Macs (PyTorch MPS), CUDA, or CPU (Pi-class).
D-MobileNet (default) ~1 GFLOPs/frame: ~15-30 fps on M1/M2/M3 Macs,
projected ~10-25 fps on Pi 4 Cortex-A72.

Usage on a Mac:
  pip install torch torchvision timm opencv-python open_clip_torch Pillow
  python demo/realtime_webcam_demo.py --query "hand"
  # Multi-query union (any of these lights up):
  python demo/realtime_webcam_demo.py --query "hand,coffee cup,laptop" --reduce max

Hotkeys (focus the cv2 window first):
  q     change queries (comma-separated, typed into terminal)
  r     cycle reduction: max -> min -> mean -> sum -> softmax
  space pause / resume
  m     cycle model (D-Mobile -> v2-Tiny -> v1)
  esc   quit

Default checkpoints are looked for relative to the repo root. Pass
--ckpt to override, or set --download_from_release to fetch them
from the GH release.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS = Path(__file__).resolve()
REPO = THIS.parent.parent

GRID = 14


# ----- Inlined head (verbatim from scripts/train_independent_scorer.py) -----
class TextScorerHead(nn.Module):
    def __init__(self, patch_dim=768, text_dim=512, hidden_dim=384,
                 n_attn_heads=6, n_attn_layers=2, grid_size=14, use_spatial=True):
        super().__init__()
        self.grid_size = grid_size
        self.use_spatial = use_spatial
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, grid_size * grid_size, hidden_dim) * 0.02)
        self.self_attn_layers = nn.ModuleList()
        for _ in range(n_attn_layers):
            self.self_attn_layers.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(hidden_dim, n_attn_heads, batch_first=True),
                "norm1": nn.LayerNorm(hidden_dim),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                ),
                "norm2": nn.LayerNorm(hidden_dim),
            }))
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_attn_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        if use_spatial:
            self.spatial = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.GELU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.GELU(),
                nn.Conv2d(64, 1, kernel_size=3, padding=1),
            )
        else:
            self.spatial = None

    def forward(self, patch_feats, text_emb):
        B = patch_feats.shape[0]
        G = self.grid_size
        x = self.patch_proj(patch_feats) + self.pos_embed
        for layer in self.self_attn_layers:
            r = x; x = layer["norm1"](x)
            xa, _ = layer["attn"](x, x, x); x = r + xa
            r = x; x = layer["norm2"](x); x = r + layer["ffn"](x)
        q = self.text_proj(text_emb).unsqueeze(1)
        cross_out, _ = self.cross_attn(x, q, q)
        x = self.cross_norm(x + cross_out)
        scores = self.score_mlp(x).squeeze(-1)
        if self.spatial is None:
            return scores
        grids = scores.reshape(B, 1, G, G)
        return (grids + self.spatial(grids)).reshape(B, G * G)


# ----- Inlined MultiQueryScorer (matches semantic_autogaze.multi_query_scorer) -----
class MultiQueryScorer:
    """Late-fusion wrapper: shared text-INdependent prefix + per-query tail."""
    def __init__(self, head):
        self.head = head
        self.grid_size = head.grid_size

    @torch.no_grad()
    def encode_patches(self, patch_feats):
        h = self.head
        x = h.patch_proj(patch_feats) + h.pos_embed
        for layer in h.self_attn_layers:
            r = x; x = layer["norm1"](x)
            xa, _ = layer["attn"](x, x, x); x = r + xa
            r = x; x = layer["norm2"](x); x = r + layer["ffn"](x)
        return x

    @torch.no_grad()
    def score_with_queries(self, x, text_embs):
        h = self.head
        B, N, H = x.shape
        Q = text_embs.shape[0]
        x_rep = x.unsqueeze(1).expand(B, Q, N, H).reshape(B * Q, N, H)
        text_rep = text_embs.unsqueeze(0).expand(B, Q, -1).reshape(B * Q, -1)
        q_proj = h.text_proj(text_rep).unsqueeze(1)
        cross_out, _ = h.cross_attn(x_rep, q_proj, q_proj)
        x_q = h.cross_norm(x_rep + cross_out)
        scores = h.score_mlp(x_q).squeeze(-1)
        if h.spatial is not None:
            G = self.grid_size
            grids = scores.reshape(B * Q, 1, G, G)
            scores = (grids + h.spatial(grids)).reshape(B * Q, N)
        return scores.reshape(B, Q, N)

    @torch.no_grad()
    def __call__(self, patches, text_embs, reduce="max", apply_sigmoid=True):
        x = self.encode_patches(patches)
        scores = self.score_with_queries(x, text_embs)
        if apply_sigmoid: scores = torch.sigmoid(scores)
        if reduce == "none": return scores
        if reduce == "max": return scores.amax(dim=1)
        if reduce == "min": return scores.amin(dim=1)
        if reduce == "mean": return scores.mean(dim=1)
        if reduce == "sum": return scores.sum(dim=1)
        if reduce == "softmax":
            w = scores.softmax(dim=1)
            return (w * scores).sum(dim=1)
        raise ValueError(f"unknown reduce: {reduce}")

# ImageNet stats (used by timm backbones); CLIP uses its own.
IM_MEAN = (0.485, 0.456, 0.406)
IM_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

DEFAULT_CKPTS = {
    "d-mobile": REPO / "results" / "sweep_v3" / "D_mobilenet_std" / "best.pt",
    "v2-tiny": REPO / "results" / "independent_scorer_v2_tiny" / "best.pt",
    "v1": REPO / "results" / "independent_scorer" / "best_v1.pt",
}

# Public release URLs (filled in once the GH release is published).
RELEASE_URLS = {
    "d-mobile": "https://github.com/Clamepending/semantic-autogaze/releases/download/v0.1.0-demo/D_mobilenet_std_best.pt",
    "v2-tiny":  "https://github.com/Clamepending/semantic-autogaze/releases/download/v0.1.0-demo/v2_tiny_best.pt",
    "v1":       "https://github.com/Clamepending/semantic-autogaze/releases/download/v0.1.0-demo/v1_best.pt",
}


def pick_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps"), "mps (Apple Silicon)"
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def maybe_download(model: str, ckpt_path: Path, force_release: bool):
    """Download checkpoint from GH release if missing locally."""
    if ckpt_path.exists() and not force_release:
        return ckpt_path
    url = RELEASE_URLS.get(model)
    if not url:
        raise FileNotFoundError(f"No checkpoint at {ckpt_path} and no release URL for {model}")
    print(f"[download] {url}")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(ckpt_path)
    print(f"[download] saved {ckpt_path}")
    return ckpt_path


def build_clip_text_encoder(device):
    """Returns (encode_text_fn, tokenizer)."""
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()
    clip_model.visual.output_tokens = True  # used by v1
    return clip_model, clip_tok


def load_head(ckpt_path: Path, device, fixed_patch_dim=None):
    ck = torch.load(str(ckpt_path), map_location=device)
    ca = ck.get("args", {}) or {}
    head = TextScorerHead(
        patch_dim=fixed_patch_dim or ck.get("embed_dim", 768),
        text_dim=512,
        hidden_dim=ca.get("head_hidden_dim", 384),
        n_attn_heads=ca.get("head_attn_heads", 6),
        n_attn_layers=ca.get("head_attn_layers", 2),
        grid_size=GRID,
        use_spatial=ca.get("head_use_spatial", True),
    ).to(device).eval()
    head.load_state_dict(ck["head"])
    return ck, head


def build_model(model_name: str, ckpt_path: Path, device):
    """Returns (backbone, head, mean, std, kind)."""
    if model_name == "v1":
        ck, head = load_head(ckpt_path, device, fixed_patch_dim=768)
        return None, head, CLIP_MEAN, CLIP_STD, "clip-visual"  # backbone = clip_model.visual
    elif model_name == "v2-tiny":
        import timm
        ck, head = load_head(ckpt_path, device)
        bb = timm.create_model(ck["backbone"], pretrained=True, num_classes=0).to(device).eval()
        return bb, head, IM_MEAN, IM_STD, "timm"
    elif model_name == "d-mobile":
        import timm
        ck, head = load_head(ckpt_path, device)
        bb = timm.create_model(ck["backbone"], pretrained=True, num_classes=0).to(device).eval()
        return bb, head, IM_MEAN, IM_STD, "timm"
    else:
        raise ValueError(f"unknown model {model_name}")


def normalize_frame(frame_bgr: np.ndarray, mean, std, device, size=224):
    """BGR uint8 -> (1, 3, size, size) float, normalized."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    arr = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().to(device) / 255.0
    arr = F.interpolate(arr.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)
    mean_t = torch.tensor(mean, device=device)[None, :, None, None]
    std_t = torch.tensor(std, device=device)[None, :, None, None]
    return (arr - mean_t) / std_t


def adapt_features(feats):
    """Normalize backbone output to (B, 196, C)."""
    if feats.dim() == 4:
        feats = F.interpolate(feats, size=(GRID, GRID), mode="bilinear", align_corners=False)
        return feats.permute(0, 2, 3, 1).reshape(feats.shape[0], GRID * GRID, feats.shape[1])
    if feats.shape[1] == 197:
        return feats[:, 1:, :]
    return feats


def overlay_heatmap(frame_bgr, heatmap_14):
    """Upsample 14×14 heatmap to frame size and blend with hot colormap."""
    H, W = frame_bgr.shape[:2]
    heat = cv2.resize(heatmap_14.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    heat_uint = np.clip(heat * 255, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_uint, cv2.COLORMAP_HOT)
    return cv2.addWeighted(frame_bgr, 0.55, heat_bgr, 0.45, 0)


REDUCE_MODES = ["max", "min", "mean", "sum", "softmax"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="hand",
                    help="initial text query (or comma-separated list, e.g. 'hand,coffee,laptop')")
    ap.add_argument("--reduce", default="max", choices=REDUCE_MODES,
                    help="how to combine per-query heatmaps when --query is a list")
    ap.add_argument("--model", default="d-mobile", choices=["d-mobile", "v2-tiny", "v1"],
                    help="which trained scorer to run")
    ap.add_argument("--ckpt", default=None, help="override default ckpt path")
    ap.add_argument("--download_from_release", action="store_true",
                    help="force-download the checkpoint from GH release even if a local copy exists")
    ap.add_argument("--cam", type=int, default=0, help="OpenCV camera index (0 = facetime camera on most Macs)")
    ap.add_argument("--cam_w", type=int, default=640)
    ap.add_argument("--cam_h", type=int, default=480)
    ap.add_argument("--device", default=None, help="override device autodetect")
    args = ap.parse_args()

    # ---- device + checkpoint ----
    if args.device:
        device = torch.device(args.device); device_label = args.device
    else:
        device, device_label = pick_device()
    print(f"[device] {device_label}")

    ckpt_path = Path(args.ckpt) if args.ckpt else DEFAULT_CKPTS[args.model]
    ckpt_path = maybe_download(args.model, Path(ckpt_path), args.download_from_release)

    # ---- model ----
    print(f"[model] loading {args.model} from {ckpt_path}", flush=True)
    backbone, head, mean, std, kind = build_model(args.model, ckpt_path, device)
    print(f"[model] kind={kind}", flush=True)

    # ---- multi-query wrapper ----
    mq = MultiQueryScorer(head)

    print("[clip] loading CLIP-B/16 text encoder ...", flush=True)
    clip_model, clip_tok = build_clip_text_encoder(device)

    def encode_texts(query_str):
        """Comma-separated string -> (Q, 512) text embeddings."""
        queries = [q.strip() for q in query_str.split(",") if q.strip()]
        if not queries: queries = [query_str]
        with torch.no_grad():
            toks = clip_tok(queries).to(device)
            embs = F.normalize(clip_model.encode_text(toks), dim=-1)
        return queries, embs  # (Q, 512)

    queries, text_embs = encode_texts(args.query)
    current_query = args.query
    reduce_mode = args.reduce
    print(f"[query] {len(queries)} querie(s): {queries} | reduce={reduce_mode}")

    # ---- camera ----
    print(f"[cam] opening camera index {args.cam} ...", flush=True)
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_h)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {args.cam}")
    print("[cam] press: q (new query), space (pause), m (cycle model), esc (quit)")

    paused = False
    last_t = time.time(); fps_ema = 0.0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("[cam] frame grab failed; stopping"); break
        if frame_bgr is None or frame_bgr.size == 0:
            continue
        if not paused:
            with torch.no_grad():
                x = normalize_frame(frame_bgr, mean, std, device)
                if kind == "clip-visual":
                    _, feats = clip_model.visual(x)  # (1, 196, 768)
                else:
                    feats = adapt_features(backbone.forward_features(x))
                heat = mq(feats, text_embs, reduce=reduce_mode, apply_sigmoid=True)  # (1, 196)
                heat14 = heat.reshape(GRID, GRID).cpu().numpy()
            display = overlay_heatmap(frame_bgr, heat14)
        else:
            display = frame_bgr

        now = time.time(); dt = now - last_t; last_t = now
        if dt > 0:
            inst = 1.0 / dt
            fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
        q_label = current_query if len(queries) == 1 else f"{current_query}  [reduce={reduce_mode}]"
        cv2.putText(display, f"query: {q_label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, f"model: {args.model} | device: {device_label} | fps: {fps_ema:.1f} | n_queries: {len(queries)}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        if paused:
            cv2.putText(display, "PAUSED", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("semantic-autogaze webcam demo", display)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord("q"):
            print("[query] enter new queries (comma-separated, in terminal): ", end="", flush=True)
            try:
                new_q = input().strip()
                if new_q:
                    current_query = new_q
                    queries, text_embs = encode_texts(new_q)
                    print(f"[query] -> {len(queries)} querie(s): {queries}")
            except EOFError:
                pass
        elif k == ord("r"):
            reduce_mode = REDUCE_MODES[(REDUCE_MODES.index(reduce_mode) + 1) % len(REDUCE_MODES)]
            print(f"[reduce] -> {reduce_mode}")
        elif k == ord(" "):
            paused = not paused
        elif k == ord("m"):
            order = ["d-mobile", "v2-tiny", "v1"]
            args.model = order[(order.index(args.model) + 1) % len(order)]
            new_ck = DEFAULT_CKPTS[args.model]
            new_ck = maybe_download(args.model, Path(new_ck), False)
            print(f"[model] -> {args.model} ({new_ck})", flush=True)
            backbone, head, mean, std, kind = build_model(args.model, Path(new_ck), device)
            mq = MultiQueryScorer(head)

    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
