"""Real-time webcam heatmap demo for the semantic-autogaze scorers.

Streams the local webcam (or FaceTime camera on Mac), runs a frozen
backbone + the trained head, and renders a per-patch heatmap overlay
showing where the scorer thinks the queried subject is.

Designed for Apple Silicon Macs (PyTorch MPS), but also runs on CUDA
or CPU. D-MobileNet (default) is ~1 GFLOPs / frame and should hit
~15-30 fps on M1/M2/M3 Macs.

Usage on a Mac:
  pip install torch torchvision timm opencv-python open_clip_torch Pillow
  python demo/realtime_webcam_demo.py --query "hand"

Hotkeys (focus the cv2 window first):
  q     change query (type into terminal)
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
import torch.nn.functional as F

THIS = Path(__file__).resolve()
REPO = THIS.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from train_independent_scorer import TextScorerHead, GRID

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="hand", help="initial text query for the scorer")
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

    print("[clip] loading CLIP-B/16 text encoder ...", flush=True)
    clip_model, clip_tok = build_clip_text_encoder(device)

    def encode_text(text):
        with torch.no_grad():
            toks = clip_tok([text]).to(device)
            emb = F.normalize(clip_model.encode_text(toks), dim=-1)
        return emb  # (1, 512)

    text_emb = encode_text(args.query)
    current_query = args.query

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
                    _, patches = clip_model.visual(x)  # (1, 196, 768)
                    feats = patches
                else:
                    feats = adapt_features(backbone.forward_features(x))
                scores = head(feats, text_emb)  # (1, 196)
                heat14 = torch.sigmoid(scores).reshape(GRID, GRID).cpu().numpy()
            display = overlay_heatmap(frame_bgr, heat14)
        else:
            display = frame_bgr

        now = time.time(); dt = now - last_t; last_t = now
        if dt > 0:
            inst = 1.0 / dt
            fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
        cv2.putText(display, f"query: {current_query}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, f"model: {args.model} | device: {device_label} | fps: {fps_ema:.1f}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        if paused:
            cv2.putText(display, "PAUSED", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("semantic-autogaze webcam demo", display)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord("q"):
            print("[query] enter new query (in terminal): ", end="", flush=True)
            try:
                new_q = input().strip()
                if new_q:
                    current_query = new_q
                    text_emb = encode_text(new_q)
                    print(f"[query] -> {new_q}")
            except EOFError:
                pass
        elif k == ord(" "):
            paused = not paused
        elif k == ord("m"):
            order = ["d-mobile", "v2-tiny", "v1"]
            args.model = order[(order.index(args.model) + 1) % len(order)]
            new_ck = DEFAULT_CKPTS[args.model]
            new_ck = maybe_download(args.model, Path(new_ck), False)
            print(f"[model] -> {args.model} ({new_ck})", flush=True)
            backbone, head, mean, std, kind = build_model(args.model, Path(new_ck), device)

    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
