"""Pi webcam server — streams annotated heatmap-overlay frames via MJPEG.

Captures from a USB webcam (cv2.VideoCapture), runs the trained scorer +
multi-query late fusion, and serves the annotated stream over HTTP. Open
http://<pi-host>:8000/ in any browser on the same network (or Tailscale).

Usage on Pi (or anywhere with a webcam + CPU):
  source ~/.venv-pi-bench/bin/activate    # has torch+timm+open_clip already
  pip install flask opencv-python
  curl -L https://github.com/Clamepending/semantic-autogaze/releases/download/v0.1.0-demo/D_mobilenet_std_best.pt -o D_mobile.pt
  python pi_webcam_server.py --ckpt D_mobile.pt --model d-mobile \\
      --query "hand,coffee cup,laptop,face" --reduce max --port 8000

Then browse http://home-raspi.tail8dd042.ts.net:8000/

Endpoints:
  GET  /           HTML page with live MJPEG + query/reduce controls
  GET  /stream     multipart/x-mixed-replace MJPEG stream
  GET  /api/state  JSON: current query, reduce mode, fps, model
  POST /api/query  body: {"query": "hand,laptop"} OR raw text "hand,laptop"
  POST /api/reduce body: {"reduce": "max"} OR raw text "max"
"""
from __future__ import annotations
import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


GRID = 14

IM_MEAN = (0.485, 0.456, 0.406)
IM_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

REDUCE_MODES = ["max", "min", "mean", "sum", "softmax"]

RELEASE_URLS = {
    "d-mobile": "https://github.com/Clamepending/semantic-autogaze/releases/download/v0.1.0-demo/D_mobilenet_std_best.pt",
    "v2-tiny":  "https://github.com/Clamepending/semantic-autogaze/releases/download/v0.1.0-demo/v2_tiny_best.pt",
    "v1":       "https://github.com/Clamepending/semantic-autogaze/releases/download/v0.1.0-demo/v1_best.pt",
    # Phase 10 SigLIP-distilled (DINOv2-s teacher) ConvNeXt family — Pi-class
    # baseline released as v0.4.0 (atto mIoU 0.696, headline pre-Phase-15).
    "phase10-atto":  "https://github.com/Clamepending/semantic-autogaze/releases/download/v0.4.0-phase10-pi-demo/phase10_convnext_atto_best_val.pt",
    "phase10-femto": "https://github.com/Clamepending/semantic-autogaze/releases/download/v0.4.0-phase10-pi-demo/phase10_convnext_femto_best_val.pt",
    "phase10-pico":  "https://github.com/Clamepending/semantic-autogaze/releases/download/v0.4.0-phase10-pi-demo/phase10_convnext_pico_best_val.pt",
    # Phase 15 ConvNeXt-atto with the longer-trained DINOv2-s teacher
    # (phase13) and source-reweighted recipe — the first Pi-class scorer the
    # project has produced above the §1 >=0.71 mIoU deployment floor.
    "phase15-atto":  "https://github.com/Clamepending/semantic-autogaze/releases/download/v0.5.0-phase15-pi-demo/phase15_convnext_atto_step35000_iou0722.pt",
}

# Mapping from `ckpt['args']['model']` (training-time identifier) to the timm
# constructor name. Pi-class ConvNeXts are the Phase 10 deployment family.
PHASE10_TIMM_NAMES = {
    "convnext-atto":  "convnext_atto.d2_in1k",
    "convnext-femto": "convnext_femto.d1_in1k",
    "convnext-pico":  "convnext_pico.d1_in1k",
    "convnext-nano":  "convnext_nano.in12k_ft_in1k",
    "convnext-tiny":  "convnext_tiny.in12k_ft_in1k",
    "fastvit-t8":     "fastvit_t8.apple_in1k",
    "mobilevit-xs":   "mobilevit_xs.cvnets_in1k",
    "repvit-m1":      "repvit_m1.dist_in1k",
    "efficientformerv2-s0": "efficientformerv2_s0.snap_dist_in1k",
}


# ---- Inlined head + multi-query (verbatim from the Mac demo) ----
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


class MultiQueryScorer:
    def __init__(self, head):
        self.head = head; self.grid_size = head.grid_size

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
    def __call__(self, patches, text_embs, reduce="max", apply_sigmoid=True):
        x = self.encode_patches(patches)
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
        scores = scores.reshape(B, Q, N)
        # Apply SigLIP calibration if loaded from a Phase 2 ckpt.
        if hasattr(h, "_siglip_t") and (h._siglip_t != 1.0 or h._siglip_bias != 0.0):
            scores = scores * h._siglip_t + h._siglip_bias
        if apply_sigmoid: scores = torch.sigmoid(scores)
        if reduce == "max": return scores.amax(dim=1)
        if reduce == "min": return scores.amin(dim=1)
        if reduce == "mean": return scores.mean(dim=1)
        if reduce == "sum": return scores.sum(dim=1)
        if reduce == "softmax":
            w = scores.softmax(dim=1)
            return (w * scores).sum(dim=1)
        return scores  # 'none'


# ---- Inference plumbing ----
def maybe_download(model: str, ckpt_path: Path):
    if ckpt_path.exists(): return ckpt_path
    url = RELEASE_URLS.get(model)
    if not url: raise FileNotFoundError(f"No ckpt at {ckpt_path}")
    print(f"[download] {url}", flush=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(ckpt_path))
    return ckpt_path


def build_model(model_name: str, ckpt_path: Path, device):
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    ca = ck.get("args", {}) or {}
    # Trust the ckpt's recorded backbone identifier when present; otherwise
    # fall back to model_name from the CLI. Phase 3-10 ckpts written by
    # train_siglip_dense_distill.py store args.model = "convnext-atto" etc.
    arch = ca.get("model") if isinstance(ca, dict) else None
    if not arch:
        arch = model_name
    # Phase 3-10 ckpts don't write `embed_dim` to the top-level dict; derive
    # patch_dim from the head's first projection layer (always
    # `patch_proj.0.weight` of shape (hidden_dim, patch_dim)).
    derived_patch_dim = None
    if "head" in ck and "patch_proj.0.weight" in ck["head"]:
        derived_patch_dim = int(ck["head"]["patch_proj.0.weight"].shape[1])
    embed_dim_default = (derived_patch_dim if derived_patch_dim is not None
                        else ck.get("embed_dim", 768))
    if arch == "v1":
        embed_dim_default = 768
    head_kwargs = dict(
        patch_dim=embed_dim_default,
        text_dim=512,
        hidden_dim=ca.get("head_hidden_dim", 384),
        n_attn_heads=ca.get("head_attn_heads", 6),
        n_attn_layers=ca.get("head_attn_layers", 2),
        grid_size=GRID,
        use_spatial=ca.get("head_use_spatial", True),
    )
    head = TextScorerHead(**head_kwargs).to(device).eval()
    head.load_state_dict(ck["head"])
    # SigLIP-Phase-2/3-10 ckpts include a "sb" module: learnable t/bias that
    # calibrates the head's raw logits. Bake into head as scalar attributes.
    head._siglip_t = 1.0
    head._siglip_bias = 0.0
    if "sb" in ck:
        sb = ck["sb"]
        log_t = sb["log_t"].item() if hasattr(sb["log_t"], "item") else float(sb["log_t"])
        bias = sb["bias"].item() if hasattr(sb["bias"], "item") else float(sb["bias"])
        head._siglip_t = float(np.exp(log_t))
        head._siglip_bias = float(bias)
        print(f"[siglip] calibration t={head._siglip_t:.2f} bias={head._siglip_bias:.2f}", flush=True)
    if arch == "v1":
        return None, head, CLIP_MEAN, CLIP_STD, "clip-visual"
    import timm
    if arch in PHASE10_TIMM_NAMES:
        bb_name = PHASE10_TIMM_NAMES[arch]
    else:
        bb_name = ck.get("backbone", "")
        if not bb_name:
            bb_name = ("vit_tiny_patch16_224.augreg_in21k_ft_in1k" if arch == "v2-tiny"
                       else "mobilenetv3_small_100")
    bb = timm.create_model(bb_name, pretrained=True, num_classes=0).to(device).eval()
    print(f"[backbone] arch={arch} timm={bb_name}", flush=True)
    return bb, head, IM_MEAN, IM_STD, "timm"


def normalize_frame(frame_bgr: np.ndarray, mean, std, device, size=224):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    arr = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
    arr = F.interpolate(arr.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)
    mean_t = torch.tensor(mean, device=device)[None, :, None, None]
    std_t = torch.tensor(std, device=device)[None, :, None, None]
    return (arr - mean_t) / std_t


def adapt_features(feats):
    if feats.dim() == 4:
        feats = F.interpolate(feats, size=(GRID, GRID), mode="bilinear", align_corners=False)
        return feats.permute(0, 2, 3, 1).reshape(feats.shape[0], GRID * GRID, feats.shape[1])
    if feats.shape[1] == 197:
        return feats[:, 1:, :]
    return feats


def overlay_heatmap(frame_bgr, heatmap_14, threshold):
    """Overlay heatmap on frame using ABSOLUTE sigmoid scores.

    `heatmap_14` is the head's per-cell sigmoid output in [0, 1]. The same
    `threshold` value means the same thing across queries and across frames:
    cells with score >= threshold are highlighted, cells below are not.
    No per-frame normalization, no percentile cutoff. The visualization
    reads honestly: when the model abstains (max < threshold), nothing
    is drawn over the frame.

    Display intensity uses the raw cell score directly. A cell at score
    0.40 is dim; a cell at 0.95 is saturated red. Calibration is the
    model's responsibility — see paper.md §4.8 caveats and the phase18
    hard-negative-mining direction."""
    H, W = frame_bgr.shape[:2]
    h = heatmap_14.astype(np.float32)

    # If nothing exceeds threshold, return the frame unchanged. This is
    # the model abstaining; the visualization should show abstention,
    # not amplified noise.
    if float(h.max()) < threshold:
        return frame_bgr

    # Mask = cells at or above absolute threshold.
    keep_mask_14 = (h >= threshold).astype(np.float32)
    # Display intensity = the raw score itself (clipped to [0, 1] for
    # safety; the head outputs in this range already).
    h_disp = np.clip(h, 0.0, 1.0) * keep_mask_14

    heat = cv2.resize(h_disp, (W, H), interpolation=cv2.INTER_LINEAR)
    heat_uint = np.clip(heat * 255, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_uint, cv2.COLORMAP_HOT)

    keep_full = cv2.resize(keep_mask_14, (W, H), interpolation=cv2.INTER_NEAREST)
    keep_full = (keep_full > 0.5)[..., None]  # (H, W, 1) bool
    dim = (frame_bgr * 0.30).astype(np.uint8)
    bright = cv2.addWeighted(frame_bgr, 0.55, heat_bgr, 0.45, 0)
    return np.where(keep_full, bright, dim)


def rotate_frame(frame_bgr: np.ndarray, deg: int) -> np.ndarray:
    if deg == 90:
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame_bgr, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame_bgr


# ---- Shared state ----
# Two live UI knobs: comma-separated keywords, and an absolute sigmoid-score
# threshold in [0, 1]. Multi-keyword always combined with `max` (union).
# Rotation is a launch-time argument, not a UI control.
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.queries = ["hand"]
        self.threshold = 0.45   # absolute sigmoid-score cutoff
        self.rotate = 0         # 0/90/180/270 (launch-time only)
        self.fps = 0.0
        self.last_jpeg = None
        self.text_embs = None   # (Q, 512) tensor

    def set_query(self, q_str: str, encode_fn):
        qs = [q.strip() for q in q_str.split(",") if q.strip()]
        if not qs: return
        embs = encode_fn(qs)
        with self.lock:
            self.queries = qs
            self.text_embs = embs

    def set_threshold(self, t: float):
        t = max(0.0, min(1.0, float(t)))
        with self.lock:
            self.threshold = t


# ---- HTML page ----
INDEX_HTML = """<!doctype html>
<html><head><title>semantic-autogaze pi webcam</title>
<style>
body { font-family: -apple-system, sans-serif; background: #111; color: #eee; margin: 16px; }
img { max-width: 100%; height: auto; border: 1px solid #444; display: block; }
input, button { font-size: 16px; padding: 6px 10px; background: #222; color: #eee; border: 1px solid #444; }
input[type=text] { width: 60%; }
input[type=range] { width: 320px; vertical-align: middle; }
button { background: #2a4; color: #fff; border: none; cursor: pointer; }
button:hover { background: #3b5; }
.row { margin: 14px 0; }
.muted { color: #888; font-size: 13px; }
.thrval { display: inline-block; width: 60px; text-align: right; font-family: monospace; color: #cfc; }
</style></head><body>
<h2>semantic-autogaze — pi webcam stream</h2>
<div class="row">
  <label>keywords (comma-separated, combined with max): <input id="q" type="text" value=""></label>
  <button onclick="apply()">apply</button>
</div>
<div class="row">
  <label>threshold (absolute sigmoid score, same meaning across queries):
    <input id="thr" type="range" min="0.0" max="1.0" step="0.01" value="0.45">
    <span id="thrval" class="thrval">0.45</span>
  </label>
  <div class="muted">cells with predicted score &lt; threshold are not drawn. when the model abstains
    (max &lt; threshold), the frame is shown unmodified.</div>
</div>
<div class="row muted" id="status"></div>
<img id="stream" src="/stream" alt="camera">
<script>
let firstLoad = true;
async function refreshState() {
  const r = await fetch('/api/state'); const j = await r.json();
  if (firstLoad) {
    document.getElementById('q').value = j.queries.join(', ');
    document.getElementById('thr').value = j.threshold;
    document.getElementById('thrval').textContent = j.threshold.toFixed(2);
    firstLoad = false;
  }
  document.getElementById('status').textContent =
    `keywords=[${j.queries.join(', ')}]  threshold=${j.threshold.toFixed(2)}  model=${j.model}  ${j.fps.toFixed(1)} fps`;
}
async function apply() {
  const q = document.getElementById('q').value;
  await fetch('/api/query', {method:'POST', body: q});
  await refreshState();
}
async function postThr(v) {
  document.getElementById('thrval').textContent = parseFloat(v).toFixed(2);
  await fetch('/api/threshold', {method:'POST', body: v});
}
window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('q').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); apply(); }
  });
  document.getElementById('thr').addEventListener('input', (e) => postThr(e.target.value));
  for (const f of ['gbox', 'gtext', 'garea', 'ggate']) {
    const el = document.getElementById(f);
    if (el) el.addEventListener('input', (e) => postGsam(f, e.target.value));
  }
  refreshState();
  setInterval(refreshState, 1000);
});
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="D_mobile.pt")
    ap.add_argument("--model", default="phase15-atto",
                    choices=["d-mobile", "v2-tiny", "v1",
                             "phase10-atto", "phase10-femto", "phase10-pico",
                             "phase15-atto"])
    ap.add_argument("--query", default="hand")
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="initial absolute sigmoid-score threshold; UI slider can change live. "
                         "Same number means the same thing across queries.")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="frame rotation (launch-time only)")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--cam_w", type=int, default=0,
                    help="0 = let camera report native resolution (preserves aspect)")
    ap.add_argument("--cam_h", type=int, default=0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--jpeg_quality", type=int, default=70)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"[torch] threads={torch.get_num_threads()}", flush=True)

    device = torch.device("cpu")
    ckpt_path = maybe_download(args.model, Path(args.ckpt))
    print(f"[model] {args.model} from {ckpt_path}", flush=True)
    backbone, head, mean, std, kind = build_model(args.model, ckpt_path, device)
    mq = MultiQueryScorer(head)

    print("[clip] loading text encoder...", flush=True)
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()
    if kind == "clip-visual":
        clip_model.visual.output_tokens = True

    def encode_texts(qs):
        with torch.no_grad():
            toks = clip_tok(qs).to(device)
            return F.normalize(clip_model.encode_text(toks), dim=-1)

    state = State()
    state.set_query(args.query, encode_texts)
    state.threshold = args.threshold
    state.rotate = args.rotate

    print(f"[cam] opening index {args.cam} ...", flush=True)
    cap = cv2.VideoCapture(args.cam)
    if args.cam_w > 0: cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_w)
    if args.cam_h > 0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_h)
    if not cap.isOpened():
        raise RuntimeError(f"camera {args.cam} not openable")
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[cam] native frame size: {actual_w}x{actual_h}", flush=True)

    def worker():
        last_t = time.time(); fps_ema = 0.0
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                time.sleep(0.05); continue
            with state.lock:
                te = state.text_embs
                qs = list(state.queries)
                thr = state.threshold
                rot = state.rotate
            if rot:
                frame_bgr = rotate_frame(frame_bgr, rot)

            try:
                with torch.no_grad():
                    x = normalize_frame(frame_bgr, mean, std, device)
                    if kind == "clip-visual":
                        _, feats = clip_model.visual(x)
                    else:
                        feats = adapt_features(backbone.forward_features(x))
                    # Multi-keyword always combined with `max` (union of per-keyword heatmaps).
                    h14 = mq(feats, te, reduce="max", apply_sigmoid=True).reshape(GRID, GRID).cpu().numpy()
                disp = overlay_heatmap(frame_bgr, h14, threshold=thr)
            except Exception as e:
                disp = frame_bgr.copy()
                cv2.putText(disp, f"err: {e}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            now = time.time(); dt = now - last_t; last_t = now
            if dt > 0:
                inst = 1.0 / dt
                fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
            label = ", ".join(qs) if len(qs) > 1 else qs[0]
            cv2.putText(disp, f"q: {label}  [{args.model}]  thr={thr:.2f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(disp, f"{fps_ema:.1f} fps",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

            ok2, jpeg = cv2.imencode(".jpg", disp,
                                     [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            if ok2:
                with state.lock:
                    state.last_jpeg = jpeg.tobytes()
                    state.fps = fps_ema

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # ---- Flask app ----
    from flask import Flask, Response, request
    app = Flask(__name__)

    @app.get("/")
    def index():
        return INDEX_HTML

    @app.get("/stream")
    def stream():
        def gen():
            boundary = b"--frame"
            while True:
                with state.lock:
                    j = state.last_jpeg
                if j is None:
                    time.sleep(0.05); continue
                yield (boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                time.sleep(1.0 / 60)  # cap stream rate so we don't over-deliver
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/state")
    def api_state():
        with state.lock:
            return {
                "queries": state.queries,
                "threshold": state.threshold,
                "fps": state.fps,
                "model": args.model,
            }

    @app.post("/api/query")
    def api_query():
        body = request.get_data(as_text=True).strip()
        if body.startswith("{"):
            body = json.loads(body).get("query", "")
        if body:
            state.set_query(body, encode_texts)
        with state.lock: return {"queries": state.queries}

    @app.post("/api/threshold")
    def api_threshold():
        body = request.get_data(as_text=True).strip()
        if body.startswith("{"):
            body = json.loads(body).get("threshold", "0")
        try:
            state.set_threshold(float(body))
        except ValueError:
            pass
        with state.lock: return {"threshold": state.threshold}

    print(f"\n[server] listening on http://{args.host}:{args.port}/", flush=True)
    print(f"[server] open in browser: http://<pi-host>:{args.port}/  (or http://<pi-tailscale-name>:{args.port}/)", flush=True)
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
