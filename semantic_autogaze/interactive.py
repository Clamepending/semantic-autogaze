"""
Interactive Gradio app for Semantic AutoGaze.

- Click on any patch in a video frame to see GT + predicted similarity heatmaps
- Type a text query (e.g. "right hand") to see predicted + GT text similarity heatmaps
"""

import os
import glob
import hashlib
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import open_clip

from autogaze.models.autogaze import AutoGaze
from semantic_autogaze.model import SemanticAutoGaze
from semantic_autogaze.data import read_video_frames


# ── Global state ──────────────────────────────────────────────────────────────

STATE = {
    "model": None,
    "device": None,
    "grid_size": None,
    "hidden_cache_dir": None,
    # Per-video cached data
    "frames": None,       # (T, H, W, 3) uint8
    "hidden": None,       # (1, T*N, hidden_dim) on device
}


def load_models(device="cuda:0", head_path=None):
    device = torch.device(device)

    print("Loading AutoGaze...")
    autogaze = AutoGaze.from_pretrained("nvidia/AutoGaze", use_flash_attn=False).to(device).eval()
    model = SemanticAutoGaze(autogaze, embedding_dim=512).to(device)
    hidden_dim = autogaze.config.gaze_model_config.gaze_decoder_config.hidden_size

    # Prefer the best BigHead warm-restart checkpoint; fall back to baseline, then old head.
    candidates = [
        head_path,
        os.path.join("results", "bighead_warmrestart", "best_bighead_student.pt"),
        os.path.join("results", "distill_bighead", "best_bighead_student.pt"),
    ]
    loaded = False
    for ckpt in candidates:
        if ckpt and os.path.exists(ckpt):
            from semantic_autogaze.train_bighead import BigSimilarityHead
            model.similarity_head = BigSimilarityHead(
                hidden_dim=hidden_dim, embedding_dim=512, expanded_dim=384,
                n_attn_heads=6, n_attn_layers=2, grid_size=14,
            ).to(device)
            model.similarity_head.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"Loaded BigHead from {ckpt}")
            loaded = True
            break
    if not loaded:
        from semantic_autogaze.model import SimilarityHead
        model.similarity_head = SimilarityHead(hidden_dim, 512, grid_size=14,
                                               num_frames=16, use_spatial=True).to(device)
        fallback = os.path.join("results", "clipseg", "best_similarity_head.pt")
        if os.path.exists(fallback):
            model.similarity_head.load_state_dict(torch.load(fallback, map_location=device))
            print(f"Loaded legacy head from {fallback}")
    model.eval()

    STATE["model"] = model
    STATE["device"] = device
    STATE["grid_size"] = model.patch_grid_size
    print("Models loaded.")


def get_text_embedding(text):
    """Get CLIP ViT-B/16 text embedding (512-dim). Loads model temporarily."""
    device = STATE["device"]
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai", device=device,
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip_model.eval()
    tokens = clip_tokenizer([text]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
    result = text_features.squeeze(0)  # (512,) on device
    del clip_model
    torch.cuda.empty_cache()
    return result


def get_clipseg_heatmaps(frames, text):
    """Run CLIPSeg on each frame and return per-frame heatmaps."""
    device = STATE["device"]
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device).eval()

    heatmaps = []
    for frame in frames:
        pil_img = Image.fromarray(frame)
        inputs = processor(text=[text], images=[pil_img], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clipseg_model(**inputs)
        # outputs.logits: (1, H', W') — raw logits, higher = more likely
        logits = outputs.logits[0].cpu()  # (H', W')
        # Resize to frame size
        logits_resized = F.interpolate(
            logits.unsqueeze(0).unsqueeze(0), size=(frame.shape[0], frame.shape[1]),
            mode="bilinear", align_corners=False,
        )[0, 0]  # (H, W)
        heatmaps.append(logits_resized)

    del clipseg_model, processor
    torch.cuda.empty_cache()
    return heatmaps  # list of (H, W) tensors


def get_openclip_dense_heatmaps(frames, text, grid_size=14):
    """
    Get dense per-patch text similarity using OpenCLIP ViT-B/16.

    Extracts patch tokens from the last transformer layer, projects them
    into CLIP's text embedding space, and computes cosine similarity.
    This is the core inference approach of PixelCLIP (without the fine-tuning).
    """
    device = STATE["device"]
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai", device=device,
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip_model.eval()

    # Text embedding
    text_tokens = clip_tokenizer([text]).to(device)
    with torch.no_grad():
        text_feat = clip_model.encode_text(text_tokens)
        text_feat = F.normalize(text_feat, dim=-1)  # (1, 512)

    heatmaps = []
    for frame in frames:
        pil_img = Image.fromarray(frame)
        img_tensor = clip_preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            result = clip_model.visual.forward_intermediates(img_tensor)
            dense_feat = result["image_intermediates"][-1]  # (1, 768, H', W')
            B, C, H, W = dense_feat.shape
            dense_flat = dense_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
            dense_normed = clip_model.visual.ln_post(dense_flat)
            if clip_model.visual.proj is not None:
                dense_proj = dense_normed @ clip_model.visual.proj  # (1, H*W, 512)
            dense_proj = F.normalize(dense_proj, dim=-1)

            # Per-patch cosine similarity
            sims = (dense_proj @ text_feat.T).squeeze(-1)  # (1, H*W)
            sim_map = sims[0].reshape(H, W)  # (14, 14) typically

            # Resize to frame size
            sim_resized = F.interpolate(
                sim_map.unsqueeze(0).unsqueeze(0),
                size=(frame.shape[0], frame.shape[1]),
                mode="bilinear", align_corners=False,
            )[0, 0].cpu()
            heatmaps.append(sim_resized)

    del clip_model
    torch.cuda.empty_cache()
    return heatmaps  # list of (H, W) tensors


def load_video(video_path, num_frames=16):
    """Load a video and fetch AutoGaze hidden states (cached when possible)."""
    import time
    device = STATE["device"]
    model = STATE["model"]

    print(f"[load_video] {os.path.basename(video_path)} num_frames={num_frames}", flush=True)
    t0 = time.time()
    frames = read_video_frames(video_path, num_frames, 224)
    t1 = time.time()
    print(f"  read_video_frames: {(t1-t0)*1000:.0f}ms", flush=True)
    if frames is None:
        return None

    # Try cache first — keyed by md5 of the relative "data/NAME.mp4" path used during training.
    hidden = None
    cache_dir = STATE.get("hidden_cache_dir")
    if cache_dir and num_frames == 16:
        rel = video_path
        if os.path.isabs(rel):
            try:
                rel = os.path.relpath(video_path, os.getcwd())
            except ValueError:
                rel = video_path
        for candidate in {rel, f"data/{os.path.basename(video_path)}"}:
            key = hashlib.md5(candidate.encode()).hexdigest()
            cache_path = os.path.join(cache_dir, f"{key}_hidden.pt")
            if os.path.exists(cache_path):
                h = torch.load(cache_path, map_location=device, weights_only=True)
                hidden = h.unsqueeze(0) if h.dim() == 2 else h
                print(f"  [cache hit] {os.path.basename(video_path)} "
                      f"({(time.time()-t1)*1000:.0f}ms)", flush=True)
                break

    if hidden is None:
        print(f"  [cache miss — running AutoGaze] {os.path.basename(video_path)}", flush=True)
        video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        video_tensor = video_tensor.unsqueeze(0).to(device)
        with torch.inference_mode():
            hidden = model.get_patch_hidden_states(video_tensor)  # (1, T*N, hidden_dim)
        print(f"  AutoGaze forward: {(time.time()-t1)*1000:.0f}ms", flush=True)

    STATE["frames"] = frames
    STATE["hidden"] = hidden
    print(f"  total load: {(time.time()-t0)*1000:.0f}ms", flush=True)

    return frames


def make_frame_selector_image(frames, frame_idx):
    """Return the selected frame as a PIL image with grid overlay."""
    frame = frames[frame_idx].copy()
    grid_size = STATE["grid_size"]
    H, W = frame.shape[:2]
    ph, pw = H // grid_size, W // grid_size

    for r in range(1, grid_size):
        frame[r * ph, :] = frame[r * ph, :] // 2 + 64
    for c in range(1, grid_size):
        frame[:, c * pw] = frame[:, c * pw] // 2 + 64

    return Image.fromarray(frame)


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _render_heatmap_row(axes, col_idx, frame, sims_t, vmin, vmax, grid_size, H, W):
    """Render one heatmap cell."""
    hmap = sims_t.reshape(grid_size, grid_size).numpy()
    hmap_norm = np.clip((hmap - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    hmap_up = np.array(
        Image.fromarray((hmap_norm * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    colored = (plt.cm.jet(hmap_up)[:, :, :3] * 255).astype(np.uint8)
    blended = (frame.astype(np.float32) * 0.45 + colored.astype(np.float32) * 0.55).astype(np.uint8)
    axes.imshow(blended)
    axes.set_title(f"[{hmap.min():.2f}, {hmap.max():.2f}]", fontsize=8)
    axes.axis("off")


def render_patch_result(frame_idx, patch_idx):
    """Render patch query visualization: 3 rows (original, predicted, GT)."""
    frames = STATE["frames"]
    grid_size = STATE["grid_size"]
    if frames is None:
        return None

    T = frames.shape[0]
    N = grid_size * grid_size
    H, W = frames.shape[1], frames.shape[2]
    model = STATE["model"]
    device = STATE["device"]
    patch_emb = STATE["patch_emb"]
    hidden = STATE["hidden"]

    query_emb = patch_emb[frame_idx, patch_idx]  # (D,)

    # GT similarities
    all_patches = patch_emb.reshape(T * N, -1)
    gt_sims = torch.mv(all_patches, query_emb).cpu()

    # Predicted similarities
    with torch.inference_mode():
        pred_sims = model.similarity_head(hidden, query_emb.unsqueeze(0).to(device))
    pred_sims = pred_sims[0].cpu()

    gt_vmin, gt_vmax = gt_sims.min().item(), gt_sims.max().item()
    pred_vmin, pred_vmax = pred_sims.min().item(), pred_sims.max().item()

    n_cols = min(T, 8)
    sample_indices = np.linspace(0, T - 1, n_cols, dtype=int)

    fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 9.5))
    if n_cols == 1:
        axes = axes[:, None]

    for col_idx, t in enumerate(sample_indices):
        frame = frames[t]

        # Row 0: Original
        display = frame.copy()
        if t == frame_idx:
            ph, pw = H // grid_size, W // grid_size
            r, c = patch_idx // grid_size, patch_idx % grid_size
            y1, y2 = r * ph, (r + 1) * ph
            x1, x2 = c * pw, (c + 1) * pw
            for thick in range(3):
                display[y1+thick, x1:x2] = [0, 255, 0]
                display[y2-1-thick, x1:x2] = [0, 255, 0]
                display[y1:y2, x1+thick] = [0, 255, 0]
                display[y1:y2, x2-1-thick] = [0, 255, 0]

        axes[0, col_idx].imshow(display)
        title = f"t={t}"
        if t == frame_idx:
            title += " (query)"
        axes[0, col_idx].set_title(title, fontsize=9)
        axes[0, col_idx].axis("off")

        # Row 1: Predicted
        pred_t = pred_sims[t * N:(t + 1) * N]
        _render_heatmap_row(axes[1, col_idx], col_idx, frame, pred_t, pred_vmin, pred_vmax, grid_size, H, W)

        # Row 2: GT
        gt_t = gt_sims[t * N:(t + 1) * N]
        _render_heatmap_row(axes[2, col_idx], col_idx, frame, gt_t, gt_vmin, gt_vmax, grid_size, H, W)

    plt.tight_layout(rect=[0.12, 0.06, 1, 0.97])
    row_labels = ["Original", "Predicted", "GT (SigLIP)"]
    for row, label in enumerate(row_labels):
        bbox = axes[row, 0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(0.01, y_center, label, fontsize=11, fontweight="bold",
                 ha="left", va="center")

    fig.subplots_adjust(bottom=0.08, left=0.12)
    sm_pred = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=pred_vmin, vmax=pred_vmax))
    fig.colorbar(sm_pred, cax=fig.add_axes([0.15, 0.03, 0.3, 0.015]),
                 orientation="horizontal", label="Predicted similarity")
    sm_gt = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=gt_vmin, vmax=gt_vmax))
    fig.colorbar(sm_gt, cax=fig.add_axes([0.55, 0.03, 0.3, 0.015]),
                 orientation="horizontal", label="GT similarity")

    qr, qc = patch_idx // grid_size, patch_idx % grid_size
    plt.suptitle(
        f"Patch Query: frame {frame_idx}, patch {patch_idx} (row {qr}, col {qc})\n"
        f"Pred: [{pred_vmin:.3f}, {pred_vmax:.3f}]   GT: [{gt_vmin:.3f}, {gt_vmax:.3f}]",
        fontsize=13, y=1.01,
    )

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return Image.fromarray(img)


def _render_pixelwise_heatmap(ax, frame, heatmap_2d, vmin, vmax, H, W):
    """Render a pixel-level heatmap (e.g. CLIPSeg) blended onto a frame."""
    hmap = heatmap_2d.numpy()
    hmap_norm = np.clip((hmap - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    colored = (plt.cm.jet(hmap_norm)[:, :, :3] * 255).astype(np.uint8)
    blended = (frame.astype(np.float32) * 0.45 + colored.astype(np.float32) * 0.55).astype(np.uint8)
    ax.imshow(blended)
    ax.set_title(f"[{hmap.min():.2f}, {hmap.max():.2f}]", fontsize=8)
    ax.axis("off")


def render_text_result(text):
    """Render text query visualization: 3 rows (original, CLIPSeg GT, predicted)."""
    frames = STATE["frames"]
    grid_size = STATE["grid_size"]
    if frames is None or not text.strip():
        return None

    T = frames.shape[0]
    N = grid_size * grid_size
    H, W = frames.shape[1], frames.shape[2]
    model = STATE["model"]
    device = STATE["device"]
    hidden = STATE["hidden"]
    text_str = text.strip()

    # Get CLIP text embedding (512-dim)
    text_emb = get_text_embedding(text_str)  # (512,) on device

    # Predicted: run similarity head with CLIP text embedding
    with torch.inference_mode():
        pred_sims = model.similarity_head(hidden, text_emb.unsqueeze(0).to(device))
    pred_sims = pred_sims[0].cpu()

    # Sample frames
    n_cols = min(T, 8)
    sample_indices = np.linspace(0, T - 1, n_cols, dtype=int)
    sampled_frames = frames[sample_indices]

    # Free GPU memory before loading CLIPSeg
    hidden_cpu = hidden.cpu()
    model.cpu()
    torch.cuda.empty_cache()

    # CLIPSeg: pixel-level segmentation heatmaps (ground truth)
    clipseg_heatmaps = get_clipseg_heatmaps(sampled_frames, text_str)

    # Restore models to GPU
    model.to(device)
    STATE["hidden"] = hidden_cpu.to(device)

    pred_vmin, pred_vmax = pred_sims.min().item(), pred_sims.max().item()
    cs_all = torch.stack(clipseg_heatmaps)
    cs_vmin, cs_vmax = cs_all.min().item(), cs_all.max().item()

    fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 9.5))
    if n_cols == 1:
        axes = axes[:, None]

    for col_idx, t in enumerate(sample_indices):
        frame = frames[t]

        # Row 0: Original
        axes[0, col_idx].imshow(frame)
        axes[0, col_idx].set_title(f"t={t}", fontsize=9)
        axes[0, col_idx].axis("off")

        # Row 1: CLIPSeg (pixel-level GT)
        _render_pixelwise_heatmap(axes[1, col_idx], frame, clipseg_heatmaps[col_idx],
                                  cs_vmin, cs_vmax, H, W)

        # Row 2: Predicted (from our CLIPSeg-trained head)
        pred_t = pred_sims[t * N:(t + 1) * N]
        _render_heatmap_row(axes[2, col_idx], col_idx, frame, pred_t, pred_vmin, pred_vmax, grid_size, H, W)

    plt.tight_layout(rect=[0.12, 0.06, 1, 0.97])
    row_labels = ["Original", "CLIPSeg\n(GT)", "Predicted\n(ours)"]
    for row, label in enumerate(row_labels):
        bbox = axes[row, 0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(0.01, y_center, label, fontsize=11, fontweight="bold",
                 ha="left", va="center")

    fig.subplots_adjust(bottom=0.08, left=0.12)
    sm_cs = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=cs_vmin, vmax=cs_vmax))
    fig.colorbar(sm_cs, cax=fig.add_axes([0.15, 0.03, 0.3, 0.015]),
                 orientation="horizontal", label="CLIPSeg GT")
    sm_pred = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=pred_vmin, vmax=pred_vmax))
    fig.colorbar(sm_pred, cax=fig.add_axes([0.55, 0.03, 0.3, 0.015]),
                 orientation="horizontal", label="Predicted")

    plt.suptitle(
        f"Text Query: \"{text_str}\"",
        fontsize=14, y=1.01,
    )

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return Image.fromarray(img)


# ── Gradio callbacks ─────────────────────────────────────────────────────────

def on_video_select(video_path, num_frames):
    if not video_path:
        return None
    frames = load_video(video_path, int(num_frames))
    if frames is None:
        return None
    return Image.fromarray(frames[0])


def on_text_query(text):
    if STATE["frames"] is None:
        return None
    return render_text_result(text)


# ── Build UI ──────────────────────────────────────────────────────────────────

def build_app(video_dir="data", val_only=False, clipseg_dir="results/distill/clipseg_cache",
              seed=42, val_frac=0.1):
    video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if val_only:
        # Reproduce the 90/10 split used during training (same shuffle + seed as DistillDataset).
        cs_files = sorted(glob.glob(os.path.join(clipseg_dir, "*_clipseg_clip*.pt")))
        rng = random.Random(seed)
        shuffled = list(cs_files)
        rng.shuffle(shuffled)
        split = int((1 - val_frac) * len(shuffled))
        val_cs = shuffled[split:]
        val_basenames = set()
        for cf in val_cs:
            d = torch.load(cf, map_location="cpu", weights_only=False)
            val_basenames.add(os.path.basename(d["video_path"]))
        video_files = [v for v in video_files if os.path.basename(v) in val_basenames]
        print(f"val_only: {len(video_files)} / {len(val_basenames)} videos from held-out split")
    video_choices = [os.path.basename(v) for v in video_files]
    video_map = {os.path.basename(v): v for v in video_files}

    with gr.Blocks(title="Semantic AutoGaze Explorer") as app:
        gr.Markdown("# Semantic AutoGaze — Interactive Explorer")
        gr.Markdown(
            "Select a video, type a **text query**, and compare our predicted "
            "heatmaps against CLIPSeg ground truth.\n\n"
            "Heatmaps use a **global color scale** (not per-frame normalized)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_dd = gr.Dropdown(
                    choices=video_choices, label="Video",
                    value=video_choices[0] if video_choices else None,
                )
                num_frames_slider = gr.Slider(4, 32, value=16, step=1, label="Num frames")
                load_btn = gr.Button("Load Video", variant="primary")

                gr.Markdown("---")
                text_input = gr.Textbox(label="Text query", placeholder="e.g. person, right hand, ball...")
                text_btn = gr.Button("Query Text", variant="secondary")

            with gr.Column(scale=2):
                frame_display = gr.Image(label="Video preview",
                                         type="pil", interactive=False)

        result_display = gr.Image(label="Heatmap comparison", type="pil", interactive=False)

        # ── Wire events ──

        def do_load(video_name, nf):
            path = video_map.get(video_name)
            return on_video_select(path, nf)

        load_btn.click(
            fn=do_load,
            inputs=[video_dd, num_frames_slider],
            outputs=[frame_display],
        )

        text_btn.click(
            fn=on_text_query,
            inputs=[text_input],
            outputs=[result_display],
        )
        text_input.submit(
            fn=on_text_query,
            inputs=[text_input],
            outputs=[result_display],
        )

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--head_ckpt", default=None,
                        help="Path to BigHead checkpoint; defaults to warm-restart > baseline")
    parser.add_argument("--val_only", action="store_true",
                        help="Show only held-out validation videos (10%% split, seed=42)")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--hidden_cache_dir", default="results/distill/hidden_cache",
                        help="AutoGaze hidden-state cache for fast video load")
    args = parser.parse_args()

    load_models(args.device, head_path=args.head_ckpt)
    STATE["hidden_cache_dir"] = args.hidden_cache_dir if os.path.isdir(args.hidden_cache_dir) else None
    app = build_app(args.video_dir, val_only=args.val_only, clipseg_dir=args.clipseg_dir)
    # When served behind the remote-vibes reverse proxy, the browser loads the app at
    # `/proxy/<port>/` — Gradio needs to know so its websocket/SSE URLs include that prefix,
    # otherwise the proxy 404s the upgrade and the UI stalls for ~30s per action.
    root_path = os.environ.get("GRADIO_ROOT_PATH", f"/proxy/{args.port}")
    app.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0", server_port=args.port, share=args.share,
        root_path=root_path,
    )
