"""Localhost Gradio UI: image + text query → semantic heatmap overlay.

Pipeline per click: CLIP text encode → AutoGaze forward (224×224) → BigHead →
bilinear upsample → jet colormap overlay.  All models pre-warmed at startup.

Run:
    python scripts/gradio_app.py --head-ckpt results/coco_seg_v7/best_head.pt
"""

from __future__ import annotations

import argparse
import time

import gradio as gr
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

from semantic_autogaze.inference import _pick_device, load_autogaze, load_head
from semantic_autogaze.model import SemanticAutoGaze

AUTOGAZE_SIZE = 224
MAX_DISPLAY = 512
_STATE: dict = {}
_CMAP = matplotlib.colormaps["jet"]


@torch.inference_mode()
def warmup(ckpt_path: str, device: torch.device):
    import open_clip

    t0 = time.time()
    print(f"[warmup] device={device}", flush=True)

    ag = load_autogaze(device=device)
    wrapper = SemanticAutoGaze(ag).to(device).eval()
    for p in wrapper.parameters():
        p.requires_grad = False

    head, head_name = load_head(ckpt_path, device)
    print(f"[warmup] head: {head_name}", flush=True)

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    dummy = torch.zeros(1, 1, 3, AUTOGAZE_SIZE, AUTOGAZE_SIZE, device=device)
    h = wrapper.get_patch_hidden_states(dummy)
    q = F.normalize(clip_model.encode_text(clip_tokenizer(["warmup"]).to(device)).float(), dim=-1)
    _ = head(h, q)
    if device.type == "mps":
        torch.mps.synchronize()

    _STATE.update(dict(device=device, wrapper=wrapper, head=head,
                       head_name=head_name, clip_model=clip_model,
                       clip_tokenizer=clip_tokenizer))
    print(f"[warmup] READY — total {time.time()-t0:.1f}s", flush=True)


def _resize_for_display(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= MAX_DISPLAY:
        return image
    scale = MAX_DISPLAY / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return t[0].permute(1, 2, 0).byte().numpy()


def _prep_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


@torch.inference_mode()
def run(image, query, alpha, normalize):
    t0 = time.time()
    if image is None:
        raise gr.Error("Upload an image first.")
    if not query or not query.strip():
        raise gr.Error("Enter a text query.")

    device = _STATE["device"]
    image = _prep_image(image)
    display_img = _resize_for_display(image)
    disp_h, disp_w = display_img.shape[:2]

    t_img = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0).to(device)
    t_img = F.interpolate(t_img.unsqueeze(0),
                          size=(AUTOGAZE_SIZE, AUTOGAZE_SIZE),
                          mode="bilinear", align_corners=False)
    video = t_img.unsqueeze(0)  # (1, 1, 3, 224, 224)

    hidden = _STATE["wrapper"].get_patch_hidden_states(video)

    toks = _STATE["clip_tokenizer"]([query]).to(device)
    q = F.normalize(_STATE["clip_model"].encode_text(toks).float(), dim=-1)

    logits = _STATE["head"](hidden, q)  # (1, N)
    n_out = logits.shape[-1]
    grid = int(round(n_out ** 0.5))
    probs = torch.sigmoid(logits).reshape(grid, grid)

    heat = F.interpolate(probs[None, None], size=(disp_h, disp_w),
                         mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

    if normalize:
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    else:
        heat = heat.clip(0, 1)

    heat_rgb = (_CMAP(heat)[..., :3] * 255).astype(np.uint8)
    overlay = ((1 - alpha) * display_img.astype(np.float32)
               + alpha * heat_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)

    dt_ms = (time.time() - t0) * 1000
    print(f"[run] {query!r} {image.shape} → {dt_ms:.0f}ms", flush=True)
    info = (f"head: {_STATE['head_name']}\n"
            f"query: {query!r}\n"
            f"device: {device}  |  output grid: {grid}×{grid}\n"
            f"input: {image.shape}  |  display: {disp_h}×{disp_w}\n"
            f"probs min/max: {probs.min().item():.4f} / {probs.max().item():.4f}\n"
            f"inference: {dt_ms:.0f} ms")
    return overlay, info


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Semantic AutoGaze Heatmap") as demo:
        gr.Markdown("# Semantic AutoGaze — Heatmap Viz\n"
                    "Upload image + text query. Pre-warmed, <100ms per click.")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="Image", type="numpy",
                                 sources=["upload", "clipboard"])
                query = gr.Textbox(label="Text query",
                                   placeholder="e.g., 'person', 'red car', 'dog'")
                alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                  label="Overlay alpha")
                normalize = gr.Checkbox(value=True, label="Min-max normalize heatmap")
                btn = gr.Button("Run", variant="primary")
            with gr.Column(scale=2):
                out_image = gr.Image(label="Heatmap overlay", type="numpy")
                info = gr.Textbox(label="Info", interactive=False, lines=6)
        btn.click(run, inputs=[image, query, alpha, normalize],
                  outputs=[out_image, info], api_name="run")
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head-ckpt", required=True, help="Path to trained head .pt")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    device = _pick_device(None)
    warmup(args.head_ckpt, device)
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name=args.host, server_port=args.port,
                show_error=True, inbrowser=False, ssr_mode=False)


if __name__ == "__main__":
    main()
