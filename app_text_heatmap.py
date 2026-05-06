"""Interactive webapp: webcam + text prompt → semantic heatmap overlay."""

import math
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

ADAPTER_PATH = Path("models/clip_text_to_dino_adapter_coco_10k_448_mlp.pt")
DINO_MODEL = "facebook/dinov2-small"
CLIP_MODEL = "openai/clip-vit-base-patch16"
DINO_SIZE = 448
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ClipTextToDinoAdapter(torch.nn.Module):
    def __init__(self, clip_dim: int, dino_dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            self.proj = torch.nn.Sequential(
                torch.nn.LayerNorm(clip_dim),
                torch.nn.Linear(clip_dim, dino_dim),
            )
        else:
            self.proj = torch.nn.Sequential(
                torch.nn.LayerNorm(clip_dim),
                torch.nn.Linear(clip_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, dino_dim),
            )
        self.logit_scale = torch.nn.Parameter(torch.tensor(10.0))

    def forward(self, text_features: torch.Tensor, patch_features: torch.Tensor) -> torch.Tensor:
        query = F.normalize(self.proj(text_features), dim=-1)
        logits = torch.einsum("bnd,bd->bn", patch_features, query)
        return logits * self.logit_scale.clamp(1.0, 100.0)


def load_models():
    dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
    dino_model = AutoModel.from_pretrained(DINO_MODEL).to(DEVICE).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL, use_safetensors=True).to(DEVICE).eval()

    ckpt = torch.load(ADAPTER_PATH, map_location=DEVICE, weights_only=True)
    config = ckpt.get("config", {})
    hidden_dim = config.get("hidden_dim", None)
    clip_dim = 512
    dino_dim = 384
    adapter = ClipTextToDinoAdapter(clip_dim, dino_dim, hidden_dim=hidden_dim).to(DEVICE)
    adapter.load_state_dict(ckpt["adapter"])
    adapter.eval()

    for m in [dino_model, clip_model, adapter]:
        for p in m.parameters():
            p.requires_grad_(False)

    return dino_processor, dino_model, clip_processor, clip_model, adapter


_models = {}
_text_cache: dict[str, torch.Tensor] = {}


def get_models():
    if not _models:
        print("Loading models...", flush=True)
        m = load_models()
        _models["dino_processor"] = m[0]
        _models["dino_model"] = m[1]
        _models["clip_processor"] = m[2]
        _models["clip_model"] = m[3]
        _models["adapter"] = m[4]
        print("Models loaded!", flush=True)
    return _models


def get_text_features(text: str) -> torch.Tensor:
    if text not in _text_cache:
        m = get_models()
        prompt = f"a photo of a {text}" if not text.startswith("a ") else text
        inputs = m["clip_processor"](text=[prompt], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            outputs = m["clip_model"].text_model(**inputs)
            features = m["clip_model"].text_projection(outputs.pooler_output)
        _text_cache[text] = F.normalize(features, dim=-1)
    return _text_cache[text]


def process_frame(frame: np.ndarray, text: str, threshold: float, opacity: float) -> np.ndarray:
    if frame is None or not text.strip():
        return frame if frame is not None else np.zeros((448, 448, 3), dtype=np.uint8)

    m = get_models()
    image = Image.fromarray(frame).resize((DINO_SIZE, DINO_SIZE), Image.BICUBIC)
    inputs = m["dino_processor"](images=[image], return_tensors="pt", do_resize=False, do_center_crop=False).to(DEVICE)

    with torch.no_grad():
        patch_features = F.normalize(m["dino_model"](**inputs).last_hidden_state[:, 1:], dim=-1)
        text_features = get_text_features(text.strip())
        logits = m["adapter"](text_features, patch_features)
        probs = torch.sigmoid(logits)

    grid_side = int(math.sqrt(probs.shape[1]))
    heatmap = probs.reshape(grid_side, grid_side).cpu().numpy()

    arr = np.array(image)
    h, w = arr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = (heatmap_resized > threshold).astype(np.float32)
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blend = arr.copy().astype(np.float32)
    overlay_mask = mask[:, :, None] * opacity
    blend = blend * (1 - overlay_mask) + heatmap_colored.astype(np.float32) * overlay_mask
    return blend.astype(np.uint8)


with gr.Blocks(title="Semantic Heatmap") as demo:
    gr.Markdown("## Semantic Text Heatmap — CLIP → DINOv2 Adapter")
    gr.Markdown("Type a category/object name and adjust the threshold to see where it appears in the camera feed.")

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(label="Text Prompt", value="person", placeholder="e.g. cat, car, cup...")
            threshold_slider = gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="Threshold")
            opacity_slider = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Overlay Opacity")
        with gr.Column(scale=3):
            webcam = gr.Image(sources=["webcam"], streaming=True, label="Camera Feed")
            output_image = gr.Image(label="Heatmap Output")

    webcam.stream(
        fn=process_frame,
        inputs=[webcam, text_input, threshold_slider, opacity_slider],
        outputs=[output_image],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
