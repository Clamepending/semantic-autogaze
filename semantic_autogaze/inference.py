"""
Inference entrypoint: load AutoGaze + a trained head checkpoint, run a
(video, text_query) → per-patch similarity map forward pass.

Supports both the committed `SimilarityHead` and the reconstructed
`BigHead`/`TemporalBigHead`. Head class is inferred from the checkpoint's
state_dict keys.

Target deployment: Modal (see `modal_inference.py`). Local MPS/CPU
fallback is supported for smoke tests.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange


@dataclasses.dataclass
class InferenceOutput:
    logits: torch.Tensor          # (T, 14, 14) raw per-patch logits
    probs: torch.Tensor           # (T, 14, 14) sigmoid probs
    upsampled: torch.Tensor       # (T, H, W) probs bilinear-upsampled to frame size
    query: str
    head_type: str


def _pick_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _infer_head_class(state_dict: dict, cfg: dict | None = None):
    keys = set(state_dict.keys())
    from .bighead import BigHead, BigHeadDecoder, TemporalBigHead
    from .model import SimilarityHead

    kind = (cfg or {}).get("model_kind")
    if kind == "BigHeadDecoder":
        return BigHeadDecoder
    if kind == "BigHead":
        return BigHead

    if any(k.startswith("temporal_blocks.") for k in keys):
        return TemporalBigHead
    if any(k.startswith("decoder_in.") or k.startswith("decoder_up.") or k.startswith("decoder_out.") for k in keys):
        return BigHeadDecoder
    if any(k.startswith("blocks.") and k.endswith(".attn.in_proj_weight") for k in keys):
        return BigHead
    return SimilarityHead


def load_head(ckpt_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, str]:
    """Load a trained head checkpoint and return (module, head_type_name)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    HeadCls = _infer_head_class(state_dict, cfg)
    if HeadCls.__name__ == "SimilarityHead":
        head = HeadCls(hidden_dim=cfg.get("hidden_dim", 192), embedding_dim=cfg.get("embedding_dim", 512))
    else:
        head = HeadCls(**{k: v for k, v in cfg.items() if k in HeadCls.__init__.__code__.co_varnames})
    head.load_state_dict(state_dict)
    head.to(device).eval()
    return head, HeadCls.__name__


def load_autogaze(model_name: str = "nvidia/AutoGaze", device: Optional[torch.device] = None):
    """Load the frozen AutoGaze backbone from HF."""
    device = device or _pick_device(None)
    # Deferred import so the module is importable without autogaze installed yet.
    from autogaze.models.autogaze import AutoGaze  # type: ignore

    model = AutoGaze.from_pretrained(model_name)
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval()


def encode_text_clip(query: str, device: torch.device) -> torch.Tensor:
    """Encode a text query with open_clip ViT-B/16 (OpenAI weights). Returns (1, 512)."""
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model = model.to(device).eval()
    with torch.no_grad():
        toks = tokenizer([query]).to(device)
        emb = model.encode_text(toks).float()
        emb = F.normalize(emb, dim=-1)
    return emb


def extract_patch_hidden(autogaze, video: torch.Tensor) -> torch.Tensor:
    """Run AutoGaze encoder+decoder to get per-patch post-decoder hidden states.

    This mirrors `SemanticAutoGaze.get_patch_hidden_states` in model.py. See
    that method for the canonical implementation.
    """
    from .model import SemanticAutoGaze

    wrapper = SemanticAutoGaze(autogaze)
    return wrapper.get_patch_hidden_states(video)


@torch.no_grad()
def run_inference(
    video: torch.Tensor,
    query: str,
    head_ckpt_path: str | Path,
    autogaze_model_name: str = "nvidia/AutoGaze",
    device: Optional[str] = None,
) -> InferenceOutput:
    """
    video : (T, C, H, W) or (B=1, T, C, H, W), uint8 or float in [0, 1].
    """
    dev = _pick_device(device)
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.dtype == torch.uint8:
        video = video.float() / 255.0
    video = video.to(dev)

    autogaze = load_autogaze(autogaze_model_name, dev)
    head, head_type = load_head(head_ckpt_path, dev)

    hidden = extract_patch_hidden(autogaze, video)      # (1, T*196, 192)
    query_embed = encode_text_clip(query, dev)          # (1, 512)

    logits = head(hidden, query_embed)                  # (1, T*196)
    probs = torch.sigmoid(logits)

    T = video.shape[1]
    grid = 14
    logits_grid = logits.reshape(T, grid, grid)
    probs_grid = probs.reshape(T, grid, grid)

    H, W = video.shape[-2:]
    upsampled = F.interpolate(probs_grid.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)
    upsampled = upsampled.squeeze(1)

    return InferenceOutput(
        logits=logits_grid.cpu(),
        probs=probs_grid.cpu(),
        upsampled=upsampled.cpu(),
        query=query,
        head_type=head_type,
    )
