"""Letterbox-square geometry helpers.

A non-square image of size (h, w) is padded to a square of side L = max(h, w)
by adding bars on the short axis. Aspect ratio is preserved; no content is
discarded. The opposite operation (un-letterbox) crops the bar regions to
recover an (h, w) array.

Conventions:
- Pad bars are split evenly: top/bottom for landscape, left/right for portrait.
- For odd-sized differences, the extra pixel goes on the bottom/right.
- Image bars use `pad_color` (default black). Mask bars are always 0.
- Letterbox metadata (top, left, h, w, L) is portable: anything in the
  letterbox-square frame can be mapped back to (h, w) using only these.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class LetterboxInfo:
    """Where the original image lives inside its letterboxed square."""

    h: int  # original height
    w: int  # original width
    L: int  # square side = max(h, w)
    top: int  # rows of pad above the image
    left: int  # cols of pad to the left of the image

    @property
    def bottom(self) -> int:
        return self.L - self.top - self.h

    @property
    def right(self) -> int:
        return self.L - self.left - self.w


def compute_info(h: int, w: int) -> LetterboxInfo:
    L = max(h, w)
    top = (L - h) // 2
    left = (L - w) // 2
    return LetterboxInfo(h=h, w=w, L=L, top=top, left=left)


def letterbox_image(img: Image.Image, pad_color: tuple[int, int, int] = (0, 0, 0)) -> tuple[Image.Image, LetterboxInfo]:
    """Pad a PIL image to a square; return (square_image, info)."""
    w, h = img.size
    info = compute_info(h, w)
    if info.L == w == h:
        return img, info
    square = Image.new("RGB", (info.L, info.L), pad_color)
    square.paste(img, (info.left, info.top))
    return square, info


def letterbox_mask(mask: np.ndarray) -> tuple[np.ndarray, LetterboxInfo]:
    """Pad a 2D mask (h, w) with zeros to a square."""
    h, w = mask.shape
    info = compute_info(h, w)
    if info.L == w == h:
        return mask, info
    out = np.zeros((info.L, info.L), dtype=mask.dtype)
    out[info.top : info.top + h, info.left : info.left + w] = mask
    return out, info


def unletterbox_array(arr_square: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Crop a (L, L) array (e.g. an upsampled heatmap) back to (h, w).

    The input is assumed to be in the letterbox-square frame at side L
    exactly; if not, resize to L first.
    """
    if arr_square.shape[-2:] != (info.L, info.L):
        raise ValueError(
            f"unletterbox_array expects shape ending in ({info.L}, {info.L}); "
            f"got {arr_square.shape}. Resize to L first."
        )
    return arr_square[..., info.top : info.top + info.h, info.left : info.left + info.w]


def heatmap_to_original(heatmap: np.ndarray, info: LetterboxInfo, mode: str = "bilinear") -> np.ndarray:
    """Take a (g, g) heatmap in letterbox-square frame, project to original (h, w).

    Steps: bilinear-resize (g, g) → (L, L) → crop bars → (h, w).
    """
    t = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=(info.L, info.L), mode=mode, align_corners=False if mode == "bilinear" else None)[0, 0].numpy()
    return unletterbox_array(up, info)
