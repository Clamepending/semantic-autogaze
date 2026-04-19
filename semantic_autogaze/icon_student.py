"""IconStudent — image-only text-conditioned segmentation head.

Frozen DINOv2-small image features + frozen CLIP text query → small
trainable cross-attention decoder → high-resolution heatmap.

No AutoGaze in the loop. Designed to test "can a small student match
CLIPSeg's heatmap quality at much lower compute" as a clean proof of
concept, with two interchangeable supervision recipes:

  - A: distill CLIPSeg soft heatmaps  (BCE on logits)
  - B: COCO instance masks            (focal + dice)

Inference cost target: one DINOv2 forward pass per image (cacheable),
then ~5-10ms decoder per text query.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CrossAttnBlock(nn.Module):
    """Self-attention over patches + cross-attention to a single query token."""

    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, patches: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # patches: (B, N, D); query: (B, 1, D)
        h = self.norm_self(patches)
        a, _ = self.self_attn(h, h, h, need_weights=False)
        patches = patches + a

        hp = self.norm_cross(patches)
        hq = self.norm_q(query)
        c, _ = self.cross_attn(hp, hq, hq, need_weights=False)
        patches = patches + c

        patches = patches + self.mlp(self.norm_mlp(patches))
        return patches


class IconStudent(nn.Module):
    """Trainable decoder that turns (DINOv2 patches, CLIP query) → heatmap.

    Frozen backbones live OUTSIDE this module — caller passes precomputed
    `patch_features` and `query_embed`. This keeps the trainable surface
    small and the data path cacheable.

    Parameters
    ----------
    patch_dim: 384 for DINOv2-small, 768 for DINOv2-base.
    query_dim: 512 for CLIP-ViT-B/16 text.
    decoder_dim: internal width of the cross-attention decoder.
    in_grid: spatial side of the patch feature map (16 for DINOv2-small at 224).
    out_grid: spatial side of the output heatmap. Must be a power-of-2 multiple of in_grid.
    n_layers: cross-attention blocks.
    n_heads: attention heads.
    """

    def __init__(
        self,
        patch_dim: int = 384,
        query_dim: int = 512,
        decoder_dim: int = 256,
        in_grid: int = 16,
        out_grid: int = 128,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert out_grid >= in_grid and (out_grid // in_grid) & ((out_grid // in_grid) - 1) == 0, \
            f"out_grid/in_grid must be power of 2; got {out_grid}/{in_grid}"

        self.in_grid = in_grid
        self.out_grid = out_grid
        self.upscale = out_grid // in_grid
        self.decoder_dim = decoder_dim

        self.patch_proj = nn.Linear(patch_dim, decoder_dim)
        self.query_proj = nn.Linear(query_dim, decoder_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, in_grid * in_grid, decoder_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            _CrossAttnBlock(decoder_dim, n_heads, dropout=dropout) for _ in range(n_layers)
        )
        self.feat_norm = nn.LayerNorm(decoder_dim)

        # PixelShuffle stack: decoder_dim → (decoder_dim*4) conv3x3 → PS×2, repeat.
        ups = []
        cur_dim = decoder_dim
        cur_scale = 1
        while cur_scale < self.upscale:
            ups.append(nn.Conv2d(cur_dim, cur_dim * 4, kernel_size=3, padding=1))
            ups.append(nn.GELU())
            ups.append(nn.PixelShuffle(2))
            cur_scale *= 2
        self.upsample = nn.Sequential(*ups) if ups else nn.Identity()
        self.head = nn.Conv2d(cur_dim, 1, kernel_size=1)

    def forward(self, patch_features: torch.Tensor, query_embed: torch.Tensor) -> torch.Tensor:
        """
        patch_features: (B, N, patch_dim)  — DINOv2 patch features (no CLS).
        query_embed:    (B, query_dim)     — CLIP text embedding.
        Returns: (B, out_grid, out_grid) raw logits.
        """
        B, N, _ = patch_features.shape
        assert N == self.in_grid * self.in_grid, f"expected {self.in_grid**2} patches, got {N}"

        h = self.patch_proj(patch_features) + self.pos_embed
        q = self.query_proj(query_embed).unsqueeze(1)  # (B, 1, D)

        for block in self.blocks:
            h = block(h, q)
        h = self.feat_norm(h)

        feat = h.transpose(1, 2).reshape(B, self.decoder_dim, self.in_grid, self.in_grid)
        feat = self.upsample(feat)
        logits = self.head(feat).squeeze(1)  # (B, out_grid, out_grid)
        return logits


def trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
