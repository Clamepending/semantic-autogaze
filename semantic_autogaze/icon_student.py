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
    in_grid: base spatial side for the learned positional embedding (16 for
        DINOv2-small at 224). At forward time, ``pos_embed`` is bilinearly
        interpolated to the actual input grid passed via ``grid_hw``; the
        autogaze pattern is to train at fixed ``(in_grid, in_grid)`` and
        infer at native non-square grids.
    out_grid: spatial side of the output heatmap when input is square at
        ``in_grid``. Output side scales linearly with input grid:
        H_out = upscale * N_h, W_out = upscale * N_w.
        ``upscale = out_grid // in_grid`` must be a power of 2.
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
        # Learned base positional embedding at (in_grid, in_grid), bilinearly
        # interpolated to the actual input grid each forward pass.
        self.pos_embed = nn.Parameter(torch.zeros(1, decoder_dim, in_grid, in_grid))
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

    def _pos_embed_for(self, n_h: int, n_w: int) -> torch.Tensor:
        """Bilinearly interpolate the learned base pos_embed to (n_h, n_w).

        Returns (1, n_h*n_w, decoder_dim) ready to add to patch features.
        """
        if n_h == self.in_grid and n_w == self.in_grid:
            pe = self.pos_embed
        else:
            pe = F.interpolate(self.pos_embed, size=(n_h, n_w),
                               mode="bilinear", align_corners=False)
        return pe.flatten(2).transpose(1, 2)  # (1, n_h*n_w, D)

    def forward(self, patch_features: torch.Tensor, query_embed: torch.Tensor,
                grid_hw: tuple[int, int] | None = None) -> torch.Tensor:
        """
        patch_features: (B, N, patch_dim)  — DINOv2 patch features (no CLS).
        query_embed:    (B, query_dim)     — CLIP text embedding.
        grid_hw:        (n_h, n_w) — spatial shape of the patch grid.
                        Defaults to (in_grid, in_grid). Must satisfy
                        n_h * n_w == N. All samples in a batch must share
                        the same grid (different shapes need separate calls).
        Returns: (B, upscale*n_h, upscale*n_w) raw logits.
        """
        B, N, _ = patch_features.shape
        if grid_hw is None:
            n_h = n_w = self.in_grid
        else:
            n_h, n_w = grid_hw
        assert n_h * n_w == N, f"grid {n_h}x{n_w}={n_h*n_w} != N={N}"

        pe = self._pos_embed_for(n_h, n_w)
        h = self.patch_proj(patch_features) + pe
        q = self.query_proj(query_embed).unsqueeze(1)  # (B, 1, D)

        for block in self.blocks:
            h = block(h, q)
        h = self.feat_norm(h)

        feat = h.transpose(1, 2).reshape(B, self.decoder_dim, n_h, n_w)
        feat = self.upsample(feat)
        logits = self.head(feat).squeeze(1)  # (B, upscale*n_h, upscale*n_w)
        return logits


def trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
