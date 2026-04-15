"""
BigHead — spatial self-attention + MLP over AutoGaze patch features, conditioned on a CLIP text-query embedding.

RECONSTRUCTED 2026-04-15 from wandb configs + output.log dumps (runs
`nkmyibxc`, `w7ly4c8w`, `lkrzsjm4`, `6unhnux3`). Ground-truth code lives on
the deactivated `cthulhu1` cluster at `/home/ogata/semantic-autogaze/`; when
that cluster returns, diff and replace. Architecture choices here are the
simplest implementation consistent with the log signals:

  - ablation A4_bighead_nospatial is worse than A1, so the block uses MHSA
    across patches (spatial attention).
  - ablation A3_bighead_nodistill is the worst of the bighead variants, so
    the distillation KD loss in the training loop does meaningful work.
  - "BigHead student" param counts in `nkmyibxc` output.log: 8011.8K params
    at expanded_dim=512, layers=3. Matches the calculation below for a
    3-layer pre-norm transformer with the given hidden sizes.

If you are reading this after the cluster is back, DELETE this file and
replace with the canonical implementation from the cluster.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PatchSelfAttnBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class BigHead(nn.Module):
    """Query-conditioned per-patch similarity head with spatial self-attention.

    Parameters
    ----------
    hidden_dim : int
        Dim of AutoGaze per-patch hidden state (192 in the observed runs).
    embedding_dim : int
        Dim of the CLIP text embedding (512 for ViT-B/16).
    expanded_dim : int
        Internal hidden dim of the BigHead. Winning run used 512 but e=384
        with L=2 reportedly scored slightly better (`A1`: 0.0668 vs
        nkmyibxc: 0.0683) — possibly overfitting on 32K samples when scaling.
    n_attn_heads : int
        Multi-head attention heads (6 or 8 in observed configs).
    n_attn_layers : int
        Stacked transformer blocks (2 or 3).
    use_spatial_attn : bool
        If False, drop the attention blocks (A4_nospatial ablation).
    """

    def __init__(
        self,
        hidden_dim: int = 192,
        embedding_dim: int = 512,
        expanded_dim: int = 512,
        n_attn_heads: int = 8,
        n_attn_layers: int = 3,
        use_spatial_attn: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.expanded_dim = expanded_dim

        # Fused projection of (patch_hidden || broadcast(query)) → expanded_dim.
        # Param counts come out close to the observed 8012K @ e=512,L=3 with
        # mlp_ratio=4 in the transformer blocks; exact match likely requires
        # the cluster code.
        self.proj = nn.Linear(hidden_dim + embedding_dim, expanded_dim)

        self.use_spatial_attn = use_spatial_attn
        self.blocks = nn.ModuleList(
            _PatchSelfAttnBlock(expanded_dim, n_attn_heads, dropout=dropout)
            for _ in range(n_attn_layers if use_spatial_attn else 0)
        )
        self.out_norm = nn.LayerNorm(expanded_dim)
        self.out_proj = nn.Linear(expanded_dim, 1)

    def forward(self, patch_hidden: torch.Tensor, query_embed: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        patch_hidden : (B, N, hidden_dim)  AutoGaze post-decoder hidden states.
        query_embed  : (B, embedding_dim)  CLIP text embedding.

        Returns
        -------
        logits : (B, N)  raw per-patch scores (apply sigmoid for probability).
        """
        B, N, _ = patch_hidden.shape
        q = query_embed.unsqueeze(1).expand(B, N, -1)
        h = self.proj(torch.cat([patch_hidden, q], dim=-1))  # (B, N, E)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(self.out_norm(h)).squeeze(-1)


class TemporalBigHead(nn.Module):
    """BigHead + temporal attention across frames.

    RECONSTRUCTED. Observed configs: `n_spatial_layers` ∈ {2, 3} and
    `n_temporal_layers` ∈ {1, 2}. In practice temporal variants did not
    decisively beat spatial-only bighead (r1zz134x plateau ≈ 0.0694).
    """

    def __init__(
        self,
        hidden_dim: int = 192,
        embedding_dim: int = 512,
        expanded_dim: int = 384,
        n_attn_heads: int = 6,
        n_spatial_layers: int = 2,
        n_temporal_layers: int = 1,
        num_frames: int = 16,
        num_patches_per_frame: int = 196,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_patches_per_frame = num_patches_per_frame

        self.query_proj = nn.Linear(embedding_dim, expanded_dim)
        self.patch_proj = nn.Linear(hidden_dim, expanded_dim)
        self.input_norm = nn.LayerNorm(expanded_dim)
        self.spatial_blocks = nn.ModuleList(
            _PatchSelfAttnBlock(expanded_dim, n_attn_heads, dropout=dropout)
            for _ in range(n_spatial_layers)
        )
        self.temporal_blocks = nn.ModuleList(
            _PatchSelfAttnBlock(expanded_dim, n_attn_heads, dropout=dropout)
            for _ in range(n_temporal_layers)
        )
        self.out_norm = nn.LayerNorm(expanded_dim)
        self.out_proj = nn.Linear(expanded_dim, 1)

    def forward(self, patch_hidden: torch.Tensor, query_embed: torch.Tensor) -> torch.Tensor:
        """patch_hidden: (B, T*N, hidden_dim); returns (B, T*N) logits."""
        B, TN, _ = patch_hidden.shape
        T = self.num_frames
        N = self.num_patches_per_frame
        assert TN == T * N, f"expected T*N={T*N} patches, got {TN}"

        q = self.query_proj(query_embed).unsqueeze(1)
        h = self.patch_proj(patch_hidden)
        h = self.input_norm(h + q)

        for block in self.spatial_blocks:
            h_spatial = h.reshape(B * T, N, -1)
            h_spatial = block(h_spatial)
            h = h_spatial.reshape(B, T * N, -1)

        for block in self.temporal_blocks:
            h_temporal = h.reshape(B, T, N, -1).permute(0, 2, 1, 3).reshape(B * N, T, -1)
            h_temporal = block(h_temporal)
            h = h_temporal.reshape(B, N, T, -1).permute(0, 2, 1, 3).reshape(B, T * N, -1)

        return self.out_proj(self.out_norm(h)).squeeze(-1)
