"""Multi-query late-fusion wrapper around TextScorerHead.

The trained TextScorerHead splits cleanly into a *text-independent* prefix
(patch_proj + pos_embed + 2× self_attn over patches; ~4 M params, ~80 %
of head compute) and a *text-conditional* tail (text_proj + cross_attn +
score_mlp + spatial; ~0.9 M params).

For N queries on the same frame, the prefix runs once and the tail runs
N times (vectorized along the query batch). Expected wall-time:

    1× backbone + 1× prefix + N × tail
  ≈ 1× backbone + 1× prefix + 0.05-0.15× prefix per added query

So 5 queries on the same frame cost only 5-25 % more than 1.

Aggregation modes for combining per-query heatmaps into one displayable map:

  "max"  — element-wise max (UNION semantics: patch relevant to ANY query)
  "min"  — element-wise min (INTERSECTION: relevant to ALL queries)
  "mean" — average score across queries
  "sum"  — additive (good when queries are intended to be combined)
  "softmax" — weighted by score (most-confident query wins, soft).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQueryScorer:
    """Reuses a trained TextScorerHead but exposes a multi-query forward.

    Not an nn.Module; just an inference-time wrapper. The wrapped head
    keeps all its params; we only re-thread the forward pass.
    """

    def __init__(self, head):
        self.head = head
        self.grid_size = head.grid_size

    @torch.no_grad()
    def encode_patches(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """Run text-INdependent prefix once. Returns (B, N_patches, H)."""
        h = self.head
        x = h.patch_proj(patch_feats) + h.pos_embed
        for layer in h.self_attn_layers:
            r = x; x = layer["norm1"](x)
            xa, _ = layer["attn"](x, x, x); x = r + xa
            r = x; x = layer["norm2"](x); x = r + layer["ffn"](x)
        return x

    @torch.no_grad()
    def score_with_queries(self, x: torch.Tensor, text_embs: torch.Tensor) -> torch.Tensor:
        """Apply text-conditional tail with Q queries.

        x:          (B, N, H) prefix output
        text_embs:  (Q, text_dim)

        Returns:    (B, Q, N) per-query patch logits.
        """
        h = self.head
        B, N, H = x.shape
        Q = text_embs.shape[0]

        # Expand to (B*Q, N, H) and (B*Q, 1, H) for batched cross-attn.
        x_rep = x.unsqueeze(1).expand(B, Q, N, H).reshape(B * Q, N, H)
        text_rep = text_embs.unsqueeze(0).expand(B, Q, -1).reshape(B * Q, -1)
        q_proj = h.text_proj(text_rep).unsqueeze(1)
        cross_out, _ = h.cross_attn(x_rep, q_proj, q_proj)
        x_q = h.cross_norm(x_rep + cross_out)
        scores = h.score_mlp(x_q).squeeze(-1)  # (B*Q, N)
        if h.spatial is not None:
            G = self.grid_size
            grids = scores.reshape(B * Q, 1, G, G)
            scores = (grids + h.spatial(grids)).reshape(B * Q, N)
        return scores.reshape(B, Q, N)

    @torch.no_grad()
    def __call__(self, patches: torch.Tensor, text_embs: torch.Tensor,
                 reduce: str = "max", apply_sigmoid: bool = True) -> torch.Tensor:
        """End-to-end: patches × Q text embs → reduced (B, N) score map.

        reduce ∈ {max, min, mean, sum, softmax, none}.
        - "none" returns (B, Q, N) without aggregation.
        - "softmax" returns (B, N) where each patch's score is the
          self-weighted average of per-query scores.
        """
        x = self.encode_patches(patches)
        scores = self.score_with_queries(x, text_embs)  # (B, Q, N)
        if apply_sigmoid:
            scores = torch.sigmoid(scores)
        if reduce == "none":
            return scores
        if reduce == "max":
            return scores.amax(dim=1)
        if reduce == "min":
            return scores.amin(dim=1)
        if reduce == "mean":
            return scores.mean(dim=1)
        if reduce == "sum":
            return scores.sum(dim=1)
        if reduce == "softmax":
            # Softmax over Q axis, weighted by self → "winner-take-some"
            w = scores.softmax(dim=1)
            return (w * scores).sum(dim=1)
        raise ValueError(f"unknown reduce: {reduce}")
