"""
Semantic filtering module for AutoGaze → VLM integration.

Provides utilities to convert semantic head scores into gazing_info dicts
compatible with AutoGaze's ViT integration (mask_with_gazing).

Two modes:
  1. Semantic-only: select patches purely by semantic relevance
  2. Semantic + Gaze: intersect AutoGaze gaze selection with semantic filtering

Usage:
    from semantic_autogaze.semantic_filter import SemanticFilter

    sf = SemanticFilter(
        head_ckpt="results/distill_bighead/best_bighead_student.pt",
        head_type="bighead",
    )

    # From raw video + text query
    gazing_info = sf.filter_video(video, text_query, keep_ratio=0.2)

    # From pre-computed hidden states
    gazing_info = sf.filter_from_hidden(hidden_states, text_embedding, keep_ratio=0.2)

    # Combined with AutoGaze gaze
    gazing_info = sf.filter_with_gaze(
        hidden_states, text_embedding, original_gazing_info, keep_ratio=0.5
    )
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union

from autogaze.utils import get_gazing_pos_from_gazing_mask


def load_semantic_head(head_type, ckpt_path, device="cuda",
                       hidden_dim=192, embedding_dim=512,
                       expanded_dim=384, n_attn_heads=6, n_attn_layers=2,
                       grid_size=14):
    """Load a trained semantic similarity head."""
    if head_type == "bighead":
        from semantic_autogaze.train_bighead import BigSimilarityHead
        head = BigSimilarityHead(
            hidden_dim=hidden_dim, embedding_dim=embedding_dim,
            expanded_dim=expanded_dim, n_attn_heads=n_attn_heads,
            n_attn_layers=n_attn_layers, grid_size=grid_size,
        )
    elif head_type == "small":
        from semantic_autogaze.model import SimilarityHead
        head = SimilarityHead(
            hidden_dim=hidden_dim, embedding_dim=embedding_dim,
            grid_size=grid_size, num_frames=16, use_spatial=True,
        )
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    state = torch.load(ckpt_path, map_location=device)
    head.load_state_dict(state)
    head.to(device).eval()
    return head


class SemanticFilter:
    """Converts semantic head scores into gazing_info for VLM integration."""

    def __init__(self, head_ckpt: str, head_type: str = "bighead",
                 device: str = "cuda", grid_size: int = 14,
                 num_frames: int = 16, **head_kwargs):
        self.device = torch.device(device)
        self.grid_size = grid_size
        self.num_frames = num_frames
        self.patches_per_frame = grid_size * grid_size  # 196

        self.head = load_semantic_head(
            head_type, head_ckpt, device=self.device,
            grid_size=grid_size, **head_kwargs,
        )

    @torch.no_grad()
    def get_scores(self, hidden_states: torch.Tensor,
                   text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute per-patch semantic relevance scores.

        Args:
            hidden_states: (B, T*196, hidden_dim) from AutoGaze encoder/decoder
            text_embedding: (B, 512) CLIP text embedding

        Returns:
            scores: (B, T*196) sigmoid probabilities in [0, 1]
        """
        logits = self.head(hidden_states, text_embedding)
        return torch.sigmoid(logits)

    def scores_to_gazing_info(self, scores: torch.Tensor,
                              keep_ratio: float = 0.2,
                              threshold: Optional[float] = None,
                              num_frames: Optional[int] = None) -> dict:
        """
        Convert semantic scores to gazing_info dict.

        Either keep_ratio or threshold must control selection:
          - keep_ratio: keep top k% of patches (deterministic budget)
          - threshold: keep all patches with score > threshold (variable budget)

        Args:
            scores: (B, T*N) semantic relevance scores
            keep_ratio: fraction of patches to keep (0.0-1.0)
            threshold: absolute score threshold (overrides keep_ratio if set)
            num_frames: number of frames (auto-detected if None)

        Returns:
            gazing_info dict with gazing_pos, if_padded_gazing, num_gazing_each_frame
        """
        B, TN = scores.shape
        T = num_frames or self.num_frames
        N = TN // T
        assert TN == T * N, f"Score length {TN} not divisible by {T} frames"

        if threshold is not None:
            # Variable budget: keep all patches above threshold
            mask = (scores > threshold).long()
            # Ensure at least 1 patch per frame
            scores_per_frame = scores.reshape(B, T, N)
            mask_per_frame = mask.reshape(B, T, N)
            for b in range(B):
                for t in range(T):
                    if mask_per_frame[b, t].sum() == 0:
                        best_idx = scores_per_frame[b, t].argmax()
                        mask[b, t * N + best_idx] = 1
        else:
            # Fixed budget: keep top-k per frame
            k = max(1, int(keep_ratio * N))
            scores_per_frame = scores.reshape(B, T, N)
            mask = torch.zeros_like(scores, dtype=torch.long)
            for t in range(T):
                _, topk_idx = scores_per_frame[:, t].topk(k, dim=-1)
                for b in range(B):
                    global_idx = t * N + topk_idx[b]
                    mask[b, global_idx] = 1

        gazing_pos, if_padded_gazing = get_gazing_pos_from_gazing_mask(mask)

        # Compute per-frame counts
        mask_per_frame = mask.reshape(B, T, N)
        # Use max across batch for uniform frame token counts
        num_gazing_each_frame = mask_per_frame[0].sum(dim=-1)  # (T,)

        return {
            "gazing_pos": gazing_pos,
            "if_padded_gazing": if_padded_gazing,
            "num_gazing_each_frame": num_gazing_each_frame,
            "semantic_scores": scores,  # pass through for downstream use
        }

    @torch.no_grad()
    def filter_from_hidden(self, hidden_states: torch.Tensor,
                           text_embedding: torch.Tensor,
                           keep_ratio: float = 0.2,
                           threshold: Optional[float] = None) -> dict:
        """
        End-to-end: hidden states + text → gazing_info.

        Args:
            hidden_states: (B, T*196, hidden_dim)
            text_embedding: (B, 512) CLIP text embedding
            keep_ratio: fraction of patches to keep
            threshold: score threshold (overrides keep_ratio)

        Returns:
            gazing_info dict
        """
        scores = self.get_scores(hidden_states, text_embedding)
        return self.scores_to_gazing_info(scores, keep_ratio=keep_ratio,
                                          threshold=threshold)

    def intersect_with_gaze(self, semantic_scores: torch.Tensor,
                            original_gazing_info: dict,
                            semantic_keep_ratio: float = 0.5) -> dict:
        """
        Intersect semantic filtering with AutoGaze gaze selection.

        Keeps only the gazed patches that also pass the semantic filter.
        This is a conjunction: patch must be BOTH gazed AND semantically relevant.

        Args:
            semantic_scores: (B, T*N) semantic relevance scores
            original_gazing_info: dict from AutoGaze with gazing_pos, if_padded_gazing
            semantic_keep_ratio: what fraction of gazed patches to keep

        Returns:
            new gazing_info dict with reduced patch set
        """
        B = semantic_scores.shape[0]
        TN = semantic_scores.shape[1]
        gazing_pos = original_gazing_info["gazing_pos"]  # (B, K)
        if_padded = original_gazing_info["if_padded_gazing"]  # (B, K)

        # Get semantic scores for gazed positions only
        gazed_scores = torch.zeros_like(gazing_pos, dtype=semantic_scores.dtype)
        for b in range(B):
            valid = ~if_padded[b]
            valid_pos = gazing_pos[b, valid].clamp(0, TN - 1)
            gazed_scores[b, valid] = semantic_scores[b, valid_pos]
            gazed_scores[b, ~valid] = -1.0  # padded → lowest priority

        # Keep top-k of the gazed patches
        K = gazing_pos.shape[1]
        num_valid = (~if_padded).sum(dim=1)  # (B,)
        keep_k = (num_valid.float() * semantic_keep_ratio).long().clamp(min=1)

        # Build new mask over the original token space
        new_mask = torch.zeros(B, TN, device=semantic_scores.device, dtype=torch.long)
        for b in range(B):
            k = keep_k[b].item()
            valid = ~if_padded[b]
            valid_scores = gazed_scores[b, valid]
            _, topk_local = valid_scores.topk(min(k, valid.sum().item()))
            valid_positions = gazing_pos[b, valid]
            kept_positions = valid_positions[topk_local].clamp(0, TN - 1)
            new_mask[b, kept_positions] = 1

        new_gazing_pos, new_if_padded = get_gazing_pos_from_gazing_mask(new_mask)

        T = self.num_frames
        N = TN // T
        mask_per_frame = new_mask.reshape(B, T, N)
        num_gazing_each_frame = mask_per_frame[0].sum(dim=-1)

        return {
            "gazing_pos": new_gazing_pos,
            "if_padded_gazing": new_if_padded,
            "num_gazing_each_frame": num_gazing_each_frame,
            "semantic_scores": semantic_scores,
        }

    def get_filtering_stats(self, scores: torch.Tensor,
                            keep_ratio: float = 0.2) -> dict:
        """Return stats about the filtering at a given keep ratio."""
        B, TN = scores.shape
        T = self.num_frames
        N = TN // T

        k = max(1, int(keep_ratio * N))
        scores_per_frame = scores.reshape(B, T, N)

        # Per-frame stats
        frame_stats = []
        for t in range(T):
            frame_scores = scores_per_frame[:, t]  # (B, N)
            topk_vals, _ = frame_scores.topk(k, dim=-1)
            frame_stats.append({
                "mean_kept_score": topk_vals.mean().item(),
                "min_kept_score": topk_vals.min().item(),
                "mean_all_score": frame_scores.mean().item(),
            })

        return {
            "keep_ratio": keep_ratio,
            "patches_kept_per_frame": k,
            "total_patches_per_frame": N,
            "tokens_saved_pct": (1 - keep_ratio) * 100,
            "mean_score": scores.mean().item(),
            "mean_kept_score": np.mean([f["mean_kept_score"] for f in frame_stats]),
            "per_frame": frame_stats,
        }
