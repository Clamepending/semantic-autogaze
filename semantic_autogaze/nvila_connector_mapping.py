"""r/nvila-attention-distill Phase 2 step 2 — NVILA connector LLM-token-to-fine-patch mapping.

# DEBUGGED 2026-04-26 — fine-grid assumption is WRONG

Per modeling_nvila.py:_encode_vision (line 431+):
  shuffle_num = 9
  Per video: for each effective frame (tile-frames + thumbnail-frames):
    - Take AutoGaze-kept patches (pad-removed)
    - Pad to multiple of 9 (replicate last token)
    - Concatenate
  Then TokenShuffle reshapes (N) -> (N//9) by grouping every 9 sequential tokens.

CRITICAL INSIGHT: AutoGaze uses MULTI-SCALE patches, NOT just the fine 14x14
grid. From actual gazing_info inspection:
  - num_vision_tokens_each_frame = 265 (= 4 + 16 + 49 + 196 across 4 scales:
    2x2, 4x4, 7x7, 14x14)
  - gazing_pos values range [0, 4240) = 16 frames × 265 multi-scale tokens
  - num_gazing_each_frame is per-frame, NON-UNIFORM across frames (e.g., qid=8
    frame-0 keeps 132 patches but frames 1-15 keep only 10 each — task_loss-driven)

So the SUPERVISION mapping is more complex than the previous fine-grid plan:
  fine 14x14 (196) + 7x7 (49) + 4x4 (16) + 2x2 (4) = 265 multi-scale per frame
  → AutoGaze keeps a subset (varies by task_loss_requirement)
  → per-frame pad to multiple of 9
  → concatenate frames
  → TokenShuffle by 9

BigHead (existing arch) operates at the fine 14x14 grid only. Phase 2 needs
either:
  A) Re-architect BigHead to score at the multi-scale grid (4 sub-grids),
     match NVILA's full pipeline. Higher fidelity but more complex training.
  B) Restrict supervision to fine-14x14 components of NVILA attention. For
     each LLM token, find which of its 9 grouped slots correspond to fine-grid
     positions; supervise BigHead only at those positions. Drops attention
     info from coarser scales.

Pickup checklist for next session:
  1. Decide architecture (A or B). B is faster to prototype.
  2. Build mapping: per-qid, gazing_pos % 265 for each kept patch identifies
     which multi-scale level (or use scale offsets:
        2x2: positions 0-3  (4 patches per frame; offset 0)
        4x4: positions 4-19 (16 patches per frame; offset 4)
        7x7: positions 20-68 (49 patches per frame; offset 20)
        14x14: positions 69-264 (196 patches per frame; offset 69)
     )
  3. For each LLM token i (NVILA attention[i]):
     - Find its 9 slot range in the padded-per-frame-concat sequence
     - For each slot that's a 14x14 (fine-grid) kept patch: supervise BigHead
       at that fine position with attention[i]
     - For coarser-scale slots: ignore (in option B)
  4. Confirm reconstruction: predicted-num-LLM-tokens (from mapping) should
     equal cached num_v.

This is multi-day architectural work.
"""
from __future__ import annotations
import numpy as np
from typing import Optional


SHUFFLE_NUM = 9


def build_llm_to_fine_mapping(
    gazing_pos: np.ndarray,              # (K_kept,) flat positions in [0, T*N) AutoGaze grid
    if_padded: np.ndarray,                # (K_kept,) bool — True = padding (drop)
    num_gazing_each_frame: np.ndarray,    # (T_tile,) per-frame kept counts (post-AutoGaze)
    n_thumbs: int = 16,                   # number of thumbnail frames in our smaller config
    thumbs_per_frame: Optional[np.ndarray] = None,  # (T_thumb,) thumb kept counts (None = all)
    n_per_frame: int = 196,               # fine grid patches per frame (14*14)
    n_thumb_per_frame: int = 196,         # thumbnail grid patches (matches SigLIP)
):
    """Build mapping from LLM token index to list of (kind, frame_idx, patch_idx) tuples.

    Returns:
      llm_to_fine: list[list[tuple]] — per LLM token, the (frame_idx, patch_idx) list
                   (kind in {'tile', 'thumb'})
    """
    # Step 1: extract per-frame kept tile positions
    T_tile = len(num_gazing_each_frame)
    tile_kept_per_frame = []  # list of (n_kept_in_frame,) arrays of patch indices
    pos_offset = 0
    for t in range(T_tile):
        n_in_frame = int(num_gazing_each_frame[t])
        # gazing_pos[pos_offset:pos_offset+n_in_frame] are positions in this frame
        frame_positions = gazing_pos[pos_offset:pos_offset + n_in_frame]
        frame_padded = if_padded[pos_offset:pos_offset + n_in_frame]
        # Keep only non-padded
        kept_frame_pos = frame_positions[~frame_padded.astype(bool)]
        # Modulo to get in-frame patch index (since gazing_pos is flat across T*N)
        kept_frame_pos = kept_frame_pos % n_per_frame
        tile_kept_per_frame.append(kept_frame_pos)
        pos_offset += n_in_frame

    # Step 2: thumbnails (assumed equally kept across all)
    if thumbs_per_frame is None:
        # Default: all thumbnail patches kept (gazing_ratio_thumbnail=1.0)
        thumbs_per_frame = np.full(n_thumbs, n_thumb_per_frame, dtype=np.int64)
    thumb_kept_per_frame = [
        np.arange(int(thumbs_per_frame[i])) for i in range(n_thumbs)
    ]

    # Step 3: build the per-video padded sequence (list of (kind, frame_idx, patch_idx))
    # Tile-frames first, then thumb-frames
    seq = []
    for t in range(T_tile):
        kept = tile_kept_per_frame[t]
        # Append each kept patch with its (kind, frame, patch)
        for p in kept:
            seq.append(("tile", t, int(p)))
        # Pad to multiple of shuffle_num = 9 (replicate the last entry)
        pad = (SHUFFLE_NUM - len(kept) % SHUFFLE_NUM) % SHUFFLE_NUM
        if pad > 0 and len(kept) > 0:
            seq.extend([("tile", t, int(kept[-1]))] * pad)
        elif pad > 0:
            # No kept patches in this frame — pad with dummy zeros
            seq.extend([("tile", t, 0)] * pad)

    for tt in range(n_thumbs):
        kept = thumb_kept_per_frame[tt]
        for p in kept:
            seq.append(("thumb", tt, int(p)))
        pad = (SHUFFLE_NUM - len(kept) % SHUFFLE_NUM) % SHUFFLE_NUM
        if pad > 0 and len(kept) > 0:
            seq.extend([("thumb", tt, int(kept[-1]))] * pad)
        elif pad > 0:
            seq.extend([("thumb", tt, 0)] * pad)

    # Step 4: TokenShuffle by 9 — group every 9 sequential entries into 1 LLM token
    n_llm_tokens = len(seq) // SHUFFLE_NUM
    llm_to_fine: list[list[tuple]] = []
    for i in range(n_llm_tokens):
        chunk = seq[i * SHUFFLE_NUM:(i + 1) * SHUFFLE_NUM]
        llm_to_fine.append(chunk)

    return llm_to_fine, len(seq), n_llm_tokens


def project_attention_to_fine_grid(
    attention: np.ndarray,                # (num_v,) NVILA attention at LLM-token level
    llm_to_fine: list[list[tuple]],       # mapping built by build_llm_to_fine_mapping
    n_frames: int = 16,                   # tile T
    n_per_frame: int = 196,               # 14*14
    include_thumb: bool = False,
) -> np.ndarray:
    """Project NVILA attention back to the fine grid (T*196,) for tile-frames only.

    Each kept fine patch's score = NVILA_attention[llm_token_that_contains_it].
    Non-kept positions get score = 0.

    Returns: (T*n_per_frame,) fine-grid attention array.
    """
    target = np.zeros(n_frames * n_per_frame, dtype=np.float32)
    n_v = len(attention)

    for i, chunk in enumerate(llm_to_fine):
        if i >= n_v:
            break  # In case our mapping has more LLM tokens than NVILA attention
        attn_val = attention[i]
        for kind, frame_idx, patch_idx in chunk:
            if kind == "tile":
                flat_idx = frame_idx * n_per_frame + patch_idx
                if 0 <= flat_idx < target.size:
                    # Multiple LLM tokens may map to the same fine position via padding;
                    # take the MAX (don't average — high attention from any LLM token wins)
                    target[flat_idx] = max(target[flat_idx], attn_val)

    return target


if __name__ == "__main__":
    # Quick test on a real cache
    import sys
    sys.path.insert(0, '/home/ogata/semantic-autogaze')
    cache_dir = '/home/ogata/semantic-autogaze/results/autogaze_hidden_for_distill'
    attn_dir = '/home/ogata/semantic-autogaze/results/nvila_attention_cache'

    qid = 8
    a = np.load(f"{cache_dir}/qid_{qid:04d}.npz", allow_pickle=True)
    b = np.load(f"{attn_dir}/qid_{qid:04d}.npz", allow_pickle=True)

    print(f"qid={qid}")
    print(f"AutoGaze cache: gazing_pos={a['gazing_pos'].shape}, "
          f"if_padded={a['if_padded_gazing'].shape}, "
          f"num_gazing_each_frame={a['num_gazing_each_frame']}")
    print(f"NVILA attn cache: attention={b['attention'].shape}, num_v={b['num_v']}")

    # Build mapping
    nge = a['num_gazing_each_frame']
    if nge.ndim == 0:
        # Scalar — broadcast across T frames
        nge = np.full(16, int(nge), dtype=np.int64)
    print(f"num_gazing_each_frame (broadcasted): {nge}")

    llm_to_fine, total_padded, n_llm = build_llm_to_fine_mapping(
        gazing_pos=a['gazing_pos'],
        if_padded=a['if_padded_gazing'],
        num_gazing_each_frame=nge,
        n_thumbs=16,
        thumbs_per_frame=None,
    )
    print(f"Built mapping: total_padded={total_padded}, n_llm_tokens_predicted={n_llm}, "
          f"actual_num_v_NVILA={int(b['num_v'])}")

    # Project to fine grid
    fine = project_attention_to_fine_grid(b['attention'], llm_to_fine, n_frames=16)
    print(f"Fine-grid target: shape={fine.shape}, "
          f"nonzero={np.sum(fine > 0)}, "
          f"max={fine.max():.4f}, mean={fine.mean():.6f}")
