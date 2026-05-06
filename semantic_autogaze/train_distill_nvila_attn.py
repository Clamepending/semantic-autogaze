"""r/nvila-attention-distill Phase 2 — distill NVILA cross-attention into BigHead.

# DESIGN STATUS — DESIGN PHASE 2026-04-26

Cycle 1.5b passed the gate (matched > shuffled by +6 paired-flip on n=122),
so distillation is justified. This module implements the training pipeline.

## Supervision target

Per cached `results/nvila_attention_cache/qid_*.npz`, NVILA's question→visual
cross-attention at layer 14 head-averaged, on the SMALLER config
(max_tiles=1, num_frames=16, gazing_ratio=0.50).

Each cache file has:
- attention: (num_v,) — attention scores at NVILA's LLM-level visual tokens
  (post-AutoGaze pruning, post-NVILA-connector 2x2 pool)
- v_positions, q_positions: positions in LLM input sequence
- num_v, num_q: counts

## Mapping problem

NVILA attention is on LLM-level tokens (varies 551-1399 per video). BigHead
operates at the AutoGaze fine-grid (T*196 per video). The chain:

    BigHead output (fine grid: T_fine * 196)
         ↓ (existing) AutoGaze pruning by gazing_ratio_tile
    AutoGaze-kept patches
         ↓ NVILA connector 2x2 spatial pool
    LLM-level visual tokens (num_v ≈ 1k)
         ↓ NVILA-attention (cached)
    NVILA attention map (num_v,)

For supervision, we want BigHead to predict scores at the FINE GRID such that
top-K of BigHead's prediction matches top-K of (gazing → pool → attention).

## Two distillation strategies

### Strategy A: Train at LLM-token level (simpler, lower-resolution supervision)

BigHead → fine-grid scores → AutoGaze-kept-mask × scores → 2x2 pool → predicted
LLM-token scores. MSE/BCE loss against NVILA attention.

Pro: works entirely in post-AutoGaze space; no need to invert anything.
Con: requires cached AutoGaze gazing_info; doesn't supervise BigHead on
positions AutoGaze drops (so BigHead may make bad predictions there at
filter-as-replacement time).

### Strategy B: Train at fine-grid level with imputation (richer, more complex)

For supervision target at fine grid:
- Positions in AutoGaze-kept set: assign target = NVILA-attention-of-that-LLM-token
- Positions AutoGaze dropped: assign target = 0 (or use a learned imputation)
- Each LLM-token corresponds to 4 fine-grid positions (2x2 pool); assign all 4
  the same value (or distribute uniformly).

Pro: BigHead learns at fine grid, ready for filter-as-replacement at K=27/196.
Con: requires gazing_info inversion; assumption that dropped positions ≈ 0.

## Recommended path

Pilot with Strategy A on a subset (n=40 train + n=10 val). Quick training (~1 hour).
Evaluate cycle 1.5b-style on the val split. If signal preserves, scale to
Strategy B for full filter-as-replacement.

## Implementation checklist

1. **Extract AutoGaze hidden states + gazing_info** for each of 122 samples at
   the smaller config (max_tiles=1, num_frames=16, gazing_ratio=0.50). ✓ DONE
   commit aea64a0. Cache at `results/autogaze_hidden_for_distill/qid_{qid:04d}.npz`
   with arrays `hidden_states (3136, 192)`, `gazing_pos (K_kept,)`, `if_padded`,
   `num_gazing_each_frame`. K_kept varies 25-696 by video.

2. **Build NVILAAttentionDataset** (PENDING). For each qid:
   - Load AutoGaze hidden_states (input)
   - Load CLIP text embedding for the question (input)
   - Load NVILA attention from cache (target) + the gazing_info (mapping)

   **The mapping problem (deferred to next session)**: NVILA's connector
   applies a 2x2 spatial pool (within each frame's 14x14 SigLIP grid → 7x7
   per frame) BEFORE AutoGaze pruning, yielding 49 LLM tokens per frame.
   Then AutoGaze prunes some of these to fit gazing_ratio. So:
     fine 14x14 → 2x2 spatial pool → 7x7 per frame (49 LLM tokens) →
     AutoGaze prune → K_kept LLM tokens

   But our AutoGaze gazing_pos is on the 14x14 fine grid (sum of frame counts
   = K_kept × 4 if we counted multi-scale). This needs reconciliation.

   **ALTERNATIVE**: Skip the precise pool inversion. Train BigHead at the
   LLM-token level directly:
   - BigHead hidden_dim=192 stays the same
   - Output dim becomes 49 per frame instead of 196
   - Loss: BCE on top-K=14% of NVILA attention positions per video

   This sidesteps the pool inversion but requires a new BigHead architecture
   variant and changes the filter-as-replacement semantics.

3. **Adapt train_distill_bighead.py loop** to use new dataset. Loss: focal
   BCE on top-K mask of NVILA attention (K=14% of num_v).

4. **Train** ~30 epochs (~1-2h GPU).

5. **Evaluate cycle 1.5b-style**: distilled BigHead → patch scores. Use as
   filter at the same smaller config. Compare matched/shuffled/random VQA
   accuracy. Decisive: matched > shuffled by ≥+3 paired-flip preserves
   through distillation → direction validated and ready for Phase 3
   (cross-config + cross-dataset + admission test).

## Engineering note for next iteration

The NVILA connector layout details need careful study before step 2 can
proceed cleanly. The path of least resistance is probably to match BigHead's
output layout to NVILA's LLM-token layout (49 per frame after 2x2 pool),
training BigHead directly on the post-pool grid. Then AT INFERENCE the
distilled BigHead can be unfolded to the 14x14 grid by replicating each
LLM-token score to its 4 fine patches.

## Open design questions

- Layer choice for distillation target: layer 14 (tested in cycle 1.5b) or
  multi-layer aggregate?
- Loss formulation: focal BCE vs MSE vs InfoNCE-style contrastive?
- Architecture: reuse `BigSimilarityHead(hidden_dim=192, embedding_dim=512,
  grid_size=14)` or design something with explicit 2x2 pool head?
"""
raise NotImplementedError(
    "Phase 2 design scaffold. See module docstring for the implementation checklist. "
    "Cycle 1.5b gate has been PASSED — Phase 2 is justified and unlocked."
)
