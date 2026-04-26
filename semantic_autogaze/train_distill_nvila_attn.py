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

## Implementation checklist (NOT YET DONE)

1. **Extract AutoGaze hidden states + gazing_info** for each of 122 samples at
   the smaller config (max_tiles=1, num_frames=16, gazing_ratio=0.50). Cache
   to `results/autogaze_hidden_for_distill/qid_{qid:04d}.npz` with arrays
   `hidden_states (T*196, hidden_dim)`, `gazing_pos (K_kept,)`, `kept_mask (T, 14, 14)`.
   ~1h GPU.

2. **Build NVILAAttentionDataset** (similar to DistillDataset). For each qid:
   - Load AutoGaze hidden states from #1 (input)
   - Load CLIP text embedding for the question (input)
   - Load NVILA attention from cache + AutoGaze gazing_info to compute fine-grid
     supervision target via Strategy A (initially)

3. **Adapt train_distill_bighead.py loop** to use new dataset. Loss: focal BCE
   on top-K=27 mask, MSE on full attention, or composite.

4. **Train**: ~30 epochs, ~1-2h on a single GPU.

5. **Evaluate**: cycle 1.5b-style — use distilled BigHead as filter (top-K=27/196
   per frame at fine grid), compare matched/shuffled/random VQA accuracy.
   Decisive if matched > shuffled by ≥+3 (replicating cycle 1.5b's signal
   through the distilled filter).

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
