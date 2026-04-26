"""r/nvila-attention-distill Phase 1 — extract NVILA's question→visual cross-attention.

# DESIGN STATUS — DESIGN PHASE 2026-04-26

## Goal

For each (video, question) pair on HLVid household + av (n=217), extract NVILA's
attention pattern from question tokens to visual tokens. Use as supervision target
for a BigHead-style distilled filter that picks query-relevant patches.

## Computational constraint

Full attention matrix at vanilla NVILA config:
- max_tiles=4, num_video_frames=32, gazing_ratio_tile=0.20
- ≈ 14k visual tokens kept after AutoGaze
- seq_len ≈ 14050 (visual + question)
- attention per layer: (1, 28 heads, 14050, 14050) ≈ 22 GB
- 28 layers → not extractable on 24-GB hardware

**We don't need the full attention matrix — only the (num_q, num_v) submatrix**.
That is, attention from question tokens (~50) to visual tokens (~14k).
For one layer, head-averaged: ≈ 50 × 14k × 4 bytes = 3 MB. Trivial.

## Three extraction options

### Option A: Custom attention layer patch (BEST, but invasive)

Monkey-patch each Qwen2 attention layer's forward to compute only the
(q_positions, v_positions) attention block. Requires intercepting `q`, `k`
projections, computing `q[q_positions] @ k[v_positions].T / sqrt(d)`, softmax
along v dim. Saves 4 orders of magnitude over the full attention matrix.

### Option B: Smaller supervision config (FASTER, but imperfect)

Use a smaller NVILA config for attention extraction:
- max_tiles=1, num_video_frames=16, gazing_ratio_tile=0.50
- ≈ 1500 visual tokens
- attention per layer: (1, 28, 1550, 1550) ≈ 270 MB
- 28 layers all-at-once: ~7 GB (but only need a few middle layers)

Use `output_attentions=True` and `attn_implementation="eager"` to materialize
attention. Extract layer 14 head-averaged.

Trade-off: this attention pattern may not match what vanilla-config NVILA does.
But the supervision is on coarse "which patches are relevant", and the answer
should be similar across configs for the same question.

### Option C: Grad-CAM proxy (SIMPLEST, but indirect)

Run NVILA forward + backward. Take ∂(answer log-prob)/∂(visual feature embeddings).
|grad| is a per-token importance score. NOT attention but correlated with it.

bnb_4bit blocks gradients through the quantized weights, but visual tokens enter
as `inputs_embeds` (not through quantized matmul) so grads should flow.

## Recommended path

1. **Pilot with Option B (smaller config)** — easiest to wire, lower OOM risk.
2. If Option B's distilled filter works, skip Option A.
3. If Option B fails, escalate to Option A.

## Implementation status

NOT YET IMPLEMENTED. This script is a planning placeholder. Engineering
checklist:

1. Load NVILA at smaller config (max_tiles=1, num_video_frames=16,
   gazing_ratio=0.50) with `attn_implementation="eager"`.
2. For each (video, question):
   a. Tokenize input. Note video_token_id=151650 positions.
   b. Run model.forward(...) with output_attentions=True, no generate.
   c. From outputs.attentions (list of 28 layers, each (1, 28, S, S) on CPU/GPU):
      - layer = 14
      - q_positions = positions in input_ids past the video tokens (question tokens)
      - v_positions = positions in input_ids equal to video_token_id (where visual features were inserted)
      - attn_q_to_v = attentions[14][0, :, q_positions, :][:, :, v_positions]  # (28 heads, num_q, num_v)
      - attn_avg = attn_q_to_v.mean(dim=(0, 1))  # (num_v,) — averaged over heads + question tokens
   d. Map num_v back to (T, 14, 14) per frame using gazing_info (which positions
      on the original 14×14 grid are kept).
   e. Save to results/nvila_attention_cache/qid_{qid}.npz with arrays
      `attention_grid (T, 14, 14)`, `gazing_pos (K_kept,)`, `kept_mask (T, 14, 14)`.

3. Sanity-visualize 5 samples: attention overlay on representative frames.

4. Cycle 1.5 gate (per result-doc): use cached attention DIRECTLY as patch
   scorer in NVILA forward. 3-config (matched/shuffled/random) HLVid household
   VQA test. Decisive: if matched > shuffled by ≥+3 paired-flip, proceed to
   Phase 2 distillation.

Estimated wall: 217 samples × ~10 s/sample = ~36 min for extraction. Plus
implementation time. Pilot run first to verify the approach.
"""
raise NotImplementedError(
    "Engineering placeholder. See module docstring for the implementation plan."
)
