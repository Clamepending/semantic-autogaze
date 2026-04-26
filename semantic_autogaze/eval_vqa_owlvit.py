"""r/owlvit-hlvid-vqa cycle 1 — DESIGN SCAFFOLD (not yet runnable).

End-to-end VQA test of OWL-ViT-as-filter on HLVid household. Tests whether
NVILA's full transformer decoder extracts query-conditional information from
OWL-ViT-selected patches even when SigLIP-pool fidelity returns null
(per r/query-direct-relevance-metric).

# Architecture

The eval needs to:

1. Read HLVid household samples (n=122) — already established subset from
   r/semantic-only-hlvid-baseline / r/filter-vs-random-baseline / etc.

2. For each (video, question, config in {match, shuffled, random}):
   a. Read raw video frames (16 frames × full HD)
   b. Compute OWL-ViT scores per (frame, query) at the SigLIP 14x14 grid:
      - Resize to 768x768
      - Run OWL-ViT image_embedder + class_head.dense0 → (T, 24, 24, 512)
      - Bilinear-resample to (T, 14, 14, 512)
      - Cosine with OWL-ViT text emb of query → (T, 14, 14, 1)
   c. Select top-K=27 patches per frame (semantic_keep_ratio=0.14)
   d. Bypass AutoGaze's tile/thumbnail gazing — substitute gazing_info dict
      with our OWL-ViT-derived selections
   e. Run NVILA forward with this gazing_info → predicted MCQ letter

3. Compute paired-flip on (matched_correct - shuffled_correct) per question.

# Implementation challenges

- NVILA's processor expects gazing_info with a specific multi-tile structure
  (gazing_pos_tiles, gazing_pos_thumbnails, num_gazing_each_frame_*). The
  positions reference the multi-tile patch grid, not a single 14x14 frame.

- The cleanest approach would be to:
  - Cherry-pick `bypass_autogaze_selection` from r/semantic-only-hlvid-baseline
    cycle 2 (commit e787e0a). This replaces gazing_pos with full-grid arange,
    making the downstream _shrink_unit_batch operate on the full 14x14 grid.
  - Replace `wrapper.semantic_filter.get_scores(hidden, query_emb)` call in
    `_shrink_unit_batch` with an OWL-ViT score provider that takes the
    AutoGaze-preprocessed unit_videos, undoes the AutoGaze normalization,
    re-normalizes for OWL-ViT, runs OWL-ViT, returns scores in the same
    (B, T*N) shape.
  - Add a `--scorer=owlvit` flag to eval_vlm_benchmark.py.

# Implementation status

NOT YET IMPLEMENTED. Pending fresh-context engineering session estimated at
1-2 hours integration + ~1.6h GPU run for the 3-config sweep.

The design above is the agreed approach. Pickup tasks:
1. Cherry-pick bypass_autogaze_selection (44 lines from e787e0a)
2. Cherry-pick random_scoring (~20 lines from r/filter-vs-random-baseline)
3. Add OWL-ViT score provider:
   - Inputs: (B, T, C, H, W) AutoGaze-format video tensor, (B, embed_dim) text
   - Outputs: (B, T*196) sigmoid-shaped scores
4. Add `score_provider` parameter to patch_processor_with_semantic_filter
   that replaces the BigHead scoring path when set
5. Driver: load HLVid n=122, run 3 configs, save predictions + paired-flip stats

# References

- r/query-direct-relevance-metric@36b4e32 — predicts owlvit_match ≈ owlvit_shuf
  with ~45% prior; auto>owlvit_match z=+5.24 suggests text-matched OWL-ViT may
  HURT vs text-blind AutoGaze
- r/semantic-only-hlvid-baseline cycle 2 (e787e0a) — bypass_autogaze_selection
- r/filter-vs-random-baseline (35d884c) — random_scoring + paired-flip
  infrastructure
- semantic_autogaze/eval_fidelity_owlvit.py (be098a9) — OWL-ViT scoring
  pipeline (vision_model + class_head.dense0 + text_projection)

# Decisive test

owlvit_match - owlvit_shuffled paired-flip net:
- ≥+3 wins → principled-fix branch via OWL-ViT IS open at VQA level (cycle 2:
  HLVid av replication + Pareto sweep vs vanilla scale 0.70)
- ≤+1 win → 5th admission class CLOSED (phrase-grounding direction
  definitively closed); routes to nvila-attention-distill (multi-week)
"""
raise NotImplementedError(
    "Engineering scaffold only. See module docstring for the implementation plan."
)
