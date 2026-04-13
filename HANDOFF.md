# Semantic AutoGaze — Session Handoff

Snapshot of where this thread left off so a fresh session can resume.

## What just shipped: K-shrinking semantic filter

**Problem:** The semantic filter was only marking patches as `if_padded=True`
(dropped at the connector before the LLM). The SigLIP transformer still
physically encoded all `K` positions, with padded entries replaced by
position-0 dummies — so SigLIP compute was unchanged. End-to-end speedup was
~1.18× even with 14× LLM-token cuts.

**Fix:** Physically rebuild `gazing_pos` to a smaller `K_new` so SigLIP only
sees the kept positions. Helpers in `semantic_autogaze/eval_vlm_benchmark.py`:

- `_pick_kept(scores, ratio, threshold)` — keep-mask logic (top-k or
  threshold), always keeps ≥1.
- `_shrink_unit_batch(unit_videos, if_padded, gazing_pos, num_gaze_per_frame, …)`
  — scores all items in one wrapper batch, computes per-frame max kept across
  items, builds shrunk `(B, K_new)` `gazing_pos` + `if_padded`. Within a frame
  slot, kept items come first; short items pad with first-kept dummy.
- `_repad_to_shared_budget(pos, pad, kt_local, kt_shared)` — pads each
  per-video shrunk tensor up to a cross-video shared `kt_shared` so NVILA's
  assertion `num_gazing_each_frame identical across videos` holds.

`patch_processor_with_semantic_filter`'s inner `patched_get_gazing_info` now:
1. For each video: batch all tiles → `_shrink_unit_batch`; same for thumbnails.
2. Reconcile per-frame budgets across videos (elementwise max of `new_kt`),
   re-pad with `_repad_to_shared_budget`.
3. Write back `gazing_pos_*`, `if_padded_gazing_*`, and replicate `new_kt`
   across rows of `num_gazing_each_frame_*`.

## Validated results

### Paper-style speedup (GPU time, n_runs=8, max_samples=3, max_tiles=2)

`results/paper_style_hlvid_v3_kshrink/paper_style_hlvid.json`

| nf | regime | ms | tokens | speedup |
|---|---|---|---|---|
| 16 | AutoGaze only | 738 | 1383 | 1.00× |
| 16 | Intersect 30% | 297 | 481 | 2.48× |
| 16 | Intersect 10% | 267 | 224 | 2.77× |
| 16 | Semantic 10% | 270 | 224 | 2.73× |
| 16 | Thresh 0.01 | 268 | 111 | 2.75× |
| 32 | AutoGaze only | 1890 | 2346 | 1.00× |
| 32 | Intersect 30% | 551 | 779 | 3.43× |
| 32 | Intersect 10% | 349 | 329 | 5.42× |
| 32 | Semantic 10% | 334 | 329 | **5.66×** |
| 32 | Thresh 0.01 | 333 | 143 | **5.68×** |

Floor at ~270ms (nf=16) / ~330ms (nf=32) — fixed cost (LLM init + 16-token
decode + projector). Speedups scale with K, so larger nf wins more.

Compared to v2 (no K-shrink): max 1.18× → max 5.68×.

### HLVid accuracy (n=11, in-progress on GPU 4)

`results/hlvid_subset_v3_kshrink/run.log` (running, see "Background processes").

Through Thresh 0.005 (sample 8 of 11 in last config we observed):

| Config | Acc | Notes |
|---|---|---|
| AutoGaze only | 1/5 (OOM 6-10) | baseline OOMs on long videos at full K |
| Intersect 30% | 1/11 | full coverage, no OOM ← K-shrink validation |
| Intersect 10% | 1/11 | same predictions as 30% |
| Semantic 10% | 1/11 | same predictions as Intersect |
| Semantic 2% | 1/11 | q86 prediction shifts (A→D); q89 (B→C) |
| Thresh 0.005+ | running | … |

Read q79 is the only correctly answered one across all regimes — the 11-sample
HLVid subset is dominated by hard questions the model gets wrong even at full
budget. Filter is **not** breaking accuracy; it's preserving the same
predictions as baseline on the shared subset and fixing the OOM that broke
baseline on samples 6-10.

## Background processes still running

| PID | What | Output | Cleanup |
|---|---|---|---|
| 1244121 | `eval_hlvid_subset` accuracy sweep on GPU 4 (~17 min in, ~3 hrs total: 9 configs × 11 samples × ~16 s) | `results/hlvid_subset_v3_kshrink/run.log` and `…/hlvid_subset.json` on completion | `kill 1244121` to abort early |
| 1253800 | `/tmp/watch_paper_speedup_v3_nf64.sh` watcher polling for ≥14 GB free GPU, will launch `paper_style_hlvid_speedup --num_frames 32 64 --max_tiles 4 --max_samples 2` once accuracy run frees GPU 4 | `/tmp/watch_v3_nf64.out` and `results/paper_style_hlvid_v3_nf64/` | `kill 1253800` to abort |

Also see `/tmp/watch_hlvid_acc_v2.sh` and `/tmp/watch_paper_speedup_v2.sh`
(older watchers, not currently running).

## Where to pick up

1. **Wait for accuracy run to finish**, then read
   `results/hlvid_subset_v3_kshrink/hlvid_subset.json` for the full
   per-config breakdown including the threshold regimes.
2. **Wait for nf=64 push to finish** (auto-launched after step 1) and read
   `results/paper_style_hlvid_v3_nf64/paper_style_hlvid.json`. Hypothesis:
   speedup keeps growing past 5.7× at nf=64 because SigLIP cost grows
   linearly with K.
3. The 11-sample HLVid subset is too small to differentiate filter quality
   from model error. To get a real accuracy signal we'd need either (a) more
   downloaded shards or (b) a different benchmark (NeXT-QA, MVBench, etc).
4. Ablation idea: turn off K-shrink (revert to the if_padded-only filter) and
   measure speedup at the same configs to isolate "SigLIP cost saved" from
   "LLM cost saved" line-by-line. The v2 numbers in
   `results/paper_style_hlvid_v2/` already give us this for max_tiles=2,
   nf={16,32}.

## Key files

| File | What |
|---|---|
| `semantic_autogaze/eval_vlm_benchmark.py` | The K-shrinking semantic filter (`_pick_kept`, `_shrink_unit_batch`, `_repad_to_shared_budget`, `patch_processor_with_semantic_filter`) |
| `semantic_autogaze/eval_hlvid_subset.py` | HLVid subset accuracy eval; runs all 9 regimes (gaze_only / intersect 30%,10% / semantic 10%,2% / thresh 0.005,0.01,0.02,0.05) |
| `semantic_autogaze/paper_style_hlvid_speedup.py` | Pre-decoded VLM-only timing benchmark (CUDA events, n_runs warmup+measure), per (nf, regime, sample) |
| `autogaze/vision_encoders/siglip/modeling_siglip.py` (lines 130-220) | `mask_with_gazing` — gather to `(B, K, …)`, dummy-fills padded entries |
| NVILA `modeling_nvila.py` (lines 478-481, 498-500) | Asserts `num_gazing_each_frame` is identical across videos for tiles/thumbnails |
| NVILA `modeling_nvila.py` (lines 511-525) | Connector drops `if_padded=True` entries before LLM |
| NVILA `processing_nvila.py` (lines 218-260) | LLM `video_token_padding_strategy` = sum_frames(`(non_padded + 8) // 9`) |

## Calibration notes

- BigSimilarityHead (BigHead) sigmoid output is saturated near 0:
  observed max ≈ 0.06–0.11, p99 ≈ 0.014, p90 ≈ 0.004 on HLVid.
  → absolute thresholds must be in `[0.005, 0.05]`. Higher thresholds collapse
  to the `_pick_kept` argmax fallback (1 keep per frame).
- `gazing_ratio_thumbnail < 1.0` is **required** for AutoGaze to actually emit
  thumbnail tensors; otherwise `pixel_values_videos_thumbnails_autogaze` is
  never built and the semantic filter has nothing to filter for thumbs.
  Both eval scripts default to `0.75`.
- `task_loss_requirement_thumbnail=0.6` is also set in `eval_hlvid_subset.py`
  to mirror tile behavior.

## Quick verify

```
cd /home/ogata/semantic-autogaze
python3 -c "
from semantic_autogaze.eval_vlm_benchmark import _pick_kept, _repad_to_shared_budget, _shrink_unit_batch
import torch
# unit tests for the pure tensor helpers (the integration was already validated end-to-end)
s = torch.tensor([0.1, 0.5, 0.3, 0.9, 0.2])
assert _pick_kept(s, 0.4, None).sum() == 2
assert _pick_kept(s, 1.0, 0.4).tolist() == [False, True, False, True, False]
assert _pick_kept(s, 1.0, 0.99).sum() == 1
print('OK')
"
```

## How to relaunch the speedup benchmark from scratch

```
HF_MODULES_CACHE=/tmp/hf_modules CUDA_VISIBLE_DEVICES=N PYTHONUNBUFFERED=1 \
  python3 -m semantic_autogaze.paper_style_hlvid_speedup \
    --device cuda:0 --num_frames 16 32 --max_tiles 2 --n_runs 8 --n_warmup 2 \
    --max_samples 3 --output_dir results/paper_style_hlvid_v3_kshrink \
    --gazing_ratio_thumbnail 0.75
```

Need ≥14 GB free on the chosen GPU.
