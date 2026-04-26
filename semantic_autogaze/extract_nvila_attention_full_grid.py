"""r/nvila-attention-distill cycle 1.5c PREP — extract NVILA cross-attention
on a FINE-GRID-ONLY gazing_info so the LLM-token -> (frame, 14x14-patch)
mapping is trivially invertible, then save a (T, 14, 14) score map per qid
for the constraint-compliant filter test (cycle 1.5c).

Why this re-extraction:
- Existing nvila_attention_cache stored attention at gazing_ratio=0.50 plus
  TRUNCATED per-frame gazing_pos in the AutoGaze hidden cache, making the
  LLM-token -> (frame, fine-patch) mapping non-recoverable. See result doc
  cycle 1.5c attempt @ 2026-04-24 in
  `mac-brain/projects/semantic-autogaze/results/nvila-attention-distill.md`.
- Multi-scale full-grid forward (1038 multi-scale patches/frame * 16 frames)
  blows up SigLIP's causal-attention mask -> 8 GB OOM. So we override gazing_info
  with a synthetic dict that selects ONLY the 14x14 sub-grid (196 patches per
  frame) from each scale block. This restricts NVILA's vision encoder to the
  fine grid and makes per-frame layout exactly:
      196 patches -> pad to 198 -> tokenshuffle/9 -> 22 LLM tokens per frame
  This is a SYNTHETIC NVILA forward (different from the natural multi-scale
  default) but is OK for cycle 1.5c because the score it produces is just used
  to pick top-K fine patches — and the actual VQA test runs NVILA's normal
  multi-scale pipeline on those top-K via bypass_autogaze_selection.

What this script does:
- Builds custom gazing_info per sample selecting per-frame positions
  [t*1038 + 58, t*1038 + 254) within the SigLIP grid (14x14 sub-block of the
  3x3+7x7+14x14+28x28 multi-scale layout). Same for thumbnails.
- Runs `model.forward(..., gazing_info=custom_gi)` and hooks
  `model.llm.model.layers[LAYER].self_attn.forward` to capture ONE layer's
  attn_weights to CPU, dropping them from the layer return so GPU memory
  doesn't peak across all 32 layers.
- Computes question->visual attention (head + question-token mean -> (num_v,)).
- Projects (num_v,) -> (T_tile=16, 14, 14) via:
      llm_local = p // 9      (each LLM token covers 9 fine patches; last
                               token of each frame contains 2 real + 7 padding)
      llm_global = t * 22 + llm_local
  for each fine-grid position p in [0, 196).

Outputs per qid in --output_dir:
  attn_compact: (num_v,) raw NVILA attention scores (LLM-token level)
  fine_grid_scores: (T_tile=16, 14, 14) projected fine-grid score map (the
    quantity the cycle 1.5c score provider will load and feed to
    bypass_autogaze_selection).
  num_gazing_each_frame_tiles, num_gazing_each_frame_thumbnails: actual values
    captured from the processor (sanity check).
  question_text: text used for the forward (for shuffled mode this is the
    deranged-Q text, not the original).

Run:
  CUDA_VISIBLE_DEVICES=N python -m semantic_autogaze.extract_nvila_attention_full_grid \\
    --device cuda:0 --mode matched \\
    --output_dir results/nvila_attention_cache_full_grid

  CUDA_VISIBLE_DEVICES=N python -m semantic_autogaze.extract_nvila_attention_full_grid \\
    --device cuda:0 --mode shuffled \\
    --output_dir results/nvila_attention_cache_full_grid_shuffled
"""
from __future__ import annotations
import argparse, os, time, random, traceback
from typing import Optional
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from semantic_autogaze.eval_hlvid_subset import load_subset, PARQUET_PATH


VIDEO_TOKEN_ID = 151650
SHUFFLE_NUM = 9
SCALES_PER_FRAME = 1038          # 9 + 49 + 196 + 784 (3x3 + 7x7 + 14x14 + 28x28)
SCALES_OFFSETS = (0, 9, 58, 254, 1038)  # left-inclusive per scale
FINE_OFFSET = 58                 # start of 14x14 within per-frame 1038
FINE_COUNT = 196                 # 14*14
FINE_END = FINE_OFFSET + FINE_COUNT  # 254
T_TILE_DEFAULT = 16


def deterministic_shuffle(qids, seed=42):
    n = len(qids)
    if n < 2:
        return list(qids)
    rng = random.Random(seed)
    result = list(qids)
    for _ in range(200):
        rng.shuffle(result)
        if all(result[i] != qids[i] for i in range(n)):
            return result
    return [qids[(i + 1) % n] for i in range(n)]


class LayerAttentionCapture:
    """Monkey-patch a single decoder layer's self_attn.forward so we capture
    its attn_weights without paying the all-layer output_attentions=True cost
    (which would peak at ~30 GB at S~3800 across 32 Qwen2 layers).
    """

    def __init__(self, attn_module):
        self.attn_module = attn_module
        self.orig_forward = attn_module.forward
        self.captured: Optional[torch.Tensor] = None

        def patched(*args, **kwargs):
            out = self.orig_forward(*args, **kwargs)
            attn_output, attn_weights = out[0], out[1]
            if attn_weights is not None:
                # Move to CPU fp32, drop from output to free GPU memory.
                self.captured = attn_weights.detach().to(torch.float32).cpu()
            # Drop attn_weights from the layer's return so it doesn't keep GPU mem.
            return (attn_output, None) + tuple(out[2:])

        self.patched = patched

    def __enter__(self):
        self.attn_module.forward = self.patched
        return self

    def __exit__(self, exc_type, exc, tb):
        self.attn_module.forward = self.orig_forward


def make_patched_get_gazing(scales_per_frame: int = SCALES_PER_FRAME,
                             fine_offset: int = FINE_OFFSET,
                             fine_end: int = FINE_END):
    """Returns a function suitable for monkey-patching processor._get_gazing_info_from_videos.
    The patched function ignores AutoGaze and produces a fine-grid-only gazing_info
    that selects positions [fine_offset, fine_end) (the 14x14 sub-block) per frame
    out of `scales_per_frame` total positions per frame in the SigLIP multi-scale
    grid.
    """
    fine_count = fine_end - fine_offset

    def patched_get_gazing(videos_inputs):
        siglip_tiles = videos_inputs["pixel_values_videos_tiles"]
        siglip_thumbs = videos_inputs["pixel_values_videos_thumbnails"]
        gi = {
            "gazing_pos_tiles": [],
            "num_gazing_each_frame_tiles": [],
            "if_padded_gazing_tiles": [],
            "gazing_pos_thumbnails": [],
            "num_gazing_each_frame_thumbnails": [],
            "if_padded_gazing_thumbnails": [],
        }
        for tile_v, thumb_v in zip(siglip_tiles, siglip_thumbs):
            num_tiles, t_tile = tile_v.shape[:2]
            n_thumbs = thumb_v.shape[0]
            device_t = torch.device("cpu")  # processor outputs are CPU; model.forward moves them.
            # Tile path: per-frame positions
            tile_per_frame = [
                torch.arange(t * scales_per_frame + fine_offset,
                             t * scales_per_frame + fine_end,
                             dtype=torch.long, device=device_t)
                for t in range(t_tile)
            ]
            tile_pos = torch.cat(tile_per_frame)  # (T_tile * fine_count,)
            tile_pos = tile_pos.unsqueeze(0).expand(num_tiles, -1).contiguous()
            tile_nge = torch.full((num_tiles, t_tile), fine_count, dtype=torch.long, device=device_t)
            tile_pad = torch.zeros(num_tiles, t_tile * fine_count, dtype=torch.bool, device=device_t)
            gi["gazing_pos_tiles"].append(tile_pos)
            gi["num_gazing_each_frame_tiles"].append(tile_nge)
            gi["if_padded_gazing_tiles"].append(tile_pad)
            # Thumbnail path: each thumb has T=1, fine_count positions in [offset, end)
            thumb_pos = torch.arange(fine_offset, fine_end, dtype=torch.long, device=device_t)
            thumb_pos = thumb_pos.unsqueeze(0).expand(n_thumbs, -1).contiguous()
            thumb_nge = torch.full((n_thumbs, 1), fine_count, dtype=torch.long, device=device_t)
            thumb_pad = torch.zeros(n_thumbs, fine_count, dtype=torch.bool, device=device_t)
            gi["gazing_pos_thumbnails"].append(thumb_pos)
            gi["num_gazing_each_frame_thumbnails"].append(thumb_nge)
            gi["if_padded_gazing_thumbnails"].append(thumb_pad)
        return gi

    return patched_get_gazing


def project_to_fine_grid(attn_compact: np.ndarray, t_tile: int = T_TILE_DEFAULT) -> np.ndarray:
    """Project per-LLM-token attention onto the per-frame 14x14 fine grid.

    Layout (fine-grid-only gazing_info): each tile-frame has 196 fine patches,
    padded to 198 (next mult of 9), -> 22 LLM tokens/frame. Only tile frames
    are projected here; thumbnail-derived LLM tokens are skipped (they encode
    global context, not patch-position information for the filter).

    For fine-grid position p in [0, 196) within tile-frame t:
        local LLM token  = p // 9
        global LLM token = t * 22 + local_llm_token

    Returns shape (t_tile, 14, 14).
    """
    n_per_frame_padded = FINE_COUNT + ((SHUFFLE_NUM - FINE_COUNT % SHUFFLE_NUM) % SHUFFLE_NUM)  # 198
    llm_per_frame = n_per_frame_padded // SHUFFLE_NUM  # 22
    out = np.zeros((t_tile, 14, 14), dtype=np.float32)
    for t in range(t_tile):
        for p in range(FINE_COUNT):
            llm_local = p // SHUFFLE_NUM
            llm_global = t * llm_per_frame + llm_local
            if llm_global >= attn_compact.shape[0]:
                continue
            py, px = divmod(p, 14)
            out[t, py, px] = attn_compact[llm_global]
    return out


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[setup] mode={args.mode}, layer={args.layer}", flush=True)
    print(f"[setup] gazing_info OVERRIDE: fine-grid-only (196 patches/frame, no other scales)",
          flush=True)

    # Processor settings don't matter for gazing_info since we override; keep
    # task_loss/gazing_ratio at defaults so the AutoGaze pre-pass runs cheaply.
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        num_video_frames=args.num_frames,
        num_video_frames_thumbnail=args.num_frames_thumbnail,
        max_tiles_video=args.max_tiles,
        gazing_ratio_tile=0.5,
        gazing_ratio_thumbnail=0.75,
        task_loss_requirement_tile=0.6,
        task_loss_requirement_thumbnail=0.6,
        max_batch_size_autogaze=8,
        autogaze_model_id="nvidia/AutoGaze",
        trust_remote_code=True,
    )
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map=args.device,
        attn_implementation="eager",  # required so attn_weights is computed
    ).eval()
    print(f"[setup] NVILA loaded; cuda mem={torch.cuda.memory_allocated()/1024**3:.2f} GB",
          flush=True)
    n_layers = len(model.llm.model.layers)
    print(f"[setup] LLM has {n_layers} layers; capturing layer {args.layer}", flush=True)

    samples = load_subset(args.video_dir, parquet_path=args.parquet_path, query_mode="stem")
    if args.category:
        samples = [s for s in samples if s.get("category") == args.category]
    if args.n_samples is not None:
        samples = samples[:args.n_samples]
    print(f"[data] {len(samples)} samples (category={args.category})", flush=True)

    qids = [s["question_id"] for s in samples]
    if args.mode == "shuffled":
        shuf_qids = deterministic_shuffle(qids, seed=42)
        qid_to_shufqid = dict(zip(qids, shuf_qids))
        qid_to_question = {s["question_id"]: s["question_raw"] for s in samples}

    target_attn = model.llm.model.layers[args.layer].self_attn
    video_token = processor.tokenizer.video_token

    # Monkey-patch the processor to produce fine-grid-only gazing_info (so
    # input_ids' VIDEO_TOKEN_ID count matches the synthetic gi at model.forward).
    processor._get_gazing_info_from_videos = make_patched_get_gazing()

    for i, s in enumerate(samples):
        qid = s["question_id"]
        out_path = os.path.join(args.output_dir, f"qid_{qid:04d}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"  [{i+1}/{len(samples)}] qid={qid} cached, skipping", flush=True)
            continue

        if args.mode == "shuffled":
            shuf_qid = qid_to_shufqid[qid]
            question_text = qid_to_question[shuf_qid]
            shuffled_from_qid = shuf_qid
        else:
            question_text = s["question_raw"]
            shuffled_from_qid = -1

        try:
            t0 = time.perf_counter()
            inputs = processor(
                text=f"{video_token}\n\n{question_text}",
                videos=s["video_path"],
                return_tensors="pt",
            )
            # The processor used the patched (fine-grid-only) gazing_info; both
            # input_ids' VIDEO_TOKEN_ID count and inputs['gazing_info'] are
            # consistent with our synthetic layout.
            inputs = {k: ([t.to(device) if isinstance(t, torch.Tensor) else t for t in v]
                          if isinstance(v, list)
                          else (v.to(device) if isinstance(v, torch.Tensor) else v))
                      for k, v in inputs.items()}
            if "gazing_info" in inputs and isinstance(inputs["gazing_info"], dict):
                gi_dev = {}
                for k, vlist in inputs["gazing_info"].items():
                    gi_dev[k] = [t.to(device) for t in vlist]
                inputs["gazing_info"] = gi_dev
            input_ids = inputs["input_ids"][0]
            video_mask = (input_ids == VIDEO_TOKEN_ID)
            v_positions = torch.where(video_mask)[0]
            num_v = int(v_positions.numel())
            last_v = int(v_positions.max().item())
            q_positions = torch.arange(last_v + 1, len(input_ids), device=device)
            num_q = int(q_positions.numel())

            with LayerAttentionCapture(target_attn) as cap:
                with torch.inference_mode():
                    _ = model.forward(
                        **inputs,
                        return_dict=True,
                    )
                attn_w = cap.captured  # (1, num_heads, S, S) on CPU fp32

            if attn_w is None:
                raise RuntimeError(
                    "Hook captured no attn_weights — check attn_implementation==eager and that the "
                    "layer's forward returns weights"
                )

            # Slice question -> visual attention; head + q-mean -> (num_v,)
            qpos_cpu = q_positions.cpu()
            vpos_cpu = v_positions.cpu()
            q_to_v = attn_w[0][:, qpos_cpu[:, None], vpos_cpu[None, :]]  # (heads, num_q, num_v)
            attn_compact = q_to_v.mean(dim=(0, 1)).numpy()                # (num_v,)

            fine_scores = project_to_fine_grid(attn_compact, t_tile=args.num_frames)

            wall = time.perf_counter() - t0
            np.savez_compressed(
                out_path,
                attn_compact=attn_compact.astype(np.float32),
                fine_grid_scores=fine_scores.astype(np.float32),
                num_v=int(num_v),
                num_q=int(num_q),
                v_positions=vpos_cpu.numpy(),
                q_positions=qpos_cpu.numpy(),
                qid=int(qid),
                question_text=question_text,
                shuffled_from_qid=int(shuffled_from_qid),
                video_path=s["video_path"],
                layer=int(args.layer),
                config_max_tiles=int(args.max_tiles),
                config_num_frames=int(args.num_frames),
                config_num_frames_thumbnail=int(args.num_frames_thumbnail),
                gazing_info_mode="fine_grid_only",
                mode=args.mode,
            )

            nz = int((fine_scores > 0).sum())
            mx = float(fine_scores.max())
            print(f"  [{i+1}/{len(samples)}] qid={qid} num_v={num_v} num_q={num_q} "
                  f"fine_nz={nz}/{fine_scores.size} fine_max={mx:.4e} "
                  f"nan={int(np.isnan(attn_compact).sum())} wall={wall:.1f}s",
                  flush=True)

            # Free GPU memory
            del attn_w, q_to_v, attn_compact, fine_scores, inputs
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [error] qid={qid}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model_path", default="nvidia/NVILA-8B-HD-Video")
    p.add_argument("--parquet_path", default=PARQUET_PATH)
    p.add_argument("--video_dir", default="hlvid_videos/extracted_household/videos")
    p.add_argument("--category", default="household")
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--num_frames_thumbnail", type=int, default=16)
    p.add_argument("--max_tiles", type=int, default=1)
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--mode", choices=["matched", "shuffled"], default="matched")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    main(args)
