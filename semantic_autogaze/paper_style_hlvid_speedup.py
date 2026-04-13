"""
Paper-faithful HLVid speedup benchmark.

Matches the AutoGaze paper's measurement boundary: we time *only* the VLM
forward (SigLIP + LLM prefill + generate), with decoded video tensors
pre-cached, so the pyav/disk cost doesn't swamp the numbers. End-to-end
HLVid timing showed no speedup because ~93% of wall time was video decode
on 4K 5-min clips — a line item the filter cannot reduce.

For each (num_frames, regime) combination we run N measurement passes with
M warmup passes, using CUDA events for precise GPU timing, and report:
  - median / mean wall time per sample (ms)
  - n_input_tokens (the LLM context size — the knob our filter moves)
  - speedup vs AutoGaze-only baseline at the same num_frames

Regimes mirror the paper:
  - AutoGaze only            (gaze_only,     keep=1.0)   baseline
  - Intersect 50%            (intersect,     keep=0.5)
  - Intersect 10%            (intersect,     keep=0.1)
  - Semantic only 10%        (semantic_only, keep=0.1)
  - Semantic only 2%         (semantic_only, keep=0.02)  aggressive

Usage:
    HF_MODULES_CACHE=/tmp/hf_modules CUDA_VISIBLE_DEVICES=N python3 \\
      -m semantic_autogaze.paper_style_hlvid_speedup --device cuda:0 \\
      --num_frames 16 32 64 --n_runs 10 --n_warmup 3
"""

import argparse
import json
import os
import statistics
import time

import open_clip
import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from semantic_autogaze.eval_hlvid_subset import PARQUET_PATH, parse_question_and_choices
from semantic_autogaze.eval_vlm_benchmark import patch_processor_with_semantic_filter
from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper


REGIMES = [
    {"name": "AutoGaze only",     "mode": "gaze_only",     "keep": 1.0,  "thr": None},
    {"name": "Intersect 30%",     "mode": "intersect",     "keep": 0.3,  "thr": None},
    {"name": "Intersect 10%",     "mode": "intersect",     "keep": 0.1,  "thr": None},
    {"name": "Semantic 10%",      "mode": "semantic_only", "keep": 0.1,  "thr": None},
    {"name": "Semantic 2%",       "mode": "semantic_only", "keep": 0.02, "thr": None},
    # Threshold-based, calibrated to the BigHead's sigmoid range
    # (observed max ~0.06-0.11, p99 ~0.015 on HLVid). Variable per-frame budget.
    {"name": "Thresh 0.005",      "mode": "semantic_only", "keep": 1.0,  "thr": 0.005},
    {"name": "Thresh 0.01",       "mode": "semantic_only", "keep": 1.0,  "thr": 0.01},
    {"name": "Thresh 0.02",       "mode": "semantic_only", "keep": 1.0,  "thr": 0.02},
    {"name": "Thresh 0.05",       "mode": "semantic_only", "keep": 1.0,  "thr": 0.05},
]


def gpu_time_generate(model, inputs, max_new_tokens):
    """CUDA-event timed single generate call. Returns (ms, n_input, n_gen, response_ids)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    n_input = int(inputs["input_ids"].shape[1])
    n_gen = int(outputs.shape[1] - n_input)
    return ms, n_input, n_gen


def load_hlvid_samples(video_dir):
    df = pd.read_parquet(PARQUET_PATH)
    samples = []
    for _, r in df.iterrows():
        full = os.path.join(video_dir, r["video_path"])
        if not os.path.exists(full):
            continue
        stem, _, sem_q = parse_question_and_choices(r["question"])
        samples.append({
            "qid": int(r["question_id"]),
            "video": full,
            "question_raw": r["question"],
            "stem": stem,
            "semantic_query": sem_q,
            "answer": r["answer"],
        })
    return samples


def set_regime(processor, wrapper, clip_model, clip_tok, cfg, query_text, device,
               log_score_dist=False):
    """Reset + re-patch the processor for a given regime."""
    if hasattr(processor, "_original_get_gazing"):
        processor._get_gazing_info_from_videos = processor._original_get_gazing
    else:
        processor._original_get_gazing = processor._get_gazing_info_from_videos
    if cfg["mode"] != "gaze_only":
        patch_processor_with_semantic_filter(
            processor, wrapper, clip_model, clip_tok,
            mode=cfg["mode"], semantic_keep_ratio=cfg["keep"],
            query_text=query_text, device=str(device),
            score_threshold=cfg.get("thr"),
            filter_thumbnails=True,
            log_score_dist=log_score_dist,
        )


def preprocess_and_cache(processor, sample, device):
    """Run processor once and return GPU-resident inputs dict."""
    video_token = processor.tokenizer.video_token
    inputs = processor(
        text=f"{video_token}\n\n{sample['question_raw']}",
        videos=sample["video"],
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    return inputs


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("[setup] NVILA-8B-HD-Video 4-bit …")
    # Load the model once; we'll hot-swap processor params for each num_frames.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    # Use the largest num_frames for the initial processor to allocate the
    # most contiguous activations first — but NVILA's processor takes
    # num_frames as an init param, so we'll need to rebuild it per-nf.
    model = None
    clip_model = None
    clip_tok = None
    wrapper = None

    samples = load_hlvid_samples(args.video_dir)
    print(f"[data] {len(samples)} HLVid QA samples")

    # Results grid: one row per (num_frames, regime, sample)
    all_rows = []

    for nf in args.num_frames:
        # Decide num_frames_thumbnail proportionally (paper uses nf/2)
        nft = max(args.num_frames_thumbnail or (nf // 2), 1)
        print(f"\n{'=' * 70}\n[setup] num_frames={nf} num_frames_thumbnail={nft} max_tiles={args.max_tiles}\n{'=' * 70}")

        # Rebuild processor for this nf (num_frames is an init param).
        # IMPORTANT: gazing_ratio_thumbnail must be < 1.0 so AutoGaze actually
        # produces thumbnail tensors that we can semantically filter further.
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            num_video_frames=nf,
            num_video_frames_thumbnail=nft,
            max_tiles_video=args.max_tiles,
            gazing_ratio_tile=args.gazing_ratio,
            gazing_ratio_thumbnail=args.gazing_ratio_thumbnail,
            task_loss_requirement_tile=0.6,
            task_loss_requirement_thumbnail=0.6,
            max_batch_size_autogaze=8,
            autogaze_model_id="nvidia/AutoGaze",
            trust_remote_code=True,
        )

        # Lazy model/clip/wrapper load on first pass
        if model is None:
            print("[setup] Loading NVILA (4-bit) …")
            model = AutoModel.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                quantization_config=bnb,
                device_map=args.device,
                max_batch_size_siglip=8,
            ).eval()
            torch.cuda.empty_cache()

            print("[setup] CLIP + wrapper …")
            clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
            clip_tok = open_clip.get_tokenizer("ViT-B-16")
            clip_model = clip_model.to(device).eval()
            wrapper = SemanticAutoGazeWrapper(
                autogaze_model_name=args.autogaze_model,
                head_ckpt=args.ckpt,
                head_type=args.head_type,
                device=str(device),
            )
            torch.cuda.empty_cache()
            free_mb = torch.cuda.mem_get_info(device)[0] / 1024**2
            print(f"[setup] Free: {free_mb:.0f} MiB")

        # For each sample we (a) pre-decode once under each regime (since the
        # regime affects gazing_pos which is *part* of the inputs), (b) then
        # time only model.generate() with the cached inputs. This isolates
        # VLM compute from pyav.
        for s_idx, s in enumerate(samples[: args.max_samples]):
            for cfg in REGIMES:
                # Log score distribution once per (nf, regime) on the first sample
                # for any non-trivial regime, so the user can pick thresholds.
                log_dist = (s_idx == 0 and cfg["mode"] != "gaze_only")
                set_regime(processor, wrapper, clip_model, clip_tok, cfg,
                           s["semantic_query"], device, log_score_dist=log_dist)
                # Pre-decode once per (sample, regime) — still counts pyav once,
                # but we DON'T time it.
                try:
                    t_pre0 = time.perf_counter()
                    inputs = preprocess_and_cache(processor, s, device)
                    t_pre = time.perf_counter() - t_pre0
                except Exception as e:
                    print(f"  skip q{s['qid']} {cfg['name']}: preprocess failed: {e}")
                    continue

                # Warmup
                try:
                    for _ in range(args.n_warmup):
                        gpu_time_generate(model, inputs, args.max_new_tokens)
                except torch.cuda.OutOfMemoryError:
                    print(f"  OOM q{s['qid']} {cfg['name']} at nf={nf}")
                    continue

                # Measurement
                runs_ms = []
                for _ in range(args.n_runs):
                    ms, n_in, n_gen = gpu_time_generate(model, inputs, args.max_new_tokens)
                    runs_ms.append(ms)

                row = {
                    "num_frames": nf,
                    "max_tiles": args.max_tiles,
                    "regime": cfg["name"],
                    "qid": s["qid"],
                    "n_input_tokens": n_in,
                    "n_gen_tokens": n_gen,
                    "ms_mean": statistics.mean(runs_ms),
                    "ms_median": statistics.median(runs_ms),
                    "ms_stdev": statistics.stdev(runs_ms) if len(runs_ms) > 1 else 0.0,
                    "preproc_s": t_pre,
                    "n_runs": args.n_runs,
                }
                all_rows.append(row)
                print(f"  nf={nf:>3}  q{s['qid']:>3}  {cfg['name']:<20}  "
                      f"gen={row['ms_median']:>7.1f}ms  tokens={n_in:>5}  pre={t_pre:>4.1f}s")

        # Save incrementally in case of OOM on a later config
        with open(os.path.join(args.output_dir, "paper_style_hlvid.json"), "w") as f:
            json.dump({"rows": all_rows, "args": vars(args)}, f, indent=2)

    # Summary table: speedup vs AutoGaze-only within each nf
    print(f"\n{'=' * 70}\n[SUMMARY] VLM-only latency (GPU time, median across {args.n_runs} runs)\n{'=' * 70}")
    by_nf = {}
    for r in all_rows:
        by_nf.setdefault((r["num_frames"], r["regime"]), []).append(r)
    # aggregate across samples
    agg = {}
    for (nf, regime), rows in by_nf.items():
        agg[(nf, regime)] = {
            "ms_median": statistics.median([r["ms_median"] for r in rows]),
            "tokens_median": statistics.median([r["n_input_tokens"] for r in rows]),
            "n_samples": len(rows),
        }

    print(f"{'num_frames':>10}  {'regime':<22}  {'ms (median)':>12}  {'tokens':>8}  {'speedup':>8}")
    for nf in args.num_frames:
        base_key = (nf, "AutoGaze only")
        base = agg.get(base_key, {}).get("ms_median", float("nan"))
        for cfg in REGIMES:
            a = agg.get((nf, cfg["name"]))
            if a is None:
                continue
            spd = base / a["ms_median"] if a["ms_median"] > 0 else 0.0
            print(f"{nf:>10}  {cfg['name']:<22}  {a['ms_median']:>12.1f}  "
                  f"{a['tokens_median']:>8.0f}  {spd:>7.2f}×")

    with open(os.path.join(args.output_dir, "paper_style_hlvid.json"), "w") as f:
        json.dump({"rows": all_rows, "agg": {f"{k[0]}|{k[1]}": v for k, v in agg.items()},
                   "args": vars(args)}, f, indent=2)
    print(f"\n[save] {args.output_dir}/paper_style_hlvid.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="nvidia/NVILA-8B-HD-Video")
    p.add_argument("--video_dir", default="hlvid_videos/extracted/videos")
    p.add_argument("--num_frames", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--num_frames_thumbnail", type=int, default=None,
                   help="Defaults to num_frames // 2 if unset")
    p.add_argument("--max_tiles", type=int, default=2)
    p.add_argument("--gazing_ratio", type=float, default=0.75)
    p.add_argument("--gazing_ratio_thumbnail", type=float, default=0.75,
                   help="Must be <1.0 to allow semantic-filtering thumbnails.")
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--ckpt", default="results/bighead_warmrestart/best_bighead_student.pt")
    p.add_argument("--head_type", default="bighead")
    p.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/paper_style_hlvid")
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--max_samples", type=int, default=4,
                   help="Run at most this many HLVid samples per regime (speed)")
    args = p.parse_args()
    main(args)
