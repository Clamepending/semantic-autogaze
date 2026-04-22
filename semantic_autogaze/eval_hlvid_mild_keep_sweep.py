"""
Mild-keep Intersect/Semantic sweep on HLVid household n=122.

Motivation (from r/filter-token-count-ablation, 2026-04-22):
at matched effective keep ratio, AutoGaze-only strictly beats the filter by
paired flip +5 to +13 across four tested cross-pairs (scale 0.3 / 0.1 / 0.1 /
0.05 vs Intersect 30 / 10 / Semantic 10 / 2 %). The existing filter sweep only
tested AGGRESSIVE keep ratios (30 %, 10 %, 2 %) — far below AutoGaze's native
75 % thumbnail / 20 % tile budget.

This move tests whether the filter's systematic loss is specific to aggressive
regimes. At MILD keep ratios (75 %, 50 %, 25 %) the filter preserves most of
AutoGaze's reconstruction-salient selections while only pruning the lowest
text-relevance patches. The decisive question:

  Does Intersect @ keep=K (with K ≥ 0.5) beat AutoGaze-only at matched
  effective count, or does it regress at ALL keep ratios?

Matched-count pairs vs the existing AutoGaze-only gaze-budget sweep:
  Intersect 75 %  ≈  AutoGaze scale 0.75  (interpolated ~0.410)
  Intersect 50 %  =   AutoGaze scale 0.50 = 0.402 (exact match)
  Intersect 25 %  ≈  AutoGaze scale 0.25  (interpolated ~0.380)

If Intersect 50 % ≥ 0.402 → filter choice ADDS value once the keep ratio is
mild — reopens the autogaze-aware filter-training direction. Train a new head
with a within-AutoGaze-set loss as the follow-up.

If Intersect 50 % < 0.402 (same pattern as 30 % / 10 % / 2 %) → filter choice
is strictly dominated by AutoGaze choice at ALL budgets; the existing BigHead
provides zero net value on top of AutoGaze. Filter direction closed; pivot to
teacher retrain (SigLIP-2) or other axes.

Usage:
    CUDA_VISIBLE_DEVICES=? PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      python -m semantic_autogaze.eval_hlvid_mild_keep_sweep \\
        --device cuda:0 \\
        --video_dir hlvid_videos/extracted_household/videos \\
        --ckpt results/bighead_warmrestart/best_bighead_student.pt \\
        --output_dir results/autogaze_aware_filter_train/mild_keep_sweep
"""

import os, json, time, random, argparse
import torch
import open_clip
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper
from semantic_autogaze.eval_vlm_benchmark import (
    patch_processor_with_semantic_filter,
    extract_answer,
)
from semantic_autogaze.eval_hlvid_subset import load_subset, run_inference


# Configs: mild-keep filter ratios + one repeat of aggressive 10% as sanity check
# that this run reproduces the r/hlvid-household-expand number (0.238).
CONFIGS = [
    {"name": "Intersect 75%",    "mode": "intersect",     "keep_ratio": 0.75, "thr": None},
    {"name": "Intersect 50%",    "mode": "intersect",     "keep_ratio": 0.50, "thr": None},
    {"name": "Intersect 25%",    "mode": "intersect",     "keep_ratio": 0.25, "thr": None},
    {"name": "Semantic 75%",     "mode": "semantic_only", "keep_ratio": 0.75, "thr": None},
    {"name": "Semantic 50%",     "mode": "semantic_only", "keep_ratio": 0.50, "thr": None},
    # Sanity-check repeat of r/hlvid-household-expand's Intersect 10 % (= 0.238):
    {"name": "Intersect 10% (sanity)", "mode": "intersect", "keep_ratio": 0.10, "thr": None},
]


def main(args):
    device = torch.device(args.device)
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print("[setup] Loading NVILA-8B-HD-Video (4-bit NF4) first...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        num_video_frames=args.num_frames,
        num_video_frames_thumbnail=args.num_frames_thumbnail,
        max_tiles_video=args.max_tiles,
        gazing_ratio_tile=args.gazing_ratio,
        gazing_ratio_thumbnail=args.gazing_ratio_thumbnail,
        task_loss_requirement_tile=0.6,
        task_loss_requirement_thumbnail=0.6,
        max_batch_size_autogaze=8,
        autogaze_model_id="nvidia/AutoGaze",
        trust_remote_code=True,
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map=args.device,
        max_batch_size_siglip=8,
    ).eval()
    torch.cuda.empty_cache()
    free_mb = torch.cuda.mem_get_info(device)[0] / 1024**2
    print(f"[setup] NVILA loaded. Free after NVILA: {free_mb:.0f} MiB")

    print("[setup] Loading CLIP...")
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()

    print("[setup] Loading SemanticAutoGazeWrapper...")
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )
    torch.cuda.empty_cache()
    free_mb = torch.cuda.mem_get_info(device)[0] / 1024**2
    print(f"[setup] All models loaded. Free: {free_mb:.0f} MiB")

    samples = load_subset(args.video_dir, query_mode="stem")
    print(f"[data] HLVid subset: {len(samples)} QA pairs on {len(set(s['video_path'] for s in samples))} videos")
    if len(samples) == 0:
        print("[warn] No samples found. Exiting.")
        return

    all_results = {}
    per_sample_all = []

    for cfg in CONFIGS:
        print(f"\n{'=' * 60}\n[config] {cfg['name']}\n{'=' * 60}")
        # Reset the processor's gazing hook between configs
        if hasattr(processor, "_original_get_gazing"):
            processor._get_gazing_info_from_videos = processor._original_get_gazing
        else:
            processor._original_get_gazing = processor._get_gazing_info_from_videos

        correct, total = 0, 0
        latencies = []
        rows = []
        for i, s in enumerate(samples):
            try:
                # Re-patch per-sample because query_text varies per question
                patch_processor_with_semantic_filter(
                    processor, wrapper, clip_model, clip_tokenizer,
                    mode=cfg["mode"], semantic_keep_ratio=cfg["keep_ratio"],
                    query_text=s["semantic_query"], device=str(device),
                    score_threshold=cfg.get("thr"),
                    filter_thumbnails=True,
                    log_score_dist=(i == 0),
                )

                t0 = time.perf_counter()
                response = run_inference(model, processor, s["video_path"], s["question_raw"], device)
                t1 = time.perf_counter()

                predicted = extract_answer(response)
                gt = s["answer"]
                is_ok = (predicted == gt)
                correct += int(is_ok)
                total += 1
                latencies.append(t1 - t0)

                rows.append({
                    "question_id": s["question_id"],
                    "gt": gt, "predicted": predicted, "correct": is_ok,
                    "latency_s": t1 - t0, "response": response,
                })
                if (i + 1) % 10 == 0 or i == 0 or i == len(samples) - 1:
                    print(f"  [{i+1}/{len(samples)}] q={s['question_id']}  gt={gt}  pred={predicted}  "
                          f"{'OK' if is_ok else 'X '}  lat={t1-t0:.1f}s")
            except Exception as e:
                print(f"  sample {i} failed: {e}")
                continue

        acc = correct / max(total, 1)
        avg_lat = sum(latencies) / max(len(latencies), 1)
        result = {
            "config": cfg, "accuracy": acc, "correct": correct, "total": total,
            "avg_latency_s": avg_lat, "rows": rows,
        }
        all_results[cfg["name"]] = result
        per_sample_all.extend([{"config": cfg["name"], **r} for r in rows])
        print(f"\n  => {cfg['name']}: acc={acc:.3f} ({correct}/{total})  avg_lat={avg_lat:.2f}s")

        with open(os.path.join(args.output_dir, "hlvid_subset.json"), "w") as f:
            json.dump({"summary": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                                   for k, v in all_results.items()},
                       "per_sample": per_sample_all,
                       "n_samples": len(samples)}, f, indent=2)

    print(f"\n{'=' * 60}\n[SUMMARY] HLVid household mild-keep sweep (n={len(samples)})\n{'=' * 60}")
    for name, r in all_results.items():
        print(f"  {name:<25}  acc={r['accuracy']:.3f} ({r['correct']}/{r['total']})  "
              f"lat={r['avg_latency_s']:.2f}s")
    print(f"[save] {args.output_dir}/hlvid_subset.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="nvidia/NVILA-8B-HD-Video")
    p.add_argument("--video_dir", default="hlvid_videos/extracted_household/videos")
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--num_frames_thumbnail", type=int, default=16)
    p.add_argument("--max_tiles", type=int, default=4)
    p.add_argument("--gazing_ratio", type=float, default=0.2)
    p.add_argument("--gazing_ratio_thumbnail", type=float, default=0.75)
    p.add_argument("--ckpt", default="results/bighead_warmrestart/best_bighead_student.pt")
    p.add_argument("--head_type", default="bighead")
    p.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/autogaze_aware_filter_train/mild_keep_sweep")
    args = p.parse_args()
    main(args)
