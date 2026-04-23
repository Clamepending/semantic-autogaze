"""
r/filter-vs-random-baseline — uniform-random patch scoring control for
r/autogaze-aware-filter-train (2026-04-23 falsified).

Question: does uniform-random selection at Intersect 50 % also score ~0.27 on
the HLVid household subset (n=122)?
  - If YES (|paired-flip| vs CLIP ≤ 2): CLIP adds zero signal, pre-empts
    teacher-retrain-siglip2 — no teacher improves on random at this keep ratio.
  - If random scores ≤ 0.22 (much worse): CLIP is anti-correlated, SigLIP-2
    retrain is worth the wall time.

Re-uses `patch_processor_with_semantic_filter(..., random_scoring=True)`:
skips the BigHead `get_scores` forward and uses `torch.rand` of the same shape,
so AutoGaze's gazed positions are re-ranked by uniform noise instead of CLIP
text-relevance, then top-50 % (or top-75 %) are kept — identical downstream
shapes as r/autogaze-aware-filter-train.

Configs mirror r/autogaze-aware-filter-train minimally (matched-count pairs):
  - Intersect 50 %  — matches AutoGaze scale 0.5 (ratio-exact pair)
  - Intersect 75 %  — matches vanilla (control; filter keeps 75 % of gazed)

Vanilla AutoGaze-only is NOT rerun (NVILA deterministic; 51/122 reproduced
across r/hlvid-household-expand, r/filter-token-count-ablation scale-1.0,
r/predecoder-bighead-full-stack AutoGaze-only, r/autogaze-aware-filter-train
smoke). Cross-reference that 51/122 for paired-flip analysis.

Usage:
    CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      HF_MODULES_CACHE=/tmp/hf_modules \\
      /home/ogata/miniconda3/envs/hunter310/bin/python -u -m \\
      semantic_autogaze.eval_hlvid_random_baseline --device cuda:0
"""

import argparse, json, os, random, time
import torch
import open_clip
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper
from semantic_autogaze.eval_vlm_benchmark import (
    patch_processor_with_semantic_filter,
    extract_answer,
)
from semantic_autogaze.eval_hlvid_subset import load_subset, run_inference


def main(args):
    device = torch.device(args.device)
    random.seed(42)
    torch.manual_seed(42)
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

    print("[setup] Loading CLIP (needed for wrapper signature; not called under random_scoring)...")
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()

    print("[setup] Loading SemanticAutoGazeWrapper (AutoGaze backbone; BigHead unused under random_scoring)...")
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )
    torch.cuda.empty_cache()
    free_mb = torch.cuda.mem_get_info(device)[0] / 1024**2
    print(f"[setup] All models loaded. Free: {free_mb:.0f} MiB")

    samples = load_subset(args.video_dir, query_mode=args.query_mode)
    print(f"[data] HLVid subset: {len(samples)} QA pairs on {len(set(s['video_path'] for s in samples))} videos  (query_mode={args.query_mode})")

    if len(samples) == 0:
        print("[warn] No HLVid samples matched downloaded videos. Exiting.")
        return

    # Parse comma-separated keep ratios. Default covers the subselection-insensitivity
    # sweep (r/subselection-insensitivity-probe): 90 %, 25 %, 10 % — the three untested
    # points across the vanilla (100 %) → attractor (50–75 %) → collapse (< 10 %) curve.
    keep_ratios = [float(x) for x in args.keep_ratios.split(",")]
    configs = [
        {"name": f"Intersect {int(r * 100)}% random", "mode": "intersect",
         "keep_ratio": r, "thr": None}
        for r in keep_ratios
    ]

    all_results = {}
    per_sample_all = []

    for cfg in configs:
        print(f"\n{'=' * 60}\n[config] {cfg['name']}\n{'=' * 60}")
        torch.manual_seed(42)  # reset random per config so runs compare cleanly

        if hasattr(processor, "_original_get_gazing"):
            processor._get_gazing_info_from_videos = processor._original_get_gazing
        else:
            processor._original_get_gazing = processor._get_gazing_info_from_videos

        correct, total = 0, 0
        latencies = []
        rows = []

        for i, s in enumerate(samples):
            try:
                patch_processor_with_semantic_filter(
                    processor, wrapper, clip_model, clip_tokenizer,
                    mode=cfg["mode"], semantic_keep_ratio=cfg["keep_ratio"],
                    query_text=s["semantic_query"], device=str(device),
                    score_threshold=cfg.get("thr"),
                    filter_thumbnails=True,
                    log_score_dist=(i == 0),
                    random_scoring=True,
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
                print(f"  [{i+1}/{len(samples)}] q={s['question_id']}  gt={gt}  pred={predicted}  "
                      f"{'OK' if is_ok else 'X'}  lat={t1-t0:.1f}s")

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

    print(f"\n{'=' * 60}\n[SUMMARY] HLVid household random-baseline (n={len(samples)})\n{'=' * 60}")
    print(f"  Reference: r/autogaze-aware-filter-train Intersect 50% CLIP filter = 33/122 = 0.270")
    print(f"  Reference: r/autogaze-aware-filter-train Intersect 75% CLIP filter = 33/122 = 0.270")
    print(f"  Reference: vanilla AutoGaze only = 51/122 = 0.418 (NVILA deterministic across 4+ runs)")
    for name, r in all_results.items():
        print(f"  {name:<28}  acc={r['accuracy']:.3f} ({r['correct']}/{r['total']})  "
              f"lat={r['avg_latency_s']:.2f}s")

    with open(os.path.join(args.output_dir, "hlvid_subset.json"), "w") as f:
        json.dump({"summary": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                               for k, v in all_results.items()},
                   "per_sample": per_sample_all,
                   "n_samples": len(samples)}, f, indent=2)
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
    p.add_argument("--output_dir", default="results/filter_vs_random_baseline")
    p.add_argument("--keep_ratios", default="0.5,0.75",
                   help="Comma-separated keep ratios for Intersect-mode random sweep. "
                        "Default 0.5,0.75 reproduces r/filter-vs-random-baseline. "
                        "For r/subselection-insensitivity-probe use 0.9,0.25,0.1.")
    p.add_argument("--query_mode", default="stem", choices=["stem", "np", "hybrid"])
    args = p.parse_args()
    main(args)
