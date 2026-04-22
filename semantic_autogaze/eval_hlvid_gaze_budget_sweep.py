"""
AutoGaze-only gaze-budget sweep on HLVid household n=122.

Motivation (from r/predecoder-bighead-full-stack, 2026-04-22):
pre-decoder BigHead (2.4x weaker teacher-mass retention than post-decoder)
produces 92.6 % identical NVILA answers to the post-decoder BigHead. Head
identity is close to asymptotically irrelevant. Filter configs regress by
0.11-0.18 on household vs vanilla AutoGaze-only 0.418.

This move distinguishes two remaining mechanism hypotheses:

  (a) "token COUNT is the binding constraint" - NVILA benefits from full
      coverage regardless of which patches are kept. Reducing the number
      of kept patches (via any mechanism - filter or reduced gazing_ratio)
      hurts accuracy proportionally to count reduction.

  (b) "token CHOICE matters" - the filter's selection is meaningfully
      worse than AutoGaze's reconstruction-salient selection at matched
      counts. Reducing AutoGaze's own budget at matched counts would hurt
      LESS than filtering does.

Design: sweep `(gazing_ratio_tile, gazing_ratio_thumbnail)` over 5 budgets
under AutoGaze-only mode (no semantic filter), report accuracy on the
same 122-QA household subset.

Budgets sweep scales `(tile, thumb)` by factors {1.0, 0.5, 0.3, 0.1, 0.05}
of the project vanilla `(0.2, 0.75)`, matching the range that filter
configs effectively achieve (Intersect 10% ~ 0.1 scale; Thresh 0.005 is
closer to 0.7-0.9 scale per score-distribution debug on 2026-04-22).

Usage:
    CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      python -m semantic_autogaze.eval_hlvid_gaze_budget_sweep \\
        --device cuda:0 \\
        --video_dir hlvid_videos/extracted_household/videos \\
        --output_dir results/filter_token_count_ablation/gaze_budget_sweep
"""

import os, json, time, random, argparse
import torch
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from semantic_autogaze.eval_hlvid_subset import load_subset, run_inference
from semantic_autogaze.eval_vlm_benchmark import extract_answer


BUDGETS = [
    {"name": "vanilla tile=0.20 thumb=0.75", "tile": 0.20, "thumb": 0.75},
    {"name": "scaled 0.5 tile=0.10 thumb=0.375", "tile": 0.10, "thumb": 0.375},
    {"name": "scaled 0.3 tile=0.06 thumb=0.225", "tile": 0.06, "thumb": 0.225},
    {"name": "scaled 0.1 tile=0.02 thumb=0.075", "tile": 0.02, "thumb": 0.075},
    {"name": "scaled 0.05 tile=0.01 thumb=0.0375", "tile": 0.01, "thumb": 0.0375},
]


def main(args):
    device = torch.device(args.device)
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print("[setup] Loading NVILA-8B-HD-Video (4-bit NF4)...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        num_video_frames=args.num_frames,
        num_video_frames_thumbnail=args.num_frames_thumbnail,
        max_tiles_video=args.max_tiles,
        gazing_ratio_tile=BUDGETS[0]["tile"],
        gazing_ratio_thumbnail=BUDGETS[0]["thumb"],
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

    samples = load_subset(args.video_dir, query_mode="stem")
    print(f"[data] HLVid subset: {len(samples)} QA pairs on {len(set(s['video_path'] for s in samples))} videos")
    if len(samples) == 0:
        print("[warn] No samples found. Exiting.")
        return

    all_results = {}
    per_sample_all = []

    for cfg in BUDGETS:
        print(f"\n{'=' * 60}\n[budget] {cfg['name']}\n{'=' * 60}")
        processor.gazing_ratio_tile = cfg["tile"]
        processor.gazing_ratio_thumbnail = cfg["thumb"]

        correct, total = 0, 0
        latencies = []
        rows = []
        for i, s in enumerate(samples):
            try:
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

    print(f"\n{'=' * 60}\n[SUMMARY] HLVid household gaze-budget sweep (n={len(samples)})\n{'=' * 60}")
    base_acc = all_results[BUDGETS[0]["name"]]["accuracy"]
    base_lat = all_results[BUDGETS[0]["name"]]["avg_latency_s"]
    for name, r in all_results.items():
        d_acc = r["accuracy"] - base_acc
        d_lat = r["avg_latency_s"] - base_lat
        spd = base_lat / max(r["avg_latency_s"], 1e-6)
        print(f"  {name:<45}  acc={r['accuracy']:.3f} ({r['correct']}/{r['total']})  "
              f"d_acc={d_acc:+.3f}  lat={r['avg_latency_s']:.2f}s  (speedup {spd:.2f}x)")

    print(f"[save] {args.output_dir}/hlvid_subset.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="nvidia/NVILA-8B-HD-Video")
    p.add_argument("--video_dir", default="hlvid_videos/extracted_household/videos")
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--num_frames_thumbnail", type=int, default=16)
    p.add_argument("--max_tiles", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/filter_token_count_ablation/gaze_budget_sweep")
    args = p.parse_args()
    main(args)
