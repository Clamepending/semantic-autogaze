"""
r/nvila-no-gaze-baseline — raw NVILA (no AutoGaze, no filter) on HLVid household n=122.

Question: what does NVILA-8B-HD-Video score on HLVid household when no AutoGaze
token reduction is applied? Compare to vanilla AutoGaze = 51/122 = 0.418.

Mechanism: NVILA processor bypasses AutoGaze entirely when gazing_ratio_tile == 1.0
AND gazing_ratio_thumbnail == 1.0 (see `_should_gaze_all_patches` in
processing_nvila.py line 775-788 — explicit comment "gazing_ratio == 1 (keep 100%)"
triggers skip_tiles_gaze=True, bypassing _get_gazing_info_from_videos).

No code changes vs eval_hlvid_subset.py — just swap the two gazing_ratio args to 1.0
and skip the wrapper/clip setup (unused under skip_gaze).

Prior expectations (from the review #2 hypothesis):
  - 60% "raw NVILA > vanilla AutoGaze 51/122" — AutoGaze actively hurts, solution
    is to not use it. Filter-on-top direction becomes moot (there's nothing to
    filter on top of).
  - 30% "raw NVILA ≈ vanilla" — AutoGaze is neutral; sub-selection-insensitivity
    insight is wrong about AutoGaze being the ceiling. Real ceiling is NVILA itself.
  - 10% "raw NVILA < vanilla" — AutoGaze's selection adds value; the insight
    correctly identifies it as the ceiling; unfreeze/replace AutoGaze is the path.

Usage:
    CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      HF_MODULES_CACHE=/tmp/hf_modules \\
      /home/ogata/miniconda3/envs/hunter310/bin/python -u -m \\
      semantic_autogaze.eval_hlvid_no_gaze --device cuda:0
"""

import argparse, json, os, random, time
import torch
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from semantic_autogaze.eval_vlm_benchmark import extract_answer
from semantic_autogaze.eval_hlvid_subset import load_subset, run_inference


def main(args):
    device = torch.device(args.device)
    random.seed(42)
    torch.manual_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    gaze_desc = f"gazing_ratio_tile={args.gazing_ratio_tile}, gazing_ratio_thumbnail={args.gazing_ratio_thumbnail}"
    bypass = (args.gazing_ratio_tile == 1.0 and args.gazing_ratio_thumbnail == 1.0)
    print(f"[setup] Loading NVILA-8B-HD-Video (4-bit NF4) with {gaze_desc} "
          f"(AutoGaze {'bypassed' if bypass else 'active'}), max_tiles={args.max_tiles}...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        num_video_frames=args.num_frames,
        num_video_frames_thumbnail=args.num_frames_thumbnail,
        max_tiles_video=args.max_tiles,
        gazing_ratio_tile=args.gazing_ratio_tile,
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
    print(f"[setup] NVILA loaded. Free: {free_mb:.0f} MiB")

    samples = load_subset(args.video_dir, query_mode=args.query_mode)
    print(f"[data] HLVid subset: {len(samples)} QA pairs on {len(set(s['video_path'] for s in samples))} videos  (query_mode={args.query_mode})")

    if len(samples) == 0:
        print("[warn] No HLVid samples matched downloaded videos. Exiting.")
        return

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
            print(f"  [{i+1}/{len(samples)}] q={s['question_id']}  gt={gt}  pred={predicted}  "
                  f"{'OK' if is_ok else 'X'}  lat={t1-t0:.1f}s")

        except Exception as e:
            print(f"  sample {i} failed: {type(e).__name__}: {e}")
            continue

    acc = correct / max(total, 1)
    avg_lat = sum(latencies) / max(len(latencies), 1)
    print(f"\n{'=' * 60}\n[SUMMARY] NVILA no-gaze on HLVid household (n={len(samples)})\n{'=' * 60}")
    print(f"  Reference: vanilla AutoGaze = 51/122 = 0.418 (NVILA deterministic across 4+ runs)")
    print(f"  Raw NVILA (no AutoGaze):  acc={acc:.3f} ({correct}/{total})  avg_lat={avg_lat:.2f}s")

    cfg_name = f"gaze_t={args.gazing_ratio_tile},th={args.gazing_ratio_thumbnail},mt={args.max_tiles}"
    result = {
        "config": {"name": cfg_name,
                   "gazing_ratio_tile": args.gazing_ratio_tile,
                   "gazing_ratio_thumbnail": args.gazing_ratio_thumbnail,
                   "max_tiles": args.max_tiles,
                   "num_frames": args.num_frames,
                   "num_frames_thumbnail": args.num_frames_thumbnail},
        "accuracy": acc, "correct": correct, "total": total,
        "avg_latency_s": avg_lat,
    }
    all_results = {cfg_name: result}
    per_sample = [{"config": cfg_name, **r} for r in rows]
    with open(os.path.join(args.output_dir, "hlvid_subset.json"), "w") as f:
        json.dump({"summary": all_results, "per_sample": per_sample,
                   "n_samples": len(samples)}, f, indent=2)
    print(f"[save] {args.output_dir}/hlvid_subset.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="nvidia/NVILA-8B-HD-Video")
    p.add_argument("--video_dir", default="hlvid_videos/extracted_household/videos")
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--num_frames_thumbnail", type=int, default=16)
    p.add_argument("--max_tiles", type=int, default=4)
    p.add_argument("--gazing_ratio_tile", type=float, default=1.0,
                   help="1.0 = bypass AutoGaze for tile patches; 0.20 = vanilla AutoGaze.")
    p.add_argument("--gazing_ratio_thumbnail", type=float, default=1.0,
                   help="1.0 = bypass AutoGaze for thumbnail patches; 0.75 = vanilla AutoGaze.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/nvila_no_gaze_baseline")
    p.add_argument("--query_mode", default="stem", choices=["stem", "np", "hybrid"])
    args = p.parse_args()
    main(args)
