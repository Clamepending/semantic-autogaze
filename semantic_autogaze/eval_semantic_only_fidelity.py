"""r/semantic-only-hlvid-baseline cycle 1: necessary-condition test for the
filter-as-replacement direction.

Three configs on HLVid household n=122 at semantic_keep_ratio matched to the
existing rank-1 vanilla AutoGaze (scale 0.70 → keep_ratio_tile = 0.14):

  A) sem_match     mode=semantic_only, keep=0.14, query=matched HLVid stem
  B) sem_shuffled  mode=semantic_only, keep=0.14, query=shuffled-other-question stem
  C) sem_random    mode=semantic_only, keep=0.14, random_scoring=True (uniform random)

Decision tree:
  * sem_match acc < vanilla AutoGaze 53/122 by > 1 sample
      → filter-as-replacement falsified at current BigHead checkpoint
  * sem_match acc ≥ 53 within tie band AND sem_match - sem_shuffled > +1
      → text-conditioning REAL and filter at least matches AutoGaze at matched K
        → cycle 2: full Pareto frontier
  * sem_match ≈ sem_shuffled (within ±1 sample)
      → filter is NOT text-conditioned; reading scene saliency only
        → triggers teacher-swap or AutoGaze-free-feature follow-up
  * sem_match >> sem_random AND sem_shuffled ≈ sem_random
      → text-conditioning is real and dominates random; admit + Pareto follow-up

Reuses eval_hlvid_subset.py infra — load_subset, run_inference, etc. — but
adds a shuffled-query map.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python -m semantic_autogaze.eval_semantic_only_fidelity \\
      --device cuda:0 \\
      --output_dir results/semantic_only_hlvid_baseline
"""
from __future__ import annotations
import os, json, time, random, argparse
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper
from semantic_autogaze.eval_hlvid_subset import load_subset, run_inference
from semantic_autogaze.eval_vlm_benchmark import (
    extract_answer,
    patch_processor_with_semantic_filter,
)


def main(args):
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- NVILA + processor ---
    print("[setup] Loading NVILA-8B-HD-Video (4-bit NF4)...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        num_video_frames=32,
        num_video_frames_thumbnail=16,
        max_tiles_video=4,
        gazing_ratio_tile=0.20,        # vanilla AutoGaze budget at the processor level
        gazing_ratio_thumbnail=0.75,
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
        args.model_path, trust_remote_code=True,
        quantization_config=bnb_config,
        device_map=args.device,
        max_batch_size_siglip=8,
    ).eval()
    torch.cuda.empty_cache()

    # --- BigHead semantic-filter wrapper ---
    print(f"[setup] Loading SemanticAutoGazeWrapper (BigHead {args.head_ckpt})...")
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name="nvidia/AutoGaze",
        head_ckpt=args.head_ckpt,
        head_type="bighead",
        device=str(device),
        grid_size=14,
        num_frames=32,
    )
    wrapper.eval()

    # --- CLIP text encoder for question embedding ---
    print("[setup] Loading CLIP ViT-B/16 text encoder...")
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    clip_tok = open_clip.get_tokenizer("ViT-B-16")

    # --- HLVid samples ---
    samples = load_subset(args.video_dir, query_mode=args.query_mode)
    print(f"[data] {len(samples)} QA pairs")
    if not samples:
        return

    # --- Shuffle map: each sample gets a different sample's semantic_query ---
    # Deterministic permutation with a fixed seed so the shuffle is reproducible.
    rng = random.Random(args.seed)
    perm = list(range(len(samples)))
    while True:
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(len(samples))):
            break  # ensure no fixed points (no sample sees its own query)
    shuffled_query = {samples[i]["question_id"]: samples[perm[i]]["semantic_query"]
                      for i in range(len(samples))}

    print(f"[shuffle] sample[0] real_q='{samples[0]['semantic_query'][:60]}'")
    print(f"          sample[0] shuffled_q='{shuffled_query[samples[0]['question_id']][:60]}'")

    configs = [
        # cycle 1 — sub-select from AutoGaze's gazed-set (filter-on-top, NOT replacement)
        {"name": "sem_match",     "mode": "semantic_only", "keep_ratio": 0.14,
         "query_mode": "matched",  "random_scoring": False, "thr": None,
         "bypass_autogaze": False},
        {"name": "sem_shuffled",  "mode": "semantic_only", "keep_ratio": 0.14,
         "query_mode": "shuffled", "random_scoring": False, "thr": None,
         "bypass_autogaze": False},
        {"name": "sem_random",    "mode": "semantic_only", "keep_ratio": 0.14,
         "query_mode": "matched",  "random_scoring": True,  "thr": None,
         "bypass_autogaze": False},
        # cycle 2 — TRUE filter-as-replacement: full 14×14 grid, BigHead picks top-K=27
        # per frame (matches K=27 of vanilla rank-1 scale 0.70 on the SigLIP grid).
        {"name": "true_match",    "mode": "semantic_only", "keep_ratio": 0.14,
         "query_mode": "matched",  "random_scoring": False, "thr": None,
         "bypass_autogaze": True},
        {"name": "true_shuffled", "mode": "semantic_only", "keep_ratio": 0.14,
         "query_mode": "shuffled", "random_scoring": False, "thr": None,
         "bypass_autogaze": True},
        {"name": "true_random",   "mode": "semantic_only", "keep_ratio": 0.14,
         "query_mode": "matched",  "random_scoring": True,  "thr": None,
         "bypass_autogaze": True},
    ]

    all_results = {}
    per_sample_all = []

    only_set = set(args.only.split(",")) if args.only else None
    if only_set:
        configs = [c for c in configs if c["name"] in only_set]
        print(f"[filter] running only: {[c['name'] for c in configs]}")

    for cfg in configs:
        print(f"\n{'=' * 60}\n[config] {cfg['name']}\n{'=' * 60}")
        if hasattr(processor, "_original_get_gazing"):
            processor._get_gazing_info_from_videos = processor._original_get_gazing
        else:
            processor._original_get_gazing = processor._get_gazing_info_from_videos

        correct, total = 0, 0
        latencies = []
        rows = []

        for i, s in enumerate(samples):
            try:
                if cfg["query_mode"] == "matched":
                    q = s["semantic_query"]
                else:
                    q = shuffled_query[s["question_id"]]

                patch_processor_with_semantic_filter(
                    processor, wrapper, clip_model, clip_tok,
                    mode=cfg["mode"],
                    semantic_keep_ratio=cfg["keep_ratio"],
                    query_text=q,
                    device=str(device),
                    score_threshold=cfg.get("thr"),
                    filter_thumbnails=True,
                    log_score_dist=(i == 0),
                    random_scoring=cfg["random_scoring"],
                    bypass_autogaze_selection=cfg.get("bypass_autogaze", False),
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
                    "query_used": q,
                })
                if (i + 1) % 10 == 0 or i == 0 or i == len(samples) - 1:
                    print(f"  [{i+1}/{len(samples)}] q={s['question_id']} gt={gt} pred={predicted} "
                          f"{'✓' if is_ok else '✗'} lat={t1-t0:.1f}s")

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

    print(f"\n{'=' * 60}\n[SUMMARY]\n{'=' * 60}")
    for name, r in all_results.items():
        print(f"  {name:<15s}  acc={r['accuracy']:.3f} ({r['correct']}/{r['total']})  "
              f"avg_lat={r['avg_latency_s']:.2f}s")
    print(f"[save] {args.output_dir}/hlvid_subset.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="nvidia/NVILA-8B-HD-Video")
    p.add_argument("--head_ckpt", default="results/bighead/best_bighead.pt")
    p.add_argument("--video_dir", default="hlvid_videos/extracted_household/videos")
    p.add_argument("--query_mode", default="stem",
                   help="how to derive semantic_query from each sample (stem|np|hybrid)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/semantic_only_hlvid_baseline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", default=None,
                   help="comma-separated config names to run (skip the others). e.g. true_match,true_shuffled,true_random")
    args = p.parse_args()
    main(args)
