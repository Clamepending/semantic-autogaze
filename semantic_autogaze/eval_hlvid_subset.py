"""
HLVid subset eval: does embedding the question as a semantic query give
comparable accuracy at a significant token cut?

Runs NVILA-8B-HD-Video (4-bit) on the 11 QA pairs from HLVid test that map
to the downloaded shard-1 videos (clip_av_video_0_000..003.mp4), under four
configs:

  1. AutoGaze only (baseline, no semantic filter)
  2. Intersect 50% with question as query
  3. Intersect 30% with question as query
  4. Semantic only 30% with question as query

For each config we report VQA accuracy and per-sample wall-clock latency.

The subset is small (n=11) but the purpose is feasibility + directional
evidence, not a publication-grade number. Shard 1 contains 4 videos totalling
7.8 GB (4 K 5-minute clips).

Usage:
    HF_MODULES_CACHE=/tmp/hf_modules CUDA_VISIBLE_DEVICES=5 python3 \\
      -m semantic_autogaze.eval_hlvid_subset \\
      --device cuda:0 --output_dir results/hlvid_subset
"""

import os, re, json, time, random, argparse
import pandas as pd
import torch
import torch.nn.functional as F
import open_clip
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper
from semantic_autogaze.eval_vlm_benchmark import (
    get_clip_text_embedding,
    patch_processor_with_semantic_filter,
    extract_answer,
)


PARQUET_PATH = "/home/ogata/.cache/huggingface/hub/datasets--bfshi--HLVid/snapshots/4956b601aec0bb9d455bb8f57552f78cbd3f0338/data/test-00000-of-00001.parquet"


def _extract_noun_phrase(stem: str) -> str:
    """Recover the subject NP from an HLVid-style stem by grabbing the first
    `on the X` target; fall back to stripping the wh-prefix."""
    m = re.search(r"\bon\s+(the\s+[^?,.]+)", stem, flags=re.I)
    if m:
        np_ = m.group(1).strip()
        np_ = re.sub(r"\s+on\s+the\s+(right|left|top|bottom|center|middle)$", "", np_, flags=re.I)
        return np_.strip(" ?.,")
    m2 = re.sub(r"^what\s+\S+\s+(is|are|does|do|say)\s+", "", stem, flags=re.I)
    return m2.strip(" ?.,")


def parse_question_and_choices(q_text, query_mode: str = "stem"):
    """HLVid's 'question' field contains stem + 4 inline options + instruction.

    Pattern per sample:
      "<stem>\\nA. ...\\nB. ...\\nC. ...\\nD. ...\\nPlease answer directly..."
    Return (stem, [A,B,C,D], semantic_query).

    query_mode:
      - "stem":   pass the whole stem (default, baseline)
      - "np":     noun-phrase extraction ("the white signboard")
      - "hybrid": "text on <NP>" — keeps the OCR signal but anchors to subject
    """
    m = re.split(r"\n(?=A\.)", q_text, maxsplit=1)
    if len(m) != 2:
        return q_text, [], q_text
    stem = m[0].strip()
    rest = re.split(r"\nPlease answer directly", m[1])[0]
    choices = re.findall(r"([A-D])\.\s*(.*?)(?=\n[A-D]\.|$)", rest, flags=re.DOTALL)
    choice_lines = [f"{L}. {txt.strip()}" for L, txt in choices]
    if query_mode == "np":
        sem_q = _extract_noun_phrase(stem)
    elif query_mode == "hybrid":
        sem_q = f"text on {_extract_noun_phrase(stem)}"
    else:
        sem_q = stem
    return stem, choice_lines, sem_q


def load_subset(video_dir, parquet_path=PARQUET_PATH, query_mode: str = "stem"):
    df = pd.read_parquet(parquet_path)
    samples = []
    for _, r in df.iterrows():
        vp = r["video_path"]
        full = os.path.join(video_dir, vp)
        if not os.path.exists(full):
            continue
        stem, choices, sem_q = parse_question_and_choices(r["question"], query_mode=query_mode)
        samples.append({
            "question_id": int(r["question_id"]),
            "category": r["category"],
            "video_path": full,
            "question_raw": r["question"],
            "question_stem": stem,
            "choices": choices,
            "semantic_query": sem_q,
            "answer": r["answer"],
        })
    return samples


def run_inference(model, processor, video_path, question_raw, device):
    """Feed the full HLVid 'question' text (already has choices + instruction)."""
    video_token = processor.tokenizer.video_token
    inputs = processor(
        text=f"{video_token}\n\n{question_raw}",
        videos=video_path,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=16, do_sample=False,
        )
    response = processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0].strip()
    return response


def main(args):
    device = torch.device(args.device)
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # Memory discipline: load the 8B model FIRST while we have the most
    # contiguous free memory; CLIP + wrapper (small) go on top afterwards.
    # Without this, the 4-bit NVILA checkpoint OOMs on the final shard when
    # ~1-2 GiB is already occupied by CLIP/AutoGaze.
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

    # ---- Load HLVid subset (only questions whose videos are downloaded) ----
    samples = load_subset(args.video_dir, query_mode=args.query_mode)
    print(f"[data] HLVid subset: {len(samples)} QA pairs on {len(set(s['video_path'] for s in samples))} videos  (query_mode={args.query_mode})")
    for s in samples[:3]:
        print(f"  q[{s['question_id']}] {os.path.basename(s['video_path'])}: {s['question_stem'][:80]}")
        print(f"      sem_query: {s['semantic_query']!r}")

    if len(samples) == 0:
        print("[warn] No HLVid samples matched downloaded videos. Exiting.")
        return

    configs = [
        {"name": "AutoGaze only",     "mode": "gaze_only",     "keep_ratio": 1.0,  "thr": None},
        {"name": "Intersect 30%",     "mode": "intersect",     "keep_ratio": 0.3,  "thr": None},
        {"name": "Intersect 10%",     "mode": "intersect",     "keep_ratio": 0.1,  "thr": None},
        {"name": "Semantic 10%",      "mode": "semantic_only", "keep_ratio": 0.1,  "thr": None},
        {"name": "Semantic 2%",       "mode": "semantic_only", "keep_ratio": 0.02, "thr": None},
        {"name": "Thresh 0.005",      "mode": "semantic_only", "keep_ratio": 1.0,  "thr": 0.005},
        {"name": "Thresh 0.01",       "mode": "semantic_only", "keep_ratio": 1.0,  "thr": 0.01},
        {"name": "Thresh 0.02",       "mode": "semantic_only", "keep_ratio": 1.0,  "thr": 0.02},
        {"name": "Thresh 0.05",       "mode": "semantic_only", "keep_ratio": 1.0,  "thr": 0.05},
    ]

    all_results = {}
    per_sample_all = []

    for cfg in configs:
        print(f"\n{'=' * 60}\n[config] {cfg['name']}\n{'=' * 60}")
        # Reset processor
        if hasattr(processor, "_original_get_gazing"):
            processor._get_gazing_info_from_videos = processor._original_get_gazing
        else:
            processor._original_get_gazing = processor._get_gazing_info_from_videos

        correct, total = 0, 0
        latencies = []
        rows = []

        for i, s in enumerate(samples):
            try:
                if cfg["mode"] != "gaze_only":
                    # Use stem as query — avoids injecting the choices into the embedding
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
                print(f"  [{i+1}/{len(samples)}] q={s['question_id']}  gt={gt}  pred={predicted}  "
                      f"{'✓' if is_ok else '✗'}  lat={t1-t0:.1f}s")

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

    # ---- Summary ----
    print(f"\n{'=' * 60}\n[SUMMARY] HLVid subset (n={len(samples)})\n{'=' * 60}")
    base_acc = all_results["AutoGaze only"]["accuracy"]
    base_lat = all_results["AutoGaze only"]["avg_latency_s"]
    for name, r in all_results.items():
        d_acc = r["accuracy"] - base_acc
        d_lat = r["avg_latency_s"] - base_lat
        spd = base_lat / max(r["avg_latency_s"], 1e-6)
        print(f"  {name:<25}  acc={r['accuracy']:.3f} ({r['correct']}/{r['total']})  "
              f"Δacc={d_acc:+.3f}  lat={r['avg_latency_s']:.2f}s  (speedup {spd:.2f}×)")

    with open(os.path.join(args.output_dir, "hlvid_subset.json"), "w") as f:
        json.dump({"summary": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                               for k, v in all_results.items()},
                   "per_sample": per_sample_all,
                   "n_samples": len(samples)}, f, indent=2)
    print(f"[save] {args.output_dir}/hlvid_subset.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="nvidia/NVILA-8B-HD-Video")
    p.add_argument("--video_dir", default="hlvid_videos/extracted/videos")
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--num_frames_thumbnail", type=int, default=16)
    p.add_argument("--max_tiles", type=int, default=4)
    p.add_argument("--gazing_ratio", type=float, default=0.2)
    p.add_argument("--gazing_ratio_thumbnail", type=float, default=0.75,
                   help="Must be <1.0 to enable AutoGaze on thumbnails (so we can semantic-filter them).")
    p.add_argument("--ckpt", default="results/bighead_warmrestart/best_bighead_student.pt")
    p.add_argument("--head_type", default="bighead")
    p.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/hlvid_subset")
    p.add_argument("--query_mode", default="stem", choices=["stem", "np", "hybrid"])
    args = p.parse_args()
    main(args)
