"""r/video-mme-ours-v1-pilot cycle 1 — Tier-3-equivalent on Video-MME.

Mirrors eval_vqa_ours_v1_bypass.py but uses Video-MME (lmms-lab/Video-MME)
instead of HLVid household. 4 configs:
  1. vanilla       - standard NVILA + AutoGaze (no bypass) — baseline
  2. ours_v1_match - bypass + Ours v1 with matched query
  3. ours_v1_shuf  - bypass + Ours v1 with shuffled query
  4. ours_v1_rand  - bypass + uniform random patch selection

Decisive gate (vs r/independent-text-scorer-v1 cycle 2 HLVid result of
match=shuf=37/122):
  match-vs-shuf >= +3 paired-flip -> IoU<->VQA gap is HLVid-OCR-specific;
    expand to full Video-MME and pivot project to non-OCR benchmarks.
  match ~~ shuf -> gap is NVILA-decoder-property; close direction.
"""
from __future__ import annotations
import os, sys, json, time, random, gc, argparse
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")

from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper
from semantic_autogaze.eval_vlm_benchmark import (
    patch_processor_with_semantic_filter,
    extract_answer,
)
from semantic_autogaze.eval_vqa_nvila_attn_bypass import deterministic_shuffle
from semantic_autogaze.eval_vqa_ours_v1_bypass import OursV1ScoreProvider, OURS_CKPT
from autogaze.datasets.video_utils import read_video_pyav

import av


N_PATCHES = 14 * 14
GRID = 14


def load_video_mme_subset(parquet_path: str, video_dir: str,
                          n_samples: Optional[int] = None,
                          duration: Optional[str] = None,
                          seed: int = 42) -> List[Dict[str, Any]]:
    """Load Video-MME questions filtered to those whose videos exist locally.

    Returns list of dicts with: question_id, video_id, video_path,
    question_stem (str), question_raw (str with options),
    answer (single letter), choices (list of str), duration.
    """
    df = pd.read_parquet(parquet_path)
    have = {os.path.splitext(f)[0] for f in os.listdir(video_dir) if f.endswith(".mp4")}
    df = df[df["videoID"].isin(have)].copy()
    if duration:
        df = df[df["duration"] == duration].copy()
    if n_samples and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    samples = []
    for _, r in df.iterrows():
        opts = list(r["options"])
        # opts look like ["A. text", "B. text", "C. text", "D. text"]
        question_raw = r["question"] + "\n" + "\n".join(opts) + "\nAnswer with the option's letter from the given choices directly."
        samples.append({
            "question_id": r["question_id"],
            "video_id": r["video_id"],
            "videoID": r["videoID"],
            "video_path": os.path.join(video_dir, f"{r['videoID']}.mp4"),
            "question_stem": r["question"],
            "question_raw": question_raw,
            "answer": r["answer"],
            "choices": opts,
            "duration": r["duration"],
            "task_type": r.get("task_type", ""),
        })
    return samples


def run_inference_video_mme(model, processor, video_path, question_raw,
                            device, num_video_frames=16,
                            num_video_frames_thumbnail=16):
    """Run NVILA inference on a Video-MME sample."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    n = stream.frames or num_video_frames
    # Sample num_video_frames evenly
    indices = np.linspace(0, n - 1, num_video_frames).round().astype(int).tolist()
    frames = read_video_pyav(container=container, indices=indices)
    container.close()

    # Convert to PIL for processor
    from PIL import Image
    pil_frames = [Image.fromarray(f) for f in frames]

    messages = [{"role": "user",
                 "content": [{"type": "video"},
                             {"type": "text", "text": question_raw}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, videos=[pil_frames],
                       return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    response = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)[0]
    return response


def main(args):
    device = torch.device(args.device)
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("r/video-mme-ours-v1-pilot cycle 1 — Tier-3 on Video-MME")
    print("=" * 60)

    # ---- Load Video-MME subset ----
    print(f"\nLoading Video-MME (parquet={args.parquet_path}, "
          f"video_dir={args.video_dir}, n_samples={args.n_samples}, duration={args.duration})...",
          flush=True)
    samples = load_video_mme_subset(args.parquet_path, args.video_dir,
                                    n_samples=args.n_samples,
                                    duration=args.duration, seed=args.seed)
    print(f"Loaded {len(samples)} samples (total {len(set(s['videoID'] for s in samples))} unique videos)",
          flush=True)
    if not samples:
        raise SystemExit("No samples found — check --video_dir and that the parquet matches.")

    # ---- CLIP for plumbing ----
    print("\nLoading CLIP for patch_processor plumbing...", flush=True)
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()

    print("Loading SemanticAutoGazeWrapper...", flush=True)
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt, head_type=args.head_type, device=str(device),
    )
    torch.cuda.empty_cache()

    print("Loading Ours v1 score provider (sharing CLIP)...", flush=True)
    ours_match = OursV1ScoreProvider(args.ours_ckpt, device,
                                     clip_model=clip_model, clip_tok=clip_tokenizer)
    ours_shuf = OursV1ScoreProvider(args.ours_ckpt, device,
                                    clip_model=clip_model, clip_tok=clip_tokenizer)
    torch.cuda.empty_cache()

    print("Loading NVILA-8B-HD-Video...", flush=True)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        autogaze_model_id=args.autogaze_model,
        num_video_frames=args.num_frames,
        num_video_frames_thumbnail=args.num_frames_thumbnail,
        max_tiles_video=args.max_tiles,
        gazing_ratio_tile=args.gazing_ratio,
        gazing_ratio_thumbnail=args.gazing_ratio_thumbnail,
        task_loss_requirement_tile=0.6,
        task_loss_requirement_thumbnail=0.6,
    )
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4")
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, quantization_config=bnb,
        device_map=args.device, max_batch_size_siglip=8,
    )
    model.eval()
    torch.cuda.empty_cache()
    print("Model loaded.", flush=True)

    qids = [s["question_id"] for s in samples]
    shuffled_qids = deterministic_shuffle(qids, seed=42)
    qid_to_question = {s["question_id"]: s["question_stem"] for s in samples}
    qid_to_shufq = {qids[i]: qid_to_question[shuffled_qids[i]] for i in range(len(qids))}

    configs = ["vanilla", "match", "shuf", "rand"]
    all_results = {}

    for cfg in configs:
        print(f"\n{'='*50}\nConfig: {cfg}\n{'='*50}", flush=True)

        if hasattr(processor, "_original_get_gazing"):
            processor._get_gazing_info_from_videos = processor._original_get_gazing
        else:
            processor._original_get_gazing = processor._get_gazing_info_from_videos

        # Per-qid resume
        per_qid_path = os.path.join(args.output_dir, f"per_qid_{cfg}.json")
        if os.path.exists(per_qid_path):
            with open(per_qid_path) as _f:
                saved = json.load(_f)
            per_q = saved.get("per_q", [])
            done_qids = {p["qid"] for p in per_q if p.get("pred", "ERROR") != "ERROR"}
            correct = sum(p["correct"] for p in per_q)
            latencies = [p["wall_s"] for p in per_q if p.get("wall_s", -1) > 0]
            print(f"  [resume] {len(per_q)} done, {correct} correct", flush=True)
        else:
            per_q = []; correct = 0; latencies = []
            done_qids = set()
        total = len(done_qids)

        for i, sample in enumerate(samples):
            qid = sample["question_id"]
            if qid in done_qids:
                continue
            try:
                processor._get_gazing_info_from_videos = processor._original_get_gazing

                if cfg == "vanilla":
                    # Standard AutoGaze; no monkey-patch needed.
                    q_for_score = sample["question_stem"]
                    score_provider = None
                elif cfg == "match":
                    q_for_score = sample["question_stem"]
                    ours_match.set_query_text(q_for_score)
                    score_provider = ours_match
                elif cfg == "shuf":
                    q_for_score = qid_to_shufq[qid]
                    ours_shuf.set_query_text(q_for_score)
                    score_provider = ours_shuf
                else:  # rand
                    q_for_score = sample["question_stem"]
                    score_provider = None

                if cfg in ("match", "shuf"):
                    patch_processor_with_semantic_filter(
                        processor, wrapper, clip_model, clip_tokenizer,
                        mode="intersect",
                        semantic_keep_ratio=args.semantic_keep_ratio,
                        query_text=q_for_score, device=str(device),
                        bypass_autogaze_selection=True,
                        score_provider=score_provider,
                        filter_thumbnails=True,
                    )
                elif cfg == "rand":
                    patch_processor_with_semantic_filter(
                        processor, wrapper, clip_model, clip_tokenizer,
                        mode="intersect",
                        semantic_keep_ratio=args.semantic_keep_ratio,
                        query_text=q_for_score, device=str(device),
                        bypass_autogaze_selection=True, random_scoring=True,
                        filter_thumbnails=True,
                    )
                # vanilla: no patch — use original AutoGaze.

                t0 = time.perf_counter()
                response = run_inference_video_mme(
                    model, processor, sample["video_path"],
                    sample["question_raw"], str(device),
                    num_video_frames=args.num_frames,
                    num_video_frames_thumbnail=args.num_frames_thumbnail,
                )
                wall_s = time.perf_counter() - t0
                pred = extract_answer(response)
                gt = sample["answer"]
                is_correct = (pred == gt)
                if is_correct: correct += 1
                total += 1; latencies.append(wall_s)
                per_q.append({"qid": qid, "gt": gt, "pred": pred,
                              "correct": int(is_correct), "scoring_q": q_for_score,
                              "wall_s": wall_s, "duration": sample["duration"],
                              "task_type": sample["task_type"]})
                if (i + 1) % 5 == 0 or i == 0:
                    print(f"  [{i+1}/{len(samples)}] qid={qid} gt={gt} pred={pred} "
                          f"acc={correct}/{total} avg_lat={sum(latencies)/len(latencies):.2f}s",
                          flush=True)
                with open(per_qid_path, "w") as _f:
                    json.dump({"cfg": cfg, "per_q": per_q,
                               "correct": correct, "total": total}, _f)
                if (i + 1) % 10 == 0:
                    gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                import traceback; tb = traceback.format_exc()
                print(f"  [error] qid={qid}: {type(e).__name__}: {e}", flush=True)
                per_q.append({"qid": qid, "gt": sample["answer"], "pred": "ERROR",
                              "correct": 0, "wall_s": -1, "error": str(e),
                              "traceback": tb, "duration": sample["duration"],
                              "task_type": sample["task_type"]})

        all_results[cfg] = {
            "correct": correct, "total": total,
            "accuracy": correct / max(total, 1),
            "avg_lat_s": sum(latencies) / max(len(latencies), 1) if latencies else 0,
            "per_q": per_q,
        }
        print(f"\n  >>> {cfg}: {correct}/{total} = {correct/max(total,1):.4f}, "
              f"avg_lat={sum(latencies)/max(len(latencies),1):.2f}s")
        with open(os.path.join(args.output_dir, f"partial_{cfg}.json"), "w") as f:
            json.dump(all_results[cfg], f, indent=2)

    print(f"\n{'='*60}\nPaired-flip\n{'='*60}")
    paired = {}
    for a, b in [("match", "shuf"), ("match", "rand"), ("shuf", "rand"),
                 ("match", "vanilla"), ("vanilla", "rand")]:
        if a not in all_results or b not in all_results: continue
        a_pq = {p["qid"]: p["correct"] for p in all_results[a]["per_q"]}
        b_pq = {p["qid"]: p["correct"] for p in all_results[b]["per_q"]}
        common = set(a_pq.keys()) & set(b_pq.keys())
        a_wins = sum(1 for q in common if a_pq[q] > b_pq[q])
        b_wins = sum(1 for q in common if b_pq[q] > a_pq[q])
        net = a_wins - b_wins
        print(f"  {a:<10} vs {b:<10}: a-only={a_wins}  b-only={b_wins}  net={net:+d}  n={len(common)}")
        paired[f"{a}_vs_{b}"] = {"a_wins": a_wins, "b_wins": b_wins, "net": net, "n": len(common)}

    out = {"configs": list(all_results.keys()), "n_samples": len(samples),
           "summary": {k: {kk: vv for kk, vv in v.items() if kk != "per_q"} for k, v in all_results.items()},
           "paired": paired,
           "per_config_per_q": {k: v["per_q"] for k, v in all_results.items()}}
    out_path = os.path.join(args.output_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model_path", default="nvidia/NVILA-8B-HD-Video")
    p.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    p.add_argument("--ckpt", default="results/bighead/best_bighead.pt")
    p.add_argument("--head_type", default="bighead")
    p.add_argument("--ours_ckpt", default=OURS_CKPT)
    p.add_argument("--parquet_path",
                   default="/home/ogata/semantic-autogaze/data/video_mme/videomme/test-00000-of-00001.parquet")
    p.add_argument("--video_dir",
                   default="/home/ogata/semantic-autogaze/data/video_mme/videos_extracted/data")
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--duration", default=None, choices=[None, "short", "medium", "long"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--num_frames_thumbnail", type=int, default=16)
    p.add_argument("--max_tiles", type=int, default=1)
    p.add_argument("--gazing_ratio", type=float, default=0.7)
    p.add_argument("--gazing_ratio_thumbnail", type=float, default=0.75)
    p.add_argument("--semantic_keep_ratio", type=float, default=0.1378)
    p.add_argument("--output_dir", default="results/video_mme_ours_v1_pilot")
    args = p.parse_args()
    main(args)
