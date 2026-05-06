"""r/nvila-attention-distill cycle 1.5b — VQA test using cached NVILA attention as filter.

For each (qid, query_mode in {matched, shuffled, random}):
1. Load cached NVILA attention (matched-Q or shuffled-Q at the same SMALLER config
   we used in cycle 1).
2. Compute top-K visual token positions (K = ratio × num_v).
3. Run NVILA forward with attention_mask=0 for non-top-K visual positions.
4. Decode answer; compare to ground truth.

This bypasses the full bypass_autogaze + score_provider plumbing in
eval_vlm_benchmark.py. Instead, we DIRECTLY mask NVILA's LLM-level visual
tokens using attention_mask. Cleaner for the cycle 1.5b gate because the
cached attention is already at the LLM-level token space (post-AutoGaze,
post-connector).

Decisive gate per result-doc:
- matched_match > matched_shuffled by ≥+3 paired-flip → NVILA's own attention
  has filterable signal at end-to-end VQA → proceed to Phase 2 distillation
  training.
- matched_match ≈ matched_shuffled → distillation cannot help; goal-change
  conversation needed.

Run:
  CUDA_VISIBLE_DEVICES=N python -m semantic_autogaze.eval_vqa_attention_filter \\
    --device cuda:0 --output_dir results/nvila_attn_filter_vqa
"""
from __future__ import annotations
import os, json, time, random, argparse
from typing import Optional
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from semantic_autogaze.eval_vlm_benchmark import extract_answer
from semantic_autogaze.eval_hlvid_subset import load_subset, PARQUET_PATH


VIDEO_TOKEN_ID = 151650


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


def load_cached_attention(cache_dir: str, qid: int) -> Optional[np.ndarray]:
    p = os.path.join(cache_dir, f"qid_{qid:04d}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    a = d["attention"]
    if np.isnan(a).any():
        return None
    return a


def run_inference_with_filter(model, processor, video_path, question_raw, attention_scores,
                               keep_ratio, device):
    """Run NVILA inference but mask out non-top-K visual tokens via attention_mask.

    attention_scores: (num_v,) numpy array of per-visual-token scores. Top-K
                      kept, rest masked. None or np.random.rand(num_v) for random.
    keep_ratio: fraction of visual tokens to keep.
    """
    video_token = processor.tokenizer.video_token
    inputs = processor(
        text=f"{video_token}\n\n{question_raw}",
        videos=video_path,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    input_ids = inputs["input_ids"][0]
    video_token_mask = (input_ids == VIDEO_TOKEN_ID)
    v_positions = torch.where(video_token_mask)[0]
    num_v = len(v_positions)

    if attention_scores is None or len(attention_scores) != num_v:
        # Random fallback
        scores = np.random.rand(num_v)
    else:
        scores = attention_scores

    K = max(1, int(keep_ratio * num_v))
    if K >= num_v:
        keep_idx = set(range(num_v))
    else:
        top_k_idx = np.argsort(scores)[-K:]
        keep_idx = set(int(i) for i in top_k_idx)

    # Build attention_mask: 1 for non-visual + kept-visual; 0 for filtered visual
    attn_mask = inputs.get("attention_mask")
    if attn_mask is None:
        attn_mask = torch.ones_like(input_ids)
    else:
        attn_mask = attn_mask[0].clone()

    for i, pos in enumerate(v_positions):
        if i not in keep_idx:
            attn_mask[pos.item()] = 0

    inputs["attention_mask"] = attn_mask.unsqueeze(0)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )
    response = processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0].strip()
    return response, K, num_v


def main(args):
    device = torch.device(args.device)
    np.random.seed(42)
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[setup] Loading NVILA at extraction config (max_tiles={args.max_tiles}, "
          f"num_frames={args.num_frames}, gazing_ratio={args.gazing_ratio})...", flush=True)
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
    ).eval()
    print(f"[setup] NVILA loaded; cuda mem={torch.cuda.memory_allocated()/1024**3:.2f} GB",
          flush=True)

    samples = load_subset(args.video_dir, parquet_path=args.parquet_path, query_mode="stem")
    if args.category:
        samples = [s for s in samples if s.get("category") == args.category]
    if args.n_samples is not None:
        samples = samples[:args.n_samples]

    qids = [s["question_id"] for s in samples]
    shuffled_qids = deterministic_shuffle(qids, seed=42)
    qid_to_shufq_qid = {qids[i]: shuffled_qids[i] for i in range(len(qids))}

    print(f"[data] {len(samples)} samples (category={args.category})", flush=True)

    configs = ["match", "shuf", "rand"]
    all_results = {c: [] for c in configs}

    for cfg in configs:
        print(f"\n{'='*50}\nConfig: nvila_attn_{cfg}\n{'='*50}", flush=True)

        per_q = []
        correct = 0
        total = 0
        latencies = []
        nan_fallback = 0

        for i, s in enumerate(samples):
            qid = s["question_id"]
            try:
                if cfg == "match":
                    cached_attn = load_cached_attention(args.matched_cache_dir, qid)
                    if cached_attn is None:
                        nan_fallback += 1
                elif cfg == "shuf":
                    # Use the SHUFFLED-Q cached attention for this qid
                    cached_attn = load_cached_attention(args.shuffled_cache_dir, qid)
                    if cached_attn is None:
                        nan_fallback += 1
                else:  # rand
                    cached_attn = None  # forces random fallback in run_inference_with_filter

                t0 = time.perf_counter()
                response, K, num_v = run_inference_with_filter(
                    model, processor, s["video_path"], s["question_raw"],
                    cached_attn, args.keep_ratio, str(device),
                )
                wall = time.perf_counter() - t0
                pred = extract_answer(response)
                gt = s["answer"]
                is_correct = (pred == gt)
                if is_correct:
                    correct += 1
                total += 1
                latencies.append(wall)

                per_q.append({
                    "qid": qid,
                    "gt": gt,
                    "pred": pred,
                    "correct": int(is_correct),
                    "K": K,
                    "num_v": num_v,
                    "fallback_random": cached_attn is None,
                    "wall_s": wall,
                })

                if (i + 1) % 5 == 0 or i == 0:
                    print(f"  [{i+1}/{len(samples)}] qid={qid} gt={gt} pred={pred} "
                          f"acc={correct}/{total} K={K}/{num_v} avg_lat={np.mean(latencies):.2f}s",
                          flush=True)
            except Exception as e:
                print(f"  [error] qid={qid}: {type(e).__name__}: {e}", flush=True)
                per_q.append({
                    "qid": qid, "gt": s["answer"], "pred": "ERROR", "correct": 0,
                    "K": 0, "num_v": 0, "fallback_random": True,
                    "wall_s": -1, "error": str(e),
                })

        all_results[cfg] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / max(total, 1),
            "avg_lat_s": float(np.mean(latencies)) if latencies else 0,
            "nan_fallback_count": nan_fallback,
            "per_q": per_q,
        }
        print(f"\n  >>> nvila_attn_{cfg}: {correct}/{total} = {correct/max(total,1):.4f}, "
              f"nan_fallback={nan_fallback}, avg_lat={np.mean(latencies):.2f}s", flush=True)

    # Paired-flip
    print(f"\n{'='*60}\nPaired-flip\n{'='*60}", flush=True)
    paired = {}
    for a, b in [("match", "shuf"), ("match", "rand"), ("shuf", "rand")]:
        a_pq = {p["qid"]: p["correct"] for p in all_results[a]["per_q"]}
        b_pq = {p["qid"]: p["correct"] for p in all_results[b]["per_q"]}
        common = set(a_pq.keys()) & set(b_pq.keys())
        a_wins = sum(1 for q in common if a_pq[q] > b_pq[q])
        b_wins = sum(1 for q in common if b_pq[q] > a_pq[q])
        net = a_wins - b_wins
        print(f"  nvila_attn_{a:<5} vs nvila_attn_{b:<5}: a-only={a_wins}  b-only={b_wins}  "
              f"net={net:+d}  n={len(common)}", flush=True)
        paired[f"{a}_vs_{b}"] = {"a_wins": a_wins, "b_wins": b_wins, "net": net,
                                 "n": len(common)}

    out = {
        "configs": configs,
        "n_samples": len(samples),
        "keep_ratio": args.keep_ratio,
        "summary": {k: {kk: vv for kk, vv in v.items() if kk != "per_q"}
                    for k, v in all_results.items()},
        "paired": paired,
        "per_config_per_q": {k: v["per_q"] for k, v in all_results.items()},
    }
    out_path = os.path.join(args.output_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}", flush=True)


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
    p.add_argument("--gazing_ratio", type=float, default=0.50)
    p.add_argument("--gazing_ratio_thumbnail", type=float, default=0.75)
    p.add_argument("--keep_ratio", type=float, default=0.14,
                   help="K/num_v ratio for top-K attention selection (matches r/owlvit-hlvid-vqa)")
    p.add_argument("--matched_cache_dir", default="results/nvila_attention_cache")
    p.add_argument("--shuffled_cache_dir", default="results/nvila_attention_cache_shuffled")
    p.add_argument("--output_dir", default="results/nvila_attn_filter_vqa")
    args = p.parse_args()
    main(args)
