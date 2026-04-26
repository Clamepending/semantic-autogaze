"""r/nvila-attention-distill cycle 1.5c — constraint-compliant VQA test using
cached NVILA-attention as a filter score via bypass_autogaze_selection.

This is the GOAL-compliant redo of cycle 1.5b, which had violated the GOAL
constraint by modifying NVILA's LLM forward through an attention_mask intercept.
Here, NVILA's pipeline runs UNMODIFIED at test time — the only change is that
the gazing_info passed to NVILA's vision encoder is built from an external
score map (NVILA-attention from a separate fine-grid-only extraction pass).

Plumbing mirrors r/owlvit-hlvid-vqa:
  - bypass_autogaze_selection=True (replace AutoGaze's gazing_pos with full
    14x14 fine-grid arange; downstream _shrink_unit_batch picks top-K from
    all 196 per frame using our score_provider).
  - score_provider returns the cached (T_tile, 14, 14) NVILA-attention scores
    for tile inputs; uniform-random for thumb inputs.
  - 3 configs (matched / shuffled / random) at semantic_keep_ratio=0.14
    (K=27/196), matching r/owlvit-hlvid-vqa.

Caches required:
  results/nvila_attention_cache_full_grid/qid_xxxx.npz  (matched)
  results/nvila_attention_cache_full_grid_shuffled/qid_xxxx.npz  (shuffled)
Built by `extract_nvila_attention_full_grid.py`.

Decisive gate:
  matched > shuffled by >= +3 paired-flip wins on n=122 -> NVILA-attention
    has a real GOAL-compliant filter signal worth distilling.
  matched ~~ shuffled -> the +6 paired-flip from cycle 1.5b was an
    attention_mask-intercept artifact. Principled-fix branch closes.
"""
from __future__ import annotations
import os, json, time, random, argparse
from typing import Optional
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper
from semantic_autogaze.eval_vlm_benchmark import (
    patch_processor_with_semantic_filter,
    extract_answer,
)
from semantic_autogaze.eval_hlvid_subset import (
    load_subset,
    run_inference,
    PARQUET_PATH,
)


N_PATCHES = 14 * 14  # 196


class NVILAAttentionScoreProvider:
    """Score provider that looks up cached NVILA-attention (T_tile, 14, 14) maps
    for tile inputs; returns uniform-random scores for thumbnail inputs (we
    only have tile-level cached scores).

    Use:
      provider = NVILAAttentionScoreProvider(cache_dir, mode='match'|'shuf')
      provider.set_qid(qid)         # per-question lookup
      patch_processor_with_semantic_filter(..., score_provider=provider, ...)
    """

    def __init__(self, cache_dir: str, device: torch.device):
        self.cache_dir = cache_dir
        self.device = device
        self.current_qid: Optional[int] = None
        self._cur_tile_scores: Optional[torch.Tensor] = None  # (T_tile, 196)

    def set_qid(self, qid: int):
        path = os.path.join(self.cache_dir, f"qid_{qid:04d}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"NVILA attention cache missing: {path}")
        d = np.load(path, allow_pickle=True)
        fg = d["fine_grid_scores"]  # (T_tile, 14, 14)
        self.current_qid = int(qid)
        # NaN handling: extraction occasionally hits fp16 overflow (~7% of qids
        # in original cache). Replace any NaN-frame with uniform random scores
        # so the top-K selection remains well-defined.
        if np.isnan(fg).any():
            fg = fg.copy()
            fg = np.where(np.isnan(fg), np.random.rand(*fg.shape).astype(fg.dtype), fg)
            self._nan_fallback_used = True
        else:
            self._nan_fallback_used = False
        self._cur_tile_scores = torch.from_numpy(fg.reshape(fg.shape[0], -1)).to(self.device)

    # No-op for compatibility with patch_processor_with_semantic_filter calling
    # set_query_text on score providers.
    def set_query_text(self, text: str):
        pass

    def __call__(self, unit_videos: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        """unit_videos: (B, T, C, H, W) AutoGaze-format video tensor.
        Returns: (B, T*196) scores in [0, 1] (rank-only matters; absolute scale
        is irrelevant since _pick_kept does top-K).
        """
        B, T, C, H, W = unit_videos.shape
        # Heuristic: tiles are (1, T_tile=16). Thumbs are (T_thumb=16, 1).
        is_tile = (B == 1 and T > 1)
        if is_tile and self._cur_tile_scores is not None:
            t_cache = self._cur_tile_scores.shape[0]
            if t_cache < T:
                # Cache shorter than expected -> fall back to random for missing frames
                pad = torch.rand(T - t_cache, N_PATCHES, device=self.device)
                scores = torch.cat([self._cur_tile_scores, pad], dim=0)
            else:
                scores = self._cur_tile_scores[:T]
            out = scores.reshape(1, T * N_PATCHES).expand(B, -1).contiguous()
            return out.to(unit_videos.device)
        # Thumbs (or unexpected shape): random
        return torch.rand(B, T * N_PATCHES, device=unit_videos.device)


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


def main(args):
    device = torch.device(args.device)
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("r/nvila-attention-distill cycle 1.5c (constraint-compliant)")
    print("=" * 60)

    # ---- CLIP for patch_processor plumbing (its scores are unused) ----
    print("\nLoading CLIP for patch_processor plumbing...")
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()

    # ---- AutoGaze wrapper (BigHead checkpoint not used in scoring path) ----
    print("Loading SemanticAutoGazeWrapper...")
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )

    # ---- Score providers (matched + shuffled caches) ----
    match_provider = NVILAAttentionScoreProvider(args.matched_cache_dir, device)
    shuf_provider = NVILAAttentionScoreProvider(args.shuffled_cache_dir, device)

    # ---- NVILA ----
    print("Loading NVILA-8B-HD-Video...")
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
        max_batch_size_siglip=8,
    )
    model.eval()
    print("Model loaded.")

    # ---- HLVid household ----
    print(f"\nLoading HLVid samples (filter category=household)...")
    all_samples = load_subset(args.video_dir, parquet_path=args.parquet_path, query_mode="stem")
    samples = [s for s in all_samples if s.get("category") == "household"]
    if args.n_samples is not None:
        samples = samples[:args.n_samples]
    qids = [s["question_id"] for s in samples]
    print(f"Loaded {len(samples)} household samples")

    # The shuffled-Q text used at extraction time was determined by the same
    # deterministic_shuffle(qids, seed=42); confirm this matches.
    shuffled_qids = deterministic_shuffle(qids, seed=42)
    qid_to_question = {s["question_id"]: s["question_stem"] for s in samples}
    qid_to_shufq = {qids[i]: qid_to_question[shuffled_qids[i]] for i in range(len(qids))}

    configs = ["match", "shuf", "rand"]
    all_results = {}

    for cfg in configs:
        print(f"\n{'='*50}\nConfig: nvila_attn_{cfg}\n{'='*50}")

        if hasattr(processor, "_original_get_gazing"):
            processor._get_gazing_info_from_videos = processor._original_get_gazing
        else:
            processor._original_get_gazing = processor._get_gazing_info_from_videos

        per_q = []
        correct = 0
        total = 0
        latencies = []
        miss_cache = 0
        for i, sample in enumerate(samples):
            qid = sample["question_id"]
            try:
                # Reset processor patch
                processor._get_gazing_info_from_videos = processor._original_get_gazing

                if cfg == "match":
                    q_for_score = sample["question_stem"]
                    cache_path = os.path.join(args.matched_cache_dir, f"qid_{qid:04d}.npz")
                    if not os.path.exists(cache_path):
                        miss_cache += 1
                        raise FileNotFoundError(f"matched cache missing for qid={qid}")
                    match_provider.set_qid(qid)
                    score_provider = match_provider
                elif cfg == "shuf":
                    q_for_score = qid_to_shufq[qid]
                    cache_path = os.path.join(args.shuffled_cache_dir, f"qid_{qid:04d}.npz")
                    if not os.path.exists(cache_path):
                        miss_cache += 1
                        raise FileNotFoundError(f"shuffled cache missing for qid={qid}")
                    shuf_provider.set_qid(qid)
                    score_provider = shuf_provider
                else:
                    q_for_score = sample["question_stem"]
                    score_provider = None  # use random_scoring

                if cfg == "rand":
                    patch_processor_with_semantic_filter(
                        processor, wrapper, clip_model, clip_tokenizer,
                        mode="intersect",
                        semantic_keep_ratio=args.semantic_keep_ratio,
                        query_text=q_for_score,
                        device=str(device),
                        bypass_autogaze_selection=True,
                        random_scoring=True,
                        filter_thumbnails=True,
                    )
                else:
                    patch_processor_with_semantic_filter(
                        processor, wrapper, clip_model, clip_tokenizer,
                        mode="intersect",
                        semantic_keep_ratio=args.semantic_keep_ratio,
                        query_text=q_for_score,
                        device=str(device),
                        bypass_autogaze_selection=True,
                        score_provider=score_provider,
                        filter_thumbnails=True,
                    )

                t0 = time.perf_counter()
                response = run_inference(
                    model, processor, sample["video_path"],
                    sample["question_raw"], str(device),
                )
                wall_s = time.perf_counter() - t0
                pred = extract_answer(response)
                gt = sample["answer"]
                is_correct = (pred == gt)
                if is_correct:
                    correct += 1
                total += 1
                latencies.append(wall_s)
                per_q.append({
                    "qid": qid,
                    "gt": gt,
                    "pred": pred,
                    "correct": int(is_correct),
                    "scoring_q": q_for_score,
                    "wall_s": wall_s,
                })
                if (i + 1) % 5 == 0 or i == 0:
                    print(f"  [{i+1}/{len(samples)}] qid={qid} gt={gt} pred={pred} "
                          f"acc={correct}/{total} avg_lat={sum(latencies)/len(latencies):.2f}s",
                          flush=True)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"  [error] qid={qid}: {type(e).__name__}: {e}", flush=True)
                per_q.append({
                    "qid": qid, "gt": sample["answer"], "pred": "ERROR",
                    "correct": 0, "scoring_q": q_for_score if cfg != "rand" else "random",
                    "wall_s": -1, "error": str(e), "traceback": tb,
                })

        all_results[f"nvila_attn_{cfg}"] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / max(total, 1),
            "avg_lat_s": sum(latencies) / max(len(latencies), 1) if latencies else 0,
            "miss_cache": miss_cache,
            "per_q": per_q,
        }
        print(f"\n  >>> nvila_attn_{cfg}: {correct}/{total} = {correct/max(total,1):.4f}, "
              f"avg_lat={sum(latencies)/max(len(latencies),1):.2f}s, miss_cache={miss_cache}")

    # ---- Paired-flip ----
    print(f"\n{'='*60}\nPaired-flip\n{'='*60}")
    paired = {}
    for a, b in [("nvila_attn_match", "nvila_attn_shuf"),
                 ("nvila_attn_match", "nvila_attn_rand"),
                 ("nvila_attn_shuf", "nvila_attn_rand")]:
        a_pq = {p["qid"]: p["correct"] for p in all_results[a]["per_q"]}
        b_pq = {p["qid"]: p["correct"] for p in all_results[b]["per_q"]}
        common = set(a_pq.keys()) & set(b_pq.keys())
        a_wins = sum(1 for q in common if a_pq[q] > b_pq[q])
        b_wins = sum(1 for q in common if b_pq[q] > a_pq[q])
        net = a_wins - b_wins
        print(f"  {a:<18} vs {b:<18}: a-only={a_wins}  b-only={b_wins}  net={net:+d}  n={len(common)}")
        paired[f"{a}_vs_{b}"] = {"a_wins": a_wins, "b_wins": b_wins, "net": net, "n": len(common)}

    out = {
        "configs": list(all_results.keys()),
        "n_samples": len(samples),
        "summary": {k: {kk: vv for kk, vv in v.items() if kk != "per_q"}
                    for k, v in all_results.items()},
        "paired": paired,
        "per_config_per_q": {k: v["per_q"] for k, v in all_results.items()},
    }
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
    p.add_argument("--parquet_path", default=PARQUET_PATH)
    p.add_argument("--video_dir", default="hlvid_videos/extracted_household/videos")
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--num_frames_thumbnail", type=int, default=16)
    p.add_argument("--max_tiles", type=int, default=1)
    p.add_argument("--gazing_ratio", type=float, default=0.7)
    p.add_argument("--gazing_ratio_thumbnail", type=float, default=0.75)
    p.add_argument("--semantic_keep_ratio", type=float, default=0.14,
                   help="K/196 -> K=27 to match r/owlvit-hlvid-vqa")
    p.add_argument("--matched_cache_dir", default="results/nvila_attention_cache_full_grid")
    p.add_argument("--shuffled_cache_dir", default="results/nvila_attention_cache_full_grid_shuffled")
    p.add_argument("--output_dir", default="results/nvila_attn_bypass_vqa")
    args = p.parse_args()
    main(args)
