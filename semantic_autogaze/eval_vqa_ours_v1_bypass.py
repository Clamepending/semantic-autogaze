"""r/independent-text-scorer-v1 cycle 2 — Tier-3 HLVid VQA bypass test.

Mirrors eval_vqa_nvila_attn_bypass.py but with an ONLINE score provider that
runs the trained Ours v1 head (frozen CLIP ViT-B/16 + small text-conditional
head) per (frame, query) instead of looking up cached scores.

Decisive admission gate (HLVid household, n=122, K=27):
  matched >= 53/122 -> ADMITS to LEADERBOARD (beats rank-1 vanilla 53/122).
  matched-vs-shuf >= +5 with matched >= 45 -> Phase-2 candidate.
  matched < 45 AND matched-vs-shuf < +3 -> close direction; the COCO-mIoU
    signal does NOT translate to HLVid VQA admission.

Loads:
  /home/ogata/semantic-autogaze/results/independent_scorer/best_v1.pt
"""
from __future__ import annotations
import os, sys, json, time, random, argparse
from typing import Optional
import numpy as np
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
from semantic_autogaze.eval_hlvid_subset import (
    load_subset,
    run_inference,
    PARQUET_PATH,
)
from semantic_autogaze.eval_vqa_nvila_attn_bypass import deterministic_shuffle
from train_independent_scorer import TextScorerHead


N_PATCHES = 14 * 14
GRID = 14

OURS_CKPT = "/home/ogata/semantic-autogaze/results/independent_scorer/best_v1.pt"

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class OursV1ScoreProvider:
    """Online score provider: runs frozen CLIP visual + Ours v1 head per frame
    with the current query text. Returns scores in (B, T*196).

    Random-mode fallback returns uniform random per call (used by the 'rand'
    config wiring path; in practice the eval code drives random_scoring=True
    via a separate path so this class is only used for matched / shuffled).
    """

    def __init__(self, head_ckpt: str, device: torch.device):
        import open_clip
        self.device = device
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai")
        self.clip_tok = open_clip.get_tokenizer("ViT-B-16")
        self.clip_model = self.clip_model.to(device).eval()
        for p in self.clip_model.parameters(): p.requires_grad_(False)

        ckpt = torch.load(head_ckpt, map_location=device)
        self.head = TextScorerHead(patch_dim=768, text_dim=512, hidden_dim=384,
                                   n_attn_heads=6, n_attn_layers=2, grid_size=GRID).to(device).eval()
        self.head.load_state_dict(ckpt["head"] if isinstance(ckpt, dict) and "head" in ckpt else ckpt)
        for p in self.head.parameters(): p.requires_grad_(False)

        self.mean = torch.tensor(CLIP_MEAN, device=device)
        self.std = torch.tensor(CLIP_STD, device=device)
        self._cur_text_emb: Optional[torch.Tensor] = None  # (1, 512)

    def set_qid(self, qid: int):
        # No per-qid state; the per-query text emb is set via set_query_text.
        pass

    @torch.no_grad()
    def set_query_text(self, text: str):
        toks = self.clip_tok([text]).to(self.device)
        emb = self.clip_model.encode_text(toks)
        self._cur_text_emb = F.normalize(emb, dim=-1)  # (1, 512)

    @torch.no_grad()
    def __call__(self, unit_videos: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        """unit_videos: (B, T, C, H, W) — AutoGaze-format [-1,1]-normalized.
        Returns: (B, T*196) sigmoid-shaped scores in [0,1]."""
        if self._cur_text_emb is None:
            return torch.rand(unit_videos.shape[0], unit_videos.shape[1] * N_PATCHES,
                              device=unit_videos.device)
        B, T, C, H, W = unit_videos.shape
        # Convert AutoGaze-format ([-1,1]) into CLIP-format (mean/std normalized at 224x224).
        # AutoGaze uses (x/127.5 - 1.0) — i.e., x in [0,255] -> [-1,1].
        # First denormalize to [0,1], then renormalize with CLIP mean/std at 224x224.
        x = unit_videos.reshape(B * T, C, H, W)
        x = (x + 1.0) / 2.0  # [-1,1] -> [0,1]
        if (H, W) != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        x = (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]

        self.clip_model.visual.output_tokens = True
        _, patch_tokens = self.clip_model.visual(x)  # (B*T, 196, 768)
        self.clip_model.visual.output_tokens = False

        text_emb = self._cur_text_emb.expand(B * T, -1)  # (B*T, 512)
        scores = self.head(patch_tokens, text_emb)  # (B*T, 196) logits
        scores = torch.sigmoid(scores)
        return scores.reshape(B, T * N_PATCHES)


def main(args):
    device = torch.device(args.device)
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("r/independent-text-scorer-v1 cycle 2 — Tier-3 HLVid VQA bypass")
    print("=" * 60)

    # ---- CLIP for patch_processor plumbing (its scores are unused) ----
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

    print("Loading Ours v1 score provider...", flush=True)
    ours_match = OursV1ScoreProvider(args.ours_ckpt, device)
    ours_shuf = OursV1ScoreProvider(args.ours_ckpt, device)

    print("Loading NVILA-8B-HD-Video...", flush=True)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
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
    print("Model loaded.", flush=True)

    print(f"\nLoading HLVid samples (filter category=household)...", flush=True)
    all_samples = load_subset(args.video_dir, parquet_path=args.parquet_path, query_mode="stem")
    samples = [s for s in all_samples if s.get("category") == "household"]
    if args.n_samples is not None:
        samples = samples[:args.n_samples]
    qids = [s["question_id"] for s in samples]
    print(f"Loaded {len(samples)} household samples", flush=True)

    shuffled_qids = deterministic_shuffle(qids, seed=42)
    qid_to_question = {s["question_id"]: s["question_stem"] for s in samples}
    qid_to_shufq = {qids[i]: qid_to_question[shuffled_qids[i]] for i in range(len(qids))}

    configs = ["match", "shuf", "rand"]
    all_results = {}

    for cfg in configs:
        print(f"\n{'='*50}\nConfig: ours_v1_{cfg}\n{'='*50}", flush=True)

        if hasattr(processor, "_original_get_gazing"):
            processor._get_gazing_info_from_videos = processor._original_get_gazing
        else:
            processor._original_get_gazing = processor._get_gazing_info_from_videos

        per_q = []; correct = 0; total = 0; latencies = []
        for i, sample in enumerate(samples):
            qid = sample["question_id"]
            try:
                processor._get_gazing_info_from_videos = processor._original_get_gazing

                if cfg == "match":
                    q_for_score = sample["question_stem"]
                    ours_match.set_query_text(q_for_score)
                    score_provider = ours_match
                elif cfg == "shuf":
                    q_for_score = qid_to_shufq[qid]
                    ours_shuf.set_query_text(q_for_score)
                    score_provider = ours_shuf
                else:
                    q_for_score = sample["question_stem"]
                    score_provider = None

                if cfg == "rand":
                    patch_processor_with_semantic_filter(
                        processor, wrapper, clip_model, clip_tokenizer,
                        mode="intersect",
                        semantic_keep_ratio=args.semantic_keep_ratio,
                        query_text=q_for_score, device=str(device),
                        bypass_autogaze_selection=True, random_scoring=True,
                        filter_thumbnails=True,
                    )
                else:
                    patch_processor_with_semantic_filter(
                        processor, wrapper, clip_model, clip_tokenizer,
                        mode="intersect",
                        semantic_keep_ratio=args.semantic_keep_ratio,
                        query_text=q_for_score, device=str(device),
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
                if is_correct: correct += 1
                total += 1; latencies.append(wall_s)
                per_q.append({"qid": qid, "gt": gt, "pred": pred,
                              "correct": int(is_correct),
                              "scoring_q": q_for_score, "wall_s": wall_s})
                if (i + 1) % 5 == 0 or i == 0:
                    print(f"  [{i+1}/{len(samples)}] qid={qid} gt={gt} pred={pred} "
                          f"acc={correct}/{total} avg_lat={sum(latencies)/len(latencies):.2f}s",
                          flush=True)
            except Exception as e:
                import traceback; tb = traceback.format_exc()
                print(f"  [error] qid={qid}: {type(e).__name__}: {e}", flush=True)
                per_q.append({"qid": qid, "gt": sample["answer"], "pred": "ERROR",
                              "correct": 0,
                              "scoring_q": q_for_score if cfg != "rand" else "random",
                              "wall_s": -1, "error": str(e), "traceback": tb})

        all_results[f"ours_v1_{cfg}"] = {
            "correct": correct, "total": total,
            "accuracy": correct / max(total, 1),
            "avg_lat_s": sum(latencies) / max(len(latencies), 1) if latencies else 0,
            "per_q": per_q,
        }
        print(f"\n  >>> ours_v1_{cfg}: {correct}/{total} = {correct/max(total,1):.4f}, "
              f"avg_lat={sum(latencies)/max(len(latencies),1):.2f}s")

    print(f"\n{'='*60}\nPaired-flip\n{'='*60}")
    paired = {}
    for a, b in [("ours_v1_match", "ours_v1_shuf"),
                 ("ours_v1_match", "ours_v1_rand"),
                 ("ours_v1_shuf", "ours_v1_rand")]:
        a_pq = {p["qid"]: p["correct"] for p in all_results[a]["per_q"]}
        b_pq = {p["qid"]: p["correct"] for p in all_results[b]["per_q"]}
        common = set(a_pq.keys()) & set(b_pq.keys())
        a_wins = sum(1 for q in common if a_pq[q] > b_pq[q])
        b_wins = sum(1 for q in common if b_pq[q] > a_pq[q])
        net = a_wins - b_wins
        print(f"  {a:<18} vs {b:<18}: a-only={a_wins}  b-only={b_wins}  net={net:+d}  n={len(common)}")
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
    p.add_argument("--parquet_path", default=PARQUET_PATH)
    p.add_argument("--video_dir", default="hlvid_videos/extracted_household/videos")
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--num_frames_thumbnail", type=int, default=16)
    p.add_argument("--max_tiles", type=int, default=1)
    p.add_argument("--gazing_ratio", type=float, default=0.7)
    p.add_argument("--gazing_ratio_thumbnail", type=float, default=0.75)
    p.add_argument("--semantic_keep_ratio", type=float, default=0.14)
    p.add_argument("--output_dir", default="results/ours_v1_bypass_vqa")
    args = p.parse_args()
    main(args)
