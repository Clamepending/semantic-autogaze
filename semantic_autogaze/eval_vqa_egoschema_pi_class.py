"""r/egoschema-pi-class-scorer-test — does the +12 m-v-s on EgoSchema survive
when we swap Ours v1 (CLIP-B/16, 91.6 M) for v2-Tiny (8.9 M) or D-Mobile (5.05 M)?

Adapts eval_vqa_egoschema_ours_v1.py to load any of the three trained scorer
sizes via --scorer_size {v1, v2-tiny, d-mobile}. If the smaller scorers also
show match-vs-shuf >= +6 at n=500, the §1 unified deployment claim
('>= 10 fps on Pi 4 + decisive text-conditioned signal at sparse-relevance')
is met by a single delivered scorer.

Pre-stated falsifiers (per QUEUE row 1):
- v2-Tiny m-v-s >= +6 at n=500 -> §1 deployment claim met for desktop class.
  D-Mobile m-v-s >= +6 -> claim met for Pi class.
- m-v-s near 0 -> the EgoSchema +12 finding is v1-specific; deployment story
  narrows back to the v1 size class (which doesn't fit Pi).
"""
from __future__ import annotations
import os, sys, json, time, random, gc, argparse
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
import timm

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")
from train_independent_scorer import TextScorerHead, GRID

from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper
from semantic_autogaze.eval_vlm_benchmark import (
    patch_processor_with_semantic_filter,
    extract_answer,
)
from semantic_autogaze.eval_vqa_nvila_attn_bypass import deterministic_shuffle
from semantic_autogaze.eval_vqa_egoschema_ours_v1 import (
    load_egoschema_subset, run_inference_egoschema, IDX_TO_LETTER,
)

import av


N_PATCHES = 14 * 14

CKPTS = {
    "v1":       "/home/ogata/semantic-autogaze/results/independent_scorer/best_v1.pt",
    "v2-tiny":  "/home/ogata/semantic-autogaze/results/independent_scorer_v2_tiny/best.pt",
    "d-mobile": "/home/ogata/semantic-autogaze/results/sweep_v3/D_mobilenet_std/best.pt",
}

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IM_MEAN = (0.485, 0.456, 0.406)
IM_STD = (0.229, 0.224, 0.225)


def _adapt_features(feats):
    """timm/CLIP output -> (B, 196, C). ViT: drop CLS. CNN: bilinear upsample."""
    if feats.dim() == 4:
        feats = F.interpolate(feats, size=(GRID, GRID), mode="bilinear", align_corners=False)
        return feats.permute(0, 2, 3, 1).reshape(feats.shape[0], GRID * GRID, feats.shape[1])
    if feats.shape[1] == 197:
        return feats[:, 1:, :]
    return feats


class GenericScoreProvider:
    """Flexible score provider for v1 (CLIP visual), v2-Tiny / D-Mobile (timm)."""

    def __init__(self, scorer_size: str, device: torch.device,
                 clip_model=None, clip_tok=None):
        self.device = device
        self.scorer_size = scorer_size

        if clip_model is None or clip_tok is None:
            import open_clip
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained="openai")
            self.clip_tok = open_clip.get_tokenizer("ViT-B-16")
            self.clip_model = self.clip_model.to(device).eval()
        else:
            self.clip_model = clip_model
            self.clip_tok = clip_tok
        for p in self.clip_model.parameters(): p.requires_grad_(False)

        ckpt_path = CKPTS[scorer_size]
        ck = torch.load(ckpt_path, map_location=device)
        ca = ck.get("args", {}) or {}

        if scorer_size == "v1":
            head_patch_dim = 768
            self.backbone = None  # uses self.clip_model.visual
            self.norm_mean = torch.tensor(CLIP_MEAN, device=device)
            self.norm_std = torch.tensor(CLIP_STD, device=device)
        else:
            head_patch_dim = ck["embed_dim"]
            self.backbone = timm.create_model(ck["backbone"], pretrained=True, num_classes=0).to(device).eval()
            for p in self.backbone.parameters(): p.requires_grad_(False)
            self.norm_mean = torch.tensor(IM_MEAN, device=device)
            self.norm_std = torch.tensor(IM_STD, device=device)

        self.head = TextScorerHead(
            patch_dim=head_patch_dim, text_dim=512,
            hidden_dim=ca.get("head_hidden_dim", 384),
            n_attn_heads=ca.get("head_attn_heads", 6),
            n_attn_layers=ca.get("head_attn_layers", 2),
            grid_size=GRID,
            use_spatial=ca.get("head_use_spatial", True),
        ).to(device).eval()
        self.head.load_state_dict(ck["head"])
        for p in self.head.parameters(): p.requires_grad_(False)

        self._cur_text_emb: Optional[torch.Tensor] = None

    def set_qid(self, qid):
        pass

    @torch.no_grad()
    def set_query_text(self, text: str):
        toks = self.clip_tok([text]).to(self.device)
        emb = self.clip_model.encode_text(toks)
        self._cur_text_emb = F.normalize(emb, dim=-1)

    @torch.no_grad()
    def __call__(self, unit_videos: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        """unit_videos: (B, T, C, H, W) AutoGaze-format [-1, 1].
        Returns: (B, T*196) sigmoid scores."""
        if self._cur_text_emb is None:
            return torch.rand(unit_videos.shape[0], unit_videos.shape[1] * N_PATCHES,
                              device=unit_videos.device)
        B, T, C, H, W = unit_videos.shape
        x = unit_videos.reshape(B * T, C, H, W)
        x = (x + 1.0) / 2.0  # [-1,1] -> [0,1]
        if (H, W) != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        x = (x - self.norm_mean[None, :, None, None]) / self.norm_std[None, :, None, None]

        if self.scorer_size == "v1":
            self.clip_model.visual.output_tokens = True
            _, patches = self.clip_model.visual(x)  # (B*T, 196, 768)
            self.clip_model.visual.output_tokens = False
        else:
            patches = _adapt_features(self.backbone.forward_features(x))

        text_emb = self._cur_text_emb.expand(B * T, -1)
        scores = self.head(patches, text_emb)
        return torch.sigmoid(scores).reshape(B, T * N_PATCHES)


def main(args):
    device = torch.device(args.device)
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"r/egoschema-pi-class-scorer-test — scorer_size={args.scorer_size}")
    print("=" * 60)

    samples = load_egoschema_subset(args.parquet_path, args.video_dir,
                                    n_samples=args.n_samples, seed=args.seed)
    print(f"Loaded {len(samples)} samples (total {len(set(s['videoID'] for s in samples))} unique videos)",
          flush=True)
    if not samples: raise SystemExit("No samples found.")

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

    print(f"Loading {args.scorer_size} score providers (sharing CLIP)...", flush=True)
    ours_match = GenericScoreProvider(args.scorer_size, device,
                                       clip_model=clip_model, clip_tok=clip_tokenizer)
    ours_shuf = GenericScoreProvider(args.scorer_size, device,
                                      clip_model=clip_model, clip_tok=clip_tokenizer)
    torch.cuda.empty_cache()

    print("Loading NVILA-8B-HD-Video...", flush=True)
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True,
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
            if qid in done_qids: continue
            try:
                processor._get_gazing_info_from_videos = processor._original_get_gazing
                if cfg == "vanilla":
                    q_for_score = sample["question_stem"]; score_provider = None
                elif cfg == "match":
                    q_for_score = sample["question_stem"]
                    ours_match.set_query_text(q_for_score)
                    score_provider = ours_match
                elif cfg == "shuf":
                    q_for_score = qid_to_shufq[qid]
                    ours_shuf.set_query_text(q_for_score)
                    score_provider = ours_shuf
                else:
                    q_for_score = sample["question_stem"]; score_provider = None

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

                t0 = time.perf_counter()
                response = run_inference_egoschema(
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
        common = set(a_pq) & set(b_pq)
        a_wins = sum(1 for q in common if a_pq[q] > b_pq[q])
        b_wins = sum(1 for q in common if b_pq[q] > a_pq[q])
        net = a_wins - b_wins
        print(f"  {a:<10} vs {b:<10}: a-only={a_wins}  b-only={b_wins}  net={net:+d}  n={len(common)}")
        paired[f"{a}_vs_{b}"] = {"a_wins": a_wins, "b_wins": b_wins, "net": net, "n": len(common)}

    out = {"configs": list(all_results.keys()), "n_samples": len(samples),
           "scorer_size": args.scorer_size,
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
    p.add_argument("--scorer_size", required=True, choices=["v1", "v2-tiny", "d-mobile"])
    p.add_argument("--model_path", default="nvidia/NVILA-8B-HD-Video")
    p.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    p.add_argument("--ckpt", default="results/bighead/best_bighead.pt")
    p.add_argument("--head_type", default="bighead")
    p.add_argument("--parquet_path",
                   default="/home/ogata/semantic-autogaze/data/egoschema/Subset/test-00000-of-00001.parquet")
    p.add_argument("--video_dir",
                   default="/home/ogata/semantic-autogaze/data/egoschema/videos")
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--num_frames_thumbnail", type=int, default=16)
    p.add_argument("--max_tiles", type=int, default=1)
    p.add_argument("--gazing_ratio", type=float, default=0.7)
    p.add_argument("--gazing_ratio_thumbnail", type=float, default=0.75)
    p.add_argument("--semantic_keep_ratio", type=float, default=0.1378)
    p.add_argument("--output_dir", required=True,
                   help="separate output dir per scorer_size, e.g. results/egoschema_v2tiny")
    args = p.parse_args()
    main(args)
