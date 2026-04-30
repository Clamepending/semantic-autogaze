"""r/phase10-atto-egoschema-vqa — does the +12 m-vs-shuf on EgoSchema scale
when we swap from v1 step-12000 (Phase 2-FN, mIoU 0.737) to:
  - Phase 10 ConvNeXt-atto best_val.pt (Pi-class SOTA, mIoU 0.696, ~3.7M backbone)
  - Phase 6 DINOv2-s optionB step18000 (project SOTA, mIoU 0.799, 22M backbone)

Adapts eval_vqa_egoschema_pi_class.py to load checkpoints in the
`train_siglip_dense_distill.py` format (used by Phase 2-FN, Phase 3-10):

  ckpt = {
    "args": {"model": "convnext-atto" | "dinov2-s" | "v2-tiny" | ..., ...},
    "head": <TextScorerHead state_dict>,
    "sb": <SiglipBias state_dict>,
    "backbone_state": <optional, when finetune_backbone_blocks > 0>,
    "step": int,
  }

Pre-stated falsifiers (per QUEUE row 1):
- Phase 10 atto m-vs-shuf >= +6 at n=500  -> §1 deployment claim met for Pi class
  AT the highest-mIoU Pi-class scorer the project has produced.
- Phase 6 DINOv2-s m-vs-shuf >= +12 at n=500 -> "higher mIoU = higher m-vs-shuf"
  scaling confirmed (the project SOTA scorer beats v1 at both Tier-1 and Tier-3).
- Both fall below +6 -> the EgoSchema +12 finding is v1-specific (or the recipe
  upgrade does not transfer to NVILA-VQA), and the Tier-3 admission gate stays
  closed for the new ckpts.
"""
from __future__ import annotations
import os, sys, json, time, random, gc, argparse
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")
from train_independent_scorer import TextScorerHead, GRID
from train_siglip_dense_distill import build_backbone, SiglipBias, CLIP_MEAN, CLIP_STD

from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper
from semantic_autogaze.eval_vlm_benchmark import (
    patch_processor_with_semantic_filter,
    extract_answer,
)
from semantic_autogaze.eval_vqa_nvila_attn_bypass import deterministic_shuffle
from semantic_autogaze.eval_vqa_egoschema_ours_v1 import (
    load_egoschema_subset, run_inference_egoschema, IDX_TO_LETTER,
)


N_PATCHES = 14 * 14

# Named ckpts for Phase 3-10 family (train_siglip_dense_distill.py format).
# Keep keys descriptive for the result doc / output dir.
PHASE10_CKPTS = {
    "phase10-atto":     "/home/ogata/semantic-autogaze/results/phase10_convnext_atto_proper/best_val.pt",
    "phase10-femto":    "/home/ogata/semantic-autogaze/results/phase10_convnext_femto_proper/best_val.pt",
    "phase10-pico":     "/home/ogata/semantic-autogaze/results/phase10_convnext_pico_proper/best_val.pt",
    "phase6-dinov2s":   "/home/ogata/semantic-autogaze/results/phase6_dinov2s_optionB/ckpt_step18000.pt",
    "phase8c-v2tiny":   "/home/ogata/semantic-autogaze/results/phase8c_v2tiny_unified_30k/ckpt_step30000.pt",
}


class Phase2bScoreProvider:
    """Score provider for `train_siglip_dense_distill.py`-format ckpts.

    Mirrors the GenericScoreProvider interface in eval_vqa_egoschema_pi_class.py:
      - .set_query_text(text)
      - __call__(unit_videos, query_emb) -> sigmoid scores (B, T*196)

    Backbone is selected from ckpt["args"]["model"] via build_backbone(); patches
    flow through the trained TextScorerHead with (text emb, learnable t/bias).
    """

    def __init__(self, ckpt_key_or_path: str, device: torch.device,
                 clip_model=None, clip_tok=None):
        self.device = device

        if ckpt_key_or_path in PHASE10_CKPTS:
            self.ckpt_path = PHASE10_CKPTS[ckpt_key_or_path]
            self.tag = ckpt_key_or_path
        else:
            self.ckpt_path = ckpt_key_or_path
            self.tag = os.path.basename(os.path.dirname(ckpt_key_or_path))

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

        ck = torch.load(self.ckpt_path, map_location=device, weights_only=False)
        ca = ck.get("args", {}) or {}
        model = ca.get("model", "v1") if isinstance(ca, dict) else getattr(ca, "model", "v1")
        finetune_blocks = (ca.get("finetune_backbone_blocks", 0) if isinstance(ca, dict)
                           else getattr(ca, "finetune_backbone_blocks", 0))

        bb_fn, patch_dim, mean, std, kind, bb_module = build_backbone(
            model, device, finetune_blocks=finetune_blocks)
        self._bb_fn = bb_fn
        self._bb_module = bb_module
        self._kind = kind
        self.norm_mean = torch.tensor(mean, device=device)
        self.norm_std = torch.tensor(std, device=device)

        head_kwargs = dict(
            patch_dim=patch_dim, text_dim=512,
            hidden_dim=(ca.get("head_hidden_dim", 384) if isinstance(ca, dict)
                        else getattr(ca, "head_hidden_dim", 384)),
            n_attn_heads=(ca.get("head_attn_heads", 6) if isinstance(ca, dict)
                          else getattr(ca, "head_attn_heads", 6)),
            n_attn_layers=(ca.get("head_attn_layers", 2) if isinstance(ca, dict)
                           else getattr(ca, "head_attn_layers", 2)),
            grid_size=GRID,
            use_spatial=(ca.get("head_use_spatial", True) if isinstance(ca, dict)
                         else getattr(ca, "head_use_spatial", True)),
        )
        self.head = TextScorerHead(**head_kwargs).to(device).eval()
        self.head.load_state_dict(ck["head"])
        for p in self.head.parameters(): p.requires_grad_(False)

        self.sb = SiglipBias().to(device).eval()
        if "sb" in ck:
            self.sb.load_state_dict(ck["sb"])
        for p in self.sb.parameters(): p.requires_grad_(False)

        if "backbone_state" in ck and self._bb_module is not None:
            try:
                self._bb_module.load_state_dict(ck["backbone_state"])
            except Exception as e:
                print(f"[{self.tag}] backbone_state load skipped: {e}", flush=True)

        # SigLIP head outputs LOGITS pre-(t, bias). For sigmoid scores at scoring
        # time, we apply learnable t and bias here. eval_phase2_ckpt.py applies
        # them via heatmap_one -> sb.forward; we do the same below.
        self._cur_text_emb: Optional[torch.Tensor] = None

        n_head = sum(p.numel() for p in self.head.parameters())
        n_bb = sum(p.numel() for p in self._bb_module.parameters()) if self._bb_module is not None else 0
        print(f"[{self.tag}] model={model} patch_dim={patch_dim} "
              f"head={n_head/1e6:.2f}M backbone={n_bb/1e6:.2f}M "
              f"t={float(self.sb.log_t.exp()):.3f} bias={float(self.sb.bias):.3f}",
              flush=True)

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
        Returns: (B, T*196) sigmoid scores after learnable SigLIP (t, bias)."""
        if self._cur_text_emb is None:
            return torch.rand(unit_videos.shape[0], unit_videos.shape[1] * N_PATCHES,
                              device=unit_videos.device)
        B, T, C, H, W = unit_videos.shape
        x = unit_videos.reshape(B * T, C, H, W)
        # AutoGaze hands us [-1, 1]; backbone expects [0, 1] then per-channel norm.
        x = (x + 1.0) / 2.0
        if (H, W) != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        x = (x - self.norm_mean[None, :, None, None]) / self.norm_std[None, :, None, None]

        patches = self._bb_fn(x)  # (B*T, 196, D) — _bb_fn handles ViT/CNN adapter

        text_emb = self._cur_text_emb.expand(B * T, -1)
        logits = self.head(patches, text_emb)         # (B*T, 196) raw logits
        logits = self.sb(logits)                       # apply learnable (t, bias)
        return torch.sigmoid(logits).reshape(B, T * N_PATCHES)


def main(args):
    device = torch.device(args.device)
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"r/phase10-atto-egoschema-vqa — scorer={args.scorer}")
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

    print(f"Loading {args.scorer} score providers (sharing CLIP)...", flush=True)
    ours_match = Phase2bScoreProvider(args.scorer, device,
                                       clip_model=clip_model, clip_tok=clip_tokenizer)
    ours_shuf = Phase2bScoreProvider(args.scorer, device,
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
                else:  # rand
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
                ok = pred == gt
                correct += int(ok); total += 1; latencies.append(wall_s)
                per_q.append({
                    "qid": qid, "gt": gt, "pred": pred,
                    "correct": int(ok), "scoring_q": q_for_score,
                    "wall_s": wall_s,
                    "duration": sample.get("duration"),
                    "task_type": sample.get("task_type"),
                })
            except Exception as e:
                print(f"  [{cfg}][{qid}] ERROR: {e}", flush=True)
                per_q.append({"qid": qid, "gt": sample.get("answer"),
                              "pred": "ERROR", "correct": 0, "wall_s": -1.0,
                              "error": str(e)})

            # Per-qid checkpoint every 10 samples + at end
            if total % 10 == 0 or i == len(samples) - 1:
                with open(per_qid_path, "w") as _f:
                    json.dump({"per_q": per_q,
                               "summary": {"n": total, "correct": correct,
                                           "acc": correct / max(1, total)}},
                              _f, indent=2)

            if i % 5 == 0:
                acc = correct / max(1, total)
                med_lat = float(np.median(latencies)) if latencies else 0.0
                print(f"  [{cfg}] {total}/{len(samples)} acc={acc:.3f} med_lat={med_lat:.2f}s",
                      flush=True)

        all_results[cfg] = {
            "n": total, "correct": correct, "acc": correct / max(1, total),
            "med_lat": float(np.median(latencies)) if latencies else 0.0,
            "per_q": per_q,
        }
        with open(per_qid_path, "w") as _f:
            json.dump(all_results[cfg], _f, indent=2)

    # Paired-flip nets (match / shuf / rand all share the same per-qid order)
    def _paired(per_a, per_b):
        a = {p["qid"]: p["correct"] for p in per_a}
        b = {p["qid"]: p["correct"] for p in per_b}
        common = sorted(set(a) & set(b))
        a_only = sum(a[q] - b[q] for q in common if a[q] > b[q])
        b_only = sum(b[q] - a[q] for q in common if b[q] > a[q])
        return a_only, b_only, a_only - b_only

    summary = {
        "scorer": args.scorer,
        "ckpt_path": ours_match.ckpt_path,
        "n_samples": len(samples),
        "configs": {cfg: {"n": all_results[cfg]["n"],
                           "correct": all_results[cfg]["correct"],
                           "acc": all_results[cfg]["acc"],
                           "med_lat": all_results[cfg]["med_lat"]}
                    for cfg in configs},
    }
    if all(c in all_results for c in configs):
        ms_w, ms_l, ms_n = _paired(all_results["match"]["per_q"], all_results["shuf"]["per_q"])
        mr_w, mr_l, mr_n = _paired(all_results["match"]["per_q"], all_results["rand"]["per_q"])
        sr_w, sr_l, sr_n = _paired(all_results["shuf"]["per_q"], all_results["rand"]["per_q"])
        mv_w, mv_l, mv_n = _paired(all_results["match"]["per_q"], all_results["vanilla"]["per_q"])
        summary["paired_flips"] = {
            "match_vs_shuf":   {"wins": ms_w, "losses": ms_l, "net": ms_n},
            "match_vs_rand":   {"wins": mr_w, "losses": mr_l, "net": mr_n},
            "shuf_vs_rand":    {"wins": sr_w, "losses": sr_l, "net": sr_n},
            "match_vs_vanilla":{"wins": mv_w, "losses": mv_l, "net": mv_n},
        }
        print(f"\n[final] match-vs-shuf net = {ms_n:+d} (wins {ms_w} / losses {ms_l})", flush=True)
        print(f"[final] match-vs-rand net = {mr_n:+d}", flush=True)
        print(f"[final] match-vs-vanilla net = {mv_n:+d}", flush=True)
    out_summary = os.path.join(args.output_dir, "summary.json")
    with open(out_summary, "w") as _f:
        json.dump(summary, _f, indent=2)
    print(f"\nSummary saved to {out_summary}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--scorer", required=True,
                   help=f"Either a key in PHASE10_CKPTS ({sorted(PHASE10_CKPTS)}) or a path.")
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
                   help="results/egoschema_phase10_<scorer> recommended")
    args = p.parse_args()
    main(args)
