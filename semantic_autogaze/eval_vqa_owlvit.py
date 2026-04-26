"""r/owlvit-hlvid-vqa cycle 1: end-to-end VQA test of OWL-ViT-as-filter on HLVid.

Replaces BigHead + CLIP scoring in eval_vlm_benchmark with an OWL-ViT score
provider that runs the trained-for-text-conditioned-detection class predictor
on each (frame, query) pair. Uses bypass_autogaze_selection so OWL-ViT picks
top-K from all 196 fine-grid patches per frame (rather than re-ranking
AutoGaze's K-subset). semantic_keep_ratio=0.14 -> K=27 patches/frame, matching
vanilla scale 0.70 = 53/122 baseline.

Three configs per question: matched-Q, shuffled-Q (deterministic seed 42),
random scoring. Decisive: matched > shuffled by ≥+3 paired-flip wins on
n=122 -> OWL-ViT direction OPEN at VQA level; ≤+1 win -> 5th admission class
CLOSED (phrase-grounding direction definitively closed under both fidelity
and VQA metrics).

Usage:
  CUDA_VISIBLE_DEVICES=N python -m semantic_autogaze.eval_vqa_owlvit \\
    --device cuda:0 --hlvid_subset hlvid_videos/extracted/household.parquet \\
    --output_dir results/owlvit_hlvid_vqa
"""
from __future__ import annotations
import os, json, time, random, argparse
from typing import Optional
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper
from semantic_autogaze.eval_vlm_benchmark import (
    patch_processor_with_semantic_filter,
    extract_answer,
    get_clip_text_embedding,
)
from semantic_autogaze.eval_hlvid_subset import (
    load_subset,
    run_inference,
    PARQUET_PATH,
)


OWLVIT_NAME = "google/owlvit-base-patch32"
SIGLIP_GRID = 14
N_PATCHES = SIGLIP_GRID * SIGLIP_GRID  # 196
# AutoGaze normalization (ImageNet mean/std at 224x224)
AUTOGAZE_MEAN = (0.485, 0.456, 0.406)
AUTOGAZE_STD = (0.229, 0.224, 0.225)


class OwlViTScoreProvider:
    """Computes OWL-ViT patch-text scores for AutoGaze-format video tensors.

    Signature matches `_shrink_unit_batch`'s expected score path:
      __call__(unit_videos: (B, T, C, H, W), query_emb: (B, embed_dim))
        -> scores: (B, T*196), sigmoid-shaped in [0, 1]

    The query_emb is IGNORED — OWL-ViT uses its own text encoding which is
    cached via set_query_text() before each call (matches the existing
    monkey-patch's per-question query-embedding pattern).
    """

    def __init__(self, device: torch.device):
        from transformers import OwlViTForObjectDetection, AutoTokenizer, AutoImageProcessor
        self.device = device
        self.model = OwlViTForObjectDetection.from_pretrained(OWLVIT_NAME).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(OWLVIT_NAME)
        self.imgproc = AutoImageProcessor.from_pretrained(OWLVIT_NAME)
        self.size = self.imgproc.size["height"] if isinstance(self.imgproc.size, dict) else 768
        self.owlvit_mean = torch.tensor(self.imgproc.image_mean, device=device)
        self.owlvit_std = torch.tensor(self.imgproc.image_std, device=device)
        self.autogaze_mean = torch.tensor(AUTOGAZE_MEAN, device=device)
        self.autogaze_std = torch.tensor(AUTOGAZE_STD, device=device)
        self._cur_query_emb = None  # (1, 512), set per-question

    @torch.no_grad()
    def set_query_text(self, text: str):
        enc = self.tok([text], padding="max_length", return_tensors="pt").to(self.device)
        out = self.model.owlvit.text_model(input_ids=enc["input_ids"])
        pooled = self.model.owlvit.text_projection(out.pooler_output)
        self._cur_query_emb = F.normalize(pooled, dim=-1)  # (1, 512)

    @torch.no_grad()
    def __call__(self, unit_videos: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        """unit_videos: (B, T, C, H, W) AutoGaze-normalized tensors.
        query_emb: (B, 512) IGNORED — uses self._cur_query_emb set by set_query_text.
        Returns: (B, T*196) sigmoid scores in [0, 1].
        """
        assert self._cur_query_emb is not None, "Call set_query_text(...) first"
        B, T, C, H, W = unit_videos.shape

        # Un-normalize from AutoGaze, resize to OWL-ViT size, re-normalize for OWL-ViT
        flat = unit_videos.view(B * T, C, H, W)
        # Undo AutoGaze normalization
        raw = flat * self.autogaze_std[None, :, None, None] + self.autogaze_mean[None, :, None, None]
        # Resize to OWL-ViT input size (768x768 default)
        raw = F.interpolate(raw, size=(self.size, self.size), mode="bicubic", align_corners=False)
        raw = raw.clamp(0.0, 1.0)
        # OWL-ViT normalization
        owl_in = (raw - self.owlvit_mean[None, :, None, None]) / self.owlvit_std[None, :, None, None]

        # OWL-ViT image_embedder + class_head.dense0 -> (B*T, h_p, w_p, 512)
        image_embeds, _ = self.model.image_embedder(pixel_values=owl_in)
        h_p = image_embeds.shape[1]
        w_p = image_embeds.shape[2]
        image_class_embeds = self.model.class_head.dense0(image_embeds)  # (B*T, h, w, 512)

        # Bilinear-resample to 14x14 grid
        x = image_class_embeds.permute(0, 3, 1, 2)  # (B*T, 512, h, w)
        x = F.interpolate(x, size=(SIGLIP_GRID, SIGLIP_GRID), mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B*T, 14, 14, 512)
        x = F.normalize(x, dim=-1)
        x = x.view(B, T, N_PATCHES, -1)  # (B, T, 196, 512)

        # Cosine with text emb -> (B, T, 196)
        cos = torch.einsum("btnd,kd->btn", x, self._cur_query_emb).clamp(-1.0, 1.0)
        # Sigmoid-warp like the existing CLIP/SigLIP-2 scorers
        scores = torch.sigmoid(cos * 10.0)
        return scores.view(B, T * N_PATCHES)  # (B, T*196)


def deterministic_shuffle(qids, seed=42):
    """Return a permutation of qids with no fixed points (each qid maps to a different qid)."""
    rng = random.Random(seed)
    n = len(qids)
    result = list(qids)
    while True:
        rng.shuffle(result)
        if all(result[i] != qids[i] for i in range(n)):
            return result


def main(args):
    device = torch.device(args.device)
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("r/owlvit-hlvid-vqa cycle 1")
    print("=" * 60)

    # ---- Load OWL-ViT (~600 MiB) ----
    print("\nLoading OWL-ViT...")
    owlvit_provider = OwlViTScoreProvider(device)
    print(f"  size={owlvit_provider.size}")

    # ---- Load CLIP (used by patch_processor; we ignore its scoring path) ----
    print("Loading CLIP for patch_processor plumbing...")
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()

    # ---- Load AutoGaze wrapper (BigHead checkpoint not used in scoring path) ----
    print("Loading SemanticAutoGazeWrapper...")
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )

    # ---- Load NVILA ----
    print("Loading NVILA-8B-HD-Video...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        num_video_frames=args.num_frames,
        num_video_frames_thumbnail=args.num_frames_thumbnail,
        max_tiles_video=args.max_tiles,
        gazing_ratio_tile=args.gazing_ratio,
        gazing_ratio_thumbnail=1.0,
        task_loss_requirement_tile=0.6,
        task_loss_requirement_thumbnail=None,
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
    )
    model.eval()
    print("Model loaded.")

    # ---- Load HLVid household ----
    print(f"\nLoading HLVid samples (filter category=household)...")
    all_samples = load_subset(args.video_dir, parquet_path=args.parquet_path, query_mode="stem")
    samples = [s for s in all_samples if s.get("category") == "household"]
    if args.n_samples is not None:
        samples = samples[:args.n_samples]
    qids = [s["question_id"] for s in samples]
    print(f"Loaded {len(samples)} household samples")

    # Build deterministic shuffle assignment for shuffled-Q config
    shuffled_qids = deterministic_shuffle(qids, seed=42)
    qid_to_question = {s["question_id"]: s["question_stem"] for s in samples}
    qid_to_shufq = {qids[i]: qid_to_question[shuffled_qids[i]] for i in range(len(qids))}

    # ---- Configurations ----
    configs = []
    for variant in ("match", "shuf", "rand"):
        configs.append({"name": f"owlvit_{variant}", "variant": variant})

    all_results = {}

    for config in configs:
        print(f"\n{'='*50}\nConfig: {config['name']}\n{'='*50}")

        # Reset processor patch
        if hasattr(processor, "_original_get_gazing"):
            processor._get_gazing_info_from_videos = processor._original_get_gazing
        else:
            processor._original_get_gazing = processor._get_gazing_info_from_videos

        # Configure scoring path for this variant
        if config["variant"] == "rand":
            patch_processor_with_semantic_filter(
                processor, wrapper, clip_model, clip_tokenizer,
                mode="intersect",  # mode != "gaze_only" so the patch runs
                semantic_keep_ratio=args.semantic_keep_ratio,
                device=str(device),
                bypass_autogaze_selection=True,
                random_scoring=True,
            )
        else:
            patch_processor_with_semantic_filter(
                processor, wrapper, clip_model, clip_tokenizer,
                mode="intersect",
                semantic_keep_ratio=args.semantic_keep_ratio,
                device=str(device),
                bypass_autogaze_selection=True,
                score_provider=owlvit_provider,
            )

        per_q = []
        correct = 0
        total = 0
        latencies = []
        for i, sample in enumerate(samples):
            qid = sample["question_id"]
            q_for_score = None
            try:
                if config["variant"] == "match":
                    q_for_score = sample["question_stem"]
                elif config["variant"] == "shuf":
                    q_for_score = qid_to_shufq[qid]
                else:
                    q_for_score = sample["question_stem"]  # ignored under random_scoring

                # Set OWL-ViT query (only used if score_provider is wired)
                if config["variant"] in ("match", "shuf"):
                    owlvit_provider.set_query_text(q_for_score)

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
                    print(f"  [{i+1}/{len(samples)}] qid={qid} gt={gt} pred={pred} acc={correct}/{total} avg_lat={sum(latencies)/len(latencies):.2f}s",
                          flush=True)
            except Exception as e:
                print(f"  [error] qid={qid}: {type(e).__name__}: {e}", flush=True)
                per_q.append({
                    "qid": qid,
                    "gt": sample["answer"],
                    "pred": "ERROR",
                    "correct": 0,
                    "scoring_q": q_for_score,
                    "wall_s": -1,
                    "error": str(e),
                })

        all_results[config["name"]] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / max(total, 1),
            "avg_lat_s": sum(latencies) / max(len(latencies), 1) if latencies else 0,
            "per_q": per_q,
        }
        print(f"\n  >>> {config['name']}: {correct}/{total} = {correct/max(total,1):.4f}, "
              f"avg_lat = {all_results[config['name']]['avg_lat_s']:.2f}s")

    # ---- Paired-flip analysis ----
    print(f"\n{'='*60}\nPaired-flip analysis\n{'='*60}")
    paired = {}
    for a, b in [("owlvit_match", "owlvit_shuf"), ("owlvit_match", "owlvit_rand"),
                 ("owlvit_shuf", "owlvit_rand")]:
        a_pq = {p["qid"]: p["correct"] for p in all_results[a]["per_q"]}
        b_pq = {p["qid"]: p["correct"] for p in all_results[b]["per_q"]}
        common = set(a_pq.keys()) & set(b_pq.keys())
        a_wins = sum(1 for q in common if a_pq[q] > b_pq[q])
        b_wins = sum(1 for q in common if b_pq[q] > a_pq[q])
        net = a_wins - b_wins
        print(f"  {a:<14} vs {b:<14}: {a}-only={a_wins}  {b}-only={b_wins}  net={net:+d}  n={len(common)}")
        paired[f"{a}_vs_{b}"] = {"a_wins": a_wins, "b_wins": b_wins, "net": net, "n": len(common)}

    # ---- Save ----
    out = {
        "configs": list(all_results.keys()),
        "n_samples": len(samples),
        "summary": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_q"}
            for k, v in all_results.items()
        },
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
    p.add_argument("--model_path", default="Efficient-Large-Model/NVILA-8B-HD-Video")
    p.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    p.add_argument("--ckpt", default="results/bighead/best_bighead.pt",
                   help="(BigHead checkpoint kept for SemanticAutoGazeWrapper plumbing only)")
    p.add_argument("--head_type", default="bighead")
    p.add_argument("--parquet_path", default=PARQUET_PATH)
    p.add_argument("--video_dir", default="hlvid_videos/extracted_household/videos")
    p.add_argument("--n_samples", type=int, default=None,
                   help="Limit to first N samples (None = full subset)")
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--num_frames_thumbnail", type=int, default=16)
    p.add_argument("--max_tiles", type=int, default=4)
    p.add_argument("--gazing_ratio", type=float, default=0.20)
    p.add_argument("--semantic_keep_ratio", type=float, default=0.14)
    p.add_argument("--output_dir", default="results/owlvit_hlvid_vqa")
    args = p.parse_args()
    main(args)
