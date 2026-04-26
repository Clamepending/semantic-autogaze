"""r/query-direct-relevance-metric cycle 1: re-evaluate scorers with a query-DIRECT
fidelity metric (NOT vs AutoGaze reference).

The cos-sim-vs-AutoGaze-reference metric used in r/filter-fidelity-non-ocr,
r/raw-clip-scoring-baseline, r/siglip2-patch-scoring, and r/phrase-grounding-teacher
is BIASED against true text-conditioning for non-main-subject queries: AutoGaze's
reference is dominated by salient main-subject patches, so a correctly text-
conditioned filter on "the sky" is penalized for picking sky patches AutoGaze
doesn't include.

This script computes a query-DIRECT metric instead:
  query_relevance = cos( SigLIP-pool(kept_patches), SigLIP-text-emb(query) )

If a scorer is text-conditioned, its matched-Q selection should produce higher
query-relevance than its shuffled-Q or random selection.

Cycle 1 focuses on OWL-ViT (the scorer that showed dichotomy in
r/phrase-grounding-teacher). Optional --include_siglip2 also runs raw SigLIP-2
as a control (expected to show no signal under the new metric, confirming the
prior "no patch-text alignment" finding for CLIP-family).

Usage:
  CUDA_VISIBLE_DEVICES=N python -m semantic_autogaze.eval_fidelity_query_direct \\
    --device cuda:0 --n_videos 50 --output_dir results/query_direct_metric
"""
from __future__ import annotations
import os, glob, random, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
import av

from autogaze.models.autogaze import AutoGazeImageProcessor
from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper


DEFAULT_QUERIES = [
    "a person",
    "a face",
    "an animal",
    "a vehicle",
    "a building",
    "the sky",
    "water",
    "food",
]

GRID_TARGET = 14
N_PATCHES_TARGET = GRID_TARGET * GRID_TARGET  # 196
OWLVIT_NAME = "google/owlvit-base-patch32"
SIGLIP2_NAME = "google/siglip2-base-patch16-224"


# ---------- Scorer 1: OWL-ViT (the candidate text-conditioned scorer) ----------

@torch.no_grad()
def encode_owlvit_patches(owlvit_det, pixel_values_b3hw, target_grid=GRID_TARGET):
    image_embeds, _ = owlvit_det.image_embedder(pixel_values=pixel_values_b3hw)
    T, hp, wp, D = image_embeds.shape
    image_class_embeds = owlvit_det.class_head.dense0(image_embeds)
    x = image_class_embeds.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(target_grid, target_grid), mode="bilinear", align_corners=False)
    x = x.permute(0, 2, 3, 1).contiguous().view(T, target_grid * target_grid, -1)
    x = F.normalize(x, dim=-1)
    return x


@torch.no_grad()
def encode_owlvit_text(owlvit_model, tokenizer, text, device):
    enc = tokenizer([text], padding="max_length", return_tensors="pt").to(device)
    out = owlvit_model.text_model(input_ids=enc["input_ids"])
    pooled = owlvit_model.text_projection(out.pooler_output)
    return F.normalize(pooled, dim=-1)


# ---------- Scorer 2: SigLIP-2 patch (control, expected null under new metric) ----------

@torch.no_grad()
def encode_siglip2_patches(siglip2_model, pixel_values_b3hw):
    out = siglip2_model.vision_model(pixel_values=pixel_values_b3hw, interpolate_pos_encoding=False)
    return F.normalize(out.last_hidden_state, dim=-1)  # (T, 196, 768)


@torch.no_grad()
def encode_siglip2_text_for_scoring(siglip2_model, tokenizer, text, device):
    """Text emb for SCORING patches (siglip2_match config) — different scaling than
    the relevance-metric text emb."""
    enc = tokenizer([text], padding="max_length", return_tensors="pt").to(device)
    out = siglip2_model.text_model(input_ids=enc["input_ids"])
    return F.normalize(out.pooler_output, dim=-1)


# ---------- Query-direct relevance metric ----------

@torch.no_grad()
def query_text_emb_for_relevance(siglip2_model, tokenizer, text, device):
    """Compute SigLIP-2 text embedding USED AS THE TARGET in the query-direct
    relevance metric. This is the same SigLIP-2 text model — pooler_output."""
    enc = tokenizer([text], padding="max_length", return_tensors="pt").to(device)
    out = siglip2_model.text_model(input_ids=enc["input_ids"])
    return F.normalize(out.pooler_output, dim=-1)  # (1, 768)


@torch.no_grad()
def pool_features(siglip_model, video_siglip, gazing_info):
    out = siglip_model(video_siglip, gazing_info=gazing_info)
    hidden = out.last_hidden_state
    pad = (~gazing_info["if_padded_gazing"].bool())[:, :hidden.shape[1]].unsqueeze(-1).float()
    pooled = (hidden * pad).sum(1) / pad.sum(1).clamp(min=1)
    return F.normalize(pooled, dim=-1)  # (1, 768) — NVILA-SigLIP-pool space


def patch_features_to_scores(patch_features_t196d, text_emb_1d, n_patches=N_PATCHES_TARGET):
    T = patch_features_t196d.shape[0]
    flat = patch_features_t196d.reshape(T * n_patches, -1)
    cos = (flat @ text_emb_1d.T).squeeze(-1)
    scores = torch.sigmoid(cos * 10.0)
    return scores.unsqueeze(0)


def main(args):
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[setup] OWL-ViT: {OWLVIT_NAME}")
    from transformers import OwlViTForObjectDetection, AutoTokenizer, AutoImageProcessor, AutoModel
    owlvit_det = OwlViTForObjectDetection.from_pretrained(OWLVIT_NAME).to(device).eval()
    owlvit_tok = AutoTokenizer.from_pretrained(OWLVIT_NAME)
    owlvit_imgproc = AutoImageProcessor.from_pretrained(OWLVIT_NAME)
    owlvit_size = owlvit_imgproc.size["height"] if isinstance(owlvit_imgproc.size, dict) else 768
    owlvit_mean = torch.tensor(owlvit_imgproc.image_mean, device=device)
    owlvit_std = torch.tensor(owlvit_imgproc.image_std, device=device)

    print(f"[setup] SigLIP-2 (text encoder for query-direct metric, AND patch scoring control): {SIGLIP2_NAME}")
    siglip2_model = AutoModel.from_pretrained(SIGLIP2_NAME).to(device).eval()
    siglip2_tok = AutoTokenizer.from_pretrained(SIGLIP2_NAME)
    siglip2_imgproc = AutoImageProcessor.from_pretrained(SIGLIP2_NAME)
    siglip2_size = (siglip2_imgproc.size["height"], siglip2_imgproc.size["width"]) \
        if isinstance(siglip2_imgproc.size, dict) else (224, 224)
    siglip2_mean = torch.tensor(siglip2_imgproc.image_mean, device=device)
    siglip2_std = torch.tensor(siglip2_imgproc.image_std, device=device)

    print(f"[setup] SemanticAutoGazeWrapper (AutoGaze ref + scores_to_gazing_info)...")
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )
    autogaze_transform = AutoGazeImageProcessor.from_pretrained(args.autogaze_model)

    print("[setup] NVILA SigLIP-2 multiscale (downstream feature pool)...")
    from autogaze.vision_encoders.siglip import SiglipVisionModel
    nvila_siglip_imgproc = AutoImageProcessor.from_pretrained(SIGLIP2_NAME)
    nvila_siglip = SiglipVisionModel.from_pretrained(
        SIGLIP2_NAME,
        scales=wrapper.autogaze.config.scales,
        attn_implementation="sdpa",
    ).to(device).eval()

    queries = [q.strip() for q in args.queries.split(",")]
    print(f"[data] queries: {queries}")

    videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    rng = random.Random(args.seed)
    rng.shuffle(videos)
    videos = videos[:args.n_videos]
    print(f"[data] {len(videos)} videos × {len(queries)} queries = {len(videos) * len(queries)} measurements")
    print(f"[data] include_siglip2 control: {args.include_siglip2}")

    # Pre-compute query text embeddings used by each system
    owlvit_query_embs = {q: encode_owlvit_text(owlvit_det.owlvit, owlvit_tok, q, device) for q in queries}
    siglip2_query_embs = {q: encode_siglip2_text_for_scoring(siglip2_model, siglip2_tok, q, device) for q in queries}
    relevance_query_embs = siglip2_query_embs  # same model & space — pooler_output of SigLIP-2 text encoder

    KEEP = args.keep_ratio

    # Scorer configs for each test variant
    scorer_keys = ["owlvit"]
    if args.include_siglip2:
        scorer_keys.append("siglip2")

    # Two metrics, three variants per scorer + auto_at_K reference
    METRICS = ["query_direct", "vs_autogaze_ref"]
    config_keys = ["auto_at_K"]
    for sk in scorer_keys:
        for v in ["match", "shuf", "rand"]:
            config_keys.append(f"{sk}_{v}")

    results = {m: {c: {q: [] for q in queries} for c in config_keys} for m in METRICS}

    for vi, vp in enumerate(videos):
        try:
            container = av.open(vp)
            stream = container.streams.video[0]
            n_frames = stream.frames
            indices = list(range(min(16, n_frames or 16)))
            raw_video = read_video_pyav(container=container, indices=indices)
            container.close()
            if raw_video.shape[0] < 16:
                continue
            video_autogaze = transform_video_for_pytorch(raw_video, autogaze_transform)[None].to(device)
            video_siglip = transform_video_for_pytorch(raw_video, nvila_siglip_imgproc)[None].to(device)

            # OWL-ViT preprocessing
            vid_owlvit = torch.from_numpy(raw_video).permute(0, 3, 1, 2).float().to(device) / 255.0
            vid_owlvit = F.interpolate(vid_owlvit, size=(owlvit_size, owlvit_size), mode="bicubic", align_corners=False)
            vid_owlvit = (vid_owlvit - owlvit_mean[None, :, None, None]) / owlvit_std[None, :, None, None]

            # SigLIP-2 preprocessing (only used if include_siglip2)
            vid_siglip2 = None
            if args.include_siglip2:
                vid_siglip2 = torch.from_numpy(raw_video).permute(0, 3, 1, 2).float().to(device) / 255.0
                vid_siglip2 = F.interpolate(vid_siglip2, size=siglip2_size, mode="bicubic", align_corners=False)
                vid_siglip2 = (vid_siglip2 - siglip2_mean[None, :, None, None]) / siglip2_std[None, :, None, None]
        except Exception as e:
            print(f"  [error] {os.path.basename(vp)}: {e}", flush=True)
            continue

        # AutoGaze reference (gaze_only at 0.75 — feeds the vs_autogaze_ref metric)
        ref_info = wrapper.forward(
            video_autogaze, owlvit_query_embs[queries[0]], mode="gaze_only",
            gazing_ratio=0.75, task_loss_requirement=0.7, semantic_keep_ratio=1.0,
        )
        ref_pool = pool_features(nvila_siglip, video_siglip, ref_info)

        # auto_at_K — query-INDEPENDENT (AutoGaze ignores text)
        auto_info = wrapper.forward(
            video_autogaze, owlvit_query_embs[queries[0]], mode="gaze_only",
            gazing_ratio=KEEP, task_loss_requirement=0.7, semantic_keep_ratio=1.0,
        )
        auto_pool = pool_features(nvila_siglip, video_siglip, auto_info)
        cs_auto_vs_ref = F.cosine_similarity(ref_pool, auto_pool).item()
        for q in queries:
            qte = relevance_query_embs[q]
            cs_auto_vs_query = F.cosine_similarity(auto_pool, qte).item()
            results["vs_autogaze_ref"]["auto_at_K"][q].append(cs_auto_vs_ref)
            results["query_direct"]["auto_at_K"][q].append(cs_auto_vs_query)

        # Pre-extract patch features per scorer (once per video)
        owlvit_patch_feats = encode_owlvit_patches(owlvit_det, vid_owlvit)
        siglip2_patch_feats = None
        if args.include_siglip2:
            siglip2_patch_feats = encode_siglip2_patches(siglip2_model, vid_siglip2)

        for q in queries:
            qte_relevance = relevance_query_embs[q]
            other_queries = [qq for qq in queries if qq != q]
            shuf_q = rng.choice(other_queries)

            # OWL-ViT scorer
            for variant in ["match", "shuf", "rand"]:
                if variant == "match":
                    text_emb = owlvit_query_embs[q]
                    scores = patch_features_to_scores(owlvit_patch_feats, text_emb)
                elif variant == "shuf":
                    text_emb = owlvit_query_embs[shuf_q]
                    scores = patch_features_to_scores(owlvit_patch_feats, text_emb)
                else:
                    scores = torch.sigmoid(torch.randn(1, owlvit_patch_feats.shape[0] * N_PATCHES_TARGET, device=device))
                info = wrapper.semantic_filter.scores_to_gazing_info(
                    scores, keep_ratio=KEEP, num_frames=owlvit_patch_feats.shape[0],
                )
                pool = pool_features(nvila_siglip, video_siglip, info)
                cs_vs_ref = F.cosine_similarity(ref_pool, pool).item()
                cs_vs_query = F.cosine_similarity(pool, qte_relevance).item()
                cfg = f"owlvit_{variant}"
                results["vs_autogaze_ref"][cfg][q].append(cs_vs_ref)
                results["query_direct"][cfg][q].append(cs_vs_query)

            # SigLIP-2 scorer (control)
            if args.include_siglip2:
                for variant in ["match", "shuf", "rand"]:
                    if variant == "match":
                        text_emb = siglip2_query_embs[q]
                        scores = patch_features_to_scores(siglip2_patch_feats, text_emb)
                    elif variant == "shuf":
                        text_emb = siglip2_query_embs[shuf_q]
                        scores = patch_features_to_scores(siglip2_patch_feats, text_emb)
                    else:
                        scores = torch.sigmoid(torch.randn(1, siglip2_patch_feats.shape[0] * N_PATCHES_TARGET, device=device))
                    info = wrapper.semantic_filter.scores_to_gazing_info(
                        scores, keep_ratio=KEEP, num_frames=siglip2_patch_feats.shape[0],
                    )
                    pool = pool_features(nvila_siglip, video_siglip, info)
                    cs_vs_ref = F.cosine_similarity(ref_pool, pool).item()
                    cs_vs_query = F.cosine_similarity(pool, qte_relevance).item()
                    cfg = f"siglip2_{variant}"
                    results["vs_autogaze_ref"][cfg][q].append(cs_vs_ref)
                    results["query_direct"][cfg][q].append(cs_vs_query)

        if (vi + 1) % 5 == 0 or vi == 0:
            print(f"  [{vi+1}/{len(videos)}] {os.path.basename(vp)}", flush=True)

    print("\n" + "=" * 78, flush=True)
    print(f"QUERY-DIRECT FIDELITY  n_videos={len(videos)}  queries={len(queries)}  keep_ratio={KEEP}")
    print("=" * 78)

    summary = {"keep_ratio": KEEP, "queries": queries, "include_siglip2": args.include_siglip2,
               "metrics": {}}
    for metric in METRICS:
        print(f"\n--- METRIC: {metric} ---")
        print(f"{'config':<14}  {'cos_sim mean':>13}  {'cos_sim std':>11}  {'n':>5}")
        print("-" * 50)
        m_results = {"results": {}, "paired": {}, "per_query": {}}
        for cfg in config_keys:
            all_sims = [s for qs in results[metric][cfg].values() for s in qs]
            mn, sd = float(np.mean(all_sims)), float(np.std(all_sims))
            print(f"{cfg:<14}  {mn:>13.4f}  {sd:>11.4f}  {len(all_sims):>5}")
            m_results["results"][cfg] = {"mean_cos_sim": mn, "std_cos_sim": sd, "n": len(all_sims),
                                          "by_query": {q: float(np.mean(v)) for q, v in results[metric][cfg].items()}}

        # Paired diffs per scorer (match vs shuf, match vs rand, shuf vs rand)
        print(f"\n{'pair':<32}  {'mean delta':>10}  {'std':>8}  {'n':>5}  {'z':>6}")
        print("-" * 70)
        pair_specs = []
        for sk in scorer_keys:
            pair_specs += [
                (f"{sk}_match - {sk}_shuf", f"{sk}_match", f"{sk}_shuf"),
                (f"{sk}_match - {sk}_rand", f"{sk}_match", f"{sk}_rand"),
                (f"{sk}_shuf  - {sk}_rand", f"{sk}_shuf", f"{sk}_rand"),
                (f"auto_at_K   - {sk}_match", "auto_at_K", f"{sk}_match"),
            ]
        for label, a, b in pair_specs:
            deltas = []
            for q in queries:
                for sa, sb in zip(results[metric][a][q], results[metric][b][q]):
                    deltas.append(sa - sb)
            if not deltas:
                continue
            mn, sd = float(np.mean(deltas)), float(np.std(deltas))
            z = mn / (sd / np.sqrt(len(deltas))) if sd > 0 else 0.0
            print(f"{label:<32}  {mn:>+10.4f}  {sd:>8.4f}  {len(deltas):>5}  z={z:.2f}")
            m_results["paired"][label] = {"mean": mn, "std": sd, "n": len(deltas), "z": z}

        # Per-query for owlvit
        print(f"\nPer-query (owlvit_match - owlvit_shuf) under {metric}:")
        per_q_dict = {}
        for q in queries:
            ms = results[metric]["owlvit_match"][q]
            ss = results[metric]["owlvit_shuf"][q]
            rs = results[metric]["owlvit_rand"][q]
            if not ms:
                continue
            d_ms = [a - b for a, b in zip(ms, ss)]
            d_mr = [a - b for a, b in zip(ms, rs)]
            print(f"  {q:<14s}  match={np.mean(ms):.4f}  shuf={np.mean(ss):.4f}  rand={np.mean(rs):.4f}  "
                  f"m-s={np.mean(d_ms):+.4f}  m-r={np.mean(d_mr):+.4f}")
            per_q_dict[q] = {
                "match_mean": float(np.mean(ms)),
                "shuf_mean": float(np.mean(ss)),
                "rand_mean": float(np.mean(rs)),
                "m_minus_s": float(np.mean(d_ms)),
                "m_minus_r": float(np.mean(d_mr)),
            }
        m_results["per_query_owlvit"] = per_q_dict
        summary["metrics"][metric] = m_results

    out_path = os.path.join(args.output_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", default="data")
    p.add_argument("--n_videos", type=int, default=50)
    p.add_argument("--queries", default=",".join(DEFAULT_QUERIES))
    p.add_argument("--keep_ratio", type=float, default=0.20)
    p.add_argument("--ckpt", default="results/bighead/best_bighead.pt")
    p.add_argument("--head_type", default="bighead")
    p.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/query_direct_metric")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include_siglip2", action="store_true",
                   help="Also run raw SigLIP-2 as a control (expected null under new metric).")
    args = p.parse_args()
    main(args)
