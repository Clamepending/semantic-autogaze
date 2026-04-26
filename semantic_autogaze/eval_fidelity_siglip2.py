"""r/siglip2-patch-scoring: drop CLIP, score patches via SigLIP-2 patch × SigLIP-2 text.

Mirrors r/raw-clip-scoring-baseline (e95928c) but replaces CLIP visual+text with
google/siglip2-base-patch16-224. SigLIP-2 was trained with sigmoid contrastive loss
on image-text pairs and may produce stronger patch-text alignment than CLIP InfoNCE.
NVILA already uses this backbone for its visual tokens, so this is the closest cheap
diagnostic before committing to multi-week phrase-grounding teacher work.

Decisive split:
  match > shuffled by z>2  → SigLIP-2 IS patch-text aligned. Cheap drop-in
                              available; row 1 of QUEUE answered.
  match ~ shuffled         → SigLIP-2 also fails. CLIP-family scorers are out;
                              phrase-grounding teacher is unavoidable.

Usage:
  CUDA_VISIBLE_DEVICES=3 python -m semantic_autogaze.eval_fidelity_siglip2 \\
    --device cuda:0 --n_videos 50 --output_dir results/siglip2_patch_scoring
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

GRID = 14
N_PATCHES_PER_FRAME = 196
SIGLIP2_NAME = "google/siglip2-base-patch16-224"


@torch.no_grad()
def encode_siglip2_patches(siglip2_model, pixel_values_b3hw):
    """Run SigLIP-2 vision encoder, return (T, 196, 768) per-patch features (L2-normalized)."""
    out = siglip2_model.vision_model(pixel_values=pixel_values_b3hw, interpolate_pos_encoding=False)
    patch = out.last_hidden_state  # (T, 196, 768)
    patch = F.normalize(patch, dim=-1)
    return patch


@torch.no_grad()
def encode_siglip2_text(siglip2_model, tokenizer, text, device):
    """Return (1, 768) L2-normalized text embedding."""
    enc = tokenizer([text], padding="max_length", return_tensors="pt").to(device)
    out = siglip2_model.text_model(input_ids=enc["input_ids"])
    pooled = out.pooler_output  # (1, 768)
    pooled = F.normalize(pooled, dim=-1)
    return pooled


def patch_features_to_scores(patch_features_t196d, text_emb_1d):
    """Cosine score per patch -> sigmoid-shaped scores in [0,1]. (Ranking is what matters.)"""
    T = patch_features_t196d.shape[0]
    flat = patch_features_t196d.reshape(T * N_PATCHES_PER_FRAME, -1)  # (T*196, 768)
    cos = (flat @ text_emb_1d.T).squeeze(-1)
    scores = torch.sigmoid(cos * 10.0)
    return scores.unsqueeze(0)


@torch.no_grad()
def pool_pooled_feature(siglip_model, video_siglip, gazing_info):
    out = siglip_model(video_siglip, gazing_info=gazing_info)
    hidden = out.last_hidden_state
    pad = (~gazing_info["if_padded_gazing"].bool())[:, :hidden.shape[1]].unsqueeze(-1).float()
    pooled = (hidden * pad).sum(1) / pad.sum(1).clamp(min=1)
    return F.normalize(pooled, dim=-1)


def main(args):
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[setup] SigLIP-2 model: {SIGLIP2_NAME}")
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
    siglip2_model = AutoModel.from_pretrained(SIGLIP2_NAME).to(device).eval()
    siglip2_tok = AutoTokenizer.from_pretrained(SIGLIP2_NAME)
    siglip2_imgproc = AutoImageProcessor.from_pretrained(SIGLIP2_NAME)
    siglip2_mean = torch.tensor(siglip2_imgproc.image_mean, device=device)
    siglip2_std = torch.tensor(siglip2_imgproc.image_std, device=device)
    siglip2_size = (siglip2_imgproc.size["height"], siglip2_imgproc.size["width"]) \
        if isinstance(siglip2_imgproc.size, dict) else (224, 224)
    print(f"[setup] SigLIP-2 size={siglip2_size} mean={siglip2_mean.tolist()} std={siglip2_std.tolist()}")

    print(f"[setup] SemanticAutoGazeWrapper (for AutoGaze ref + scores_to_gazing_info)...")
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )
    autogaze_transform = AutoGazeImageProcessor.from_pretrained(args.autogaze_model)

    print("[setup] NVILA SigLIP-2 (multiscale) for downstream feature pool...")
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

    query_embs = {q: encode_siglip2_text(siglip2_model, siglip2_tok, q, device) for q in queries}

    KEEP = args.keep_ratio
    results = {c: {q: [] for q in queries} for c in
               ["auto_at_K", "siglip2_match", "siglip2_shuf", "siglip2_rand"]}
    timings_ms = {c: [] for c in ["siglip2_visual_extract", "siglip2_score"]}
    tokens_kept = {c: [] for c in ["ref_auto_75", "auto_at_K", "siglip2_match"]}

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

            # SigLIP-2 preprocessing: bicubic-resize to 224 + normalize with SigLIP mean/std
            vid = torch.from_numpy(raw_video).permute(0, 3, 1, 2).float().to(device) / 255.0
            vid = F.interpolate(vid, size=siglip2_size, mode="bicubic", align_corners=False)
            vid = (vid - siglip2_mean[None, :, None, None]) / siglip2_std[None, :, None, None]
        except Exception as e:
            print(f"  [error] {os.path.basename(vp)}: {e}")
            continue

        # Reference: AutoGaze gaze_only at 0.75
        ref_info = wrapper.forward(
            video_autogaze, query_embs[queries[0]], mode="gaze_only",
            gazing_ratio=0.75, task_loss_requirement=0.7, semantic_keep_ratio=1.0,
        )
        ref_feat = pool_pooled_feature(nvila_siglip, video_siglip, ref_info)
        tokens_kept["ref_auto_75"].append(int((~ref_info["if_padded_gazing"]).sum().item()))

        auto_info = wrapper.forward(
            video_autogaze, query_embs[queries[0]], mode="gaze_only",
            gazing_ratio=KEEP, task_loss_requirement=0.7, semantic_keep_ratio=1.0,
        )
        auto_feat = pool_pooled_feature(nvila_siglip, video_siglip, auto_info)
        cs_auto = F.cosine_similarity(ref_feat, auto_feat).item()
        tokens_kept["auto_at_K"].append(int((~auto_info["if_padded_gazing"]).sum().item()))
        for q in queries:
            results["auto_at_K"][q].append(cs_auto)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        patch_feats = encode_siglip2_patches(siglip2_model, vid)  # (T, 196, 768)
        torch.cuda.synchronize()
        timings_ms["siglip2_visual_extract"].append((time.perf_counter() - t0) * 1000)

        for q in queries:
            text_emb = query_embs[q]

            torch.cuda.synchronize(); t0 = time.perf_counter()
            scores_m = patch_features_to_scores(patch_feats, text_emb)
            torch.cuda.synchronize()
            timings_ms["siglip2_score"].append((time.perf_counter() - t0) * 1000)
            info_m = wrapper.semantic_filter.scores_to_gazing_info(
                scores_m, keep_ratio=KEEP, num_frames=patch_feats.shape[0],
            )
            feat_m = pool_pooled_feature(nvila_siglip, video_siglip, info_m)
            cs_m = F.cosine_similarity(ref_feat, feat_m).item()
            results["siglip2_match"][q].append(cs_m)
            if vi == 0:
                tokens_kept["siglip2_match"].append(int((~info_m["if_padded_gazing"]).sum().item()))

            other_queries = [qq for qq in queries if qq != q]
            shuffled_q = rng.choice(other_queries)
            text_emb_s = query_embs[shuffled_q]
            scores_s = patch_features_to_scores(patch_feats, text_emb_s)
            info_s = wrapper.semantic_filter.scores_to_gazing_info(
                scores_s, keep_ratio=KEEP, num_frames=patch_feats.shape[0],
            )
            feat_s = pool_pooled_feature(nvila_siglip, video_siglip, info_s)
            cs_s = F.cosine_similarity(ref_feat, feat_s).item()
            results["siglip2_shuf"][q].append(cs_s)

            scores_r = torch.rand_like(scores_m)
            info_r = wrapper.semantic_filter.scores_to_gazing_info(
                scores_r, keep_ratio=KEEP, num_frames=patch_feats.shape[0],
            )
            feat_r = pool_pooled_feature(nvila_siglip, video_siglip, info_r)
            cs_r = F.cosine_similarity(ref_feat, feat_r).item()
            results["siglip2_rand"][q].append(cs_r)

        if (vi + 1) % 5 == 0 or vi == 0:
            print(f"  [{vi+1}/{len(videos)}] {os.path.basename(vp)}")

    print("\n" + "=" * 78)
    print(f"SIGLIP-2 FIDELITY  n_videos={len(videos)}  queries={len(queries)}  keep_ratio={KEEP}")
    print("=" * 78)

    summary = {"keep_ratio": KEEP, "queries": queries, "results": {}, "model": SIGLIP2_NAME}

    print(f"\n{'config':<14}  {'cos_sim mean':>13}  {'cos_sim std':>11}  {'n':>5}")
    print("-" * 50)
    for cfg in ["auto_at_K", "siglip2_match", "siglip2_shuf", "siglip2_rand"]:
        all_sims = [s for q_list in results[cfg].values() for s in q_list]
        mn, sd = float(np.mean(all_sims)), float(np.std(all_sims))
        print(f"{cfg:<14}  {mn:>13.4f}  {sd:>11.4f}  {len(all_sims):>5}")
        summary["results"][cfg] = {"mean_cos_sim": mn, "std_cos_sim": sd, "n": len(all_sims),
                                   "by_query": {q: float(np.mean(v)) for q, v in results[cfg].items()}}

    print(f"\n{'pair':<32}  {'mean delta':>10}  {'std':>8}  {'n':>5}  {'z':>6}")
    print("-" * 70)
    pairs = [
        ("siglip2_match - siglip2_shuf", "siglip2_match", "siglip2_shuf"),
        ("siglip2_match - siglip2_rand", "siglip2_match", "siglip2_rand"),
        ("siglip2_shuf  - siglip2_rand", "siglip2_shuf", "siglip2_rand"),
        ("auto_at_K     - siglip2_match", "auto_at_K", "siglip2_match"),
    ]
    paired = {}
    for label, a, b in pairs:
        deltas = []
        for q in queries:
            for sa, sb in zip(results[a][q], results[b][q]):
                deltas.append(sa - sb)
        mn, sd = float(np.mean(deltas)), float(np.std(deltas))
        z = mn / (sd / np.sqrt(len(deltas))) if sd > 0 else 0.0
        print(f"{label:<32}  {mn:>+10.4f}  {sd:>8.4f}  {len(deltas):>5}  z={z:.2f}")
        paired[label] = {"mean": mn, "std": sd, "n": len(deltas), "z": z}
    summary["paired"] = paired

    print(f"\nPer-query (siglip2_match - siglip2_shuf):")
    for q in queries:
        ms = results["siglip2_match"][q]
        ss = results["siglip2_shuf"][q]
        rs = results["siglip2_rand"][q]
        if not ms:
            continue
        deltas_ms = [a - b for a, b in zip(ms, ss)]
        deltas_mr = [a - b for a, b in zip(ms, rs)]
        print(f"  {q:<14s}  match={np.mean(ms):.4f}  shuf={np.mean(ss):.4f}  rand={np.mean(rs):.4f}  "
              f"m-s={np.mean(deltas_ms):+.4f}  m-r={np.mean(deltas_mr):+.4f}")

    print(f"\nLatency:")
    for k, v in timings_ms.items():
        if v:
            print(f"  {k:<28s}  {np.mean(v):6.2f} ± {np.std(v):5.2f} ms  (n={len(v)})")
    summary["latency_ms"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                             for k, v in timings_ms.items() if v}
    summary["tokens_kept_mean"] = {k: float(np.mean(v)) for k, v in tokens_kept.items() if v}

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
    p.add_argument("--ckpt", default="results/bighead/best_bighead.pt",
                   help="(only used for SemanticAutoGazeWrapper plumbing — scoring is by SigLIP-2)")
    p.add_argument("--head_type", default="bighead")
    p.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/siglip2_patch_scoring")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
