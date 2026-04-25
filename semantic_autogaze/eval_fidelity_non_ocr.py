"""r/filter-fidelity-non-ocr: text-conditioning fidelity benchmark on non-OCR
Kinetics-style videos (data/, 1002 YouTube clips).

Question (the user's): does the BigHead+CLIP filter actually pick text-relevant
patches when the videos are about generic semantic subjects (not OCR-heavy)?

Metric (no NVILA, no LLM): cosine similarity between
  (mean-pooled SigLIP features over filter's chosen patches)
and
  (mean-pooled SigLIP features over AutoGaze's gaze_only ratio=0.75 patches)

The "reference" pooled features represent what NVILA's downstream SigLIP would
encode if it saw the full AutoGaze gazed set. The candidate pooled features are
what it'd encode under the filter at a much lower keep ratio. High cosine = the
filter's selection captured the same "gist" as the reference at ~5× fewer tokens.

Configs at semantic_keep_ratio=0.20 (≈ 39 patches/frame on 14×14 grid + scales):
  A) auto_at_K   — vanilla AutoGaze gaze_only at gazing_ratio=0.20 (NO filter)
  B) filt_match  — BigHead with matched query, top-20%
  C) filt_shuf   — BigHead with shuffled-other-query text, top-20%
  D) filt_rand   — uniform-random scoring, top-20%

Key paired comparisons:
  match vs shuffled cos sim → text-conditioning fidelity (the missing diagnostic)
  match vs random cos sim   → does the filter beat random selection
  match vs auto_at_K        → does the filter approach AutoGaze's quality

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m semantic_autogaze.eval_fidelity_non_ocr \\
    --device cuda:0 --n_videos 50 --output_dir results/filter_fidelity_non_ocr
"""
from __future__ import annotations
import os, glob, random, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
import av

from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper


# Generic non-OCR semantic queries — all "subjects" / "objects" / "scenes"
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


def get_clip_text_embedding(text, model, tokenizer, device):
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = F.normalize(feats, dim=-1)
    return feats


@torch.no_grad()
def pool_pooled_feature(siglip_model, video_siglip, gazing_info):
    """Mean-pool SigLIP last_hidden_state over the kept positions."""
    out = siglip_model(video_siglip, gazing_info=gazing_info)
    hidden = out.last_hidden_state  # (1, K_kept, C) typically
    pad = (~gazing_info["if_padded_gazing"].bool())[:, :hidden.shape[1]].unsqueeze(-1).float()
    pooled = (hidden * pad).sum(1) / pad.sum(1).clamp(min=1)
    return F.normalize(pooled, dim=-1)


def main(args):
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("[setup] CLIP ViT-B/16 text encoder...")
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()

    print(f"[setup] SemanticAutoGazeWrapper (BigHead {args.ckpt})...")
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )
    autogaze_transform = AutoGazeImageProcessor.from_pretrained(args.autogaze_model)

    print("[setup] SigLIP-2 base patch16-224 (NVILA's vision encoder)...")
    from autogaze.vision_encoders.siglip import SiglipVisionModel
    from transformers import AutoImageProcessor
    siglip_transform = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
    siglip_model = SiglipVisionModel.from_pretrained(
        "google/siglip2-base-patch16-224",
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

    # Pre-embed queries (matched and the shuffle pool — same set, distinct)
    query_embs = {q: get_clip_text_embedding(q, clip_model, clip_tok, device) for q in queries}

    KEEP = args.keep_ratio
    # results[config][query] = list of cos sims
    results = {c: {q: [] for q in queries} for c in
               ["auto_at_K", "filt_match", "filt_shuf", "filt_rand"]}
    tokens_kept = {c: [] for c in
                   ["ref_auto_75", "auto_at_K", "filt_match", "filt_shuf", "filt_rand"]}
    timings_ms = {c: [] for c in ["wrapper_extract_hidden", "filter_get_scores"]}

    for vi, vp in enumerate(videos):
        try:
            container = av.open(vp)
            stream = container.streams.video[0]
            n_frames = stream.frames
            indices = list(range(min(16, n_frames or 16)))
            raw_video = read_video_pyav(container=container, indices=indices)
            container.close()
            if raw_video.shape[0] < 16:
                print(f"  [skip] {os.path.basename(vp)} — only {raw_video.shape[0]} frames")
                continue
            video_autogaze = transform_video_for_pytorch(raw_video, autogaze_transform)[None].to(device)
            video_siglip = transform_video_for_pytorch(raw_video, siglip_transform)[None].to(device)
        except Exception as e:
            print(f"  [error] {vp}: {e}")
            continue

        # ---- Extract hidden states once per video (cache for all queries) ----
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hidden = wrapper.extract_hidden_states(video_autogaze)  # (1, T*196, 192)
        torch.cuda.synchronize()
        timings_ms["wrapper_extract_hidden"].append((time.perf_counter() - t0) * 1000)

        # ---- Reference: AutoGaze gaze_only at 0.75 (the established reference) ----
        ref_info = wrapper.forward(
            video_autogaze, query_embs[queries[0]], mode="gaze_only",
            gazing_ratio=0.75, task_loss_requirement=0.7, semantic_keep_ratio=1.0,
        )
        ref_feat = pool_pooled_feature(siglip_model, video_siglip, ref_info)
        tokens_kept["ref_auto_75"].append(int((~ref_info["if_padded_gazing"]).sum().item()))

        # ---- Config A: AutoGaze gaze_only at K (matched-K reference) ----
        auto_info = wrapper.forward(
            video_autogaze, query_embs[queries[0]], mode="gaze_only",
            gazing_ratio=KEEP, task_loss_requirement=0.7, semantic_keep_ratio=1.0,
        )
        auto_feat = pool_pooled_feature(siglip_model, video_siglip, auto_info)
        cs_auto = F.cosine_similarity(ref_feat, auto_feat).item()
        tokens_kept["auto_at_K"].append(int((~auto_info["if_padded_gazing"]).sum().item()))
        for q in queries:
            results["auto_at_K"][q].append(cs_auto)  # query-independent

        for q in queries:
            text_emb_match = query_embs[q]

            # filt_match: BigHead with matched query
            torch.cuda.synchronize(); t0 = time.perf_counter()
            scores_m = wrapper.semantic_filter.get_scores(hidden, text_emb_match)
            torch.cuda.synchronize()
            timings_ms["filter_get_scores"].append((time.perf_counter() - t0) * 1000)
            info_m = wrapper.semantic_filter.scores_to_gazing_info(
                scores_m, keep_ratio=KEEP, num_frames=hidden.shape[1] // 196,
            )
            feat_m = pool_pooled_feature(siglip_model, video_siglip, info_m)
            cs_m = F.cosine_similarity(ref_feat, feat_m).item()
            results["filt_match"][q].append(cs_m)
            if vi == 0:
                tokens_kept["filt_match"].append(int((~info_m["if_padded_gazing"]).sum().item()))

            # filt_shuf: BigHead with shuffled query (any other query)
            other_queries = [qq for qq in queries if qq != q]
            shuffled_q = rng.choice(other_queries)
            text_emb_shuf = query_embs[shuffled_q]
            scores_s = wrapper.semantic_filter.get_scores(hidden, text_emb_shuf)
            info_s = wrapper.semantic_filter.scores_to_gazing_info(
                scores_s, keep_ratio=KEEP, num_frames=hidden.shape[1] // 196,
            )
            feat_s = pool_pooled_feature(siglip_model, video_siglip, info_s)
            cs_s = F.cosine_similarity(ref_feat, feat_s).item()
            results["filt_shuf"][q].append(cs_s)

            # filt_rand: uniform-random scores
            scores_r = torch.rand_like(scores_m)
            info_r = wrapper.semantic_filter.scores_to_gazing_info(
                scores_r, keep_ratio=KEEP, num_frames=hidden.shape[1] // 196,
            )
            feat_r = pool_pooled_feature(siglip_model, video_siglip, info_r)
            cs_r = F.cosine_similarity(ref_feat, feat_r).item()
            results["filt_rand"][q].append(cs_r)

        if (vi + 1) % 5 == 0 or vi == 0:
            print(f"  [{vi+1}/{len(videos)}] {os.path.basename(vp)}")

    # --- Aggregate ---
    print("\n" + "=" * 78)
    print(f"FIDELITY RESULTS  n_videos={len(videos)}  queries={len(queries)}  keep_ratio={KEEP}")
    print("=" * 78)

    summary = {"keep_ratio": KEEP, "n_videos_attempted": len(videos),
               "queries": queries, "results": {}, "tokens_kept": {}}

    print(f"\n{'config':<14}  {'cos_sim mean':>13}  {'cos_sim std':>11}  {'n':>5}")
    print("-" * 50)
    for cfg in ["auto_at_K", "filt_match", "filt_shuf", "filt_rand"]:
        all_sims = [s for q_list in results[cfg].values() for s in q_list]
        mn, sd = float(np.mean(all_sims)), float(np.std(all_sims))
        print(f"{cfg:<14}  {mn:>13.4f}  {sd:>11.4f}  {len(all_sims):>5}")
        summary["results"][cfg] = {"mean_cos_sim": mn, "std_cos_sim": sd, "n": len(all_sims),
                                   "by_query": {q: float(np.mean(v)) for q, v in results[cfg].items()}}

    # Paired diff per (video, query)
    print(f"\n{'pair':<28}  {'mean delta':>10}  {'std':>8}  {'n':>5}")
    print("-" * 60)
    pairs = [
        ("filt_match - filt_shuf", "filt_match", "filt_shuf"),
        ("filt_match - filt_rand", "filt_match", "filt_rand"),
        ("filt_shuf - filt_rand", "filt_shuf", "filt_rand"),
        ("auto_at_K - filt_match", "auto_at_K", "filt_match"),
    ]
    paired = {}
    for label, a, b in pairs:
        deltas = []
        for q in queries:
            for sa, sb in zip(results[a][q], results[b][q]):
                deltas.append(sa - sb)
        mn, sd = float(np.mean(deltas)), float(np.std(deltas))
        # naive z-test for mean=0
        z = mn / (sd / np.sqrt(len(deltas))) if sd > 0 else 0.0
        print(f"{label:<28}  {mn:>+10.4f}  {sd:>8.4f}  {len(deltas):>5}  z={z:.2f}")
        paired[label] = {"mean": mn, "std": sd, "n": len(deltas), "z": z}
    summary["paired"] = paired

    # Per-query breakdown for the headline match-shuf comparison
    print(f"\nPer-query (filt_match - filt_shuf):")
    for q in queries:
        ms = results["filt_match"][q]
        ss = results["filt_shuf"][q]
        if len(ms) == 0:
            continue
        deltas = [a - b for a, b in zip(ms, ss)]
        print(f"  {q:<20s}  match={np.mean(ms):.4f}  shuf={np.mean(ss):.4f}  delta={np.mean(deltas):+.4f}  n={len(ms)}")

    # Latency summary
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
    p.add_argument("--ckpt", default="results/bighead/best_bighead.pt")
    p.add_argument("--head_type", default="bighead")
    p.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/filter_fidelity_non_ocr")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
