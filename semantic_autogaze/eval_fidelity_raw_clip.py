"""r/raw-clip-scoring-baseline: drop the BigHead, score patches via raw CLIP.

Mirrors r/filter-fidelity-non-ocr's setup but replaces BigHead with:

  patch_features = CLIP_visual(frame).patch_tokens × CLIP.visual.proj   # (T*196, 512)
  patch_scores   = patch_features @ text_emb                             # (T*196,)

No head, no training. Tests whether *raw CLIP* shows text-conditioning on the
same 50 Kinetics-style videos × 8 generic queries × 4 configs benchmark.

Decisive split:
  match > shuffled by z>2  → BigHead's CLIPSeg distillation is the bottleneck
                              (the architecture CAN attend to text but the
                              training pipeline doesn't make it).
                              Route to row 2: bighead-contrastive-retrain.
  match ≈ shuffled         → CLIP itself is the bottleneck.
                              Route to row 3: phrase-grounding-teacher.

Usage:
  CUDA_VISIBLE_DEVICES=2 python -m semantic_autogaze.eval_fidelity_raw_clip \\
    --device cuda:0 --n_videos 50 --output_dir results/raw_clip_scoring_baseline
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


@torch.no_grad()
def encode_clip_patches(clip_model, video_thwc_norm_clip, device):
    """Run CLIP visual encoder on each frame, return (T, 196, 512) projected & normalized.

    `video_thwc_norm_clip` is already CLIP-preprocessed (normalize + resize-to-224).
    """
    clip_model.visual.output_tokens = True
    T = video_thwc_norm_clip.shape[0]
    feats = []
    for t in range(T):
        frame = video_thwc_norm_clip[t:t+1]  # (1, 3, 224, 224)
        pooled, patch_tokens = clip_model.visual(frame)  # (1,512), (1,196,768)
        # Project patch tokens 768 → 512 (same projection used for cls)
        if clip_model.visual.proj is not None:
            patch_proj = patch_tokens @ clip_model.visual.proj  # (1, 196, 512)
        else:
            patch_proj = patch_tokens
        patch_proj = F.normalize(patch_proj, dim=-1)
        feats.append(patch_proj.squeeze(0))  # (196, 512)
    feats = torch.stack(feats, dim=0)  # (T, 196, 512)
    clip_model.visual.output_tokens = False
    return feats


@torch.no_grad()
def get_clip_text_embedding(text, clip_model, tokenizer, device):
    tokens = tokenizer([text]).to(device)
    feats = clip_model.encode_text(tokens)
    feats = F.normalize(feats, dim=-1)
    return feats  # (1, 512)


def patch_features_to_scores(patch_features_t196d, text_emb_1d):
    """Cosine score per patch.

    patch_features_t196d: (T, 196, 512) normalized
    text_emb_1d: (1, 512) normalized
    returns: (1, T*196) sigmoid-shaped scores in [0, 1]
    """
    T = patch_features_t196d.shape[0]
    flat = patch_features_t196d.reshape(T * N_PATCHES_PER_FRAME, -1)  # (T*196, 512)
    cos = (flat @ text_emb_1d.T).squeeze(-1)  # (T*196,)
    # Map to [0,1] for compatibility with scores_to_gazing_info via shifted sigmoid
    # (the function only ranks; exact values don't matter as long as ordering is preserved)
    scores = torch.sigmoid(cos * 10.0)  # sharpen mildly to avoid all-ties at top-K
    return scores.unsqueeze(0)  # (1, T*196)


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

    print("[setup] CLIP ViT-B/16 (visual + text)...")
    clip_model, _, clip_preproc = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()

    # We need a CLIP-style transform that produces (3, 224, 224) tensors in CLIP's normalization.
    # open_clip's preproc takes a PIL image and returns a tensor. Easier: reuse open_clip's
    # mean/std for normalization and resize manually.
    # Use the SAME CLIP normalization as the encode_image path expects.
    # open_clip 3.3 uses CLIP's standard mean/std.
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)

    print(f"[setup] SemanticAutoGazeWrapper (only used for AutoGaze + scores_to_gazing_info)...")
    # We still need the wrapper for: extract_hidden_states (for ref + auto_at_K paths) and
    # scores_to_gazing_info to convert raw-CLIP scores into NVILA-compatible gazing_info.
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )
    autogaze_transform = AutoGazeImageProcessor.from_pretrained(args.autogaze_model)

    print("[setup] SigLIP-2 base patch16-224...")
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

    query_embs = {q: get_clip_text_embedding(q, clip_model, clip_tok, device) for q in queries}

    KEEP = args.keep_ratio
    results = {c: {q: [] for q in queries} for c in
               ["auto_at_K", "rawclip_match", "rawclip_shuf", "rawclip_rand"]}
    timings_ms = {c: [] for c in ["clip_visual_extract", "clip_score"]}
    tokens_kept = {c: [] for c in ["ref_auto_75", "auto_at_K", "rawclip_match"]}

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
            video_siglip = transform_video_for_pytorch(raw_video, siglip_transform)[None].to(device)

            # CLIP-preprocessed video: resize to 224 + normalize with CLIP mean/std
            # raw_video shape: (T, H, W, 3) uint8
            vid = torch.from_numpy(raw_video).permute(0, 3, 1, 2).float().to(device) / 255.0  # (T,3,H,W)
            vid = F.interpolate(vid, size=(224, 224), mode="bicubic", align_corners=False)
            vid = (vid - CLIP_MEAN[None, :, None, None]) / CLIP_STD[None, :, None, None]
        except Exception as e:
            print(f"  [error] {os.path.basename(vp)}: {e}")
            continue

        # Reference: AutoGaze gaze_only at 0.75
        ref_info = wrapper.forward(
            video_autogaze, query_embs[queries[0]], mode="gaze_only",
            gazing_ratio=0.75, task_loss_requirement=0.7, semantic_keep_ratio=1.0,
        )
        ref_feat = pool_pooled_feature(siglip_model, video_siglip, ref_info)
        tokens_kept["ref_auto_75"].append(int((~ref_info["if_padded_gazing"]).sum().item()))

        # auto_at_K
        auto_info = wrapper.forward(
            video_autogaze, query_embs[queries[0]], mode="gaze_only",
            gazing_ratio=KEEP, task_loss_requirement=0.7, semantic_keep_ratio=1.0,
        )
        auto_feat = pool_pooled_feature(siglip_model, video_siglip, auto_info)
        cs_auto = F.cosine_similarity(ref_feat, auto_feat).item()
        tokens_kept["auto_at_K"].append(int((~auto_info["if_padded_gazing"]).sum().item()))
        for q in queries:
            results["auto_at_K"][q].append(cs_auto)

        # CLIP visual patch features, ONCE per video (all queries reuse)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        patch_feats = encode_clip_patches(clip_model, vid, device)  # (T, 196, 512)
        torch.cuda.synchronize()
        timings_ms["clip_visual_extract"].append((time.perf_counter() - t0) * 1000)

        for q in queries:
            text_emb = query_embs[q]

            # rawclip_match
            torch.cuda.synchronize(); t0 = time.perf_counter()
            scores_m = patch_features_to_scores(patch_feats, text_emb)  # (1, T*196)
            torch.cuda.synchronize()
            timings_ms["clip_score"].append((time.perf_counter() - t0) * 1000)
            info_m = wrapper.semantic_filter.scores_to_gazing_info(
                scores_m, keep_ratio=KEEP, num_frames=patch_feats.shape[0],
            )
            feat_m = pool_pooled_feature(siglip_model, video_siglip, info_m)
            cs_m = F.cosine_similarity(ref_feat, feat_m).item()
            results["rawclip_match"][q].append(cs_m)
            if vi == 0:
                tokens_kept["rawclip_match"].append(int((~info_m["if_padded_gazing"]).sum().item()))

            # rawclip_shuf — use a different query's text emb
            other_queries = [qq for qq in queries if qq != q]
            shuffled_q = rng.choice(other_queries)
            text_emb_s = query_embs[shuffled_q]
            scores_s = patch_features_to_scores(patch_feats, text_emb_s)
            info_s = wrapper.semantic_filter.scores_to_gazing_info(
                scores_s, keep_ratio=KEEP, num_frames=patch_feats.shape[0],
            )
            feat_s = pool_pooled_feature(siglip_model, video_siglip, info_s)
            cs_s = F.cosine_similarity(ref_feat, feat_s).item()
            results["rawclip_shuf"][q].append(cs_s)

            # rawclip_rand
            scores_r = torch.rand_like(scores_m)
            info_r = wrapper.semantic_filter.scores_to_gazing_info(
                scores_r, keep_ratio=KEEP, num_frames=patch_feats.shape[0],
            )
            feat_r = pool_pooled_feature(siglip_model, video_siglip, info_r)
            cs_r = F.cosine_similarity(ref_feat, feat_r).item()
            results["rawclip_rand"][q].append(cs_r)

        if (vi + 1) % 5 == 0 or vi == 0:
            print(f"  [{vi+1}/{len(videos)}] {os.path.basename(vp)}")

    # --- Aggregate ---
    print("\n" + "=" * 78)
    print(f"RAW-CLIP FIDELITY  n_videos={len(videos)}  queries={len(queries)}  keep_ratio={KEEP}")
    print("=" * 78)

    summary = {"keep_ratio": KEEP, "queries": queries, "results": {}}

    print(f"\n{'config':<14}  {'cos_sim mean':>13}  {'cos_sim std':>11}  {'n':>5}")
    print("-" * 50)
    for cfg in ["auto_at_K", "rawclip_match", "rawclip_shuf", "rawclip_rand"]:
        all_sims = [s for q_list in results[cfg].values() for s in q_list]
        mn, sd = float(np.mean(all_sims)), float(np.std(all_sims))
        print(f"{cfg:<14}  {mn:>13.4f}  {sd:>11.4f}  {len(all_sims):>5}")
        summary["results"][cfg] = {"mean_cos_sim": mn, "std_cos_sim": sd, "n": len(all_sims),
                                   "by_query": {q: float(np.mean(v)) for q, v in results[cfg].items()}}

    print(f"\n{'pair':<32}  {'mean delta':>10}  {'std':>8}  {'n':>5}  {'z':>6}")
    print("-" * 70)
    pairs = [
        ("rawclip_match - rawclip_shuf", "rawclip_match", "rawclip_shuf"),
        ("rawclip_match - rawclip_rand", "rawclip_match", "rawclip_rand"),
        ("rawclip_shuf  - rawclip_rand", "rawclip_shuf", "rawclip_rand"),
        ("auto_at_K     - rawclip_match", "auto_at_K", "rawclip_match"),
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

    print(f"\nPer-query (rawclip_match - rawclip_shuf):")
    for q in queries:
        ms = results["rawclip_match"][q]
        ss = results["rawclip_shuf"][q]
        rs = results["rawclip_rand"][q]
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
                   help="(only used for SemanticAutoGazeWrapper plumbing — scoring is by raw CLIP)")
    p.add_argument("--head_type", default="bighead")
    p.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/raw_clip_scoring_baseline")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
