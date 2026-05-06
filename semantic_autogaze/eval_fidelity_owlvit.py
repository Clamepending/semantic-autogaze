"""r/phrase-grounding-teacher cycle 1: drop CLIP/SigLIP-2, score patches via OWL-ViT.

Mirrors r/siglip2-patch-scoring (f8e9379) but replaces SigLIP-2 visual+text with
google/owlvit-base-patch32 — an open-vocabulary detection model trained EXPLICITLY
for text-conditioned per-patch localization (text query → which patches contain
that object). Unlike CLIP/SigLIP-2 which train alignment on POOLED features only,
OWL-ViT's class predictor scores patches against text per-patch as part of the
detection objective. This is the cleanest test of "patch-text alignment becomes
non-trivial when the training objective requires it" — separating the "no patch
alignment in CLIP-family" finding from the "patch alignment is intrinsically
unachievable" alternative.

Decisive split:
  match > shuffled by z>2  → phrase-grounding teacher works as a drop-in.
                              cycle 2: evaluate VQA accuracy at iso-K on HLVid.
  match ≈ shuffled         → even phrase-grounding teachers don't transfer to
                              this task. Rules out the principled fix entirely;
                              routes to NVILA-attention distillation.

OWL-ViT B/32 produces 24×24=576 patches per 768-input frame. We resample the
score grid bilinearly to 14×14=196 to align with the SigLIP grid that
scores_to_gazing_info expects.

Usage:
  CUDA_VISIBLE_DEVICES=4 python -m semantic_autogaze.eval_fidelity_owlvit \\
    --device cuda:0 --n_videos 50 --output_dir results/owlvit_patch_scoring
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

GRID_TARGET = 14  # SigLIP grid for scores_to_gazing_info compatibility
N_PATCHES_TARGET = GRID_TARGET * GRID_TARGET  # 196
OWLVIT_NAME = "google/owlvit-base-patch32"


@torch.no_grad()
def encode_owlvit_patches(owlvit_det, pixel_values_b3hw, target_grid=GRID_TARGET):
    """Run OWL-ViT image_embedder + class_head.dense0, return per-patch
    embeddings projected to 512-d (matching the text-embedding space) AND
    bilinear-resampled to the target SigLIP grid.

    Returns: (T, target_grid*target_grid, 512), L2-normalized.
    """
    # Use the detection model's image_embedder which applies post_layernorm
    # + class-token multiplication + OD-head layer_norm. This is the canonical
    # patch representation used by OWL-ViT for text matching.
    image_embeds, _ = owlvit_det.image_embedder(pixel_values=pixel_values_b3hw)
    # image_embeds shape: (T, h_patches, w_patches, 768)
    T, hp, wp, D = image_embeds.shape

    # Project to 512-d via class_head.dense0 (the trained projection that
    # OWL-ViT uses to compare patches with text).
    image_class_embeds = owlvit_det.class_head.dense0(image_embeds)  # (T, hp, wp, 512)

    # Bilinear-resample to the SigLIP grid (14, 14)
    # Move D to channel dim for interpolate
    x = image_class_embeds.permute(0, 3, 1, 2)  # (T, 512, hp, wp)
    x = F.interpolate(x, size=(target_grid, target_grid), mode="bilinear", align_corners=False)
    x = x.permute(0, 2, 3, 1).contiguous()  # (T, target_grid, target_grid, 512)
    x = x.view(T, target_grid * target_grid, -1)
    x = F.normalize(x, dim=-1)
    return x  # (T, 196, 512)


@torch.no_grad()
def encode_owlvit_text(owlvit_model, tokenizer, text, device):
    """Return (1, 512) L2-normalized text embedding. Matches OWL-ViT's
    text-encoding pipeline used during detection: text_model.pooler_output
    → text_projection → normalize."""
    enc = tokenizer([text], padding="max_length", return_tensors="pt").to(device)
    out = owlvit_model.text_model(input_ids=enc["input_ids"])
    pooled = out.pooler_output  # (1, 512)
    pooled = owlvit_model.text_projection(pooled)  # (1, 512) — applies the trained 512→512 proj
    pooled = F.normalize(pooled, dim=-1)
    return pooled


def patch_features_to_scores(patch_features_t196d, text_emb_1d):
    """Cosine score per patch -> sigmoid-shaped scores in [0,1]. Ranking-only."""
    T = patch_features_t196d.shape[0]
    flat = patch_features_t196d.reshape(T * N_PATCHES_TARGET, -1)  # (T*196, 512)
    cos = (flat @ text_emb_1d.T).squeeze(-1)
    scores = torch.sigmoid(cos * 10.0)
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

    print(f"[setup] OWL-ViT detection model: {OWLVIT_NAME}")
    from transformers import OwlViTForObjectDetection, AutoTokenizer, AutoImageProcessor
    owlvit_det = OwlViTForObjectDetection.from_pretrained(OWLVIT_NAME).to(device).eval()
    owlvit_tok = AutoTokenizer.from_pretrained(OWLVIT_NAME)
    owlvit_imgproc = AutoImageProcessor.from_pretrained(OWLVIT_NAME)
    owlvit_size = owlvit_imgproc.size["height"] if isinstance(owlvit_imgproc.size, dict) else 768
    owlvit_mean = torch.tensor(owlvit_imgproc.image_mean, device=device)
    owlvit_std = torch.tensor(owlvit_imgproc.image_std, device=device)
    print(f"[setup] OWL-ViT size={owlvit_size} mean={owlvit_mean.tolist()} std={owlvit_std.tolist()}")
    h_patches = owlvit_size // owlvit_det.config.vision_config.patch_size
    print(f"[setup] OWL-ViT patch grid: {h_patches}x{h_patches}={h_patches*h_patches} -> resample to 14x14=196")

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
    nvila_siglip_imgproc = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
    nvila_siglip = SiglipVisionModel.from_pretrained(
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

    query_embs = {q: encode_owlvit_text(owlvit_det.owlvit, owlvit_tok, q, device) for q in queries}

    KEEP = args.keep_ratio
    results = {c: {q: [] for q in queries} for c in
               ["auto_at_K", "owlvit_match", "owlvit_shuf", "owlvit_rand"]}
    timings_ms = {c: [] for c in ["owlvit_visual_extract", "owlvit_score"]}
    tokens_kept = {c: [] for c in ["ref_auto_75", "auto_at_K", "owlvit_match"]}

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

            # OWL-ViT preprocessing: bicubic-resize to 768 + normalize with OWL-ViT mean/std
            vid = torch.from_numpy(raw_video).permute(0, 3, 1, 2).float().to(device) / 255.0
            vid = F.interpolate(vid, size=(owlvit_size, owlvit_size), mode="bicubic", align_corners=False)
            vid = (vid - owlvit_mean[None, :, None, None]) / owlvit_std[None, :, None, None]
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
        patch_feats = encode_owlvit_patches(owlvit_det, vid)  # (T, 196, 512)
        torch.cuda.synchronize()
        timings_ms["owlvit_visual_extract"].append((time.perf_counter() - t0) * 1000)

        for q in queries:
            text_emb = query_embs[q]

            torch.cuda.synchronize(); t0 = time.perf_counter()
            scores_m = patch_features_to_scores(patch_feats, text_emb)
            torch.cuda.synchronize()
            timings_ms["owlvit_score"].append((time.perf_counter() - t0) * 1000)
            info_m = wrapper.semantic_filter.scores_to_gazing_info(
                scores_m, keep_ratio=KEEP, num_frames=patch_feats.shape[0],
            )
            feat_m = pool_pooled_feature(nvila_siglip, video_siglip, info_m)
            cs_m = F.cosine_similarity(ref_feat, feat_m).item()
            results["owlvit_match"][q].append(cs_m)
            if vi == 0:
                tokens_kept["owlvit_match"].append(int((~info_m["if_padded_gazing"]).sum().item()))

            other_queries = [qq for qq in queries if qq != q]
            shuffled_q = rng.choice(other_queries)
            text_emb_s = query_embs[shuffled_q]
            scores_s = patch_features_to_scores(patch_feats, text_emb_s)
            info_s = wrapper.semantic_filter.scores_to_gazing_info(
                scores_s, keep_ratio=KEEP, num_frames=patch_feats.shape[0],
            )
            feat_s = pool_pooled_feature(nvila_siglip, video_siglip, info_s)
            cs_s = F.cosine_similarity(ref_feat, feat_s).item()
            results["owlvit_shuf"][q].append(cs_s)

            scores_r = torch.rand_like(scores_m)
            info_r = wrapper.semantic_filter.scores_to_gazing_info(
                scores_r, keep_ratio=KEEP, num_frames=patch_feats.shape[0],
            )
            feat_r = pool_pooled_feature(nvila_siglip, video_siglip, info_r)
            cs_r = F.cosine_similarity(ref_feat, feat_r).item()
            results["owlvit_rand"][q].append(cs_r)

        if (vi + 1) % 5 == 0 or vi == 0:
            print(f"  [{vi+1}/{len(videos)}] {os.path.basename(vp)}", flush=True)

    print("\n" + "=" * 78, flush=True)
    print(f"OWL-ViT FIDELITY  n_videos={len(videos)}  queries={len(queries)}  keep_ratio={KEEP}")
    print("=" * 78)

    summary = {"keep_ratio": KEEP, "queries": queries, "results": {}, "model": OWLVIT_NAME}

    print(f"\n{'config':<14}  {'cos_sim mean':>13}  {'cos_sim std':>11}  {'n':>5}")
    print("-" * 50)
    for cfg in ["auto_at_K", "owlvit_match", "owlvit_shuf", "owlvit_rand"]:
        all_sims = [s for q_list in results[cfg].values() for s in q_list]
        mn, sd = float(np.mean(all_sims)), float(np.std(all_sims))
        print(f"{cfg:<14}  {mn:>13.4f}  {sd:>11.4f}  {len(all_sims):>5}")
        summary["results"][cfg] = {"mean_cos_sim": mn, "std_cos_sim": sd, "n": len(all_sims),
                                   "by_query": {q: float(np.mean(v)) for q, v in results[cfg].items()}}

    print(f"\n{'pair':<32}  {'mean delta':>10}  {'std':>8}  {'n':>5}  {'z':>6}")
    print("-" * 70)
    pairs = [
        ("owlvit_match - owlvit_shuf", "owlvit_match", "owlvit_shuf"),
        ("owlvit_match - owlvit_rand", "owlvit_match", "owlvit_rand"),
        ("owlvit_shuf  - owlvit_rand", "owlvit_shuf", "owlvit_rand"),
        ("auto_at_K    - owlvit_match", "auto_at_K", "owlvit_match"),
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

    print(f"\nPer-query (owlvit_match - owlvit_shuf):")
    for q in queries:
        ms = results["owlvit_match"][q]
        ss = results["owlvit_shuf"][q]
        rs = results["owlvit_rand"][q]
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
                   help="(only used for SemanticAutoGazeWrapper plumbing — scoring is by OWL-ViT)")
    p.add_argument("--head_type", default="bighead")
    p.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/owlvit_patch_scoring")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
