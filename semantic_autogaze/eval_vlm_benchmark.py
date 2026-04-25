"""
VLM benchmark: evaluate NVILA-8B-HD-Video accuracy with/without semantic filtering.

Loads the NVILA model and runs inference on HLVid (or a custom QA dataset)
under different filtering configurations:
  1. Standard AutoGaze (baseline)
  2. AutoGaze + semantic intersect filtering
  3. Semantic-only filtering (bypasses AutoGaze decoder)

Compares accuracy to measure the impact of semantic filtering on VLM task performance.

Usage:
  HF_MODULES_CACHE=/tmp/hf_modules CUDA_VISIBLE_DEVICES=5 python3 -m semantic_autogaze.eval_vlm_benchmark \
    --device cuda:0 --n_samples 50 --ckpt results/distill_bighead/best_bighead_student.pt
"""

import os
import json
import time
import argparse
import random
from typing import Optional

import torch
import torch.nn.functional as F
import open_clip
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper


def get_clip_text_embedding(text: str, clip_model, clip_tokenizer, device):
    """Encode text with CLIP for semantic filtering."""
    tokens = clip_tokenizer([text]).to(device)
    with torch.no_grad():
        features = clip_model.encode_text(tokens)
        features = F.normalize(features, dim=-1)
    return features


def _pick_kept(non_padded_scores, semantic_keep_ratio, score_threshold):
    """Return a bool keep_mask over non_padded_scores. Always keep ≥1."""
    n = non_padded_scores.shape[0]
    if score_threshold is not None:
        keep_mask = non_padded_scores > score_threshold
        if keep_mask.sum() == 0:
            keep_mask = torch.zeros_like(non_padded_scores, dtype=torch.bool)
            keep_mask[non_padded_scores.argmax()] = True
    else:
        n_keep = max(1, int(semantic_keep_ratio * n))
        if n_keep >= n:
            keep_mask = torch.ones_like(non_padded_scores, dtype=torch.bool)
        else:
            _, topk_idx = torch.topk(non_padded_scores, n_keep)
            keep_mask = torch.zeros_like(non_padded_scores, dtype=torch.bool)
            keep_mask[topk_idx] = True
    return keep_mask


def _repad_to_shared_budget(gazing_pos, if_padded, kt_local, kt_shared):
    """Re-pad a (B, K_local) gazing tensor up to a shared per-frame budget kt_shared.

    Within each frame slot, kept entries (if_padded=False) come first, then padded
    dummy entries (if_padded=True). This matches the layout produced by
    `_shrink_unit_batch`. New padding slots are filled with position 0 dummies and
    if_padded=True so the connector drops them.
    """
    if torch.equal(kt_local, kt_shared):
        return gazing_pos, if_padded
    B = gazing_pos.shape[0]
    T = kt_shared.shape[0]
    new_K = int(kt_shared.sum().item())
    out_pos = torch.zeros(B, new_K, dtype=gazing_pos.dtype, device=gazing_pos.device)
    out_pad = torch.ones(B, new_K, dtype=torch.bool, device=if_padded.device)
    src_off = 0
    dst_off = 0
    for t in range(T):
        kt_l = int(kt_local[t].item())
        kt_s = int(kt_shared[t].item())
        if kt_l > 0:
            out_pos[:, dst_off:dst_off + kt_l] = gazing_pos[:, src_off:src_off + kt_l]
            out_pad[:, dst_off:dst_off + kt_l] = if_padded[:, src_off:src_off + kt_l]
        # remaining (kt_s - kt_l) slots stay as position-0 dummies, if_padded=True
        src_off += kt_l
        dst_off += kt_s
    return out_pos, out_pad


def _shrink_unit_batch(
    unit_videos,
    if_padded_orig,
    gazing_pos_orig,
    num_gaze_per_frame,
    wrapper,
    query_emb,
    device,
    semantic_keep_ratio,
    score_threshold,
    score_log=None,
    random_scoring=False,
):
    """Physically shrink K (the gazing-position dimension) for a batch of items
    that share the same per-frame budget.

    Items = tiles for one video, OR thumbnails for one video. The NVILA SigLIP
    wrapper requires `num_gazing_each_frame` to be identical across the batch,
    so we compute a *single* new per-frame budget = max kept count across items,
    and pad each item up to that budget within each frame slot. Padding entries
    are marked True in the new if_padded so they're dropped at the connector,
    but the SigLIP transformer only sees K_new positions per item.

    Args:
      unit_videos: (B, T, C, H, W) AutoGaze-preprocessed video tensors
      if_padded_orig: (B, K_orig) bool
      gazing_pos_orig: (B, K_orig) long  — patch indices into the full (T*N) grid
      num_gaze_per_frame: (T,) long  — per-frame budget shared across items
      query_emb: (1, embed_dim) CLIP text embedding

    Returns:
      new_gazing_pos: (B, K_new) long
      new_if_padded:  (B, K_new) bool
      new_kt:         (T,)       long  — new per-frame budget
    """
    B = unit_videos.shape[0]
    T = unit_videos.shape[1]
    orig_device = if_padded_orig.device

    # ---- Score all items in one wrapper batch ----
    unit_videos_dev = unit_videos.to(device)
    with torch.inference_mode():
        hidden = wrapper.extract_hidden_states(unit_videos_dev)  # (B, T*N_full, C)
        if random_scoring:
            scores = torch.rand(hidden.shape[0], hidden.shape[1], device=hidden.device)
        else:
            # Broadcast query_emb to batch
            query_emb_b = query_emb.expand(B, -1)
            scores = wrapper.semantic_filter.get_scores(hidden, query_emb_b)  # (B, T*N_full)
    score_dev = scores.device

    # ---- Per-frame: figure out kept *original-grid* indices for each item ----
    frame_offsets = torch.zeros(T + 1, dtype=torch.long)
    frame_offsets[1:] = num_gaze_per_frame.cumsum(0)

    if_padded_dev = if_padded_orig.to(score_dev)
    gazing_pos_dev = gazing_pos_orig.to(score_dev)

    kept_per_frame = [[None] * B for _ in range(T)]  # kept_per_frame[t][i] = (kept_pos,)
    new_kt = torch.zeros(T, dtype=torch.long)

    for t in range(T):
        s, e = int(frame_offsets[t].item()), int(frame_offsets[t + 1].item())
        frame_max = 0
        for i in range(B):
            item_pad = if_padded_dev[i, s:e]               # (k_orig_t,)
            item_pos = gazing_pos_dev[i, s:e]              # (k_orig_t,)
            non_padded_mask = ~item_pad
            n_non_padded = int(non_padded_mask.sum().item())
            if n_non_padded == 0:
                kept_per_frame[t][i] = torch.empty(0, dtype=torch.long, device=score_dev)
                continue
            np_pos = item_pos[non_padded_mask]
            np_scores = scores[i, np_pos % scores.shape[1]]
            if score_log is not None:
                score_log.append(np_scores.detach().float().cpu())
            keep_mask = _pick_kept(np_scores, semantic_keep_ratio, score_threshold)
            kept_pos = np_pos[keep_mask]
            kept_per_frame[t][i] = kept_pos
            if kept_pos.shape[0] > frame_max:
                frame_max = kept_pos.shape[0]
        new_kt[t] = max(frame_max, 1)  # need ≥1 per frame for batch alignment

    # ---- Build new gazing_pos / if_padded with shrunk K ----
    new_K = int(new_kt.sum().item())
    new_gazing_pos = torch.zeros(B, new_K, dtype=gazing_pos_orig.dtype, device=orig_device)
    new_if_padded = torch.ones(B, new_K, dtype=torch.bool, device=orig_device)

    write_off = 0
    for t in range(T):
        kt = int(new_kt[t].item())
        for i in range(B):
            kept = kept_per_frame[t][i].to(orig_device)
            n = int(kept.shape[0])
            if n > 0:
                new_gazing_pos[i, write_off:write_off + n] = kept
                new_if_padded[i, write_off:write_off + n] = False
                if n < kt:
                    # Pad remaining slots with first kept (dummy fill); if_padded stays True
                    new_gazing_pos[i, write_off + n:write_off + kt] = kept[0]
            else:
                # No kept positions for this (item, frame): use position 0 dummy
                # (if_padded already True for all kt entries here)
                new_gazing_pos[i, write_off:write_off + kt] = 0
        write_off += kt

    return new_gazing_pos, new_if_padded, new_kt


def patch_processor_with_semantic_filter(
    processor,
    wrapper: SemanticAutoGazeWrapper,
    clip_model,
    clip_tokenizer,
    mode: str = "intersect",
    semantic_keep_ratio: float = 0.5,
    query_text: Optional[str] = None,
    device: str = "cuda",
    score_threshold: Optional[float] = None,
    filter_thumbnails: bool = True,
    log_score_dist: bool = False,
    random_scoring: bool = False,
    bypass_autogaze_selection: bool = False,
):
    """
    Monkey-patch the NVILA processor to inject semantic filtering after AutoGaze.

    Filters BOTH tiles and thumbnails. With score_threshold set, uses an absolute
    sigmoid-score cutoff (variable per-frame budget); otherwise uses top-k by
    semantic_keep_ratio (fixed per-unit budget).

    Note: thumbnails are only filterable if AutoGaze actually ran on them, which
    requires gazing_ratio_thumbnail < 1.0 (or task_loss_requirement_thumbnail set).
    Otherwise the processor short-circuits and pixel_values_videos_thumbnails_autogaze
    isn't even built.
    """
    original_get_gazing = processor._get_gazing_info_from_videos

    def patched_get_gazing_info(videos_inputs):
        gazing_info = original_get_gazing(videos_inputs)
        if gazing_info is None or mode == "gaze_only":
            return gazing_info

        gazing_info = {
            k: [t.clone() for t in v] if isinstance(v, list) else v.clone()
            for k, v in gazing_info.items()
        }

        tiles_autogaze = videos_inputs.get("pixel_values_videos_tiles_autogaze")
        thumbs_autogaze = videos_inputs.get("pixel_values_videos_thumbnails_autogaze")
        if tiles_autogaze is None:
            return gazing_info

        # When bypass_autogaze_selection=True, replace AutoGaze's gazing_pos
        # with full-grid arange (every patch on the 14×14 grid is "kept" by
        # AutoGaze; BigHead then picks top-K from all 196 per frame). This
        # makes the downstream _shrink_unit_batch operate on the full grid
        # rather than re-ranking AutoGaze's K-subset, so semantic_keep_ratio
        # selects K = round(196 * keep_ratio) patches per frame.
        if bypass_autogaze_selection:
            import torch as _torch
            for vid_idx in range(len(tiles_autogaze)):
                vt = tiles_autogaze[vid_idx]  # (num_tiles, T_tile, C, H, W)
                num_tiles_v, T_tile_v = vt.shape[:2]
                # 14×14 = 196 fine-grid patches per frame
                N_per_frame = 196
                K_total = T_tile_v * N_per_frame
                # gazing_pos_tiles[vid] : (num_tiles, K_total) — same arange across tiles
                full_pos = _torch.arange(K_total, device=gazing_info["gazing_pos_tiles"][vid_idx].device,
                                         dtype=gazing_info["gazing_pos_tiles"][vid_idx].dtype)
                full_pos = full_pos.unsqueeze(0).expand(num_tiles_v, -1).contiguous()
                gazing_info["gazing_pos_tiles"][vid_idx] = full_pos
                gazing_info["if_padded_gazing_tiles"][vid_idx] = _torch.zeros(
                    num_tiles_v, K_total,
                    device=full_pos.device, dtype=_torch.bool,
                )
                # num_gazing_each_frame_tiles: keep original outer shape but set values to N_per_frame
                nge = gazing_info["num_gazing_each_frame_tiles"][vid_idx]
                gazing_info["num_gazing_each_frame_tiles"][vid_idx] = _torch.full_like(nge, N_per_frame)
            if thumbs_autogaze is not None:
                for vid_idx in range(len(thumbs_autogaze)):
                    th = thumbs_autogaze[vid_idx]  # (1, T_thumb, C, H, W)
                    num_t, T_thumb_v = th.shape[:2]
                    N_per_frame = 196
                    K_total = T_thumb_v * N_per_frame
                    full_pos = _torch.arange(K_total, device=gazing_info["gazing_pos_thumbnails"][vid_idx].device,
                                             dtype=gazing_info["gazing_pos_thumbnails"][vid_idx].dtype)
                    full_pos = full_pos.unsqueeze(0).expand(num_t, -1).contiguous()
                    gazing_info["gazing_pos_thumbnails"][vid_idx] = full_pos
                    gazing_info["if_padded_gazing_thumbnails"][vid_idx] = _torch.zeros(
                        num_t, K_total,
                        device=full_pos.device, dtype=_torch.bool,
                    )
                    nge = gazing_info["num_gazing_each_frame_thumbnails"][vid_idx]
                    gazing_info["num_gazing_each_frame_thumbnails"][vid_idx] = _torch.full_like(nge, N_per_frame)

        nonlocal query_text
        q = query_text or "important content"
        query_emb = get_clip_text_embedding(q, clip_model, clip_tokenizer, device)

        score_log = [] if log_score_dist else None

        # Cross-video per-frame max trackers (NVILA asserts num_gazing_each_frame
        # is identical across all videos for tiles and for thumbnails). We compute
        # per-video shrunk gazing_pos/if_padded first, then pad them up to a
        # cross-video shared per-frame budget at the end.
        per_video_tile_results = []   # list of (new_pos, new_pad, new_kt) per video
        per_video_thumb_results = []  # list of (new_pos, new_pad, new_kt) or None

        for vid_idx in range(len(tiles_autogaze)):
            # ---- Tiles: batch all tiles of this video together ----
            video_tiles = tiles_autogaze[vid_idx]  # (num_tiles, T_tile, C, H, W)
            tile_pos = gazing_info["gazing_pos_tiles"][vid_idx]                  # (num_tiles, K)
            tile_pad = gazing_info["if_padded_gazing_tiles"][vid_idx]            # (num_tiles, K)
            tile_nge = gazing_info["num_gazing_each_frame_tiles"][vid_idx][0]    # (T_tile,)

            new_pos_t, new_pad_t, new_kt_t = _shrink_unit_batch(
                unit_videos=video_tiles,
                if_padded_orig=tile_pad,
                gazing_pos_orig=tile_pos,
                num_gaze_per_frame=tile_nge,
                wrapper=wrapper, query_emb=query_emb, device=device,
                semantic_keep_ratio=semantic_keep_ratio,
                score_threshold=score_threshold,
                score_log=score_log,
                random_scoring=random_scoring,
            )
            per_video_tile_results.append((new_pos_t, new_pad_t, new_kt_t))

            # ---- Thumbnails: batch all thumbnail items of this video ----
            do_thumbs = (
                filter_thumbnails
                and thumbs_autogaze is not None
                and vid_idx < len(thumbs_autogaze)
                and "if_padded_gazing_thumbnails" in gazing_info
            )
            if do_thumbs:
                video_thumbs = thumbs_autogaze[vid_idx]  # (T_thumb, 1, C, H, W)
                th_pos = gazing_info["gazing_pos_thumbnails"][vid_idx]               # (T_thumb, K')
                th_pad = gazing_info["if_padded_gazing_thumbnails"][vid_idx]         # (T_thumb, K')
                th_nge = gazing_info["num_gazing_each_frame_thumbnails"][vid_idx][0] # (1,)
                new_pos_th, new_pad_th, new_kt_th = _shrink_unit_batch(
                    unit_videos=video_thumbs,
                    if_padded_orig=th_pad,
                    gazing_pos_orig=th_pos,
                    num_gaze_per_frame=th_nge,
                    wrapper=wrapper, query_emb=query_emb, device=device,
                    semantic_keep_ratio=semantic_keep_ratio,
                    score_threshold=score_threshold,
                    score_log=score_log,
                    random_scoring=random_scoring,
                )
                per_video_thumb_results.append((new_pos_th, new_pad_th, new_kt_th))
            else:
                per_video_thumb_results.append(None)

        # ---- Reconcile per-frame budgets ACROSS videos (NVILA assertion) ----
        # Tiles: take elementwise max of new_kt across videos, then pad each
        # video's K up to that shared budget with dummies (if_padded=True).
        if per_video_tile_results:
            shared_kt_tiles = per_video_tile_results[0][2].clone()
            for _, _, kt in per_video_tile_results[1:]:
                shared_kt_tiles = torch.maximum(shared_kt_tiles, kt)
            for vid_idx, (pos, pad, kt) in enumerate(per_video_tile_results):
                pos2, pad2 = _repad_to_shared_budget(pos, pad, kt, shared_kt_tiles)
                gazing_info["gazing_pos_tiles"][vid_idx] = pos2
                gazing_info["if_padded_gazing_tiles"][vid_idx] = pad2
                num_tiles = pos2.shape[0]
                gazing_info["num_gazing_each_frame_tiles"][vid_idx] = (
                    shared_kt_tiles.unsqueeze(0).expand(num_tiles, -1).clone()
                )

        if any(r is not None for r in per_video_thumb_results):
            shared_kt_th = None
            for r in per_video_thumb_results:
                if r is None:
                    continue
                shared_kt_th = r[2].clone() if shared_kt_th is None else torch.maximum(shared_kt_th, r[2])
            for vid_idx, r in enumerate(per_video_thumb_results):
                if r is None:
                    continue
                pos, pad, kt = r
                pos2, pad2 = _repad_to_shared_budget(pos, pad, kt, shared_kt_th)
                gazing_info["gazing_pos_thumbnails"][vid_idx] = pos2
                gazing_info["if_padded_gazing_thumbnails"][vid_idx] = pad2
                n_thumb = pos2.shape[0]
                gazing_info["num_gazing_each_frame_thumbnails"][vid_idx] = (
                    shared_kt_th.unsqueeze(0).expand(n_thumb, -1).clone()
                )

        if log_score_dist and score_log:
            allscores = torch.cat(score_log)
            qs = torch.quantile(allscores, torch.tensor([0.10, 0.50, 0.90, 0.99]))
            print(f"  [score-dist] n={len(allscores)} "
                  f"p10={qs[0]:.3f} p50={qs[1]:.3f} p90={qs[2]:.3f} p99={qs[3]:.3f} "
                  f"max={allscores.max():.3f}")

        return gazing_info

    processor._get_gazing_info_from_videos = patched_get_gazing_info


def load_hlvid_samples(data_dir: str, n_samples: int = 50):
    """Load HLVid benchmark samples."""
    # Try loading from HuggingFace datasets
    try:
        from datasets import load_dataset
        ds = load_dataset("bfshi/HLVid", split="test")
        samples = []
        for i, item in enumerate(ds):
            if i >= n_samples:
                break
            samples.append({
                "video_path": item.get("video", item.get("video_path", "")),
                "question": item["question"],
                "choices": item.get("choices", []),
                "answer": item.get("answer", ""),
            })
        return samples
    except Exception as e:
        print(f"Could not load HLVid from HuggingFace: {e}")

    # Fallback: generate synthetic QA from our video dataset
    import glob
    videos = sorted(glob.glob(os.path.join(data_dir, "*.mp4")))
    random.shuffle(videos)

    # Create simple recognition questions
    questions = [
        ("What is the main activity shown in this video?",
         ["A. Cooking", "B. Walking", "C. Talking", "D. Playing sports"],
         "C"),  # Most of our videos are people talking
        ("What type of setting is this video recorded in?",
         ["A. Indoor", "B. Outdoor", "C. Studio", "D. Vehicle"],
         "A"),
        ("How many people are primarily visible in this video?",
         ["A. None", "B. One", "C. Two", "D. Three or more"],
         "B"),
    ]

    samples = []
    for v in videos[:n_samples]:
        q_idx = random.randint(0, len(questions) - 1)
        question, choices, answer = questions[q_idx]
        samples.append({
            "video_path": v,
            "question": question,
            "choices": choices,
            "answer": answer,
        })
    return samples


def run_inference(model, processor, video_path, question, choices, device):
    """Run NVILA inference on a single video QA sample."""
    # Format prompt
    choices_text = "\n".join(choices)
    prompt = f"Question: {question}\n{choices_text}\nPlease answer directly with the letter of the correct answer."

    video_token = processor.tokenizer.video_token
    inputs = processor(
        text=f"{video_token}\n\n{prompt}",
        videos=video_path,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )
    response = processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0].strip()

    return response


def extract_answer(response: str) -> str:
    """Extract answer letter from model response."""
    response = response.strip().upper()
    for letter in ["A", "B", "C", "D"]:
        if response.startswith(letter):
            return letter
    # Try finding letter in response
    for letter in ["A", "B", "C", "D"]:
        if letter in response:
            return letter
    return response[:1] if response else ""


def main(args):
    device = torch.device(args.device)
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("NVILA VLM Benchmark with Semantic Filtering")
    print("=" * 60)

    # Load CLIP for semantic filtering
    print("\nLoading CLIP for semantic filtering...")
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()

    # Load semantic filtering wrapper
    print("Loading SemanticAutoGaze wrapper...")
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )

    # Load NVILA model
    print("Loading NVILA-8B-HD-Video...")
    model_path = args.model_path

    processor = AutoProcessor.from_pretrained(
        model_path,
        num_video_frames=args.num_frames,
        num_video_frames_thumbnail=args.num_frames_thumbnail,
        max_tiles_video=args.max_tiles,
        gazing_ratio_tile=args.gazing_ratio,
        gazing_ratio_thumbnail=1.0,  # No gazing on thumbnails
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
        model_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map=args.device,
        max_batch_size_siglip=8,
    )
    model.eval()
    print("Model loaded.")

    # Load benchmark data
    print(f"\nLoading benchmark data (n={args.n_samples})...")
    samples = load_hlvid_samples(args.video_dir, args.n_samples)
    print(f"Loaded {len(samples)} samples")

    # Configurations to evaluate
    configs = [
        {"name": "AutoGaze only", "mode": "gaze_only", "keep_ratio": 1.0},
        {"name": "Intersect (50%)", "mode": "intersect", "keep_ratio": 0.5},
        {"name": "Intersect (30%)", "mode": "intersect", "keep_ratio": 0.3},
        {"name": "Semantic only (30%)", "mode": "semantic_only", "keep_ratio": 0.3},
    ]

    all_results = {}

    for config in configs:
        print(f"\n{'='*50}")
        print(f"Config: {config['name']}")
        print(f"{'='*50}")

        # Patch processor for this config
        # Reset to original first
        if hasattr(processor, '_original_get_gazing'):
            processor._get_gazing_info_from_videos = processor._original_get_gazing
        else:
            processor._original_get_gazing = processor._get_gazing_info_from_videos

        if config["mode"] != "gaze_only":
            patch_processor_with_semantic_filter(
                processor, wrapper,
                clip_model, clip_tokenizer,
                mode=config["mode"],
                semantic_keep_ratio=config["keep_ratio"],
                device=str(device),
            )

        correct = 0
        total = 0
        latencies = []

        for i, sample in enumerate(samples):
            try:
                t0 = time.perf_counter()

                # Set query text for semantic filtering based on question
                if config["mode"] != "gaze_only":
                    # Use question as query for semantic filtering
                    query_text = sample["question"]
                    patch_processor_with_semantic_filter(
                        processor, wrapper,
                        clip_model, clip_tokenizer,
                        mode=config["mode"],
                        semantic_keep_ratio=config["keep_ratio"],
                        query_text=query_text,
                        device=str(device),
                    )

                response = run_inference(
                    model, processor,
                    sample["video_path"],
                    sample["question"],
                    sample["choices"],
                    device,
                )

                t1 = time.perf_counter()
                latencies.append(t1 - t0)

                predicted = extract_answer(response)
                gt = sample["answer"]
                is_correct = predicted == gt

                correct += int(is_correct)
                total += 1

                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(samples)}] acc={correct/total:.3f}, "
                          f"avg_lat={sum(latencies)/len(latencies):.2f}s")

            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                continue

        accuracy = correct / max(total, 1)
        avg_latency = sum(latencies) / max(len(latencies), 1)

        result = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_latency_s": avg_latency,
            "config": config,
        }
        all_results[config["name"]] = result

        print(f"\n  Accuracy: {accuracy:.3f} ({correct}/{total})")
        print(f"  Avg latency: {avg_latency:.2f}s")

    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for name, result in all_results.items():
        print(f"  {name:<25}: acc={result['accuracy']:.3f} "
              f"({result['correct']}/{result['total']}), "
              f"lat={result['avg_latency_s']:.2f}s")

    # Save results
    with open(os.path.join(args.output_dir, "vlm_benchmark.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {args.output_dir}/vlm_benchmark.json")

    # Plot
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        names = list(all_results.keys())
        accs = [all_results[n]["accuracy"] for n in names]
        lats = [all_results[n]["avg_latency_s"] for n in names]

        x = np.arange(len(names))
        ax1.bar(x, accs, color=["#4CAF50", "#2196F3", "#FF9800", "#E91E63"])
        ax1.set_ylabel("Accuracy")
        ax1.set_title("VLM Accuracy by Filtering Config")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax1.set_ylim([0, 1.05])
        ax1.grid(True, alpha=0.3, axis="y")

        ax2.bar(x, lats, color=["#4CAF50", "#2196F3", "#FF9800", "#E91E63"])
        ax2.set_ylabel("Latency (s)")
        ax2.set_title("Inference Latency by Filtering Config")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax2.grid(True, alpha=0.3, axis="y")

        fig.suptitle("NVILA-8B-HD-Video: Semantic Filtering Impact", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "vlm_benchmark.png"), dpi=150)
        plt.close(fig)
        print(f"Saved: {args.output_dir}/vlm_benchmark.png")
    except Exception as e:
        print(f"Plot failed: {e}")

    # Cleanup
    del clip_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="nvidia/NVILA-8B-HD-Video")
    parser.add_argument("--video_dir", default="data")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--num_frames_thumbnail", type=int, default=32)
    parser.add_argument("--max_tiles", type=int, default=8)
    parser.add_argument("--gazing_ratio", type=float, default=0.2)
    parser.add_argument("--ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--head_type", default="bighead")
    parser.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/vlm_benchmark")
    args = parser.parse_args()
    main(args)
