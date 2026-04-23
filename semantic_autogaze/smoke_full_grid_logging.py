"""
Smoke test for r/fix-gazed-subset-logging cycle 2.

Runs 2 samples on HLVid household at Intersect 50 % with log_score_dist=True,
to verify:
  - no runtime errors,
  - [score-dist gazed-subset] matches the previous [score-dist] numbers
    (p99 well under 0.05 on AutoGaze's reconstruction-salient subset),
  - [score-dist full-grid] shows a meaningfully higher p99 on the same
    sample (p99 >~ 0.20-0.35 per r/debug-score-distribution's heatmap finding).

Usage:
    CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      python -u -m semantic_autogaze.smoke_full_grid_logging --device cuda:0
"""

import argparse, random
import torch
import open_clip
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper
from semantic_autogaze.eval_vlm_benchmark import (
    patch_processor_with_semantic_filter,
    extract_answer,
)
from semantic_autogaze.eval_hlvid_subset import load_subset, run_inference


def main(args):
    device = torch.device(args.device)
    random.seed(42)

    print("[setup] Loading NVILA-8B-HD-Video (4-bit NF4)...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        num_video_frames=32,
        num_video_frames_thumbnail=16,
        max_tiles_video=4,
        gazing_ratio_tile=0.2,
        gazing_ratio_thumbnail=0.75,
        task_loss_requirement_tile=0.6,
        task_loss_requirement_thumbnail=0.6,
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
    ).eval()
    torch.cuda.empty_cache()
    print("[setup] Loading CLIP...")
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()
    print("[setup] Loading SemanticAutoGazeWrapper...")
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name="nvidia/AutoGaze",
        head_ckpt=args.ckpt,
        head_type="bighead",
        device=str(device),
    )

    samples = load_subset(args.video_dir, query_mode="stem")
    print(f"[data] HLVid subset: {len(samples)} QA pairs")

    # Just run 2 samples with score-dist logging
    for i, s in enumerate(samples[:2]):
        print(f"\n{'=' * 60}\n[smoke] sample {i+1}/2  q={s['question_id']}\n{'=' * 60}")
        patch_processor_with_semantic_filter(
            processor, wrapper, clip_model, clip_tokenizer,
            mode="intersect", semantic_keep_ratio=0.5,
            query_text=s["semantic_query"], device=str(device),
            score_threshold=None,
            filter_thumbnails=True,
            log_score_dist=True,
        )
        response = run_inference(model, processor, s["video_path"], s["question_raw"], device)
        predicted = extract_answer(response)
        gt = s["answer"]
        print(f"  => gt={gt}  pred={predicted}  {'OK' if predicted == gt else 'X'}")

    print("\n[smoke] done. Check that BOTH [score-dist gazed-subset] and "
          "[score-dist full-grid] lines appear, and the full-grid p99 is "
          "meaningfully higher than the subset p99.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="nvidia/NVILA-8B-HD-Video")
    p.add_argument("--video_dir", default="hlvid_videos/extracted_household/videos")
    p.add_argument("--ckpt", default="results/bighead_warmrestart/best_bighead_student.pt")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    main(args)
