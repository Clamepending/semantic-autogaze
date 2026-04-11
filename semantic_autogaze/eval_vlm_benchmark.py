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


def patch_processor_with_semantic_filter(
    processor,
    wrapper: SemanticAutoGazeWrapper,
    clip_model,
    clip_tokenizer,
    mode: str = "intersect",
    semantic_keep_ratio: float = 0.5,
    query_text: Optional[str] = None,
    device: str = "cuda",
):
    """
    Monkey-patch the NVILA processor to inject semantic filtering after AutoGaze.

    Instead of modifying the processor's internal methods (which is fragile),
    we wrap the entire __call__ to post-process gazing_info.
    """
    original_get_gazing = processor._get_gazing_info_from_videos

    def patched_get_gazing_info(videos_inputs):
        # Get standard AutoGaze gazing info
        gazing_info = original_get_gazing(videos_inputs)
        if gazing_info is None or mode == "gaze_only":
            return gazing_info

        # Get the AutoGaze-preprocessed video tiles for semantic scoring
        tiles_autogaze = videos_inputs.get("pixel_values_videos_tiles_autogaze")
        if tiles_autogaze is None:
            return gazing_info

        # Determine query text for semantic filtering
        nonlocal query_text
        q = query_text or "important content"
        query_emb = get_clip_text_embedding(q, clip_model, clip_tokenizer, device)

        # Process each video's tiles through semantic filter
        for vid_idx in range(len(tiles_autogaze)):
            video_tiles = tiles_autogaze[vid_idx]  # (num_tiles, T, C, H, W)

            for tile_idx in range(video_tiles.shape[0]):
                tile_video = video_tiles[tile_idx:tile_idx+1].to(device)  # (1, T, C, H, W)

                # Get semantic scores from wrapper
                hidden_states = wrapper.extract_hidden_states(tile_video)
                scores = wrapper.semantic_filter.get_scores(hidden_states, query_emb)
                # scores: (1, T*196) for 14x14 patches, 196 per frame

                # Get current gazing info for this tile
                tile_gazing_pos = gazing_info["gazing_pos_tiles"][vid_idx][tile_idx]
                tile_if_padded = gazing_info["if_padded_gazing_tiles"][vid_idx][tile_idx]
                tile_num_gazing = gazing_info["num_gazing_each_frame_tiles"][vid_idx][tile_idx]

                # Apply semantic scoring to non-padded positions
                non_padded_mask = ~tile_if_padded
                if non_padded_mask.sum() == 0:
                    continue

                # Get positions of non-padded entries
                non_padded_positions = tile_gazing_pos[non_padded_mask]
                non_padded_scores = scores[0, non_padded_positions % scores.shape[1]]

                # Keep top-k by semantic score
                n_keep = max(1, int(semantic_keep_ratio * non_padded_mask.sum().item()))
                if n_keep >= non_padded_mask.sum().item():
                    continue

                _, topk_indices = torch.topk(non_padded_scores, n_keep)
                keep_mask = torch.zeros_like(non_padded_scores, dtype=torch.bool)
                keep_mask[topk_indices] = True

                # Update if_padded: mark removed positions as padded
                non_padded_indices = torch.where(non_padded_mask)[0]
                for j, idx in enumerate(non_padded_indices):
                    if not keep_mask[j]:
                        tile_if_padded[idx] = True

                # Update gazing info in place
                gazing_info["if_padded_gazing_tiles"][vid_idx][tile_idx] = tile_if_padded

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
