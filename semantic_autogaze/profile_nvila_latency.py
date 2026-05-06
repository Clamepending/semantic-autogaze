"""
Profile NVILA-8B-HD-Video's latency breakdown on one HLVid sample under two
filter regimes: AutoGaze-only (baseline tokens) and Semantic-only-30% (tight
tokens). Splits the wall-clock into:

  1. preprocess + AutoGaze/filter gazing-pos computation
  2. model.generate (the whole forward+decode)

…plus we report the resulting kept-token count so we can verify the filter
actually shrank the LLM's visual input. If cost (2) is constant across
regimes despite a 4–5× drop in visual tokens, the bottleneck is SigLIP
encoding or LLM generation, not prefill on visual tokens.
"""

import argparse
import os
import time

import open_clip
import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from semantic_autogaze.eval_hlvid_subset import (
    PARQUET_PATH,
    parse_question_and_choices,
)
from semantic_autogaze.eval_vlm_benchmark import patch_processor_with_semantic_filter
from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper


def run_one(model, processor, video_path, question_raw, device, warmup=False):
    video_token = processor.tokenizer.video_token

    t0 = time.perf_counter()
    inputs = processor(
        text=f"{video_token}\n\n{question_raw}",
        videos=video_path,
        return_tensors="pt",
    )
    t1 = time.perf_counter()

    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    # Count kept visual tokens: everything between special video tokens
    n_input = int(inputs["input_ids"].shape[1])
    n_gen = int(outputs.shape[1] - n_input)

    response = processor.batch_decode(
        outputs[:, n_input:], skip_special_tokens=True,
    )[0].strip()
    if warmup:
        return None
    return {
        "preprocess_s": t1 - t0,
        "togpu_s": t2 - t1,
        "generate_s": t3 - t2,
        "total_s": t3 - t0,
        "n_input_tokens": n_input,
        "n_gen_tokens": n_gen,
        "response": response,
    }


def main(args):
    device = torch.device(args.device)

    print("[setup] NVILA 4-bit …")
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
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map=args.device,
        max_batch_size_siglip=8,
    ).eval()
    torch.cuda.empty_cache()

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()

    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )

    # First HLVid QA sample on the shard
    df = pd.read_parquet(PARQUET_PATH)
    for _, r in df.iterrows():
        full = os.path.join(args.video_dir, r["video_path"])
        if os.path.exists(full):
            sample = {"video": full, "q": r["question"],
                      "stem": parse_question_and_choices(r["question"])[0]}
            break
    print(f"[data] using sample: {os.path.basename(sample['video'])}")
    print(f"       stem: {sample['stem'][:80]}")

    # Warmup once
    print("[warmup] running one pass…")
    run_one(model, processor, sample["video"], sample["q"], device, warmup=True)

    regimes = [
        {"name": "AutoGaze only",      "mode": "gaze_only",     "keep": 1.0},
        {"name": "Intersect 30%",      "mode": "intersect",     "keep": 0.3},
        {"name": "Semantic only 30%",  "mode": "semantic_only", "keep": 0.3},
    ]

    print()
    print(f"{'regime':<22} {'preproc(s)':>10} {'togpu(s)':>9} {'generate(s)':>11} "
          f"{'total(s)':>9}  {'n_input':>8} {'n_gen':>5}")

    # Reset once
    if hasattr(processor, "_original_get_gazing"):
        processor._get_gazing_info_from_videos = processor._original_get_gazing
    else:
        processor._original_get_gazing = processor._get_gazing_info_from_videos

    for cfg in regimes:
        # Reset patch
        processor._get_gazing_info_from_videos = processor._original_get_gazing
        if cfg["mode"] != "gaze_only":
            patch_processor_with_semantic_filter(
                processor, wrapper, clip_model, clip_tok,
                mode=cfg["mode"], semantic_keep_ratio=cfg["keep"],
                query_text=sample["stem"], device=str(device),
            )

        # Run N times
        runs = [run_one(model, processor, sample["video"], sample["q"], device) for _ in range(args.n_runs)]
        avg = {k: sum(r[k] for r in runs) / len(runs) for k in
               ["preprocess_s", "togpu_s", "generate_s", "total_s", "n_input_tokens", "n_gen_tokens"]}
        print(f"{cfg['name']:<22} {avg['preprocess_s']:>10.2f} {avg['togpu_s']:>9.2f} "
              f"{avg['generate_s']:>11.2f} {avg['total_s']:>9.2f}  "
              f"{avg['n_input_tokens']:>8.0f} {avg['n_gen_tokens']:>5.0f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="nvidia/NVILA-8B-HD-Video")
    p.add_argument("--video_dir", default="hlvid_videos/extracted/videos")
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--num_frames_thumbnail", type=int, default=16)
    p.add_argument("--max_tiles", type=int, default=2)
    p.add_argument("--gazing_ratio", type=float, default=0.2)
    p.add_argument("--ckpt", default="results/bighead_warmrestart/best_bighead_student.pt")
    p.add_argument("--head_type", default="bighead")
    p.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_runs", type=int, default=3)
    args = p.parse_args()
    main(args)
