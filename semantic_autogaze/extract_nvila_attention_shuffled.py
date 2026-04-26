"""r/nvila-attention-distill cycle 1.5 prep — extract NVILA attention with SHUFFLED queries.

For each (qid, video) on HLVid household, re-run NVILA forward at the same smaller
config (max_tiles=1, num_frames=16, gazing_ratio=0.50) but with the question
text replaced by a deterministic-shuffled OTHER question. Cache attention to
results/nvila_attention_cache_shuffled/.

This is the cheap diagnostic for whether NVILA's cross-attention is
query-conditional at all. If matched-Q vs shuffled-Q top-K overlap is high
(≈ K/num_v), attention is query-INVARIANT → distillation pre-empted.

Run:
  CUDA_VISIBLE_DEVICES=N python -m semantic_autogaze.extract_nvila_attention_shuffled \\
    --device cuda:0 --output_dir results/nvila_attention_cache_shuffled
"""
from __future__ import annotations
import os, json, time, argparse, random
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from semantic_autogaze.eval_hlvid_subset import load_subset, PARQUET_PATH


VIDEO_TOKEN_ID = 151650


def deterministic_shuffle(qids, seed=42):
    """Same as r/owlvit-hlvid-vqa: derangement (no fixed points). Falls back to
    cyclic rotation if rejection sampling doesn't find one in 200 tries."""
    n = len(qids)
    if n < 2:
        return list(qids)
    rng = random.Random(seed)
    result = list(qids)
    for _ in range(200):
        rng.shuffle(result)
        if all(result[i] != qids[i] for i in range(n)):
            return result
    return [qids[(i + 1) % n] for i in range(n)]


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[setup] Loading NVILA at smaller config for shuffled-Q extraction...", flush=True)
    print(f"        max_tiles={args.max_tiles}, num_frames={args.num_frames}, "
          f"gazing_ratio={args.gazing_ratio}", flush=True)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        num_video_frames=args.num_frames,
        num_video_frames_thumbnail=args.num_frames_thumbnail,
        max_tiles_video=args.max_tiles,
        gazing_ratio_tile=args.gazing_ratio,
        gazing_ratio_thumbnail=args.gazing_ratio_thumbnail,
        task_loss_requirement_tile=0.6,
        task_loss_requirement_thumbnail=0.6,
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
        attn_implementation="eager",
    ).eval()
    print(f"[setup] NVILA loaded; cuda mem={torch.cuda.memory_allocated()/1024**3:.2f} GB",
          flush=True)

    samples = load_subset(args.video_dir, parquet_path=args.parquet_path, query_mode="stem")
    if args.category:
        samples = [s for s in samples if s.get("category") == args.category]
    if args.n_samples is not None:
        samples = samples[:args.n_samples]

    qids = [s["question_id"] for s in samples]
    shuffled_qids = deterministic_shuffle(qids, seed=42)
    qid_to_question_raw = {s["question_id"]: s["question_raw"] for s in samples}
    qid_to_shuffled_raw = {qids[i]: qid_to_question_raw[shuffled_qids[i]] for i in range(len(qids))}

    print(f"[data] {len(samples)} samples (category={args.category}); "
          f"each will run NVILA with the SHUFFLED question_raw text", flush=True)
    print(f"[shuffle] sample[0] real_q='{samples[0]['question_stem'][:60]}'", flush=True)
    print(f"          sample[0] shuffled_q='{qid_to_shuffled_raw[qids[0]][:60].splitlines()[0]}'",
          flush=True)

    LAYER = args.layer

    for i, s in enumerate(samples):
        qid = s["question_id"]
        out_path = os.path.join(args.output_dir, f"qid_{qid:04d}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"  [{i+1}/{len(samples)}] qid={qid} already cached, skipping", flush=True)
            continue

        try:
            t0 = time.perf_counter()
            video_token = processor.tokenizer.video_token
            shuffled_q_raw = qid_to_shuffled_raw[qid]
            inputs = processor(
                text=f"{video_token}\n\n{shuffled_q_raw}",
                videos=s["video_path"],
                return_tensors="pt",
            )
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            input_ids = inputs["input_ids"][0]
            video_token_mask = (input_ids == VIDEO_TOKEN_ID)
            v_positions = torch.where(video_token_mask)[0]
            num_v = len(v_positions)
            last_v = v_positions.max().item()
            q_positions = torch.arange(last_v + 1, len(input_ids), device=device)
            num_q = len(q_positions)

            with torch.inference_mode():
                outputs = model.forward(
                    **inputs,
                    output_attentions=True,
                    return_dict=True,
                )
            attns = outputs.attentions
            layer_attn = attns[LAYER][0]

            q_to_v = layer_attn[:, q_positions[:, None], v_positions[None, :]]
            attn_map = q_to_v.mean(dim=(0, 1))
            attn_map_cpu = attn_map.detach().to(torch.float32).cpu().numpy()
            wall = time.perf_counter() - t0

            np.savez_compressed(
                out_path,
                attention=attn_map_cpu,
                num_v=int(num_v),
                num_q=int(num_q),
                v_positions=v_positions.detach().cpu().numpy(),
                q_positions=q_positions.detach().cpu().numpy(),
                qid=int(qid),
                question_raw=shuffled_q_raw,
                shuffled_from_qid=int(shuffled_qids[i]) if shuffled_qids[i] != qid else -1,
                video_path=s["video_path"],
                layer=LAYER,
                config_max_tiles=args.max_tiles,
                config_num_frames=args.num_frames,
                config_gazing_ratio=args.gazing_ratio,
            )
            valid = not np.isnan(attn_map_cpu).any()
            print(f"  [{i+1}/{len(samples)}] qid={qid} (shufQ from qid={shuffled_qids[i]}) "
                  f"num_v={num_v} num_q={num_q} "
                  f"valid={valid} max={attn_map_cpu.max() if valid else 'NaN'} wall={wall:.1f}s",
                  flush=True)
        except Exception as e:
            import traceback
            print(f"  [error] qid={qid}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model_path", default="nvidia/NVILA-8B-HD-Video")
    p.add_argument("--parquet_path", default=PARQUET_PATH)
    p.add_argument("--video_dir", default="hlvid_videos/extracted_household/videos")
    p.add_argument("--category", default="household")
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--num_frames_thumbnail", type=int, default=16)
    p.add_argument("--max_tiles", type=int, default=1)
    p.add_argument("--gazing_ratio", type=float, default=0.50)
    p.add_argument("--gazing_ratio_thumbnail", type=float, default=0.75)
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--output_dir", default="results/nvila_attention_cache_shuffled")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    main(args)
