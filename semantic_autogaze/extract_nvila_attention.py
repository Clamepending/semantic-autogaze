"""r/nvila-attention-distill Phase 1 — extract NVILA's question→visual cross-attention.

Implementation of Option B (smaller supervision config + output_attentions=True).
For each (video, question) on HLVid household + av:
1. Load NVILA at max_tiles=1, num_video_frames=16, gazing_ratio=0.50 (small enough
   that attention matrices fit in 24 GB).
2. Forward pass with output_attentions=True (NOT generate — we only need prefill
   attention).
3. Extract attentions[LAYER] (1, num_heads, S, S). Slice (q_positions, v_positions),
   head-mean. Map back to (T, 14, 14) using gazing_info.
4. Save to results/nvila_attention_cache/qid_{qid}.npz.

Layer choice: pilot at layer 14 (middle of 28-layer LLaMA-style decoder).
Future cycle 2 may sweep layers/heads.

Run:
  CUDA_VISIBLE_DEVICES=N python -m semantic_autogaze.extract_nvila_attention \\
    --device cuda:0 --output_dir results/nvila_attention_cache --layer 14
"""
from __future__ import annotations
import os, json, time, argparse
from typing import Optional
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from semantic_autogaze.eval_hlvid_subset import load_subset, PARQUET_PATH


VIDEO_TOKEN_ID = 151650


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[setup] Loading NVILA at smaller config for attention extraction...", flush=True)
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
        attn_implementation="eager",  # required for output_attentions
    ).eval()
    print(f"[setup] NVILA loaded; cuda mem={torch.cuda.memory_allocated()/1024**3:.2f} GB",
          flush=True)
    print(f"[setup] LLM has {len(model.llm.model.layers)} layers", flush=True)

    # Load HLVid samples
    samples = load_subset(args.video_dir, parquet_path=args.parquet_path, query_mode="stem")
    if args.category:
        samples = [s for s in samples if s.get("category") == args.category]
    if args.n_samples is not None:
        samples = samples[:args.n_samples]
    print(f"[data] Loaded {len(samples)} samples (category={args.category})", flush=True)

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
            inputs = processor(
                text=f"{video_token}\n\n{s['question_raw']}",
                videos=s["video_path"],
                return_tensors="pt",
            )
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            input_ids = inputs["input_ids"][0]  # (S,)
            video_token_mask = (input_ids == VIDEO_TOKEN_ID)
            v_positions = torch.where(video_token_mask)[0]  # positions of video tokens
            num_v = len(v_positions)
            # Question tokens = everything AFTER the last video token
            last_v = v_positions.max().item()
            q_positions = torch.arange(last_v + 1, len(input_ids), device=device)
            num_q = len(q_positions)

            with torch.inference_mode():
                outputs = model.forward(
                    **inputs,
                    output_attentions=True,
                    return_dict=True,
                )
            attns = outputs.attentions  # tuple of (1, num_heads, S, S) for each layer
            layer_attn = attns[LAYER][0]  # (num_heads, S, S)

            # Extract question→visual attention
            # attn[h, q_pos, v_pos]: how much head h attends from q_pos to v_pos
            q_to_v = layer_attn[:, q_positions[:, None], v_positions[None, :]]  # (heads, num_q, num_v)
            # Average over heads + question tokens
            attn_map = q_to_v.mean(dim=(0, 1))  # (num_v,)

            # Map back to (T, 14, 14) per frame using gazing_info
            # gazing_info contains gazing_pos_tiles and gazing_pos_thumbnails which are
            # (num_units, K) tensors of patch positions on the original 14x14 grid.
            # Visual tokens in the LLM sequence come from per_video_features which
            # concatenates kept tile + thumb patches per video.
            #
            # For our smaller config (max_tiles=1, num_frames=16), there is 1 tile
            # holding all 16 frames + 16 thumbnail single-frames. After AutoGaze with
            # gazing_ratio=0.50, ~50% of patches/frame are kept. The ordering follows
            # `_encode_vision`: tiles first (per spatial_tile per frame), then thumbs.
            #
            # We DON'T have direct access to gazing_info here. NVILA computed it
            # internally and used it to select kept patches. We need to either:
            # (a) hook into model.forward to capture gazing_info, or
            # (b) re-derive gazing_info via a separate AutoGaze call.
            #
            # Cleanest path: hook _encode_vision to capture gazing_info.
            # For now, save the raw attention map and num_v; reconstruct (T, 14, 14)
            # in a post-processing pass that re-runs AutoGaze.

            attn_map_cpu = attn_map.detach().to(torch.float32).cpu().numpy()
            wall = time.perf_counter() - t0

            np.savez_compressed(
                out_path,
                attention=attn_map_cpu,             # (num_v,) — raw NVILA attention scores
                num_v=int(num_v),
                num_q=int(num_q),
                v_positions=v_positions.detach().cpu().numpy(),
                q_positions=q_positions.detach().cpu().numpy(),
                qid=int(qid),
                question_stem=s["question_stem"],
                video_path=s["video_path"],
                layer=LAYER,
                config_max_tiles=args.max_tiles,
                config_num_frames=args.num_frames,
                config_gazing_ratio=args.gazing_ratio,
            )
            print(f"  [{i+1}/{len(samples)}] qid={qid} num_v={num_v} num_q={num_q} "
                  f"min={attn_map_cpu.min():.4f} max={attn_map_cpu.max():.4f} "
                  f"wall={wall:.1f}s", flush=True)
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
    p.add_argument("--category", default="household",
                   help="HLVid category filter; '' to keep all")
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--num_frames_thumbnail", type=int, default=16)
    p.add_argument("--max_tiles", type=int, default=1)
    p.add_argument("--gazing_ratio", type=float, default=0.50)
    p.add_argument("--gazing_ratio_thumbnail", type=float, default=0.75)
    p.add_argument("--layer", type=int, default=14, help="LLM layer index for attention extraction")
    p.add_argument("--output_dir", default="results/nvila_attention_cache")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    main(args)
