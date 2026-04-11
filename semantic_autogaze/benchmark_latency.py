"""
Latency benchmark for Semantic AutoGaze variants.

Measures end-to-end inference time for:
1. Vanilla AutoGaze (no semantic head)
2. AutoGaze + small SimilarityHead (201K params)
3. AutoGaze + BigSimilarityHead (3.4M params)
4. Pre-decoder + BigSimilarityHead (skip LLaMA decoder for semantic branch)

Each variant is measured with warm-up, then averaged over multiple runs.
Reports: total time, per-component breakdown, and throughput.
"""

import os
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from autogaze.models.autogaze import AutoGaze
from semantic_autogaze.model import SimilarityHead
from semantic_autogaze.train_bighead import BigSimilarityHead


def benchmark_vanilla_autogaze(autogaze, video, n_warmup=5, n_runs=50, device="cuda"):
    """Benchmark vanilla AutoGaze (encoder + decoder, no semantic head)."""
    autogaze.eval()
    gaze_model = autogaze.gazing_model

    B, T = video.shape[:2]
    video_resized = rearrange(video, 'b t c h w -> (b t) c h w')
    video_resized = F.interpolate(
        video_resized,
        size=(gaze_model.input_img_size, gaze_model.input_img_size),
        mode="bicubic", align_corners=False,
    )
    video_resized = rearrange(video_resized, '(b t) c h w -> b t c h w', b=B)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            vision_features, _ = gaze_model.vision_model(video_resized)
            vision_features = vision_features.transpose(1, 2)
            vision_features = rearrange(vision_features, 'b t c h w -> b t (h w) c')
            vision_features = gaze_model.connector(vision_features)
            B2, T2, N, C = vision_features.shape
            inputs_embeds = vision_features.reshape(B2, T2 * N, C)
            attention_mask = torch.ones(B2, T2 * N, device=device, dtype=torch.long)
            decoder_outputs = gaze_model.gaze_decoder.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=attention_mask.cumsum(dim=-1) - 1,
            )
    torch.cuda.synchronize()

    # Benchmark components separately
    encoder_times = []
    decoder_times = []
    total_times = []

    with torch.no_grad():
        for _ in range(n_runs):
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            vision_features, _ = gaze_model.vision_model(video_resized)
            vision_features = vision_features.transpose(1, 2)
            vision_features = rearrange(vision_features, 'b t c h w -> b t (h w) c')
            vision_features = gaze_model.connector(vision_features)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            B2, T2, N, C = vision_features.shape
            inputs_embeds = vision_features.reshape(B2, T2 * N, C)
            attention_mask = torch.ones(B2, T2 * N, device=device, dtype=torch.long)
            decoder_outputs = gaze_model.gaze_decoder.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=attention_mask.cumsum(dim=-1) - 1,
            )
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            encoder_times.append(t1 - t0)
            decoder_times.append(t2 - t1)
            total_times.append(t2 - t0)

    return {
        "encoder_ms": np.mean(encoder_times) * 1000,
        "decoder_ms": np.mean(decoder_times) * 1000,
        "total_ms": np.mean(total_times) * 1000,
        "encoder_std": np.std(encoder_times) * 1000,
        "decoder_std": np.std(decoder_times) * 1000,
        "total_std": np.std(total_times) * 1000,
    }


def benchmark_with_head(autogaze, head, video, query_emb, use_predecoder=False,
                         n_warmup=5, n_runs=50, device="cuda"):
    """Benchmark AutoGaze + similarity head."""
    autogaze.eval()
    head.eval()
    gaze_model = autogaze.gazing_model

    B, T = video.shape[:2]
    video_resized = rearrange(video, 'b t c h w -> (b t) c h w')
    video_resized = F.interpolate(
        video_resized,
        size=(gaze_model.input_img_size, gaze_model.input_img_size),
        mode="bicubic", align_corners=False,
    )
    video_resized = rearrange(video_resized, '(b t) c h w -> b t c h w', b=B)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            vision_features, _ = gaze_model.vision_model(video_resized)
            vision_features = vision_features.transpose(1, 2)
            vision_features = rearrange(vision_features, 'b t c h w -> b t (h w) c')
            vision_features = gaze_model.connector(vision_features)

            if use_predecoder:
                hidden = vision_features.reshape(B, T * 196, 192)
            else:
                B2, T2, N, C = vision_features.shape
                inputs_embeds = vision_features.reshape(B2, T2 * N, C)
                attention_mask = torch.ones(B2, T2 * N, device=device, dtype=torch.long)
                decoder_outputs = gaze_model.gaze_decoder.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=attention_mask.cumsum(dim=-1) - 1,
                )
                hidden = decoder_outputs.last_hidden_state

            scores = head(hidden, query_emb)
    torch.cuda.synchronize()

    encoder_times = []
    decoder_times = []
    head_times = []
    total_times = []

    with torch.no_grad():
        for _ in range(n_runs):
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            vision_features, _ = gaze_model.vision_model(video_resized)
            vision_features = vision_features.transpose(1, 2)
            vision_features = rearrange(vision_features, 'b t c h w -> b t (h w) c')
            vision_features = gaze_model.connector(vision_features)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            if use_predecoder:
                hidden = vision_features.reshape(B, T * 196, 192)
                torch.cuda.synchronize()
                t2 = t1  # no decoder
            else:
                B2, T2, N, C = vision_features.shape
                inputs_embeds = vision_features.reshape(B2, T2 * N, C)
                attention_mask = torch.ones(B2, T2 * N, device=device, dtype=torch.long)
                decoder_outputs = gaze_model.gaze_decoder.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=attention_mask.cumsum(dim=-1) - 1,
                )
                hidden = decoder_outputs.last_hidden_state
                torch.cuda.synchronize()
                t2 = time.perf_counter()

            scores = head(hidden, query_emb)
            torch.cuda.synchronize()
            t3 = time.perf_counter()

            encoder_times.append(t1 - t0)
            decoder_times.append(t2 - t1)
            head_times.append(t3 - t2)
            total_times.append(t3 - t0)

    return {
        "encoder_ms": np.mean(encoder_times) * 1000,
        "decoder_ms": np.mean(decoder_times) * 1000,
        "head_ms": np.mean(head_times) * 1000,
        "total_ms": np.mean(total_times) * 1000,
        "encoder_std": np.std(encoder_times) * 1000,
        "decoder_std": np.std(decoder_times) * 1000,
        "head_std": np.std(head_times) * 1000,
        "total_std": np.std(total_times) * 1000,
    }


def main(args):
    device = torch.device(args.device)

    print("Loading AutoGaze...")
    autogaze = AutoGaze.from_pretrained(args.autogaze_model, use_flash_attn=False).to(device).eval()
    hidden_dim = autogaze.config.gaze_model_config.gaze_decoder_config.hidden_size

    # Create dummy input
    print(f"Creating dummy video: B=1, T={args.num_frames}, 224x224")
    video = torch.randn(1, args.num_frames, 3, 224, 224, device=device)
    query_emb = torch.randn(1, 512, device=device)  # CLIP text embedding

    # 1. Vanilla AutoGaze
    print("\n" + "="*60)
    print("1. Vanilla AutoGaze (encoder + decoder)")
    print("="*60)
    vanilla = benchmark_vanilla_autogaze(autogaze, video, args.n_warmup, args.n_runs, device)
    print(f"  Encoder:  {vanilla['encoder_ms']:.2f} ± {vanilla['encoder_std']:.2f} ms")
    print(f"  Decoder:  {vanilla['decoder_ms']:.2f} ± {vanilla['decoder_std']:.2f} ms")
    print(f"  Total:    {vanilla['total_ms']:.2f} ± {vanilla['total_std']:.2f} ms")

    # 2. + Small SimilarityHead
    print("\n" + "="*60)
    print("2. AutoGaze + Small SimilarityHead (201K params)")
    print("="*60)
    small_head = SimilarityHead(hidden_dim=hidden_dim, embedding_dim=512,
                                 grid_size=14, num_frames=args.num_frames,
                                 use_spatial=True).to(device).eval()
    if args.small_ckpt and os.path.exists(args.small_ckpt):
        small_head.load_state_dict(torch.load(args.small_ckpt, map_location=device))
    small = benchmark_with_head(autogaze, small_head, video, query_emb,
                                 use_predecoder=False, n_warmup=args.n_warmup,
                                 n_runs=args.n_runs, device=device)
    print(f"  Encoder:  {small['encoder_ms']:.2f} ± {small['encoder_std']:.2f} ms")
    print(f"  Decoder:  {small['decoder_ms']:.2f} ± {small['decoder_std']:.2f} ms")
    print(f"  Head:     {small['head_ms']:.2f} ± {small['head_std']:.2f} ms")
    print(f"  Total:    {small['total_ms']:.2f} ± {small['total_std']:.2f} ms")
    print(f"  Overhead: +{small['total_ms'] - vanilla['total_ms']:.2f} ms ({(small['total_ms']/vanilla['total_ms'] - 1)*100:.1f}%)")

    # 3. + BigSimilarityHead (post-decoder)
    print("\n" + "="*60)
    print("3. AutoGaze + BigSimilarityHead post-decoder (3.4M params)")
    print("="*60)
    big_head = BigSimilarityHead(hidden_dim=hidden_dim, embedding_dim=512,
                                  expanded_dim=384, n_attn_heads=6,
                                  n_attn_layers=2, grid_size=14).to(device).eval()
    if args.bighead_ckpt and os.path.exists(args.bighead_ckpt):
        big_head.load_state_dict(torch.load(args.bighead_ckpt, map_location=device))
    big = benchmark_with_head(autogaze, big_head, video, query_emb,
                               use_predecoder=False, n_warmup=args.n_warmup,
                               n_runs=args.n_runs, device=device)
    print(f"  Encoder:  {big['encoder_ms']:.2f} ± {big['encoder_std']:.2f} ms")
    print(f"  Decoder:  {big['decoder_ms']:.2f} ± {big['decoder_std']:.2f} ms")
    print(f"  Head:     {big['head_ms']:.2f} ± {big['head_std']:.2f} ms")
    print(f"  Total:    {big['total_ms']:.2f} ± {big['total_std']:.2f} ms")
    print(f"  Overhead: +{big['total_ms'] - vanilla['total_ms']:.2f} ms ({(big['total_ms']/vanilla['total_ms'] - 1)*100:.1f}%)")

    # 4. Pre-decoder + BigSimilarityHead (skip LLaMA decoder)
    print("\n" + "="*60)
    print("4. Pre-decoder + BigSimilarityHead (skip decoder, 3.4M params)")
    print("="*60)
    big_head_pre = BigSimilarityHead(hidden_dim=hidden_dim, embedding_dim=512,
                                      expanded_dim=384, n_attn_heads=6,
                                      n_attn_layers=2, grid_size=14).to(device).eval()
    if args.predecoder_ckpt and os.path.exists(args.predecoder_ckpt):
        big_head_pre.load_state_dict(torch.load(args.predecoder_ckpt, map_location=device))
    pre = benchmark_with_head(autogaze, big_head_pre, video, query_emb,
                               use_predecoder=True, n_warmup=args.n_warmup,
                               n_runs=args.n_runs, device=device)
    print(f"  Encoder:  {pre['encoder_ms']:.2f} ± {pre['encoder_std']:.2f} ms")
    print(f"  Decoder:  {pre['decoder_ms']:.2f} ms (skipped)")
    print(f"  Head:     {pre['head_ms']:.2f} ± {pre['head_std']:.2f} ms")
    print(f"  Total:    {pre['total_ms']:.2f} ± {pre['total_std']:.2f} ms")
    print(f"  Overhead: +{pre['total_ms'] - vanilla['total_ms']:.2f} ms ({(pre['total_ms']/vanilla['total_ms'] - 1)*100:.1f}%)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Variant':<45} {'Total (ms)':<15} {'Overhead':<15}")
    print("-" * 75)
    print(f"{'1. Vanilla AutoGaze':<45} {vanilla['total_ms']:<15.2f} {'baseline':<15}")

    small_oh = small['total_ms'] - vanilla['total_ms']
    small_pct = (small['total_ms'] / vanilla['total_ms'] - 1) * 100
    print(f"{'2. + Small head (201K)':<45} {small['total_ms']:<15.2f} +{small_oh:.2f}ms ({small_pct:.1f}%)")

    big_oh = big['total_ms'] - vanilla['total_ms']
    big_pct = (big['total_ms'] / vanilla['total_ms'] - 1) * 100
    print(f"{'3. + BigHead post-decoder (3.4M)':<45} {big['total_ms']:<15.2f} +{big_oh:.2f}ms ({big_pct:.1f}%)")

    pre_oh = pre['total_ms'] - vanilla['total_ms']
    pre_pct = (pre['total_ms'] / vanilla['total_ms'] - 1) * 100
    print(f"{'4. Pre-decoder + BigHead (3.4M)':<45} {pre['total_ms']:<15.2f} +{pre_oh:.2f}ms ({pre_pct:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--small_ckpt", default="results/distill/best_student.pt")
    parser.add_argument("--bighead_ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--predecoder_ckpt", default="results/distill_predecoder/best_predecoder_bighead.pt")
    args = parser.parse_args()
    main(args)
