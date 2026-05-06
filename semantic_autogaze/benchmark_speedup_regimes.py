"""
Speedup characterization: where does semantic filtering actually help?

Sweeps num_frames ∈ {16, 32, 64, 128} and reports wall-clock latency for:

  (a) AutoGaze only (encoder + decoder + SigLIP @ 75% tokens) — baseline
  (b) Intersect 50% (AutoGaze + sem head + SigLIP @ 50% of 75% ≈ 37.5%)
  (c) Intersect 10% (AutoGaze + sem head + SigLIP @ 10% of 75% ≈ 7.5%)
  (d) Semantic-only 10% (skip AutoGaze decoder; encoder + head + SigLIP @ 10%)
  (e) Semantic-only 2%  (extreme "4K 1K-frame" regime)

Component times are measured independently then summed, so we can attribute
speedup to: (1) decoder bypass, (2) SigLIP token reduction.

Uses a dummy random video (fair for latency; doesn't touch data pipeline).

Usage:
  CUDA_VISIBLE_DEVICES=2 python3 -m semantic_autogaze.benchmark_speedup_regimes \
    --device cuda:0 --output_dir results/speedup_regimes
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange

from autogaze.models.autogaze import AutoGaze
from semantic_autogaze.train_bighead import BigSimilarityHead


GRID = 14
N = GRID * GRID  # 196 patches per frame


def cuda_time(fn, n_warmup=3, n_runs=20):
    """Run fn() and return mean / std latency in ms."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        fn()
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))
    return float(np.mean(times)), float(np.std(times))


def time_autogaze_encoder(gaze_model, video_resized, n_warmup, n_runs):
    def _fn():
        with torch.no_grad():
            vf, _ = gaze_model.vision_model(video_resized)
            vf = vf.transpose(1, 2)
            vf = rearrange(vf, 'b t c h w -> b t (h w) c')
            vf = gaze_model.connector(vf)
            return vf
    return cuda_time(_fn, n_warmup, n_runs)


def time_autogaze_decoder(gaze_model, vision_features, n_warmup, n_runs):
    B, T, N_, C = vision_features.shape
    inputs_embeds = vision_features.reshape(B, T * N_, C)
    attn = torch.ones(B, T * N_, device=vision_features.device, dtype=torch.long)
    pos = attn.cumsum(dim=-1) - 1
    def _fn():
        with torch.no_grad():
            gaze_model.gaze_decoder.model(
                inputs_embeds=inputs_embeds, attention_mask=attn, position_ids=pos,
            )
    return cuda_time(_fn, n_warmup, n_runs)


def time_similarity_head(head, hidden, query, n_warmup, n_runs):
    def _fn():
        with torch.no_grad():
            head(hidden, query)
    return cuda_time(_fn, n_warmup, n_runs)


def time_siglip(siglip_model, video_siglip, gazing_info, n_warmup, n_runs):
    def _fn():
        with torch.no_grad():
            siglip_model(video_siglip, gazing_info=gazing_info)
    return cuda_time(_fn, n_warmup, n_runs)


def make_gazing_info(T, N, num_tokens_kept, device):
    """Build a synthetic gazing_info keeping num_tokens_kept patches total."""
    from autogaze.utils import get_gazing_pos_from_gazing_mask
    mask = torch.zeros(1, T * N, dtype=torch.long, device=device)
    # Keep roughly uniform across frames
    per_frame = max(1, num_tokens_kept // T)
    for t in range(T):
        start = t * N
        mask[0, start:start + per_frame] = 1
    gazing_pos, if_padded = get_gazing_pos_from_gazing_mask(mask)
    num_gazing_each_frame = mask.reshape(1, T, N)[0].sum(dim=-1)
    return {
        "gazing_pos": gazing_pos,
        "if_padded_gazing": if_padded,
        "num_gazing_each_frame": num_gazing_each_frame,
    }


def run_for_frames(T, device, autogaze, head, siglip_model,
                   siglip_transform_info, args):
    """Returns a dict of component timings + derived total latencies."""
    gaze_model = autogaze.gazing_model
    results = {"num_frames": T, "components": {}, "regimes": {}}

    # Dummy video at AutoGaze input size
    video_ag = torch.randn(1, T, 3, gaze_model.input_img_size,
                           gaze_model.input_img_size, device=device)
    # SigLIP video size (224)
    video_sg = torch.randn(1, T, 3, 224, 224, device=device)

    # (1) Encoder
    enc_mean, enc_std = time_autogaze_encoder(gaze_model, video_ag,
                                               args.n_warmup, args.n_runs)
    results["components"]["encoder_ms"] = enc_mean
    results["components"]["encoder_std"] = enc_std

    # Need vision_features for (2)
    with torch.no_grad():
        vf, _ = gaze_model.vision_model(video_ag)
        vf = vf.transpose(1, 2)
        vf = rearrange(vf, 'b t c h w -> b t (h w) c')
        vf = gaze_model.connector(vf)

    # (2) Decoder
    dec_mean, dec_std = time_autogaze_decoder(gaze_model, vf,
                                               args.n_warmup, args.n_runs)
    results["components"]["decoder_ms"] = dec_mean
    results["components"]["decoder_std"] = dec_std

    with torch.no_grad():
        B = 1
        inputs_embeds = vf.reshape(B, T * N, -1)
        attn = torch.ones(B, T * N, device=device, dtype=torch.long)
        pos = attn.cumsum(dim=-1) - 1
        hidden = gaze_model.gaze_decoder.model(
            inputs_embeds=inputs_embeds, attention_mask=attn, position_ids=pos,
        ).last_hidden_state
        query = torch.randn(1, 512, device=device)

    # (3) Semantic head (post-decoder)
    head_mean, head_std = time_similarity_head(head, hidden, query,
                                                args.n_warmup, args.n_runs)
    results["components"]["head_ms"] = head_mean
    results["components"]["head_std"] = head_std

    # (3b) Semantic head on pre-decoder features (if using that route)
    pre_hidden = vf.reshape(B, T * N, -1)
    pre_head_mean, _ = time_similarity_head(head, pre_hidden, query,
                                             args.n_warmup, args.n_runs)
    results["components"]["head_predecoder_ms"] = pre_head_mean

    # (4) SigLIP at multiple token counts
    siglip_times = {}
    if siglip_model is not None:
        # Representative budgets:  75% of T*N, 50% of 75%, 10% of 75%, 10%, 2%
        budgets = {
            "gaze_75pct": int(0.75 * T * N),
            "intersect_50_of_75": int(0.50 * 0.75 * T * N),
            "intersect_10_of_75": int(0.10 * 0.75 * T * N),
            "semantic_10pct": int(0.10 * T * N),
            "semantic_2pct": int(0.02 * T * N),
        }
        for label, budget in budgets.items():
            budget = max(T, budget)
            info = make_gazing_info(T, N, budget, device)
            try:
                sg_mean, sg_std = time_siglip(siglip_model, video_sg, info,
                                               args.n_warmup, args.n_runs)
                siglip_times[label] = {"budget": budget, "ms": sg_mean, "std": sg_std}
            except Exception as e:
                siglip_times[label] = {"budget": budget, "error": str(e)}
        results["components"]["siglip"] = siglip_times

    # Derive regime totals
    enc = enc_mean
    dec = dec_mean
    head_post = head_mean
    head_pre = pre_head_mean

    def sglip(label):
        v = siglip_times.get(label, {})
        return v.get("ms", 0.0)

    # (a) AutoGaze only: encoder + decoder + SigLIP@75%
    results["regimes"]["a_autogaze_baseline"] = enc + dec + sglip("gaze_75pct")
    # (b) Intersect 50%: enc + dec + head + SigLIP@37.5%
    results["regimes"]["b_intersect_50"] = enc + dec + head_post + sglip("intersect_50_of_75")
    # (c) Intersect 10%: enc + dec + head + SigLIP@7.5%
    results["regimes"]["c_intersect_10"] = enc + dec + head_post + sglip("intersect_10_of_75")
    # (d) Semantic-only 10%: enc + pre-head + SigLIP@10% (no decoder)
    results["regimes"]["d_semantic_10"] = enc + head_pre + sglip("semantic_10pct")
    # (e) Semantic-only 2%: enc + pre-head + SigLIP@2% (extreme)
    results["regimes"]["e_semantic_2"] = enc + head_pre + sglip("semantic_2pct")

    return results


def main(args):
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading AutoGaze...")
    autogaze = AutoGaze.from_pretrained(args.autogaze_model,
                                        use_flash_attn=False).to(device).eval()
    hidden_dim = autogaze.config.gaze_model_config.gaze_decoder_config.hidden_size

    print("Loading BigSimilarityHead...")
    head = BigSimilarityHead(
        hidden_dim=hidden_dim, embedding_dim=512, expanded_dim=384,
        n_attn_heads=6, n_attn_layers=2, grid_size=GRID,
    ).to(device).eval()
    if args.ckpt and os.path.exists(args.ckpt):
        head.load_state_dict(torch.load(args.ckpt, map_location=device))

    siglip_model = None
    siglip_info = None
    if not args.no_siglip:
        try:
            print("Loading SigLIP...")
            from autogaze.vision_encoders.siglip import SiglipVisionModel
            siglip_model = SiglipVisionModel.from_pretrained(
                "google/siglip2-base-patch16-224",
                scales=autogaze.config.scales,
                attn_implementation="sdpa",
            ).to(device).eval()
        except Exception as e:
            print(f"SigLIP not available: {e}")
            siglip_model = None

    frame_counts = [int(t) for t in args.frames.split(",")]
    all_results = []

    for T in frame_counts:
        print(f"\n{'='*60}")
        print(f"num_frames = {T}")
        print(f"{'='*60}")
        try:
            res = run_for_frames(T, device, autogaze, head, siglip_model,
                                 siglip_info, args)
            all_results.append(res)
            c = res["components"]
            print(f"  Encoder:         {c['encoder_ms']:7.2f} ms")
            print(f"  Decoder:         {c['decoder_ms']:7.2f} ms")
            print(f"  Head (post-dec): {c['head_ms']:7.2f} ms")
            print(f"  Head (pre-dec):  {c['head_predecoder_ms']:7.2f} ms")
            if siglip_model is not None:
                for k, v in c.get("siglip", {}).items():
                    if "ms" in v:
                        print(f"  SigLIP {k:<22}: {v['ms']:7.2f} ms  ({v['budget']} tokens)")
                    else:
                        print(f"  SigLIP {k:<22}: ERROR {v['error'][:60]}")
            print(f"  Regime totals:")
            for k, v in res["regimes"].items():
                print(f"    {k:<25}: {v:7.2f} ms")
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at T={T}: {e}")
            all_results.append({"num_frames": T, "error": "OOM"})
            torch.cuda.empty_cache()

    # Save JSON
    with open(os.path.join(args.output_dir, "speedup_regimes.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {args.output_dir}/speedup_regimes.json")

    # Plot
    valid = [r for r in all_results if "components" in r]
    if valid:
        frames = [r["num_frames"] for r in valid]
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        regime_keys = [
            ("a_autogaze_baseline", "AutoGaze only (75%)", "tab:gray"),
            ("b_intersect_50", "Intersect 50%", "tab:blue"),
            ("c_intersect_10", "Intersect 10%", "tab:green"),
            ("d_semantic_10", "Semantic-only 10%", "tab:orange"),
            ("e_semantic_2", "Semantic-only 2%", "tab:red"),
        ]
        for k, label, color in regime_keys:
            ys = [r["regimes"].get(k, float("nan")) for r in valid]
            ax.plot(frames, ys, "o-", label=label, color=color, lw=2)
        ax.set_xlabel("num_frames")
        ax.set_ylabel("Total latency (ms)")
        ax.set_title("End-to-end latency by frame count")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9)

        baseline = np.array([r["regimes"]["a_autogaze_baseline"] for r in valid])
        for k, label, color in regime_keys[1:]:
            ys = np.array([r["regimes"].get(k, np.nan) for r in valid])
            speedup = baseline / ys
            ax2.plot(frames, speedup, "o-", label=label, color=color, lw=2)
        ax2.axhline(1.0, color="black", linestyle=":", lw=1)
        ax2.set_xlabel("num_frames")
        ax2.set_ylabel("Speedup vs AutoGaze baseline")
        ax2.set_title("Speedup by regime (higher = faster)")
        ax2.set_xscale("log", base=2)
        ax2.grid(True, alpha=0.3, which="both")
        ax2.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "speedup_regimes.png"), dpi=150)
        plt.close(fig)
        print(f"Saved: {args.output_dir}/speedup_regimes.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    parser.add_argument("--ckpt", default="results/bighead_warmrestart/best_bighead_student.pt")
    parser.add_argument("--frames", default="16,32,64,128")
    parser.add_argument("--n_warmup", type=int, default=3)
    parser.add_argument("--n_runs", type=int, default=20)
    parser.add_argument("--no_siglip", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/speedup_regimes")
    args = parser.parse_args()
    main(args)
