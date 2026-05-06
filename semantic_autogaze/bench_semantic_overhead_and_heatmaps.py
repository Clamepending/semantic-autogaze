"""
Answers (with real numbers, no synthetic baselines):

  1) Does semantic filtering give a speedup vs NORMAL (adaptive) AutoGaze?
     -> End-to-end wall-clock on real videos, per-video + averaged.

  2) Does accuracy degrade when the semantic vector is *accurate*?
     -> Cosine similarity between SigLIP pooled features from full AutoGaze
        and SigLIP pooled features after semantic filtering, with a correct
        query ("people") vs a wrong query ("hanging clock"). A good filter
        preserves pooled features (~1.0) on correct query; a blind one degrades.

  3) How much latency does the semantic head ITSELF add?
     -> Isolated head forward time + param count, plus head-induced overhead
        in the full pipeline (hidden-state extraction included).

  4) Qualitative heatmaps: for validation videos, plot per-frame heatmaps of
     the semantic score for query "people", and overlay the Intersect-10% and
     Intersect-50% selection masks.

Outputs:
    results/semantic_overhead/overhead.json          - all numbers
    results/semantic_overhead/overhead_summary.png   - summary bar chart
    results/semantic_overhead/heatmaps/<video>.png   - qualitative overlay per video

Usage:
    CUDA_VISIBLE_DEVICES=5 python3 -m semantic_autogaze.bench_semantic_overhead_and_heatmaps \
        --device cuda:0 --n_videos 12 --query people --wrong_query "hanging clock"
"""

import os
import glob
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import open_clip
import av
from einops import rearrange

from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper


T_FRAMES = 16
GRID = 14
N_PATCHES = GRID * GRID  # 196
TOTAL = T_FRAMES * N_PATCHES  # 3136


def cuda_time(fn, n_warmup=2, n_runs=5):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record(); fn(); t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))
    return float(np.mean(times)), float(np.std(times))


def clip_text_embed(text, clip_model, tokenizer, device):
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        f = clip_model.encode_text(tokens)
        f = F.normalize(f, dim=-1)
    return f


def pooled_siglip_feature(siglip_model, video_sg, gazing_info=None):
    """Return a single vector per video: mean over non-padded token features."""
    with torch.no_grad():
        out = siglip_model(video_sg, gazing_info=gazing_info)
    hidden = out.last_hidden_state  # (B, K, D)
    if gazing_info is not None and "if_padded_gazing" in gazing_info:
        mask = ~gazing_info["if_padded_gazing"]
        mask = mask[:, :hidden.shape[1]].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        pooled = hidden.mean(dim=1)
    return F.normalize(pooled, dim=-1)


def param_count(module):
    return sum(p.numel() for p in module.parameters())


def main(args):
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    random.seed(42); np.random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    heat_dir = os.path.join(args.output_dir, "heatmaps")
    os.makedirs(heat_dir, exist_ok=True)

    print("[setup] Loading AutoGaze + head + SigLIP + CLIP...")
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )
    autogaze_transform = AutoGazeImageProcessor.from_pretrained(args.autogaze_model)

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()

    from autogaze.vision_encoders.siglip import SiglipVisionModel
    siglip_model = SiglipVisionModel.from_pretrained(
        "google/siglip2-base-patch16-224",
        scales=wrapper.autogaze.config.scales,
        attn_implementation="sdpa",
    ).to(device).eval()
    from transformers import AutoImageProcessor
    siglip_tf = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")

    # =========================================================
    # (A) HEAD-IN-ISOLATION: params, FLOPs proxy, raw latency
    # =========================================================
    head = wrapper.semantic_filter.head
    head_params = param_count(head)
    autogaze_params = param_count(wrapper.autogaze)
    print(f"[head] BigSimilarityHead params: {head_params/1e6:.3f}M  "
          f"(AutoGaze: {autogaze_params/1e6:.1f}M)")

    # Isolated head forward time on typical hidden-state input
    dummy_hidden = torch.randn(1, TOTAL, wrapper.autogaze.config.gaze_model_config.gaze_decoder_config.hidden_size, device=device)
    dummy_query = torch.randn(1, 512, device=device)
    head_ms, head_std = cuda_time(lambda: head(dummy_hidden, dummy_query))
    print(f"[head] isolated forward (T=16): {head_ms:.2f} ± {head_std:.2f} ms")

    # =========================================================
    # (B) Real videos: per-video latency + feature fidelity
    # =========================================================
    videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    random.shuffle(videos)
    videos = videos[: args.n_videos]
    print(f"[eval] n_videos = {len(videos)}")

    q_correct = clip_text_embed(args.query, clip_model, clip_tok, device)
    q_wrong = clip_text_embed(args.wrong_query, clip_model, clip_tok, device)

    per_video = []
    heatmap_saved = 0

    for vi, vp in enumerate(videos):
        name = os.path.basename(vp)
        try:
            c = av.open(vp)
            raw = read_video_pyav(container=c, indices=list(range(T_FRAMES)))
            c.close()
            if raw.shape[0] < T_FRAMES:
                continue
            v_ag = transform_video_for_pytorch(raw, autogaze_transform)[None].to(device)
            v_sg = transform_video_for_pytorch(raw, siglip_tf)[None].to(device)
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue

        row = {"video": name}

        # -- AutoGaze ADAPTIVE baseline (normal AutoGaze: encoder+decoder+SigLIP@kept)
        def run_gaze():
            return wrapper.forward(
                v_ag, q_correct, mode="gaze_only",
                gazing_ratio=args.gazing_ratio,
                task_loss_requirement=args.task_loss_req,
                semantic_keep_ratio=1.0,
            )
        info_gaze = run_gaze()
        n_gaze = int((~info_gaze["if_padded_gazing"]).sum().item())
        row["tokens_autogaze"] = n_gaze
        row["tokens_total"] = TOTAL

        t_ag_fwd, _ = cuda_time(run_gaze)
        t_sg_baseline, _ = cuda_time(lambda: siglip_model(v_sg, gazing_info=info_gaze))
        row["lat_autogaze_fwd_ms"] = t_ag_fwd
        row["lat_siglip_baseline_ms"] = t_sg_baseline
        row["lat_baseline_total_ms"] = t_ag_fwd + t_sg_baseline

        # Pooled feature of the baseline (this is our "accuracy reference")
        feat_baseline = pooled_siglip_feature(siglip_model, v_sg, info_gaze)

        # -- Intersect 50% with correct query
        def run_int(kr, q):
            return wrapper.forward(
                v_ag, q, mode="intersect",
                gazing_ratio=args.gazing_ratio,
                task_loss_requirement=args.task_loss_req,
                semantic_keep_ratio=kr,
            )

        info_i50 = run_int(0.5, q_correct)
        n_i50 = int((~info_i50["if_padded_gazing"]).sum().item())
        t_i50_fwd, _ = cuda_time(lambda: run_int(0.5, q_correct))
        t_sg_i50, _ = cuda_time(lambda: siglip_model(v_sg, gazing_info=info_i50))
        row["tokens_intersect50"] = n_i50
        row["lat_intersect50_fwd_ms"] = t_i50_fwd
        row["lat_siglip_i50_ms"] = t_sg_i50
        row["lat_i50_total_ms"] = t_i50_fwd + t_sg_i50
        # The overhead of adding the head is (t_i50_fwd - t_ag_fwd): extra
        # hidden-state extraction + head forward (intersect path doesn't re-run
        # decoder separately; it shares cost with gaze forward).
        row["head_overhead_ms"] = t_i50_fwd - t_ag_fwd

        feat_i50 = pooled_siglip_feature(siglip_model, v_sg, info_i50)
        row["feat_cos_i50_vs_baseline"] = float(F.cosine_similarity(feat_i50, feat_baseline).item())

        # -- Intersect 10% with correct query
        info_i10 = run_int(0.1, q_correct)
        n_i10 = int((~info_i10["if_padded_gazing"]).sum().item())
        t_i10_fwd, _ = cuda_time(lambda: run_int(0.1, q_correct))
        t_sg_i10, _ = cuda_time(lambda: siglip_model(v_sg, gazing_info=info_i10))
        row["tokens_intersect10"] = n_i10
        row["lat_intersect10_fwd_ms"] = t_i10_fwd
        row["lat_siglip_i10_ms"] = t_sg_i10
        row["lat_i10_total_ms"] = t_i10_fwd + t_sg_i10
        feat_i10 = pooled_siglip_feature(siglip_model, v_sg, info_i10)
        row["feat_cos_i10_vs_baseline"] = float(F.cosine_similarity(feat_i10, feat_baseline).item())

        # -- Intersect 50% with WRONG query (accuracy sanity check)
        try:
            info_wrong = run_int(0.5, q_wrong)
            feat_wrong = pooled_siglip_feature(siglip_model, v_sg, info_wrong)
            row["feat_cos_wrongquery_i50_vs_baseline"] = float(F.cosine_similarity(feat_wrong, feat_baseline).item())
        except Exception as e:
            print(f"    wrong-query path failed for {name}: {e}")

        per_video.append(row)

        # Heatmap for first few videos.
        # Important: we avoid any indexing with AutoGaze's `gazing_pos` (which
        # can live in a wider token space than T*196), and derive overlay
        # masks from semantic scores directly.  The Intersect-10% overlay uses
        # a global top-10% threshold on scores; Intersect-50% uses top-50%.
        if heatmap_saved < args.n_heatmaps:
            try:
                hidden = wrapper.extract_hidden_states(v_ag)
                scores = wrapper.semantic_filter.get_scores(hidden, q_correct)  # (1, 3136)
                scores_np = scores[0].reshape(T_FRAMES, GRID, GRID).cpu().numpy()

                frames_np = v_sg[0].cpu().numpy()  # (T, 3, 224, 224) normalized
                frames_disp = frames_np.transpose(0, 2, 3, 1)
                frames_disp = (frames_disp - frames_disp.min()) / (frames_disp.max() - frames_disp.min() + 1e-9)

                flat = scores_np.flatten()
                thr50 = float(np.sort(flat)[-max(1, int(0.5 * len(flat)))])
                thr10 = float(np.sort(flat)[-max(1, int(0.1 * len(flat)))])
                m_i50 = (scores_np >= thr50).astype(float)
                m_i10 = (scores_np >= thr10).astype(float)

                fig, axes = plt.subplots(4, T_FRAMES, figsize=(T_FRAMES * 1.1, 4.6))
                extent = [0, frames_disp.shape[2], frames_disp.shape[1], 0]
                for t in range(T_FRAMES):
                    axes[0, t].imshow(frames_disp[t]); axes[0, t].set_xticks([]); axes[0, t].set_yticks([])
                    if t == 0: axes[0, t].set_ylabel("frame", fontsize=7)

                    axes[1, t].imshow(frames_disp[t])
                    axes[1, t].imshow(scores_np[t], cmap="jet", alpha=0.55, extent=extent, vmin=0, vmax=flat.max())
                    axes[1, t].set_xticks([]); axes[1, t].set_yticks([])
                    if t == 0: axes[1, t].set_ylabel(f'"{args.query}"\nheatmap', fontsize=7)

                    axes[2, t].imshow(frames_disp[t])
                    axes[2, t].imshow(m_i50[t], cmap="Greens", alpha=0.6, extent=extent, vmin=0, vmax=1)
                    axes[2, t].set_xticks([]); axes[2, t].set_yticks([])
                    if t == 0: axes[2, t].set_ylabel(f"top-50%\n({int(m_i50.sum())})", fontsize=7)

                    axes[3, t].imshow(frames_disp[t])
                    axes[3, t].imshow(m_i10[t], cmap="Reds", alpha=0.7, extent=extent, vmin=0, vmax=1)
                    axes[3, t].set_xticks([]); axes[3, t].set_yticks([])
                    if t == 0: axes[3, t].set_ylabel(f"top-10%\n({int(m_i10.sum())})", fontsize=7)

                fig.suptitle(
                    f'{name}   query="{args.query}"   '
                    f'AG keep={n_gaze}/{TOTAL}={100*n_gaze/TOTAL:.0f}%   '
                    f'cos(i50↔baseline)={row["feat_cos_i50_vs_baseline"]:.3f}',
                    fontsize=9,
                )
                fig.tight_layout()
                safe = name.replace(".mp4", "").replace("/", "_")
                out = os.path.join(heat_dir, f"{safe}.png")
                fig.savefig(out, dpi=120, bbox_inches="tight")
                plt.close(fig)
                heatmap_saved += 1
                print(f"  [{vi+1}/{len(videos)}] {name[:30]:<30} heatmap -> {out}")
            except Exception as e:
                print(f"  heatmap failed for {name}: {e}")
        else:
            print(f"  [{vi+1}/{len(videos)}] {name[:30]:<30} "
                  f"AG={n_gaze}({100*n_gaze/TOTAL:.0f}%) "
                  f"cos50={row['feat_cos_i50_vs_baseline']:.3f} "
                  f"cos10={row['feat_cos_i10_vs_baseline']:.3f} "
                  f"baseline={row['lat_baseline_total_ms']:.0f}ms "
                  f"i50={row['lat_i50_total_ms']:.0f}ms "
                  f"({row['lat_baseline_total_ms']/row['lat_i50_total_ms']:.2f}x)")

    if not per_video:
        print("No videos processed.")
        return

    # =========================================================
    # Aggregate
    # =========================================================
    def agg(key):
        vals = [r[key] for r in per_video if key in r]
        return float(np.mean(vals)), float(np.std(vals)), float(np.median(vals))

    summary = {
        "head": {
            "params_M": head_params / 1e6,
            "autogaze_params_M": autogaze_params / 1e6,
            "params_ratio_pct": 100 * head_params / autogaze_params,
            "isolated_forward_ms_T16": head_ms,
            "isolated_forward_ms_T16_std": head_std,
        },
        "latency_ms": {
            "autogaze_fwd": agg("lat_autogaze_fwd_ms"),
            "siglip_baseline": agg("lat_siglip_baseline_ms"),
            "baseline_total": agg("lat_baseline_total_ms"),
            "intersect50_fwd": agg("lat_intersect50_fwd_ms"),
            "siglip_i50": agg("lat_siglip_i50_ms"),
            "intersect50_total": agg("lat_i50_total_ms"),
            "intersect10_total": agg("lat_i10_total_ms"),
            "head_added_overhead": agg("head_overhead_ms"),
        },
        "tokens": {
            "autogaze": agg("tokens_autogaze"),
            "intersect50": agg("tokens_intersect50"),
            "intersect10": agg("tokens_intersect10"),
        },
        "accuracy_proxy_cosine": {
            "i50_vs_baseline": agg("feat_cos_i50_vs_baseline"),
            "i10_vs_baseline": agg("feat_cos_i10_vs_baseline"),
            "wrongquery_i50_vs_baseline": agg("feat_cos_wrongquery_i50_vs_baseline"),
        },
        "speedup_vs_autogaze": {
            "intersect50_mean": float(np.mean([r["lat_baseline_total_ms"] / max(r["lat_i50_total_ms"], 1e-9) for r in per_video])),
            "intersect10_mean": float(np.mean([r["lat_baseline_total_ms"] / max(r["lat_i10_total_ms"], 1e-9) for r in per_video])),
        },
        "query_correct": args.query,
        "query_wrong": args.wrong_query,
        "n_videos": len(per_video),
    }

    print("\n" + "=" * 80)
    print(f'SUMMARY  (query="{args.query}" vs wrong="{args.wrong_query}",  n={len(per_video)} videos)')
    print("=" * 80)
    print(f'Head params:                 {head_params/1e6:.3f}M  '
          f'({100*head_params/autogaze_params:.1f}% of AutoGaze {autogaze_params/1e6:.1f}M)')
    print(f'Head isolated forward (T16): {head_ms:.2f} ms')
    print(f'Latency (mean ± std ms):')
    for label, key in [
        ("  AutoGaze fwd",       "lat_autogaze_fwd_ms"),
        ("  SigLIP @ baseline",  "lat_siglip_baseline_ms"),
        ("  TOTAL baseline",     "lat_baseline_total_ms"),
        ("  AutoGaze+head fwd",  "lat_intersect50_fwd_ms"),
        ("  SigLIP @ I50",       "lat_siglip_i50_ms"),
        ("  TOTAL I50",          "lat_i50_total_ms"),
        ("  TOTAL I10",          "lat_i10_total_ms"),
        ("  head overhead (fwd diff)", "head_overhead_ms"),
    ]:
        m, s, med = agg(key)
        print(f"{label:<34}{m:7.1f} ± {s:6.1f}   (median {med:6.1f})")
    print(f'Mean speedup vs AutoGaze: I50={summary["speedup_vs_autogaze"]["intersect50_mean"]:.2f}x  '
          f'I10={summary["speedup_vs_autogaze"]["intersect10_mean"]:.2f}x')
    print(f'Accuracy (pooled SigLIP feature cosine vs full AutoGaze):')
    for label, key in [
        ("  Intersect50 (correct query)", "feat_cos_i50_vs_baseline"),
        ("  Intersect10 (correct query)", "feat_cos_i10_vs_baseline"),
        ("  Intersect50 (WRONG query)",    "feat_cos_wrongquery_i50_vs_baseline"),
    ]:
        vals = [r[key] for r in per_video if key in r]
        if vals:
            print(f"{label:<34}{np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Save
    with open(os.path.join(args.output_dir, "overhead.json"), "w") as f:
        json.dump({"summary": summary, "per_video": per_video}, f, indent=2)
    print(f'\n[save] {args.output_dir}/overhead.json')

    # =========================================================
    # Summary bar chart (3 panels)
    # =========================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: latency breakdown
    ax = axes[0]
    bars = [
        ("AutoGaze\n(baseline)", agg("lat_autogaze_fwd_ms")[0], agg("lat_siglip_baseline_ms")[0], "tab:gray"),
        ("+ head\n(Intersect 50%)", agg("lat_intersect50_fwd_ms")[0], agg("lat_siglip_i50_ms")[0], "tab:blue"),
        ("+ head\n(Intersect 10%)", agg("lat_intersect10_fwd_ms")[0] if "lat_intersect10_fwd_ms" in per_video[0] else agg("lat_intersect50_fwd_ms")[0], agg("lat_siglip_i10_ms")[0], "tab:green"),
    ]
    labels = [b[0] for b in bars]
    fwds = [b[1] for b in bars]
    siglips = [b[2] for b in bars]
    x = np.arange(len(labels))
    ax.bar(x, fwds, color="#555", label="AutoGaze fwd (+head)")
    ax.bar(x, siglips, bottom=fwds, color=[b[3] for b in bars], label="SigLIP")
    for i, (f_, s_) in enumerate(zip(fwds, siglips)):
        ax.text(i, f_ + s_ + 5, f"{f_ + s_:.0f}ms", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("End-to-end latency (mean across videos)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: feature cosine
    ax = axes[1]
    cats = [
        ("Intersect 50%\n(correct)",      "feat_cos_i50_vs_baseline", "tab:blue"),
        ("Intersect 10%\n(correct)",      "feat_cos_i10_vs_baseline", "tab:green"),
        ("Intersect 50%\n(WRONG query)",  "feat_cos_wrongquery_i50_vs_baseline", "tab:red"),
    ]
    xs = np.arange(len(cats))
    means = [np.mean([r[k] for r in per_video if k in r]) for _, k, _ in cats]
    stds = [np.std([r[k] for r in per_video if k in r]) for _, k, _ in cats]
    ax.bar(xs, means, yerr=stds, color=[c for _, _, c in cats], capsize=4)
    ax.axhline(1.0, color="black", ls=":", lw=1, label="full AutoGaze = 1.0")
    ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in cats], fontsize=9)
    ax.set_ylabel("Cosine similarity vs baseline SigLIP feature")
    ax.set_title("Accuracy proxy: does filtering preserve features?")
    ax.set_ylim([min(0.5, min(means) - 0.1), 1.02])
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: tokens kept
    ax = axes[2]
    cats = [
        ("Full\n(T×196)",            TOTAL, "lightgray"),
        ("AutoGaze\n(adaptive)",    agg("tokens_autogaze")[0], "tab:gray"),
        ("Intersect 50%",           agg("tokens_intersect50")[0], "tab:blue"),
        ("Intersect 10%",           agg("tokens_intersect10")[0], "tab:green"),
    ]
    xs = np.arange(len(cats))
    ys = [c[1] for c in cats]
    ax.bar(xs, ys, color=[c[2] for c in cats])
    for i, y in enumerate(ys):
        ax.text(i, y + 40, f"{y:.0f}", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in cats], fontsize=9)
    ax.set_ylabel("SigLIP input tokens")
    ax.set_title("Tokens fed to SigLIP (mean)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f'Semantic AutoGaze overhead & accuracy — query="{args.query}" (n={len(per_video)})',
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.output_dir, "overhead_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", default="data")
    p.add_argument("--n_videos", type=int, default=12)
    p.add_argument("--n_heatmaps", type=int, default=6)
    p.add_argument("--query", default="people")
    p.add_argument("--wrong_query", default="hanging clock")
    p.add_argument("--gazing_ratio", type=float, default=0.75)
    p.add_argument("--task_loss_req", type=float, default=0.7)
    p.add_argument("--ckpt", default="results/bighead_warmrestart/best_bighead_student.pt")
    p.add_argument("--head_type", default="bighead")
    p.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/semantic_overhead")
    args = p.parse_args()
    main(args)
