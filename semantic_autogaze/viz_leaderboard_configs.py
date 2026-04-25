"""r/qual-leaderboard-viz cycle 1: AutoGaze patch-selection visualization for the
LEADERBOARD configs.

For each sample clip, render a 5-row image:

    Row 0: original video frames (t=0..T-1)
    Row 1: vanilla     gazing_ratio_tile=0.20 (K=39 tokens/frame, 39/196 = 19.9%)
    Row 2: rank 1      gazing_ratio_tile=0.14 (K=27 tokens/frame, 27/196 = 13.8%)
    Row 3: rank 2      gazing_ratio_tile=0.08 (K=16 tokens/frame, 16/196 = 8.2%)
    Row 4: legend / per-config K + retention rate

Each row shows the SAME frames with kept-patch overlay (kept = bright, dropped = dimmed).

Also measures filter-stage forward latency for each gazing_ratio
(should be near-identical since gazing_ratio doesn't change the AutoGaze forward —
it only changes top-K selection AFTER scoring).

Usage:
    CUDA_VISIBLE_DEVICES=0 \\
      python -m semantic_autogaze.viz_leaderboard_configs \\
        --device cuda:0 \\
        --output_dir results/qual_leaderboard_viz
"""
from __future__ import annotations
import os, sys, time, argparse, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange


SAMPLE_CLIPS = [
    {"path": "hlvid_videos/extracted_household/videos/clip_household_video_0_000.mp4",
     "label": "household_0_000", "category": "household"},
    {"path": "hlvid_videos/extracted_household/videos/clip_household_video_11_000.mp4",
     "label": "household_11_000", "category": "household"},
    {"path": "hlvid_videos/extracted/videos/clip_av_video_0_000.mp4",
     "label": "av_0_000", "category": "av"},
    {"path": "hlvid_videos/extracted/videos/clip_av_video_1_000.mp4",
     "label": "av_1_000", "category": "av"},
]

# AutoGaze gazing_ratio settings under test (tile only; thumbnail follows in
# practice but the visual is dominated by the tile branch).
CONFIGS = [
    {"name": "vanilla",       "tile_ratio": 0.20, "thumb_ratio": 0.75, "color": "#22c55e"},  # green
    {"name": "rank1_h_0.70",  "tile_ratio": 0.14, "thumb_ratio": 0.525, "color": "#ef4444"}, # red
    {"name": "rank2_av_0.40", "tile_ratio": 0.08, "thumb_ratio": 0.30,  "color": "#3b82f6"}, # blue
]

NUM_DISPLAY_FRAMES = 6   # how many tile frames to show per clip
# AutoGaze emits multi-scale tokens — sizes per side at scales [32, 64, 112, 224]:
#   scale 32 → 2×2 = 4 tokens, each "covers" 112×112 image pixels
#   scale 64 → 4×4 = 16 tokens, each covers 56×56
#   scale 112 → 7×7 = 49 tokens, each covers 32×32
#   scale 224 → 14×14 = 196 tokens, each covers 16×16
SCALE_GRIDS = [(2, 4), (4, 16), (7, 49), (14, 196)]  # (grid_side, n_tokens)
N_PER_FRAME = 265


def read_video_frames(path, num_frames=6, size=224):
    """Cheap uniform-sampled reader using cv2."""
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    indices = np.linspace(0, total - 1, num_frames).astype(int)
    out = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        out.append(frame)
    cap.release()
    if not out:
        return None
    return np.stack(out, axis=0).astype(np.uint8)  # (T, H, W, 3)


@torch.no_grad()
def run_autogaze(autogaze, video_thwc_uint8, max_gaze_tokens_each_frame, device):
    """Run AutoGaze and return gazing_pos as a (T, G*G) bool mask of kept patches.

    Args:
        autogaze: nn.Module loaded from nvidia/AutoGaze
        video_thwc_uint8: (T, H, W, C) numpy uint8
        max_gaze_tokens_each_frame: int K (top-K patches kept per frame)
        device: torch device

    Returns: (T, G*G) bool numpy mask of kept patch positions per frame.
    Latency: AutoGaze forward time in ms.
    """
    T = video_thwc_uint8.shape[0]
    # (T, H, W, C) → (1, T, C, H, W) float in [-1, 1]
    vid = torch.from_numpy(video_thwc_uint8).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    vid = vid.unsqueeze(0).to(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = autogaze.gazing_model.generate(vid, max_gaze_tokens_each_frame=int(max_gaze_tokens_each_frame))
    torch.cuda.synchronize()
    lat_ms = (time.perf_counter() - t0) * 1000.0

    gazing_pos = out["gazing_pos"]              # (1, N) ints, indexed into T*G*G
    if_padded = out["if_padded_gazing"]         # (1, N) bool
    num_per_frame = out["num_gazing_each_frame"]  # (1, T) ints

    gp = gazing_pos[0].cpu().numpy()
    pad = if_padded[0].cpu().numpy()

    # Decode each kept index into (frame, scale_idx, in_scale_idx).
    # Within a frame's 265 tokens: first 4 = scale 32 (2×2), next 16 = scale 64 (4×4),
    # next 49 = scale 112 (7×7), last 196 = scale 224 (14×14).
    kept = [[] for _ in range(T)]   # per-frame list of (scale_idx, gi, gj)
    counts_per_scale = [np.zeros(T, dtype=int) for _ in SCALE_GRIDS]
    n_kept_total = np.zeros(T, dtype=int)
    for n in range(gp.shape[0]):
        if pad[n]:
            continue
        global_idx = int(gp[n])
        if global_idx >= T * N_PER_FRAME:
            continue
        t = global_idx // N_PER_FRAME
        local = global_idx % N_PER_FRAME
        # bucket into a scale
        cum = 0
        for si, (gside, ntok) in enumerate(SCALE_GRIDS):
            if local < cum + ntok:
                in_scale = local - cum
                gi = in_scale // gside
                gj = in_scale % gside
                kept[t].append((si, gi, gj))
                counts_per_scale[si][t] += 1
                n_kept_total[t] += 1
                break
            cum += ntok
    return kept, lat_ms, n_kept_total, counts_per_scale


def render_clip(frames_thwc, kept_lists_by_config, kept_per_config, configs, save_path, title=""):
    """Render the multi-config kept-patch overlay across multiple AutoGaze scales.

    Each kept token (scale_idx, gi, gj) is drawn as a colored rectangle on the
    frame. Scale 32 patches are large (112×112 image px); scale 224 patches are
    tiny (16×16). The display dims non-overlapping pixels to make kept-region
    coverage easier to read.
    """
    from matplotlib.patches import Rectangle
    T = frames_thwc.shape[0]
    n_configs = len(configs)
    n_rows = 1 + n_configs
    fig, axes = plt.subplots(n_rows, T, figsize=(2.4 * T, 2.4 * n_rows + 0.7))
    if T == 1:
        axes = axes.reshape(n_rows, 1)

    H = frames_thwc.shape[1]
    W = frames_thwc.shape[2]

    for t in range(T):
        ax = axes[0, t]
        ax.imshow(frames_thwc[t])
        ax.set_title(f"t={t}", fontsize=8)
        ax.axis("off")
        if t == 0:
            ax.text(-0.10, 0.5, "frame\n(input)", transform=ax.transAxes,
                    rotation=90, va="center", ha="right", fontsize=10, fontweight="bold")

    for r, cfg in enumerate(configs):
        kept_lists = kept_lists_by_config[cfg["name"]]
        per_scale_counts = kept_per_config[cfg["name"]]["per_scale_counts"]
        n_total_per = kept_per_config[cfg["name"]]["total_per_frame"]
        for t in range(T):
            ax = axes[1 + r, t]
            f = frames_thwc[t].astype(np.float32) / 255.0
            # Build coverage mask at pixel level
            coverage = np.zeros((H, W), dtype=bool)
            for (si, gi, gj) in kept_lists[t]:
                gside, ntok = SCALE_GRIDS[si]
                cell_h = H / gside
                cell_w = W / gside
                y0 = int(round(gi * cell_h)); y1 = int(round((gi + 1) * cell_h))
                x0 = int(round(gj * cell_w)); x1 = int(round((gj + 1) * cell_w))
                coverage[y0:y1, x0:x1] = True
            display = np.where(coverage[..., None], f, f * 0.25)
            ax.imshow(display)
            # Draw rectangles, thinner for finer scales so they don't dominate
            for (si, gi, gj) in kept_lists[t]:
                gside, ntok = SCALE_GRIDS[si]
                cell_h = H / gside
                cell_w = W / gside
                lw = {0: 2.0, 1: 1.4, 2: 1.0, 3: 0.5}[si]
                ax.add_patch(Rectangle((gj * cell_w, gi * cell_h),
                                       cell_w, cell_h,
                                       fill=False, edgecolor=cfg["color"],
                                       linewidth=lw, alpha=0.9))
            ax.set_title(
                f"K={int(n_total_per[t])}  by scale [2:{int(per_scale_counts[0][t])}, "
                f"4:{int(per_scale_counts[1][t])}, 7:{int(per_scale_counts[2][t])}, "
                f"14:{int(per_scale_counts[3][t])}]",
                fontsize=6.5,
            )
            ax.axis("off")
            if t == 0:
                ax.text(-0.10, 0.5,
                        f"{cfg['name']}\nratio_tile={cfg['tile_ratio']}",
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=10, fontweight="bold",
                        color=cfg["color"])

    fig.suptitle(title, fontsize=10, y=0.998)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[setup] Loading AutoGaze from {args.autogaze_id} ...")
    from autogaze.models.autogaze import AutoGaze
    autogaze = AutoGaze.from_pretrained(args.autogaze_id, use_flash_attn=False).to(device).eval()
    gaze_model = autogaze.gazing_model  # AutoGazeModel sub-module
    print(f"[setup] AutoGaze ready. input_img_size={gaze_model.input_img_size}; "
          f"num_vision_tokens_each_frame={autogaze.num_vision_tokens_each_frame}")

    # Compute K = floor(num_vision_tokens_each_frame * tile_ratio)
    N_per_frame = int(autogaze.num_vision_tokens_each_frame)
    for cfg in CONFIGS:
        cfg["K"] = max(1, int(np.floor(N_per_frame * cfg["tile_ratio"])))

    print(f"[setup] N_per_frame={N_per_frame}; K per config: " +
          ", ".join(f"{c['name']}={c['K']}" for c in CONFIGS))

    summary = {"clips": [], "configs": CONFIGS, "N_per_frame": N_per_frame}
    latencies_ms = {c["name"]: [] for c in CONFIGS}

    for clip in SAMPLE_CLIPS:
        if not os.path.exists(clip["path"]):
            print(f"[skip] missing {clip['path']}")
            continue
        print(f"[run] {clip['label']}")
        frames = read_video_frames(clip["path"], num_frames=NUM_DISPLAY_FRAMES, size=224)
        kept_lists_by_config = {}
        kept_per_config = {}
        for cfg in CONFIGS:
            # warmup
            _ = run_autogaze(autogaze, frames, cfg["K"], device)
            kept_lists, lat_ms, n_total, n_per_scale = run_autogaze(autogaze, frames, cfg["K"], device)
            kept_lists_by_config[cfg["name"]] = kept_lists
            kept_per_config[cfg["name"]] = {"total_per_frame": n_total.tolist(),
                                            "per_scale_counts": [s.tolist() for s in n_per_scale]}
            latencies_ms[cfg["name"]].append(lat_ms)

        save_path = os.path.join(args.output_dir, f"viz_{clip['label']}.png")
        title = (f"{clip['label']} ({clip['category']}) — AutoGaze multi-scale patch retention vs gazing_ratio_tile  "
                 f"|  scales [2×2, 4×4, 7×7, 14×14] = [4, 16, 49, 196] tokens (total 265/frame)")
        render_clip(frames, kept_lists_by_config, kept_per_config, CONFIGS, save_path, title=title)
        print(f"  saved → {save_path}")
        summary["clips"].append({"label": clip["label"], "path": clip["path"],
                                 "save_path": save_path,
                                 "category": clip["category"],
                                 "kept_per_config": kept_per_config})

    print("\n[latency] AutoGaze forward time per gazing_ratio (mean ± std over clips, ms):")
    for cfg in CONFIGS:
        lats = latencies_ms[cfg["name"]]
        m = float(np.mean(lats)); s = float(np.std(lats))
        cfg["latency_ms_mean"] = m
        cfg["latency_ms_std"] = s
        cfg["latency_ms_per_clip"] = lats
        print(f"  {cfg['name']:<20s}  K={cfg['K']:>3d}  {m:6.2f} ± {s:5.2f} ms  (n={len(lats)})")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] wrote {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--autogaze_id", default="nvidia/AutoGaze")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/qual_leaderboard_viz")
    args = p.parse_args()
    main(args)
