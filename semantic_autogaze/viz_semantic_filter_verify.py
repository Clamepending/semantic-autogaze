"""r/qual-semantic-filter-verify: visualize + benchmark the text-conditioned
semantic filter on top of AutoGaze using ACTUAL HLVid questions as the text query.

Goal — answer 3 specific questions from the user:

  Q1. Is the latency overhead of the semantic-filter forward NEGLIGIBLE
      compared to the AutoGaze hidden-state extraction?
      → measure both (filter ms + AutoGaze ms) over n>=3 clips, report ratio.

  Q2. Is accuracy preserved when the filter is added on top of vanilla AutoGaze?
      → reference existing accuracy data (r/autogaze-aware-filter-train,
        r/filter-vs-random-baseline, r/predecoder-bighead-full-stack) and
        run a small same-sample replay here.

  Q3. Does the filter actually focus on patches relevant to the question?
      → render heatmaps + kept-patch overlays at keep_ratio in
        {1.0, 0.75, 0.50, 0.25}, using the actual HLVid question stem as text.

Outputs:
  results/qual_semantic_filter_verify/
    viz_<clip>.png     # 5 rows: input | gaze-only | gaze+filter@0.75 | @0.50 | @0.25
    summary.json       # {clips, latencies_ms, accuracy_refs, kept_per_keep}
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
     "label": "household_0_000", "category": "household",
     "qid": 8,
     "question": "What does the white text say on the green background of the pack on the top of the shelf?"},
    {"path": "hlvid_videos/extracted_household/videos/clip_household_video_11_000.mp4",
     "label": "household_11_000", "category": "household",
     "qid": 177,
     "question": "What does the black text on the book say?"},
    {"path": "hlvid_videos/extracted/videos/clip_av_video_0_000.mp4",
     "label": "av_0_000", "category": "av",
     "qid": 79,
     "question": "What are the two texts written on the yellow and green signs?"},
    {"path": "hlvid_videos/extracted/videos/clip_av_video_1_000.mp4",
     "label": "av_1_000", "category": "av",
     "qid": 232,
     "question": "What is the number on the red sign on the brown building?"},
]

KEEP_RATIOS = [1.00, 0.75, 0.50, 0.25]
KEEP_COLORS = {1.00: "#22c55e", 0.75: "#f59e0b", 0.50: "#ef4444", 0.25: "#3b82f6"}

NUM_DISPLAY_FRAMES = 6
GRID = 14
N_PATCHES = GRID * GRID  # 196


def read_video_frames(path, num_frames=6, size=224):
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
    return np.stack(out, axis=0).astype(np.uint8)


def render_clip(frames_thwc, scores_per_frame, kept_masks_per_keep, kept_count_per_keep,
                question, save_path, title=""):
    """5-row grid: input frames + 4 keep_ratio rows. Each keep row shows kept patches
    as colored 14×14 boxes, dropped pixels dimmed, plus a heatmap of filter scores."""
    from matplotlib.patches import Rectangle
    T = frames_thwc.shape[0]
    n_rows = 1 + len(KEEP_RATIOS)  # input + per keep_ratio
    fig, axes = plt.subplots(n_rows, T, figsize=(2.4 * T, 2.4 * n_rows + 1.0))
    if T == 1:
        axes = axes.reshape(n_rows, 1)
    H, W = frames_thwc.shape[1:3]

    # Top row: input frames
    for t in range(T):
        ax = axes[0, t]
        ax.imshow(frames_thwc[t])
        ax.set_title(f"t={t}", fontsize=8)
        ax.axis("off")
        if t == 0:
            ax.text(-0.10, 0.5, "frame\n(input)", transform=ax.transAxes,
                    rotation=90, va="center", ha="right", fontsize=10, fontweight="bold")

    # Score normalization for heatmap (per-clip, across all frames + patches)
    smin, smax = float(scores_per_frame.min()), float(scores_per_frame.max())
    if smax - smin < 1e-6:
        smax = smin + 1.0

    for r, kr in enumerate(KEEP_RATIOS):
        kept = kept_masks_per_keep[kr]            # (T, 14, 14) bool
        kept_count = kept_count_per_keep[kr]      # (T,) ints
        col = KEEP_COLORS[kr]
        for t in range(T):
            ax = axes[1 + r, t]
            f = frames_thwc[t].astype(np.float32) / 255.0
            # Coverage mask for dim-non-kept
            up = np.repeat(np.repeat(kept[t], H // GRID, axis=0), W // GRID, axis=1)
            up = up[:H, :W]
            display = np.where(up[..., None], f, f * 0.25)
            # Score heatmap overlay (faint, normalized to clip range)
            scores = scores_per_frame[t]
            sup = np.repeat(np.repeat(scores, H // GRID, axis=0), W // GRID, axis=1)
            sup = sup[:H, :W]
            sup_norm = (sup - smin) / (smax - smin)
            ax.imshow(display)
            ax.imshow(sup_norm, alpha=0.25, cmap="hot", vmin=0, vmax=1)
            cell_h = H / GRID; cell_w = W / GRID
            for gi in range(GRID):
                for gj in range(GRID):
                    if kept[t, gi, gj]:
                        ax.add_patch(Rectangle((gj * cell_w, gi * cell_h),
                                               cell_w, cell_h,
                                               fill=False, edgecolor=col,
                                               linewidth=0.8, alpha=0.95))
            ax.set_title(f"K_filter={int(kept_count[t])}/196", fontsize=8)
            ax.axis("off")
            if t == 0:
                ax.text(-0.10, 0.5,
                        f"keep_ratio={kr:.2f}",
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=10, fontweight="bold",
                        color=col)

    fig.suptitle(f"{title}\nquestion: \"{question[:140]}\"",
                 fontsize=10, y=0.998)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[setup] Loading SemanticAutoGazeWrapper (BigHead {args.head_ckpt})...")
    from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_id,
        head_ckpt=args.head_ckpt,
        head_type="bighead",
        device=str(device),
        grid_size=GRID,
        num_frames=NUM_DISPLAY_FRAMES,
    )
    wrapper.eval()

    print("[setup] Loading CLIP ViT-B/16 (text encoder for question embedding)...")
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    clip_tok = open_clip.get_tokenizer("ViT-B-16")

    summary = {"clips": [], "keep_ratios": KEEP_RATIOS}
    autogaze_lat_ms = []
    filter_lat_ms = []

    for clip in SAMPLE_CLIPS:
        if not os.path.exists(clip["path"]):
            print(f"[skip] missing {clip['path']}")
            continue
        print(f"[run] {clip['label']}  qid={clip['qid']}  q=\"{clip['question'][:60]}...\"")
        frames = read_video_frames(clip["path"], num_frames=NUM_DISPLAY_FRAMES, size=224)
        if frames is None:
            print(f"  [warn] could not read frames")
            continue
        T = frames.shape[0]

        # Preprocess for AutoGaze: (1, T, C, H, W), in [-1, 1]
        vid = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        vid = vid.unsqueeze(0).to(device)

        # CLIP text embedding for the question stem
        with torch.no_grad():
            tok = clip_tok([clip["question"]]).to(device)
            text_emb = clip_model.encode_text(tok)
            text_emb = F.normalize(text_emb, dim=-1)  # (1, 512)

        # ----- Latency: AutoGaze hidden-state extraction (the "pre-ViT" cost) -----
        torch.cuda.synchronize()
        # warmup
        _ = wrapper.extract_hidden_states(vid)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hidden = wrapper.extract_hidden_states(vid)
        torch.cuda.synchronize()
        ag_ms = (time.perf_counter() - t0) * 1000.0

        # ----- Latency: SemanticFilter forward (DELTA over AutoGaze) -----
        sf = wrapper.semantic_filter
        torch.cuda.synchronize()
        _ = sf.get_scores(hidden, text_emb)  # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        scores = sf.get_scores(hidden, text_emb)  # (1, T*196)
        torch.cuda.synchronize()
        ft_ms = (time.perf_counter() - t0) * 1000.0

        autogaze_lat_ms.append(ag_ms)
        filter_lat_ms.append(ft_ms)
        print(f"  AutoGaze hidden extract = {ag_ms:6.2f} ms")
        print(f"  SemanticFilter scores   = {ft_ms:6.2f} ms  (delta over pre-ViT)")
        print(f"  ratio                   = {ft_ms / ag_ms * 100:.2f}% of AutoGaze")

        # Decode scores to (T, 14, 14)
        scores_np = scores.reshape(1, T, GRID, GRID).squeeze(0).cpu().numpy()

        # For each keep_ratio, select top-K patches PER FRAME (independent ranking)
        kept_masks = {}
        kept_counts = {}
        for kr in KEEP_RATIOS:
            K = max(1, int(round(N_PATCHES * kr)))
            mask = np.zeros((T, GRID, GRID), dtype=bool)
            counts = np.zeros(T, dtype=int)
            for t in range(T):
                flat = scores_np[t].flatten()
                if K >= N_PATCHES:
                    idx = np.arange(N_PATCHES)
                else:
                    idx = np.argpartition(-flat, K)[:K]
                mask_flat = mask[t].flatten()
                mask_flat[idx] = True
                mask[t] = mask_flat.reshape(GRID, GRID)
                counts[t] = mask[t].sum()
            kept_masks[kr] = mask
            kept_counts[kr] = counts

        save_path = os.path.join(args.output_dir, f"viz_filter_{clip['label']}.png")
        title = (f"{clip['label']} ({clip['category']}) — semantic-filter top-K patches "
                 f"on the 14×14 grid as keep_ratio drops")
        render_clip(frames, scores_np, kept_masks, kept_counts,
                    clip["question"], save_path, title=title)
        print(f"  saved → {save_path}")

        summary["clips"].append({
            "label": clip["label"],
            "qid": clip["qid"],
            "question": clip["question"],
            "category": clip["category"],
            "save_path": save_path,
            "autogaze_lat_ms": ag_ms,
            "filter_lat_ms": ft_ms,
            "filter_pct_of_autogaze": ft_ms / ag_ms * 100,
            "kept_per_keep_ratio": {f"{kr:.2f}": kept_counts[kr].tolist()
                                    for kr in KEEP_RATIOS},
        })

    summary["aggregate"] = {
        "autogaze_lat_ms_mean": float(np.mean(autogaze_lat_ms)),
        "autogaze_lat_ms_std": float(np.std(autogaze_lat_ms)),
        "filter_lat_ms_mean": float(np.mean(filter_lat_ms)),
        "filter_lat_ms_std": float(np.std(filter_lat_ms)),
        "filter_pct_of_autogaze_mean": float(np.mean(filter_lat_ms) / np.mean(autogaze_lat_ms) * 100),
    }

    print("\n[summary] Aggregate latency over {} clips:".format(len(autogaze_lat_ms)))
    print(f"  AutoGaze hidden extract: {summary['aggregate']['autogaze_lat_ms_mean']:.2f} ± "
          f"{summary['aggregate']['autogaze_lat_ms_std']:.2f} ms")
    print(f"  SemanticFilter forward:  {summary['aggregate']['filter_lat_ms_mean']:.2f} ± "
          f"{summary['aggregate']['filter_lat_ms_std']:.2f} ms")
    print(f"  filter / AutoGaze:       {summary['aggregate']['filter_pct_of_autogaze_mean']:.2f}%")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] wrote {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--autogaze_id", default="nvidia/AutoGaze")
    p.add_argument("--head_ckpt", default="results/bighead/best_bighead.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/qual_semantic_filter_verify")
    args = p.parse_args()
    main(args)
