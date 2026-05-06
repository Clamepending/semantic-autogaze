"""
Heatmap crispness investigation — what causes fuzzy predictions?

Three candidate causes for our head's heatmaps looking soft:
  (1) CLIPSeg teacher itself is fuzzy for certain object classes.
  (2) Our CLIP ViT-B-16 text embedding is not as discriminative as CLIPSeg's
      dedicated text encoder.
  (3) Our BigSimilarityHead architecture is the bottleneck.

This script separates these factors by visualizing, for a set of validation
samples, three things side-by-side per frame:

  Row A:  Raw video frame (8 frames).
  Row B:  CLIPSeg TEACHER heatmap (sigmoid of cached target_scores).
          — The upper bound: if this is fuzzy, cause (1) dominates.
  Row C:  OUR HEAD STUDENT heatmap (sigmoid of head(hidden, q)).
          — If close to B, we are not losing information in the head.
          — If much worse than B, cause (3) dominates.
  Row D:  Diff = |student - teacher|  (white = disagreement).

We also pick queries where the teacher clearly localized something
(max teacher sigmoid > 0.6) so we're measuring crispness, not signal absence.

Quantitatively, we report, per sample:
  - teacher_crispness     = (max-mean)/max of sigmoid(target)
  - student_crispness     = same for sigmoid(head output)
  - teacher-student MSE
  - teacher-student correlation

Aggregated across samples this tells us:
  - If teacher_crispness is LOW across the board, the teacher is fuzzy →
    we need better supervision (segmentation datasets, or CLIPSeg+).
  - If teacher_crispness is HIGH but student_crispness is LOW, the head is
    the bottleneck → bigger/better head.
  - If both are fine, we look elsewhere (text encoder, resolution).

Usage:
  CUDA_VISIBLE_DEVICES=5 python3 -m semantic_autogaze.crispness_investigation \\
    --device cuda:0 --n_samples 12 --output_dir results/crispness
"""

import os, glob, json, random, hashlib, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import av

from autogaze.datasets.video_utils import read_video_pyav

from semantic_autogaze.train_bighead import BigSimilarityHead

T_FRAMES = 16
GRID = 14
N = GRID * GRID
TOTAL = T_FRAMES * N


def load_head(ckpt, device, expanded_dim=384, n_attn_heads=6, n_attn_layers=2):
    head = BigSimilarityHead(
        hidden_dim=192, embedding_dim=512,
        expanded_dim=expanded_dim,
        n_attn_heads=n_attn_heads, n_attn_layers=n_attn_layers,
        grid_size=GRID,
    )
    state = torch.load(ckpt, map_location=device)
    head.load_state_dict(state)
    return head.to(device).eval()


def crispness(sig_scores):
    """How peaked is the distribution? (max - mean) / (max + 1e-9). Range ~[0,1]."""
    m = sig_scores.max()
    mu = sig_scores.mean()
    return float((m - mu) / (m + 1e-9))


def try_read_video(vp, T=T_FRAMES):
    try:
        c = av.open(vp)
        # sample T evenly across total frames
        n_total = c.streams.video[0].frames or 32
        idxs = np.linspace(0, max(n_total - 1, 0), T).astype(int).tolist()
        raw = read_video_pyav(container=c, indices=idxs)
        c.close()
        if raw.shape[0] < T: return None
        return raw
    except Exception:
        return None


def main(args):
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    head = load_head(args.ckpt, device,
                     expanded_dim=args.expanded_dim,
                     n_attn_heads=args.n_attn_heads,
                     n_attn_layers=args.n_attn_layers)
    print(f"[setup] head params: {sum(p.numel() for p in head.parameters())/1e6:.3f}M")

    clipseg_files = sorted(glob.glob(os.path.join(args.clipseg_dir, "*_clipseg_clip.pt")))
    random.shuffle(clipseg_files)
    # Use a held-out slice (last 10%) the head did not train on
    clipseg_files = clipseg_files[int(0.9 * len(clipseg_files)):]
    print(f"[setup] val clipseg files: {len(clipseg_files)}")

    # Collect (teacher, student, metadata) for samples where teacher had signal
    per_sample = []
    scanned = 0
    target_n = args.n_samples
    for cf in clipseg_files:
        if len(per_sample) >= target_n:
            break
        data = torch.load(cf, map_location="cpu", weights_only=False)
        vp = data["video_path"]
        key = hashlib.md5(vp.encode()).hexdigest()
        hp = os.path.join(args.hidden_dir, f"{key}_hidden.pt")
        if not os.path.exists(hp):
            continue
        hidden = torch.load(hp, map_location=device, weights_only=True)[None]  # (1, TN, 192)

        for q in data["queries"]:
            if len(per_sample) >= target_n: break
            target_logits = q["target_scores"].to(device)  # (TN,)
            text_emb = q["text_embedding"].to(device)[None]  # (1, 512)
            query_text = q.get("text", q.get("query_text", "?"))

            teacher_sig = torch.sigmoid(target_logits).cpu().numpy()
            # Filter: we want queries where the teacher actually found something
            if teacher_sig.max() < args.min_teacher_max:
                scanned += 1
                continue

            # Student forward
            student_logits = head(hidden, text_emb)  # (1, TN)
            student_sig = torch.sigmoid(student_logits[0]).cpu().numpy()

            # Metrics
            t_crisp = crispness(teacher_sig)
            s_crisp = crispness(student_sig)
            mse = float(((teacher_sig - student_sig) ** 2).mean())
            # Pearson correlation
            tv = teacher_sig - teacher_sig.mean()
            sv = student_sig - student_sig.mean()
            corr = float((tv * sv).sum() / (np.sqrt((tv**2).sum() * (sv**2).sum()) + 1e-12))

            per_sample.append({
                "video": vp, "query": query_text,
                "teacher_crispness": t_crisp,
                "student_crispness": s_crisp,
                "teacher_max": float(teacher_sig.max()),
                "student_max": float(student_sig.max()),
                "mse": mse, "pearson": corr,
                "teacher_sig": teacher_sig.reshape(T_FRAMES, GRID, GRID),
                "student_sig": student_sig.reshape(T_FRAMES, GRID, GRID),
            })
            scanned += 1

    print(f"[collect] got {len(per_sample)} qualifying samples (scanned {scanned})")

    # ---- Aggregate stats ----
    def agg(key):
        v = np.array([r[key] for r in per_sample])
        return float(v.mean()), float(v.std())

    summary = {
        "n_samples": len(per_sample),
        "teacher_crispness_mean_std": agg("teacher_crispness"),
        "student_crispness_mean_std": agg("student_crispness"),
        "teacher_max_mean_std": agg("teacher_max"),
        "student_max_mean_std": agg("student_max"),
        "mse_mean_std": agg("mse"),
        "pearson_mean_std": agg("pearson"),
    }
    print("\n" + "=" * 70)
    print("CRISPNESS SUMMARY (higher = more peaked distribution)")
    print("=" * 70)
    for k, v in summary.items():
        if isinstance(v, tuple):
            m, s = v
            print(f"  {k:<38}  {m:.4f} ± {s:.4f}")
        else:
            print(f"  {k:<38}  {v}")

    # Interpretation
    tm, _ = agg("teacher_crispness")
    sm, _ = agg("student_crispness")
    tm_max, _ = agg("teacher_max")
    pc, _ = agg("pearson")
    print("\nInterpretation:")
    if tm < 0.35:
        print(f"  * TEACHER fuzziness: mean crispness = {tm:.3f} (<0.35 is fuzzy).")
        print("    → CLIPSeg teacher is a likely root cause of soft heatmaps.")
        print("    → Training on segmentation datasets with hard masks should help.")
    else:
        print(f"  * Teacher crispness is OK ({tm:.3f}).")
    gap = tm - sm
    if gap > 0.08:
        print(f"  * STUDENT loses crispness: teacher {tm:.3f} → student {sm:.3f} (gap {gap:.3f}).")
        print("    → Head architecture / training may be smoothing predictions.")
    elif gap > 0.03:
        print(f"  * Small student→teacher gap ({gap:.3f}). Head is mostly tracking teacher.")
    else:
        print(f"  * Student matches teacher crispness ({sm:.3f} vs {tm:.3f}). Head is fine.")
    if pc > 0.7:
        print(f"  * High teacher↔student Pearson ({pc:.3f}): head has learned the target shape.")
    elif pc > 0.4:
        print(f"  * Moderate Pearson ({pc:.3f}): head captures broad regions but not detail.")
    else:
        print(f"  * Low Pearson ({pc:.3f}): head is not well-aligned with teacher.")

    # ---- Save JSON (drop heavy arrays) ----
    json_rows = [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in per_sample]
    with open(os.path.join(args.output_dir, "crispness.json"), "w") as f:
        json.dump({"summary": summary, "samples": json_rows}, f, indent=2)
    print(f"[save] {args.output_dir}/crispness.json")

    # ---- Visualizations ----
    n_frames_show = 8
    frame_step = T_FRAMES // n_frames_show
    shown_frames = list(range(0, T_FRAMES, frame_step))[:n_frames_show]

    # Sort samples so high-teacher-crispness examples come first — these are
    # the "the teacher was sharp" cases, which show us whether the student is
    # keeping up.
    per_sample_sorted = sorted(per_sample, key=lambda r: -r["teacher_crispness"])

    n_vis = min(args.n_vis, len(per_sample_sorted))
    for i, r in enumerate(per_sample_sorted[:n_vis]):
        vp = r["video"]
        raw = try_read_video(vp, T=T_FRAMES)
        if raw is None:
            raw = np.zeros((T_FRAMES, 224, 224, 3), dtype=np.uint8)

        # normalize for display
        disp = raw.astype(np.float32) / 255.0

        fig, axes = plt.subplots(4, n_frames_show, figsize=(n_frames_show * 2, 8))
        vmax = max(r["teacher_sig"].max(), r["student_sig"].max())

        for col, t in enumerate(shown_frames):
            ax = axes[0, col]
            ax.imshow(disp[t] if t < disp.shape[0] else np.zeros_like(disp[0]))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"frame {t}", fontsize=9)
            if col == 0: ax.set_ylabel("raw", fontsize=10)

            ax = axes[1, col]
            ax.imshow(disp[t] if t < disp.shape[0] else np.zeros_like(disp[0]))
            ax.imshow(r["teacher_sig"][t], cmap="jet", alpha=0.55, vmin=0, vmax=vmax,
                      extent=[0, disp.shape[2], disp.shape[1], 0])
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0: ax.set_ylabel("CLIPSeg\nteacher", fontsize=10)

            ax = axes[2, col]
            ax.imshow(disp[t] if t < disp.shape[0] else np.zeros_like(disp[0]))
            ax.imshow(r["student_sig"][t], cmap="jet", alpha=0.55, vmin=0, vmax=vmax,
                      extent=[0, disp.shape[2], disp.shape[1], 0])
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0: ax.set_ylabel("Our head\nstudent", fontsize=10)

            diff = np.abs(r["teacher_sig"][t] - r["student_sig"][t])
            ax = axes[3, col]
            ax.imshow(disp[t] if t < disp.shape[0] else np.zeros_like(disp[0]))
            ax.imshow(diff, cmap="magma", alpha=0.6, vmin=0, vmax=0.5,
                      extent=[0, disp.shape[2], disp.shape[1], 0])
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0: ax.set_ylabel("|student−teacher|", fontsize=10)

        fig.suptitle(
            f'query="{r["query"]}"   {os.path.basename(vp)}\n'
            f'teacher crisp={r["teacher_crispness"]:.3f} (max {r["teacher_max"]:.2f})   '
            f'student crisp={r["student_crispness"]:.3f} (max {r["student_max"]:.2f})   '
            f'MSE={r["mse"]:.4f}  Pearson={r["pearson"]:.3f}',
            fontsize=10,
        )
        fig.tight_layout()
        safe_q = r["query"].replace(" ", "_").replace("/", "_")[:25]
        safe_v = os.path.basename(vp).replace(".mp4", "").replace("/", "_")[:25]
        out = os.path.join(vis_dir, f"crisp_{i:02d}_{safe_q}__{safe_v}.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  vis [{i+1}/{n_vis}]  q='{r['query'][:20]:<20}'  t={r['teacher_crispness']:.2f} s={r['student_crispness']:.2f}  -> {out}")

    # ---- Summary scatter: teacher crispness vs student crispness ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    tc = np.array([r["teacher_crispness"] for r in per_sample])
    sc = np.array([r["student_crispness"] for r in per_sample])
    ax.scatter(tc, sc, alpha=0.7, color="tab:blue")
    lims = [0, max(tc.max(), sc.max()) * 1.05]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("Teacher (CLIPSeg) crispness")
    ax.set_ylabel("Student (our head) crispness")
    ax.set_title(f"Student tracks teacher crispness\n(mean T={tc.mean():.3f}, S={sc.mean():.3f})")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    ax.hist(tc, bins=25, alpha=0.6, label=f"teacher (mean={tc.mean():.3f})", color="tab:orange")
    ax.hist(sc, bins=25, alpha=0.6, label=f"student (mean={sc.mean():.3f})", color="tab:blue")
    ax.set_xlabel("Crispness (max-mean)/max")
    ax.set_ylabel("# samples")
    ax.set_title("Crispness distribution")
    ax.grid(True, alpha=0.3); ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "crispness_summary.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {args.output_dir}/crispness_summary.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    p.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    p.add_argument("--ckpt", default="results/bighead_warmrestart/best_bighead_student.pt")
    p.add_argument("--expanded_dim", type=int, default=384)
    p.add_argument("--n_attn_heads", type=int, default=6)
    p.add_argument("--n_attn_layers", type=int, default=2)
    p.add_argument("--n_samples", type=int, default=48)
    p.add_argument("--n_vis", type=int, default=10)
    p.add_argument("--min_teacher_max", type=float, default=0.55,
                   help="Only include queries where teacher sigmoid max exceeds this.")
    p.add_argument("--output_dir", default="results/crispness")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
