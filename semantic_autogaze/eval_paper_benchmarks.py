"""
Paper-style benchmarks answering four questions:

  Q1 (preservation): at matched token budgets, how does semantic filtering
      compare to a size-matched gaze proxy?  (uses "top-k by GT" as a proxy for
      "AutoGaze at the same budget" — i.e. an oracle patch selection).

  Q2 (usefulness): semantic vs random vs per-frame-random vs oracle.
      Does the learned head beat non-semantic baselines at matched budgets?

  Q3 (subject known): correct-query vs wrong-query semantic filtering.
      If the question's subject is known, does telling the filter help?

  Q4 (speedup): see benchmark_speedup_regimes.py (separate script, GPU-heavy).

All metrics here are computed on cached hidden states + CLIPSeg targets, so
this runs fast (~minutes) without loading AutoGaze or SigLIP.

Bootstraps 95% CIs for every reported number.

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 -m semantic_autogaze.eval_paper_benchmarks \
    --ckpt results/bighead_warmrestart/best_bighead_student.pt \
    --device cuda:0 --output_dir results/paper_benchmarks
"""

import os
import glob
import random
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from semantic_autogaze.eval_filtering import EvalDataset, load_head


T = 16
N = 196
TOTAL = T * N  # 3136


def select_mask(pred_row, target_row, budget, strategy, rng):
    """Return bool mask of shape (TOTAL,) selecting `budget` patches."""
    mask = np.zeros(TOTAL, dtype=bool)
    if strategy == "semantic_global":
        top = np.argpartition(pred_row, -budget)[-budget:]
        mask[top] = True
    elif strategy == "semantic_per_frame":
        k = max(1, budget // T)
        pred_pf = pred_row.reshape(T, N)
        for t in range(T):
            top = np.argpartition(pred_pf[t], -k)[-k:]
            mask[t * N + top] = True
    elif strategy == "random_global":
        idx = rng.choice(TOTAL, size=budget, replace=False)
        mask[idx] = True
    elif strategy == "random_per_frame":
        k = max(1, budget // T)
        for t in range(T):
            idx = rng.choice(N, size=k, replace=False)
            mask[t * N + idx] = True
    elif strategy == "uniform_stride":
        # Evenly spaced patches across flat array — a cheap non-random baseline
        step = max(1, TOTAL // budget)
        idx = np.arange(0, TOTAL, step)[:budget]
        mask[idx] = True
    elif strategy == "oracle_global":
        top = np.argpartition(target_row, -budget)[-budget:]
        mask[top] = True
    else:
        raise ValueError(strategy)
    return mask


def retention_metrics(pred, target, budget, strategy, rng):
    """Compute retention / gt-topk-overlap for one batch."""
    B = pred.shape[0]
    retentions, overlaps = [], []
    for b in range(B):
        t = target[b]
        gt_sum = t.sum()
        if gt_sum < 1e-6:
            continue
        mask = select_mask(pred[b], t, budget, strategy, rng)
        retentions.append(t[mask].sum() / gt_sum)
        gt_top = np.argpartition(t, -budget)[-budget:]
        overlaps.append(len(set(np.where(mask)[0]) & set(gt_top)) / budget)
    return np.array(retentions), np.array(overlaps)


def bootstrap_ci(values, n_boot=1000, q=(2.5, 97.5), rng=None):
    """Return (mean, lo, hi) bootstrap CI."""
    if rng is None:
        rng = np.random.default_rng(0)
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, q[0])), float(np.percentile(means, q[1]))


def run_inference(head, dataloader, device):
    """Returns (preds, targets, video_ids, query_texts)."""
    preds, targets, vids, texts = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            hidden = batch["hidden_states"].to(device)
            query = batch["text_embedding"].to(device)
            target = batch["target_scores"].to(device)
            pred = torch.sigmoid(head(hidden, query)).cpu().numpy()
            preds.append(pred)
            targets.append(torch.sigmoid(target).cpu().numpy())
            vids.extend(batch["video_path"])
            texts.extend(batch["query_text"])
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return preds, targets, vids, texts


def build_wrong_query_preds(head, dataset, device, rng):
    """Run the head with query embeddings from an unrelated video.

    Returns preds_wrong of shape (N_samples, TOTAL).
    """
    # Group sample indices by video so we can shuffle queries across videos.
    by_video = {}
    for i, s in enumerate(dataset.samples):
        by_video.setdefault(s["video_path"], []).append(i)
    videos = list(by_video.keys())

    n = len(dataset)
    preds_wrong = np.zeros((n, TOTAL), dtype=np.float32)
    order = np.arange(n)

    # Build permuted queries: sample i gets a query embedding from a random OTHER video.
    bs = 16
    batch_hidden, batch_query, batch_indices = [], [], []
    with torch.no_grad():
        for i in tqdm(order, desc="Wrong-query inference"):
            s = dataset.samples[i]
            # Pick query from a different video
            other_vid = s["video_path"]
            while other_vid == s["video_path"] and len(videos) > 1:
                other_vid = videos[rng.integers(0, len(videos))]
            other_idx = by_video[other_vid][rng.integers(0, len(by_video[other_vid]))]
            other_q = dataset.samples[other_idx]["text_embedding"]

            batch_hidden.append(s["hidden_states"])
            batch_query.append(other_q)
            batch_indices.append(i)

            if len(batch_hidden) == bs or i == order[-1]:
                hidden = torch.stack(batch_hidden).to(device)
                query = torch.stack(batch_query).to(device)
                pred = torch.sigmoid(head(hidden, query)).cpu().numpy()
                for j, idx in enumerate(batch_indices):
                    preds_wrong[idx] = pred[j]
                batch_hidden, batch_query, batch_indices = [], [], []
    return preds_wrong


def main(args):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Val split (same 90/10 shuffle as training)
    files = sorted(glob.glob(os.path.join(args.clipseg_dir, "*_clipseg_clip*.pt")))
    random.shuffle(files)
    split = int(0.9 * len(files))
    val_files = files[split:][: args.max_files]
    print(f"Val files: {len(val_files)}")

    dataset = EvalDataset(val_files, args.hidden_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    head = load_head(args, device)

    # (1) Correct-query predictions
    preds, targets, vids, texts = run_inference(head, dataloader, device)
    print(f"Loaded {len(preds)} (video, query) samples")

    # (2) Wrong-query predictions — same head, shuffled queries
    preds_wrong = build_wrong_query_preds(head, dataset, device, rng)

    # ========== Q2 + Q3: strategies across budgets ==========
    budget_fracs = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    strategies = [
        ("semantic_global", "Semantic (global top-k)"),
        ("semantic_per_frame", "Semantic (per-frame)"),
        ("random_global", "Random (global)"),
        ("random_per_frame", "Random (per-frame)"),
        ("uniform_stride", "Uniform stride"),
        ("oracle_global", "Oracle (GT top-k)"),
    ]

    results = {name: [] for name, _ in strategies}
    wrong_results = []

    for frac in budget_fracs:
        budget = max(T, int(frac * TOTAL))
        print(f"\nBudget {frac*100:.0f}% ({budget} tokens):")
        for sname, _ in strategies:
            ret, ovlp = retention_metrics(preds, targets, budget, sname, rng)
            m, lo, hi = bootstrap_ci(ret, rng=rng)
            o_m, o_lo, o_hi = bootstrap_ci(ovlp, rng=rng)
            results[sname].append({
                "budget_frac": frac, "budget": budget, "n": int(len(ret)),
                "retention": m, "retention_lo": lo, "retention_hi": hi,
                "overlap": o_m, "overlap_lo": o_lo, "overlap_hi": o_hi,
            })
            print(f"  {sname:<22}: retention={m:.4f} [{lo:.4f}, {hi:.4f}], "
                  f"overlap={o_m:.4f}")

        # Wrong-query using the semantic global strategy
        ret_w, _ = retention_metrics(preds_wrong, targets, budget, "semantic_global", rng)
        m_w, lo_w, hi_w = bootstrap_ci(ret_w, rng=rng)
        wrong_results.append({
            "budget_frac": frac, "budget": budget, "n": int(len(ret_w)),
            "retention": m_w, "retention_lo": lo_w, "retention_hi": hi_w,
        })
        print(f"  {'wrong_query':<22}: retention={m_w:.4f} [{lo_w:.4f}, {hi_w:.4f}]")

    # Save JSON
    payload = {
        "n_val_samples": int(len(preds)),
        "total_patches": TOTAL,
        "num_frames": T,
        "budgets": budget_fracs,
        "correct_query": {name: rows for name, rows in results.items()},
        "wrong_query_semantic_global": wrong_results,
    }
    with open(os.path.join(args.output_dir, "paper_benchmarks.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {args.output_dir}/paper_benchmarks.json")

    # ========== Plots ==========
    colors = {
        "semantic_global": "#2196F3",
        "semantic_per_frame": "#4CAF50",
        "random_global": "#9E9E9E",
        "random_per_frame": "#616161",
        "uniform_stride": "#795548",
        "oracle_global": "#E91E63",
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    for sname, label in strategies:
        rows = results[sname]
        x = [r["budget_frac"] * 100 for r in rows]
        y = [r["retention"] for r in rows]
        lo = [r["retention_lo"] for r in rows]
        hi = [r["retention_hi"] for r in rows]
        ax.plot(x, y, "o-", label=label, color=colors[sname], lw=2)
        ax.fill_between(x, lo, hi, alpha=0.15, color=colors[sname])
    # Wrong-query line
    xw = [r["budget_frac"] * 100 for r in wrong_results]
    yw = [r["retention"] for r in wrong_results]
    low = [r["retention_lo"] for r in wrong_results]
    hiw = [r["retention_hi"] for r in wrong_results]
    ax.plot(xw, yw, "v--", label="Semantic w/ WRONG query", color="#FF5722", lw=2)
    ax.fill_between(xw, low, hiw, alpha=0.15, color="#FF5722")

    ax.set_xlabel("Token Budget (% of 3136)")
    ax.set_ylabel("CLIPSeg Score Mass Retention")
    ax.set_title(f"Score Retention by Strategy (n={len(preds)} val samples, 95% CI)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "retention_with_baselines.png"), dpi=150)
    plt.close(fig)

    # ========== Summary table ==========
    print(f"\n{'='*80}")
    print("SUMMARY: CLIPSeg score retention at key budgets")
    print(f"{'='*80}")
    print(f"{'Strategy':<28}" + "".join([f"{int(f*100)}%".rjust(9) for f in budget_fracs]))
    for sname, label in strategies:
        row = [r["retention"] for r in results[sname]]
        print(f"{label:<28}" + "".join([f"{v:.3f}".rjust(9) for v in row]))
    row_w = [r["retention"] for r in wrong_results]
    print(f"{'WRONG-query (sem global)':<28}" + "".join([f"{v:.3f}".rjust(9) for v in row_w]))
    print(f"\nSavings vs random (global) at 10%: "
          f"{results['semantic_global'][2]['retention'] / max(results['random_global'][2]['retention'], 1e-6):.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--ckpt", default="results/bighead_warmrestart/best_bighead_student.pt")
    parser.add_argument("--head_type", choices=["small", "bighead"], default="bighead")
    parser.add_argument("--output_dir", default="results/paper_benchmarks")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--expanded_dim", type=int, default=384)
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--n_attn_layers", type=int, default=2)
    parser.add_argument("--max_files", type=int, default=500,
                        help="Cap val clipseg files for fast iteration")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args)
