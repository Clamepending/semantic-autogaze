"""
Frame-level correctness: does the semantic filter pick the right *frames*?

For each (video, query), CLIPSeg produces 3136 per-patch scores = 16 frames ×
196 patches. For each frame we compute the total GT mass. A frame with high
mass contains the subject; a frame with low mass does not.

Then for our semantic filter (at various budgets), we count how many tokens
it allocates to each frame. The question is whether its allocation correlates
with GT mass — i.e. does it pick high-mass frames?

We also compare against the "absolute score threshold" view: given sigmoid
scores in [0, 1], what absolute threshold gives the best precision/recall
tradeoff? This is the concrete answer to "what threshold should I use".
"""

import os, glob, json, random, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from semantic_autogaze.eval_filtering import EvalDataset, load_head

T = 16
N = 196
TOTAL = T * N


def main(args):
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    files = sorted(glob.glob(os.path.join(args.clipseg_dir, "*_clipseg_clip*.pt")))
    random.shuffle(files)
    val_files = files[int(0.9 * len(files)):][: args.max_files]

    dataset = EvalDataset(val_files, args.hidden_dir)
    dl = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    head = load_head(args, device)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(dl, desc="Inference"):
            h = batch["hidden_states"].to(device)
            q = batch["text_embedding"].to(device)
            t = batch["target_scores"].to(device)
            all_preds.append(torch.sigmoid(head(h, q)).cpu().numpy())
            all_targets.append(torch.sigmoid(t).cpu().numpy())
    preds = np.concatenate(all_preds)  # (B, 3136)
    targets = np.concatenate(all_targets)
    print(f"Samples: {len(preds)}")

    # ========== (A) Frame-level mass allocation ==========
    preds_f = preds.reshape(-1, T, N)
    targets_f = targets.reshape(-1, T, N)

    # For each sample, compute GT frame mass (normalized) and our frame allocation.
    gt_mass = targets_f.sum(axis=-1)  # (B, T)
    # Correlate with our allocation at different budgets
    print("\nFrame-level allocation correlation with GT frame mass:")
    print(f"{'budget':>7} {'spearman_corr':>16} {'top-1 frame hit':>18} {'top-3 frames hit':>18}")
    budgets = [int(0.02 * TOTAL), int(0.05 * TOTAL), int(0.10 * TOTAL),
               int(0.20 * TOTAL), int(0.30 * TOTAL), int(0.50 * TOTAL)]
    frame_analysis = []
    for budget in budgets:
        # Pick top-k globally; count tokens per frame
        corrs, top1_hits, top3_hits = [], [], []
        for b in range(preds.shape[0]):
            gt = targets[b]
            if gt.sum() < 1e-6: continue
            topk = np.argpartition(preds[b], -budget)[-budget:]
            mask = np.zeros(TOTAL, dtype=bool); mask[topk] = True
            alloc = mask.reshape(T, N).sum(axis=-1)  # (T,) our token allocation per frame
            gmass = gt_mass[b]
            # Spearman via rank correlation
            from scipy.stats import spearmanr
            c = spearmanr(alloc, gmass).correlation
            if not np.isnan(c): corrs.append(c)
            # Did we put any tokens in the GT top-1 frame?
            top1_gt = int(np.argmax(gmass))
            top1_hits.append(alloc[top1_gt] > 0)
            # Top-3 frames hit: did we allocate to all 3 of the top-3 GT frames?
            top3_gt = set(np.argsort(gmass)[-3:])
            top3_hits.append(sum(alloc[t] > 0 for t in top3_gt) / 3.0)
        frame_analysis.append({
            "budget_frac": budget / TOTAL,
            "budget": int(budget),
            "spearman_frame_mass": float(np.mean(corrs)) if corrs else 0.0,
            "top1_frame_hit_rate": float(np.mean(top1_hits)),
            "top3_frame_coverage": float(np.mean(top3_hits)),
        })
        print(f"{int(budget/TOTAL*100):>6}% {np.mean(corrs):>16.3f} "
              f"{np.mean(top1_hits):>18.3f} {np.mean(top3_hits):>18.3f}")

    # ========== (B) Absolute score threshold analysis ==========
    # At each sigmoid threshold τ in [0, 1], compute average precision/recall
    # vs the GT (binary: patch is in top-20% of its video) and the effective
    # keep ratio. Answer: what τ gives good quality at what budget?
    print("\nAbsolute sigmoid threshold analysis:")
    print(f"{'tau':>6} {'mean_keep_frac':>16} {'precision':>12} {'recall':>10} {'F1':>8} {'retention':>12}")
    thr_table = []
    gt_binary = (targets >= np.sort(targets, axis=1)[:, -int(0.2 * TOTAL), None])  # top-20% of each video
    for tau in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        sel = preds > tau
        # Ensure at least 1 patch per sample
        keep_fracs, precs, recs, f1s, rets = [], [], [], [], []
        for b in range(preds.shape[0]):
            s = sel[b]; g = gt_binary[b]; t = targets[b]
            if s.sum() == 0:
                s = np.zeros_like(s); s[np.argmax(preds[b])] = True
            keep_fracs.append(s.mean())
            tp = (s & g).sum(); fp = (s & ~g).sum(); fn = (~s & g).sum()
            p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
            precs.append(p); recs.append(r)
            f1s.append(2*p*r / max(p + r, 1e-9))
            if t.sum() > 1e-6:
                rets.append(t[s].sum() / t.sum())
        thr_table.append({
            "tau": tau,
            "mean_keep_frac": float(np.mean(keep_fracs)),
            "precision": float(np.mean(precs)),
            "recall": float(np.mean(recs)),
            "f1": float(np.mean(f1s)),
            "retention": float(np.mean(rets)) if rets else 0.0,
        })
        print(f"{tau:>6.2f} {np.mean(keep_fracs):>16.3f} {np.mean(precs):>12.3f} "
              f"{np.mean(recs):>10.3f} {np.mean(f1s):>8.3f} {np.mean(rets):>12.3f}")

    with open(os.path.join(args.output_dir, "frame_level.json"), "w") as f:
        json.dump({"frame_analysis": frame_analysis,
                   "threshold_analysis": thr_table,
                   "n_samples": int(len(preds))}, f, indent=2)
    print(f"\nSaved: {args.output_dir}/frame_level.json")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x = [r["budget_frac"] * 100 for r in frame_analysis]
    ax.plot(x, [r["spearman_frame_mass"] for r in frame_analysis], "o-", label="Spearman(alloc, GT mass)", color="tab:blue", lw=2)
    ax.plot(x, [r["top1_frame_hit_rate"] for r in frame_analysis], "s-", label="Top-1 GT frame hit rate", color="tab:green", lw=2)
    ax.plot(x, [r["top3_frame_coverage"] for r in frame_analysis], "^-", label="Top-3 GT frames covered", color="tab:orange", lw=2)
    ax.set_xlabel("Token budget (%)"); ax.set_ylabel("Metric")
    ax.set_title("Does the filter pick the correct frames?")
    ax.axhline(0, color="gray", lw=0.5); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    ax.set_ylim([-0.1, 1.05])

    ax = axes[1]
    taus = [r["tau"] for r in thr_table]
    ax.plot(taus, [r["mean_keep_frac"] for r in thr_table], "o-", label="mean keep frac", color="tab:gray", lw=2)
    ax.plot(taus, [r["precision"] for r in thr_table], "s-", label="precision", color="tab:blue", lw=2)
    ax.plot(taus, [r["recall"] for r in thr_table], "^-", label="recall", color="tab:green", lw=2)
    ax.plot(taus, [r["f1"] for r in thr_table], "D-", label="F1", color="tab:red", lw=2)
    ax.plot(taus, [r["retention"] for r in thr_table], "v-", label="retention", color="tab:purple", lw=2)
    ax.set_xlabel("Sigmoid threshold τ"); ax.set_ylabel("Metric")
    ax.set_title("Absolute threshold: what τ should I use?")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "frame_level.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {args.output_dir}/frame_level.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    p.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    p.add_argument("--ckpt", default="results/bighead_warmrestart/best_bighead_student.pt")
    p.add_argument("--head_type", default="bighead")
    p.add_argument("--output_dir", default="results/frame_level")
    p.add_argument("--expanded_dim", type=int, default=384)
    p.add_argument("--n_attn_heads", type=int, default=6)
    p.add_argument("--n_attn_layers", type=int, default=2)
    p.add_argument("--max_files", type=int, default=300)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    main(args)
