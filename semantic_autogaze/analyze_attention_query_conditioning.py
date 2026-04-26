"""r/nvila-attention-distill cycle 1.5 GATE — does NVILA's cross-attention
condition on the question text?

For each (qid, video) on HLVid household, we have two cached NVILA attention
maps at the same smaller config:
  - matched-Q: attention when NVILA receives the actual question for the video
  - shuffled-Q: attention when NVILA receives a deterministic-shuffled OTHER
    question (same seed-42 derangement as r/owlvit-hlvid-vqa)

For attention to be a useful distillation target, matched and shuffled top-K
patches should DIFFER substantially. Specifically, the IoU (intersection over
union) of top-K positions should be CLOSE TO RANDOM EXPECTED IoU (= K/N for
random, asymptotically).

If avg matched∩shuffled IoU > 0.7 → attention is query-INVARIANT → distillation
direction pre-empted (filter learns the same selection regardless of query).

If avg IoU close to random expectation → attention is query-CONDITIONAL →
distillation worth pursuing.

Run:
  python -m semantic_autogaze.analyze_attention_query_conditioning \\
    --matched_dir results/nvila_attention_cache \\
    --shuffled_dir results/nvila_attention_cache_shuffled \\
    --keep_ratio 0.14
"""
from __future__ import annotations
import os, json, argparse, glob
import numpy as np


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int):
    """IoU of top-k positions of a and b. Both 1-D same length."""
    if len(a) != len(b):
        return None
    if np.isnan(a).any() or np.isnan(b).any():
        return None
    top_a = set(np.argsort(a)[-k:].tolist())
    top_b = set(np.argsort(b)[-k:].tolist())
    inter = len(top_a & top_b)
    union = len(top_a | top_b)
    return inter / union if union > 0 else 0.0


def main(args):
    matched_files = sorted(glob.glob(os.path.join(args.matched_dir, "qid_*.npz")))
    print(f"[data] matched cache: {len(matched_files)} files in {args.matched_dir}")

    overlaps = []
    by_qid = {}
    nan_qids_matched = []
    nan_qids_shuffled = []
    missing_shuffled = []

    for mf in matched_files:
        m = np.load(mf, allow_pickle=True)
        qid = int(m["qid"])
        m_attn = m["attention"]

        sf = os.path.join(args.shuffled_dir, f"qid_{qid:04d}.npz")
        if not os.path.exists(sf):
            missing_shuffled.append(qid)
            continue
        s = np.load(sf, allow_pickle=True)
        s_attn = s["attention"]

        if np.isnan(m_attn).any():
            nan_qids_matched.append(qid)
            continue
        if np.isnan(s_attn).any():
            nan_qids_shuffled.append(qid)
            continue

        if len(m_attn) != len(s_attn):
            print(f"  [warn] qid={qid} length mismatch: matched={len(m_attn)} shuffled={len(s_attn)}")
            continue

        n = len(m_attn)
        k = max(1, int(args.keep_ratio * n))
        iou = topk_overlap(m_attn, s_attn, k)
        random_expected_iou = k / (2 * n - k)  # IoU of two random k-of-n sets in expectation
        overlaps.append({
            "qid": qid,
            "n": n,
            "k": k,
            "iou": iou,
            "random_expected_iou": random_expected_iou,
            "iou_above_random": iou - random_expected_iou,
        })
        by_qid[qid] = overlaps[-1]

    print(f"\n[stats] matched-vs-shuffled top-K overlap (k = {args.keep_ratio:.2f} × n)")
    print(f"  pairs analyzed: {len(overlaps)}")
    print(f"  matched NaN qids: {len(nan_qids_matched)}: {nan_qids_matched[:10]}{'...' if len(nan_qids_matched)>10 else ''}")
    print(f"  shuffled NaN qids: {len(nan_qids_shuffled)}: {nan_qids_shuffled[:10]}{'...' if len(nan_qids_shuffled)>10 else ''}")
    print(f"  missing shuffled cache: {len(missing_shuffled)}")

    if not overlaps:
        print("  no pairs to analyze; exiting")
        return

    ious = np.array([o["iou"] for o in overlaps])
    rand_ious = np.array([o["random_expected_iou"] for o in overlaps])
    diffs = ious - rand_ious

    print(f"\n  observed IoU       : mean={ious.mean():.4f}  std={ious.std():.4f}  min={ious.min():.4f}  max={ious.max():.4f}")
    print(f"  random expected IoU: mean={rand_ious.mean():.4f}  std={rand_ious.std():.4f}")
    print(f"  IoU − random       : mean={diffs.mean():+.4f}  std={diffs.std():.4f}")

    # Decision verdict
    print(f"\n[verdict]")
    if ious.mean() > 0.85:
        print("  ❌ NVILA attention is HIGHLY query-INVARIANT (top-K essentially identical for matched and shuffled)")
        print("  → Distillation pre-empted: filter would learn the same selection regardless of query.")
        print("  → Routes to GOAL-CHANGE conversation per result-doc cycle 1.5 gate.")
    elif ious.mean() > 0.70:
        print("  ⚠️  NVILA attention is MOSTLY query-INVARIANT (high overlap; small query-conditional component)")
        print("  → Distillation likely to produce a near-saliency filter, not a true text-conditioned one.")
        print("  → Marginal — consider escalating to full-config attention extraction before committing to Phase 2.")
    elif ious.mean() > 0.50:
        print("  ✓  NVILA attention is PARTIALLY query-CONDITIONAL")
        print("  → Distillation may produce a useful filter; decision rests on whether the conditional signal transfers.")
        print("  → Proceed to attention-as-filter VQA test (full cycle 1.5).")
    else:
        print("  ✓✓ NVILA attention is STRONGLY query-CONDITIONAL")
        print("  → Distillation is worth pursuing. Proceed to Phase 2.")

    # Save per-qid stats
    out_path = os.path.join(args.output_dir, "matched_vs_shuffled_overlap.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "matched_dir": args.matched_dir,
                "shuffled_dir": args.shuffled_dir,
                "keep_ratio": args.keep_ratio,
            },
            "summary": {
                "n_pairs": len(overlaps),
                "iou_mean": float(ious.mean()),
                "iou_std": float(ious.std()),
                "random_expected_iou_mean": float(rand_ious.mean()),
                "iou_above_random_mean": float(diffs.mean()),
            },
            "per_qid": overlaps,
            "nan_qids_matched": nan_qids_matched,
            "nan_qids_shuffled": nan_qids_shuffled,
        }, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--matched_dir", default="results/nvila_attention_cache")
    p.add_argument("--shuffled_dir", default="results/nvila_attention_cache_shuffled")
    p.add_argument("--keep_ratio", type=float, default=0.14,
                   help="K/N ratio for top-K extraction (matches semantic_keep_ratio in r/owlvit-hlvid-vqa)")
    p.add_argument("--output_dir", default="results/nvila_attn_query_conditioning")
    args = p.parse_args()
    main(args)
