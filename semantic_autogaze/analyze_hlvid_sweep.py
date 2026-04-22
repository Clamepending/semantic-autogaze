"""Post-hoc analysis for eval_hlvid_subset sweep output.

Reads hlvid_subset.json, computes bootstrap 95% CIs for per-config accuracy,
pairwise flips (vanilla correct -> filter wrong, and vice versa), and prints
a leaderboard-ready summary.

Usage:
  python -m semantic_autogaze.analyze_hlvid_sweep --input results/hlvid_expand_n/tier3_full_sweep/hlvid_subset.json
"""
import argparse, json, os, random
from collections import defaultdict


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    samples = []
    for _ in range(n_boot):
        s = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(s) / n)
    samples.sort()
    lo = samples[int(alpha / 2 * n_boot)]
    hi = samples[int((1 - alpha / 2) * n_boot)]
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--baseline", default="AutoGaze only")
    ap.add_argument("--parquet", default="/home/ogata/.cache/huggingface/hub/datasets--bfshi--HLVid/snapshots/4956b601aec0bb9d455bb8f57552f78cbd3f0338/data/test-00000-of-00001.parquet")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    per_sample = data["per_sample"]
    summary = data["summary"]
    n = data["n_samples"]

    # group per-sample by config -> list of correct(0/1)
    by_cfg = defaultdict(list)
    # and per (cfg, qid) -> correct
    per_q = defaultdict(dict)
    for row in per_sample:
        by_cfg[row["config"]].append(int(row["correct"]))
        per_q[row["config"]][row["question_id"]] = int(row["correct"])

    # optional per-category breakdown (from parquet)
    cat_by_qid = {}
    if os.path.exists(args.parquet):
        try:
            import pandas as pd
            df = pd.read_parquet(args.parquet)
            for _, r in df.iterrows():
                cat_by_qid[int(r["question_id"])] = r["category"]
        except Exception as e:
            print(f"[warn] could not load parquet for category breakdown: {e}")

    print(f"HLVid sweep analysis  n={n}  file={args.input}\n")

    # ---- per-config accuracy + CI ----
    print(f"{'config':<25}  {'acc':>6}  {'95% CI':>18}  {'lat_s':>7}  {'n':>4}")
    baseline_corrects = by_cfg[args.baseline]
    for name in summary:
        corrects = by_cfg[name]
        acc = sum(corrects) / max(len(corrects), 1)
        lo, hi = bootstrap_ci([float(c) for c in corrects])
        lat = summary[name]["avg_latency_s"]
        print(f"{name:<25}  {acc:>6.3f}  [{lo:.3f}, {hi:.3f}]  {lat:>7.2f}  {len(corrects):>4}")

    # ---- pairwise flips vs baseline ----
    print(f"\nFlip analysis vs '{args.baseline}':")
    print(f"{'config':<25}  {'win (0->1)':>10}  {'lose (1->0)':>11}  {'net':>5}")
    for name in summary:
        if name == args.baseline:
            continue
        wins = loses = 0
        for qid, base_c in per_q[args.baseline].items():
            filt_c = per_q[name].get(qid)
            if filt_c is None:
                continue
            if base_c == 0 and filt_c == 1:
                wins += 1
            elif base_c == 1 and filt_c == 0:
                loses += 1
        print(f"{name:<25}  {wins:>10}  {loses:>11}  {wins - loses:>+5}")

    # ---- per-category breakdown ----
    if cat_by_qid:
        cats = sorted(set(cat_by_qid.values()))
        print("\nPer-category accuracy:")
        header = "config".ljust(25) + "  " + "  ".join(f"{c:>12}" for c in cats)
        print(header)
        for name in summary:
            parts = []
            for c in cats:
                subs = [per_q[name][q] for q in per_q[name] if cat_by_qid.get(q) == c]
                if not subs:
                    parts.append("   -/-    ")
                else:
                    parts.append(f"{sum(subs)/len(subs):.3f} ({sum(subs)}/{len(subs)})")
            print(f"{name:<25}  " + "  ".join(f"{p:>12}" for p in parts))

    # ---- verdict text for result doc ----
    base_acc = sum(baseline_corrects) / max(len(baseline_corrects), 1)
    base_lo, base_hi = bootstrap_ci([float(c) for c in baseline_corrects])
    print(f"\nVerdict notes (for result doc):")
    print(f"  baseline ('{args.baseline}') = {base_acc:.3f} 95% CI [{base_lo:.3f}, {base_hi:.3f}]")
    regressions = 0
    ties = 0
    for name in summary:
        if name == args.baseline:
            continue
        corrects = by_cfg[name]
        acc = sum(corrects) / max(len(corrects), 1)
        lo, hi = bootstrap_ci([float(c) for c in corrects])
        if acc < base_acc and hi < base_acc:
            regressions += 1
            tag = "REGRESSION (CI below baseline)"
        elif acc < base_acc:
            tag = "worse but CI overlaps"
            ties += 1
        elif acc > base_acc and lo > base_acc:
            tag = "IMPROVEMENT (CI above baseline)"
        else:
            tag = "better but CI overlaps"
            ties += 1
        print(f"  {name:<25}  {acc:.3f}  [{lo:.3f},{hi:.3f}]  -> {tag}")
    print(f"\n  decisive regressions: {regressions}/8   ties (CI overlap): {ties}/8")


if __name__ == "__main__":
    main()
