"""r/hlvid-av-category-replication cycle 1 analyzer.

Loads the av n=63 9-config sweep from r/hlvid-expand-n@92d7e21 and the household
n=122 9-config sweep from r/hlvid-household-expand (already on disk at either the
in-tree path or pulled via `git show`), then computes:

- per-config accuracy + 95% bootstrap CI on av
- paired-flip net vs vanilla on av + binomial p-values
- 9-config answer matrix on av -> ensemble ceiling / universal-fail / universal-success / variable
- cross-category comparison: same stats on household, plus rank inversions across categories
- stable-outcome set: configs that regress on BOTH categories (pattern-confirming) vs configs that flip

Pure analytical; no inference. Both input JSONs are already committed to the code repo.

Usage:
  python -m semantic_autogaze.analyze_av_vs_household \
    --av results/hlvid_expand_n/tier3_full_sweep/hlvid_subset.json \
    --household results/predecoder_household_expand/tier3_full_sweep/hlvid_subset.json \
    --out results/hlvid_av_category_replication/analysis.txt
"""
from __future__ import annotations
import argparse
import json
import os
import random
import math
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


def binom_two_sided_p(wins, losses):
    """Exact two-sided binomial p for H0: P(win)=0.5 given wins+losses discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    # tail probability P(X <= k) for X ~ Binomial(n, 0.5)
    # p = 2 * sum_{i=0..k} C(n,i) * 0.5^n
    def logcomb(n, k):
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    log_half_n = n * math.log(0.5)
    tail = 0.0
    for i in range(k + 1):
        tail += math.exp(logcomb(n, i) + log_half_n)
    return min(1.0, 2 * tail)


def load_sweep(path):
    with open(path) as f:
        d = json.load(f)
    return d


def per_q_correct(data):
    # returns: {config_name: {qid: 0/1}}
    out = defaultdict(dict)
    for row in data["per_sample"]:
        out[row["config"]][int(row["question_id"])] = int(row["correct"])
    return dict(out)


def ensemble_decomp(per_q, configs):
    """Given per_q {cfg:{qid:0/1}}, return (n_total, ensemble_correct, universal_fail, universal_success, variable)."""
    # qids present in all configs
    qids = None
    for c in configs:
        if c not in per_q:
            continue
        qset = set(per_q[c].keys())
        qids = qset if qids is None else qids & qset
    qids = sorted(qids or [])
    ens_correct = 0
    u_fail = 0
    u_succ = 0
    variable = 0
    for q in qids:
        row = [per_q[c][q] for c in configs if c in per_q]
        if any(row):
            ens_correct += 1
        s = sum(row)
        if s == 0:
            u_fail += 1
        elif s == len(row):
            u_succ += 1
        else:
            variable += 1
    return len(qids), ens_correct, u_fail, u_succ, variable


def analyze_single(data, name, baseline, out_lines):
    per_q = per_q_correct(data)
    by_cfg = {c: list(per_q[c].values()) for c in per_q}
    n = data["n_samples"]
    configs_sorted = list(data["summary"].keys())
    out_lines.append(f"=== {name} (n={n}) ===")
    out_lines.append(
        f"{'config':<25}  {'acc':>6}  {'95% CI':>18}  {'net_vs_van':>10}  {'binom_p':>8}"
    )
    base_corrects = by_cfg[baseline]
    base_acc = sum(base_corrects) / max(len(base_corrects), 1)
    base_lo, base_hi = bootstrap_ci([float(c) for c in base_corrects])
    out_lines.append(
        f"{baseline:<25}  {base_acc:>6.3f}  [{base_lo:.3f}, {base_hi:.3f}]  {'(ref)':>10}  {'-':>8}"
    )
    flip_rows = []
    for c in configs_sorted:
        if c == baseline:
            continue
        corr = by_cfg[c]
        acc = sum(corr) / max(len(corr), 1)
        lo, hi = bootstrap_ci([float(x) for x in corr])
        wins = losses = 0
        for q, b in per_q[baseline].items():
            f = per_q[c].get(q)
            if f is None:
                continue
            if b == 0 and f == 1:
                wins += 1
            elif b == 1 and f == 0:
                losses += 1
        net = wins - losses
        p = binom_two_sided_p(wins, losses)
        out_lines.append(
            f"{c:<25}  {acc:>6.3f}  [{lo:.3f}, {hi:.3f}]  {net:>+10d}  {p:>8.4f}"
        )
        flip_rows.append(
            {
                "config": c,
                "acc": acc,
                "ci": (lo, hi),
                "wins": wins,
                "losses": losses,
                "net": net,
                "p": p,
            }
        )
    # ensemble decomposition
    nt, ens, uf, us, var = ensemble_decomp(per_q, configs_sorted)
    out_lines.append("")
    out_lines.append(
        f"Answer-matrix decomposition (9 configs x {nt} qids):"
    )
    out_lines.append(
        f"  ensemble_correct (>=1 config right):  {ens}/{nt} = {ens / nt:.3f}"
    )
    out_lines.append(
        f"  universal_success (all 9 right):      {us}/{nt} = {us / nt:.3f}"
    )
    out_lines.append(
        f"  universal_fail (all 9 wrong):         {uf}/{nt} = {uf / nt:.3f}"
    )
    out_lines.append(
        f"  variable (some right, some wrong):    {var}/{nt} = {var / nt:.3f}"
    )
    out_lines.append(
        f"  headroom ceil - individual-max:       {ens}/{nt} - {sum(base_corrects) / len(base_corrects) * len(base_corrects):.0f}/{nt}"
    )
    out_lines.append("")
    return {
        "n": n,
        "base_acc": base_acc,
        "base_ci": (base_lo, base_hi),
        "flips": flip_rows,
        "ensemble_correct": ens,
        "universal_fail": uf,
        "universal_success": us,
        "variable": var,
        "nt": nt,
    }


def cross_compare(av, hh, out_lines, av_base_corrects=None, hh_base_corrects=None):
    out_lines.append("=== Cross-category comparison (av vs household) ===")
    out_lines.append(
        f"{'config':<25}  {'av net':>7}  {'hh net':>7}  {'av p':>7}  {'hh p':>7}  {'pattern':<20}"
    )
    # build lookup
    av_map = {r["config"]: r for r in av["flips"]}
    hh_map = {r["config"]: r for r in hh["flips"]}
    common = [c for c in av_map if c in hh_map]
    both_regress = 0
    flip_confirmed = 0
    either_positive = 0
    for c in common:
        a = av_map[c]
        h = hh_map[c]
        if a["net"] < 0 and h["net"] < 0:
            pattern = "both regress"
            both_regress += 1
            flip_confirmed += 1
        elif a["net"] >= 0 and h["net"] < 0:
            pattern = "av_up hh_down"
            either_positive += 1
        elif a["net"] < 0 and h["net"] >= 0:
            pattern = "av_down hh_up"
            either_positive += 1
        else:
            pattern = "both non-neg"
            either_positive += 1
        out_lines.append(
            f"{c:<25}  {a['net']:>+7d}  {h['net']:>+7d}  {a['p']:>7.4f}  {h['p']:>7.4f}  {pattern:<20}"
        )
    out_lines.append("")
    out_lines.append(
        f"  configs regressing on BOTH categories: {both_regress}/{len(common)}"
    )
    out_lines.append(
        f"  configs with any non-neg category:     {either_positive}/{len(common)}"
    )
    out_lines.append(
        f"  pattern-confirm fraction:              {flip_confirmed / max(len(common), 1):.3f}"
    )
    out_lines.append("")
    # ensemble ceilings
    out_lines.append(
        f"Ensemble decomposition across categories:"
    )
    out_lines.append(
        f"  av  n={av['nt']:>3}  ens={av['ensemble_correct']:>3} ({av['ensemble_correct'] / av['nt']:.3f})  u_fail={av['universal_fail']:>3}  u_succ={av['universal_success']:>3}  var={av['variable']:>3}"
    )
    out_lines.append(
        f"  hh  n={hh['nt']:>3}  ens={hh['ensemble_correct']:>3} ({hh['ensemble_correct'] / hh['nt']:.3f})  u_fail={hh['universal_fail']:>3}  u_succ={hh['universal_success']:>3}  var={hh['variable']:>3}"
    )
    out_lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av", required=True)
    ap.add_argument("--household", required=True)
    ap.add_argument("--baseline", default="AutoGaze only")
    ap.add_argument("--out", default="results/hlvid_av_category_replication/analysis.txt")
    ap.add_argument("--matrix-json", default="results/hlvid_av_category_replication/matrix.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_lines = []

    av_data = load_sweep(args.av)
    hh_data = load_sweep(args.household)

    out_lines.append(f"Source files:")
    out_lines.append(f"  av:        {args.av}")
    out_lines.append(f"  household: {args.household}")
    out_lines.append(f"  baseline:  {args.baseline}")
    out_lines.append("")

    av_summary = analyze_single(av_data, "HLVid av", args.baseline, out_lines)
    hh_summary = analyze_single(hh_data, "HLVid household", args.baseline, out_lines)

    cross_compare(av_summary, hh_summary, out_lines)

    # save matrix.json with both per_q tables for downstream
    per_q_av = per_q_correct(av_data)
    per_q_hh = per_q_correct(hh_data)
    matrix = {
        "av": {
            "n": av_summary["nt"],
            "configs": list(av_data["summary"].keys()),
            "per_q": per_q_av,
            "summary": av_summary,
        },
        "household": {
            "n": hh_summary["nt"],
            "configs": list(hh_data["summary"].keys()),
            "per_q": per_q_hh,
            "summary": hh_summary,
        },
    }
    # scrub tuples (CIs) into lists for JSON
    def _scrub(o):
        if isinstance(o, tuple):
            return list(o)
        if isinstance(o, dict):
            return {k: _scrub(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_scrub(x) for x in o]
        return o

    with open(args.matrix_json, "w") as f:
        json.dump(_scrub(matrix), f, indent=2)

    out = "\n".join(out_lines)
    with open(args.out, "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
