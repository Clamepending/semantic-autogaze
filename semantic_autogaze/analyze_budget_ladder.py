"""r/vanilla-budget-ladder cycle 1 analyzer.

Loads the new budget-ladder sweep JSON plus the prior r/filter-token-count-ablation
sweep JSON (via `git show`-like fallback), merges configs into one table, runs
paired-flip vs vanilla, computes bootstrap 95% CI, and emits a full curve.

Usage:
  python -m semantic_autogaze.analyze_budget_ladder \
    --new results/vanilla_budget_ladder/hlvid_subset.json \
    --prior results/filter_token_count_ablation/gaze_budget_sweep/hlvid_subset.json \
    --out results/vanilla_budget_ladder/analysis.txt
"""
from __future__ import annotations
import argparse
import json
import os
import random
import math
import subprocess
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
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    def logcomb(n, k):
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    log_half_n = n * math.log(0.5)
    tail = 0.0
    for i in range(k + 1):
        tail += math.exp(logcomb(n, i) + log_half_n)
    return min(1.0, 2 * tail)


def load_sweep(path, git_ref=None):
    """Load a sweep JSON. If `path` doesn't exist on disk, try `git show git_ref:path`."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    if git_ref:
        out = subprocess.check_output(
            ["git", "show", f"{git_ref}:{path}"], cwd=os.getcwd()
        )
        return json.loads(out.decode())
    raise FileNotFoundError(f"Neither {path} on disk nor git ref {git_ref}")


def per_q_correct(data):
    out = defaultdict(dict)
    for row in data["per_sample"]:
        out[row["config"]][int(row["question_id"])] = int(row["correct"])
    return dict(out)


def cfg_scale(name):
    """Extract numeric scale from config name like 'scaled 0.9 tile=0.18 thumb=0.675' or 'vanilla ...'."""
    if name.startswith("vanilla"):
        return 1.0
    parts = name.split()
    try:
        return float(parts[1])
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--prior", default="results/filter_token_count_ablation/gaze_budget_sweep/hlvid_subset.json")
    ap.add_argument("--prior-git-ref", default="3b49755")
    ap.add_argument("--baseline", default="vanilla tile=0.20 thumb=0.75")
    ap.add_argument("--out", default="results/vanilla_budget_ladder/analysis.txt")
    args = ap.parse_args()

    new_data = load_sweep(args.new)
    prior_data = load_sweep(args.prior, git_ref=args.prior_git_ref)

    new_per_q = per_q_correct(new_data)
    prior_per_q = per_q_correct(prior_data)

    # merge: new wins on vanilla (deterministic replay) if both present
    merged = {**prior_per_q}
    for c, d in new_per_q.items():
        merged[c] = d  # new overwrites prior (for vanilla determinism check)

    # sanity: check vanilla determinism between old and new
    out_lines = []
    if args.baseline in new_per_q and args.baseline in prior_per_q:
        old_q = prior_per_q[args.baseline]
        new_q = new_per_q[args.baseline]
        common = set(old_q) & set(new_q)
        same = sum(1 for q in common if old_q[q] == new_q[q])
        out_lines.append(
            f"Vanilla determinism check (NVILA deterministic by construction): "
            f"{same}/{len(common)} identical across prior and new sweep."
        )
        out_lines.append("")

    out_lines.append(f"Full budget-ladder curve (merged new + prior):")
    out_lines.append(
        f"{'config':<50}  {'scale':>6}  {'acc':>6}  {'n':>5}  {'95% CI':>18}  {'net':>6}  {'p':>7}"
    )
    base = merged.get(args.baseline, {})
    base_corrects_list = [base[q] for q in sorted(base)]
    base_acc = sum(base_corrects_list) / max(len(base_corrects_list), 1)
    n = len(base_corrects_list)
    base_lo, base_hi = bootstrap_ci([float(c) for c in base_corrects_list])
    out_lines.append(
        f"{args.baseline:<50}  {1.0:>6.2f}  {base_acc:>6.3f}  {n:>5}  "
        f"[{base_lo:.3f}, {base_hi:.3f}]  {'(ref)':>6}  {'-':>7}"
    )

    # sorted by scale descending
    entries = [(c, cfg_scale(c)) for c in merged if c != args.baseline]
    entries.sort(key=lambda x: -x[1])
    for c, sc in entries:
        q_map = merged[c]
        corr = [q_map[q] for q in sorted(q_map)]
        acc = sum(corr) / max(len(corr), 1)
        lo, hi = bootstrap_ci([float(x) for x in corr])
        # paired flip vs merged baseline
        wins = losses = 0
        for q, b in base.items():
            f = q_map.get(q)
            if f is None:
                continue
            if b == 0 and f == 1:
                wins += 1
            elif b == 1 and f == 0:
                losses += 1
        net = wins - losses
        p = binom_two_sided_p(wins, losses)
        out_lines.append(
            f"{c:<50}  {sc:>6.2f}  {acc:>6.3f}  {len(corr):>5}  "
            f"[{lo:.3f}, {hi:.3f}]  {net:>+6d}  {p:>7.4f}"
        )
    out_lines.append("")

    # tie-band call-outs
    out_lines.append("Tie-band analysis (|net| <= 1 vs vanilla 51/122 = admission-adjacent):")
    any_tie = False
    for c, sc in entries:
        q_map = merged[c]
        wins = losses = 0
        for q, b in base.items():
            f = q_map.get(q)
            if f is None:
                continue
            if b == 0 and f == 1:
                wins += 1
            elif b == 1 and f == 0:
                losses += 1
        net = wins - losses
        if abs(net) <= 1:
            out_lines.append(f"  TIE  {c}  net={net:+d}")
            any_tie = True
    if not any_tie:
        out_lines.append("  (no config within 1-sample band)")
    out_lines.append("")

    out = "\n".join(out_lines)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
