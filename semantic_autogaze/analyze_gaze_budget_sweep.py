"""
Analyze the AutoGaze-only gaze-budget sweep (cycle-3 of r/filter-token-count-ablation).

Reports per-budget bootstrap CI on accuracy + paired flip vs the vanilla budget,
then cross-tabulates against the filter sweep from r/hlvid-household-expand so we
can see whether matched-effective-keep filter configs regress by the same amount
that AutoGaze-only does at matched gazing_ratio.

The decisive comparison is:

  AutoGaze-only @ scale 0.1 (tile=0.02, thumb=0.075)  ↔  Intersect 10 % or
                                                            Semantic 10 %

If those land at similar accuracy, **token COUNT is the binding constraint** and
no filter architecture on top of AutoGaze can close the regression. If the
AutoGaze-only scale-0.1 beats Intersect 10 % by >1 sample, filter choice is
strictly worse than AutoGaze's own salience — the filter design space reopens.

Usage:
    python -m semantic_autogaze.analyze_gaze_budget_sweep \\
      --budget_sweep results/filter_token_count_ablation/gaze_budget_sweep/hlvid_subset.json \\
      --filter_sweep results/hlvid_household_expand/tier3_full_sweep/hlvid_subset.json
"""

import argparse, json, random
from collections import defaultdict


def bootstrap_ci(correct_flags, n_resample=1000, ci=0.95, seed=0):
    rng = random.Random(seed)
    n = len(correct_flags)
    means = []
    for _ in range(n_resample):
        resample = [correct_flags[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo = means[int((1 - ci) / 2 * n_resample)]
    hi = means[int((1 + ci) / 2 * n_resample) - 1]
    return lo, hi


def load_per_sample(path):
    """Return {config_name: {question_id: correct_bool}}."""
    with open(path) as f:
        data = json.load(f)
    per = defaultdict(dict)
    for row in data["per_sample"]:
        per[row["config"]][row["question_id"]] = row["correct"]
    return per, data["n_samples"]


def paired_flip(a_map, b_map):
    """Return (a_wins, b_wins, both_ok, both_wrong, total_shared)."""
    shared = sorted(set(a_map.keys()) & set(b_map.keys()))
    a_wins = b_wins = both_ok = both_wrong = 0
    for q in shared:
        a, b = a_map[q], b_map[q]
        if a and not b:
            a_wins += 1
        elif b and not a:
            b_wins += 1
        elif a and b:
            both_ok += 1
        else:
            both_wrong += 1
    return a_wins, b_wins, both_ok, both_wrong, len(shared)


def main(args):
    budget_per, budget_n = load_per_sample(args.budget_sweep)
    filter_per, filter_n = load_per_sample(args.filter_sweep) if args.filter_sweep else (None, None)

    print(f"\n=== GAZE-BUDGET SWEEP (n={budget_n}) ===\n")
    budget_names = list(budget_per.keys())
    baseline_name = budget_names[0]
    baseline_flags = budget_per[baseline_name]
    base_correct_list = [v for _, v in sorted(baseline_flags.items())]
    base_acc = sum(base_correct_list) / len(base_correct_list)
    base_lo, base_hi = bootstrap_ci(base_correct_list)
    print(f"  [{baseline_name}] acc={base_acc:.3f} ({sum(base_correct_list)}/{len(base_correct_list)})  "
          f"95% CI [{base_lo:.3f}, {base_hi:.3f}]")

    print(f"\n{'config':<45} {'acc':>7} {'95% CI':>18} {'Δ vs vanilla':>14} {'paired flip':>18}")
    for cfg in budget_names:
        flags = budget_per[cfg]
        correct_list = [v for _, v in sorted(flags.items())]
        acc = sum(correct_list) / len(correct_list)
        lo, hi = bootstrap_ci(correct_list)
        delta = acc - base_acc
        if cfg == baseline_name:
            flip_str = "—"
        else:
            a_w, b_w, *_ = paired_flip(flags, baseline_flags)
            flip_str = f"{a_w} W / {b_w} L ({a_w - b_w:+d})"
        print(f"  {cfg:<43} {acc:7.3f} [{lo:.3f},{hi:.3f}] {delta:+14.3f}  {flip_str:>18}")

    if filter_per is None:
        return

    print(f"\n=== FILTER SWEEP (n={filter_n}) ===\n")
    filter_baseline = "AutoGaze only"
    filter_base_flags = filter_per[filter_baseline]
    filter_base_correct_list = [v for _, v in sorted(filter_base_flags.items())]
    filter_base_acc = sum(filter_base_correct_list) / len(filter_base_correct_list)
    filter_base_lo, filter_base_hi = bootstrap_ci(filter_base_correct_list)
    print(f"  [{filter_baseline}] acc={filter_base_acc:.3f} ({sum(filter_base_correct_list)}/{len(filter_base_correct_list)})  "
          f"95% CI [{filter_base_lo:.3f}, {filter_base_hi:.3f}]")
    print()
    for cfg in filter_per:
        if cfg == filter_baseline:
            continue
        flags = filter_per[cfg]
        correct_list = [v for _, v in sorted(flags.items())]
        acc = sum(correct_list) / len(correct_list)
        lo, hi = bootstrap_ci(correct_list)
        delta = acc - filter_base_acc
        a_w, b_w, *_ = paired_flip(flags, filter_base_flags)
        print(f"  {cfg:<43} {acc:7.3f} [{lo:.3f},{hi:.3f}] {delta:+14.3f}  {a_w} W / {b_w} L ({a_w - b_w:+d})")

    print("\n=== CROSS-COMPARISON ===\n")
    print("Paired flip between matched AutoGaze-only gaze-budget configs and filter configs:")
    print("(positive net → filter beats budget; negative → budget beats filter)\n")
    cross_pairs = [
        ("scaled 0.1 tile=0.02 thumb=0.075", "Intersect 10%"),
        ("scaled 0.1 tile=0.02 thumb=0.075", "Semantic 10%"),
        ("scaled 0.3 tile=0.06 thumb=0.225", "Intersect 30%"),
        ("scaled 0.05 tile=0.01 thumb=0.0375", "Semantic 2%"),
    ]
    for budget_cfg, filter_cfg in cross_pairs:
        if budget_cfg not in budget_per or filter_cfg not in filter_per:
            continue
        b_flags = budget_per[budget_cfg]
        f_flags = filter_per[filter_cfg]
        b_correct_list = [v for _, v in sorted(b_flags.items())]
        f_correct_list = [v for _, v in sorted(f_flags.items())]
        b_acc = sum(b_correct_list) / len(b_correct_list)
        f_acc = sum(f_correct_list) / len(f_correct_list)
        f_win, b_win, both_ok, both_bad, shared = paired_flip(f_flags, b_flags)
        print(f"  BUDGET={budget_cfg!r} (acc={b_acc:.3f})")
        print(f"  FILTER={filter_cfg!r} (acc={f_acc:.3f})")
        print(f"     filter_only_correct={f_win}  budget_only_correct={b_win}  "
              f"both_ok={both_ok}  both_wrong={both_bad}  (shared={shared})")
        if f_win + b_win == 0:
            verdict = "IDENTICAL per-question outcomes"
        elif abs(f_win - b_win) <= 1:
            verdict = "within 1-sample noise — token COUNT likely binding"
        elif f_win > b_win:
            verdict = "filter beats matched budget — token CHOICE helps"
        else:
            verdict = "budget beats matched filter — filter makes choice WORSE"
        print(f"     verdict: {verdict}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--budget_sweep", required=True)
    p.add_argument("--filter_sweep",
                   default="results/hlvid_household_expand/tier3_full_sweep/hlvid_subset.json")
    args = p.parse_args()
    main(args)
