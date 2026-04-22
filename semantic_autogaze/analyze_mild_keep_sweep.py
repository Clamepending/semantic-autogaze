"""
Analyze the mild-keep filter sweep (cycle 3 of r/autogaze-aware-filter-train).

For each filter config in the mild-keep sweep, compare against AutoGaze-only
at matched effective keep count (from r/filter-token-count-ablation's gaze
budget sweep):

  Intersect 75 %  <->  AutoGaze scale 0.75  (interpolated ~0.410)
  Intersect 50 %  <->  AutoGaze scale 0.50  = 0.402 (exact vanilla sweep point)
  Intersect 25 %  <->  AutoGaze scale 0.25  (interpolated ~0.385)
  Intersect 10 %  <->  AutoGaze scale 0.10  = 0.328 (exact sanity reference)

The decisive verdict per pair: if the filter STRICTLY beats matched
AutoGaze-only on paired flip (+2 or more net), then filter choice adds
value at that keep ratio. If paired flip is within |1| or filter loses,
filter choice is dominated.

Usage:
    python -m semantic_autogaze.analyze_mild_keep_sweep \\
      --mild_keep_sweep results/autogaze_aware_filter_train/mild_keep_sweep/hlvid_subset.json \\
      --budget_sweep results/filter_token_count_ablation/gaze_budget_sweep/hlvid_subset.json
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
    with open(path) as f:
        data = json.load(f)
    per = defaultdict(dict)
    for row in data["per_sample"]:
        per[row["config"]][row["question_id"]] = row["correct"]
    return per, data["n_samples"]


def paired_flip(a_map, b_map):
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


def print_acc_line(name, flags):
    correct_list = [v for _, v in sorted(flags.items())]
    acc = sum(correct_list) / len(correct_list)
    lo, hi = bootstrap_ci(correct_list)
    print(f"  [{name:<30}] acc={acc:.3f} ({sum(correct_list)}/{len(correct_list)})  "
          f"95% CI [{lo:.3f}, {hi:.3f}]")
    return acc


def main(args):
    mild_per, mild_n = load_per_sample(args.mild_keep_sweep)
    budget_per, budget_n = load_per_sample(args.budget_sweep)

    print(f"\n=== MILD-KEEP FILTER SWEEP (n={mild_n}) ===\n")
    for cfg in mild_per:
        print_acc_line(cfg, mild_per[cfg])

    print(f"\n=== AUTO-GAZE ONLY GAZE-BUDGET SWEEP (n={budget_n}) ===\n")
    for cfg in budget_per:
        print_acc_line(cfg, budget_per[cfg])

    print("\n=== SANITY CHECK: Intersect 10 % reproduces r/hlvid-household-expand? ===\n")
    sanity_cfg = "Intersect 10% (sanity)"
    if sanity_cfg in mild_per:
        mild_intersect_10 = mild_per[sanity_cfg]
        print_acc_line(sanity_cfg, mild_intersect_10)
        print("  (expected: 29/122 = 0.238 from r/hlvid-household-expand)")
    else:
        print("  (no sanity config found)")

    print("\n=== CROSS-COMPARISON — MILD-KEEP FILTER vs MATCHED AUTOGAZE-ONLY BUDGET ===\n")
    print("Paired flip: (positive net → filter beats matched AutoGaze-only budget;")
    print("             negative → budget beats filter at matched count)\n")
    cross_pairs = [
        # (filter_cfg,         matched_autogaze_cfg)
        ("Intersect 75%",      "vanilla tile=0.20 thumb=0.75"),  # 75% of AutoGaze's 75% = 56% keep ≈ scale 0.75
        ("Intersect 50%",      "scaled 0.5 tile=0.10 thumb=0.375"),   # exact match
        ("Intersect 25%",      "scaled 0.3 tile=0.06 thumb=0.225"),   # 25% of 75%=18.75% ≈ scale 0.25-0.3
        ("Semantic 75%",       "vanilla tile=0.20 thumb=0.75"),
        ("Semantic 50%",       "scaled 0.5 tile=0.10 thumb=0.375"),
        ("Intersect 10% (sanity)", "scaled 0.1 tile=0.02 thumb=0.075"),   # reference pair
    ]
    for filter_cfg, budget_cfg in cross_pairs:
        if filter_cfg not in mild_per or budget_cfg not in budget_per:
            print(f"  ({filter_cfg!r} or {budget_cfg!r} missing — skip)")
            continue
        f_flags = mild_per[filter_cfg]
        b_flags = budget_per[budget_cfg]
        f_correct_list = [v for _, v in sorted(f_flags.items())]
        b_correct_list = [v for _, v in sorted(b_flags.items())]
        f_acc = sum(f_correct_list) / len(f_correct_list)
        b_acc = sum(b_correct_list) / len(b_correct_list)
        f_win, b_win, both_ok, both_bad, shared = paired_flip(f_flags, b_flags)
        print(f"  FILTER={filter_cfg!r}   (acc={f_acc:.3f})")
        print(f"  BUDGET={budget_cfg!r}   (acc={b_acc:.3f})")
        print(f"     filter_only_correct={f_win}  budget_only_correct={b_win}  "
              f"both_ok={both_ok}  both_wrong={both_bad}  (shared={shared})")
        net = f_win - b_win
        if f_win + b_win == 0:
            verdict = "IDENTICAL per-question outcomes"
        elif abs(net) <= 1:
            verdict = "within 1-sample noise — filter and budget are indistinguishable"
        elif net >= 2:
            verdict = f"FILTER BEATS MATCHED BUDGET (+{net}) — token CHOICE helps at this keep"
        else:
            verdict = f"BUDGET BEATS MATCHED FILTER ({net:+d}) — filter choice dominated at this keep"
        print(f"     verdict: {verdict}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mild_keep_sweep", required=True)
    p.add_argument("--budget_sweep",
                   default="results/filter_token_count_ablation/gaze_budget_sweep/hlvid_subset.json")
    args = p.parse_args()
    main(args)
