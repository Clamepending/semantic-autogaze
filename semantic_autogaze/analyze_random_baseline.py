"""
Analyze r/filter-vs-random-baseline.

Compares uniform-random scoring at Intersect 50 % / 75 % keep against:
  (a) r/autogaze-aware-filter-train's CLIP-distilled BigHead filter at the
      same Intersect 50 % / 75 % (the DECISIVE comparison)
  (b) r/filter-token-count-ablation's AutoGaze-only at scale 0.5 (matched
      count for Intersect 50 %) and vanilla scale 1.0 (matched count for
      Intersect 75 %: AutoGaze keeps all gazed, then random 75 % of those)
  (c) r/hlvid-household-expand vanilla (additional control)

Decision tree (walked by outcome on Intersect 50 % random):
  - IF random ~= CLIP (|paired flip| <= 2 vs CLIP 33/122):
      CLIP adds zero signal; pre-empts teacher-retrain-siglip2.
  - ELIF random <= 0.22 (<=27/122):
      CLIP is anti-correlated with relevance; SigLIP-2 retrain remains a
      live hypothesis (different teacher could flip the sign).
  - ELIF random ~= 0.40 (>=0.40):
      count effect dominates; filter choice is actively anti-correlated at
      this keep ratio — SigLIP-2 becomes an urgent experiment.
  - ELSE (0.22 < random < 0.40, strictly above CLIP 0.27 but below vanilla):
      random has some positive signal over CLIP but not enough to match
      AutoGaze-only — teacher-retrain-siglip2 worth running.

Usage:
    python -m semantic_autogaze.analyze_random_baseline \\
      --random_sweep results/filter_vs_random_baseline/hlvid_subset.json \\
      --mild_keep_sweep <r/autogaze-aware-filter-train results> \\
      --budget_sweep   <r/filter-token-count-ablation results>
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


def compare(label, a_name, a_flags, b_name, b_flags):
    a_correct = [v for _, v in sorted(a_flags.items())]
    b_correct = [v for _, v in sorted(b_flags.items())]
    a_acc = sum(a_correct) / len(a_correct)
    b_acc = sum(b_correct) / len(b_correct)
    a_win, b_win, both_ok, both_bad, shared = paired_flip(a_flags, b_flags)
    print(f"\n[{label}]")
    print(f"  A = {a_name!r}   (acc={a_acc:.3f} = {sum(a_correct)}/{len(a_correct)})")
    print(f"  B = {b_name!r}   (acc={b_acc:.3f} = {sum(b_correct)}/{len(b_correct)})")
    print(f"     A_only_correct={a_win}  B_only_correct={b_win}  "
          f"both_ok={both_ok}  both_wrong={both_bad}  (shared={shared})")
    net = a_win - b_win
    if a_win + b_win == 0:
        verdict = "IDENTICAL per-question outcomes"
    elif abs(net) <= 1:
        verdict = "A ~= B (within 1-sample noise)"
    elif net >= 2:
        verdict = f"A BEATS B (+{net})"
    else:
        verdict = f"B BEATS A ({net:+d})"
    print(f"     verdict: {verdict}")


def main(args):
    rand_per, rand_n = load_per_sample(args.random_sweep)
    print(f"\n=== RANDOM BASELINE (n={rand_n}) ===\n")
    for cfg in rand_per:
        print_acc_line(cfg, rand_per[cfg])

    if args.mild_keep_sweep:
        mild_per, _ = load_per_sample(args.mild_keep_sweep)
        print(f"\n=== CLIP BIGHEAD FILTER (r/autogaze-aware-filter-train reference) ===\n")
        for cfg in ("Intersect 50%", "Intersect 75%"):
            if cfg in mild_per:
                print_acc_line(cfg, mild_per[cfg])

    if args.budget_sweep:
        budget_per, _ = load_per_sample(args.budget_sweep)
        print(f"\n=== AUTOGAZE-ONLY BUDGET SWEEP (r/filter-token-count-ablation reference) ===\n")
        for cfg in ("vanilla tile=0.20 thumb=0.75", "scaled 0.5 tile=0.10 thumb=0.375"):
            if cfg in budget_per:
                print_acc_line(cfg, budget_per[cfg])

    print("\n=== DECISIVE COMPARISONS ===\n")

    # (1) random 50 % vs CLIP 50 %  — the move's core question
    if args.mild_keep_sweep and "Intersect 50% random" in rand_per and "Intersect 50%" in mild_per:
        compare("Random 50 %  vs  CLIP 50 %  (r/autogaze-aware-filter-train)",
                "Intersect 50% random", rand_per["Intersect 50% random"],
                "Intersect 50% CLIP",   mild_per["Intersect 50%"])

    # (2) random 75 % vs CLIP 75 %
    if args.mild_keep_sweep and "Intersect 75% random" in rand_per and "Intersect 75%" in mild_per:
        compare("Random 75 %  vs  CLIP 75 %  (r/autogaze-aware-filter-train)",
                "Intersect 75% random", rand_per["Intersect 75% random"],
                "Intersect 75% CLIP",   mild_per["Intersect 75%"])

    # (3) random 50 % vs AutoGaze scale 0.5 (matched count)
    if args.budget_sweep and "Intersect 50% random" in rand_per and "scaled 0.5 tile=0.10 thumb=0.375" in budget_per:
        compare("Random 50 %  vs  AutoGaze scale 0.5  (matched count)",
                "Intersect 50% random", rand_per["Intersect 50% random"],
                "AutoGaze scale 0.5", budget_per["scaled 0.5 tile=0.10 thumb=0.375"])

    # (4) random 75 % vs vanilla (matched count: all of AutoGaze's 75 %)
    if args.budget_sweep and "Intersect 75% random" in rand_per and "vanilla tile=0.20 thumb=0.75" in budget_per:
        compare("Random 75 %  vs  vanilla AutoGaze  (matched count = 100 % of AutoGaze set)",
                "Intersect 75% random", rand_per["Intersect 75% random"],
                "vanilla AutoGaze", budget_per["vanilla tile=0.20 thumb=0.75"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--random_sweep", default="results/filter_vs_random_baseline/hlvid_subset.json")
    p.add_argument("--mild_keep_sweep", default="results/autogaze_aware_filter_train/mild_keep_sweep/hlvid_subset.json")
    p.add_argument("--budget_sweep", default="results/filter_token_count_ablation/gaze_budget_sweep/hlvid_subset.json")
    args = p.parse_args()
    main(args)
