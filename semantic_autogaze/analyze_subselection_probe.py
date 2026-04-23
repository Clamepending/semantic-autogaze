"""
Analyzer for r/subselection-insensitivity-probe — characterizes the random-scoring
curve over keep ratios {90, 75, 50, 25, 10} %% on HLVid household n=122.

Pulls:
  - this move's 3 new configs (90, 25, 10) from results/subselection_insensitivity_probe/
  - existing 50 / 75 random from r/filter-vs-random-baseline
  - CLIP references at 25 and 10 from r/autogaze-aware-filter-train (mild-keep sweep)
  - AutoGaze-only budget sweep (vanilla 1.0, scale 0.5/0.3/0.1/0.05) from
    r/filter-token-count-ablation for matched-count paired-flip tests.

Run:
    python -m semantic_autogaze.analyze_subselection_probe \\
      --probe   results/subselection_insensitivity_probe/hlvid_subset.json \\
      --random_50_75 /tmp/random_75_50.json \\
      --mild_keep    /tmp/clip_mild_keep.json \\
      --budget       /tmp/budget_sweep.json
"""

import argparse, json, random
from collections import defaultdict


def bootstrap_ci(flags, n_resample=1000, ci=0.95, seed=0):
    rng = random.Random(seed)
    n = len(flags)
    means = []
    for _ in range(n_resample):
        means.append(sum(flags[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int((1 - ci) / 2 * n_resample)], means[int((1 + ci) / 2 * n_resample) - 1]


def load_per_sample(path):
    with open(path) as f:
        d = json.load(f)
    per = defaultdict(dict)
    for r in d["per_sample"]:
        per[r["config"]][r["question_id"]] = r["correct"]
    return per


def compare(label, a_name, a_flags, b_name, b_flags):
    shared = sorted(set(a_flags) & set(b_flags))
    a_correct = sum(a_flags[q] for q in shared)
    b_correct = sum(b_flags[q] for q in shared)
    a_only = sum(1 for q in shared if a_flags[q] and not b_flags[q])
    b_only = sum(1 for q in shared if b_flags[q] and not a_flags[q])
    both_ok = sum(1 for q in shared if a_flags[q] and b_flags[q])
    both_bad = sum(1 for q in shared if not a_flags[q] and not b_flags[q])
    net = a_only - b_only
    if a_only + b_only == 0:
        verdict = "IDENTICAL per-question outcomes"
    elif abs(net) <= 1:
        verdict = f"A ~= B (within 1-sample noise; net {net:+d})"
    elif net >= 2:
        verdict = f"A BEATS B (+{net})"
    else:
        verdict = f"B BEATS A ({net:+d})"
    print(f"\n[{label}]")
    print(f"  A = {a_name!r}   {a_correct}/{len(shared)}")
    print(f"  B = {b_name!r}   {b_correct}/{len(shared)}")
    print(f"     A_only={a_only}  B_only={b_only}  both_ok={both_ok}  both_wrong={both_bad}  shared={len(shared)}")
    print(f"     verdict: {verdict}")


def pacc(name, flags):
    correct = [v for _, v in sorted(flags.items())]
    acc = sum(correct) / len(correct)
    lo, hi = bootstrap_ci(correct)
    print(f"  [{name:<30}] acc={acc:.3f} ({sum(correct)}/{len(correct)})  95% CI [{lo:.3f}, {hi:.3f}]")


def main(args):
    probe = load_per_sample(args.probe)
    prior_random = load_per_sample(args.random_50_75)
    mild = load_per_sample(args.mild_keep)
    budget = load_per_sample(args.budget)

    print("=" * 70)
    print("RANDOM CURVE on HLVid household n=122")
    print("=" * 70)
    pacc("vanilla (100 %) AutoGaze-only", budget["vanilla tile=0.20 thumb=0.75"])
    pacc("Intersect 90 % random",        probe["Intersect 90% random"])
    pacc("Intersect 75 % random",        prior_random["Intersect 75% random"])
    pacc("Intersect 50 % random",        prior_random["Intersect 50% random"])
    pacc("Intersect 25 % random",        probe["Intersect 25% random"])
    pacc("Intersect 10 % random",        probe["Intersect 10% random"])

    print("\n" + "=" * 70)
    print("CLIP BIGHEAD (r/autogaze-aware-filter-train) for matched keep")
    print("=" * 70)
    pacc("Intersect 75 % CLIP",           mild["Intersect 75%"])
    pacc("Intersect 50 % CLIP",           mild["Intersect 50%"])
    pacc("Intersect 25 % CLIP",           mild["Intersect 25%"])
    pacc("Intersect 10 % CLIP (sanity)",  mild["Intersect 10% (sanity)"])

    print("\n" + "=" * 70)
    print("AUTOGAZE-ONLY BUDGET (r/filter-token-count-ablation) for matched count")
    print("=" * 70)
    pacc("scale 1.0 (vanilla)",         budget["vanilla tile=0.20 thumb=0.75"])
    pacc("scale 0.5",                   budget["scaled 0.5 tile=0.10 thumb=0.375"])
    pacc("scale 0.1",                   budget["scaled 0.1 tile=0.02 thumb=0.075"])

    print("\n" + "=" * 70)
    print("DECISIVE PAIRED-FLIP COMPARISONS")
    print("=" * 70)

    # (A) vanilla-to-attractor transition — "does 10 % uniform drop cost anything?"
    compare("Intersect 90 % random  vs  vanilla AutoGaze",
            "Intersect 90% random", probe["Intersect 90% random"],
            "vanilla",              budget["vanilla tile=0.20 thumb=0.75"])

    # (B) 90 % random vs attractor samples (75 % and 50 % random)
    compare("Intersect 90 % random  vs  Intersect 75 % random",
            "Intersect 90% random",  probe["Intersect 90% random"],
            "Intersect 75% random",  prior_random["Intersect 75% random"])
    compare("Intersect 90 % random  vs  Intersect 50 % random",
            "Intersect 90% random",  probe["Intersect 90% random"],
            "Intersect 50% random",  prior_random["Intersect 50% random"])

    # (C) CLIP-vs-random test at low keep regimes (attractor below 25 %?)
    compare("Intersect 25 % random  vs  Intersect 25 % CLIP",
            "Intersect 25% random", probe["Intersect 25% random"],
            "Intersect 25% CLIP",   mild["Intersect 25%"])
    compare("Intersect 10 % random  vs  Intersect 10 % CLIP",
            "Intersect 10% random", probe["Intersect 10% random"],
            "Intersect 10% CLIP",   mild["Intersect 10% (sanity)"])

    # (D) random curve monotonicity — does 10 % actually regress vs 25 %?
    compare("Intersect 10 % random  vs  Intersect 25 % random",
            "Intersect 10% random", probe["Intersect 10% random"],
            "Intersect 25% random", probe["Intersect 25% random"])
    compare("Intersect 25 % random  vs  Intersect 50 % random",
            "Intersect 25% random", probe["Intersect 25% random"],
            "Intersect 50% random", prior_random["Intersect 50% random"])

    # (E) matched-count: random at 10 % vs scale-0.1 AutoGaze-only
    compare("Intersect 10 % random  vs  AutoGaze scale 0.1 (matched ~10 % keep)",
            "Intersect 10% random", probe["Intersect 10% random"],
            "AutoGaze scale 0.1",   budget["scaled 0.1 tile=0.02 thumb=0.075"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--probe",         default="results/subselection_insensitivity_probe/hlvid_subset.json")
    p.add_argument("--random_50_75",  default="/tmp/random_75_50.json")
    p.add_argument("--mild_keep",     default="/tmp/clip_mild_keep.json")
    p.add_argument("--budget",        default="/tmp/budget_sweep.json")
    args = p.parse_args()
    main(args)
