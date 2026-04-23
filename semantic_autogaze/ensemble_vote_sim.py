"""Simulate test-time-ensemble-vote from existing per-config predictions (CPU-only).

Builds a 36-config × 122-question PREDICTED-LETTER matrix from the same
hlvid_subset.json files used by cross_move_matrix.py, then simulates
majority-vote ensembles of various k and config subsets.

Decisive against:
  - ensemble k=3 or k=5 or k=all36 > 51 by > 1-sample tie band -> admission candidate
  - all ensemble variants <= 52 -> direction closed

Key questions:
  1. top-k-by-accuracy majority vote: k ∈ {3, 5, 7, 9, 11, 15, 21, 36}
  2. disagreement-weighted (configs whose predictions most spread the vote)
  3. oracle upper bound: if any k-subset can hit the 77/122 ensemble ceiling

Outputs:
  results/ensemble_vote_sim/analysis.txt
  results/ensemble_vote_sim/per_ensemble.json
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

OUT_DIR = Path("/home/ogata/semantic-autogaze/results/ensemble_vote_sim")


SOURCES: list[tuple[str, str, str]] = [
    ("r/autogaze-aware-filter-train",        "results/autogaze_aware_filter_train/mild_keep_sweep/hlvid_subset.json",     "aware_filter"),
    ("3b49755",                              "results/filter_token_count_ablation/gaze_budget_sweep/hlvid_subset.json",   "budget_sweep"),
    ("r/filter-vs-random-baseline",          "results/filter_vs_random_baseline/hlvid_subset.json",                        "filter_vs_rand"),
    ("r/hlvid-household-expand",             "results/hlvid_household_expand/tier3_full_sweep/hlvid_subset.json",           "hh_expand"),
    ("r/nvila-no-gaze-reduced-budget",       "results/nvila_no_gaze_m1/hlvid_subset.json",                                  "raw_m1"),
    ("r/nvila-no-gaze-reduced-budget",       "results/nvila_vanilla_m1/hlvid_subset.json",                                  "vanilla_m1"),
    ("r/predecoder-bighead-full-stack",      "results/predecoder_household_expand/tier3_full_sweep/hlvid_subset.json",      "predecoder"),
    ("r/subselection-insensitivity-probe",   "results/subselection_insensitivity_probe/hlvid_subset.json",                  "subsel_probe"),
]


def load_rows(ref: str, path: str) -> list[dict]:
    raw = subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)
    data = json.loads(raw)
    if isinstance(data, dict):
        for k in ("per_sample", "samples", "results", "rows"):
            if k in data and isinstance(data[k], list):
                return data[k]
        raise AssertionError(f"no list field in {ref}:{path}")
    assert isinstance(data, list), f"unexpected shape in {ref}:{path}"
    return data


def row_config_key(row: dict, src_tag: str) -> str:
    cfg = row.get("config")
    if cfg:
        return f"{src_tag}::{cfg}"
    return src_tag


def build_pred_matrix() -> tuple[dict[str, dict[str, str]], dict[str, str], list[str], list[str]]:
    pred: dict[str, dict[str, str]] = defaultdict(dict)
    gt: dict[str, str] = {}
    for ref, path, tag in SOURCES:
        rows = load_rows(ref, path)
        for row in rows:
            qid = str(row["question_id"])
            cfg = row_config_key(row, tag)
            p = str(row.get("predicted", "")).strip().upper()[:1]
            g = str(row.get("gt", "")).strip().upper()[:1]
            if not p or p not in "ABCD":
                continue
            pred[qid][cfg] = p
            if qid not in gt and g in "ABCD":
                gt[qid] = g
    qids = sorted(pred.keys(), key=int)
    configs = sorted({c for q in pred.values() for c in q.keys()})
    return pred, gt, qids, configs


def config_accuracy(pred: dict[str, dict[str, str]], gt: dict[str, str], configs: list[str]) -> list[tuple[str, int, int]]:
    acc = []
    for c in configs:
        correct = 0
        total = 0
        for qid, g in gt.items():
            p = pred[qid].get(c)
            if p is None:
                continue
            total += 1
            if p == g:
                correct += 1
        acc.append((c, correct, total))
    acc.sort(key=lambda x: -x[1])
    return acc


def majority_vote(predictions: list[str], tiebreak: str = "A") -> str:
    if not predictions:
        return tiebreak
    counter = Counter(predictions)
    top = counter.most_common()
    max_count = top[0][1]
    candidates = [letter for letter, cnt in top if cnt == max_count]
    if len(candidates) == 1:
        return candidates[0]
    priority = {letter: i for i, letter in enumerate("ABCD")}
    candidates.sort(key=lambda l: priority.get(l, 99))
    return candidates[0]


def ensemble_accuracy(pred: dict[str, dict[str, str]], gt: dict[str, str], configs: list[str],
                      qids: list[str]) -> tuple[int, int, list[tuple[str, str, str, list[str]]]]:
    correct = 0
    per_q: list[tuple[str, str, str, list[str]]] = []
    for qid in qids:
        votes = [pred[qid].get(c) for c in configs if pred[qid].get(c) is not None]
        winner = majority_vote(votes)
        g = gt.get(qid, "")
        if winner == g:
            correct += 1
        per_q.append((qid, g, winner, votes))
    return correct, len(qids), per_q


def oracle_per_q(pred: dict[str, dict[str, str]], gt: dict[str, str], configs: list[str], qids: list[str]) -> int:
    hit = 0
    for qid in qids:
        g = gt.get(qid)
        if any(pred[qid].get(c) == g for c in configs):
            hit += 1
    return hit


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("building prediction matrix from 8 SOURCES...")
    pred, gt, qids, configs = build_pred_matrix()
    print(f"  qids={len(qids)}  configs={len(configs)}  gt={len(gt)}")

    acc = config_accuracy(pred, gt, configs)
    print("\ntop-10 configs by accuracy:")
    for c, k, n in acc[:10]:
        print(f"  {k:3}/{n:3}  ({k/n:.3f})  {c}")

    best_fixed = acc[0][1]
    print(f"\nbest-fixed: {best_fixed}/122")

    # Ensemble variants.
    variants: list[tuple[str, list[str]]] = []
    for k in (3, 5, 7, 9, 11, 15, 21, len(configs)):
        top_k = [c for c, _, _ in acc[:k]]
        variants.append((f"top{k}_by_acc", top_k))

    # Also try: all-raw (no AutoGaze variants), all-AutoGaze configs, pre-decoder set, etc.
    # Tag-based grouping:
    tag_groups: dict[str, list[str]] = defaultdict(list)
    for c in configs:
        tag = c.split("::", 1)[0]
        tag_groups[tag].append(c)
    for tag, cs in tag_groups.items():
        variants.append((f"tag={tag}", cs))

    # Oracle upper bound.
    oracle_total = oracle_per_q(pred, gt, configs, qids)
    print(f"\noracle-per-q (any config correct): {oracle_total}/122")

    results: list[dict] = []
    print("\nmajority-vote ensembles:")
    for name, subset in variants:
        correct, total, per_q = ensemble_accuracy(pred, gt, subset, qids)
        # paired-flip vs best-fixed
        best_cfg = acc[0][0]
        ens_correct_set = {qid for qid, g, w, _ in per_q if w == g}
        best_correct_set = {qid for qid in qids if pred[qid].get(best_cfg) == gt.get(qid)}
        ens_only = len(ens_correct_set - best_correct_set)
        best_only = len(best_correct_set - ens_correct_set)
        both = len(ens_correct_set & best_correct_set)
        print(f"  {name:<30} {correct:3}/{total:3} ({correct/total:.3f})  "
              f"vs best {acc[0][0][:20]}: +{ens_only}/-{best_only} net {ens_only-best_only}")
        results.append({
            "name": name,
            "k": len(subset),
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
            "vs_best_plus": ens_only,
            "vs_best_minus": best_only,
            "vs_best_net": ens_only - best_only,
            "vs_best_both": both,
            "configs": subset,
        })

    out_json = {
        "n_qids": len(qids),
        "n_configs": len(configs),
        "best_fixed_cfg": acc[0][0],
        "best_fixed_correct": best_fixed,
        "oracle_total": oracle_total,
        "config_acc": [{"cfg": c, "correct": k, "total": n} for c, k, n in acc],
        "ensembles": results,
    }
    (OUT_DIR / "per_ensemble.json").write_text(json.dumps(out_json, indent=2))

    # Human summary.
    lines: list[str] = []
    w = lines.append
    w("=" * 72)
    w("Test-time-ensemble-vote simulation on 36 × 122 prediction matrix")
    w("=" * 72)
    w(f"  n_qids          {len(qids)}")
    w(f"  n_configs       {len(configs)}")
    w(f"  best_fixed_cfg  {acc[0][0]}")
    w(f"  best_fixed      {best_fixed}/122 ({best_fixed/len(qids):.3f})")
    w(f"  oracle_total    {oracle_total}/122 ({oracle_total/len(qids):.3f}) (any config correct)")
    w("")
    w("Top-k by-acc ensembles:")
    for r in results:
        if r["name"].startswith("top"):
            w(f"  {r['name']:<20} {r['correct']:3}/{r['total']:3} ({r['accuracy']:.3f})  "
              f"+{r['vs_best_plus']}/-{r['vs_best_minus']} net {r['vs_best_net']}")
    w("")
    w("Tag-group ensembles:")
    for r in results:
        if r["name"].startswith("tag="):
            w(f"  {r['name']:<30} k={r['k']:2}  {r['correct']:3}/{r['total']:3} ({r['accuracy']:.3f})  "
              f"+{r['vs_best_plus']}/-{r['vs_best_minus']} net {r['vs_best_net']}")
    (OUT_DIR / "analysis.txt").write_text("\n".join(lines))
    print("")
    print("\n".join(lines))
    print(f"\nwrote {OUT_DIR / 'per_ensemble.json'}")
    print(f"wrote {OUT_DIR / 'analysis.txt'}")


if __name__ == "__main__":
    main()
