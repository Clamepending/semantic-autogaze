"""Cross-move correct/wrong matrix over the HLVid household subset (n=122).

Loads per-sample JSON outputs from every resolved tier-3 household-subset
move via `git show <ref>:<path>` (no local checkout needed), pivots to a
{question_id: {config: correct}} matrix, and reports:

  - total configs across all moves
  - universal-failure count (wrong in ALL configs)
  - universal-success count (correct in ALL configs)
  - ensemble ceiling (at-least-once-correct across all configs)
  - per-question correct-count histogram
  - per-config unique-correct count (questions only this config got right)

Prints to stdout AND writes results/cross_move_matrix/analysis.txt.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


# (branch_or_sha, path-in-that-tree, short-tag).
# hlvid_subset.json files across every tier-3 household-subset move.
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
    # All files use either a list-of-dicts at top level or {"samples":[...]}.
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    assert isinstance(data, list), f"unexpected shape in {ref}:{path}"
    return data


def row_config_key(row: dict, src_tag: str) -> str:
    cfg = row.get("config")
    if cfg:
        return f"{src_tag}::{cfg}"
    return src_tag


def main() -> None:
    matrix: dict[str, dict[str, int]] = defaultdict(dict)  # qid -> {config: 0/1}
    per_config_correct: Counter[str] = Counter()
    per_config_total: Counter[str] = Counter()
    all_configs: set[str] = set()

    for ref, path, tag in SOURCES:
        rows = load_rows(ref, path)
        for row in rows:
            qid = str(row["question_id"])
            cfg = row_config_key(row, tag)
            correct = 1 if row.get("correct") else 0
            matrix[qid][cfg] = correct
            per_config_correct[cfg] += correct
            per_config_total[cfg] += 1
            all_configs.add(cfg)

    n_configs = len(all_configs)
    qids = sorted(matrix.keys())
    n_q = len(qids)

    # Universal fail / success / ensemble ceiling.
    universal_fail = 0
    universal_success = 0
    ensemble_ceiling = 0
    per_q_correct_hist: Counter[int] = Counter()
    per_q_correct_count: dict[str, int] = {}
    for qid in qids:
        row = matrix[qid]
        # Only count qids that have coverage from ALL configs. (Sanity check:
        # they should, since every source uses the same 122-question subset.)
        n_cov = len(row)
        n_correct = sum(row.values())
        per_q_correct_count[qid] = n_correct
        per_q_correct_hist[n_correct] += 1
        if n_correct == 0:
            universal_fail += 1
        if n_correct == n_cov:
            universal_success += 1
        if n_correct >= 1:
            ensemble_ceiling += 1

    # Per-config unique-correct: questions this config got right AND no other did.
    unique_correct: Counter[str] = Counter()
    for qid in qids:
        row = matrix[qid]
        correct_cfgs = [c for c, v in row.items() if v == 1]
        if len(correct_cfgs) == 1:
            unique_correct[correct_cfgs[0]] += 1

    # Assemble output.
    lines: list[str] = []

    def w(s: str = "") -> None:
        print(s)
        lines.append(s)

    w("=" * 72)
    w("cross_move_matrix :: HLVid household subset (n=122)")
    w("=" * 72)
    w(f"sources loaded   : {len(SOURCES)} files")
    w(f"configs covered  : {n_configs}")
    w(f"questions covered: {n_q}")
    w("")

    w("-- Universal failure / success / ensemble ceiling --")
    w(f"universal FAIL (wrong in all configs)    : {universal_fail} / {n_q}")
    w(f"universal SUCCESS (correct in all configs): {universal_success} / {n_q}")
    w(f"ensemble ceiling (correct in >=1 config)  : {ensemble_ceiling} / {n_q}")
    w(f"variable (1..n-1 configs correct)         : {n_q - universal_fail - universal_success} / {n_q}")
    w("")

    w("-- Per-question correct-count histogram (x = #configs correct) --")
    for k in sorted(per_q_correct_hist.keys()):
        w(f"  {k:3d} configs correct : {per_q_correct_hist[k]} questions")
    w("")

    w("-- Per-config total correct (higher = stronger individual config) --")
    for cfg, c in sorted(per_config_correct.items(), key=lambda kv: -kv[1]):
        t = per_config_total[cfg]
        w(f"  {c:3d} / {t:3d}  {cfg}")
    w("")

    w("-- Per-config UNIQUE-correct (qids only this config gets) --")
    for cfg, n in sorted(unique_correct.items(), key=lambda kv: -kv[1]):
        if n:
            w(f"  {n:3d}  {cfg}")
    w("")

    # Write.
    out_dir = Path("results/cross_move_matrix")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.txt").write_text("\n".join(lines) + "\n")

    # Also dump machine-readable json for downstream moves.
    (out_dir / "matrix.json").write_text(json.dumps({
        "n_configs": n_configs,
        "n_questions": n_q,
        "universal_fail": universal_fail,
        "universal_success": universal_success,
        "ensemble_ceiling": ensemble_ceiling,
        "per_config_correct": dict(per_config_correct),
        "per_config_total": dict(per_config_total),
        "per_config_unique_correct": dict(unique_correct),
        "per_q_correct_count": per_q_correct_count,
        "per_q_correct_hist": {str(k): v for k, v in per_q_correct_hist.items()},
        "sources": [{"ref": r, "path": p, "tag": t} for r, p, t in SOURCES],
    }, indent=2))


if __name__ == "__main__":
    main()
