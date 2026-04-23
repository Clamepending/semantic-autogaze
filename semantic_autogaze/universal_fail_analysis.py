"""Characterize the 45/122 household questions universally wrong across all 36 configs.

Follow-up to cross_move_matrix.py. Identifies the universal-fail set, pulls
question text + gt-answer from HLVid parquet, breaks down by:
  - video distribution (do they cluster on specific videos?)
  - gt-answer letter distribution (A/B/C/D bias?)
  - question-stem length
  - keyword / template patterns (counting / action / object / attribute)
  - choice-length / numeric-vs-text answer type

Compares each axis to (a) the full n=122 distribution and (b) the 14
universal-success questions — to highlight what's *specific* to the fail set.

Outputs:
  results/universal_fail_analysis/analysis.txt
  results/universal_fail_analysis/fail_qids.json
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PARQUET_PATH = "/home/ogata/.cache/huggingface/hub/datasets--bfshi--HLVid/snapshots/4956b601aec0bb9d455bb8f57552f78cbd3f0338/data/test-00000-of-00001.parquet"


KEYWORD_PATTERNS = {
    "counting":   [r"\bhow many\b", r"\bnumber of\b", r"\bcount\b"],
    "color":      [r"\bcolou?r\b", r"\bred\b", r"\bblue\b", r"\bgreen\b", r"\byellow\b", r"\bwhite\b", r"\bblack\b"],
    "action":     [r"\bdoing\b", r"\baction\b", r"\bactivity\b", r"\bperforming\b", r"\busing\b"],
    "object":     [r"\bwhat (is|object|item)\b", r"\bwhich (object|item|tool)\b"],
    "location":   [r"\bwhere\b", r"\blocation\b", r"\bplace\b"],
    "person":     [r"\bwho\b", r"\bperson\b", r"\bhuman\b", r"\bman\b", r"\bwoman\b"],
    "attribute":  [r"\bwhat type\b", r"\bwhat kind\b", r"\bsize\b", r"\bshape\b"],
    "ordering":   [r"\bfirst\b", r"\blast\b", r"\bbefore\b", r"\bafter\b", r"\bthen\b", r"\bnext\b"],
    "yesno":      [r"^(is|are|does|do|has|have|can)\b"],
}


def parse_stem(q_text: str) -> str:
    lines = q_text.split("\n")
    stem_lines = []
    for ln in lines:
        if re.match(r"^[A-D]\.", ln.strip()):
            break
        if ln.strip().startswith("Please answer"):
            break
        stem_lines.append(ln)
    return " ".join(stem_lines).strip()


def parse_choices(q_text: str) -> dict[str, str]:
    choices: dict[str, str] = {}
    for ln in q_text.split("\n"):
        m = re.match(r"^([A-D])\.\s*(.+)$", ln.strip())
        if m:
            choices[m.group(1)] = m.group(2).strip()
    return choices


def keyword_tags(stem: str) -> list[str]:
    s = stem.lower()
    tags = []
    for tag, patterns in KEYWORD_PATTERNS.items():
        for p in patterns:
            if re.search(p, s):
                tags.append(tag)
                break
    return tags


def main() -> None:
    from semantic_autogaze.cross_move_matrix import SOURCES, load_rows, row_config_key

    correct: dict[str, dict[str, int]] = defaultdict(dict)
    for ref, path, tag in SOURCES:
        rows = load_rows(ref, path)
        for row in rows:
            qid = str(row["question_id"])
            cfg = row_config_key(row, tag)
            correct[qid][cfg] = 1 if row.get("correct") else 0

    qids = sorted(correct.keys(), key=int)
    configs = sorted({c for q in correct.values() for c in q.keys()})
    Y = np.array([[correct[q].get(c, 0) for c in configs] for q in qids], dtype=np.int32)

    per_q_sum = Y.sum(axis=1)
    universal_fail_qids = [qids[i] for i in range(len(qids)) if per_q_sum[i] == 0]
    universal_succ_qids = [qids[i] for i in range(len(qids)) if per_q_sum[i] == len(configs)]
    variable_qids = [qids[i] for i in range(len(qids)) if 0 < per_q_sum[i] < len(configs)]

    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["category"] == "household"].copy()
    df["question_id"] = df["question_id"].astype(int).astype(str)
    df = df.set_index("question_id")

    def slice_subset(subset_qids: list[str]) -> dict:
        sub = df.loc[subset_qids]
        stems = [parse_stem(q) for q in sub["question"].tolist()]
        choices_list = [parse_choices(q) for q in sub["question"].tolist()]
        answers = sub["answer"].tolist()
        videos = sub["video_path"].tolist()

        video_counts = Counter(videos)
        answer_counts = Counter(answers)
        tag_counts: Counter[str] = Counter()
        no_tag = 0
        for s in stems:
            tags = keyword_tags(s)
            if not tags:
                no_tag += 1
            for t in tags:
                tag_counts[t] += 1

        stem_lens = [len(s) for s in stems]
        choice_lens_flat = [len(c) for cs in choices_list for c in cs.values()]
        choice_max = [max((len(c) for c in cs.values()), default=0) for cs in choices_list]

        # Numeric-answer detection: gt letter's text is a digit/number-word.
        numeric_words = {"one", "two", "three", "four", "five", "six", "seven",
                         "eight", "nine", "ten", "zero"}
        numeric_answers = 0
        for ans_letter, cs in zip(answers, choices_list):
            if ans_letter not in cs:
                continue
            txt = cs[ans_letter].lower().strip().rstrip(".")
            if re.match(r"^\d+$", txt) or txt in numeric_words:
                numeric_answers += 1

        return {
            "n": len(subset_qids),
            "video_counts": dict(video_counts.most_common()),
            "answer_counts": dict(answer_counts.most_common()),
            "tag_counts": dict(tag_counts.most_common()),
            "no_keyword_tag": no_tag,
            "stem_len_mean": float(np.mean(stem_lens)),
            "stem_len_median": float(np.median(stem_lens)),
            "stem_len_max": int(np.max(stem_lens)),
            "choice_len_mean": float(np.mean(choice_lens_flat)) if choice_lens_flat else 0.0,
            "choice_max_mean": float(np.mean(choice_max)) if choice_max else 0.0,
            "numeric_answers": numeric_answers,
            "numeric_pct": numeric_answers / max(1, len(subset_qids)),
        }

    fail = slice_subset(universal_fail_qids)
    succ = slice_subset(universal_succ_qids)
    var = slice_subset(variable_qids)
    allq = slice_subset(qids)

    lines: list[str] = []
    def w(s: str = "") -> None:
        print(s); lines.append(s)

    w("=" * 72)
    w("universal_fail_analysis :: 45 / 122 universal-fail vs 14 universal-success vs 63 variable")
    w("=" * 72)
    w(f"matrix: {len(qids)} qids x {len(configs)} configs")
    w(f"universal-fail: {len(universal_fail_qids)}  universal-succ: {len(universal_succ_qids)}  variable: {len(variable_qids)}")
    w("")

    def report(name: str, s: dict, baseline: dict | None = None) -> None:
        def ratio(a: float, b: float) -> str:
            return f"{a/b:.2f}" if b > 0 else "-"
        w(f"-- {name} (n={s['n']}) --")
        w(f"  top 5 videos:")
        for i, (v, n) in enumerate(list(s["video_counts"].items())[:5]):
            base_n = baseline["video_counts"].get(v, 0) if baseline else 0
            share_in_subset = n / s["n"]
            share_in_base = base_n / baseline["n"] if baseline and baseline["n"] else 0
            lift = share_in_subset / share_in_base if share_in_base else float("inf")
            lift_str = f"(lift vs all {share_in_subset:.2f} / {share_in_base:.2f} = {lift:.2f}x)" if baseline else ""
            w(f"    {n:3d}  {v}  {lift_str}")
        w(f"  gt-letter split: {s['answer_counts']}")
        w(f"  top keyword tags: {dict(list(s['tag_counts'].items())[:5])}")
        w(f"  untagged stems: {s['no_keyword_tag']} / {s['n']}")
        w(f"  stem len mean/median/max: {s['stem_len_mean']:.1f} / {s['stem_len_median']:.1f} / {s['stem_len_max']}")
        w(f"  choice len mean / max-choice mean: {s['choice_len_mean']:.1f} / {s['choice_max_mean']:.1f}")
        w(f"  numeric answers: {s['numeric_answers']} / {s['n']} ({s['numeric_pct']:.2%})")
        w("")

    report("ALL-122 (baseline)", allq)
    report("UNIVERSAL-FAIL (45)", fail, baseline=allq)
    report("UNIVERSAL-SUCC (14)", succ, baseline=allq)
    report("VARIABLE (63)", var, baseline=allq)

    # Cross-comparison table: axis lifts.
    w("-- Axis lifts (fail vs all, succ vs all) --")
    def axis_row(label: str, f_val: float, s_val: float, a_val: float) -> str:
        def ratio(a, b):
            return f"{a/b:.2f}x" if b else "-"
        return f"  {label:28s} fail={f_val:.2f}  succ={s_val:.2f}  all={a_val:.2f}  fail-lift={ratio(f_val, a_val):>6s}  succ-lift={ratio(s_val, a_val):>6s}"
    w(axis_row("numeric-answer rate", fail["numeric_pct"], succ["numeric_pct"], allq["numeric_pct"]))
    w(axis_row("stem-len mean", fail["stem_len_mean"], succ["stem_len_mean"], allq["stem_len_mean"]))
    w(axis_row("max-choice-len mean", fail["choice_max_mean"], succ["choice_max_mean"], allq["choice_max_mean"]))

    # Tag-specific lifts.
    w("")
    w("-- Keyword-tag frequency rates (subset) --")
    for tag in KEYWORD_PATTERNS:
        f_rate = fail["tag_counts"].get(tag, 0) / fail["n"]
        s_rate = succ["tag_counts"].get(tag, 0) / succ["n"]
        a_rate = allq["tag_counts"].get(tag, 0) / allq["n"]
        f_lift = f_rate / a_rate if a_rate else 0
        s_lift = s_rate / a_rate if a_rate else 0
        w(f"  {tag:12s} fail={f_rate:.2%}  succ={s_rate:.2%}  all={a_rate:.2%}  fail-lift={f_lift:.2f}x  succ-lift={s_lift:.2f}x")

    w("")
    w("-- Video concentration test --")
    total_vids = len(allq["video_counts"])
    top3_all = sum(list(allq["video_counts"].values())[:3]) / allq["n"]
    top3_fail = sum(list(fail["video_counts"].values())[:3]) / fail["n"]
    top3_succ = sum(list(succ["video_counts"].values())[:3]) / succ["n"] if succ["n"] else 0
    w(f"  total unique videos across 122: {total_vids}")
    w(f"  top-3-video share  all={top3_all:.2%}  fail={top3_fail:.2%}  succ={top3_succ:.2%}")

    # Sample 5 fail questions.
    w("")
    w("-- Sample of 5 universal-fail questions --")
    for qid in universal_fail_qids[:5]:
        stem = parse_stem(df.loc[qid]["question"])
        ans = df.loc[qid]["answer"]
        choices = parse_choices(df.loc[qid]["question"])
        vid = df.loc[qid]["video_path"]
        w(f"  qid={qid} video={vid} gt={ans}")
        w(f"    stem: {stem}")
        for L, C in choices.items():
            mark = "*" if L == ans else " "
            w(f"    {mark} {L}. {C}")

    out = Path("results/universal_fail_analysis")
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.txt").write_text("\n".join(lines) + "\n")
    (out / "fail_qids.json").write_text(json.dumps({
        "universal_fail_qids": universal_fail_qids,
        "universal_succ_qids": universal_succ_qids,
        "variable_qids": variable_qids,
        "fail_summary": fail,
        "succ_summary": succ,
        "var_summary": var,
        "all_summary": allq,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
