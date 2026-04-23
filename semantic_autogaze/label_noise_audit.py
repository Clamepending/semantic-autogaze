"""Cycle 1 — label-noise audit on 36-config × 122-question predicted-letter matrix.

Reuses SOURCES + predicted-letter loading from ensemble_vote_sim. For each qid:
  - universal-fail: compute letter distribution; flag if a single non-gt letter has ≥28/36 votes.
  - universal-success: check "trivially correct" patterns.
  - variable: compute Shannon entropy of predictions; flag near-universal-success candidates.

Output: results/label_noise_audit/audit.json + audit.txt.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from semantic_autogaze.ensemble_vote_sim import SOURCES, load_rows, row_config_key

OUT_DIR = Path("/home/ogata/semantic-autogaze/results/label_noise_audit")
PARQUET_PATH = "/home/ogata/.cache/huggingface/hub/datasets--bfshi--HLVid/snapshots/4956b601aec0bb9d455bb8f57552f78cbd3f0338/data/test-00000-of-00001.parquet"


def build_matrices() -> tuple[
    dict[str, dict[str, str]],  # pred
    dict[str, str],  # gt
    list[str],  # qids sorted
    list[str],  # configs sorted
]:
    pred: dict[str, dict[str, str]] = defaultdict(dict)
    gt: dict[str, str] = {}
    for ref, path, tag in SOURCES:
        rows = load_rows(ref, path)
        for row in rows:
            qid = str(row["question_id"])
            cfg = row_config_key(row, tag)
            p = str(row.get("predicted", "")).strip().upper()[:1]
            g = str(row.get("gt", "")).strip().upper()[:1]
            if p not in "ABCD":
                continue
            pred[qid][cfg] = p
            if qid not in gt and g in "ABCD":
                gt[qid] = g
    qids = sorted(pred.keys(), key=int)
    configs = sorted({c for q in pred.values() for c in q.keys()})
    return pred, gt, qids, configs


def classify(qids: list[str], pred: dict[str, dict[str, str]], gt: dict[str, str],
             configs: list[str]) -> tuple[list[str], list[str], list[str]]:
    universal_fail: list[str] = []
    universal_success: list[str] = []
    variable: list[str] = []
    for qid in qids:
        g = gt[qid]
        c_correct = sum(1 for c in configs if pred[qid].get(c) == g)
        c_with_pred = sum(1 for c in configs if pred[qid].get(c) is not None)
        if c_correct == 0:
            universal_fail.append(qid)
        elif c_correct == c_with_pred:
            universal_success.append(qid)
        else:
            variable.append(qid)
    return universal_fail, universal_success, variable


def letter_distribution(qid: str, pred: dict[str, dict[str, str]], configs: list[str]) -> Counter:
    return Counter(pred[qid].get(c) for c in configs if pred[qid].get(c) is not None)


def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in counter.values() if v > 0)


def parse_choices(q_text: str) -> dict[str, str]:
    import re
    choices: dict[str, str] = {}
    for ln in q_text.split("\n"):
        m = re.match(r"^([A-D])\.\s*(.+)$", ln.strip())
        if m:
            choices[m.group(1)] = m.group(2).strip()
    return choices


def parse_stem(q_text: str) -> str:
    import re
    stem_lines: list[str] = []
    for ln in q_text.split("\n"):
        if re.match(r"^[A-D]\.", ln.strip()):
            break
        if ln.strip().startswith("Please answer"):
            break
        stem_lines.append(ln)
    return " ".join(stem_lines).strip()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("building prediction matrix...")
    pred, gt, qids, configs = build_matrices()
    print(f"  qids={len(qids)} configs={len(configs)}")

    universal_fail, universal_success, variable = classify(qids, pred, gt, configs)
    print(f"  fail={len(universal_fail)} succ={len(universal_success)} var={len(variable)}")

    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["category"] == "household"].copy()
    df["question_id"] = df["question_id"].astype(int).astype(str)
    df = df.set_index("question_id")

    # Audit universal-fail: convergent-wrong-letter signatures.
    fail_audit: list[dict] = []
    convergent_wrong: list[dict] = []  # anomaly candidates
    for qid in universal_fail:
        dist = letter_distribution(qid, pred, configs)
        g = gt[qid]
        total = sum(dist.values())
        top_letter, top_count = dist.most_common(1)[0]
        q_text = df.loc[qid, "question"]
        stem = parse_stem(q_text)
        choices = parse_choices(q_text)
        entry = {
            "qid": qid,
            "gt": g,
            "gt_choice": choices.get(g, ""),
            "stem": stem[:180],
            "choices": choices,
            "dist": {k: v for k, v in dist.most_common()},
            "top_letter": top_letter,
            "top_count": top_count,
            "top_pct": top_count / total if total else 0.0,
            "entropy": shannon_entropy(dist),
            "is_convergent_wrong": top_letter != g and top_count >= 28,
        }
        fail_audit.append(entry)
        if entry["is_convergent_wrong"]:
            convergent_wrong.append(entry)

    # Audit universal-success: trivially-correct patterns.
    succ_audit: list[dict] = []
    for qid in universal_success:
        g = gt[qid]
        q_text = df.loc[qid, "question"]
        stem = parse_stem(q_text)
        choices = parse_choices(q_text)
        gt_choice = choices.get(g, "")
        # Trivially correct heuristics: gt choice is longest by a wide margin; gt letter is C (modal).
        choice_lens = {L: len(choices.get(L, "")) for L in "ABCD"}
        longest_L = max(choice_lens, key=lambda L: choice_lens[L])
        longest_len = choice_lens[longest_L]
        second_longest = sorted(choice_lens.values(), reverse=True)[1] if len(choice_lens) >= 2 else 0
        gt_dominant = longest_L == g and longest_len > 1.5 * second_longest
        gt_in_stem = any(word.lower() in stem.lower() for word in gt_choice.split() if len(word) >= 4)
        succ_audit.append({
            "qid": qid,
            "gt": g,
            "gt_choice": gt_choice,
            "stem": stem[:180],
            "choices": choices,
            "gt_letter_is_modal_C": g == "C",
            "gt_dominant_length": gt_dominant,
            "gt_in_stem": gt_in_stem,
        })

    # Audit variable: near-universal-success candidates (1-2 wrong).
    var_audit: list[dict] = []
    near_success: list[dict] = []  # 34-35/36 correct — could flip ceiling interpretation
    for qid in variable:
        dist = letter_distribution(qid, pred, configs)
        g = gt[qid]
        total = sum(dist.values())
        c_correct = dist.get(g, 0)
        entry = {
            "qid": qid,
            "gt": g,
            "c_correct": c_correct,
            "c_total": total,
            "entropy": shannon_entropy(dist),
            "dist": {k: v for k, v in dist.most_common()},
        }
        var_audit.append(entry)
        if c_correct >= 34:
            near_success.append(entry)

    # Label-corrected ceiling upper bound: if every convergent-wrong qid's gt flips to top_letter,
    # how many configs would be correct on that qid?
    # (Each convergent-wrong qid adds top_count / total ~ 28-36 per config; per-config gain = 1 each.)
    corrected_ceiling_upper = 51 + len(convergent_wrong)

    summary = {
        "n_qids": len(qids),
        "n_configs": len(configs),
        "n_universal_fail": len(universal_fail),
        "n_universal_success": len(universal_success),
        "n_variable": len(variable),
        "n_convergent_wrong_fail": len(convergent_wrong),
        "n_near_success_variable": len(near_success),
        "corrected_ceiling_upper_bound": corrected_ceiling_upper,
        "best_fixed_accuracy": 51,
    }

    out = {
        "summary": summary,
        "fail_audit": fail_audit,
        "succ_audit": succ_audit,
        "var_audit": var_audit,
        "convergent_wrong_candidates": convergent_wrong,
        "near_success_variables": near_success,
    }
    (OUT_DIR / "audit.json").write_text(json.dumps(out, indent=2))

    # Human summary.
    lines: list[str] = []
    w = lines.append
    w("=" * 72)
    w("Label-noise audit on 36×122 predicted-letter matrix")
    w("=" * 72)
    for k, v in summary.items():
        w(f"  {k:<36} {v}")
    w("")
    w(f"Convergent-wrong (top non-gt letter ≥ 28/36 = 78%): {len(convergent_wrong)}")
    w("-" * 72)
    for e in convergent_wrong:
        w(f"qid={e['qid']} gt={e['gt']}({e['gt_choice'][:30]!r}) "
          f"top={e['top_letter']}={e['top_count']}/36 ent={e['entropy']:.2f}")
        w(f"  Q: {e['stem'][:140]}")
        for L in "ABCD":
            marker = "*" if L == e['gt'] else ("!" if L == e['top_letter'] else " ")
            ch = e['choices'].get(L, '')[:40]
            w(f"  [{marker}] {L}. {ch}  ({e['dist'].get(L, 0)} votes)")
        w("")
    w(f"Near-universal-success variables (≥ 34/36 correct): {len(near_success)}")
    w("-" * 72)
    for e in near_success[:15]:
        w(f"qid={e['qid']} gt={e['gt']} correct={e['c_correct']}/36 dist={dict(e['dist'])}")
    w("")
    w("Universal-success pattern audit (14 qids):")
    trivial_C = sum(1 for e in succ_audit if e["gt_letter_is_modal_C"])
    gt_dominant = sum(1 for e in succ_audit if e["gt_dominant_length"])
    gt_in_stem = sum(1 for e in succ_audit if e["gt_in_stem"])
    w(f"  gt_letter == C:         {trivial_C}/14")
    w(f"  gt choice dominates by length (> 1.5x next): {gt_dominant}/14")
    w(f"  gt choice words appear in stem: {gt_in_stem}/14")

    (OUT_DIR / "audit.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {OUT_DIR / 'audit.json'}")
    print(f"wrote {OUT_DIR / 'audit.txt'}")


if __name__ == "__main__":
    main()
