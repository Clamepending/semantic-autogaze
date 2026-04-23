"""Cycle 1 — measure EasyOCR recall on universal-fail OCR questions (CPU-only).

For each OCR-like question in r/universal-fail-45-analysis's 45 universal-fail
set (~40 after OCR filter), sample 4 frames from the video, run EasyOCR, and
substring-match the detected text against each of the 4 choice strings (A/B/C/D).

Question: of the 40-ish OCR fail questions, how many does EasyOCR actually read
well enough that one-and-only-one choice matches? That's the upper bound on
cycle-2 prompt-injection lift.

Outputs:
  results/ocr_preprocessing/cycle1_recall.json  — per-question match bits
  results/ocr_preprocessing/cycle1_recall.txt   — human summary
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pandas as pd

PARQUET_PATH = "/home/ogata/.cache/huggingface/hub/datasets--bfshi--HLVid/snapshots/4956b601aec0bb9d455bb8f57552f78cbd3f0338/data/test-00000-of-00001.parquet"
VIDEO_DIR = Path("/home/ogata/semantic-autogaze/hlvid_videos/extracted_household/videos")
OUT_DIR = Path("/home/ogata/semantic-autogaze/results/ocr_preprocessing")
FRAME_FRACTIONS = (0.1, 0.35, 0.65, 0.9)
FAIL_QIDS_REF = "r/universal-fail-45-analysis"
FAIL_QIDS_PATH = "results/universal_fail_analysis/fail_qids.json"

OCR_PATTERNS = [
    r"\bwhat does.*\b(say|read|write|display)\b",
    r"\bwhat.*(text|writing|word|letter|number|digit|name|label|sign|symbol|logo)\b",
    r"\bwhat is.*(displayed|visible|written|printed|shown).*text\b",
    r"\btext.*(on|visible|displayed|written|printed|shown|read|say|reads)\b",
    r"\bwritten on\b",
    r"\bsays\b",
    r"\b(brand|model|title|name).*(on|label|visible|displayed)\b",
    r"\b(logo|icon)\b",
]


def is_ocr(q_text: str) -> bool:
    s = q_text.lower()
    return any(re.search(p, s) for p in OCR_PATTERNS)


def parse_choices(q_text: str) -> dict[str, str]:
    choices: dict[str, str] = {}
    for ln in q_text.split("\n"):
        m = re.match(r"^([A-D])\.\s*(.+)$", ln.strip())
        if m:
            choices[m.group(1)] = m.group(2).strip()
    return choices


def parse_stem(q_text: str) -> str:
    stem_lines: list[str] = []
    for ln in q_text.split("\n"):
        if re.match(r"^[A-D]\.", ln.strip()):
            break
        if ln.strip().startswith("Please answer"):
            break
        stem_lines.append(ln)
    return " ".join(stem_lines).strip()


def load_fail_qids() -> list[str]:
    raw = subprocess.check_output(
        ["git", "show", f"{FAIL_QIDS_REF}:{FAIL_QIDS_PATH}"],
        cwd="/home/ogata/semantic-autogaze",
    )
    d = json.loads(raw)
    return list(d["universal_fail_qids"])


def sample_frames(video_path: Path, fractions: tuple[float, ...] = FRAME_FRACTIONS):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    frames = []
    for f in fractions:
        idx = int(f * total)
        idx = max(0, min(total - 1, idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if ok and bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
    cap.release()
    return frames


def run_ocr_on_video(reader: easyocr.Reader, video_path: Path) -> list[str]:
    frames = sample_frames(video_path)
    detections: list[str] = []
    for frame in frames:
        try:
            lines = reader.readtext(frame, detail=0, paragraph=False)
        except Exception as exc:
            print(f"  ocr error on {video_path.name}: {exc}")
            continue
        for ln in lines:
            t = str(ln).strip()
            if t:
                detections.append(t)
    return detections


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def substring_match(detected: list[str], target: str) -> bool:
    if not target.strip():
        return False
    tnorm = norm_text(target)
    if not tnorm:
        return False
    tokens = tnorm.split()
    det_joined = " ".join(norm_text(d) for d in detected)
    if not det_joined:
        return False
    if tnorm in det_joined:
        return True
    if len(tokens) >= 2:
        ngrams = [" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]
        for ng in ngrams:
            if ng in det_joined:
                return True
    if len(tokens) == 1 and len(tokens[0]) >= 4:
        return tokens[0] in det_joined
    return False


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading fail qids...")
    fail_qids = load_fail_qids()
    print(f"  {len(fail_qids)} universal-fail qids")

    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["category"] == "household"].copy()
    df["question_id"] = df["question_id"].astype(int).astype(str)
    df = df.set_index("question_id")

    ocr_qids = [q for q in fail_qids if is_ocr(df.loc[q, "question"])]
    print(f"  {len(ocr_qids)} are OCR-like after is_ocr filter")

    print("initializing EasyOCR (English, CPU)...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    per_q: list[dict] = []
    gt_matches = 0
    unique_matches = 0
    ambiguous_matches = 0
    no_matches = 0

    for i, qid in enumerate(ocr_qids):
        row = df.loc[qid]
        q_text = row["question"]
        gt_letter = row["answer"]
        video_path = VIDEO_DIR / row["video_path"]
        stem = parse_stem(q_text)
        choices = parse_choices(q_text)

        if not video_path.exists():
            print(f"[{i + 1}/{len(ocr_qids)}] qid={qid} MISSING VIDEO {video_path}")
            continue

        detected = run_ocr_on_video(reader, video_path)

        matches: dict[str, bool] = {}
        for letter in ("A", "B", "C", "D"):
            target = choices.get(letter, "")
            matches[letter] = substring_match(detected, target)

        gt_matched = matches.get(gt_letter, False)
        num_matched = sum(matches.values())
        if num_matched == 0:
            no_matches += 1
            verdict = "no_match"
        elif num_matched == 1:
            if gt_matched:
                unique_matches += 1
                verdict = "unique_gt"
            else:
                verdict = "unique_wrong"
        else:
            if gt_matched:
                ambiguous_matches += 1
                verdict = "ambiguous_gt_in"
            else:
                verdict = "ambiguous_gt_out"
        if gt_matched:
            gt_matches += 1

        per_q.append({
            "qid": qid,
            "video": row["video_path"],
            "stem": stem[:160],
            "gt_letter": gt_letter,
            "gt_choice": choices.get(gt_letter, ""),
            "choices": choices,
            "matches": matches,
            "detected_n": len(detected),
            "detected": detected[:30],
            "verdict": verdict,
        })
        print(f"[{i + 1}/{len(ocr_qids)}] qid={qid} gt={gt_letter} detected={len(detected)} matches={''.join('1' if matches[L] else '0' for L in 'ABCD')} -> {verdict}")

    n = len(per_q)
    summary = {
        "n_ocr_fails": n,
        "gt_match_rate": gt_matches / max(1, n),
        "unique_gt_rate": unique_matches / max(1, n),
        "ambiguous_gt_rate": ambiguous_matches / max(1, n),
        "no_match_rate": no_matches / max(1, n),
        "gt_matches_absolute": gt_matches,
        "unique_gt_absolute": unique_matches,
        "ambiguous_gt_absolute": ambiguous_matches,
        "no_match_absolute": no_matches,
    }

    out_json = {"summary": summary, "per_question": per_q}
    (OUT_DIR / "cycle1_recall.json").write_text(json.dumps(out_json, indent=2))

    lines: list[str] = []
    w = lines.append
    w("=" * 72)
    w(f"Cycle 1 — EasyOCR recall on {n} universal-fail OCR-like questions")
    w("=" * 72)
    w("")
    for k, v in summary.items():
        if isinstance(v, float):
            w(f"  {k:<28} {v:.3f}")
        else:
            w(f"  {k:<28} {v}")
    w("")
    w("Per-question detail:")
    for p in per_q:
        w(f"  qid={p['qid']:>3}  gt={p['gt_letter']} ({p['gt_choice'][:30]!r}) "
          f"ABCD={''.join('1' if p['matches'][L] else '0' for L in 'ABCD')} "
          f"dets={p['detected_n']} -> {p['verdict']}")
    (OUT_DIR / "cycle1_recall.txt").write_text("\n".join(lines))
    print("")
    print("\n".join(lines[:20]))
    print(f"wrote {OUT_DIR / 'cycle1_recall.json'}")
    print(f"wrote {OUT_DIR / 'cycle1_recall.txt'}")


if __name__ == "__main__":
    main()
