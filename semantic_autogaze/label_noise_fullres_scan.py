"""Cycle 2 — scan all 40 convergent-wrong qids for object-absent signal.

For each qid, sample 6 native-res frames, call rv-browser describe-file
with a simplified prompt asking "is the scene/object this question refers
to visible in this frame?" Collect yes/no across 6 frames.

Classification per qid:
  - object_absent (0/6 frames) → strong label-noise signal (question-video mismatch)
  - object_rare (1-2/6 frames) → borderline
  - object_present (3+/6 frames) → pipeline-bias candidate
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PARQUET_PATH = "/home/ogata/.cache/huggingface/hub/datasets--bfshi--HLVid/snapshots/4956b601aec0bb9d455bb8f57552f78cbd3f0338/data/test-00000-of-00001.parquet"
VIDEO_DIR = Path("/home/ogata/semantic-autogaze/hlvid_videos/extracted_household/videos")
AUDIT_PATH = Path("/home/ogata/semantic-autogaze/results/label_noise_audit/audit.json")
OUT_DIR = Path("/home/ogata/semantic-autogaze/results/label_noise_audit/fullres_scan")
N_FRAMES = 6
TIMEOUT_S = 90


def extract_native_frames(video_path: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    frames = []
    for i in range(n):
        frac = (i + 0.5) / n
        idx = int(frac * total)
        idx = max(0, min(total - 1, idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if ok and bgr is not None:
            frames.append(bgr)
    cap.release()
    return frames


def run_vlm(img_path: Path, prompt: str) -> dict:
    try:
        r = subprocess.run(
            ["rv-browser", "describe-file", str(img_path),
             "--provider", "claude", "--prompt", prompt],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
        if r.returncode != 0:
            return {"ok": False, "error": r.stderr[:200]}
        return json.loads(r.stdout)
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def build_prompt(entry: dict) -> str:
    stem = entry["stem"].strip()
    return (
        f"The question asks: \"{stem}\"\n\n"
        "Looking at this single video frame, is the scene or object the question refers to "
        "(the shelf/pack/screen/button/outlet/etc described in the question) visible in this frame? "
        "Answer with one word only at the start: YES if the described scene/object is clearly visible, "
        "NO if it is not in this frame. "
        "Then optionally a short phrase describing what you see instead."
    )


def parse_yes_no(analysis: str) -> str:
    if not analysis:
        return "ERR"
    first = re.split(r"[^A-Za-z]", analysis.strip())[0].upper()
    if first in ("YES", "Y"):
        return "YES"
    if first in ("NO", "N"):
        return "NO"
    return "OTHER"


def classify(yes_no_counts: dict[str, int], n: int) -> str:
    yes = yes_no_counts.get("YES", 0)
    if yes == 0:
        return "object_absent"
    if yes <= 2:
        return "object_rare"
    return "object_present"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = json.loads(AUDIT_PATH.read_text())
    candidates = audit["convergent_wrong_candidates"]

    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["category"] == "household"].copy()
    df["question_id"] = df["question_id"].astype(int).astype(str)
    df = df.set_index("question_id")

    results: dict[str, dict] = {}
    for entry in candidates:
        qid = entry["qid"]
        vp = VIDEO_DIR / df.loc[qid, "video_path"]
        if not vp.exists():
            print(f"  qid={qid} MISSING {vp}")
            continue
        frames = extract_native_frames(vp, N_FRAMES)
        qid_dir = OUT_DIR / f"qid_{int(qid):03d}"
        qid_dir.mkdir(exist_ok=True, parents=True)
        prompt = build_prompt(entry)
        counts: dict[str, int] = {}
        per_frame: list[dict] = []
        for i, frame in enumerate(frames):
            frac = (i + 0.5) / N_FRAMES
            fp = qid_dir / f"frame_{i:02d}_{frac:.2f}.png"
            cv2.imwrite(str(fp), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            resp = run_vlm(fp, prompt)
            analysis = resp.get("analysis", "") if resp.get("ok") else f"ERR:{resp.get('error','')}"
            verdict = parse_yes_no(analysis)
            counts[verdict] = counts.get(verdict, 0) + 1
            per_frame.append({"frame": i, "frac": frac, "v": verdict, "raw": analysis[:180]})
        cls = classify(counts, N_FRAMES)
        results[qid] = {
            "gt": entry["gt"], "top": entry["top_letter"], "top_count": entry["top_count"],
            "counts": counts, "classification": cls, "per_frame": per_frame,
        }
        print(f"  qid={qid:>4s} cls={cls:>15s}  counts={counts}  gt={entry['gt']} top={entry['top_letter']}={entry['top_count']}/36")
        # Incremental save so any partial run is usable.
        (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))

    # Aggregate.
    by_cls: dict[str, list[str]] = {"object_absent": [], "object_rare": [], "object_present": []}
    for q, r in results.items():
        by_cls[r["classification"]].append(q)
    agg = {
        "n_total": len(results),
        "by_classification": {k: {"count": len(v), "qids": v} for k, v in by_cls.items()},
    }
    (OUT_DIR / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print("\n=== AGGREGATE ===")
    for k, v in by_cls.items():
        print(f"  {k:>16s}: {len(v)}/{len(results)}  qids={v}")


if __name__ == "__main__":
    main()
