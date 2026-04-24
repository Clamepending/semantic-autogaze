"""Cycle 1 — scan the 14 universal-success + 63 variable qids for object-absent signal.

Symmetry check to r/vlm-high-res-label-noise: is the 77.5 % object_absent rate
on the 40 convergent-wrong fail qids specific to the fail set, or a dataset-wide
property of HLVid household?

For each qid in succ_audit (14) + var_audit (63) = 77 total, sample 6 native-res
frames, call rv-browser describe-file with the YES/NO visibility prompt. Collect
yes/no across frames. Classify per qid same as fullres_scan:
  - object_absent (0/6 YES) → structurally ill-posed
  - object_rare (1-2/6 YES) → borderline
  - object_present (3+/6 YES) → answerable
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
OUT_DIR = Path("/home/ogata/semantic-autogaze/results/label_noise_audit/fullres_scan_supervar")
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


def build_prompt(stem: str) -> str:
    return (
        f"The question asks: \"{stem.strip()}\"\n\n"
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


def classify(yes_no_counts: dict[str, int]) -> str:
    yes = yes_no_counts.get("YES", 0)
    if yes == 0:
        return "object_absent"
    if yes <= 2:
        return "object_rare"
    return "object_present"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = json.loads(AUDIT_PATH.read_text())

    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["category"] == "household"].copy()
    df["question_id"] = df["question_id"].astype(int).astype(str)
    df = df.set_index("question_id")

    # Build combined task list with a 'set' label for later aggregation.
    tasks = []
    for entry in audit["succ_audit"]:
        tasks.append({"qid": entry["qid"], "gt": entry["gt"], "stem": entry["stem"], "set": "universal_success"})
    for entry in audit["var_audit"]:
        qid = entry["qid"]
        # var_audit lacks stem; pull from parquet and strip choices.
        q_full = str(df.loc[qid, "question"])
        stem = q_full.split("\n")[0].strip() if "\n" in q_full else q_full
        tasks.append({"qid": qid, "gt": entry["gt"], "stem": stem, "set": "variable"})

    print(f"loaded {len(tasks)} tasks: 14 succ + 63 var = 77 qids × {N_FRAMES} frames = {77*N_FRAMES} VLM calls")

    results: dict[str, dict] = {}
    for i, task in enumerate(tasks):
        qid = task["qid"]
        vp = VIDEO_DIR / df.loc[qid, "video_path"]
        if not vp.exists():
            print(f"  qid={qid} MISSING {vp}")
            continue
        frames = extract_native_frames(vp, N_FRAMES)
        qid_dir = OUT_DIR / f"qid_{int(qid):03d}"
        qid_dir.mkdir(exist_ok=True, parents=True)
        prompt = build_prompt(task["stem"])
        counts: dict[str, int] = {}
        per_frame: list[dict] = []
        for j, frame in enumerate(frames):
            frac = (j + 0.5) / N_FRAMES
            fp = qid_dir / f"frame_{j:02d}_{frac:.2f}.png"
            cv2.imwrite(str(fp), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            resp = run_vlm(fp, prompt)
            analysis = resp.get("analysis", "") if resp.get("ok") else f"ERR:{resp.get('error','')}"
            verdict = parse_yes_no(analysis)
            counts[verdict] = counts.get(verdict, 0) + 1
            per_frame.append({"frame": j, "frac": frac, "v": verdict, "raw": analysis[:180]})
        cls = classify(counts)
        results[qid] = {
            "gt": task["gt"], "set": task["set"], "stem": task["stem"][:120],
            "counts": counts, "classification": cls, "per_frame": per_frame,
        }
        print(f"  [{i+1:>3d}/{len(tasks)}] qid={qid:>4s} set={task['set']:>16s} cls={cls:>15s}  counts={counts}")
        (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))

    # Aggregate by set.
    agg: dict[str, dict] = {}
    for split in ["universal_success", "variable"]:
        by_cls: dict[str, list[str]] = {"object_absent": [], "object_rare": [], "object_present": []}
        for q, r in results.items():
            if r["set"] == split:
                by_cls[r["classification"]].append(q)
        n = sum(len(v) for v in by_cls.values())
        agg[split] = {
            "n_total": n,
            "by_classification": {k: {"count": len(v), "qids": v, "fraction": len(v)/n if n else 0.0} for k, v in by_cls.items()},
        }

    # Also overall.
    by_cls_all: dict[str, list[str]] = {"object_absent": [], "object_rare": [], "object_present": []}
    for q, r in results.items():
        by_cls_all[r["classification"]].append(q)
    agg["combined_77"] = {
        "n_total": len(results),
        "by_classification": {k: {"count": len(v), "qids": v, "fraction": len(v)/len(results) if results else 0.0} for k, v in by_cls_all.items()},
    }
    (OUT_DIR / "aggregate.json").write_text(json.dumps(agg, indent=2))

    print("\n=== AGGREGATE ===")
    for split, a in agg.items():
        print(f"\n  --- {split} (n={a['n_total']}) ---")
        for k, v in a["by_classification"].items():
            print(f"    {k:>16s}: {v['count']:>3d}/{a['n_total']}  ({100*v['fraction']:.1f}%)  qids={v['qids']}")


if __name__ == "__main__":
    main()
