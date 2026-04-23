"""Cycle 1 — probe whether rv-browser describe-file with Claude as VLM can classify
a single convergent-wrong qid by extracting all frames at native resolution.

Picks qid=8 (ORGANIC vs VEGGIE) and qid=153 (5 vs 3 spikes) as probes.
For each:
  - extract ~20 frames at native resolution
  - save each as frame_<frac>.png
  - invoke rv-browser describe-file on each with a targeted prompt
  - collect 20 verdicts, majority-vote

If VLM returns a confident letter on most frames, the apparatus works
and we can scale to 40 qids. If UNKNOWN dominates, the approach fails.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PARQUET_PATH = "/home/ogata/.cache/huggingface/hub/datasets--bfshi--HLVid/snapshots/4956b601aec0bb9d455bb8f57552f78cbd3f0338/data/test-00000-of-00001.parquet"
VIDEO_DIR = Path("/home/ogata/semantic-autogaze/hlvid_videos/extracted_household/videos")
AUDIT_PATH = Path("/home/ogata/semantic-autogaze/results/label_noise_audit/audit.json")
OUT_DIR = Path("/home/ogata/semantic-autogaze/results/label_noise_audit/fullres_probe")
N_FRAMES = 20

PROBE_QIDS = ["8", "153", "257", "258"]


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
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            return {"ok": False, "error": r.stderr[:200]}
        return json.loads(r.stdout)
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def build_prompt(entry: dict) -> str:
    choices_str = "\n".join(f"  {L}. {entry['choices'].get(L, '')}" for L in "ABCD")
    stem = entry["stem"].strip()
    return (
        f"QUESTION: {stem}\n"
        f"CHOICES:\n{choices_str}\n\n"
        "Looking at this single video frame, can you see the object the question asks about? "
        "If yes and you can read/discern the answer, return just one letter A/B/C/D. "
        "If the object is visible but the answer is unclear, return UNKNOWN_UNCLEAR. "
        "If the object is not in this frame, return UNKNOWN_ABSENT. "
        "Start your response with the exact verdict token followed by a colon and one short sentence."
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = json.loads(AUDIT_PATH.read_text())
    by_qid = {e["qid"]: e for e in audit["convergent_wrong_candidates"]}

    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["category"] == "household"].copy()
    df["question_id"] = df["question_id"].astype(int).astype(str)
    df = df.set_index("question_id")

    results: dict[str, dict] = {}
    for qid in PROBE_QIDS:
        entry = by_qid[qid]
        vp = VIDEO_DIR / df.loc[qid, "video_path"]
        frames = extract_native_frames(vp, N_FRAMES)
        qid_dir = OUT_DIR / f"qid_{int(qid):03d}"
        qid_dir.mkdir(exist_ok=True, parents=True)
        prompt = build_prompt(entry)
        verdicts: list[str] = []
        per_frame: list[dict] = []
        for i, frame in enumerate(frames):
            frac = (i + 0.5) / N_FRAMES
            fp = qid_dir / f"frame_{i:02d}_{frac:.2f}.png"
            cv2.imwrite(str(fp), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            resp = run_vlm(fp, prompt)
            analysis = resp.get("analysis", "") if resp.get("ok") else f"ERR:{resp.get('error','')}"
            # parse first token (verdict)
            first = analysis.strip().split(":", 1)[0].strip().split()[0] if analysis.strip() else ""
            first = first.upper().strip().rstrip(".").rstrip(",")
            verdicts.append(first)
            per_frame.append({"frame": i, "frac": frac, "first": first, "raw": analysis[:200]})
            print(f"  qid={qid} f{i:02d}: {first}  |  {analysis[:120]}")
        vc = {v: verdicts.count(v) for v in set(verdicts)}
        results[qid] = {
            "gt": entry["gt"], "top": entry["top_letter"],
            "verdicts": verdicts, "counts": vc, "per_frame": per_frame,
        }
        print(f"\nqid={qid}: counts={vc}\n")

    (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
