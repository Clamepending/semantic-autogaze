"""Cycle 1 — apparatus-validity calibration at NVILA-matched temporal budget.

If the 6-frame VLM has a 64 % false-negative rate on universal-success
(r/vlm-audit-success-and-variable), does a 32-frame uniform scan (matching
NVILA's 32-frame video input) bring the succ object_absent rate toward 0 %?

For each of the 14 universal-success qids, sample 32 native-res frames
uniformly, call rv-browser describe-file with the SAME YES/NO prompt. Class:
  - object_absent (0/32 YES) → apparatus still misses even at NVILA budget
  - object_rare (1-5/32 YES) → borderline
  - object_present (6+/32 YES) → apparatus sees object at dense sampling

Decisive either way:
  - succ object_absent ≤ 15 % at 32 frames → apparatus IS trustworthy at
    NVILA budget; proceed to fail-40 rerun in cycle 2 for a clean estimate.
  - succ object_absent ≥ 50 % at 32 frames → dense temporal sampling is
    insufficient; small-text requires spatial cropping. Label-noise-via-VLM
    direction is closed.
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
OUT_DIR = Path("/home/ogata/semantic-autogaze/results/label_noise_audit/fullres_scan_densecal")
N_FRAMES = 32
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


def classify(yes_no_counts: dict[str, int], n: int) -> str:
    # At n=32: object_absent if 0 YES; object_rare if 1 ≤ yes ≤ 5; object_present if yes ≥ 6.
    yes = yes_no_counts.get("YES", 0)
    if yes == 0:
        return "object_absent"
    if yes <= 5:
        return "object_rare"
    return "object_present"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = json.loads(AUDIT_PATH.read_text())

    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["category"] == "household"].copy()
    df["question_id"] = df["question_id"].astype(int).astype(str)
    df = df.set_index("question_id")

    tasks = [{"qid": e["qid"], "gt": e["gt"], "stem": e["stem"]} for e in audit["succ_audit"]]
    print(f"dense-frame calibration: {len(tasks)} universal-success qids × {N_FRAMES} frames = {len(tasks)*N_FRAMES} VLM calls")

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
        cls = classify(counts, N_FRAMES)
        yes = counts.get("YES", 0)
        results[qid] = {
            "gt": task["gt"], "stem": task["stem"][:120],
            "counts": counts, "yes_count": yes, "classification": cls, "per_frame": per_frame,
        }
        print(f"  [{i+1:>2d}/{len(tasks)}] qid={qid:>4s} cls={cls:>15s}  yes={yes}/{N_FRAMES}  counts={counts}")
        (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))

    # Aggregate and compare to prior 6-frame result on same qids.
    by_cls: dict[str, list[str]] = {"object_absent": [], "object_rare": [], "object_present": []}
    for q, r in results.items():
        by_cls[r["classification"]].append(q)
    n = sum(len(v) for v in by_cls.values())
    agg = {
        "n_total": n,
        "n_frames": N_FRAMES,
        "by_classification": {k: {"count": len(v), "qids": v, "fraction": len(v)/n if n else 0.0} for k, v in by_cls.items()},
    }
    # Also compare against the 6-frame succ result.
    prior_6frame_path = Path("/home/ogata/semantic-autogaze/results/label_noise_audit/fullres_scan_supervar/summary.json")
    if prior_6frame_path.exists():
        prior = json.loads(prior_6frame_path.read_text())
        comparison = []
        for q in results:
            p = prior.get(q, {})
            if p.get("set") == "universal_success":
                comparison.append({
                    "qid": q,
                    "cls_6frame": p.get("classification"), "yes_6frame": p.get("counts", {}).get("YES", 0),
                    "cls_32frame": results[q]["classification"], "yes_32frame": results[q]["yes_count"],
                    "delta_yes": results[q]["yes_count"] - p.get("counts", {}).get("YES", 0),
                })
        agg["comparison_6vs32"] = comparison

    (OUT_DIR / "aggregate.json").write_text(json.dumps(agg, indent=2))

    print("\n=== AGGREGATE (dense-frame calibration, 32 frames) ===")
    for k, v in by_cls.items():
        print(f"  {k:>16s}: {len(v)}/{n}  ({100*len(v)/n:.1f}%)  qids={v}")

    print("\n=== 6-frame vs 32-frame comparison on universal-success ===")
    if "comparison_6vs32" in agg:
        for c in agg["comparison_6vs32"]:
            print(f"  qid={c['qid']:>4s}  6f:{c['cls_6frame']:>15s} yes={c['yes_6frame']}/6   |   32f:{c['cls_32frame']:>15s} yes={c['yes_32frame']}/32   Δ yes={c['delta_yes']:+d}")


if __name__ == "__main__":
    main()
