"""Cycle 1 — extract 4-frame montages for each convergent-wrong qid for agent visual inspection.

Reads the audit.json from r/label-noise-audit. For each convergent_wrong_candidate:
  - load video via HLVid parquet's video_path
  - sample 4 frames (0.1/0.35/0.65/0.9 duration)
  - build 2x2 montage at high resolution
  - overlay qid, question stem, all 4 choices with gt/majority markers, vote counts
  - save to results/label_noise_audit/inspection/qid_<NNN>.png

Agent then reads each image and classifies as label_noise / pipeline_bias / ambiguous.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PARQUET_PATH = "/home/ogata/.cache/huggingface/hub/datasets--bfshi--HLVid/snapshots/4956b601aec0bb9d455bb8f57552f78cbd3f0338/data/test-00000-of-00001.parquet"
VIDEO_DIR = Path("/home/ogata/semantic-autogaze/hlvid_videos/extracted_household/videos")
AUDIT_PATH = Path("/home/ogata/semantic-autogaze/results/label_noise_audit/audit.json")
OUT_DIR = Path("/home/ogata/semantic-autogaze/results/label_noise_audit/inspection")
FRAME_FRACTIONS = (0.1, 0.35, 0.65, 0.9)
TARGET_FRAME_W = 640
TARGET_FRAME_H = 360


def sample_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    frames = []
    for f in FRAME_FRACTIONS:
        idx = int(f * total)
        idx = max(0, min(total - 1, idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if ok and bgr is not None:
            frames.append(bgr)
    cap.release()
    return frames


def resize_to(bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)


def draw_text(img: np.ndarray, text: str, org: tuple[int, int],
              scale: float = 0.55, color=(255, 255, 255), thick: int = 1,
              bg=(0, 0, 0)) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + 4), bg, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def build_montage(frames: list[np.ndarray], entry: dict) -> np.ndarray:
    # 2x2 grid of frames (4*TARGET_FRAME_W x 1*TARGET_FRAME_H or 2x2 layout).
    # Use 2x2 layout: grid is (2*TARGET_FRAME_W) x (2*TARGET_FRAME_H).
    # Above grid: caption strip of height 260.
    cap_h = 280
    grid_w = 2 * TARGET_FRAME_W
    grid_h = 2 * TARGET_FRAME_H
    total_w = grid_w
    total_h = cap_h + grid_h
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:cap_h, :] = (30, 30, 30)

    # Place frames or black if missing.
    resized = []
    for i in range(4):
        if i < len(frames):
            resized.append(resize_to(frames[i], TARGET_FRAME_W, TARGET_FRAME_H))
        else:
            resized.append(np.zeros((TARGET_FRAME_H, TARGET_FRAME_W, 3), dtype=np.uint8))

    # Layout: tl, tr, bl, br for fracs 0.1, 0.35, 0.65, 0.9.
    positions = [(0, 0), (TARGET_FRAME_W, 0), (0, TARGET_FRAME_H), (TARGET_FRAME_W, TARGET_FRAME_H)]
    for (x, y), img, frac in zip(positions, resized, FRAME_FRACTIONS):
        canvas[cap_h + y:cap_h + y + TARGET_FRAME_H, x:x + TARGET_FRAME_W] = img
        draw_text(canvas, f"t={frac:.2f}", (x + 8, cap_h + y + 24))

    # Caption overlay.
    draw_text(canvas, f"qid={entry['qid']}   gt={entry['gt']}   top={entry['top_letter']}={entry['top_count']}/36  ent={entry['entropy']:.2f}",
              (12, 28), scale=0.7, thick=2)
    stem_full = entry.get("stem", "").strip()
    # wrap stem to ~80 chars/line
    stem_lines = []
    words = stem_full.split()
    cur = ""
    for w in words:
        if len(cur) + 1 + len(w) > 85:
            stem_lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        stem_lines.append(cur)
    for i, ln in enumerate(stem_lines[:2]):
        draw_text(canvas, f"Q: {ln}" if i == 0 else f"   {ln}",
                  (12, 60 + i * 22), scale=0.55)

    # Choices.
    dist = entry.get("dist", {})
    for i, L in enumerate("ABCD"):
        ch = entry.get("choices", {}).get(L, "")
        v = dist.get(L, 0)
        marker = " * gt     " if L == entry["gt"] else (" ! top    " if L == entry["top_letter"] else "         ")
        color = (120, 255, 120) if L == entry["gt"] else ((120, 120, 255) if L == entry["top_letter"] else (200, 200, 200))
        txt = f"{marker} {L}. {ch[:70]}  ({v} votes)"
        draw_text(canvas, txt, (12, 110 + i * 22), scale=0.5, color=color)

    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = json.loads(AUDIT_PATH.read_text())
    candidates = audit["convergent_wrong_candidates"]
    print(f"  {len(candidates)} convergent-wrong qids to inspect")

    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["category"] == "household"].copy()
    df["question_id"] = df["question_id"].astype(int).astype(str)
    df = df.set_index("question_id")

    ok = 0
    missing = 0
    for entry in candidates:
        qid = entry["qid"]
        row = df.loc[qid]
        vp = VIDEO_DIR / row["video_path"]
        if not vp.exists():
            print(f"  qid={qid} MISSING {vp}")
            missing += 1
            continue
        frames = sample_frames(vp)
        canvas = build_montage(frames, entry)
        out_path = OUT_DIR / f"qid_{int(qid):03d}.png"
        cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        ok += 1
    print(f"  wrote {ok} montages, {missing} missing videos -> {OUT_DIR}")


if __name__ == "__main__":
    main()
