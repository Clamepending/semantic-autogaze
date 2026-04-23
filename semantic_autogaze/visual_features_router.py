"""Visual-features router: CLIP-L/14 IMAGE embeddings replace video one-hot.

Follow-up to rich_features_router.py (r/rich-features-router). Same 36-config
x 122-question labels, same LOO framework, same CLIP-L stem text embeddings
(cycle 2 winner = 792-dim stem-only router at 51/122). The one change: replace
the 21-dim video one-hot with 768-dim per-video CLIP-L IMAGE embeddings from
sampled frames.

Features per question:
  - CLIP-L image embedding of the question's video (768)
  - CLIP-L text stem (768)
  - Length features: qlen + max-choice-len + mean-choice-len (3)
  -> 1539 total dims

Prior: 30 % beats best-fixed 51.

Outputs:
  results/visual_features_router/analysis.txt
  results/visual_features_router/router.json
  results/visual_features_router/video_embs.npy  (cache for reruns)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModel

PARQUET_PATH = "/home/ogata/.cache/huggingface/hub/datasets--bfshi--HLVid/snapshots/4956b601aec0bb9d455bb8f57552f78cbd3f0338/data/test-00000-of-00001.parquet"
CLIP_MODEL = "openai/clip-vit-large-patch14"
VIDEO_DIR = Path("/home/ogata/semantic-autogaze/hlvid_videos/extracted_household/videos")
FRAME_FRACTIONS = (0.25, 0.75)  # 2 frames per video


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


def parse_choices(q_text: str) -> list[str]:
    choices = []
    for ln in q_text.split("\n"):
        m = re.match(r"^[A-D]\.\s*(.+)$", ln.strip())
        if m:
            choices.append(m.group(1))
    return choices


def sample_frames(video_path: Path, fractions: tuple[float, ...] = FRAME_FRACTIONS) -> list[np.ndarray]:
    """Read 1+ frames at given duration-fractions. Returns RGB HxWx3 uint8 arrays."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for frac in fractions:
        target = max(0, min(n - 1, int(frac * n)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return frames


def clip_text_encode(texts: list[str]) -> np.ndarray:
    tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    mdl = CLIPTextModel.from_pretrained(CLIP_MODEL).eval()
    embs = []
    with torch.no_grad():
        for t in texts:
            inputs = tok([t], return_tensors="pt", truncation=True, max_length=77, padding=True)
            out = mdl(**inputs)
            embs.append(out.pooler_output[0].cpu().numpy())
    return np.stack(embs, axis=0).astype(np.float32)


def clip_image_encode_videos(video_map: dict[str, Path]) -> dict[str, np.ndarray]:
    """Encode 1-2 frames per video, mean-pool into per-video 768-dim embedding."""
    proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
    mdl = CLIPVisionModel.from_pretrained(CLIP_MODEL).eval()
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for vid, path in video_map.items():
            frames = sample_frames(path)
            batch = proc(images=frames, return_tensors="pt")
            vout = mdl(**batch)
            # pooler_output is [len(frames), 768]; mean-pool across frames
            emb = vout.pooler_output.mean(dim=0).cpu().numpy()
            out[vid] = emb.astype(np.float32)
            print(f"  [img] {vid}: {len(frames)} frames -> norm {np.linalg.norm(emb):.2f}")
    return out


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
    print(f"[load] Y matrix: {Y.shape}  ({len(qids)} qids x {len(configs)} configs)")

    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["category"] == "household"].copy()
    df["question_id"] = df["question_id"].astype(int).astype(str)
    df = df.set_index("question_id")
    df = df.loc[qids]
    print(f"[load] HLVid household rows: {len(df)}")

    stems = [parse_stem(q) for q in df["question"].tolist()]
    video_ids = df["video_path"].tolist()

    # Length features
    q_len = np.array([[len(s)] for s in stems], dtype=np.float32)
    choice_lens = np.array([
        [max(len(c) for c in parse_choices(q) or [""]),
         np.mean([len(c) for c in parse_choices(q) or [""]])]
        for q in df["question"].tolist()
    ], dtype=np.float32)

    # --- CLIP-L image embeddings per unique video ---
    unique_vids = sorted(set(video_ids))
    print(f"[video] {len(unique_vids)} unique videos")
    video_map: dict[str, Path] = {}
    for vid in unique_vids:
        p = VIDEO_DIR / vid
        if not p.exists():
            raise FileNotFoundError(f"Missing video: {p}")
        video_map[vid] = p

    out_dir = Path("results/visual_features_router")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "video_embs.npz"

    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        vid_emb: dict[str, np.ndarray] = {k: z[k] for k in z.files}
        print(f"[video] loaded cached embeddings for {len(vid_emb)} videos")
    else:
        print(f"[video] encoding {len(unique_vids)} videos with CLIP-L image encoder...")
        vid_emb = clip_image_encode_videos(video_map)
        np.savez(cache_path, **vid_emb)
        print(f"[video] cached to {cache_path}")

    X_img = np.stack([vid_emb[v] for v in video_ids], axis=0)
    X_img = X_img / (np.linalg.norm(X_img, axis=1, keepdims=True) + 1e-8)
    print(f"[features] X_img shape: {X_img.shape}")

    # --- CLIP-L text stem embeddings (cycle-2 winner from rich-features) ---
    print(f"[clip-text] encoding {len(stems)} stems...")
    X_txt = clip_text_encode(stems)
    X_txt = X_txt / (np.linalg.norm(X_txt, axis=1, keepdims=True) + 1e-8)

    # --- Feature matrices ---
    X_visual_only = np.hstack([X_img, q_len, choice_lens])  # (122, 772)
    X_combined = np.hstack([X_img, X_txt, q_len, choice_lens])  # (122, 1539)

    per_config_correct = Y.sum(axis=0)
    best_fixed = int(per_config_correct.max())
    best_fixed_c = configs[int(per_config_correct.argmax())]
    best_fixed_j = int(per_config_correct.argmax())
    random_mean = float(per_config_correct.mean())
    oracle = int((Y.sum(axis=1) >= 1).sum())
    N = Y.shape[0]
    print(f"[baselines] random={random_mean:.2f} best_fixed={best_fixed}  oracle={oracle}")

    def run_loo_router(X: np.ndarray, label: str, C: float) -> dict:
        chosen_correct = np.zeros(N, dtype=np.int32)
        chosen = np.zeros(N, dtype=np.int32)
        for i in range(N):
            mask = np.ones(N, dtype=bool); mask[i] = False
            Xtr, Ytr, Xte = X[mask], Y[mask], X[i:i+1]
            scores = np.zeros(len(configs), dtype=np.float32)
            for j in range(len(configs)):
                y = Ytr[:, j]
                if y.sum() == 0 or y.sum() == len(y):
                    scores[j] = float(y.mean())
                else:
                    lr = LogisticRegression(max_iter=500, C=C)
                    lr.fit(Xtr, y)
                    scores[j] = float(lr.predict_proba(Xte)[0, 1])
            best_j = int(scores.argmax())
            chosen[i] = best_j
            chosen_correct[i] = Y[i, best_j]
        router_correct = int(chosen_correct.sum())
        wins = 0; losses = 0
        for i in range(N):
            if chosen_correct[i] and not Y[i, best_fixed_j]:
                wins += 1
            if Y[i, best_fixed_j] and not chosen_correct[i]:
                losses += 1
        hist: dict[str, int] = defaultdict(int)
        for c_idx in chosen:
            hist[configs[int(c_idx)]] += 1
        print(f"[loo] {label}: {router_correct}/{N}  vs best {router_correct - best_fixed:+d}  "
              f"(+{wins}/-{losses} net {wins - losses:+d})")
        return {
            "label": label,
            "features_dim": int(X.shape[1]),
            "C": C,
            "correct": router_correct,
            "vs_best_fixed": router_correct - best_fixed,
            "paired_wins": wins,
            "paired_losses": losses,
            "net": wins - losses,
            "chosen_hist": dict(hist),
        }

    print(f"\n[loo] training routers: {N} * {len(configs)} = {N * len(configs)} LR models per variant")
    res_visual = run_loo_router(X_visual_only, "visual-only (img+len)", C=0.3)
    res_combined = run_loo_router(X_combined, "visual+text (img+stem+len)", C=0.3)

    lines: list[str] = []
    def w(s: str = "") -> None:
        print(s); lines.append(s)

    w("=" * 72)
    w("visual_features_router :: CLIP-L IMG + CLIP-L TEXT stem -> LOO per-config LR")
    w("=" * 72)
    w(f"matrix: {N} questions x {len(configs)} configs")
    w(f"video features: CLIP-L image ({X_img.shape[1]}) per video at frame-fractions {FRAME_FRACTIONS}")
    w(f"text features: CLIP-L text stem ({X_txt.shape[1]})")
    w(f"length features: qlen (1) + choice-len (2) = 3")
    w("")
    w("-- Baselines --")
    w(f"  random config (mean)      : {random_mean:.2f} / {N}")
    w(f"  best fixed config         : {best_fixed} / {N}  [{best_fixed_c}]")
    w(f"  oracle                    : {oracle} / {N}")
    w(f"  cheap-router (prior)      : 47 / {N}")
    w(f"  rich-router stem (prior)  : 51 / {N}  [= best-fixed parity]")
    w(f"  rich-router combo (prior) : 47 / {N}")
    w(f"  individual-vs-oracle gap  : {oracle - best_fixed}")
    w("")
    for res in (res_visual, res_combined):
        w(f"-- LOO router: {res['label']} --")
        w(f"  feature dim         : {res['features_dim']}")
        w(f"  router correct      : {res['correct']} / {N}")
        w(f"  vs best fixed       : {res['vs_best_fixed']:+d}")
        w(f"  paired +/-          : +{res['paired_wins']}  -{res['paired_losses']}  net {res['net']:+d}")
        w(f"  top picks:")
        for c, n in sorted(res["chosen_hist"].items(), key=lambda kv: -kv[1])[:10]:
            if n:
                w(f"    {n:3d}  {c}")
        w("")

    (out_dir / "analysis.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "router.json").write_text(json.dumps({
        "n_questions": N,
        "n_configs": len(configs),
        "baselines": {
            "random_mean": random_mean,
            "best_fixed": best_fixed,
            "best_fixed_config": best_fixed_c,
            "oracle": oracle,
            "cheap_router_prior": 47,
            "rich_router_stem_prior": 51,
            "rich_router_combo_prior": 47,
        },
        "visual_only": res_visual,
        "visual_plus_text": res_combined,
        "frame_fractions": list(FRAME_FRACTIONS),
        "clip_model": CLIP_MODEL,
    }, indent=2))


if __name__ == "__main__":
    main()
