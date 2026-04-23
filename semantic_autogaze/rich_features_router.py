"""Rich-features router: CLIP-L/14 text embeddings replacing TF-IDF.

Follow-up to ensemble_router.py (r/ensemble-router-bootstrap). Same 36-config
× 122-question labels, same LOO framework, same video one-hot + length
features — but question text is encoded with CLIP-L/14's text encoder
(768-dim) instead of TF-IDF (200-dim).

Outputs:
  results/rich_features_router/analysis.txt
  results/rich_features_router/router.json
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from transformers import CLIPTextModel, CLIPTokenizer

MATRIX_PATH = "results/cross_move_matrix/matrix.json"
PARQUET_PATH = "/home/ogata/.cache/huggingface/hub/datasets--bfshi--HLVid/snapshots/4956b601aec0bb9d455bb8f57552f78cbd3f0338/data/test-00000-of-00001.parquet"
CLIP_MODEL = "openai/clip-vit-large-patch14"


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


def clip_encode(texts: list[str]) -> np.ndarray:
    """Return pooled mean-token embeddings (per-text EOS-position embedding).

    CLIP's text encoder outputs per-token last_hidden_state; the standard
    pooled output is taken at the EOS position. We use that — it's what CLIP
    contrastive training aligned with image embeddings.
    """
    device = "cpu"
    tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    mdl = CLIPTextModel.from_pretrained(CLIP_MODEL).to(device).eval()

    embs = []
    with torch.no_grad():
        for t in texts:
            inputs = tok([t], return_tensors="pt", truncation=True, max_length=77, padding=True).to(device)
            out = mdl(**inputs)
            # CLIPTextModel returns pooler_output (EOS-pooled, 768-dim for ViT-L).
            embs.append(out.pooler_output[0].cpu().numpy())
    return np.stack(embs, axis=0).astype(np.float32)


def main() -> None:
    # --- Load matrix (labels) ---
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
    print(f"[load] Y matrix: {Y.shape}  ({len(qids)} qids × {len(configs)} configs)")

    # --- Load HLVid parquet ---
    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["category"] == "household"].copy()
    df["question_id"] = df["question_id"].astype(int).astype(str)
    df = df.set_index("question_id")
    df = df.loc[qids]
    print(f"[load] HLVid household rows: {len(df)}")

    stems = [parse_stem(q) for q in df["question"].tolist()]
    video_ids = df["video_path"].tolist()

    # --- Video one-hot ---
    ohe = OneHotEncoder(sparse_output=False)
    X_video = ohe.fit_transform(np.array(video_ids).reshape(-1, 1))

    # --- Length features ---
    q_len = np.array([[len(s)] for s in stems], dtype=np.float32)
    choice_lens = np.array([
        [max(len(c) for c in parse_choices(q) or [""]),
         np.mean([len(c) for c in parse_choices(q) or [""]])]
        for q in df["question"].tolist()
    ], dtype=np.float32)

    # --- CLIP-L text embeddings ---
    print(f"[clip] encoding {len(stems)} stems with {CLIP_MODEL}...")
    X_clip = clip_encode(stems)  # shape (122, 768)
    print(f"[clip] done, shape {X_clip.shape}, norm-mean={np.linalg.norm(X_clip, axis=1).mean():.2f}")

    # L2-normalize CLIP embeddings so LR sees unit vectors (matches CLIP contrastive).
    X_clip = X_clip / (np.linalg.norm(X_clip, axis=1, keepdims=True) + 1e-8)

    X = np.hstack([X_video, q_len, choice_lens, X_clip])
    print(f"[features] X shape: {X.shape} "
          f"(video={X_video.shape[1]}, qlen=1, choice=2, clip={X_clip.shape[1]})")

    # --- Baselines ---
    per_config_correct = Y.sum(axis=0)
    best_fixed = int(per_config_correct.max())
    best_fixed_c = configs[int(per_config_correct.argmax())]
    random_mean = float(per_config_correct.mean())
    oracle = int((Y.sum(axis=1) >= 1).sum())
    N = X.shape[0]
    print(f"[baselines] random={random_mean:.2f} best_fixed={best_fixed}  oracle={oracle}")

    # --- LOO router ---
    chosen = np.zeros(N, dtype=np.int32)
    chosen_correct = np.zeros(N, dtype=np.int32)
    scores_by_config = np.zeros((N, len(configs)), dtype=np.float32)

    print(f"[loo] training {N} * {len(configs)} = {N * len(configs)} LR models...")
    for i in range(N):
        mask = np.ones(N, dtype=bool); mask[i] = False
        Xtr, Ytr, Xte = X[mask], Y[mask], X[i:i+1]
        for j in range(len(configs)):
            y = Ytr[:, j]
            if y.sum() == 0 or y.sum() == len(y):
                p = float(y.mean())
            else:
                lr = LogisticRegression(max_iter=500, C=0.3)  # stronger regularization for 790-dim
                lr.fit(Xtr, y)
                p = float(lr.predict_proba(Xte)[0, 1])
            scores_by_config[i, j] = p
        best_j = int(scores_by_config[i].argmax())
        chosen[i] = best_j
        chosen_correct[i] = Y[i, best_j]

    router_correct = int(chosen_correct.sum())

    # Paired-flip vs best-fixed.
    best_fixed_j = int(per_config_correct.argmax())
    wrong_best_right_router = 0
    right_best_wrong_router = 0
    for i in range(N):
        rc = chosen_correct[i]
        bc = Y[i, best_fixed_j]
        if rc and not bc:
            wrong_best_right_router += 1
        if bc and not rc:
            right_best_wrong_router += 1

    # Chosen-config histogram.
    chosen_hist: dict[str, int] = defaultdict(int)
    for c_idx in chosen:
        chosen_hist[configs[int(c_idx)]] += 1

    # --- Video-conditional router (for ablation comparison to cheap router) ---
    vid_chosen_correct = np.zeros(N, dtype=np.int32)
    for i in range(N):
        v = video_ids[i]
        peer_mask = np.array([vid == v for vid in video_ids])
        peer_mask[i] = False
        if peer_mask.sum() == 0:
            chosen_j = int(per_config_correct.argmax())
        else:
            chosen_j = int(Y[peer_mask].sum(axis=0).argmax())
        vid_chosen_correct[i] = Y[i, chosen_j]
    vid_router_correct = int(vid_chosen_correct.sum())

    # --- Output ---
    lines: list[str] = []

    def w(s: str = "") -> None:
        print(s); lines.append(s)

    w("=" * 72)
    w("rich_features_router :: CLIP-L text + video + lengths -> LOO per-config LR")
    w("=" * 72)
    w(f"matrix: {N} questions × {len(configs)} configs")
    w(f"features: video-onehot ({X_video.shape[1]}) + qlen (1) + choice-len (2) + CLIP-L ({X_clip.shape[1]}) = {X.shape[1]} dims")
    w(f"CLIP model: {CLIP_MODEL}")
    w("")
    w("-- Baselines --")
    w(f"  random config (mean)     : {random_mean:.2f} / {N}")
    w(f"  best fixed config        : {best_fixed} / {N}  [{best_fixed_c}]")
    w(f"  oracle                   : {oracle} / {N}")
    w(f"  cheap-router (prior)     : 47 / {N}")
    w(f"  video-cond router (prior): 42 / {N}")
    w(f"  individual-vs-oracle gap : {oracle - best_fixed}")
    w("")
    w("-- Rich-features LOO router --")
    w(f"  router correct           : {router_correct} / {N}")
    w(f"  vs best fixed            : {router_correct - best_fixed:+d}")
    w(f"  vs oracle (% gap closed) : {(router_correct - best_fixed) / max(1, oracle - best_fixed):.2%}")
    w(f"  vs cheap-router          : {router_correct - 47:+d}")
    w(f"  qs router>best-fixed     : +{wrong_best_right_router}")
    w(f"  qs best-fixed>router     : -{right_best_wrong_router}")
    w(f"  net                      : {wrong_best_right_router - right_best_wrong_router:+d}")
    w("")
    w("-- Video-conditional router (reproduction of cheap variant) --")
    w(f"  router correct           : {vid_router_correct} / {N}")
    w("")
    w("-- Router chosen-config histogram --")
    for c, n in sorted(chosen_hist.items(), key=lambda kv: -kv[1]):
        if n:
            w(f"  {n:3d}  {c}")

    out = Path("results/rich_features_router")
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.txt").write_text("\n".join(lines) + "\n")
    (out / "router.json").write_text(json.dumps({
        "n_questions": N,
        "n_configs": len(configs),
        "features": {"video": int(X_video.shape[1]), "clip": int(X_clip.shape[1]), "total": int(X.shape[1])},
        "baselines": {
            "random_mean": random_mean,
            "best_fixed": best_fixed,
            "best_fixed_config": best_fixed_c,
            "oracle": oracle,
            "cheap_router_prior": 47,
            "video_cond_router_prior": 42,
        },
        "rich_router": {
            "correct": router_correct,
            "vs_best_fixed": router_correct - best_fixed,
            "vs_cheap_router": router_correct - 47,
            "pct_gap_closed": (router_correct - best_fixed) / max(1, oracle - best_fixed),
            "qs_router_beats_best": wrong_best_right_router,
            "qs_best_beats_router": right_best_wrong_router,
        },
        "video_cond_router": {
            "correct": vid_router_correct,
            "vs_best_fixed": vid_router_correct - best_fixed,
        },
        "chosen_config_hist": dict(chosen_hist),
    }, indent=2))


if __name__ == "__main__":
    main()
