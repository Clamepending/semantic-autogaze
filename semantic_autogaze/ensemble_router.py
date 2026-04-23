"""Per-question config router bootstrap.

Uses results/cross_move_matrix/matrix.json as labels (36 configs × 122 qids).
Features: video id (21 unique videos, one-hot), question-length, choice-length,
TF-IDF on question stem. Trains a leave-one-out (LOO) logistic-regression
router: for each held-out question, train per-config binary classifiers on the
other 121 questions, pick argmax config, tally correct.

Baselines:
  - random config (mean over the 36 configs)
  - best fixed config (max_c per_config_correct)
  - oracle (at-least-once-correct)

Outputs:
  results/ensemble_router/analysis.txt  (human readable)
  results/ensemble_router/router.json   (machine readable)
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


MATRIX_PATH = "results/cross_move_matrix/matrix.json"
PARQUET_PATH = "/home/ogata/.cache/huggingface/hub/datasets--bfshi--HLVid/snapshots/4956b601aec0bb9d455bb8f57552f78cbd3f0338/data/test-00000-of-00001.parquet"


def parse_stem(q_text: str) -> str:
    """Strip the trailing 'A./B./C./D./Please answer...' from a HLVid question."""
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


def main() -> None:
    # --- Load matrix.json (labels) ---
    matrix = json.loads(Path(MATRIX_PATH).read_text())
    n_q = matrix["n_questions"]
    configs = sorted(matrix["per_config_correct"].keys())
    n_c = len(configs)
    print(f"[load] matrix: {n_q} questions, {n_c} configs")

    # Need per-(qid, config) correctness. matrix.json has per_q_correct_count
    # but not the full matrix — reload directly from the sources.
    from semantic_autogaze.cross_move_matrix import SOURCES, load_rows, row_config_key

    correct: dict[str, dict[str, int]] = defaultdict(dict)
    for ref, path, tag in SOURCES:
        rows = load_rows(ref, path)
        for row in rows:
            qid = str(row["question_id"])
            cfg = row_config_key(row, tag)
            correct[qid][cfg] = 1 if row.get("correct") else 0

    qids = sorted(correct.keys(), key=int)
    Y = np.array([[correct[q].get(c, 0) for c in configs] for q in qids], dtype=np.int32)
    print(f"[load] Y matrix shape: {Y.shape} (qids × configs)")

    # --- Load HLVid parquet (features) ---
    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["category"] == "household"].copy()
    df["question_id"] = df["question_id"].astype(int).astype(str)
    df = df.set_index("question_id")
    # Keep only qids that appear in the matrix (sanity).
    df = df.loc[[q for q in qids if q in df.index]]
    assert len(df) == len(qids), f"coverage mismatch: df={len(df)} qids={len(qids)}"
    print(f"[load] HLVid household rows: {len(df)}")

    # --- Features ---
    stems = [parse_stem(q) for q in df["question"].tolist()]
    video_ids = df["video_path"].tolist()
    n_videos = len(set(video_ids))
    print(f"[features] unique videos: {n_videos}")

    # Video one-hot.
    ohe = OneHotEncoder(sparse_output=False)
    X_video = ohe.fit_transform(np.array(video_ids).reshape(-1, 1))
    # Question length features.
    q_len = np.array([[len(s)] for s in stems], dtype=np.float32)
    choice_lens = np.array([
        [max(len(c) for c in parse_choices(q) or [""]),
         np.mean([len(c) for c in parse_choices(q) or [""]])]
        for q in df["question"].tolist()
    ], dtype=np.float32)
    # TF-IDF on stems.
    tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2), stop_words="english")
    X_tfidf = tfidf.fit_transform(stems).toarray().astype(np.float32)

    X = np.hstack([X_video, q_len, choice_lens, X_tfidf])
    print(f"[features] X shape: {X.shape} "
          f"(video={X_video.shape[1]}, qlen=1, choice=2, tfidf={X_tfidf.shape[1]})")

    # --- Baselines ---
    per_config_correct = Y.sum(axis=0)  # shape (n_c,)
    best_fixed = int(per_config_correct.max())
    best_fixed_c = configs[int(per_config_correct.argmax())]
    random_mean = float(per_config_correct.mean())
    oracle = int((Y.sum(axis=1) >= 1).sum())
    print(f"[baselines] random_mean={random_mean:.2f} best_fixed={best_fixed} ({best_fixed_c})  oracle={oracle}")

    # --- LOO router ---
    # For each held-out qid i, train per-config LR on the other 121; for the
    # held-out question, predict p(correct | X_i) for each config and pick
    # argmax. Tally whether the chosen config is actually correct.
    N = X.shape[0]
    chosen = np.zeros(N, dtype=np.int32)
    chosen_correct = np.zeros(N, dtype=np.int32)
    scores_by_config = np.zeros((N, n_c), dtype=np.float32)

    print(f"[loo] training {N} * {n_c} = {N * n_c} LR models... (cheap, ~1-2 min)")
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        Xtr = X[mask]
        Ytr = Y[mask]
        Xte = X[i:i + 1]

        for j, cfg in enumerate(configs):
            y = Ytr[:, j]
            if y.sum() == 0 or y.sum() == len(y):
                # Degenerate: all-0 or all-1 training labels → constant predictor.
                p = float(y.mean())
            else:
                lr = LogisticRegression(max_iter=200, C=1.0)
                lr.fit(Xtr, y)
                p = float(lr.predict_proba(Xte)[0, 1])
            scores_by_config[i, j] = p

        best_j = int(scores_by_config[i].argmax())
        chosen[i] = best_j
        chosen_correct[i] = Y[i, best_j]

    router_correct = int(chosen_correct.sum())

    # --- Per-video breakdown ---
    per_video_stats: dict[str, dict] = {}
    for v in sorted(set(video_ids)):
        v_mask = np.array([vid == v for vid in video_ids])
        n_v = int(v_mask.sum())
        v_oracle = int((Y[v_mask].sum(axis=1) >= 1).sum())
        v_router = int(chosen_correct[v_mask].sum())
        v_best_fixed = int(Y[v_mask].sum(axis=0).max())
        per_video_stats[v] = {
            "n": n_v,
            "oracle": v_oracle,
            "router": v_router,
            "best_fixed": v_best_fixed,
        }

    # --- Per-config selection frequency ---
    chosen_config_hist: dict[str, int] = defaultdict(int)
    for c_idx in chosen:
        chosen_config_hist[configs[int(c_idx)]] += 1

    # --- Router beats best-fixed on how many questions? ---
    wrong_best_right_router = 0  # router picked correctly on qs best-fixed misses
    right_best_wrong_router = 0  # router picked wrong where best-fixed gets it
    best_fixed_j = int(per_config_correct.argmax())
    for i in range(N):
        rc = chosen_correct[i]
        bc = Y[i, best_fixed_j]
        if rc and not bc:
            wrong_best_right_router += 1
        if bc and not rc:
            right_best_wrong_router += 1

    # --- Output ---
    lines: list[str] = []

    def w(s: str = "") -> None:
        print(s)
        lines.append(s)

    w("=" * 72)
    w("ensemble_router :: leave-one-out router bootstrap")
    w("=" * 72)
    w(f"matrix: {n_q} questions × {n_c} configs")
    w(f"features: video-one-hot ({X_video.shape[1]}) + qlen (1) + choice-len (2) + TF-IDF ({X_tfidf.shape[1]}) = {X.shape[1]} dims")
    w("")
    w("-- Baselines --")
    w(f"  random config (mean)    : {random_mean:.2f} / {N}")
    w(f"  best fixed config       : {best_fixed} / {N}  [{best_fixed_c}]")
    w(f"  oracle (any config right): {oracle} / {N}")
    w(f"  individual-vs-oracle gap : {oracle - best_fixed} points")
    w("")
    w("-- LOO router --")
    w(f"  router correct          : {router_correct} / {N}")
    w(f"  vs best fixed           : {router_correct - best_fixed:+d}")
    w(f"  vs oracle (% gap closed): {(router_correct - best_fixed) / max(1, oracle - best_fixed):.2%}")
    w(f"  qs router>best-fixed    : +{wrong_best_right_router}")
    w(f"  qs best-fixed>router    : -{right_best_wrong_router}")
    w(f"  net                     : {wrong_best_right_router - right_best_wrong_router:+d}")
    w("")
    w("-- Router's chosen-config distribution --")
    for c, n in sorted(chosen_config_hist.items(), key=lambda kv: -kv[1]):
        if n:
            w(f"  {n:3d}  {c}")
    w("")
    w("-- Per-video router score --")
    w(f"  {'video':<54} {'n':>3} {'best':>5} {'router':>7} {'oracle':>7}")
    for v in sorted(per_video_stats):
        s = per_video_stats[v]
        w(f"  {v[:54]:<54} {s['n']:>3} {s['best_fixed']:>5} {s['router']:>7} {s['oracle']:>7}")

    out_dir = Path("results/ensemble_router")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "router.json").write_text(json.dumps({
        "n_questions": N,
        "n_configs": n_c,
        "features": {"video_onehot": X_video.shape[1], "tfidf": X_tfidf.shape[1], "total_dim": X.shape[1]},
        "baselines": {
            "random_mean": random_mean,
            "best_fixed": best_fixed,
            "best_fixed_config": best_fixed_c,
            "oracle": oracle,
        },
        "router": {
            "correct": router_correct,
            "vs_best_fixed": router_correct - best_fixed,
            "pct_gap_closed": (router_correct - best_fixed) / max(1, oracle - best_fixed),
            "qs_router_beats_best": wrong_best_right_router,
            "qs_best_beats_router": right_best_wrong_router,
        },
        "chosen_config_hist": dict(chosen_config_hist),
        "per_video": per_video_stats,
    }, indent=2))


if __name__ == "__main__":
    main()
