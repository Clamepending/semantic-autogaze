"""Modal launcher for r/autogaze-frozen-head cycle 2: scale to train2017.

Pipeline (single function so `modal run --detach` survives local disconnect):
    1. Cache AutoGaze gaze-decoder features on COCO train2017 (~118K images).
    2. Cache IconStudent-118K teacher heatmaps on train2017 (depends on
       pre-existing dinov2_train_native cache from r/native-aspect-train2017).
    3. Train BilinearCosineHead on train2017 pairs, val on val2017 pairs
       (val features+teacher already cached locally; we re-cache them on
       Modal too so a single launcher is self-contained).
    4. Commit checkpoint + metrics to results volume.

Cost estimate (T4 @ ~$0.59/hr):
    cache autogaze features val2017      ~5 min
    cache autogaze features train2017    ~30-60 min   (118K * ~10ms)
    cache teacher heatmaps val2017       ~5 min       (uses dinov2_val_native)
    cache teacher heatmaps train2017     ~30-60 min   (uses dinov2_train_native)
    train bilinear_cosine ~700K pairs    ~30-60 min
    total                                ~2-3 hr  ->  ~$1.5-2

Usage:
    modal run --detach scripts/modal_train_frozen_head.py
    # Skip caching steps if already populated:
    modal run --detach scripts/modal_train_frozen_head.py --skip-features --skip-teacher

Pull artifacts back:
    modal volume get sem-autogaze-results autogaze_frozen_head/cycle2 ./results/
"""
from __future__ import annotations

import os
import modal

REPO_URL = "https://github.com/Clamepending/semantic-autogaze.git"
BRANCH = "r/autogaze-frozen-head"
DATA_PATH = "/data"
RESULTS_PATH = "/results"

data_vol = modal.Volume.from_name("sem-autogaze-data", create_if_missing=False)
results_vol = modal.Volume.from_name("sem-autogaze-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.4.0",
        "torchvision",
        "transformers~=4.51",
        "huggingface_hub",
        "open_clip_torch",
        "pycocotools",
        "tqdm",
        "numpy",
        "Pillow",
        "matplotlib",
        "einops",
        "timm>=1.0.15",
        "hydra-core>=1.3.2",
        "wandb",
        "loguru",
        "omegaconf",
        "av",
        "imageio",
    )
    .run_commands(
        f"git clone --branch {BRANCH} {REPO_URL} /opt/sa",
        "cd /opt/sa && pip install -e .",
    )
)

app = modal.App("sa-autogaze-frozen-head", image=image)

FEAT_TRAIN = f"{RESULTS_PATH}/autogaze_probe/features_gaze_train"
FEAT_VAL = f"{RESULTS_PATH}/autogaze_probe/features_gaze_val"
TEACH_TRAIN = f"{RESULTS_PATH}/autogaze_probe/teacher_14x14_train"
TEACH_VAL = f"{RESULTS_PATH}/autogaze_probe/teacher_14x14_val"
CACHE_TRAIN_DV2 = f"{RESULTS_PATH}/dinov2_train_native"
CACHE_VAL_DV2 = f"{RESULTS_PATH}/dinov2_val_native"
BUNDLE_DIR = f"{RESULTS_PATH}/autogaze_probe/bundles_v2"
N_BUNDLE_CHUNKS = 16
RUN_NAME = "autogaze_frozen_head/cycle2"
OUT_DIR = f"{RESULTS_PATH}/{RUN_NAME}"


def _stream(cmd: str):
    import subprocess
    print(f"[cmd] {cmd}", flush=True)
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"command failed (exit {proc.returncode}): {cmd}")


def _setup():
    import subprocess
    os.chdir("/opt/sa")
    print(f"[code] git pull origin {BRANCH}", flush=True)
    subprocess.run(["git", "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "reset", "--hard", f"origin/{BRANCH}"], check=True)
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    print(f"[code] now at {sha}", flush=True)
    # Diagnostic: show disk + inode usage so we can see what's full when Modal volumes throw ENOSPC.
    print("[df] disk usage:", flush=True)
    subprocess.run(["df", "-h"], check=False)
    print("[df] inode usage:", flush=True)
    subprocess.run(["df", "-i"], check=False)

    if not os.path.exists("data"):
        os.symlink(DATA_PATH, "data")
    os.makedirs("results", exist_ok=True)
    for sub in ("autogaze_probe/features_gaze_train",
                "autogaze_probe/features_gaze_val",
                "autogaze_probe/teacher_14x14_train",
                "autogaze_probe/teacher_14x14_val",
                "dinov2_train_native",
                "dinov2_val_native",
                RUN_NAME):
        target = f"{RESULTS_PATH}/{sub}"
        os.makedirs(target, exist_ok=True)
        link = f"results/{sub}"
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if not os.path.lexists(link):
            os.symlink(target, link)


@app.function(
    cpu=2,
    memory=4096,
    volumes={RESULTS_PATH: results_vol},
    timeout=2 * 3600,
)
def bundle_chunk(chunk_idx: int, n_chunks: int, img_ids: list,
                 feat_dir: str, teach_dir: str, out_path: str,
                 split: str) -> int:
    """Read this container's slice of (feature, teacher) per-image files
    from the Modal Volume and pack them into a single bundle file.

    Spawned as part of a parallel .starmap() across N_BUNDLE_CHUNKS containers
    so the per-file open latency on Modal Volumes (~250ms single-container,
    ~5 it/s saturating point) is shared across many network connections.
    Round-robin slicing (img_ids[i::n_chunks]) gives uniform coverage.
    """
    import os
    import time
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    if os.path.exists(out_path):
        n = sum(1 for _ in torch.load(out_path, weights_only=False, map_location="cpu"))
        print(f"[bundle:{split}:{chunk_idx}] exists ({n} imgs), skipping", flush=True)
        return n

    chunk = img_ids[chunk_idx::n_chunks]
    print(f"[bundle:{split}:{chunk_idx}] {len(chunk)} imgs assigned", flush=True)
    fd = Path(feat_dir); td = Path(teach_dir)
    bundle = {}
    t0 = time.time()

    def _load(img_id):
        fp = fd / f"{img_id}.pt"
        tp = td / f"{img_id}.pt"
        if not fp.exists() or not tp.exists():
            return img_id, None
        try:
            f = torch.load(fp, weights_only=True).half()
            t = torch.load(tp, weights_only=True)
            teach = {int(k): v.half() for k, v in t.items()}
            return img_id, (f, teach)
        except Exception as e:
            print(f"[bundle:{split}:{chunk_idx}] load fail {img_id}: {e}", flush=True)
            return img_id, None

    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, (img_id, val) in enumerate(ex.map(_load, chunk)):
            if val is not None:
                bundle[int(img_id)] = val
            if (i + 1) % 1000 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                print(f"[bundle:{split}:{chunk_idx}] {i+1}/{len(chunk)} ({rate:.1f}/s)", flush=True)

    print(f"[bundle:{split}:{chunk_idx}] writing {len(bundle)} imgs to {out_path}", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(bundle, out_path)
    results_vol.commit()
    print(f"[bundle:{split}:{chunk_idx}] done in {time.time()-t0:.1f}s", flush=True)
    return len(bundle)


@app.function(
    gpu="T4",
    volumes={DATA_PATH: data_vol, RESULTS_PATH: results_vol},
    timeout=20 * 3600,
    ephemeral_disk=524288,
    memory=24576,
)
def cycle2(skip_features: bool = False, skip_teacher: bool = False):
    import torch
    assert torch.cuda.is_available()
    print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)
    _setup()

    # Ensure 118K teacher artifacts are present on the volume (they were
    # uploaded to results/icon_student_B_native_train by the prior r/native-aspect-train2017 run).
    teacher_ckpt = f"{RESULTS_PATH}/icon_student_B_native_train/best.pt"
    clip_text = f"{RESULTS_PATH}/icon_student_B_native_train/clip_text_embeddings.pt"
    for fp in (teacher_ckpt, clip_text):
        if not os.path.exists(fp):
            raise RuntimeError(f"missing {fp} — needed from r/native-aspect-train2017 run")
        print(f"[ok] {fp} ({os.path.getsize(fp)/1e6:.1f} MB)", flush=True)

    # Diagnostic: dump per-mount sizes to see which directory dominates.
    import subprocess as _sp
    print("[du] /results top-level (max-depth=2):", flush=True)
    _sp.run(["du", "-h", "--max-depth=2", "/results"], check=False)

    # Sanity: dinov2_train_native must be populated (teacher cache reads from it).
    n_dv2_train = len(os.listdir(CACHE_TRAIN_DV2)) if os.path.isdir(CACHE_TRAIN_DV2) else 0
    n_dv2_val = len(os.listdir(CACHE_VAL_DV2)) if os.path.isdir(CACHE_VAL_DV2) else 0
    print(f"[cache] dinov2_train_native: {n_dv2_train} files", flush=True)
    print(f"[cache] dinov2_val_native:   {n_dv2_val} files", flush=True)
    if n_dv2_train < 100000:
        raise RuntimeError(f"dinov2_train_native has only {n_dv2_train} files — need r/native-aspect-train2017 caches")
    if n_dv2_val < 4000:
        raise RuntimeError(f"dinov2_val_native has only {n_dv2_val} files — need r/native-aspect-train2017 caches")

    if not skip_features:
        # AutoGaze features val2017 first (small, fail fast).
        _stream(
            "python -u scripts/autogaze_probe/cache_features.py "
            "--data-dir data/coco "
            "--ann annotations/instances_val2017.json "
            "--img-subdir val2017 "
            f"--cache-dir {FEAT_VAL} "
            "--device cuda"
        )
        results_vol.commit()
        print("[autogaze-feat] val2017 done, committed", flush=True)

        _stream(
            "python -u scripts/autogaze_probe/cache_features.py "
            "--data-dir data/coco "
            "--ann annotations/instances_train2017.json "
            "--img-subdir train2017 "
            f"--cache-dir {FEAT_TRAIN} "
            "--device cuda"
        )
        results_vol.commit()
        print("[autogaze-feat] train2017 done, committed", flush=True)

    if not skip_teacher:
        # Teacher heatmaps val2017 first (already cached locally; rebuild on Modal too).
        _stream(
            "python -u scripts/autogaze_probe/cache_teacher.py "
            f"--ckpt {teacher_ckpt} "
            f"--clip-text-embeddings {clip_text} "
            f"--dinov2-cache-native {CACHE_VAL_DV2} "
            "--data-dir data/coco "
            "--ann annotations/instances_val2017.json "
            "--img-subdir val2017 "
            f"--out-dir {TEACH_VAL} "
            "--device cuda"
        )
        results_vol.commit()
        print("[teacher] val2017 done, committed", flush=True)

        # Chunked: write CHUNK_SIZE teacher files, commit, repeat. The
        # uncommitted writeback cache on the worker's ephemeral disk
        # filled up at ~70K files in the prior un-chunked run.
        TOTAL_TRAIN = 118287
        CHUNK_SIZE = 15000
        for start in range(0, TOTAL_TRAIN, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, TOTAL_TRAIN)
            print(f"[teacher] train2017 chunk [{start}, {end})", flush=True)
            _stream(
                "python -u scripts/autogaze_probe/cache_teacher.py "
                f"--ckpt {teacher_ckpt} "
                f"--clip-text-embeddings {clip_text} "
                f"--dinov2-cache-native {CACHE_TRAIN_DV2} "
                "--data-dir data/coco "
                "--ann annotations/instances_train2017.json "
                "--img-subdir train2017 "
                f"--out-dir {TEACH_TRAIN} "
                "--device cuda "
                f"--start-idx {start} --end-idx {end}"
            )
            results_vol.commit()
            print(f"[teacher] train2017 chunk [{start}, {end}) committed", flush=True)
        print("[teacher] train2017 all chunks done", flush=True)

    # Sanity counts before training.
    for d in (FEAT_VAL, FEAT_TRAIN, TEACH_VAL, TEACH_TRAIN):
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        print(f"[count] {d}: {n} files", flush=True)

    # Bundle phase: spawn N parallel CPU containers to read per-image files
    # from the volume in parallel and pack them into N chunk files. This
    # bypasses the single-container ~5 it/s read cap by sharing the work
    # across many independent volume-client connections.
    from pycocotools.coco import COCO
    coco_val = COCO("data/coco/annotations/instances_val2017.json")
    coco_train = COCO("data/coco/annotations/instances_train2017.json")
    val_ids = sorted(coco_val.getImgIds())
    train_ids = sorted(coco_train.getImgIds())
    print(f"[bundle] val_ids={len(val_ids)} train_ids={len(train_ids)}", flush=True)

    bundle_dir_val = f"{BUNDLE_DIR}/val"
    bundle_dir_train = f"{BUNDLE_DIR}/train"
    os.makedirs(bundle_dir_val, exist_ok=True)
    os.makedirs(bundle_dir_train, exist_ok=True)
    results_vol.commit()

    # Val: small, single chunk (1 container).
    val_args = [(0, 1, val_ids, FEAT_VAL, TEACH_VAL,
                 f"{bundle_dir_val}/chunk_0.pt", "val")]
    # Train: N chunks, fanned out.
    train_args = [
        (i, N_BUNDLE_CHUNKS, train_ids, FEAT_TRAIN, TEACH_TRAIN,
         f"{bundle_dir_train}/chunk_{i}.pt", "train")
        for i in range(N_BUNDLE_CHUNKS)
    ]

    print(f"[bundle] dispatching {len(val_args) + len(train_args)} bundle jobs", flush=True)
    counts = list(bundle_chunk.starmap(val_args + train_args))
    print(f"[bundle] all chunks done; counts={counts}", flush=True)
    results_vol.reload()

    # Now run training: T4 container reads ~17 large bundle files (instead of
    # 226K small per-image files) and trains in-RAM.
    bundle_train_glob = f"{bundle_dir_train}/chunk_*.pt"
    bundle_val_glob = f"{bundle_dir_val}/chunk_*.pt"
    _stream(
        "python -u scripts/autogaze_probe/train_frozen_head.py "
        f"--bundle-glob '{bundle_train_glob}' "
        f"--val-bundle-glob '{bundle_val_glob}' "
        f"--clip-text {clip_text} "
        f"--out-dir {OUT_DIR} "
        "--device cuda "
        "--n-epochs 10 "
        "--batch-size 256 "
        "--num-workers 0 "
        "--lr 1e-3"
    )
    results_vol.commit()
    print(f"[done] artifacts at {OUT_DIR}", flush=True)


@app.local_entrypoint()
def main(skip_features: bool = False, skip_teacher: bool = False):
    cycle2.remote(skip_features=skip_features, skip_teacher=skip_teacher)
