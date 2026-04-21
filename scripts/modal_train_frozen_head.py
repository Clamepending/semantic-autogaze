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

    # Bulk-copy from network volume to ephemeral SSD before training.
    # Modal Volume's per-file random-access latency made the in-process
    # preload (113K * ~250ms) project ~7 hr; tar streaming uses sequential
    # I/O which is ~10-50x faster for many small files.
    LOCAL_FEAT_TRAIN = "/tmp/sa/features_gaze_train"
    LOCAL_FEAT_VAL = "/tmp/sa/features_gaze_val"
    LOCAL_TEACH_TRAIN = "/tmp/sa/teacher_14x14_train"
    LOCAL_TEACH_VAL = "/tmp/sa/teacher_14x14_val"
    os.makedirs("/tmp/sa", exist_ok=True)
    for src, dst in [
        (FEAT_VAL, LOCAL_FEAT_VAL),
        (TEACH_VAL, LOCAL_TEACH_VAL),
        (FEAT_TRAIN, LOCAL_FEAT_TRAIN),
        (TEACH_TRAIN, LOCAL_TEACH_TRAIN),
    ]:
        if os.path.isdir(dst) and len(os.listdir(dst)) > 100:
            print(f"[copy] {dst} already populated, skipping", flush=True)
            continue
        os.makedirs(dst, exist_ok=True)
        # tar->untar pipe: single sequential read on the volume, single write
        # on the SSD. Skip permissions/owner bits to avoid spurious errors.
        _stream(
            f"tar -C {os.path.dirname(src)} -cf - {os.path.basename(src)} "
            f"| tar -C {os.path.dirname(dst)} -xf -"
        )
        n = len(os.listdir(dst))
        print(f"[copy] {dst}: {n} files", flush=True)

    _stream(
        "python -u scripts/autogaze_probe/train_frozen_head.py "
        f"--feature-dir {LOCAL_FEAT_TRAIN} "
        f"--teacher-dir {LOCAL_TEACH_TRAIN} "
        f"--val-feature-dir {LOCAL_FEAT_VAL} "
        f"--val-teacher-dir {LOCAL_TEACH_VAL} "
        f"--clip-text {clip_text} "
        f"--out-dir {OUT_DIR} "
        "--device cuda "
        "--n-epochs 10 "
        "--batch-size 256 "
        "--num-workers 0 "
        "--preload "
        "--lr 1e-3"
    )
    results_vol.commit()
    print(f"[done] artifacts at {OUT_DIR}", flush=True)


@app.local_entrypoint()
def main(skip_features: bool = False, skip_teacher: bool = False):
    cycle2.remote(skip_features=skip_features, skip_teacher=skip_teacher)
