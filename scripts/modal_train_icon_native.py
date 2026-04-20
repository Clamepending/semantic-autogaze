"""
Modal launcher for r/native-aspect-train2017: cache DINOv2-small at native
aspect over train2017 + val2017, then train IconStudent Recipe B with
external val-ann (train on 118K of train2017, val on 5K of val2017).

Usage:
    modal run --detach scripts/modal_train_icon_native.py

    # Inspect logs of detached run
    modal app logs <app-id>

    # Pull artifacts back
    modal volume get sem-autogaze-results icon_student_B_native_train ./results/
    modal volume get sem-autogaze-results icon_student_B_native_train/best.pt ./

Volumes:
    sem-autogaze-data     /data       — coco/{train2017,val2017,annotations}
                                          already populated by v11/v12 work
    sem-autogaze-results  /results    — caches + checkpoints written here

Cost guesstimate (T4 @ ~$0.59/hr):
    cache val2017       ~10 min  (5K images, batch=1, variable shape)
    cache train2017     ~80-120 min  (118K images, same)
    train 10 epochs     ~3-5 hr   (decoder is small; bottleneck is feature load)
    eval + render        ~10 min
    total               ~5-7 hr  →  ~$3-4
"""
from __future__ import annotations

import os
import modal

REPO_URL = "https://github.com/Clamepending/semantic-autogaze.git"
BRANCH = "r/native-aspect-train2017"
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
        "transformers",
        "huggingface_hub",
        "open_clip_torch",
        "pycocotools",
        "tqdm",
        "numpy",
        "Pillow",
        "matplotlib",
    )
    .run_commands(
        f"git clone --branch {BRANCH} {REPO_URL} /opt/sa",
        "cd /opt/sa && pip install -e .",
    )
)

app = modal.App("sa-icon-native-train2017", image=image)

CACHE_TRAIN_DIR = f"{RESULTS_PATH}/dinov2_train_native"
CACHE_VAL_DIR = f"{RESULTS_PATH}/dinov2_val_native"
RUN_NAME = "icon_student_B_native_train"
OUT_DIR = f"{RESULTS_PATH}/{RUN_NAME}"


def _stream(cmd: str):
    """Run a shell command, streaming stdout/stderr live."""
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
    """Sync code to BRANCH head, symlink data + results into the repo."""
    import subprocess
    os.chdir("/opt/sa")
    print(f"[code] git pull origin {BRANCH}", flush=True)
    subprocess.run(["git", "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "reset", "--hard", f"origin/{BRANCH}"], check=True)
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    print(f"[code] now at {sha}", flush=True)

    # data volume → ./data ; results volume → ./results
    if not os.path.exists("data"):
        os.symlink(DATA_PATH, "data")
    os.makedirs("results", exist_ok=True)
    for sub in ("dinov2_train_native", "dinov2_val_native", RUN_NAME):
        target = f"{RESULTS_PATH}/{sub}"
        os.makedirs(target, exist_ok=True)
        link = f"results/{sub}"
        if not os.path.lexists(link):
            os.symlink(target, link)


@app.function(
    gpu="T4",
    volumes={DATA_PATH: data_vol, RESULTS_PATH: results_vol},
    timeout=20 * 3600,
    ephemeral_disk=524288,  # 512 GiB — Modal's enforced minimum for custom ephemeral_disk
)
def cache_and_train(skip_cache: bool = False):
    """Cache DINOv2 native (val2017 + train2017) then train IconStudent
    Recipe B with external val-ann.

    Single function so `modal run --detach` survives local disconnect:
    detach guarantees the *most recently triggered* remote call lives on,
    so we collapse cache+train into one call.
    """
    import torch
    assert torch.cuda.is_available()
    print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)
    _setup()

    if not skip_cache:
        # val2017 first (small, fast — fail early if anything's wrong)
        _stream(
            "python -m semantic_autogaze.cache_dinov2 "
            "--ann data/coco/annotations/instances_val2017.json "
            "--img-dir data/coco/val2017 "
            f"--cache-dir {CACHE_VAL_DIR} "
            "--mode native --device cuda --num-workers 4"
        )
        results_vol.commit()
        print("[cache] val2017 done, committed", flush=True)

        _stream(
            "python -m semantic_autogaze.cache_dinov2 "
            "--ann data/coco/annotations/instances_train2017.json "
            "--img-dir data/coco/train2017 "
            f"--cache-dir {CACHE_TRAIN_DIR} "
            "--mode native --device cuda --num-workers 4"
        )
        results_vol.commit()
        print("[cache] train2017 done, committed", flush=True)

    # Sanity: caches must be populated.
    for d in (CACHE_VAL_DIR, CACHE_TRAIN_DIR):
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        print(f"[cache] {d}: {n} files", flush=True)
        if n < 100:
            raise RuntimeError(f"cache {d} has only {n} files — run cache first")

    _stream(
        "python -u -m semantic_autogaze.train_icon_student "
        "--supervision B "
        "--data-dir data/coco "
        "--ann annotations/instances_train2017.json "
        "--img-subdir train2017 "
        f"--dinov2-cache {CACHE_TRAIN_DIR} "
        "--val-ann annotations/instances_val2017.json "
        "--val-img-subdir val2017 "
        f"--val-dinov2-cache {CACHE_VAL_DIR} "
        f"--output-dir {OUT_DIR} "
        "--device cuda "
        "--num-epochs 3 "
        "--batch-size 16 "
        "--num-workers 8 "
        "--lr 3e-4"
    )
    results_vol.commit()
    print(f"[done] artifacts at {OUT_DIR}", flush=True)


@app.local_entrypoint()
def main(skip_cache: bool = False):
    """Single .remote() so --detach guarantees survival."""
    cache_and_train.remote(skip_cache=skip_cache)
