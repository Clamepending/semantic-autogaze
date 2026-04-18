"""
Modal training launcher for semantic-autogaze experiments.

Usage:
    # IMPORTANT: Always use --detach so training survives local disconnects!
    # Run v12 with defaults (focal+neg5+CLIP vision+train2017, out_grid=28)
    modal run --detach scripts/modal_train.py --run-name coco_seg_v12

    # Run with custom args
    modal run --detach scripts/modal_train.py --run-name coco_seg_v12b \
        --train-args "--device cuda --num_epochs 10 --batch_size 24 --dropout 0.10 \
        --dice_weight 1.0 --focal --focal_alpha 0.25 --focal_gamma 2.0 \
        --decoder --out_grid 56 --decoder_dim 128 --clipseg_mix 0.0 \
        --neg_per_image 5 --clip_vision_online --train_split train --num_qual_examples 20"

    # Monitor a running detached app
    modal app logs <app-id>

    # Download checkpoint locally after training
    modal volume get sem-autogaze-results <run_name>/best_head.pt ./

    # Check what's on the volume
    modal volume ls sem-autogaze-results
"""

import modal
import os

REPO_URL = "https://github.com/Clamepending/semantic-autogaze.git"
BRANCH = "experiment/v13e-within-image-neg"
VOLUME_PATH = "/data"
RESULTS_PATH = "/results"

# Persistent volume for COCO data + AutoGaze model (shared across runs)
data_vol = modal.Volume.from_name("sem-autogaze-data", create_if_missing=True)
# Separate volume for results (checkpoints, logs)
results_vol = modal.Volume.from_name("sem-autogaze-results", create_if_missing=True)

# Container image with all dependencies pre-installed
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "unzip")
    .pip_install(
        "torch>=2.4.0",
        "torchvision",
        "open_clip_torch",
        "wandb",
        "pycocotools",
        "tqdm",
        "numpy",
        "Pillow",
        "transformers",
        "huggingface_hub",
        "gradio",
    )
    .run_commands(
        f"git clone --branch {BRANCH} {REPO_URL} /opt/semantic-autogaze",
        "cd /opt/semantic-autogaze && pip install -e .",
    )
)

app = modal.App("sem-autogaze-train", image=image)

DEFAULT_TRAIN_ARGS = (
    "--device cuda "
    "--num_epochs 10 "
    "--batch_size 24 "
    "--dropout 0.10 "
    "--dice_weight 1.0 "
    "--focal "
    "--focal_alpha 0.25 "
    "--focal_gamma 2.0 "
    "--decoder "
    "--out_grid 28 "
    "--decoder_dim 128 "
    "--clipseg_mix 0.0 "
    "--neg_per_image 5 "
    "--clip_vision_online "
    "--train_split train "
    "--num_qual_examples 20"
)


@app.function(
    gpu="T4",
    volumes={VOLUME_PATH: data_vol, RESULTS_PATH: results_vol},
    timeout=18 * 3600,  # 18h — full train2017 + cache + eval, with headroom
    secrets=[
        modal.Secret.from_dict({
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        })
    ],
)
def train(run_name: str = "coco_seg_v12", train_args: str = ""):
    """Run a training experiment on GPU."""
    import subprocess
    import torch

    # Verify GPU
    assert torch.cuda.is_available(), "No CUDA GPU available!"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[gpu] {gpu_name}, {gpu_mem:.1f}GB")

    # Auth
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    if wandb_key:
        subprocess.run(["wandb", "login", "--relogin", wandb_key],
                       capture_output=True)
        print("[auth] wandb OK")
    if hf_token:
        subprocess.run(["huggingface-cli", "login", "--token", hf_token],
                       capture_output=True)
        print("[auth] huggingface OK")

    os.chdir("/opt/semantic-autogaze")

    # Pull latest commit on the configured branch so we don't run a stale
    # snapshot baked into the cached image. Modal caches `run_commands` layers,
    # so without this any new commit on BRANCH would be silently ignored.
    print(f"[code] git pull origin {BRANCH}")
    subprocess.run(["git", "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "reset", "--hard", f"origin/{BRANCH}"], check=True)
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    print(f"[code] now at {sha}")

    # Symlink data volume
    data_dir = f"{VOLUME_PATH}/coco"
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists("data"):
        os.symlink(VOLUME_PATH, "data")

    # Download COCO data if not cached on volume
    print("[data] Checking COCO data on volume...")
    from semantic_autogaze.train_coco_seg import download_coco_val, download_coco_train
    download_coco_val("data/coco")
    print("[data] val2017 ready")
    effective_args = train_args if train_args else DEFAULT_TRAIN_ARGS
    if "--train_split train" in effective_args or "--train_split=train" in effective_args:
        download_coco_train("data/coco")
        print("[data] train2017 ready")
    data_vol.commit()

    # Download AutoGaze model if not cached
    hf_cache = f"{VOLUME_PATH}/hf_cache"
    os.environ["HF_HOME"] = hf_cache
    os.makedirs(hf_cache, exist_ok=True)
    print("[data] Checking AutoGaze model...")
    from huggingface_hub import snapshot_download
    snapshot_download("nvidia/AutoGaze")
    print("[data] AutoGaze ready")
    data_vol.commit()

    # Set up results directory on results volume
    results_dir = f"{RESULTS_PATH}/{run_name}"
    os.makedirs(results_dir, exist_ok=True)
    # Symlink so training script writes to the volume
    os.makedirs("results", exist_ok=True)
    if not os.path.exists(f"results/{run_name}"):
        os.symlink(results_dir, f"results/{run_name}")

    # Build training command — WANDB_MODE=offline avoids 90s init timeouts
    # that have been killing remote runs (wandb cloud occasionally slow)
    final_args = train_args if train_args else DEFAULT_TRAIN_ARGS
    cmd = (
        f"WANDB_MODE=offline python -u -m semantic_autogaze.train_coco_seg "
        f"--output_dir results/{run_name} "
        f"--run_name {run_name} "
        f"{final_args}"
    )
    print(f"[train] {cmd}")
    print(f"[train] Starting {run_name}...")

    # Run training with real-time output
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    if proc.returncode != 0:
        print(f"[train] FAILED with exit code {proc.returncode}")
        results_vol.commit()
        raise RuntimeError(f"Training failed with exit code {proc.returncode}")

    print(f"[train] {run_name} completed successfully!")
    results_vol.commit()

    # Run eval
    eval_dir = f"{RESULTS_PATH}/eval_{run_name}"
    os.makedirs(eval_dir, exist_ok=True)
    if not os.path.exists(f"results/eval_{run_name}"):
        os.symlink(eval_dir, f"results/eval_{run_name}")

    eval_cmd = (
        f"python scripts/eval_quantitative.py "
        f"--head-ckpt results/{run_name}/best_head.pt "
        f"--hidden-cache-dir results/{run_name}/val_hidden_cache "
        f"--clip-vision-online "
        f"--device cuda "
        f"--output-dir results/eval_{run_name}"
    )
    print(f"[eval] {eval_cmd}")
    proc = subprocess.Popen(
        eval_cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    results_vol.commit()

    print(f"[done] {run_name} training + eval complete!")
    print(f"[done] Checkpoint: modal volume get sem-autogaze-results {run_name}/best_head.pt ./")


@app.local_entrypoint()
def main(
    run_name: str = "coco_seg_v12",
    train_args: str = "",
):
    """Launch training from CLI."""
    print(f"Launching {run_name} on Modal...")
    print(f"Logs stream in real-time. Ctrl+C to detach (training continues).")
    train.remote(run_name=run_name, train_args=train_args)
