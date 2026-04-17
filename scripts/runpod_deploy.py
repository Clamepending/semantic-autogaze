"""
RunPod deployment helper for semantic-autogaze.

Reads the API key from `~/.runpod/config.toml` (written by
`runpodctl config --apiKey <KEY>`) or from `RUNPOD_API_KEY` env var — never
from arguments or the repo.

Usage:

    # One-time volume prep: clone repo, install deps, download nvidia/AutoGaze
    python scripts/runpod_deploy.py prep --volume <ID> --dc <REGION>

    # Launch an on-demand 3090 pod (for setup / smoke tests)
    python scripts/runpod_deploy.py launch --mode on-demand --gpu "RTX 3090" \
        --volume <ID> --dc <REGION>

    # Launch a spot 3090 pod (for real training)
    python scripts/runpod_deploy.py launch --mode spot --gpu "RTX 3090" \
        --volume <ID> --dc <REGION>

    # List / stop
    python scripts/runpod_deploy.py list
    python scripts/runpod_deploy.py stop --pod-id <ID>
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import tomllib
from pathlib import Path
from typing import Optional

import runpod

REPO_URL = "https://github.com/Clamepending/semantic-autogaze.git"
DEFAULT_BRANCH = "reconstruct-from-wandb-2026-04-15"
VOLUME_MOUNT = "/workspace"

# GPU type IDs as RunPod's API expects them.
GPU_IDS = {
    "RTX 3090": "NVIDIA GeForce RTX 3090",
    "RTX 4090": "NVIDIA GeForce RTX 4090",
    "RTX A5000": "NVIDIA RTX A5000",
    "RTX A4000": "NVIDIA RTX A4000",
    "A40": "NVIDIA A40",
    "L4": "NVIDIA L4",
}

# A PyTorch image with CUDA + Python 3.11 preinstalled. Saves ~5–10 min
# vs a bare CUDA image that has to install torch on each cold start.
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def load_api_key() -> str:
    # 1. env var wins (useful for CI)
    if env := os.environ.get("RUNPOD_API_KEY"):
        return env
    # 2. runpodctl config
    cfg = Path.home() / ".runpod/config.toml"
    if cfg.exists():
        data = tomllib.loads(cfg.read_text())
        if k := data.get("apikey") or data.get("apiKey"):
            return k
    sys.exit(
        "No RunPod API key found. Run `runpodctl config --apiKey <KEY>` "
        "or set RUNPOD_API_KEY."
    )


def startup_script(mode: str) -> str:
    """First-boot script. Idempotent: safe to re-run on a resumed pod."""
    return f"""#!/bin/bash
set -eux
mkdir -p {VOLUME_MOUNT}/semantic-autogaze
cd {VOLUME_MOUNT}/semantic-autogaze

# Clone or update repo
if [ ! -d .git ]; then
    git clone --branch {DEFAULT_BRANCH} {REPO_URL} .
else
    git fetch origin {DEFAULT_BRANCH} && git checkout {DEFAULT_BRANCH} && git pull --ff-only
fi

# Install project deps if needed
if ! python -c 'import semantic_autogaze' 2>/dev/null; then
    pip install -e . open_clip_torch wandb
fi

# Wandb + HF auth: the caller should set WANDB_API_KEY and HF_TOKEN when
# creating the pod. The volume persists ~/.cache across restarts.
[ -n "${{WANDB_API_KEY:-}}" ] && wandb login --relogin "$WANDB_API_KEY" || true
[ -n "${{HF_TOKEN:-}}" ] && huggingface-cli login --token "$HF_TOKEN" || true

echo "[setup] pod is ready. mode={mode}"
# Keep the pod alive so we can SSH in. Training is launched as a separate step.
sleep infinity
"""


def _gpu_type_id(gpu: str) -> str:
    if gpu not in GPU_IDS:
        raise SystemExit(f"unknown GPU {gpu!r}; known: {list(GPU_IDS)}")
    return GPU_IDS[gpu]


def _encoded_docker_args(script: str) -> str:
    # RunPod's SDK inlines docker_args into the GraphQL mutation with no
    # escaping: `dockerArgs: "{docker_args}"`. So docker_args must contain
    # no `"` and no `\`. Base64 output is [A-Za-z0-9+/=] only, and we wrap
    # the shell command in single quotes so spaces/pipes/`$` are fine.
    b64 = base64.b64encode(script.encode()).decode()
    return f"bash -c 'echo {b64} | base64 -d | bash'"


def launch(args):
    runpod.api_key = load_api_key()

    if args.mode == "spot":
        sys.exit(
            "spot mode requires raw GraphQL (`podRentInterruptable`); the "
            "`runpod` SDK's create_pod only does on-demand. Use --mode on-demand "
            "for now, or extend this script with a raw-GraphQL path."
        )

    pod = runpod.create_pod(
        name=f"sem-autogaze-{args.mode}-{args.gpu.lower().replace(' ', '')}",
        image_name=args.image,
        gpu_type_id=_gpu_type_id(args.gpu),
        gpu_count=1,
        volume_in_gb=0,  # ephemeral disk only; real state goes on network volume
        container_disk_in_gb=40,
        network_volume_id=args.volume,
        data_center_id=args.dc,
        volume_mount_path=VOLUME_MOUNT,
        ports="22/tcp,8888/http",
        docker_args=_encoded_docker_args(startup_script(args.mode)),
        cloud_type=("SECURE" if args.secure else "COMMUNITY"),
        env={
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        },
    )
    print(f"pod id: {pod['id']}  status: {pod.get('desiredStatus', '?')}")
    print(f"ssh: runpodctl ssh {pod['id']}")
    print(f"logs: runpodctl logs {pod['id']} -f")


def prep(args):
    """One-time: launch a cheap on-demand pod, run the volume-population
    script to `git clone` + `pip install` + download `nvidia/AutoGaze` into
    the network volume, then stop the pod. Cost: ~$0.22 × ~15 min ≈ $0.06."""
    runpod.api_key = load_api_key()
    prep_script = f"""#!/bin/bash
set -eux
mkdir -p {VOLUME_MOUNT}/semantic-autogaze
cd {VOLUME_MOUNT}/semantic-autogaze
if [ -d .git ]; then
    git fetch origin {DEFAULT_BRANCH} && git checkout {DEFAULT_BRANCH} && git pull --ff-only
else
    git clone --branch {DEFAULT_BRANCH} {REPO_URL} .
fi
pip install -e . open_clip_torch wandb pycocotools
[ -n "${{HF_TOKEN:-}}" ] && huggingface-cli login --token "$HF_TOKEN" || true

# Pre-download AutoGaze weights into the volume's HF cache.
export HF_HOME={VOLUME_MOUNT}/hf_cache
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/AutoGaze')"

# Download COCO train2017 + val2017 + annotations
python -c "
from semantic_autogaze.train_coco_seg import download_coco_val, download_coco_train
download_coco_val('data/coco')
download_coco_train('data/coco')
print('[prep] COCO data downloaded')
"

echo "[prep] volume is ready at {VOLUME_MOUNT}/semantic-autogaze"
"""
    pod = runpod.create_pod(
        name="sem-autogaze-prep",
        image_name=args.image,
        gpu_type_id=_gpu_type_id(args.gpu),
        gpu_count=1,
        volume_in_gb=0,
        container_disk_in_gb=40,
        network_volume_id=args.volume,
        data_center_id=args.dc,
        volume_mount_path=VOLUME_MOUNT,
        ports="22/tcp",
        docker_args=_encoded_docker_args(prep_script + "\nsleep infinity\n"),
        cloud_type="COMMUNITY",
        env={"HF_TOKEN": os.environ.get("HF_TOKEN", "")},
    )
    print(f"prep pod id: {pod['id']}")
    print("watch it with: runpodctl logs", pod["id"], "-f")
    print("when you see [prep] volume is ready, stop it:")
    print(f"  python {sys.argv[0]} stop --pod-id {pod['id']}")


def list_pods(_args):
    runpod.api_key = load_api_key()
    pods = runpod.get_pods()
    for p in pods:
        print(f"{p['id']}  {p.get('name'):30s}  {p.get('desiredStatus')}  {p.get('machine', {}).get('gpuDisplayName')}")


def stop(args):
    runpod.api_key = load_api_key()
    runpod.stop_pod(args.pod_id)
    print(f"stopped {args.pod_id}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--image", default=DEFAULT_IMAGE)
    shared.add_argument("--volume", required=True, help="Network volume ID")
    shared.add_argument("--dc", required=True, help="Data center ID, e.g. US-OR-01")

    p_prep = sub.add_parser("prep", parents=[shared])
    p_prep.add_argument("--gpu", default="RTX 3090", choices=list(GPU_IDS))

    p_launch = sub.add_parser("launch", parents=[shared])
    p_launch.add_argument("--gpu", default="RTX 3090", choices=list(GPU_IDS))
    p_launch.add_argument("--mode", choices=["on-demand", "spot"], default="spot")
    p_launch.add_argument("--bid", type=float, default=0.14, help="Max $/hr/GPU for spot")
    p_launch.add_argument("--secure", action="store_true", help="Use Secure Cloud instead of Community")

    sub.add_parser("list")

    p_stop = sub.add_parser("stop")
    p_stop.add_argument("--pod-id", required=True)

    args = p.parse_args()
    {"prep": prep, "launch": launch, "list": list_pods, "stop": stop}[args.cmd](args)


if __name__ == "__main__":
    main()
