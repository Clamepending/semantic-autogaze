#!/bin/bash
# Retry RunPod launch in EU-CZ-1 until stock is available.
# Alternates 3090 → 4090 each iteration. Writes progress to stdout.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

VOLUME="${VOLUME:-82egl4egf5}"
DC="${DC:-EU-CZ-1}"
INTERVAL="${INTERVAL:-90}"
GPUS=("RTX 3090" "RTX 4090")

i=0
while true; do
  gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  ts=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
  echo "[$ts] attempt $((i+1)): gpu='$gpu' dc=$DC volume=$VOLUME"
  out=$(python scripts/runpod_deploy.py launch --mode on-demand \
        --gpu "$gpu" --volume "$VOLUME" --dc "$DC" 2>&1)
  rc=$?
  if [ $rc -eq 0 ] && echo "$out" | grep -q '^pod id:'; then
    echo "[$ts] SUCCESS:"
    echo "$out"
    exit 0
  fi
  echo "[$ts] no luck. last err: $(echo "$out" | tail -1)"
  i=$((i+1))
  sleep "$INTERVAL"
done
