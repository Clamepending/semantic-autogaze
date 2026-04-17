#!/usr/bin/env bash
# Wait for v7 (PID $V7_PID) to exit, then launch v8 with the multi-objective recipe.
# Usage: V7_PID=61577 bash scripts/launch_v8.sh
set -u
cd "$(dirname "$0")/.."

V7_PID="${V7_PID:-61577}"
echo "[chain] waiting for PID $V7_PID..."
while kill -0 "$V7_PID" 2>/dev/null; do
  sleep 30
done
echo "[chain] v7 exited; launching v8"

mkdir -p results/coco_seg_v8
source .venv/bin/activate
nohup python -u -m semantic_autogaze.train_coco_seg \
  --device mps \
  --num_epochs 15 \
  --batch_size 24 \
  --dropout 0.15 \
  --dice_weight 1.0 \
  --decoder \
  --out_grid 28 \
  --decoder_dim 128 \
  --clipseg_mix 0.3 \
  --output_dir results/coco_seg_v8 \
  --run_name coco-seg-v8-decoder28-dice1.0-clipseg0.3-drop0.15 \
  --num_qual_examples 20 \
  > results/coco_seg_v8/train.log 2>&1 &
echo "[chain] v8 PID=$!"
