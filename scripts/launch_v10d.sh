#!/usr/bin/env bash
# Launch v10d: CLIP vision features + AutoGaze features combined.
# Wait for chain PID if provided.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

WAIT_PID="${WAIT_PID:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[v10d] waiting for PID $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
  echo "[v10d] predecessor exited"
fi

mkdir -p results/coco_seg_v10d

echo "[v10d] launching training with CLIP vision features..."
python -u -m semantic_autogaze.train_coco_seg \
  --device mps \
  --num_epochs 15 \
  --batch_size 24 \
  --dropout 0.10 \
  --dice_weight 1.0 \
  --decoder \
  --out_grid 28 \
  --decoder_dim 128 \
  --clipseg_mix 0.5 \
  --clip_vision \
  --output_dir results/coco_seg_v10d \
  --run_name coco-seg-v10d-clip-vision-dice1.0-clipseg0.5-drop0.10 \
  --num_qual_examples 20 \
  > results/coco_seg_v10d/train.log 2>&1

echo "[v10d] training done; running eval"
python scripts/eval_quantitative.py \
  --head-ckpt results/coco_seg_v10d/best_head.pt \
  --hidden-cache-dir results/coco_seg_v10d/hidden_cache \
  --clip-vision-cache-dir results/coco_seg_v10d/clip_vision_cache \
  --device mps --output-dir results/eval_v10d

echo "[v10d] ALL DONE"
