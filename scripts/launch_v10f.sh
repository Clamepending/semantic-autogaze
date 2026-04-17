#!/usr/bin/env bash
# v10f: focal + 5x negatives + CLIP vision features — combining all three best findings.
# Wait for predecessor PID if provided.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

WAIT_PID="${WAIT_PID:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[v10f] waiting for PID $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
  echo "[v10f] predecessor exited"
fi

mkdir -p results/coco_seg_v10f

# Symlink shared hidden cache
if [ ! -e results/coco_seg_v10f/hidden_cache ]; then
  ln -s "$(pwd)/results/shared_hidden_cache" results/coco_seg_v10f/hidden_cache
  echo "[v10f] symlinked shared hidden_cache"
fi

echo "[v10f] launching: focal + neg5 + clip_vision + dice=1.0 + clipseg=0.5 + drop0.10"
python -u -m semantic_autogaze.train_coco_seg \
  --device mps \
  --num_epochs 15 \
  --batch_size 24 \
  --dropout 0.10 \
  --dice_weight 1.0 \
  --focal \
  --focal_alpha 0.25 \
  --focal_gamma 2.0 \
  --decoder \
  --out_grid 28 \
  --decoder_dim 128 \
  --clipseg_mix 0.5 \
  --clip_vision \
  --neg_per_image 5 \
  --output_dir results/coco_seg_v10f \
  --run_name coco-seg-v10f-focal-neg5-clipvision-dice1.0-clipseg0.5-drop0.10 \
  --num_qual_examples 20 \
  > results/coco_seg_v10f/train.log 2>&1

echo "[v10f] training done; running eval"
python scripts/eval_quantitative.py \
  --head-ckpt results/coco_seg_v10f/best_head.pt \
  --hidden-cache-dir results/coco_seg_v10f/hidden_cache \
  --clip-vision-cache-dir results/coco_seg_v10f/clip_vision_cache \
  --device mps --output-dir results/eval_v10f

echo "[v10f] ALL DONE"
