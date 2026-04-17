#!/usr/bin/env bash
# v10g: focal (alpha=0.75 → up-weight FG) + 5x negatives + dice.
# Tests whether correct class-balanced focal alpha improves FG confidence
# beyond v10e's unweighted focal.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

WAIT_PID="${WAIT_PID:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[v10g] waiting for PID $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
  echo "[v10g] predecessor exited"
fi

mkdir -p results/coco_seg_v10g

# Symlink shared hidden cache
if [ ! -e results/coco_seg_v10g/hidden_cache ]; then
  ln -s "$(pwd)/results/shared_hidden_cache" results/coco_seg_v10g/hidden_cache
  echo "[v10g] symlinked shared hidden_cache"
fi

echo "[v10g] launching: focal(alpha=0.75) + neg5 + dice=1.0 + clipseg=0.5 + drop0.10"
python -u -m semantic_autogaze.train_coco_seg \
  --device mps \
  --num_epochs 15 \
  --batch_size 24 \
  --dropout 0.10 \
  --dice_weight 1.0 \
  --focal \
  --focal_alpha 0.75 \
  --focal_gamma 2.0 \
  --decoder \
  --out_grid 28 \
  --decoder_dim 128 \
  --clipseg_mix 0.5 \
  --neg_per_image 5 \
  --output_dir results/coco_seg_v10g \
  --run_name coco-seg-v10g-focal-a0.75-neg5-dice1.0-clipseg0.5-drop0.10 \
  --num_qual_examples 20 \
  > results/coco_seg_v10g/train.log 2>&1

echo "[v10g] training done; running eval"
python scripts/eval_quantitative.py \
  --head-ckpt results/coco_seg_v10g/best_head.pt \
  --hidden-cache-dir results/coco_seg_v10g/hidden_cache \
  --device mps --output-dir results/eval_v10g

echo "[v10g] ALL DONE"
