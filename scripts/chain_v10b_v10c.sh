#!/usr/bin/env bash
# Chain v10b and v10c after v10 finishes.
# Usage: V10_PID=97448 bash scripts/chain_v10b_v10c.sh
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

V10_PID="${V10_PID:-97448}"
echo "[chain] waiting for v10 PID $V10_PID..."
while kill -0 "$V10_PID" 2>/dev/null; do
  sleep 30
done
echo "[chain] v10 exited; running eval then launching v10b"

# Eval v10
python scripts/eval_quantitative.py \
  --head-ckpt results/coco_seg_v10/best_head.pt \
  --hidden-cache-dir results/coco_seg_v10/hidden_cache \
  --device mps --output-dir results/eval_v10

# v10b: v7 recipe + 5 negatives per image
mkdir -p results/coco_seg_v10b
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
  --neg_per_image 5 \
  --output_dir results/coco_seg_v10b \
  --run_name coco-seg-v10b-neg5-dice1.0-clipseg0.5-drop0.10 \
  --num_qual_examples 20 \
  > results/coco_seg_v10b/train.log 2>&1

echo "[chain] v10b done; running eval then launching v10c"

# Eval v10b
python scripts/eval_quantitative.py \
  --head-ckpt results/coco_seg_v10b/best_head.pt \
  --hidden-cache-dir results/coco_seg_v10b/hidden_cache \
  --device mps --output-dir results/eval_v10b

# v10c: pure CLIPSeg distillation
mkdir -p results/coco_seg_v10c
python -u -m semantic_autogaze.train_coco_seg \
  --device mps \
  --num_epochs 15 \
  --batch_size 24 \
  --dropout 0.10 \
  --dice_weight 1.0 \
  --decoder \
  --out_grid 28 \
  --decoder_dim 128 \
  --clipseg_mix 1.0 \
  --output_dir results/coco_seg_v10c \
  --run_name coco-seg-v10c-pure-clipseg-dice1.0-drop0.10 \
  --num_qual_examples 20 \
  > results/coco_seg_v10c/train.log 2>&1

echo "[chain] v10c done; running eval"

# Eval v10c
python scripts/eval_quantitative.py \
  --head-ckpt results/coco_seg_v10c/best_head.pt \
  --hidden-cache-dir results/coco_seg_v10c/hidden_cache \
  --device mps --output-dir results/eval_v10c

echo "[chain] ALL DONE — v10, v10b, v10c complete"
