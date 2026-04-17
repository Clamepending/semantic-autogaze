#!/usr/bin/env bash
# v11: v10e recipe (focal+neg5+dice+clipseg0.5) scaled to COCO train2017 (118K images).
# Tests whether 24x more training data improves metrics.
# No CLIP vision features (too much disk for 118K images).
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

mkdir -p results/coco_seg_v11

echo "[v11] launching: focal + neg5 + dice=1.0 + clipseg=0.5 + drop0.10 on train2017"
WANDB_MODE=offline python -u -m semantic_autogaze.train_coco_seg \
  --device mps \
  --num_epochs 10 \
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
  --neg_per_image 5 \
  --train_split train \
  --output_dir results/coco_seg_v11 \
  --run_name coco-seg-v11-train2017-focal-neg5 \
  --num_qual_examples 20 \
  > results/coco_seg_v11/train.log 2>&1

echo "[v11] training done; running eval"
python scripts/eval_quantitative.py \
  --head-ckpt results/coco_seg_v11/best_head.pt \
  --hidden-cache-dir results/coco_seg_v11/val_hidden_cache \
  --device mps --output-dir results/eval_v11

echo "[v11] ALL DONE"
