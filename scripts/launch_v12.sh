#!/usr/bin/env bash
# v12: v10h recipe (focal+neg5+CLIP vision) scaled to COCO train2017 (118K images).
# Uses --clip_vision_online to compute CLIP ViT-B/16 features on-the-fly,
# avoiding the ~35GB disk cost of caching features for 118K images.
# Tests whether data scaling + CLIP vision features stack multiplicatively.
# Expected overhead vs v11: ~30-60% slower per batch due to CLIP forward passes.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

mkdir -p results/coco_seg_v12

# Symlink shared hidden caches to avoid re-caching
if [ ! -d results/coco_seg_v12/hidden_cache ] && [ -d results/coco_seg_v11/hidden_cache ]; then
  ln -s "$(pwd)/results/coco_seg_v11/hidden_cache" results/coco_seg_v12/hidden_cache
  echo "[v12] symlinked hidden_cache from v11"
fi
if [ ! -d results/coco_seg_v12/val_hidden_cache ] && [ -d results/coco_seg_v11/val_hidden_cache ]; then
  ln -s "$(pwd)/results/coco_seg_v11/val_hidden_cache" results/coco_seg_v12/val_hidden_cache
  echo "[v12] symlinked val_hidden_cache from v11"
fi

echo "[v12] launching: focal + neg5 + dice=1.0 + clipseg=0.5 + CLIP vision online + drop0.10 on train2017"
WANDB_MODE=offline python -u -m semantic_autogaze.train_coco_seg \
  --device mps \
  --num_epochs 10 \
  --batch_size 16 \
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
  --clip_vision_online \
  --train_split train \
  --output_dir results/coco_seg_v12 \
  --run_name coco-seg-v12-train2017-focal-neg5-clipvis-online \
  --num_qual_examples 20 \
  > results/coco_seg_v12/train.log 2>&1

echo "[v12] training done; running eval"
python scripts/eval_quantitative.py \
  --head-ckpt results/coco_seg_v12/best_head.pt \
  --hidden-cache-dir results/coco_seg_v12/val_hidden_cache \
  --clip-vision-online \
  --device mps --output-dir results/eval_v12

echo "[v12] ALL DONE"
