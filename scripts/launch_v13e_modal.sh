#!/usr/bin/env bash
# v13e: within-image hard negatives + v10h/v11 recipe at scale.
#
# Hypothesis (see mac-brain/research/semantic-autogaze/v11-failure-audit.md):
#   v11 qual audit on 20 random val2017 examples shows ~73% of clear failures
#   fire on a labeled COCO co-occurring object (bowl→pizza, skis→person,
#   frisbee→person, fork→dining table+person+pizza, remote→laptop+book+mouse,
#   chair→person, apple→bottle, bench→teddy bear). Random absent negatives
#   never gave the head this within-image discrimination signal — the model
#   learned a "predict the salient object" shortcut.
#
#   v13e adds a per-patch BCE term (weighted) pushing pred→0 inside the
#   union of OTHER present categories' patch coverage in the same image.
#   On the bowl/pizza image that's a 469-patch penalty zone vs 20 patches
#   of positive — should overwhelm the saliency shortcut.
#
# Cost estimate: T4 @ $0.59/hr × ~6 hr ≈ $4. Always uses --detach.
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_NAME="${RUN_NAME:-coco_seg_v13e}"
OTHER_NEG_WEIGHT="${OTHER_NEG_WEIGHT:-2.0}"

TRAIN_ARGS=(
  --device cuda
  --num_epochs 10
  --batch_size 32
  --dropout 0.10
  --dice_weight 1.0
  --focal
  --focal_alpha 0.25
  --focal_gamma 2.0
  --decoder
  --out_grid 28
  --decoder_dim 128
  --clipseg_mix 0.0
  --neg_per_image 5
  --within_image_neg
  --other_neg_weight "${OTHER_NEG_WEIGHT}"
  --train_split train
  --num_qual_examples 20
)

echo "[v13e] launching detached on Modal as run_name=${RUN_NAME}"
echo "[v13e] other_neg_weight=${OTHER_NEG_WEIGHT}"
echo "[v13e] train args: ${TRAIN_ARGS[*]}"

modal run --detach scripts/modal_train.py \
  --run-name "${RUN_NAME}" \
  --train-args "${TRAIN_ARGS[*]}"
