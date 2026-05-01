#!/usr/bin/env bash
# Wait for the first GPU 0 10K run (phase24a) to finish, then launch
# phase26 (things-only objectness) on the freed GPU. Decoupled from
# the main sweep launcher so we don't block on it.
set -u
cd "$(dirname "$0")/.."
PYBIN="${PYBIN:-/home/ogata/miniconda3/envs/hunter/bin/python}"
TARGET_LOG="results/phase24a_atto_mpp025_10k/train.log"

# Wait for the marker line. The trainer prints it once both 10K is reached
# and final ckpt is saved.
until grep -q "step 10000\|max_steps reached\|saved final\|step  9990" "$TARGET_LOG" 2>/dev/null; do
  sleep 30
done
echo "[phase26] phase24a finished at $(date -Iseconds), launching"

OUT=results/phase26_atto_thingsonly_obj_10k
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
PYTHONUNBUFFERED=1 \
nohup "$PYBIN" -m scripts.train_siglip_dense_distill \
  --model convnext-atto \
  --target_dir results/clean_targets_unified_AB \
  --image_dir data/clean_images \
  --output_dir "$OUT" \
  --batch_size 16 \
  --max_steps 10000 \
  --epochs 200 \
  --lr 5e-4 \
  --bias_init -2.0 \
  --bce_pos_weight 30 \
  --balanced_pos_frac 0.6 \
  --fn_filter \
  --augment \
  --source_weights pp:5,stuff:1.5,lvis:1,coco:1,ade20k:1.5 \
  --distill_teacher_ckpt /tmp/phase13_dinov2s_long_bestval_snapshot.pt \
  --lambda_distill 0.5 \
  --val_split_frac 0.05 \
  --val_eval_every 2000 \
  --resume_from results/phase19b_atto_perquery_10k/best_val.pt \
  --per_query_bias \
  --objectness_weight 0.5 \
  --objectness_pos_weight 5.0 \
  --objectness_things_only \
  --wandb_project semantic-autogaze \
  --wandb_run_name phase26_atto_thingsonly_obj_10k \
  > "$OUT/train.log" 2>&1 &
PID=$!
echo "[phase26] PID=$PID out=$OUT GPU=0"
