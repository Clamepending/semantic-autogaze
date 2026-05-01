#!/bin/bash
# phase23 — v0.6.0 simplification ablation (5-way knob removal).
# Launches A1-A4 in parallel on GPUs 0/2/3/4. A5 (vanilla class balance)
# is run separately after the first slot frees, leaving GPUs 1+5 for the
# parallel open-vocab agent.
#
# Each variant: 10K-step fine-tune from v0.5.0 (phase15_atto_long_50k step 35000).
# Wall: ~30 min per variant on RTX 4090.

set -euo pipefail
cd "$(dirname "$0")/.."

PYBIN=/home/ogata/miniconda3/envs/hunter/bin/python
RESUME=results/phase15_atto_long_50k/ckpt_step35000.pt
TEACHER=/tmp/phase13_dinov2s_long_bestval_snapshot.pt
COMMON=(--model convnext-atto
        --target_dir results/clean_targets_unified_AB
        --image_dir data/clean_images
        --batch_size 16 --max_steps 10000 --epochs 200
        --lr 5e-4 --bias_init -2.0 --balanced_pos_frac 0.6
        --augment
        --val_split_frac 0.05 --val_eval_every 2000
        --resume_from "$RESUME"
        --wandb_project semantic-autogaze)

mkdir -p results/phase23_ablation_logs

launch () {
  local gpu=$1; local name=$2; shift 2
  local logf="results/phase23_ablation_logs/${name}.log"
  echo "[launch] gpu=$gpu name=$name -> $logf"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
    nohup "$PYBIN" -m scripts.train_siglip_dense_distill \
      --device cuda:0 \
      --output_dir "results/${name}" \
      --wandb_run_name "$name" \
      "${COMMON[@]}" \
      "$@" > "$logf" 2>&1 &
  echo "  pid=$!"
}

# A1 - no distillation
launch 0 phase23a_atto_nodistill_10k \
  --bce_pos_weight 30 --fn_filter \
  --source_weights "pp:5,stuff:1.5,lvis:1,coco:1,ade20k:1.5" \
  --per_query_bias \
  --lambda_distill 0.0

# A2 - linear bias (no MLP hidden+GELU)
launch 2 phase23b_atto_perquery_linear_10k \
  --bce_pos_weight 30 --fn_filter \
  --source_weights "pp:5,stuff:1.5,lvis:1,coco:1,ade20k:1.5" \
  --per_query_bias --per_query_bias_kind linear \
  --distill_teacher_ckpt "$TEACHER" --lambda_distill 0.5

# A3 - drop per-query bias (just more v0.5.0 training)
launch 3 phase23c_atto_noperquery_10k \
  --bce_pos_weight 30 --fn_filter \
  --source_weights "pp:5,stuff:1.5,lvis:1,coco:1,ade20k:1.5" \
  --distill_teacher_ckpt "$TEACHER" --lambda_distill 0.5

# A4 - COCO-only data (drop pp/stuff/lvis explicit upweighting)
launch 4 phase23d_atto_cocoonly_10k \
  --bce_pos_weight 30 --fn_filter \
  --source_weights "coco:1,pp:0,stuff:0,lvis:0" \
  --per_query_bias \
  --distill_teacher_ckpt "$TEACHER" --lambda_distill 0.5

echo
echo "[launch] 4 trainings launched. Tail with:"
echo "  tail -f results/phase23_ablation_logs/phase23{a,b,c,d}*.log"
echo "[launch] A5 (vanilla class balance) will start when first slot frees."
