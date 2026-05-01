#!/bin/bash
# Watcher: when any of phase23a-d finishes (ckpt_step10000.pt appears),
# launch A5 (vanilla-class-balance ablation) on the freed GPU.
# Polls every 30s. Exits after launching A5.
#
# Plays nice with the parallel open-vocab agent: GPUs 1 and 5 are NEVER
# claimed by this watcher; A5 only fills a slot vacated by A1-A4.

set -u
cd "$(dirname "$0")/.."
PYBIN=/home/ogata/miniconda3/envs/hunter/bin/python
RESUME=results/phase15_atto_long_50k/ckpt_step35000.pt
TEACHER=/tmp/phase13_dinov2s_long_bestval_snapshot.pt
mkdir -p results/phase23_ablation_logs
LOG=results/phase23_ablation_logs/queue_phase23e.log
echo "[queue] watcher starting at $(date -Iseconds)" > "$LOG"

# slot map: which GPU each variant uses (per launch_phase23_ablation.sh)
declare -A SLOTS=(
  [phase23a_atto_nodistill_10k]=0
  [phase23b_atto_perquery_linear_10k]=2
  [phase23c_atto_noperquery_10k]=3
  [phase23d_atto_cocoonly_10k]=4
)

while true; do
  for slug in "${!SLOTS[@]}"; do
    if [ -f "results/$slug/ckpt_step10000.pt" ]; then
      gpu=${SLOTS[$slug]}
      echo "[queue] $slug finished -> launching phase23e on GPU $gpu at $(date -Iseconds)" | tee -a "$LOG"
      A5LOG="results/phase23_ablation_logs/phase23e_atto_vanillaclassbalance_10k.log"
      CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
        nohup "$PYBIN" -m scripts.train_siglip_dense_distill \
          --device cuda:0 \
          --output_dir results/phase23e_atto_vanillaclassbalance_10k \
          --wandb_run_name phase23e_atto_vanillaclassbalance_10k \
          --model convnext-atto \
          --target_dir results/clean_targets_unified_AB \
          --image_dir data/clean_images \
          --batch_size 16 --max_steps 10000 --epochs 200 \
          --lr 5e-4 --bias_init -2.0 --balanced_pos_frac 0.6 \
          --augment \
          --val_split_frac 0.05 --val_eval_every 2000 \
          --resume_from "$RESUME" \
          --wandb_project semantic-autogaze \
          --bce_pos_weight 1.0 \
          --source_weights "pp:5,stuff:1.5,lvis:1,coco:1,ade20k:1.5" \
          --per_query_bias \
          --distill_teacher_ckpt "$TEACHER" --lambda_distill 0.5 \
          > "$A5LOG" 2>&1 &
      echo "[queue] phase23e pid=$! gpu=$gpu" | tee -a "$LOG"
      exit 0
    fi
  done
  sleep 30
done
