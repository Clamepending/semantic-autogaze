#!/usr/bin/env bash
# Parallel sister recipes for v070 (in flight on GPU 0). Provides
# falsifier-targeted fallbacks if the primary fails its pre-stated band,
# and characterizes which of the 3 stacked knobs matter.
#
# GPU 5 is reserved (do not target).
set -u
cd "$(dirname "$0")/.."
PYBIN="${PYBIN:-/home/ogata/miniconda3/envs/hunter/bin/python}"
BASE_ARGS=(
  --model convnext-atto
  --target_dir results/clean_targets_unified_AB
  --image_dir data/clean_images
  --batch_size 16
  --max_steps 10000
  --epochs 200
  --lr 5e-4
  --bias_init -2.0
  --balanced_pos_frac 0.6
  --bce_pos_weight 30
  --fn_filter
  --per_query_bias
  --augment
  --distill_teacher_ckpt /tmp/phase13_dinov2s_long_bestval_snapshot.pt
  --lambda_distill 0.5
  --val_split_frac 0.05
  --val_eval_every 2000
  --resume_from results/phase15_atto_long_50k/ckpt_step35000.pt
  --wandb_project semantic-autogaze
)

launch_one() {
  local gpu="$1" slug="$2" outdir="$3"; shift 3
  local extra=("$@")
  echo "[launch] GPU=$gpu slug=$slug $(date -Iseconds)"
  mkdir -p "$outdir"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  PYTHONUNBUFFERED=1 \
  nohup "$PYBIN" -m scripts.train_siglip_dense_distill \
    "${BASE_ARGS[@]}" \
    --output_dir "$outdir" \
    --wandb_run_name "$slug" \
    "${extra[@]}" \
    > "$outdir/train.log" 2>&1 &
  echo "  PID=$! out=$outdir"
}

# 1. mpp 0.75 — phase24b's optimum, combined with cocoonly + aggrAug
launch_one 1 v070b_cocoonly_mpp075_aggrAug_10k results/v070b_cocoonly_mpp075_aggrAug_10k \
  --source_weights "coco:1,pp:0,stuff:0,lvis:0,ade20k:0" \
  --augment_aggressive \
  --multi_prompt_training --multi_prompt_p 0.75
sleep 60

# 2. cocoonly + multi-prompt only (NO aggrAug) — isolates aggrAug contribution
launch_one 2 v070c_cocoonly_mpp05_noAggr_10k results/v070c_cocoonly_mpp05_noAggr_10k \
  --source_weights "coco:1,pp:0,stuff:0,lvis:0,ade20k:0" \
  --multi_prompt_training --multi_prompt_p 0.5
sleep 60

# 3. cocoonly + aggrAug only (NO multi-prompt) — isolates multi-prompt contribution
launch_one 3 v070d_cocoonly_aggrAug_noMpp_10k results/v070d_cocoonly_aggrAug_noMpp_10k \
  --source_weights "coco:1,pp:0,stuff:0,lvis:0,ade20k:0" \
  --augment_aggressive
sleep 60

# 4. fullsource + multi-prompt + aggrAug (NOT cocoonly) — isolates A4 contribution
#    Tests whether the A4 simplification is hurting OOD vocab when combined.
launch_one 4 v070e_fullsource_mpp05_aggrAug_10k results/v070e_fullsource_mpp05_aggrAug_10k \
  --source_weights "pp:5,stuff:1.5,lvis:1,coco:1,ade20k:1.5" \
  --augment_aggressive \
  --multi_prompt_training --multi_prompt_p 0.5

echo "[done] all 4 sister recipes launched at $(date -Iseconds)"
