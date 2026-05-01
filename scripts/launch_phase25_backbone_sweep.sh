#!/usr/bin/env bash
# Phase 25 cycle 1: 5-way backbone sweep on GPUs 0-4 (GPU 5 reserved).
# Tests whether a backbone with weaker spatial prior breaks the
# horizon-band failure that vshift augmentation could not.
#
# Variants:
#   GPU 0  phase25       DINOv2-s   + phase24d recipe (mpp + aggrAug)
#   GPU 1  phase25b      DINOv2-s   + minimal recipe  (no mpp/aggrAug)
#   GPU 2  phase26       MobileCLIPs2 + phase24d recipe
#   GPU 3  phase26b      MobileCLIPs2 + minimal recipe
#   GPU 4  phase27       FastViT-T8 + phase24d recipe
#
# Distillation: skip --distill_teacher_ckpt for DINOv2-s variants (target == teacher).
# Keep distill for MobileCLIP + FastViT.
set -u
cd "$(dirname "$0")/.."
PYBIN="${PYBIN:-/home/ogata/miniconda3/envs/hunter/bin/python}"

COMMON_BASE=(
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
  --source_weights pp:5,stuff:1.5,lvis:1,coco:1,ade20k:1.5
  --val_split_frac 0.05
  --val_eval_every 2000
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
    "${COMMON_BASE[@]}" \
    --output_dir "$outdir" \
    --wandb_run_name "$slug" \
    "${extra[@]}" \
    > "$outdir/train.log" 2>&1 &
  echo "  PID=$! out=$outdir"
}

# 1. DINOv2-s + phase24d recipe (PRIMARY)
launch_one 0 phase25_dinov2s_recipe_10k results/phase25_dinov2s_recipe_10k \
  --model dinov2-s \
  --augment_aggressive \
  --multi_prompt_training --multi_prompt_p 0.5
sleep 60

# 2. DINOv2-s baseline (no mpp/aggrAug; isolates backbone effect)
launch_one 1 phase25b_dinov2s_baseline_10k results/phase25b_dinov2s_baseline_10k \
  --model dinov2-s
sleep 60

# 3. MobileCLIP-S2 + phase24d recipe
launch_one 2 phase26_mobileclip_recipe_10k results/phase26_mobileclip_recipe_10k \
  --model mobileclip-s2 \
  --distill_teacher_ckpt /tmp/phase13_dinov2s_long_bestval_snapshot.pt \
  --lambda_distill 0.5 \
  --augment_aggressive \
  --multi_prompt_training --multi_prompt_p 0.5
sleep 60

# 4. MobileCLIP-S2 baseline
launch_one 3 phase26b_mobileclip_baseline_10k results/phase26b_mobileclip_baseline_10k \
  --model mobileclip-s2 \
  --distill_teacher_ckpt /tmp/phase13_dinov2s_long_bestval_snapshot.pt \
  --lambda_distill 0.5
sleep 60

# 5. FastViT-T8 + phase24d recipe
launch_one 4 phase27_fastvit_recipe_10k results/phase27_fastvit_recipe_10k \
  --model fastvit-t8 \
  --distill_teacher_ckpt /tmp/phase13_dinov2s_long_bestval_snapshot.pt \
  --lambda_distill 0.5 \
  --augment_aggressive \
  --multi_prompt_training --multi_prompt_p 0.5

echo "[done] all 5 backbone variants launched at $(date -Iseconds)"
