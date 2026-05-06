#!/usr/bin/env bash
# Launches phase20 + phase21 architectural experiments in parallel.
# Use only after phase19 evals come back negative (Branch C in NEXT_STEPS.md).
#
# phase20: 28x28 grid output via TextScorerHeadGrid28 + on-the-fly mask28
#   resampling from mask_full. Addresses C2 (scale invariance) +
#   C3 (spatial offset).
# phase21: query-agnostic objectness head per OWLv2 (arxiv 2306.09683).
#   Addresses C4 (abstention).
set -u
cd /home/ogata/semantic-autogaze
PYBIN="/home/ogata/miniconda3/envs/hunter/bin/python"
mkdir -p results/phase20_atto_grid28_10k results/phase21_atto_objectness_10k

echo "[launch] phase20 (GPU 0) — 28x28 grid, resume from v0.5.0"
nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
  $PYBIN -m scripts.train_siglip_dense_distill \
    --model convnext-atto --target_dir results/clean_targets_unified_AB \
    --image_dir data/clean_images \
    --output_dir results/phase20_atto_grid28_10k \
    --batch_size 16 --max_steps 10000 --epochs 200 \
    --lr 5e-4 --bias_init -2.0 --bce_pos_weight 30 --balanced_pos_frac 0.6 \
    --fn_filter --augment \
    --source_weights "pp:5,stuff:1.5,lvis:1,coco:1,ade20k:1.5" \
    --distill_teacher_ckpt /tmp/phase13_dinov2s_long_bestval_snapshot.pt \
    --lambda_distill 0.5 \
    --val_split_frac 0.05 --val_eval_every 2000 \
    --resume_from results/phase15_atto_long_50k/ckpt_step35000.pt \
    --grid_size_out 28 \
    --wandb_project semantic-autogaze --wandb_run_name phase20_atto_grid28_10k \
    > results/phase20_atto_grid28_10k/train.log 2>&1 &
echo "phase20 PID=$!"

sleep 60   # NFS-stagger to avoid scan thrashing

echo "[launch] phase21 (GPU 2) — objectness head, resume from v0.5.0"
nohup env CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
  $PYBIN -m scripts.train_siglip_dense_distill \
    --model convnext-atto --target_dir results/clean_targets_unified_AB \
    --image_dir data/clean_images \
    --output_dir results/phase21_atto_objectness_10k \
    --batch_size 16 --max_steps 10000 --epochs 200 \
    --lr 5e-4 --bias_init -2.0 --bce_pos_weight 30 --balanced_pos_frac 0.6 \
    --fn_filter --augment \
    --source_weights "pp:5,stuff:1.5,lvis:1,coco:1,ade20k:1.5" \
    --distill_teacher_ckpt /tmp/phase13_dinov2s_long_bestval_snapshot.pt \
    --lambda_distill 0.5 \
    --val_split_frac 0.05 --val_eval_every 2000 \
    --resume_from results/phase15_atto_long_50k/ckpt_step35000.pt \
    --objectness_weight 0.5 --objectness_pos_weight 5.0 \
    --wandb_project semantic-autogaze --wandb_run_name phase21_atto_objectness_10k \
    > results/phase21_atto_objectness_10k/train.log 2>&1 &
echo "phase21 PID=$!"

echo "[launch] both fired. ETA ~25-30 min wall (NFS scan + 10K steps)."
