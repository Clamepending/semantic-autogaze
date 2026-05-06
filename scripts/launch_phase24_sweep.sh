#!/usr/bin/env bash
# Parallel multi-prompt / aux-objectness sweep launched after phase22d+phase24
# settled. Each entry: GPU + slug + extra trainer args.
#
# Stagger 60s between launches so the NFS dataset scans don't thrash each
# other (each scan touches 1.23M target files).
set -u
cd "$(dirname "$0")/.."
PYBIN="${PYBIN:-/home/ogata/miniconda3/envs/hunter/bin/python}"
BASE_ARGS=(
  --model convnext-atto
  --target_dir results/clean_targets_unified_AB
  --image_dir data/clean_images
  --batch_size 16
  --epochs 200
  --lr 5e-4
  --bias_init -2.0
  --bce_pos_weight 30
  --balanced_pos_frac 0.6
  --fn_filter
  --augment
  --source_weights pp:5,stuff:1.5,lvis:1,coco:1,ade20k:1.5
  --distill_teacher_ckpt /tmp/phase13_dinov2s_long_bestval_snapshot.pt
  --lambda_distill 0.5
  --val_split_frac 0.05
  --val_eval_every 2000
  --per_query_bias
  --wandb_project semantic-autogaze
)

launch_one() {
  local gpu="$1" slug="$2" outdir="$3"; shift 3
  local extra=("$@")
  echo "[launch] GPU=$gpu slug=$slug $(date -Iseconds)"
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

# Pre-create dirs so train.log paths exist
for d in \
  results/phase24a_atto_mpp025_10k \
  results/phase24b_atto_mpp075_10k \
  results/phase24d_atto_mpp05_aggrAug_10k \
  results/phase21b_atto_obj_from_v060_10k \
  results/phase24f_atto_mpp05_fromphase15_50k \
  results/phase24g_atto_mpp05_nodistill_10k; do
  mkdir -p "$d"
done

# 1. mpp025 — gentler multi-prompt
launch_one 0 phase24a_atto_mpp025_10k results/phase24a_atto_mpp025_10k \
  --max_steps 10000 \
  --resume_from results/phase19b_atto_perquery_10k/best_val.pt \
  --multi_prompt_training --multi_prompt_p 0.25
sleep 60

# 2. mpp075 — aggressive multi-prompt
launch_one 1 phase24b_atto_mpp075_10k results/phase24b_atto_mpp075_10k \
  --max_steps 10000 \
  --resume_from results/phase19b_atto_perquery_10k/best_val.pt \
  --multi_prompt_training --multi_prompt_p 0.75
sleep 60

# 3. mpp05 + aggressive aug — does multi-prompt + heavy geometric aug compose?
launch_one 2 phase24d_atto_mpp05_aggrAug_10k results/phase24d_atto_mpp05_aggrAug_10k \
  --max_steps 10000 \
  --resume_from results/phase19b_atto_perquery_10k/best_val.pt \
  --multi_prompt_training --multi_prompt_p 0.5 \
  --augment_aggressive
sleep 60

# 4. phase21b — aux objectness loss starting from v0.6.0 (not phase15)
#    Tests whether the regularization-on-per-query-bias effect from phase21
#    holds when the bias MLP is already trained.
launch_one 3 phase21b_atto_obj_from_v060_10k results/phase21b_atto_obj_from_v060_10k \
  --max_steps 10000 \
  --resume_from results/phase19b_atto_perquery_10k/best_val.pt \
  --objectness_weight 0.5 --objectness_pos_weight 5.0
sleep 60

# 5. mpp05 from phase15 (50K) — does multi-prompt benefit from longer training?
#    NOT a fine-tune: resumes from the 35K-step ImageNet-distill ckpt and
#    trains 50K on top. Heavier (~75 min) but biggest signal on whether
#    multi-prompt is just a fine-tune trick or a fundamental upgrade.
launch_one 4 phase24f_atto_mpp05_fromphase15_50k results/phase24f_atto_mpp05_fromphase15_50k \
  --max_steps 50000 \
  --resume_from results/phase15_atto_long_50k/ckpt_step35000.pt \
  --multi_prompt_training --multi_prompt_p 0.5
sleep 60

# GPU 5 is reserved (user constraint 2026-05-01) — phase24g is dropped from
# this sweep launcher. If you want to run it later, target a free GPU in
# {0,1,2,3,4}.

echo "[done] all 6 launched at $(date -Iseconds)"
