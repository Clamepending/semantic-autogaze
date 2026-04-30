#!/usr/bin/env bash
# Run after phase10-atto cycle 2 finishes (all 4 per_qid_*.json populated to 500).
# Aggregates -> launches the next wave of work on freed GPUs (skipping GPU 5).
#
# Usage:  bash scripts/post_cycle2_launch.sh
set -euo pipefail
cd /home/ogata/semantic-autogaze

OUT_ATTO=results/egoschema_phase10_atto
OUT_DINO=results/egoschema_phase6_dinov2s

# 1. Verify cycle 2 atto fully done
for cfg in vanilla match shuf rand; do
  N=$(/home/ogata/miniconda3/envs/hunter/bin/python -c "import json; print(len(json.load(open('$OUT_ATTO/per_qid_$cfg.json'))['per_q']))")
  echo "  atto $cfg n=$N"
  if [ "$N" -lt 500 ]; then
    echo "  [SKIP] $cfg not finished (n=$N < 500); aborting post-launch"
    exit 1
  fi
done

# 2. Run the aggregator (writes summary.json with paired flips)
/home/ogata/miniconda3/envs/hunter/bin/python scripts/agg_egoschema_phase10.py "$OUT_ATTO"

# 3. Launch wave 2:
#    - GPU 0: phase10-atto-train-48k (push toward >=0.71 mIoU demo target)
#    - GPU 2: phase10-femto-train-48k (close-second backbone)
#    - GPU 3: phase6-dinov2s match (queue row 2; partial — only match config to start)
#    - GPU 4: phase6-dinov2s shuf
#    - GPU 5: free (user constraint)
mkdir -p logs/wave2

# atto 48k on GPU 0
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup /home/ogata/miniconda3/envs/hunter/bin/python -u scripts/train_siglip_dense_distill.py \
    --model convnext-atto \
    --target_dir results/clean_targets_unified_AB \
    --max_steps 48000 \
    --lambda_distill 0.5 \
    --distill_teacher_ckpt results/phase4b_dinov2s_catbal/ckpt_step12000.pt \
    --output_dir results/phase12_convnext_atto_48k \
    --wandb_project semantic-autogaze \
    --wandb_run_name phase12-atto-48k \
    > logs/wave2/atto_48k_gpu0.log 2>&1 &
echo "[wave2] atto 48k PID=$!"

# femto 48k on GPU 2
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 nohup /home/ogata/miniconda3/envs/hunter/bin/python -u scripts/train_siglip_dense_distill.py \
    --model convnext-femto \
    --target_dir results/clean_targets_unified_AB \
    --max_steps 48000 \
    --lambda_distill 0.5 \
    --distill_teacher_ckpt results/phase4b_dinov2s_catbal/ckpt_step12000.pt \
    --output_dir results/phase12_convnext_femto_48k \
    --wandb_project semantic-autogaze \
    --wandb_run_name phase12-femto-48k \
    > logs/wave2/femto_48k_gpu2.log 2>&1 &
echo "[wave2] femto 48k PID=$!"

# phase6-dinov2s match on GPU 3
mkdir -p $OUT_DINO
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 nohup /home/ogata/miniconda3/envs/hunter/bin/python -u -m semantic_autogaze.eval_vqa_egoschema_phase10 \
    --device cuda:0 --scorer phase6-dinov2s --n_samples 500 \
    --output_dir $OUT_DINO --configs match \
    > logs/wave2/dinov2s_match_gpu3.log 2>&1 &
echo "[wave2] dinov2s match PID=$!"

# phase6-dinov2s shuf on GPU 4
CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 nohup /home/ogata/miniconda3/envs/hunter/bin/python -u -m semantic_autogaze.eval_vqa_egoschema_phase10 \
    --device cuda:0 --scorer phase6-dinov2s --n_samples 500 \
    --output_dir $OUT_DINO --configs shuf \
    > logs/wave2/dinov2s_shuf_gpu4.log 2>&1 &
echo "[wave2] dinov2s shuf PID=$!"

echo
echo "[wave2] launched.  GPU 5 left free per user constraint."
echo "       phase6-dinov2s rand will be launched once femto-48k frees a GPU"
echo "       (training takes ~85 min @ 24K * ~42 min, so ~85 min for 48k)."
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
