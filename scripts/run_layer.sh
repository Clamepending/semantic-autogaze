#!/usr/bin/env bash
# Run nvila-attn extraction (matched + shuffled) and VQA bypass at a given layer.
# Usage: bash scripts/run_layer.sh <gpu_id> <layer>
set -euo pipefail
GPU=${1:?gpu id}
LAYER=${2:?layer index}
cd /home/ogata/semantic-autogaze

CACHE_M="results/nvila_attention_cache_layer${LAYER}"
CACHE_S="results/nvila_attention_cache_layer${LAYER}_shuffled"
VQA_OUT="results/nvila_attn_bypass_vqa_layer${LAYER}_k27"

PY=/home/ogata/miniconda3/envs/hunter/bin/python
LOG=/tmp/run_layer_${LAYER}.log

echo "[layer ${LAYER}] GPU=${GPU} starting matched extraction" | tee "$LOG"
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY \
  -m semantic_autogaze.extract_nvila_attention_full_grid \
  --device cuda:0 --mode matched --layer "$LAYER" \
  --output_dir "$CACHE_M" 2>&1 | tee -a "$LOG"

echo "[layer ${LAYER}] starting shuffled extraction" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY \
  -m semantic_autogaze.extract_nvila_attention_full_grid \
  --device cuda:0 --mode shuffled --layer "$LAYER" \
  --output_dir "$CACHE_S" 2>&1 | tee -a "$LOG"

echo "[layer ${LAYER}] starting VQA bypass eval at K=27" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 $PY \
  -m semantic_autogaze.eval_vqa_nvila_attn_bypass \
  --device cuda:0 \
  --semantic_keep_ratio 0.1378 \
  --matched_cache_dir "$CACHE_M" \
  --shuffled_cache_dir "$CACHE_S" \
  --output_dir "$VQA_OUT" 2>&1 | tee -a "$LOG"

echo "[layer ${LAYER}] DONE" | tee -a "$LOG"
