#!/bin/bash
# Cycle 1 of r/nvila-attn-layer-head-sweep: extract NVILA attention at
# layer 16 head-mean (matched + shuffled), then VQA at K=27.
# ~84 min total wall on a 24 GiB GPU.
set -uo pipefail
cd /home/ogata/semantic-autogaze
mkdir -p logs

GPU="${1:-0}"
echo "=== layer-16 cycle 1 starting on GPU ${GPU} at $(date -Is) ===" | tee logs/layer16_orchestrator.log

run_step () {
  local name=$1
  local logfile=$2
  shift 2
  echo "--- $name @ $(date -Is) ---" | tee -a logs/layer16_orchestrator.log
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/ogata/miniconda3/envs/hunter/bin/python "$@" >> "${logfile}" 2>&1
  local rc=$?
  echo "--- $name done rc=${rc} @ $(date -Is) ---" | tee -a logs/layer16_orchestrator.log
  if [ "${rc}" -ne 0 ]; then
    echo "!!! $name FAILED — aborting orchestrator" | tee -a logs/layer16_orchestrator.log
    exit "${rc}"
  fi
}

# Step 1: extract matched at layer 16
run_step "extract matched layer16" logs/layer16_extract_matched.log \
  -m semantic_autogaze.extract_nvila_attention_full_grid \
  --device cuda:0 --mode matched --layer 16 \
  --output_dir results/nvila_attention_cache_layer16

# Step 2: extract shuffled at layer 16
run_step "extract shuffled layer16" logs/layer16_extract_shuffled.log \
  -m semantic_autogaze.extract_nvila_attention_full_grid \
  --device cuda:0 --mode shuffled --layer 16 \
  --output_dir results/nvila_attention_cache_layer16_shuffled

# Step 3: VQA at K=27 (semantic_keep_ratio=0.1378)
run_step "vqa layer16 K=27" logs/layer16_vqa_k27.log \
  -m semantic_autogaze.eval_vqa_nvila_attn_bypass \
  --device cuda:0 \
  --semantic_keep_ratio 0.1378 \
  --matched_cache_dir results/nvila_attention_cache_layer16 \
  --shuffled_cache_dir results/nvila_attention_cache_layer16_shuffled \
  --output_dir results/nvila_attn_bypass_vqa_layer16_k27

echo "=== layer-16 cycle 1 ALL DONE on GPU ${GPU} at $(date -Is) ===" | tee -a logs/layer16_orchestrator.log
