#!/bin/bash
# Sequential K-sweep on GPU 5 for r/nvila-attn-k-sweep.
# K in {39, 60, 90}; semantic_keep_ratio in {0.1990, 0.3061, 0.4592}.
# Each cycle: ~50 min wall on a 24 GiB GPU. Total ~150 min.
set -euo pipefail
cd /home/ogata/semantic-autogaze
mkdir -p logs

run_k () {
  local k=$1
  local ratio=$2
  local outdir=$3
  local logfile=$4
  echo "=== Cycle K=${k} (semantic_keep_ratio=${ratio}) starting at $(date -Is) ===" | tee -a "${logfile}"
  CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 \
    /home/ogata/miniconda3/envs/hunter/bin/python \
    -m semantic_autogaze.eval_vqa_nvila_attn_bypass \
    --device cuda:0 \
    --semantic_keep_ratio "${ratio}" \
    --output_dir "${outdir}" >> "${logfile}" 2>&1
  echo "=== Cycle K=${k} done at $(date -Is) ===" | tee -a "${logfile}"
}

run_k 39 0.1990 results/nvila_attn_bypass_vqa_k039 logs/k_sweep_k039.log
run_k 60 0.3061 results/nvila_attn_bypass_vqa_k060 logs/k_sweep_k060.log
run_k 90 0.4592 results/nvila_attn_bypass_vqa_k090 logs/k_sweep_k090.log

echo "=== ALL THREE K cycles done at $(date -Is) ==="
