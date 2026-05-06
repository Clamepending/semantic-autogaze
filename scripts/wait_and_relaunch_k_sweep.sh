#!/bin/bash
# Wait until a GPU has >=14000 MiB free, then run the K-sweep on it.
# Robust to one-time OOMs: any cycle that fails is logged and skipped; the
# next cycle still runs. Total wall ~150 min on a single 24 GiB GPU.
set -uo pipefail
cd /home/ogata/semantic-autogaze
mkdir -p logs

pick_gpu () {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' '$2+0 >= 14000 { print $1; exit 0 }'
}

while :; do
  GPU=$(pick_gpu || true)
  if [ -n "${GPU:-}" ]; then
    echo "$(date -Is) picked GPU ${GPU} (>=14000 MiB free)"
    break
  fi
  echo "$(date -Is) no GPU with >=14000 MiB free; sleeping 60s"
  sleep 60
done

run_k () {
  local k=$1
  local ratio=$2
  local outdir=$3
  local logfile=$4
  echo "=== Cycle K=${k} (semantic_keep_ratio=${ratio}) on GPU ${GPU} starting at $(date -Is) ===" | tee -a "${logfile}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/ogata/miniconda3/envs/hunter/bin/python \
    -m semantic_autogaze.eval_vqa_nvila_attn_bypass \
    --device cuda:0 \
    --semantic_keep_ratio "${ratio}" \
    --output_dir "${outdir}" >> "${logfile}" 2>&1
  local rc=$?
  if [ "${rc}" -ne 0 ]; then
    echo "=== Cycle K=${k} FAILED with rc=${rc} at $(date -Is) ===" | tee -a "${logfile}"
  else
    echo "=== Cycle K=${k} done at $(date -Is) ===" | tee -a "${logfile}"
  fi
}

run_k 39 0.1990 results/nvila_attn_bypass_vqa_k039 logs/k_sweep_k039.log
run_k 60 0.3061 results/nvila_attn_bypass_vqa_k060 logs/k_sweep_k060.log
run_k 90 0.4592 results/nvila_attn_bypass_vqa_k090 logs/k_sweep_k090.log

echo "=== ALL THREE K cycles done at $(date -Is) on GPU ${GPU} ==="
