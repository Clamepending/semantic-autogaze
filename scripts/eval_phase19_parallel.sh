#!/usr/bin/env bash
# Parallel-across-GPUs version of eval_phase19_sweep.sh.
# Each phase19 variant gets its own GPU for the (8-img + n=50 + qual) eval.
# 5x faster than the CPU-sequential version when GPUs are free.
#
# Usage: bash scripts/eval_phase19_parallel.sh

set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PYBIN="${PYBIN:-/home/ogata/miniconda3/envs/hunter/bin/python}"
WIKI_FIG="/home/ogata/mac-brain/projects/semantic-autogaze/figures"
OUT_ROOT="results/eval_phase19"
mkdir -p "$OUT_ROOT"

# (slug, ckpt, gpu) — one variant per GPU.
TRIPLES=(
  "phase19a_atto_aggrAugment_10k|results/phase19a_atto_aggrAugment_10k/ckpt_step10000.pt|0"
  "phase19a_atto_aggrAugment_best|results/phase19a_atto_aggrAugment_10k/best_val.pt|0"
  "phase19b_atto_perquery_10k|results/phase19b_atto_perquery_10k/ckpt_step10000.pt|2"
  "phase19b_atto_perquery_best|results/phase19b_atto_perquery_10k/best_val.pt|2"
  "phase19c_atto_calib_10k|results/phase19c_atto_calib_10k/ckpt_step10000.pt|3"
  "phase19c_atto_calib_best|results/phase19c_atto_calib_10k/best_val.pt|3"
  "phase19e_atto_kitchen_sink_10k|results/phase19e_atto_kitchen_sink_10k/ckpt_step10000.pt|4"
  "phase19e_atto_kitchen_sink_best|results/phase19e_atto_kitchen_sink_10k/best_val.pt|4"
  "phase19f_atto_perquery_aggrAug_10k|results/phase19f_atto_perquery_aggrAug_10k/ckpt_step10000.pt|5"
  "phase19f_atto_perquery_aggrAug_best|results/phase19f_atto_perquery_aggrAug_10k/best_val.pt|5"
)

run_one() {
  local slug="$1" ckpt="$2" gpu="$3"
  local out="$OUT_ROOT/$slug"
  if [ ! -f "$ckpt" ]; then echo "[skip] $slug ckpt missing"; return 0; fi
  if [ -f "$out/coco50.json" ]; then echo "[skip] $slug already done"; return 0; fi
  echo "[start] $slug GPU=$gpu $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -m scripts.eval_phase2_ckpt \
    --device cuda:0 --ckpt "$ckpt" --output_dir "$out" --skip_demo \
    >> "$OUT_ROOT/sweep_${slug}.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -m scripts.eval_coco_val_miou \
    --device cuda:0 --ckpt "$ckpt" --n_images 50 --seed 2026 \
    --output_path "$out/coco50.json" \
    >> "$OUT_ROOT/sweep_${slug}.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -m scripts.qual_failure_categories \
    --ckpt "$ckpt" --device cuda:0 \
    --output_dir "$WIKI_FIG/v05x_failure_qual/$slug" \
    >> "$OUT_ROOT/sweep_${slug}.log" 2>&1
  echo "[done]  $slug $(date -Iseconds)"
}

# Group by GPU: each GPU gets a serial chain of its own ckpts. Then run all
# 5 chains in parallel (one per GPU).
run_chain_for_gpu() {
  local target_gpu="$1"
  for triple in "${TRIPLES[@]}"; do
    local slug="${triple%%|*}"
    local rest="${triple#*|}"
    local ckpt="${rest%%|*}"
    local gpu="${rest##*|}"
    if [ "$gpu" = "$target_gpu" ]; then
      run_one "$slug" "$ckpt" "$gpu"
    fi
  done
}

echo "[parallel-sweep] starting at $(date -Iseconds), 5 GPU chains"
for g in 0 2 3 4 5; do
  run_chain_for_gpu "$g" &
done
wait
echo "[parallel-sweep] done at $(date -Iseconds)"
