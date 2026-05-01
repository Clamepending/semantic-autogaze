#!/usr/bin/env bash
# Parallel-across-GPUs eval for the phase23 v0.6.0-simplification ablation.
# Mirrors eval_phase19_parallel.sh structure: each variant gets one GPU,
# eval = (8-img qual-grid + 50-img + qual-failure-categories panel).
#
# Usage: bash scripts/eval_phase23_parallel.sh [GPUS]
# Default GPUS = "0 2 3 4 5"
set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PYBIN="${PYBIN:-/home/ogata/miniconda3/envs/hunter/bin/python}"
WIKI_FIG="/home/ogata/mac-brain/projects/semantic-autogaze/figures"
OUT_ROOT="results/eval_phase23"
mkdir -p "$OUT_ROOT"

# (slug, ckpt, gpu)
TRIPLES=(
  "phase23a_atto_nodistill_10k|results/phase23a_atto_nodistill_10k/ckpt_step10000.pt|0"
  "phase23a_atto_nodistill_best|results/phase23a_atto_nodistill_10k/best_val.pt|0"
  "phase23b_atto_perquery_linear_10k|results/phase23b_atto_perquery_linear_10k/ckpt_step10000.pt|2"
  "phase23b_atto_perquery_linear_best|results/phase23b_atto_perquery_linear_10k/best_val.pt|2"
  "phase23c_atto_noperquery_10k|results/phase23c_atto_noperquery_10k/ckpt_step10000.pt|3"
  "phase23c_atto_noperquery_best|results/phase23c_atto_noperquery_10k/best_val.pt|3"
  "phase23d_atto_cocoonly_10k|results/phase23d_atto_cocoonly_10k/ckpt_step10000.pt|4"
  "phase23d_atto_cocoonly_best|results/phase23d_atto_cocoonly_10k/best_val.pt|4"
  "phase23e_atto_vanillaclassbalance_10k|results/phase23e_atto_vanillaclassbalance_10k/ckpt_step10000.pt|5"
  "phase23e_atto_vanillaclassbalance_best|results/phase23e_atto_vanillaclassbalance_10k/best_val.pt|5"
)

run_one() {
  local slug="$1" ckpt="$2" gpu="$3"
  local out="$OUT_ROOT/$slug"
  if [ ! -f "$ckpt" ]; then echo "[skip] $slug ckpt missing: $ckpt"; return 0; fi
  if [ -f "$out/coco50.json" ]; then echo "[skip] $slug already done"; return 0; fi
  echo "[start] $slug GPU=$gpu $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -m scripts.eval_phase2_ckpt \
    --device cuda:0 --ckpt "$ckpt" --output_dir "$out" --skip_demo \
    >> "$OUT_ROOT/sweep_${slug}.log" 2>&1 || echo "[warn] $slug 8-img failed"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -m scripts.eval_coco_val_miou \
    --device cuda:0 --ckpt "$ckpt" --n_images 50 --seed 2026 \
    --output_path "$out/coco50.json" \
    >> "$OUT_ROOT/sweep_${slug}.log" 2>&1 || echo "[warn] $slug 50-img failed"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -m scripts.qual_failure_categories \
    --ckpt "$ckpt" --device cuda:0 \
    --output_dir "$WIKI_FIG/v06x_failure_qual/$slug" \
    >> "$OUT_ROOT/sweep_${slug}.log" 2>&1 || echo "[warn] $slug qual failed"
  echo "[done]  $slug $(date -Iseconds)"
}

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

GPUS="${1:-0 2 3 4 5}"
echo "[parallel-sweep] starting at $(date -Iseconds), GPUs: $GPUS"
for g in $GPUS; do
  run_chain_for_gpu "$g" &
done
wait
echo "[parallel-sweep] done at $(date -Iseconds)"
