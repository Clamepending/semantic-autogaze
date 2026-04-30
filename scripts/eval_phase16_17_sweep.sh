#!/usr/bin/env bash
# Tier-1 COCO 8-image qual-grid mIoU sweep for the Phase 16 + Phase 17
# training wave (kicked off ~2026-04-30 16:00 UTC, all candidates done by 19:06).
#
# Replicates the scripts.eval_phase2_ckpt --skip_demo recipe used by the
# resolved phase15-mIoU-eval move, on CPU only (no GPU contention).
#
# Output goes to results/eval_phase16_17/<slug>/qual_grid_metrics.json so the
# aggregator can pick winners directly.
#
# Usage:
#   bash scripts/eval_phase16_17_sweep.sh   (run in background; ~75 min total)

set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
PYBIN="${PYBIN:-/home/ogata/miniconda3/envs/hunter/bin/python}"

OUT_ROOT="results/eval_phase16_17"
mkdir -p "$OUT_ROOT"

# (slug, ckpt path) pairs — best_val.pt of each finished candidate.
PAIRS=(
  "phase16_atto_long_60k_best_val|results/phase16_atto_long_60k/best_val.pt"
  "phase16_atto_long_60k_step30000|results/phase16_atto_long_60k/ckpt_step30000.pt"
  "phase16_atto_long_60k_step32500|results/phase16_atto_long_60k/ckpt_step32500.pt"
  "phase16_atto_long_60k_step35000|results/phase16_atto_long_60k/ckpt_step35000.pt"
  "phase16_atto_long_60k_step37500|results/phase16_atto_long_60k/ckpt_step37500.pt"
  "phase16_atto_long_60k_step40000|results/phase16_atto_long_60k/ckpt_step40000.pt"
  "phase16_atto_long_60k_step42500|results/phase16_atto_long_60k/ckpt_step42500.pt"
  "phase16_atto_long_60k_step45000|results/phase16_atto_long_60k/ckpt_step45000.pt"
  "phase16_atto_long_60k_step50000|results/phase16_atto_long_60k/ckpt_step50000.pt"
  "phase16_atto_long_60k_step55000|results/phase16_atto_long_60k/ckpt_step55000.pt"
  "phase16_atto_long_60k_step60000|results/phase16_atto_long_60k/ckpt_step60000.pt"
  "phase16_atto_coco_heavy_40k_best_val|results/phase16_atto_coco_heavy_40k/best_val.pt"
  "phase16_atto_lambda03_best_val|results/phase16_atto_lambda03/best_val.pt"
  "phase16_atto_lambda07_best_val|results/phase16_atto_lambda07/best_val.pt"
  "phase16_atto_pool2_best_val|results/phase16_atto_pool2/best_val.pt"
  "phase16_mobilevit_xs_40k_best_val|results/phase16_mobilevit_xs_40k/best_val.pt"
  "phase17_convnext_femto_50k_best_val|results/phase17_convnext_femto_50k/best_val.pt"
  "phase17_convnext_pico_50k_best_val|results/phase17_convnext_pico_50k/best_val.pt"
  "phase17_atto_input280_best_val|results/phase17_atto_input280/best_val.pt"
)

echo "[sweep] starting at $(date -Iseconds), ${#PAIRS[@]} ckpts"
for pair in "${PAIRS[@]}"; do
  slug="${pair%%|*}"
  ckpt="${pair##*|}"
  out="$OUT_ROOT/$slug"
  if [ -f "$out/qual_grid_metrics.json" ]; then
    echo "[sweep] SKIP $slug (already evaluated)"
    continue
  fi
  if [ ! -f "$ckpt" ]; then
    echo "[sweep] SKIP $slug (ckpt missing: $ckpt)"
    continue
  fi
  echo "[sweep] $(date -Iseconds) eval $slug"
  "$PYBIN" -m scripts.eval_phase2_ckpt \
    --device cpu --ckpt "$ckpt" --output_dir "$out" --skip_demo \
    >> "$OUT_ROOT/sweep.log" 2>&1 || echo "[sweep] FAILED $slug (see sweep.log)"
done
echo "[sweep] done at $(date -Iseconds)"
