#!/usr/bin/env bash
# Tier-1 8-image qual-grid + n=50 follow-up + failure-category panel for the
# phase18 hard-negative-mining sweep (4 variants on top of phase15 atto step 35000).
set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PYBIN="${PYBIN:-/home/ogata/miniconda3/envs/hunter/bin/python}"
WIKI_FIG="/home/ogata/mac-brain/projects/semantic-autogaze/figures"

OUT_ROOT="results/eval_phase18"
mkdir -p "$OUT_ROOT"

# (slug, ckpt-glob) — we eval the LAST step ckpt (10000) and best.pt of each.
PAIRS=(
  "phase18_atto_hardneg_w1_10k|results/phase18_atto_hardneg_10k/ckpt_step10000.pt"
  "phase18_atto_hardneg_w1_best|results/phase18_atto_hardneg_10k/best_val.pt"
  "phase18b_atto_hardneg_w3_10k|results/phase18b_atto_hardneg_w3_10k/ckpt_step10000.pt"
  "phase18b_atto_hardneg_w3_best|results/phase18b_atto_hardneg_w3_10k/best_val.pt"
  "phase18c_atto_lowposw_10k|results/phase18c_atto_lowposw_10k/ckpt_step10000.pt"
  "phase18c_atto_lowposw_best|results/phase18c_atto_lowposw_10k/best_val.pt"
  "phase18d_atto_combined_10k|results/phase18d_atto_combined_10k/ckpt_step10000.pt"
  "phase18d_atto_combined_best|results/phase18d_atto_combined_10k/best_val.pt"
)

echo "[sweep] starting at $(date -Iseconds), ${#PAIRS[@]} ckpts"
for pair in "${PAIRS[@]}"; do
  slug="${pair%%|*}"
  ckpt="${pair##*|}"
  out="$OUT_ROOT/$slug"
  if [ ! -f "$ckpt" ]; then
    echo "[sweep] SKIP $slug (ckpt missing: $ckpt)"
    continue
  fi
  if [ -f "$out/phase2_qual_grid.png" ] && [ -f "$out/coco50.json" ]; then
    echo "[sweep] SKIP $slug (already evaluated)"
    continue
  fi
  echo "[sweep] $(date -Iseconds) eval $slug -- 8-img + n=50"
  CUDA_VISIBLE_DEVICES="" "$PYBIN" -m scripts.eval_phase2_ckpt \
    --device cpu --ckpt "$ckpt" --output_dir "$out" --skip_demo \
    >> "$OUT_ROOT/sweep.log" 2>&1
  CUDA_VISIBLE_DEVICES="" "$PYBIN" -m scripts.eval_coco_val_miou \
    --device cpu --ckpt "$ckpt" --n_images 50 --seed 2026 \
    --output_path "$out/coco50.json" \
    >> "$OUT_ROOT/sweep.log" 2>&1
  # Failure-category qual panel (also CPU)
  CUDA_VISIBLE_DEVICES="" "$PYBIN" -m scripts.qual_failure_categories \
    --ckpt "$ckpt" --device cpu \
    --output_dir "$WIKI_FIG/v05x_failure_qual/$slug" \
    >> "$OUT_ROOT/sweep.log" 2>&1
done
echo "[sweep] done at $(date -Iseconds)"
