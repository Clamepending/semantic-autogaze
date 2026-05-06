#!/usr/bin/env bash
# Tier-1 8-image qual-grid + n=50 follow-up + failure-category panel for the
# phase19 calibration / augmentation sweep (5 variants on top of phase15
# atto step 35000).
set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PYBIN="${PYBIN:-/home/ogata/miniconda3/envs/hunter/bin/python}"
WIKI_FIG="/home/ogata/mac-brain/projects/semantic-autogaze/figures"

OUT_ROOT="results/eval_phase19"
mkdir -p "$OUT_ROOT"

# (slug, ckpt) — eval the LAST step ckpt (10000) and best_val.pt of each.
PAIRS=(
  "phase19a_atto_aggrAugment_10k|results/phase19a_atto_aggrAugment_10k/ckpt_step10000.pt"
  "phase19a_atto_aggrAugment_best|results/phase19a_atto_aggrAugment_10k/best_val.pt"
  "phase19b_atto_perquery_10k|results/phase19b_atto_perquery_10k/ckpt_step10000.pt"
  "phase19b_atto_perquery_best|results/phase19b_atto_perquery_10k/best_val.pt"
  "phase19c_atto_calib_10k|results/phase19c_atto_calib_10k/ckpt_step10000.pt"
  "phase19c_atto_calib_best|results/phase19c_atto_calib_10k/best_val.pt"
  "phase19e_atto_kitchen_sink_10k|results/phase19e_atto_kitchen_sink_10k/ckpt_step10000.pt"
  "phase19e_atto_kitchen_sink_best|results/phase19e_atto_kitchen_sink_10k/best_val.pt"
  "phase19f_atto_perquery_aggrAug_10k|results/phase19f_atto_perquery_aggrAug_10k/ckpt_step10000.pt"
  "phase19f_atto_perquery_aggrAug_best|results/phase19f_atto_perquery_aggrAug_10k/best_val.pt"
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
  if [ -f "$out/coco50.json" ]; then
    echo "[sweep] SKIP $slug (already evaluated)"
    continue
  fi
  echo "[sweep] $(date -Iseconds) eval $slug"
  CUDA_VISIBLE_DEVICES="" "$PYBIN" -m scripts.eval_phase2_ckpt \
    --device cpu --ckpt "$ckpt" --output_dir "$out" --skip_demo \
    >> "$OUT_ROOT/sweep.log" 2>&1
  CUDA_VISIBLE_DEVICES="" "$PYBIN" -m scripts.eval_coco_val_miou \
    --device cpu --ckpt "$ckpt" --n_images 50 --seed 2026 \
    --output_path "$out/coco50.json" \
    >> "$OUT_ROOT/sweep.log" 2>&1
  CUDA_VISIBLE_DEVICES="" "$PYBIN" -m scripts.qual_failure_categories \
    --ckpt "$ckpt" --device cpu \
    --output_dir "$WIKI_FIG/v05x_failure_qual/$slug" \
    >> "$OUT_ROOT/sweep.log" 2>&1
done
echo "[sweep] done at $(date -Iseconds)"
