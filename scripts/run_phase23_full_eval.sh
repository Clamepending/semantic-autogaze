#!/usr/bin/env bash
# Full eval sweep for phase23 ablation + the TTA/prompt-ensemble companion
# experiments. Runs on a single specified GPU (or chains across many) in
# sequence so it works even when only one GPU is free.
#
# Usage: bash scripts/run_phase23_full_eval.sh [GPU]
#        Default GPU=1 (skip GPU 0 if A5/etc. is still going there).
set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PYBIN="${PYBIN:-/home/ogata/miniconda3/envs/hunter/bin/python}"
GPU="${1:-1}"
WIKI_FIG="/home/ogata/mac-brain/projects/semantic-autogaze/figures"
OUT_ROOT="results/eval_phase23"
mkdir -p "$OUT_ROOT"

# -------- (1) phase23 ckpts: 50-img + qual-failure-cats --------
# (slug, ckpt) pairs
PAIRS=(
  "phase23a_atto_nodistill_10k|results/phase23a_atto_nodistill_10k/ckpt_step10000.pt"
  "phase23a_atto_nodistill_best|results/phase23a_atto_nodistill_10k/best_val.pt"
  "phase23b_atto_perquery_linear_10k|results/phase23b_atto_perquery_linear_10k/ckpt_step10000.pt"
  "phase23b_atto_perquery_linear_best|results/phase23b_atto_perquery_linear_10k/best_val.pt"
  "phase23c_atto_noperquery_10k|results/phase23c_atto_noperquery_10k/ckpt_step10000.pt"
  "phase23c_atto_noperquery_best|results/phase23c_atto_noperquery_10k/best_val.pt"
  "phase23d_atto_cocoonly_10k|results/phase23d_atto_cocoonly_10k/ckpt_step10000.pt"
  "phase23d_atto_cocoonly_best|results/phase23d_atto_cocoonly_10k/best_val.pt"
  "phase23e_atto_vanillaclassbalance_10k|results/phase23e_atto_vanillaclassbalance_10k/ckpt_step10000.pt"
  "phase23e_atto_vanillaclassbalance_best|results/phase23e_atto_vanillaclassbalance_10k/best_val.pt"
)

for pair in "${PAIRS[@]}"; do
  slug="${pair%%|*}"; ckpt="${pair##*|}"
  out="$OUT_ROOT/$slug"
  if [ ! -f "$ckpt" ]; then echo "[skip] $slug ckpt missing"; continue; fi
  if [ -f "$out/coco50.json" ]; then echo "[skip] $slug coco50.json done"; continue; fi
  echo "[start] 50-img $slug $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYBIN" -m scripts.eval_coco_val_miou \
    --device cuda:0 --ckpt "$ckpt" --n_images 50 --seed 2026 \
    --output_path "$out/coco50.json" \
    >> "$OUT_ROOT/sweep_${slug}.log" 2>&1 || echo "[warn] $slug 50-img failed"
  echo "[start] qual $slug $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYBIN" -m scripts.qual_failure_categories \
    --ckpt "$ckpt" --device cuda:0 \
    --output_dir "$WIKI_FIG/v06x_failure_qual/$slug" \
    >> "$OUT_ROOT/sweep_${slug}.log" 2>&1 || echo "[warn] $slug qual failed"
done

# -------- (2) TTA + prompt-ensemble on v0.5.0, v0.6.0, A2 (winner) --------
TTA_CKPTS=(
  "v050_phase15_step35000|results/phase15_atto_long_50k/ckpt_step35000.pt"
  "v060_phase19b_best|results/phase19b_atto_perquery_10k/best_val.pt"
  "phase23b_atto_perquery_linear_best|results/phase23b_atto_perquery_linear_10k/best_val.pt"
)
TTA_MODES=(
  "tta|--tta"
  "pe|--prompt_ensemble"
  "tta_pe|--tta --prompt_ensemble"
)

for tckpt in "${TTA_CKPTS[@]}"; do
  slug="${tckpt%%|*}"; ckpt="${tckpt##*|}"
  if [ ! -f "$ckpt" ]; then echo "[skip-tta] $slug ckpt missing"; continue; fi
  for mode in "${TTA_MODES[@]}"; do
    mname="${mode%%|*}"; mflags="${mode##*|}"
    outpath="$OUT_ROOT/${slug}_${mname}/coco50.json"
    if [ -f "$outpath" ]; then echo "[skip] $slug $mname done"; continue; fi
    echo "[start] $slug $mname $(date -Iseconds)"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYBIN" -m scripts.tta_prompt_eval \
      --device cuda:0 --ckpt "$ckpt" --n_images 50 --seed 2026 $mflags \
      --output_path "$outpath" \
      >> "$OUT_ROOT/sweep_${slug}_${mname}.log" 2>&1 || echo "[warn] $slug $mname failed"
  done
done

echo "[done] full eval sweep at $(date -Iseconds)"
