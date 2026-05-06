#!/usr/bin/env bash
# Tier-1 eval for phase21 + phase22 wave (4 variants, 2 ckpts each = 8 ckpts).
# Distributes across the 5 GPUs that aren't running training (GPU 1 = unrelated).
# Idempotent.
set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PYBIN="${PYBIN:-/home/ogata/miniconda3/envs/hunter/bin/python}"
WIKI_FIG="/home/ogata/mac-brain/projects/semantic-autogaze/figures"
OUT_ROOT="results/eval_phase21_22"
mkdir -p "$OUT_ROOT"

# (slug, ckpt, gpu) — staggered across 4 free GPUs.
TRIPLES=(
  "phase21_atto_objectness_10k|results/phase21_atto_objectness_10k/ckpt_step10000.pt|3"
  "phase21_atto_objectness_best|results/phase21_atto_objectness_10k/best_val.pt|3"
  "phase22c_atto_v060_reg_10k|results/phase22c_atto_v060_reg_10k/ckpt_step10000.pt|4"
  "phase22c_atto_v060_reg_best|results/phase22c_atto_v060_reg_10k/best_val.pt|4"
  "phase22a_atto_perquery_aug_50k|results/phase22a_atto_perquery_aug_50k/ckpt_step50000.pt|5"
  "phase22a_atto_perquery_aug_best|results/phase22a_atto_perquery_aug_50k/best_val.pt|5"
  "phase22b_atto_perquery_aug_reg_50k|results/phase22b_atto_perquery_aug_reg_50k/ckpt_step50000.pt|5"
  "phase22b_atto_perquery_aug_reg_best|results/phase22b_atto_perquery_aug_reg_50k/best_val.pt|5"
)

run_one() {
  local slug="$1" ckpt="$2" gpu="$3"
  local out="$OUT_ROOT/$slug"
  if [ ! -f "$ckpt" ]; then echo "[skip] $slug ckpt missing: $ckpt"; return 0; fi
  if [ -f "$out/coco50.json" ]; then echo "[skip] $slug already done"; return 0; fi
  echo "[start] $slug GPU=$gpu $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -m scripts.eval_phase2_ckpt \
    --device cuda:0 --ckpt "$ckpt" --output_dir "$out" --skip_demo \
    >> "$OUT_ROOT/sweep_${slug}.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -m scripts.eval_coco_val_miou \
    --device cuda:0 --ckpt "$ckpt" --n_images 50 --seed 2026 \
    --output_path "$out/coco50.json" \
    >> "$OUT_ROOT/sweep_${slug}.log" 2>&1
  echo "[done]  $slug $(date -Iseconds)"
}

# 3 GPU chains. GPU 3 has phase21 (2 ckpts). GPU 4 has phase22c (2 ckpts).
# GPU 5 has phase22a + phase22b (4 ckpts) — only fires when those trainings
# finish (the run_one SKIP-on-missing-ckpt makes this safe).
run_chain() {
  local target_gpu="$1"
  for triple in "${TRIPLES[@]}"; do
    local slug="${triple%%|*}"; local rest="${triple#*|}"
    local ckpt="${rest%%|*}"; local gpu="${rest##*|}"
    if [ "$gpu" = "$target_gpu" ]; then run_one "$slug" "$ckpt" "$gpu"; fi
  done
}

echo "[parallel-sweep] starting at $(date -Iseconds)"
for g in 3 4 5; do run_chain "$g" & done
wait
echo "[parallel-sweep] done at $(date -Iseconds)"
