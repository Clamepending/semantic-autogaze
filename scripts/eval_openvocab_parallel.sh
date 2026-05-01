#!/usr/bin/env bash
# Parallel open-vocab qualitative eval across all available ckpts.
# Each ckpt gets one GPU; runs qual_openvocab_eval.py producing a panel
# per validation image. Idempotent (skips already-rendered).
#
# Usage:
#   bash scripts/eval_openvocab_parallel.sh
set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PYBIN="${PYBIN:-/home/ogata/miniconda3/envs/hunter/bin/python}"
WIKI_FIG="/home/ogata/mac-brain/projects/semantic-autogaze/figures/openvocab_eval"
mkdir -p "$WIKI_FIG"

# (slug, ckpt_path, gpu_id, extra_flags) — one ckpt per GPU. GPU 1 is unrelated work.
# When the ckpt has an objectness head, it is applied by default; pass
# "--no_objectness" in extra_flags to A/B compare ungated.
TRIPLES=(
  "v050_phase15_step35000|results/phase15_atto_long_50k/ckpt_step35000.pt|0|"
  "v060_phase19b_perquery_best|results/phase19b_atto_perquery_10k/best_val.pt|2|"
  "phase21_atto_objectness_gated|results/phase21_atto_objectness_10k/best_val.pt|3|"
  "phase21_atto_objectness_ungated|results/phase21_atto_objectness_10k/best_val.pt|3|--no_objectness"
  "phase22c_atto_v060_reg_best|results/phase22c_atto_v060_reg_10k/best_val.pt|4|"
  "phase22a_atto_perquery_aug_best|results/phase22a_atto_perquery_aug_50k/best_val.pt|5|"
  "phase22b_atto_perquery_aug_reg_best|results/phase22b_atto_perquery_aug_reg_50k/best_val.pt|5|"
  "phase22d_atto_v060_reg005_best|results/phase22d_atto_v060_reg005_10k/best_val.pt|0|"
  "phase24_atto_multiprompt_best|results/phase24_atto_multiprompt_10k/best_val.pt|2|"
  "phase18c_atto_lowposw_10k|results/phase18c_atto_lowposw_10k/ckpt_step10000.pt|0|"
  "phase19a_atto_aggrAugment_best|results/phase19a_atto_aggrAugment_10k/best_val.pt|2|"
  "phase19c_atto_calib_best|results/phase19c_atto_calib_10k/best_val.pt|3|"
  "phase19f_atto_perquery_aggrAug_best|results/phase19f_atto_perquery_aggrAug_10k/best_val.pt|4|"
)

run_one() {
  local slug="$1" ckpt="$2" gpu="$3" extra="$4"
  local out="$WIKI_FIG/$slug"
  if [ ! -f "$ckpt" ]; then echo "[skip] $slug ckpt missing: $ckpt"; return 0; fi
  if [ -f "$out/summary.csv" ]; then echo "[skip] $slug already done"; return 0; fi
  mkdir -p "$out"
  echo "[start] $slug GPU=$gpu extra='$extra' $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -m scripts.qual_openvocab_eval \
    --device cuda:0 --ckpt "$ckpt" --output_dir "$out" $extra \
    >> "$out/eval.log" 2>&1
  echo "[done]  $slug $(date -Iseconds)"
}

run_chain() {
  local target_gpu="$1"
  for entry in "${TRIPLES[@]}"; do
    local slug rest ckpt gpu extra
    IFS='|' read -r slug ckpt gpu extra <<< "$entry"
    if [ "$gpu" = "$target_gpu" ]; then run_one "$slug" "$ckpt" "$gpu" "$extra"; fi
  done
}

echo "[parallel-openvocab-sweep] starting at $(date -Iseconds)"
for g in 0 2 3 4 5; do run_chain "$g" & done
wait
echo "[parallel-openvocab-sweep] done at $(date -Iseconds)"
