#!/bin/bash
# Autonomous GPU dispatcher for phase29 / phase30 sweep.
#
# Polls every 30s. Two responsibilities, in order:
#   1) For each results/phase29*/ that has best_val.pt but no
#      figures/openvocab_eval/<SLUG>/summary.csv → run bench eval on a free GPU.
#   2) For each pending experiment in scripts/phase29_queue.json (priority
#      ascending) where results/<output_dir>/best_val.pt does NOT exist →
#      launch on a free GPU.
#
# A GPU is "free" if its VRAM used < 4 GB AND no train_siglip_dense_distill
# process pinned to it via CUDA_VISIBLE_DEVICES is alive.
#
# Idempotent: safe to restart, no state file required (uses presence of
# best_val.pt and summary.csv as canonical state).
#
# Stop with: pkill -f phase29_dispatcher.sh   (in-flight trainings keep going)
set -u
cd /home/ogata/semantic-autogaze
export PYTHONUNBUFFERED=1
PYBIN=/home/ogata/miniconda3/envs/hunter/bin/python
QUEUE=/home/ogata/semantic-autogaze/scripts/phase29_queue.json
LOG=/tmp/phase29_dispatch.log
WIKI=/home/ogata/mac-brain/projects/semantic-autogaze/figures
LEADERBOARD_TXT=/tmp/phase29_leaderboard.txt
RESULTS=/home/ogata/semantic-autogaze/results

ALLOWED_GPUS=(0 1 2 3 4)
VRAM_FREE_THRESHOLD_MB=4000
POLL_SLEEP=30

log(){ echo "[$(date -Iseconds)] $*" | tee -a "$LOG" ; }

vram_used(){
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '
}

# Returns 0 (true) if any compute process is running on this GPU.
# Uses nvidia-smi --query-compute-apps which sees processes regardless of how
# CUDA_VISIBLE_DEVICES was set (env var doesn't appear in argv).
gpu_in_use(){
  local gpu=$1
  local count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null | grep -cv '^$')
  [[ "$count" -gt 0 ]] && return 0
  return 1
}

# Also check VRAM as backup — nvidia-smi compute-apps sometimes lags by a few
# seconds. A GPU with > 500 MB used has SOMETHING running on it.
gpu_has_vram(){
  local gpu=$1
  local v=$(vram_used "$gpu")
  [[ -z "$v" ]] && return 1
  (( v > 500 ))
}

find_free_gpu(){
  for g in "${ALLOWED_GPUS[@]}"; do
    if gpu_in_use "$g"; then continue; fi
    if gpu_has_vram "$g"; then continue; fi
    echo "$g"; return 0
  done
  return 1
}

find_free_gpu_excluding(){
  local skip="$1"
  for g in "${ALLOWED_GPUS[@]}"; do
    [[ "$g" == "$skip" ]] && continue
    if gpu_in_use "$g"; then continue; fi
    if gpu_has_vram "$g"; then continue; fi
    echo "$g"; return 0
  done
  return 1
}

# Find a results dir whose training completed ([done]) and has best_val.pt but no eval CSV.
# IMPORTANT: requires "[done] training finished" in train.log — prevents
# the dispatcher from running bench eval on intermediate-checkpoint best_val.pt
# while training is still in flight (would produce stale numbers).
needs_eval(){
  for d in "$RESULTS"/phase29* "$RESULTS"/phase30* "$RESULTS"/phase31* "$RESULTS"/phase32* "$RESULTS"/phase33*; do
    [[ -d "$d" ]] || continue
    [[ -f "$d/best_val.pt" ]] || continue
    [[ -f "$d/train.log" ]] || continue
    grep -q "^\[done\] training finished" "$d/train.log" 2>/dev/null || continue
    local slug=$(basename "$d")
    local bench="$WIKI/openvocab_eval/${slug}"
    [[ -f "$bench/summary.csv" ]] && continue
    if pgrep -af "qual_openvocab_eval.*$d/best_val.pt" >/dev/null 2>&1; then continue; fi
    echo "$d"; return 0
  done
  return 1
}

run_bench_eval(){
  local results_dir=$1
  local gpu=$2
  local slug=$(basename "$results_dir")
  local bench="$WIKI/openvocab_eval/${slug}"
  mkdir -p "$bench"
  log "[eval] $slug on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PYBIN" -m scripts.qual_openvocab_eval \
    --device cuda:0 --ckpt "$results_dir/best_val.pt" \
    --output_dir "$bench" > "$bench/eval.log" 2>&1 &
}

# Pull the highest-priority queued experiment whose output_dir lacks best_val.pt
next_queued_command(){
  "$PYBIN" - <<EOF
import json, os, sys
queue = json.load(open("$QUEUE"))["experiments"]
queue.sort(key=lambda e: e["priority"])
for e in queue:
    cmd = e["command"]
    out = None
    if "--output_dir" in cmd:
        tok = cmd.split()
        try:
            i = tok.index("--output_dir")
            out = tok[i+1]
        except ValueError:
            continue
    if out and os.path.exists(os.path.join(out, "best_val.pt")):
        continue
    # Skip if train.log exists — that means a training was already started
    # for this output_dir (it may still be running, may have died, may have
    # completed without best_val). Either way, do not auto-relaunch; let a
    # human inspect.
    if out and os.path.exists(os.path.join(out, "train.log")):
        continue
    # Skip if a train process is already running for this output_dir (covers
    # the gap between launch and train.log appearing). pgrep matches against
    # the FULL command line, so the path appears with various neighbors.
    import subprocess
    rc = subprocess.run(["pgrep", "-af", out], capture_output=True).returncode
    if rc == 0:
        continue
    print(e["slug"])
    print(e["command"])
    sys.exit(0)
sys.exit(1)
EOF
}

launch_experiment(){
  local slug=$1
  local cmd=$2
  local gpu=$3
  local out_dir=$(echo "$cmd" | grep -oP -- '--output_dir \S+' | awk '{print $2}')
  mkdir -p "$out_dir"
  # Touch train.log IMMEDIATELY so the next_queued_command's train.log check
  # detects this output_dir as already-launched even before the trainer
  # starts writing to it. This avoids the race where dispatcher polls again
  # before the bash subprocess and python interpreter have spun up.
  echo "[dispatcher] launching at $(date -Iseconds) on GPU $gpu" > "$out_dir/train.log"
  log "[launch] $slug on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu nohup bash -c "$cmd >> $out_dir/train.log 2>&1" \
    > "$out_dir/wrapper.log" 2>&1 &
}

post_leaderboard(){
  local slugs
  slugs=$(ls "$WIKI/openvocab_eval/" 2>/dev/null | grep -E "^phase2[9]|^phase3[0-9]|^phase24d_atto_mpp05_aggrAug_best$|^baseline_clipseg_rd64$|^v060_phase19b_perquery_best$" | tr '\n' ',' | sed 's/,$//')
  [[ -z "$slugs" ]] && return
  "$PYBIN" -m scripts.compare_openvocab_sweep --slugs "$slugs" 2>/dev/null > "$LEADERBOARD_TXT"
}

log "=== dispatcher started ==="
log "queue: $QUEUE  log: $LOG"

while true; do
  # 1) Run AT MOST ONE bench eval this cycle (race-safe with nvidia-smi lag)
  used_gpu=""
  results_dir=$(needs_eval || true)
  if [[ -n "$results_dir" ]]; then
    gpu=$(find_free_gpu || true)
    if [[ -n "$gpu" ]]; then
      run_bench_eval "$results_dir" "$gpu"
      used_gpu="$gpu"
    fi
  fi

  # 2) Update leaderboard
  post_leaderboard

  # 3) Launch AT MOST ONE new training this cycle, NOT on the GPU we just
  #    started an eval on (nvidia-smi lag means find_free_gpu can't see it
  #    yet). next cycle will pick up the next experiment.
  qcmd=$(next_queued_command || true)
  if [[ -n "$qcmd" ]]; then
    slug=$(echo "$qcmd" | head -1)
    cmd=$(echo "$qcmd" | sed -n '2p')
    if [[ -n "$cmd" ]]; then
      gpu=$(find_free_gpu_excluding "$used_gpu" || true)
      if [[ -n "$gpu" ]]; then
        launch_experiment "$slug" "$cmd" "$gpu"
      fi
    fi
  fi

  sleep "$POLL_SLEEP"
done
