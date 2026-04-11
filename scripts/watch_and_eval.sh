#!/bin/bash
# Watch for training completion and run evaluations automatically.
# Usage: bash scripts/watch_and_eval.sh

set -e

PREDECODER_CKPT="results/distill_predecoder/best_predecoder_bighead.pt"
POSTDECODER_CKPT="results/distill_bighead/best_bighead_student.pt"

echo "Watching for pre-decoder training completion..."
echo "Checking every 60 seconds for: $PREDECODER_CKPT"

while true; do
    if [ -f "$PREDECODER_CKPT" ]; then
        echo ""
        echo "=========================================="
        echo "Pre-decoder checkpoint found! Running evaluation..."
        echo "=========================================="

        CUDA_VISIBLE_DEVICES=0 python3 -m semantic_autogaze.eval_predecoder \
            --predecoder_ckpt "$PREDECODER_CKPT" \
            --postdecoder_ckpt "$POSTDECODER_CKPT" \
            --predecoder_dir results/distill/predecoder_cache \
            --hidden_dir results/distill/hidden_cache \
            --clipseg_dir results/distill/clipseg_cache \
            --output_dir results/eval_predecoder_comparison \
            --batch_size 32 \
            --device cuda:0

        echo ""
        echo "Evaluation complete! Results in results/eval_predecoder_comparison/"
        break
    fi

    # Show current training status
    TEACHER_EPOCH=$(cat /proc/$(pgrep -f train_distill_predecoder | head -1)/fd/2 2>/dev/null | grep -oP "(Teacher|Student) \d+/\d+" | tail -1)
    echo "$(date '+%H:%M:%S') — Training: ${TEACHER_EPOCH:-unknown}"
    sleep 60
done
