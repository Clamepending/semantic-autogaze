#!/usr/bin/env bash
# pi_setup.sh — one-shot install + first run for the Phase 10 SigLIP Pi demo.
# Tested target: Raspberry Pi 5 (Cortex-A76, 4 GB), Bookworm 64-bit. Pi 4
# Cortex-A72 also works at lower fps.
#
# Usage on the Pi:
#   curl -L https://raw.githubusercontent.com/Clamepending/semantic-autogaze/main/demo/pi_setup.sh -o pi_setup.sh
#   bash pi_setup.sh                     # default model = phase10-atto
#   MODEL=phase10-femto bash pi_setup.sh
#
# Then point a browser on the same network at http://<pi-host>:8000/.

set -euo pipefail

MODEL="${MODEL:-phase24d-atto}"
QUERY="${QUERY:-hand,coffee cup,laptop,face}"
PORT="${PORT:-8000}"
VENV="${VENV:-$HOME/.venv-pi-demo}"
WORKDIR="${WORKDIR:-$HOME/semantic-autogaze-pi-demo}"
case "$MODEL" in
  phase24d-atto)
    # v0.7.0: ConvNeXt-atto with per-query SigLIP bias (MLP form, same as
    # v0.6.0) + multi-prompt training (p=0.5, ECO/LMSeg-style training-time
    # prompt-template ensembling) + aggressive geometric aug. Identical
    # Pi 5 latency to v0.6.0 (~37 ms/frame, ~27 fps). Openvocab composite
    # +22 vs v0.6.0's +17 on the user's street validation panels (rank 1
    # of 29 ckpts evaluated 2026-05-01; OPENVOCAB_REFLECTION.md).
    # Indoor close-up + Pi-on-chest down-view scenes still have the
    # horizon-band prior (deployment-relevant indoor failure mode is
    # backbone-bound, fixed by phase25 DINOv2-s at 4-5x latency cost).
    RELEASE_TAG="v0.7.0-phase24d-pi-demo"
    CKPT="phase24d_convnext_atto_mpp05_aggrAug_best_v070.pt"
    ;;
  phase19-atto)
    # v0.6.0: ConvNeXt-atto with per-query SigLIP bias (DAC-style). +0.13
    # mean failure-cat IoU lift over v0.5.0; net 50-img mIoU 0.694 vs 0.681.
    RELEASE_TAG="v0.6.0-phase19-pi-demo"
    CKPT="phase19b_convnext_atto_perquery_best_v060.pt"
    ;;
  phase15-atto)
    RELEASE_TAG="v0.5.0-phase15-pi-demo"
    CKPT="phase15_convnext_atto_step35000_iou0722.pt"
    ;;
  phase10-atto)
    RELEASE_TAG="v0.4.0-phase10-pi-demo"
    CKPT="phase10_convnext_atto_best_val.pt"
    ;;
  phase10-femto)
    RELEASE_TAG="v0.4.0-phase10-pi-demo"
    CKPT="phase10_convnext_femto_best_val.pt"
    ;;
  phase10-pico)
    RELEASE_TAG="v0.4.0-phase10-pi-demo"
    CKPT="phase10_convnext_pico_best_val.pt"
    ;;
  *) echo "Unknown MODEL=$MODEL (expected phase24d-atto|phase19-atto|phase15-atto|phase10-atto|phase10-femto|phase10-pico)"; exit 1 ;;
esac

echo "[pi_setup] model=$MODEL ckpt=$CKPT port=$PORT venv=$VENV"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [ ! -d "$VENV" ]; then
  echo "[pi_setup] creating venv at $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet --upgrade pip wheel
pip install --quiet --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision timm numpy open_clip_torch flask opencv-python

if [ ! -f "$CKPT" ]; then
  URL="https://github.com/Clamepending/semantic-autogaze/releases/download/$RELEASE_TAG/$CKPT"
  echo "[pi_setup] fetching $URL"
  curl -fL "$URL" -o "$CKPT"
fi
if [ ! -f pi_webcam_server.py ]; then
  echo "[pi_setup] fetching pi_webcam_server.py"
  curl -fL "https://raw.githubusercontent.com/Clamepending/semantic-autogaze/main/demo/pi_webcam_server.py" -o pi_webcam_server.py
fi

echo
echo "[pi_setup] starting server on port $PORT — open http://$(hostname -I | awk '{print $1}'):$PORT/"
echo "[pi_setup]   query='$QUERY' model=$MODEL"
exec python pi_webcam_server.py \
    --ckpt "$CKPT" \
    --model "$MODEL" \
    --query "$QUERY" \
    --port "$PORT"
