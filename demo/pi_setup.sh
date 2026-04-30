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

MODEL="${MODEL:-phase15-atto}"
QUERY="${QUERY:-hand,coffee cup,laptop,face}"
REDUCE="${REDUCE:-max}"
PORT="${PORT:-8000}"
VENV="${VENV:-$HOME/.venv-pi-demo}"
WORKDIR="${WORKDIR:-$HOME/semantic-autogaze-pi-demo}"
case "$MODEL" in
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
  *) echo "Unknown MODEL=$MODEL (expected phase15-atto|phase10-atto|phase10-femto|phase10-pico)"; exit 1 ;;
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
echo "[pi_setup]   query='$QUERY' reduce=$REDUCE model=$MODEL"
exec python pi_webcam_server.py \
    --ckpt "$CKPT" \
    --model "$MODEL" \
    --query "$QUERY" \
    --reduce "$REDUCE" \
    --port "$PORT"
