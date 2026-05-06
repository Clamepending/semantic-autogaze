# CLIP → DINOv2 Text-to-Patch Semantic Adapter

A lightweight adapter that maps CLIP text embeddings into DINOv2's patch feature space, enabling open-vocabulary semantic heatmaps from any text prompt.

## Architecture

```
CLIP ViT-B/16 (frozen)              DINOv2-small (frozen)
       │                                    │
  text encoder                        image encoder
       │                                    │
  512-d embedding                384-d × 1024 patches (32×32 @ 448px)
       │                                    │
       ▼                                    │
┌──────────────────┐                        │
│  LayerNorm(512)  │                        │
│  Linear(512→512) │                        │
│  GELU            │  ← 460K trainable      │
│  Linear(512→384) │     parameters         │
│  L2-normalize    │                        │
└────────┬─────────┘                        │
         │                                  │
         └───── cosine similarity ──────────┘
                        │
                × logit_scale (learned)
                        │
                     sigmoid
                        │
                 32×32 heatmap
```

**Key design choices:**
- Only the small adapter is trained; both CLIP and DINOv2 remain frozen
- Uses DINOv2's spatial patch features (not CLS token) for localization
- Soft binary cross-entropy supervision from COCO instance segmentation masks
- All instances of the same category in an image are merged into one target mask

## Training

Trained on COCO 2017 train (337K image×category pairs) for 10K steps:
- Input: 448×448 images → 32×32 DINOv2-small patch grid
- Supervision: per-patch mask coverage fraction (soft BCE loss)
- Final train metrics: loss=0.19, patch-IoU=0.71
- wandb: https://wandb.ai/839/semantic-autogaze/runs/wwfu6rc4

```bash
uv run python scripts/train_clip_to_dino_adapter.py \
  --dino-size 448 --hidden-dim 512 --steps 10000 \
  --output-dir results/clip_text_to_dino_adapter_coco_10k_448_mlp
```

## Weights

| File | Size | Description |
|------|------|-------------|
| `clip_text_to_dino_adapter_coco_10k_448_mlp.pt` | 1.8 MB | PyTorch adapter weights |
| `onnx_int8/dinov2_small_patches_int8.onnx` | 25 MB | INT8 quantized DINOv2 (per-frame) |
| `onnx_int8/text_adapter.onnx` | 16 KB | ONNX adapter (once per prompt) |
| `onnx_int8/clip_text_embeddings.npz` | 140 KB | 65 pre-computed COCO category embeddings |

## Usage

### Interactive webapp (GPU server)

```bash
uv run python app_text_heatmap.py
# Open http://localhost:7861 — webcam + text prompt + threshold slider
```

### Raspberry Pi / edge deployment (CPU, no PyTorch)

```bash
pip install onnxruntime numpy opencv-python
python infer_rpi.py --category person --threshold 0.3
# Keys: +/- adjust threshold, q to quit
```

### Re-export ONNX INT8

```bash
uv run python scripts/export_onnx_int8.py \
  --adapter models/clip_text_to_dino_adapter_coco_10k_448_mlp.pt \
  --output-dir models/onnx_int8
```

## Quantization

Dynamic INT8 quantization via ONNX Runtime:
- FP32: 88 MB → INT8: **25 MB** (3.5× compression)
- Cosine similarity vs FP32: **0.978** (minimal quality loss)
- Target: Raspberry Pi 4/5 ARM CPU via `onnxruntime` CPUExecutionProvider
