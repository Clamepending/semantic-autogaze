# Real-time webcam demo (Mac)

Stream your webcam (or FaceTime camera), type a query like `"hand"` or `"laptop"`, and see the trained scorer's heatmap overlaid live.

## What you're seeing

The demo runs the same trained scorer the rest of this project benchmarks (CLIP text encoder + a small visual backbone + a learned text-conditional head trained on COCO + CLIPSeg-soft + Ours-v1-soft). It produces a 14×14 sigmoid score map per frame, upsampled and overlaid on the camera feed with a `HOT` colormap.

Three model sizes are selectable:

| model | backbone | total params | desktop FPS class | Mac FPS class |
|---|---|---:|---:|---:|
| `d-mobile` (default) | MobileNet-V3-Small | 5.05 M | ~50-100 | **~15-30 (M1/M2)** |
| `v2-tiny` | ViT-Tiny/16 | 8.9 M | ~80-120 | ~5-15 (M1), ~15-25 (M3) |
| `v1` | CLIP ViT-B/16 | 91.6 M | ~30-50 | ~2-5 (CPU), ~10-20 (MPS) |

D-Mobile is the Pi-class winner; v2-Tiny is the desktop-class winner; v1 is the highest-quality (mIoU 0.79 on COCO ≈ CLIPSeg's 0.81). Cycle between them at runtime with the `m` key.

## Install (Mac)

```bash
git clone https://github.com/Clamepending/semantic-autogaze.git
cd semantic-autogaze
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision timm opencv-python open_clip_torch Pillow
```

PyTorch should auto-detect Apple Silicon (MPS). Verify with:

```bash
python3 -c "import torch; print('mps:', torch.backends.mps.is_available())"
```

## Run

```bash
python demo/realtime_webcam_demo.py --query "hand"
```

The first run downloads the head checkpoint (~14 MB) from the GH release and pulls the backbone weights via `timm` / `open_clip` (one-time, ~50 MB combined). After that it's all local.

### Hotkeys

Focus the OpenCV window first, then:

| key | action |
|---|---|
| `q` | type a new query in the terminal (e.g. `"laptop"`, `"the person's face"`, `"a coffee cup"`) |
| `space` | pause / resume the inference loop (still streams camera) |
| `m` | cycle model: D-Mobile → v2-Tiny → v1 |
| `esc` | quit |

### Useful flags

```bash
python demo/realtime_webcam_demo.py \
  --query "hand"               # initial query
  --model d-mobile             # d-mobile / v2-tiny / v1
  --cam 0                      # 0 is the FaceTime / built-in camera on most Macs
  --cam_w 640 --cam_h 480      # capture resolution; lower = faster
  --device mps                 # override autodetect; mps / cuda / cpu
  --ckpt /path/to/best.pt      # use a local checkpoint instead of downloading
  --download_from_release      # force-download the latest release ckpt
```

## What the heatmap means

Each frame is fed into the (frozen) backbone → 196 patch features (14×14 grid) → text-conditional head with the CLIP text emb of your query → sigmoid score per patch. **High score (red/yellow) = "this patch is relevant to the query"**, low score (dark) = "irrelevant".

Try queries like:
- concrete objects: `"hand"`, `"face"`, `"laptop"`, `"coffee cup"`
- relational: `"the person's eyes"`, `"a screen"`, `"food on a plate"`
- scene: `"kitchen"`, `"office desk"`, `"hardwood floor"`

The scorer is trained on COCO category masks + CLIPSeg-soft, so concrete COCO-like categories work best. Abstract / temporal queries like `"playing dominoes"` (which need motion context) will be flatter.

## Troubleshooting

- **"cannot open camera 0"**: try `--cam 1` (FaceTime is sometimes index 1 if you have an iPhone Continuity Camera connected).
- **MPS error / NaN heatmaps**: `--device cpu` will work everywhere but is slower.
- **First run slow**: `timm` and `open_clip` download backbone weights on first run (~50 MB). Subsequent runs start in ~3 s.
- **Permission prompt for camera**: macOS will ask the first time. Grant access to your terminal app (Terminal / iTerm / VS Code).

## What this is NOT

- Not a full video-VQA pipeline. This demo only shows the *scorer's* per-patch heatmap. The downstream NVILA SigLIP-ViT + LLaMA decoder is not in the loop. Use this to inspect what the trained scorer "sees", not to answer questions about your video.
- Not the AutoGaze model. AutoGaze is text-blind saliency; this demo runs the text-conditioned scorer that complements / replaces AutoGaze in the §1 pipeline.
- Not a benchmark. For benchmarks see `paper.md` and the per-move result docs.
