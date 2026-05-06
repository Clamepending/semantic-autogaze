"""Lightweight inference for Raspberry Pi using ONNX Runtime INT8.

Dependencies (no PyTorch needed):
    pip install onnxruntime numpy opencv-python

Usage:
    python infer_rpi.py --category person
    python infer_rpi.py --category "tennis racket" --threshold 0.4
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_DIR = Path(__file__).parent / "models" / "onnx_int8"
DINO_SIZE = 448
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_sessions(model_dir: Path):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 4

    dino_sess = ort.InferenceSession(
        str(model_dir / "dinov2_small_patches_int8.onnx"),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    adapter_sess = ort.InferenceSession(
        str(model_dir / "text_adapter.onnx"),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    return dino_sess, adapter_sess


def preprocess_frame(frame: np.ndarray, size: int = DINO_SIZE) -> np.ndarray:
    img = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)[None]  # (1, 3, H, W)
    return img


def get_query(adapter_sess, text_features: np.ndarray, meta: dict) -> tuple[np.ndarray, float]:
    query = adapter_sess.run(None, {"text_features": text_features})[0]
    logit_scale = float(meta["logit_scale"][0])
    return query, logit_scale


def compute_heatmap(
    dino_sess, pixel_values: np.ndarray, query: np.ndarray, logit_scale: float, threshold: float
) -> np.ndarray:
    patch_features = dino_sess.run(None, {"pixel_values": pixel_values})[0]  # (1, N, 384)
    logits = np.einsum("bnd,bd->bn", patch_features, query) * logit_scale  # (1, N)
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    grid_side = int(np.sqrt(probs.shape[1]))
    heatmap = probs.reshape(grid_side, grid_side)
    return heatmap


def overlay_heatmap(
    frame: np.ndarray, heatmap: np.ndarray, threshold: float = 0.3, opacity: float = 0.6
) -> np.ndarray:
    h, w = frame.shape[:2]
    heat_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = (heat_resized > threshold).astype(np.float32)
    heat_color = cv2.applyColorMap((heat_resized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    overlay = frame.copy().astype(np.float32)
    mask_3ch = mask[:, :, None] * opacity
    overlay = overlay * (1 - mask_3ch) + heat_color.astype(np.float32) * mask_3ch
    return overlay.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="RPi INT8 semantic heatmap inference")
    parser.add_argument("--category", default="person", help="Object category to detect")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--opacity", type=float, default=0.6)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--display-size", type=int, default=640)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    print(f"Loading models from {model_dir}...")
    dino_sess, adapter_sess = load_sessions(model_dir)

    embeddings = np.load(model_dir / "clip_text_embeddings.npz")
    categories = list(embeddings["categories"])
    features = embeddings["features"]
    meta = np.load(model_dir / "adapter_meta.npz")

    if args.category in categories:
        idx = categories.index(args.category)
        text_features = features[idx:idx+1].astype(np.float32)
        print(f"Using pre-computed embedding for '{args.category}'")
    else:
        print(f"WARNING: '{args.category}' not in pre-computed categories. Using closest match.")
        print(f"Available: {', '.join(categories)}")
        return

    query, logit_scale = get_query(adapter_sess, text_features, meta)
    print(f"Query ready. logit_scale={logit_scale:.1f}")
    print(f"Starting camera (threshold={args.threshold})... Press 'q' to quit, '+'/'-' to adjust threshold.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    fps_history = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        pixel_values = preprocess_frame(frame)
        heatmap = compute_heatmap(dino_sess, pixel_values, query, logit_scale, args.threshold)
        dt = time.perf_counter() - t0
        fps_history.append(1.0 / dt)
        if len(fps_history) > 30:
            fps_history.pop(0)

        vis = overlay_heatmap(frame, heatmap, args.threshold, args.opacity)
        vis = cv2.resize(vis, (args.display_size, args.display_size))

        fps_avg = np.mean(fps_history)
        cv2.putText(vis, f"{args.category} | {fps_avg:.1f} FPS | thr={args.threshold:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Semantic Heatmap (INT8)", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("+") or key == ord("="):
            args.threshold = min(1.0, args.threshold + 0.05)
        elif key == ord("-"):
            args.threshold = max(0.0, args.threshold - 0.05)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
