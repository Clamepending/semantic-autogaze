"""Grounded-SAM HTTP inference server.

Exposes POST /infer that takes (jpeg_bytes, query) and returns a binary mask
as a 1-channel PNG (uint8: 0 or 255). Designed to run on a GPU host while
the camera-side Pi captures frames and overlays masks.

Endpoint:
  POST /infer
    multipart/form-data:
      image: JPEG bytes (any resolution)
      query: text query string
    optional headers:
      X-Box-Threshold: 0.30 (default)
      X-Text-Threshold: 0.25 (default)
      X-Top1: '1' (default; if '0', union all detected boxes' SAM masks)
  Returns:
    PNG: binary mask 1-channel uint8 at the original frame resolution (0/255).
    Headers: X-N-Boxes, X-Top-Score, X-Inference-Ms

  GET /health  -> 'ok'
"""
from __future__ import annotations
import argparse
import io
import time

import numpy as np
import torch
from PIL import Image
from flask import Flask, Response, request


GDINO_NAME = "IDEA-Research/grounding-dino-base"
SAM_NAME = "facebook/sam-vit-base"   # huge=2.5GB, base=375MB+5x faster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--gdino", default=GDINO_NAME)
    ap.add_argument("--sam", default=SAM_NAME)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[device] {device}", flush=True)

    print(f"[gdino] loading {args.gdino} ...", flush=True)
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    gdino_proc = AutoProcessor.from_pretrained(args.gdino)
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino).to(device).eval()
    print(f"[gdino] ok, {sum(p.numel() for p in gdino.parameters()) / 1e6:.1f} M params", flush=True)

    print(f"[sam] loading {args.sam} ...", flush=True)
    from transformers import SamModel, SamProcessor
    sam_proc = SamProcessor.from_pretrained(args.sam)
    sam = SamModel.from_pretrained(args.sam).to(device).eval()
    print(f"[sam] ok, {sum(p.numel() for p in sam.parameters()) / 1e6:.1f} M params", flush=True)

    @torch.no_grad()
    def infer(pil: Image.Image, query: str, box_thr: float, text_thr: float, top1: bool):
        H, W = pil.height, pil.width
        text = query.lower().strip()
        if not text.endswith("."): text += "."
        inputs = gdino_proc(images=pil, text=text, return_tensors="pt").to(device)
        outs = gdino(**inputs)
        results = gdino_proc.post_process_grounded_object_detection(
            outs, inputs.input_ids, threshold=box_thr,
            text_threshold=text_thr, target_sizes=[(H, W)],
        )[0]
        boxes = results["boxes"]; scores = results.get("scores", torch.zeros(len(boxes)))
        full_mask = np.zeros((H, W), dtype=np.uint8)
        if boxes.numel() == 0:
            return full_mask, 0, 0.0
        if top1:
            idx = int(torch.argmax(scores).item())
            boxes = boxes[idx:idx + 1]
            top_score = float(scores[idx].item())
        else:
            top_score = float(scores.max().item())
        sam_inputs = sam_proc(pil, input_boxes=[boxes.cpu().numpy().tolist()],
                              return_tensors="pt").to(device)
        sam_outs = sam(**sam_inputs, multimask_output=False)
        masks = sam_proc.image_processor.post_process_masks(
            sam_outs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )[0]
        m_np = masks.cpu().numpy() if hasattr(masks, "cpu") else np.array(masks)
        if m_np.ndim == 4:
            for box_ms in m_np:
                for m in box_ms:
                    full_mask = np.maximum(full_mask, (m.astype(bool).astype(np.uint8) * 255))
        elif m_np.ndim == 3:
            for m in m_np:
                full_mask = np.maximum(full_mask, (m.astype(bool).astype(np.uint8) * 255))
        elif m_np.ndim == 2:
            full_mask = (m_np.astype(bool).astype(np.uint8) * 255)
        return full_mask, len(boxes), top_score

    app = Flask(__name__)

    @app.get("/health")
    def health():
        return "ok"

    @app.post("/infer")
    def infer_route():
        t0 = time.perf_counter()
        if "image" not in request.files:
            return ("missing 'image' file", 400)
        query = request.form.get("query", "").strip() or request.headers.get("X-Query", "").strip()
        if not query:
            return ("missing 'query'", 400)
        box_thr = float(request.headers.get("X-Box-Threshold", "0.30"))
        text_thr = float(request.headers.get("X-Text-Threshold", "0.25"))
        top1 = request.headers.get("X-Top1", "1") == "1"
        img_bytes = request.files["image"].read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        full_mask, n_boxes, top_score = infer(pil, query, box_thr, text_thr, top1)
        # PNG-encode the mask (1-channel uint8 0/255)
        out = io.BytesIO()
        Image.fromarray(full_mask, mode="L").save(out, format="PNG")
        out.seek(0)
        ms = (time.perf_counter() - t0) * 1000
        resp = Response(out.read(), mimetype="image/png")
        resp.headers["X-N-Boxes"] = str(n_boxes)
        resp.headers["X-Top-Score"] = f"{top_score:.4f}"
        resp.headers["X-Inference-Ms"] = f"{ms:.1f}"
        return resp

    print(f"\n[server] listening on http://{args.host}:{args.port}/", flush=True)
    print(f"[server] curl http://{args.host}:{args.port}/health", flush=True)
    app.run(host=args.host, port=args.port, threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()
