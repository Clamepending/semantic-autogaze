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

    # CLIP for the open-set gate: image-text cosine tells us whether the
    # queried concept is plausibly in the frame at all (much better-calibrated
    # for absence than gdino's per-box score).
    print("[clip] loading CLIP-B/16 for gate ...", flush=True)
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()
    CLIP_MEAN_T = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
    CLIP_STD_T  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)

    @torch.no_grad()
    def clip_gate_score(pil: Image.Image, query: str) -> float:
        """Returns CLIP image-text cosine similarity in roughly [-0.1, 0.4].
        Empirically: ~0.10-0.15 when concept absent; 0.20-0.30 when present."""
        arr = np.array(pil.resize((224, 224), Image.BICUBIC))
        x = torch.from_numpy(arr).permute(2, 0, 1).float().to(device) / 255.0
        x = (x.unsqueeze(0) - CLIP_MEAN_T[None, :, None, None]) / CLIP_STD_T[None, :, None, None]
        img_emb = torch.nn.functional.normalize(clip_model.encode_image(x), dim=-1)
        toks = clip_tok([query]).to(device)
        txt_emb = torch.nn.functional.normalize(clip_model.encode_text(toks), dim=-1)
        return float((img_emb @ txt_emb.T).squeeze().item())

    @torch.no_grad()
    def infer(pil: Image.Image, query: str, box_thr: float, text_thr: float, top1: bool,
              max_box_area_frac: float = 0.55):
        """If GroundingDINO returns a box whose area covers more than
        max_box_area_frac of the frame, treat it as a 'no detection' (the
        common false-positive mode where DINO emits a frame-spanning box
        when the queried object isn't actually present)."""
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

        # Reject huge "whole-frame" boxes — usually false positives.
        frame_area = float(H * W)
        keep = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].tolist()
            barea = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if barea / max(frame_area, 1.0) <= max_box_area_frac:
                keep.append(i)
        if not keep:
            return full_mask, 0, 0.0
        keep_idx = torch.tensor(keep, device=boxes.device)
        boxes = boxes[keep_idx]; scores = scores[keep_idx]

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
        box_thr = float(request.headers.get("X-Box-Threshold", "0.40"))
        text_thr = float(request.headers.get("X-Text-Threshold", "0.30"))
        top1 = request.headers.get("X-Top1", "1") == "1"
        max_area = float(request.headers.get("X-Max-Box-Area-Frac", "0.55"))
        clip_gate = float(request.headers.get("X-Clip-Gate", "0.18"))
        img_bytes = request.files["image"].read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # Stage 1: CLIP gate — is the concept plausibly in the frame at all?
        sim = clip_gate_score(pil, query)
        if sim < clip_gate:
            full_mask = np.zeros((pil.height, pil.width), dtype=np.uint8)
            n_boxes = 0; top_score = 0.0
        else:
            # Stage 2: full Grounded-SAM segmentation.
            full_mask, n_boxes, top_score = infer(pil, query, box_thr, text_thr, top1, max_area)
        # PNG-encode the mask (1-channel uint8 0/255)
        out = io.BytesIO()
        Image.fromarray(full_mask, mode="L").save(out, format="PNG")
        out.seek(0)
        ms = (time.perf_counter() - t0) * 1000
        resp = Response(out.read(), mimetype="image/png")
        resp.headers["X-N-Boxes"] = str(n_boxes)
        resp.headers["X-Top-Score"] = f"{top_score:.4f}"
        resp.headers["X-Clip-Sim"] = f"{sim:.4f}"
        resp.headers["X-Clip-Gate"] = f"{clip_gate:.4f}"
        resp.headers["X-Gated-Out"] = "1" if sim < clip_gate else "0"
        resp.headers["X-Inference-Ms"] = f"{ms:.1f}"
        return resp

    print(f"\n[server] listening on http://{args.host}:{args.port}/", flush=True)
    print(f"[server] curl http://{args.host}:{args.port}/health", flush=True)
    app.run(host=args.host, port=args.port, threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()
