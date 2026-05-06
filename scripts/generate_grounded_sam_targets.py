"""Phase 2a: generate Grounded-SAM teacher signals on COCO val2017 + curated vocab.

For each (image, query) pair:
  1. CLIP image-text cosine similarity (presence gate)
  2. If sim > gate_threshold:
     - Run GroundingDINO + SAM
     - Save full-resolution binary mask (PNG)
     - Save 14×14 pooled mask (max-pooled, in npz)
     - Save presence flag = 1 if a real box was found, else 0
  3. Else:
     - Save zero mask + presence = 0 (skipping gdino+SAM saves compute)

Output layout:
  results/phase2_targets/<image_id>__<query_slug>.npz
    fields: clip_sim (float), gate_passed (bool), presence (bool),
            top_box_score (float), n_boxes (int), mask14 (14, 14) uint8,
            mask_full (H, W) uint8 (only if presence)

Parallelizable: pass --query_slice "start:end" to process a subset of the vocab.
Across 2 GPUs, split queries 50/50.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# Vocabulary: COCO 80 classes + curated deployment-relevant terms.
COCO_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# Deployment-relevant additions. Some overlap with COCO; that's OK.
DEPLOYMENT_EXTRA = [
    "hand", "face", "robot gripper", "robot arm", "pen", "pencil", "marker",
    "paper", "cardboard box", "screen", "monitor", "computer", "headphones",
    "package", "bag", "wallet", "keys", "watch", "glasses", "mask",
]

DEFAULT_VOCAB = COCO_80 + DEPLOYMENT_EXTRA

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def slugify(s: str) -> str:
    return s.lower().strip().replace(" ", "_").replace("/", "_")


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load CLIP for the gate
    print(f"[clip] loading ViT-B/16 ...", flush=True)
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()
    CLIP_MEAN_T = torch.tensor(CLIP_MEAN, device=device)
    CLIP_STD_T = torch.tensor(CLIP_STD, device=device)

    # Load Grounded-SAM
    print(f"[gdino] loading IDEA-Research/grounding-dino-base ...", flush=True)
    from transformers import AutoProcessor as _AP, AutoModelForZeroShotObjectDetection
    gdino_proc = _AP.from_pretrained("IDEA-Research/grounding-dino-base")
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base").to(device).eval()

    print(f"[sam] loading {args.sam} ...", flush=True)
    from transformers import SamModel, SamProcessor
    sam_proc = SamProcessor.from_pretrained(args.sam)
    sam = SamModel.from_pretrained(args.sam).to(device).eval()

    # Pre-encode all text embeddings once
    vocab = args.vocab
    if args.query_slice:
        a, b = args.query_slice.split(":")
        a = int(a) if a else 0
        b = int(b) if b else len(vocab)
        vocab = vocab[a:b]
        print(f"[slice] using vocab[{a}:{b}] = {len(vocab)} queries", flush=True)
    print(f"[vocab] {len(vocab)} queries: {vocab[:5]} ... {vocab[-3:]}", flush=True)

    with torch.no_grad():
        toks = clip_tok(vocab).to(device)
        text_embs = F.normalize(clip_model.encode_text(toks), dim=-1)  # (V, 512)
    print(f"[text] cached embs: {text_embs.shape}", flush=True)

    # Image list
    img_dir = Path(args.image_dir)
    img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() == ".jpg"])
    if args.n_images:
        rng = np.random.default_rng(args.seed)
        if len(img_files) > args.n_images:
            indices = rng.choice(len(img_files), args.n_images, replace=False)
            img_files = [img_files[i] for i in sorted(indices)]
    print(f"[images] processing {len(img_files)} images", flush=True)

    @torch.no_grad()
    def clip_image_emb(pil):
        arr = np.array(pil.resize((224, 224), Image.BICUBIC))
        x = torch.from_numpy(arr).permute(2, 0, 1).float().to(device) / 255.0
        x = (x.unsqueeze(0) - CLIP_MEAN_T[None, :, None, None]) / CLIP_STD_T[None, :, None, None]
        return F.normalize(clip_model.encode_image(x), dim=-1)  # (1, 512)

    @torch.no_grad()
    def grounded_sam_mask(pil, query, box_thr=0.30, text_thr=0.25, max_box_area=0.55):
        H, W = pil.height, pil.width
        text = query.lower().strip() + "."
        inputs = gdino_proc(images=pil, text=text, return_tensors="pt").to(device)
        outs = gdino(**inputs)
        results = gdino_proc.post_process_grounded_object_detection(
            outs, inputs.input_ids, threshold=box_thr, text_threshold=text_thr,
            target_sizes=[(H, W)],
        )[0]
        boxes = results["boxes"]; scores = results.get("scores", torch.zeros(len(boxes)))
        full_mask = np.zeros((H, W), dtype=np.uint8)
        if boxes.numel() == 0:
            return full_mask, 0, 0.0
        # Filter big false-positive boxes
        frame_area = float(H * W)
        keep = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].tolist()
            barea = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if barea / max(frame_area, 1.0) <= max_box_area:
                keep.append(i)
        if not keep: return full_mask, 0, 0.0
        keep_idx = torch.tensor(keep, device=boxes.device)
        boxes = boxes[keep_idx]; scores = scores[keep_idx]
        top_score = float(scores.max().item())
        # Run SAM with all kept boxes (union)
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
        return full_mask, len(keep), top_score

    skipped = 0
    n_done = 0
    n_present = 0
    n_gated = 0
    t_start = time.time()

    for img_idx, img_path in enumerate(img_files):
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            skipped += 1; continue
        img_id = img_path.stem
        H, W = pil.height, pil.width
        img_emb = clip_image_emb(pil)  # (1, 512)
        sims = (img_emb @ text_embs.T).squeeze(0).cpu().numpy()  # (V,)

        for q_idx, query in enumerate(vocab):
            sim = float(sims[q_idx])
            slug = slugify(query)
            out_path = os.path.join(args.output_dir, f"{img_id}__{slug}.npz")
            if os.path.exists(out_path) and not args.overwrite:
                continue

            gate_passed = sim >= args.gate
            presence = False; n_boxes = 0; top_score = 0.0
            mask14 = np.zeros((14, 14), dtype=np.uint8)
            mask_full = None
            if gate_passed:
                full_mask, n_boxes, top_score = grounded_sam_mask(
                    pil, query, box_thr=args.box_thr, text_thr=args.text_thr,
                    max_box_area=args.max_box_area,
                )
                presence = (n_boxes > 0) and (full_mask.sum() > 0)
                if presence:
                    # Pool to 14x14 via max-pool
                    m_t = torch.from_numpy(full_mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
                    m14 = F.adaptive_max_pool2d(m_t, (14, 14)).squeeze().numpy()
                    mask14 = (m14 > 0.5).astype(np.uint8) * 255
                    mask_full = full_mask
                    n_present += 1
                n_gated += 1

            np.savez_compressed(
                out_path,
                clip_sim=np.float32(sim),
                gate_passed=np.bool_(gate_passed),
                presence=np.bool_(presence),
                top_box_score=np.float32(top_score),
                n_boxes=np.int32(n_boxes),
                mask14=mask14,
                # Save full-res mask only when presence; saves space.
                mask_full=mask_full if mask_full is not None else np.zeros((1, 1), dtype=np.uint8),
                H=np.int32(H), W=np.int32(W),
                img_id=img_id, query=query,
            )
            n_done += 1

        if (img_idx + 1) % 25 == 0:
            elapsed = time.time() - t_start
            rate = n_done / max(elapsed, 1)
            remaining_pairs = len(img_files) * len(vocab) - n_done
            eta_s = remaining_pairs / max(rate, 1)
            print(f"  [img {img_idx+1}/{len(img_files)}] pairs={n_done} "
                  f"gated={n_gated} present={n_present} skipped={skipped} | "
                  f"{rate:.1f} pairs/s, ETA {eta_s/60:.1f} min",
                  flush=True)

    elapsed = time.time() - t_start
    print(f"\n[done] {n_done} pairs in {elapsed/60:.1f} min "
          f"(gated={n_gated}, presence={n_present}, skipped_imgs={skipped})", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sam", default="facebook/sam-vit-base",
                   help="huge=better mask quality but ~5x slower")
    p.add_argument("--image_dir", default="/home/ogata/semantic-autogaze/data/coco_val2017/val2017")
    p.add_argument("--output_dir", default="/home/ogata/semantic-autogaze/results/phase2_targets")
    p.add_argument("--vocab", nargs="+", default=DEFAULT_VOCAB,
                   help="space-separated; or use --vocab_file path")
    p.add_argument("--vocab_file", default=None,
                   help="optional: file with one query per line")
    p.add_argument("--query_slice", default=None,
                   help="parallelize: 'a:b' python-slice for vocab")
    p.add_argument("--n_images", type=int, default=2500,
                   help="random subset of COCO val2017; default 2500 of 5000")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gate", type=float, default=0.18,
                   help="CLIP cosine gate; below this -> save zero mask, skip gdino+SAM")
    p.add_argument("--box_thr", type=float, default=0.30)
    p.add_argument("--text_thr", type=float, default=0.25)
    p.add_argument("--max_box_area", type=float, default=0.55)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.vocab_file:
        with open(args.vocab_file) as f:
            args.vocab = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    main(args)
