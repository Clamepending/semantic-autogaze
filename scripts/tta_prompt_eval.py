"""COCO val mIoU eval with optional test-time augmentation (TTA) and
prompt ensembling. Reuses the same N-image sampling + iou_topk metric as
`eval_coco_val_miou.py`, so output JSON is byte-comparable for diffing.

Two flags:
  --tta                 H-flip TTA. Run inference on original AND h-flipped
                        image, average the two heatmaps (un-flipping the
                        flipped one before averaging). 2x inference cost
                        per image.
  --prompt_ensemble     Average text embeddings across 7 OpenCLIP zero-shot
                        templates (e.g. "a photo of a {cat}", "the {cat}",
                        ...) before running inference. 7x text-encode cost
                        per query (1x per keyword change at deploy, since
                        text embs cache); 1x patch-inference cost per image.

Either / both / neither flag is valid.

Usage:
  python -m scripts.tta_prompt_eval --device cuda:0 \
      --ckpt results/phase23b_atto_perquery_linear_10k/best_val.pt \
      --n_images 50 --seed 2026 --tta --prompt_ensemble \
      --output_path results/eval_phase23/phase23b_perquery_linear_best/coco50_tta_pe.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")
from pycocotools.coco import COCO
from scripts.eval_phase2_ckpt import (
    load_ckpt, iou_topk, GRID,
)

COCO_ROOT = "/home/ogata/semantic-autogaze/data/coco_val2017"

# OpenCLIP-style zero-shot ensemble templates (subset of the standard 80)
PROMPT_TEMPLATES = [
    "a photo of a {}",
    "a photo of the {}",
    "a photo of a small {}",
    "a photo of a large {}",
    "a {}",
    "the {}",
    "a clear photo of a {}",
]


@torch.no_grad()
def encode_text(query, prompt_ensemble, clip_model_text, clip_tok, device):
    """Returns (1, 512) text embedding (averaged across templates if PE)."""
    if prompt_ensemble:
        texts = [tmpl.format(query) for tmpl in PROMPT_TEMPLATES]
    else:
        texts = [query]
    toks = clip_tok(texts).to(device)
    embs = F.normalize(clip_model_text.encode_text(toks), dim=-1)  # (P, 512)
    if prompt_ensemble:
        # Average then re-normalize (standard PE recipe).
        embs = F.normalize(embs.mean(0, keepdim=True), dim=-1)  # (1, 512)
    return embs  # (1, 512)


@torch.no_grad()
def heatmap_one(pil, text_emb, bb_fn, head, sb, mean, std, device):
    """Like eval_phase2_ckpt.heatmap_one but takes a precomputed text_emb."""
    arr = np.array(pil.resize((224, 224), Image.BICUBIC))
    x = (arr.astype(np.float32) / 255.0 -
         np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(device)
    patches = bb_fn(x)  # (1, 196, D)
    logits = head(patches, text_emb).reshape(GRID, GRID)
    if getattr(sb, "_is_per_query", False):
        logits_w = logits.unsqueeze(0).unsqueeze(0)  # (1, 1, GRID, GRID)
        cal = sb(logits_w, text_emb).squeeze(0).squeeze(0)
    else:
        cal = sb(logits)
    return torch.sigmoid(cal).cpu().numpy()


def heatmap_tta(pil, text_emb, bb_fn, head, sb, mean, std, device):
    """H-flip TTA: average heatmap from original + h-flipped (un-flipped)."""
    h_orig = heatmap_one(pil, text_emb, bb_fn, head, sb, mean, std, device)
    pil_f = pil.transpose(Image.FLIP_LEFT_RIGHT)
    h_flip = heatmap_one(pil_f, text_emb, bb_fn, head, sb, mean, std, device)
    h_flip_corrected = h_flip[:, ::-1].copy()  # un-flip cols
    return (h_orig + h_flip_corrected) * 0.5


def main(args):
    device = torch.device(args.device)
    bb_fn, head, sb, mean, std, model, kind, bb_module, _obj_head = load_ckpt(args.ckpt, device)
    print(f"[ckpt] loaded {args.ckpt}", flush=True)
    print(f"[mode] tta={args.tta}  prompt_ensemble={args.prompt_ensemble} "
          f"({len(PROMPT_TEMPLATES) if args.prompt_ensemble else 1} prompts/query)", flush=True)

    import open_clip
    clip_model_text, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai")
    clip_model_text = clip_model_text.to(device).eval()
    clip_tok = open_clip.get_tokenizer("ViT-B-16")

    coco = COCO(os.path.join(COCO_ROOT, "annotations", "instances_val2017.json"))
    cat_ids = sorted(coco.getCatIds())
    rng = np.random.default_rng(args.seed)

    # Same sampling as eval_coco_val_miou (one-per-image, balanced cats)
    samples = []
    img_ids_used = set()
    n_per_cat = max(1, args.n_images // len(cat_ids) + 1)
    for cat_id in cat_ids:
        cat_name = coco.loadCats([cat_id])[0]["name"]
        ann_ids = coco.getAnnIds(catIds=[cat_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        per_img = {}
        for a in anns:
            if a["image_id"] in img_ids_used: continue
            if a["image_id"] not in per_img or a.get("area", 0) > per_img[a["image_id"]].get("area", 0):
                per_img[a["image_id"]] = a
        candidates = sorted(per_img.values(), key=lambda a: -a.get("area", 0))
        if not candidates: continue
        chosen = candidates[:n_per_cat]
        rng.shuffle(chosen)
        for ann in chosen[:n_per_cat]:
            samples.append((cat_name, ann["image_id"], ann))
            img_ids_used.add(ann["image_id"])
        if len(samples) >= args.n_images: break

    samples = samples[:args.n_images]
    print(f"[sampled] {len(samples)} (image, category) pairs", flush=True)

    ious = []
    per_cat_ious = {}
    for i, (cat_name, img_id, ann) in enumerate(samples):
        gt = coco.annToMask(ann).astype(np.float32)
        info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(COCO_ROOT, "val2017", info["file_name"])
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [skip {i}] {img_path}: {e}")
            continue
        text_emb = encode_text(cat_name, args.prompt_ensemble,
                               clip_model_text, clip_tok, device)
        if args.tta:
            h = heatmap_tta(pil, text_emb, bb_fn, head, sb, mean, std, device)
        else:
            h = heatmap_one(pil, text_emb, bb_fn, head, sb, mean, std, device)
        iou = iou_topk(h, gt)
        ious.append(iou)
        per_cat_ious.setdefault(cat_name, []).append(iou)
        if i % 10 == 0 or i == len(samples) - 1:
            print(f"  {i+1:3d}/{len(samples)} {cat_name:18s} img={img_id:6d} "
                  f"IoU={iou:.3f}  running_mean={np.mean(ious):.3f}", flush=True)

    miou = float(np.mean(ious))
    print(f"\n[final] N={len(ious)} mIoU={miou:.4f}  (std={np.std(ious):.3f}) "
          f"tta={args.tta} pe={args.prompt_ensemble}")
    cat_means = {c: float(np.mean(vs)) for c, vs in per_cat_ious.items()}

    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump({"ckpt": args.ckpt, "n": len(ious), "miou": miou,
                       "std": float(np.std(ious)),
                       "tta": args.tta, "prompt_ensemble": args.prompt_ensemble,
                       "n_prompts": len(PROMPT_TEMPLATES) if args.prompt_ensemble else 1,
                       "per_cat": cat_means,
                       "samples": [{"cat": s[0], "img_id": s[1]} for s in samples],
                       "ious": ious},
                       f, indent=2)
        print(f"[saved] {args.output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_images", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output_path", default=None)
    p.add_argument("--tta", action="store_true",
                   help="H-flip TTA: average heatmap from original + h-flipped image.")
    p.add_argument("--prompt_ensemble", action="store_true",
                   help="Average CLIP text embeddings across 7 zero-shot templates.")
    args = p.parse_args()
    main(args)
