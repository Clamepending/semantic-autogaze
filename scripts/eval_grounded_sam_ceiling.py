"""Phase 1: Grounded-SAM ceiling teacher validation.

For each of the 8 COCO qual-grid (category, query) pairs:
  1. Load the canonical qual-grid image.
  2. Compute the COCO GT mask (largest instance of the category).
  3. Run our existing Ours v1 head → soft mask → IoU vs GT.
  4. Run CLIPSeg → soft mask → IoU vs GT.
  5. Run Grounded-SAM (GroundingDino text→bbox + SAM bbox→mask) → IoU vs GT.

Render a side-by-side figure with all four columns + per-cell IoU,
and dump per-method mIoU to JSON.

If Grounded-SAM clearly beats CLIPSeg on the qual grid → Phase 2:
distill from Grounded-SAM at scale (COCO train2017).
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")
from train_independent_scorer import TextScorerHead, GRID
from pycocotools.coco import COCO

# Same 8 (category, query) pairs as the existing qual grid for direct comparison
QUAL_PAIRS = [
    ("bird", 337987, "bird"),
    ("person", 32861, "people"),
    ("bicycle", 370208, "bicycle"),
    ("tv", 346638, "screen"),
    ("cat", 223747, "cat"),
    ("dog", 267300, "dog"),
    ("car", 151962, "car"),
    ("pizza", 232489, "pizza"),
]

COCO_ROOT = "/home/ogata/semantic-autogaze/data/coco_val2017"
OURS_CKPT = "/home/ogata/semantic-autogaze/results/independent_scorer/best_v1.pt"
CLIPSEG_NAME = "CIDAS/clipseg-rd64-refined"

# Grounded-SAM
GDINO_NAME = "IDEA-Research/grounding-dino-base"
SAM_NAME = "facebook/sam-vit-huge"  # gold-standard mask quality

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_qual_image_and_gt(coco, cat_name, img_id):
    cat_id = coco.getCatIds(catNms=[cat_name])[0]
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=False)
    anns = sorted(coco.loadAnns(ann_ids), key=lambda a: -a.get("area", 0))
    ann = anns[0]
    gt_mask = coco.annToMask(ann).astype(np.uint8)  # (H, W) {0,1}
    info = coco.loadImgs([img_id])[0]
    img_path = os.path.join(COCO_ROOT, "val2017", info["file_name"])
    pil = Image.open(img_path).convert("RGB")
    return pil, gt_mask, img_path


def iou_topk(soft_14, gt_full_mask, K=None):
    """soft_14: (14, 14) sigmoid scores. gt_full_mask: (H, W) {0,1}.
    Pool gt to 14x14 (max-pool), pick top-K patches from soft, compute IoU."""
    gt_t = torch.from_numpy(gt_full_mask).float().unsqueeze(0).unsqueeze(0)
    gt14 = F.adaptive_max_pool2d(gt_t, (GRID, GRID)).squeeze().numpy() > 0.5
    K = K or int(gt14.sum())
    if K <= 0: return 0.0
    flat = soft_14.flatten()
    topk = np.argpartition(-flat, K - 1)[:K]
    m = np.zeros(GRID * GRID, bool); m[topk] = True
    m14 = m.reshape(GRID, GRID)
    inter = np.logical_and(m14, gt14).sum()
    union = np.logical_or(m14, gt14).sum()
    return float(inter / max(1, union))


def iou_full_resolution(pred_mask_HW, gt_mask_HW):
    """Full-resolution IoU between two binary masks of the same shape."""
    pred = pred_mask_HW.astype(bool); gt = gt_mask_HW.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / max(1, union))


def heatmap_ours_v1(clip_model, head, pil, query, device):
    """Returns (14, 14) sigmoid scores on the image."""
    arr = np.array(pil)
    t = torch.from_numpy(arr).permute(2, 0, 1).float().to(device) / 255.0
    t = F.interpolate(t.unsqueeze(0), size=(224, 224), mode="bicubic", align_corners=False)
    mean = torch.tensor(CLIP_MEAN, device=device)[None, :, None, None]
    std = torch.tensor(CLIP_STD, device=device)[None, :, None, None]
    t = (t - mean) / std
    import open_clip
    with torch.no_grad():
        clip_model.visual.output_tokens = True
        _, patches = clip_model.visual(t)
        clip_model.visual.output_tokens = False
        toks = open_clip.get_tokenizer("ViT-B-16")([query]).to(device)
        text = F.normalize(clip_model.encode_text(toks), dim=-1).expand(1, -1)
        scores = head(patches, text)
        h = torch.sigmoid(scores).reshape(GRID, GRID).cpu().numpy()
    return h


def heatmap_clipseg(clipseg_model, clipseg_proc, pil, query, device):
    inp = clipseg_proc(text=[query], images=[pil], return_tensors="pt").to(device)
    with torch.no_grad():
        preds = clipseg_model(**inp).logits  # (1, 352, 352)
    preds = torch.sigmoid(preds.unsqueeze(1))
    h = F.interpolate(preds, size=(GRID, GRID), mode="bilinear", align_corners=False)
    return h.squeeze().cpu().numpy()


def mask_grounded_sam(gdino_model, gdino_proc, sam_model, sam_proc,
                     pil, query, device, box_threshold=0.30, text_threshold=0.25,
                     top1_only=True):
    """Run GroundingDINO → top-scored box → SAM → mask. Returns
    (full_res_binary_mask, boxes, top_score)."""
    H, W = pil.height, pil.width
    full_mask = np.zeros((H, W), dtype=bool)

    text_for_dino = query.lower().strip()
    if not text_for_dino.endswith("."): text_for_dino += "."
    inputs = gdino_proc(images=pil, text=text_for_dino, return_tensors="pt").to(device)
    with torch.no_grad():
        outs = gdino_model(**inputs)
    results = gdino_proc.post_process_grounded_object_detection(
        outs, inputs.input_ids, threshold=box_threshold,
        text_threshold=text_threshold, target_sizes=[(H, W)],
    )[0]
    boxes = results["boxes"]
    scores = results.get("scores", torch.zeros(len(boxes)))
    if boxes.numel() == 0:
        return full_mask, [], 0.0

    # Pick top-1 by score (avoids over-detection union problem)
    if top1_only:
        top_idx = int(torch.argmax(scores).item())
        boxes = boxes[top_idx:top_idx + 1]
        used_score = float(scores[top_idx].item())
    else:
        used_score = float(scores.max().item())

    sam_inputs = sam_proc(pil, input_boxes=[boxes.cpu().numpy().tolist()],
                          return_tensors="pt").to(device)
    with torch.no_grad():
        sam_outs = sam_model(**sam_inputs, multimask_output=False)
    masks = sam_proc.image_processor.post_process_masks(
        sam_outs.pred_masks.cpu(),
        sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu(),
    )[0]
    masks_np = masks.cpu().numpy() if hasattr(masks, "cpu") else np.array(masks)
    if masks_np.ndim == 4:
        for box_masks in masks_np:
            for m in box_masks:
                full_mask |= m.astype(bool)
    elif masks_np.ndim == 3:
        for m in masks_np:
            full_mask |= m.astype(bool)
    elif masks_np.ndim == 2:
        full_mask |= masks_np.astype(bool)
    return full_mask, boxes.cpu().numpy().tolist(), used_score


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("loading CLIP ViT-B/16 ...", flush=True)
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_model = clip_model.to(device).eval()

    print(f"loading Ours v1 head from {OURS_CKPT}...", flush=True)
    ck = torch.load(OURS_CKPT, map_location=device)
    ca = ck.get("args", {}) or {}
    head = TextScorerHead(
        patch_dim=768, text_dim=512,
        hidden_dim=ca.get("head_hidden_dim", 384),
        n_attn_heads=ca.get("head_attn_heads", 6),
        n_attn_layers=ca.get("head_attn_layers", 2),
        grid_size=GRID, use_spatial=ca.get("head_use_spatial", True),
    ).to(device).eval()
    head.load_state_dict(ck["head"])

    print(f"loading CLIPSeg {CLIPSEG_NAME}...", flush=True)
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    clipseg_proc = CLIPSegProcessor.from_pretrained(CLIPSEG_NAME)
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_NAME).to(device).eval()

    print(f"loading GroundingDINO {GDINO_NAME}...", flush=True)
    from transformers import AutoProcessor as _AP, AutoModelForZeroShotObjectDetection
    gdino_proc = _AP.from_pretrained(GDINO_NAME)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_NAME).to(device).eval()

    print(f"loading SAM {SAM_NAME}...", flush=True)
    from transformers import SamModel, SamProcessor
    sam_proc = SamProcessor.from_pretrained(SAM_NAME)
    sam_model = SamModel.from_pretrained(SAM_NAME).to(device).eval()

    print(f"loading COCO val2017 annotations...", flush=True)
    coco = COCO(os.path.join(COCO_ROOT, "annotations", "instances_val2017.json"))

    results = {}
    rows = []
    for cat, img_id, query in QUAL_PAIRS:
        print(f"\n=== {cat} (img {img_id}, query '{query}') ===", flush=True)
        pil, gt, img_path = load_qual_image_and_gt(coco, cat, img_id)
        H, W = pil.height, pil.width

        h_v1 = heatmap_ours_v1(clip_model, head, pil, query, device)
        h_cs = heatmap_clipseg(clipseg_model, clipseg_proc, pil, query, device)
        gsam_mask, gsam_boxes, gsam_score = mask_grounded_sam(
            gdino_model, gdino_proc, sam_model, sam_proc, pil, query, device)

        iou_v1 = iou_topk(h_v1, gt)
        iou_cs = iou_topk(h_cs, gt)
        # Grounded-SAM is a HARD mask, so compute IoU at full-res, plus 14x14 top-K for fair compare
        iou_gsam_full = iou_full_resolution(gsam_mask, gt)
        # 14x14 version
        gsam14 = F.adaptive_max_pool2d(
            torch.from_numpy(gsam_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0),
            (GRID, GRID)).squeeze().numpy()
        iou_gsam14 = iou_topk(gsam14, gt) if gsam14.sum() > 0 else 0.0

        print(f"  IoU vs GT: ours_v1={iou_v1:.3f}  clipseg={iou_cs:.3f}  "
              f"gsam_full={iou_gsam_full:.3f}  gsam14={iou_gsam14:.3f}  "
              f"gsam_n_boxes={len(gsam_boxes)} gsam_max_score={gsam_score:.2f}",
              flush=True)
        rows.append((cat, query, img_path, gt, h_v1, h_cs, gsam_mask,
                     iou_v1, iou_cs, iou_gsam_full, iou_gsam14))
        results[cat] = dict(query=query, img_id=img_id,
                            iou_ours_v1=iou_v1, iou_clipseg=iou_cs,
                            iou_gsam_full=iou_gsam_full, iou_gsam14=iou_gsam14,
                            gsam_n_boxes=len(gsam_boxes))

    # ---- Render figure ----
    methods = ["input", "GT mask", "CLIPSeg", "Ours v1", "Grounded-SAM"]
    n_rows = len(rows); n_cols = len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols + 0.5, 1.7 * n_rows + 0.5))
    if n_rows == 1: axes = axes[None, :]

    for i, (cat, query, img_path, gt, h_v1, h_cs, gsam_mask,
            iou_v1, iou_cs, iou_gsam_full, iou_gsam14) in enumerate(rows):
        pil = Image.open(img_path).convert("RGB")
        arr = np.array(pil)

        ax = axes[i, 0]; ax.imshow(arr); ax.set_xticks([]); ax.set_yticks([])
        if i == 0: ax.set_title(methods[0], fontsize=10)
        ax.set_ylabel(f"{cat}\n'{query}'", fontsize=9, rotation=0, ha="right", va="center", labelpad=24)

        ax = axes[i, 1]; ax.imshow(arr); ax.imshow(gt, alpha=0.5, cmap="Greens")
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0: ax.set_title(methods[1], fontsize=10)

        for col_i, (h14, iou, label) in enumerate([
            (h_cs, iou_cs, "CLIPSeg"),
            (h_v1, iou_v1, "Ours v1"),
        ]):
            ax = axes[i, 2 + col_i]
            ax.imshow(arr, alpha=0.6)
            h_up = np.kron(h14, np.ones((arr.shape[0] // GRID + 1, arr.shape[1] // GRID + 1)))[
                :arr.shape[0], :arr.shape[1]]
            ax.imshow(h_up, alpha=0.5, cmap="hot", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.02, 0.95, f"IoU={iou:.2f}", color="white", fontsize=9,
                    transform=ax.transAxes, va="top",
                    bbox=dict(facecolor="black", alpha=0.6, pad=2, edgecolor="none"))
            if i == 0: ax.set_title(label, fontsize=10)

        ax = axes[i, 4]
        ax.imshow(arr, alpha=0.6)
        ax.imshow(gsam_mask.astype(np.float32), alpha=0.55, cmap="hot", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.02, 0.95, f"IoU={iou_gsam_full:.2f}\n(14x14: {iou_gsam14:.2f})",
                color="white", fontsize=9, transform=ax.transAxes, va="top",
                bbox=dict(facecolor="black", alpha=0.6, pad=2, edgecolor="none"))
        if i == 0: ax.set_title(methods[4], fontsize=10)

    miou_v1 = float(np.mean([r["iou_ours_v1"] for r in results.values()]))
    miou_cs = float(np.mean([r["iou_clipseg"] for r in results.values()]))
    miou_gsam_full = float(np.mean([r["iou_gsam_full"] for r in results.values()]))
    miou_gsam14 = float(np.mean([r["iou_gsam14"] for r in results.values()]))

    plt.suptitle(f"Phase 1 ceiling-teacher comparison on COCO qual grid (n=8)\n"
                 f"mIoU: CLIPSeg={miou_cs:.3f} | Ours v1={miou_v1:.3f} | "
                 f"Grounded-SAM full-res={miou_gsam_full:.3f} | Grounded-SAM 14x14={miou_gsam14:.3f}",
                 fontsize=11, y=1.005)
    plt.tight_layout()
    out_png = os.path.join(args.output_dir, "ceiling_teacher_grid.png")
    plt.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"\nsaved {out_png}")

    if args.library_figures_dir:
        os.makedirs(args.library_figures_dir, exist_ok=True)
        import shutil
        lib_png = os.path.join(args.library_figures_dir, "phase1-grounded-sam-ceiling.png")
        shutil.copy(out_png, lib_png)
        print(f"  -> {lib_png}")

    out = {"per_image": results, "miou": {
        "ours_v1": miou_v1, "clipseg": miou_cs,
        "grounded_sam_full": miou_gsam_full, "grounded_sam_14x14": miou_gsam14,
    }}
    with open(os.path.join(args.output_dir, "phase1_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== mIoU summary on 8 COCO qual-grid images ===")
    print(f"  CLIPSeg:                     {miou_cs:.3f}")
    print(f"  Ours v1:                     {miou_v1:.3f}")
    print(f"  Grounded-SAM (full-res mask): {miou_gsam_full:.3f}")
    print(f"  Grounded-SAM (14x14 pooled): {miou_gsam14:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/phase1_ceiling_teacher")
    p.add_argument("--library_figures_dir",
                   default="/home/ogata/mac-brain/projects/semantic-autogaze/figures")
    args = p.parse_args()
    main(args)
