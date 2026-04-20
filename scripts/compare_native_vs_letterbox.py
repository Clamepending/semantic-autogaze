"""Side-by-side: native-aspect Recipe B vs letterbox Recipe B vs COCO GT.

Uses the same v11 audit (img_id, cat) pairs as compare_icon_student so
panels are directly comparable to the prior letterbox-cycle write-up.

Panels: [image | COCO GT | letterbox B pred | native B pred]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO

from semantic_autogaze.icon_student import IconStudent
from semantic_autogaze.letterbox import (
    LetterboxInfo,
    compute_info,
    heatmap_to_original,
)
from semantic_autogaze.train_coco_seg import build_category_mask, get_image_categories


FILENAME_RE = re.compile(r"^\d{3}_(\d+)_(.+)\.png$")


def parse_qual_dir(qual_dir: Path) -> list[tuple[int, str]]:
    pairs = []
    for p in sorted(qual_dir.iterdir()):
        m = FILENAME_RE.match(p.name)
        if m:
            pairs.append((int(m.group(1)), m.group(2)))
    return pairs


def load_student(ckpt_path: str, device: torch.device) -> IconStudent:
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = raw["config"]
    model = IconStudent(
        patch_dim=cfg["patch_dim"], query_dim=cfg["query_dim"],
        decoder_dim=cfg["decoder_dim"],
        in_grid=cfg["in_grid"], out_grid=cfg["out_grid"],
        n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(raw["state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def predict_letterbox(model, patches_lb, query, device, info: LetterboxInfo) -> np.ndarray:
    """Letterbox cache: bare tensor (256, 384). Output mapped via un-letterbox."""
    logits = model(patches_lb.unsqueeze(0).to(device), query.unsqueeze(0).to(device))
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    hw = heatmap_to_original(probs, info, mode="bilinear")
    return np.clip(hw, 0.0, 1.0)


@torch.inference_mode()
def predict_native(model, native_obj, query, device, h: int, w: int) -> np.ndarray:
    """Native cache: dict with patches + grid. Output is at (8*N_h, 8*N_w),
    then bilinear up to original (h, w). No letterbox math."""
    patches = native_obj["patches"].float()
    n_h, n_w = native_obj["grid"]
    logits = model(patches.unsqueeze(0).to(device), query.unsqueeze(0).to(device),
                   grid_hw=(n_h, n_w))
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    # Map post-encode-frame heatmap (N_h*8, N_w*8) → original (h, w).
    # The encode frame covers a centered crop of the resized image; its
    # corresponding pixel-rect on the original is encode_hw ∶ resize_hw,
    # offset by crop_top_left scaled by resize→original.
    enc_h, enc_w = native_obj["encode_hw"]
    new_h, new_w = native_obj["resize_hw"]
    top_pad, left_pad = native_obj["crop_top_left"]
    sy = h / float(new_h); sx = w / float(new_w)
    og_top = max(0, int(round(top_pad * sy)))
    og_left = max(0, int(round(left_pad * sx)))
    og_bot = min(h, int(round((top_pad + enc_h) * sy)))
    og_right = min(w, int(round((left_pad + enc_w) * sx)))
    # Resize the heatmap to the encode-frame's pixel size on the original,
    # then place it at (og_top:og_bot, og_left:og_right). Pixels outside
    # this rect were center-cropped away during DINOv2 encoding and have
    # no model output — fill with zeros so the panel reflects reality.
    region_h = og_bot - og_top
    region_w = og_right - og_left
    if region_h <= 0 or region_w <= 0:
        return np.zeros((h, w), dtype=np.float32)
    t = torch.from_numpy(probs).float().unsqueeze(0).unsqueeze(0)
    region = F.interpolate(t, size=(region_h, region_w), mode="bilinear",
                            align_corners=False)[0, 0].numpy()
    out = np.zeros((h, w), dtype=np.float32)
    out[og_top:og_bot, og_left:og_right] = np.clip(region, 0.0, 1.0)
    return out


def overlay(img_np, gray, cmap):
    rgb = (cmap(gray)[..., :3] * 255).astype(np.uint8)
    return (0.5 * img_np.astype(np.float32) + 0.5 * rgb).clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qual-dir", required=True)
    p.add_argument("--ckpt-letterbox", required=True, help="Recipe B trained on letterbox cache")
    p.add_argument("--ckpt-native", required=True, help="Recipe B trained on native cache")
    p.add_argument("--dinov2-cache-letterbox", required=True)
    p.add_argument("--dinov2-cache-native", required=True)
    p.add_argument("--clip-text-embeddings", required=True)
    p.add_argument("--data-dir", default="data/coco")
    p.add_argument("--ann", default="annotations/instances_val2017.json")
    p.add_argument("--img-subdir", default="val2017")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pairs = parse_qual_dir(Path(args.qual_dir))
    print(f"[diag] {len(pairs)} pairs from {args.qual_dir}")

    coco = COCO(os.path.join(args.data_dir, args.ann))
    img_dir = os.path.join(args.data_dir, args.img_subdir)
    name_to_id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}

    clip_text = torch.load(args.clip_text_embeddings, map_location="cpu", weights_only=True)
    print(f"[diag] loading letterbox B: {args.ckpt_letterbox}")
    model_lb = load_student(args.ckpt_letterbox, device)
    print(f"[diag] loading native B: {args.ckpt_native}")
    model_nv = load_student(args.ckpt_native, device)

    cmap = matplotlib.colormaps["jet"]
    summary = []
    for idx, (img_id, cat_slug) in enumerate(pairs):
        cat_name = cat_slug.replace("_", " ")
        if cat_name not in name_to_id:
            print(f"  [skip] {idx:03d}: unknown cat {cat_name!r}"); continue
        cat_id = name_to_id[cat_name]

        lb_path = Path(args.dinov2_cache_letterbox) / f"{img_id}.pt"
        nv_path = Path(args.dinov2_cache_native) / f"{img_id}.pt"
        if not lb_path.exists() or not nv_path.exists():
            print(f"  [skip] {idx:03d}: missing cache for {img_id}"); continue

        img_info = coco.imgs[img_id]
        img = Image.open(os.path.join(img_dir, img_info["file_name"])).convert("RGB")
        img_np = np.array(img); h, w = img_np.shape[:2]
        info_lb = compute_info(h, w)

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=0))
        if not anns:
            print(f"  [skip] {idx:03d}: no GT for {cat_name} in {img_id}"); continue
        gt = build_category_mask(coco, img_id, cat_id, anns).astype(np.float32)
        gt = np.clip(gt, 0, 1)

        query = clip_text[str(cat_id)].float()

        patches_lb = torch.load(lb_path, map_location="cpu", weights_only=True).float()
        native_obj = torch.load(nv_path, map_location="cpu", weights_only=True)

        p_lb = predict_letterbox(model_lb, patches_lb, query, device, info_lb)
        p_nv = predict_native(model_nv, native_obj, query, device, h, w)

        other_cats = [coco.cats[c]["name"] for c in get_image_categories(coco, img_id).keys() if c != cat_id]

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        axes[0].imshow(img_np); axes[0].set_title(f"image {img_id}"); axes[0].axis("off")
        axes[1].imshow(overlay(img_np, gt, cmap)); axes[1].set_title(f"COCO GT \"{cat_name}\""); axes[1].axis("off")
        axes[2].imshow(overlay(img_np, p_lb, cmap)); axes[2].set_title(f"letterbox B (max={p_lb.max():.2f})"); axes[2].axis("off")
        axes[3].imshow(overlay(img_np, p_nv, cmap)); axes[3].set_title(f"native B (max={p_nv.max():.2f})"); axes[3].axis("off")
        others = ", ".join(other_cats[:8]) + ("…" if len(other_cats) > 8 else "")
        plt.suptitle(f"query: \"{cat_name}\" | also present: {others}", fontsize=12)
        plt.tight_layout()
        out = out_dir / f"{idx:03d}_{img_id}_{cat_slug}.png"
        plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
        summary.append({"idx": idx, "img_id": img_id, "cat": cat_name, "out": str(out)})
        print(f"  {idx:03d}: {cat_name:14s} on {img_id} → {out.name}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[diag] wrote {len(summary)} panels to {out_dir}")


if __name__ == "__main__":
    main()
