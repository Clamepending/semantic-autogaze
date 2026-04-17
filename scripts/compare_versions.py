"""Side-by-side comparison of v1/v2/v3/v4 predictions on the same image-query pairs.

Usage:
    python scripts/compare_versions.py --data_dir data/coco --out results/comparison \\
        --num 12 --device mps
"""

from __future__ import annotations

import argparse
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO

from semantic_autogaze.bighead import BigHead, BigHeadDecoder
from semantic_autogaze.train_coco_seg import (
    PATCH_GRID,
    build_category_mask,
    cache_clip_text_embeddings,
    download_coco_val,
    get_image_categories,
    mask_to_patch_target,
)


def load_head(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    kind = cfg.get("model_kind", "BigHead")
    if kind == "BigHeadDecoder":
        head = BigHeadDecoder(
            hidden_dim=cfg["hidden_dim"],
            embedding_dim=cfg["embedding_dim"],
            expanded_dim=cfg["expanded_dim"],
            n_attn_heads=cfg["n_attn_heads"],
            n_attn_layers=cfg["n_attn_layers"],
            decoder_dim=cfg.get("decoder_dim", 128),
            out_grid=cfg.get("out_grid", 28),
        ).to(device)
    else:
        head = BigHead(
            hidden_dim=cfg["hidden_dim"],
            embedding_dim=cfg["embedding_dim"],
            expanded_dim=cfg["expanded_dim"],
            n_attn_heads=cfg["n_attn_heads"],
            n_attn_layers=cfg["n_attn_layers"],
        ).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/coco")
    p.add_argument("--device", default="mps")
    p.add_argument("--out", default="results/comparison")
    p.add_argument("--num", type=int, default=12)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    device = torch.device(args.device)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    versions = [
        ("v2", "results/coco_seg_v2/best_head.pt", "results/coco_seg_v2/hidden_cache"),
        ("v4", "results/coco_seg_v4/best_head.pt", "results/coco_seg_v4/hidden_cache"),
        ("v6", "results/coco_seg_v6/best_head.pt", "results/coco_seg_v6/hidden_cache"),
        ("v7", "results/coco_seg_v7/best_head.pt", "results/coco_seg_v7/hidden_cache"),
        ("v8", "results/coco_seg_v8/best_head.pt", "results/coco_seg_v8/hidden_cache"),
        ("v9", "results/coco_seg_v9/best_head.pt", "results/coco_seg_v9/hidden_cache"),
    ]
    versions = [(name, ck, h) for name, ck, h in versions if os.path.exists(ck)]
    print(f"[compare] Loaded versions: {[v[0] for v in versions]}")

    img_dir, ann_file = download_coco_val(args.data_dir)
    coco = COCO(ann_file)
    cat_info = coco.loadCats(coco.getCatIds())
    categories = {c["id"]: c["name"] for c in cat_info}

    clip_path = "results/coco_seg_v2/clip_text_embeddings.pt"
    clip_embeddings = cache_clip_text_embeddings(categories, clip_path, device)

    heads = {name: load_head(ck, device) for name, ck, _ in versions}
    hidden_dirs = {name: h for name, _, h in versions}

    img_ids = sorted(coco.getImgIds())
    random.shuffle(img_ids)

    cmap = matplotlib.colormaps["jet"]
    saved = 0

    for img_id in img_ids:
        if saved >= args.num:
            break

        first_hidden_dir = hidden_dirs[versions[0][0]]
        cache_path = os.path.join(first_hidden_dir, f"{img_id}.pt")
        if not os.path.exists(cache_path):
            continue

        cats = get_image_categories(coco, img_id)
        if not cats:
            continue
        cat_id = random.choice(list(cats.keys()))
        if str(cat_id) not in clip_embeddings:
            continue
        cat_name = categories.get(cat_id, f"cat_{cat_id}")
        anns = cats[cat_id]

        img_info = coco.imgs[img_id]
        img = Image.open(os.path.join(img_dir, img_info["file_name"])).convert("RGB")
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        mask = build_category_mask(coco, img_id, cat_id, anns)
        gt_14 = mask_to_patch_target(mask, PATCH_GRID)
        gt_up = F.interpolate(
            torch.from_numpy(gt_14).unsqueeze(0).unsqueeze(0),
            size=(h, w), mode="nearest",
        )[0, 0].numpy()
        gt_rgb = (cmap(gt_up)[..., :3] * 255).astype(np.uint8)
        gt_overlay = (0.5 * img_np.astype(np.float32) + 0.5 * gt_rgb).clip(0, 255).astype(np.uint8)

        # Predict per version
        preds = {}
        for name in heads:
            hidden_path = os.path.join(hidden_dirs[name], f"{img_id}.pt")
            if not os.path.exists(hidden_path):
                continue
            hidden = torch.load(hidden_path, map_location="cpu", weights_only=True).float()
            query = clip_embeddings[str(cat_id)]
            with torch.inference_mode():
                logits = heads[name](
                    hidden.unsqueeze(0).to(device),
                    query.unsqueeze(0).to(device),
                )
            n_out = logits.shape[-1]
            grid = int(round(n_out ** 0.5))
            probs = torch.sigmoid(logits).reshape(grid, grid).cpu().numpy()
            probs_up = F.interpolate(
                torch.from_numpy(probs).unsqueeze(0).unsqueeze(0),
                size=(h, w), mode="bilinear", align_corners=False,
            )[0, 0].numpy()
            probs_up = (probs_up - probs_up.min()) / (probs_up.max() - probs_up.min() + 1e-8)
            heat_rgb = (cmap(probs_up)[..., :3] * 255).astype(np.uint8)
            overlay = (0.5 * img_np.astype(np.float32) + 0.5 * heat_rgb).clip(0, 255).astype(np.uint8)
            preds[name] = overlay

        cols = 2 + len(preds)
        fig, axes = plt.subplots(1, cols, figsize=(cols * 4, 4))
        axes[0].imshow(img_np)
        axes[0].set_title(f"{cat_name} (img {img_id})")
        axes[0].axis("off")
        axes[1].imshow(gt_overlay)
        axes[1].set_title("GT mask (14×14 down)")
        axes[1].axis("off")
        for i, (name, overlay) in enumerate(preds.items()):
            axes[2 + i].imshow(overlay)
            axes[2 + i].set_title(name)
            axes[2 + i].axis("off")
        out_path = os.path.join(args.out, f"{saved:03d}_{img_id}_{cat_name}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close()
        saved += 1

    print(f"[compare] Saved {saved} comparisons to {args.out}")


if __name__ == "__main__":
    main()
