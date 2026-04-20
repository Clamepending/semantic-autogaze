"""Train IconStudent with one of two supervisions:

  --supervision A   distill CLIPSeg soft heatmaps (BCE on logits)
  --supervision B   COCO instance masks            (focal + dice)

Same architecture, same data, same images, different target. Lets us
ask: which supervision produces better heatmaps for a small student?

Pipeline: DINOv2-small patch features (frozen, cached) + CLIP-ViT-B/16
text queries (frozen, cached) + tiny cross-attention decoder (~9.5M
trainable) → 128×128 heatmap.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from semantic_autogaze.icon_student import IconStudent, trainable_param_count
from semantic_autogaze.letterbox import letterbox_mask
from semantic_autogaze.train_coco_seg import (
    build_category_mask,
    cache_clip_text_embeddings,
    generate_clipseg_targets,
    get_image_categories,
)


CLIPSEG_TARGET_GRID = 128


class _IconDataset(Dataset):
    """Yields (patch_features, query_embed, target_128, meta) per (img_id, cat_id)."""

    def __init__(
        self,
        coco: COCO,
        img_ids: list[int],
        dinov2_cache: str,
        clip_text_path: str,
        supervision: str,
        clipseg_cache: str | None,
        max_cats_per_image: int = 5,
        include_clipseg_negatives: bool = False,
        target_grid: int = CLIPSEG_TARGET_GRID,
    ):
        assert supervision in ("A", "B"), supervision
        self.coco = coco
        self.dinov2_cache = dinov2_cache
        self.supervision = supervision
        self.clipseg_cache = clipseg_cache
        self.target_grid = target_grid

        self.cat_names = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
        self.clip_text = torch.load(clip_text_path, map_location="cpu", weights_only=True)

        # Build (img_id, cat_id) sample list — only for images we have all data for.
        self.samples: list[tuple[int, int]] = []
        all_cat_ids = list(self.cat_names.keys())
        for img_id in img_ids:
            cache_path = Path(dinov2_cache) / f"{img_id}.pt"
            if not cache_path.exists():
                continue
            present = get_image_categories(coco, img_id)
            cat_ids = list(present.keys())[:max_cats_per_image]
            if supervision == "A":
                # CLIPSeg cache may also include negatives — load all keys.
                cs_path = Path(clipseg_cache) / f"{img_id}_g{target_grid}.pt"
                if not cs_path.exists():
                    continue
                cs_keys = torch.load(cs_path, map_location="cpu", weights_only=True).keys()
                cat_ids = [int(k) for k in cs_keys]
                if include_clipseg_negatives:
                    absent = [c for c in all_cat_ids if c not in present]
                    random.shuffle(absent)
                    cat_ids = list(set(cat_ids) | set(absent[:2]))
            for cid in cat_ids:
                if str(cid) in self.clip_text:
                    self.samples.append((img_id, cid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, cat_id = self.samples[idx]
        patches = torch.load(Path(self.dinov2_cache) / f"{img_id}.pt", map_location="cpu",
                             weights_only=True).float()
        query = self.clip_text[str(cat_id)].float()

        if self.supervision == "A":
            cs = torch.load(Path(self.clipseg_cache) / f"{img_id}_g{self.target_grid}.pt",
                            map_location="cpu", weights_only=True)
            target = cs.get(str(cat_id))
            if target is None:
                # category not in CLIPSeg cache → zeros (treat as absent)
                target = torch.zeros(self.target_grid, self.target_grid, dtype=torch.float16)
            target = target.float()
        else:
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=0))
            if anns:
                m = build_category_mask(self.coco, img_id, cat_id, anns)
                # Letterbox the native-resolution mask to a square before
                # downsampling so the target's geometry matches DINOv2's
                # letterbox-square input frame.
                m_sq, _ = letterbox_mask(m)
                t = torch.from_numpy(m_sq.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                t = F.interpolate(t, size=(self.target_grid, self.target_grid),
                                  mode="bilinear", align_corners=False)
                target = t.squeeze(0).squeeze(0).clamp(0, 1)
            else:
                target = torch.zeros(self.target_grid, self.target_grid)
        return patches, query, target


def soft_dice(probs, target, eps=1e-6):
    p = probs.flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    denom = p.sum(1) + t.sum(1)
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


def focal(logits, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.exp(-bce)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    return (a_t * (1 - pt) ** gamma * bce).mean()


def loss_recipe_A(logits, target):
    """CLIPSeg soft target: BCE on raw logits is the natural distillation loss."""
    return F.binary_cross_entropy_with_logits(logits, target)


def loss_recipe_B(logits, target):
    """COCO hard mask: focal + dice."""
    return focal(logits, target) + 1.0 * soft_dice(torch.sigmoid(logits), target)


def train_one_epoch(model, loader, optim, device, supervision, epoch, total_epochs):
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"epoch {epoch+1}/{total_epochs} [{supervision}]")
    for patches, query, target in pbar:
        patches = patches.to(device)
        query = query.to(device)
        target = target.to(device)
        logits = model(patches, query)
        if supervision == "A":
            loss = loss_recipe_A(logits, target)
        else:
            loss = loss_recipe_B(logits, target)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return sum(losses) / len(losses)


@torch.inference_mode()
def validate(model, loader, device, supervision):
    model.eval()
    losses = []
    for patches, query, target in tqdm(loader, desc="val"):
        patches = patches.to(device)
        query = query.to(device)
        target = target.to(device)
        logits = model(patches, query)
        if supervision == "A":
            loss = loss_recipe_A(logits, target)
        else:
            loss = loss_recipe_B(logits, target)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--supervision", required=True, choices=["A", "B"])
    p.add_argument("--data-dir", default="data/coco")
    p.add_argument("--ann", default="annotations/instances_val2017.json")
    p.add_argument("--img-subdir", default="val2017")
    p.add_argument("--dinov2-cache", required=True)
    p.add_argument("--clipseg-cache", default=None,
                   help="Required for --supervision A; precomputed at target_grid=128")
    p.add_argument("--clip-text-cache", default=None,
                   help="If absent, will be generated to <output>/clip_text_embeddings.pt")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--decoder-dim", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    coco = COCO(os.path.join(args.data_dir, args.ann))
    img_ids = sorted(coco.getImgIds())
    if args.max_images is not None:
        img_ids = img_ids[: args.max_images]

    cat_names = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    if args.clip_text_cache is None:
        args.clip_text_cache = os.path.join(args.output_dir, "clip_text_embeddings.pt")
        if not os.path.exists(args.clip_text_cache):
            cache_clip_text_embeddings(cat_names, args.clip_text_cache, device=str(device))

    if args.supervision == "A" and args.clipseg_cache is None:
        raise SystemExit("--supervision A requires --clipseg-cache")
    if args.supervision == "A":
        # Generate CLIPSeg targets at 128 if not present
        generate_clipseg_targets(
            img_dir=os.path.join(args.data_dir, args.img_subdir),
            coco=coco,
            img_ids=img_ids,
            categories=cat_names,
            cache_dir=args.clipseg_cache,
            device=device,
            target_grid=CLIPSEG_TARGET_GRID,
            max_cats_per_image=5,
            include_negatives=False,
        )

    # split
    rng = random.Random(args.seed)
    shuffled = list(img_ids)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_frac))
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    print(f"[data] train={len(train_ids)} val={len(val_ids)}")

    common = dict(
        coco=coco,
        dinov2_cache=args.dinov2_cache,
        clip_text_path=args.clip_text_cache,
        supervision=args.supervision,
        clipseg_cache=args.clipseg_cache,
    )
    train_ds = _IconDataset(img_ids=train_ids, **common)
    val_ds = _IconDataset(img_ids=val_ids, **common)
    print(f"[data] train samples={len(train_ds)} val samples={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = IconStudent(
        patch_dim=384, query_dim=512,
        decoder_dim=args.decoder_dim, in_grid=16, out_grid=CLIPSEG_TARGET_GRID,
        n_layers=args.n_layers, n_heads=8, dropout=0.0,
    ).to(device)
    print(f"[model] trainable params: {trainable_param_count(model)/1e6:.2f}M")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    history = []
    for epoch in range(args.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optim, device,
                                     args.supervision, epoch, args.num_epochs)
        val_loss = validate(model, val_loader, device, args.supervision)
        history.append({"epoch": epoch + 1, "train": train_loss, "val": val_loss})
        print(f"[epoch {epoch+1}] train={train_loss:.4f} val={val_loss:.4f}")

        ckpt = {
            "state_dict": model.state_dict(),
            "config": {
                "patch_dim": 384, "query_dim": 512, "decoder_dim": args.decoder_dim,
                "in_grid": 16, "out_grid": CLIPSEG_TARGET_GRID,
                "n_layers": args.n_layers, "n_heads": 8,
            },
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "supervision": args.supervision,
        }
        torch.save(ckpt, os.path.join(args.output_dir, "latest.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))
            print(f"  → new best val={val_loss:.4f}")

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[done] best val={best_val:.4f}")


if __name__ == "__main__":
    main()
