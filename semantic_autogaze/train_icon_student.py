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


def _is_native_cache(dinov2_cache: str) -> bool:
    """Sniff one .pt file to decide if cache is native-aspect (dict) or
    legacy letterbox (bare tensor)."""
    cache = Path(dinov2_cache)
    for p in cache.iterdir():
        if p.suffix == ".pt":
            obj = torch.load(p, map_location="cpu", weights_only=True)
            return isinstance(obj, dict) and "patches" in obj and "grid" in obj
    raise RuntimeError(f"no .pt files in {dinov2_cache}")


class _IconDataset(Dataset):
    """Yields (patch_features, query_embed, target) per (img_id, cat_id).

    Two modes (auto-detected from cache layout):

    - **letterbox** (legacy): cache has bare ``(256, 384)`` tensors. Target
      is the letterboxed COCO mask (or CLIPSeg cache) downsampled to
      ``target_grid``. Output grid = 16×16 → ``target_grid``.

    - **native**: cache has dicts ``{patches, grid, img_hw, encode_hw,
      crop_top_left, patch_size}``. At train time we crop a random
      ``in_grid``×``in_grid`` sub-grid of patches (default 16×16); the
      corresponding pixel-rectangle of the original image is computed
      and the COCO mask is cropped to it, then resized to
      ``target_grid``. This is autogaze-style RandomCrop in patch
      space, which keeps batches at fixed token count and trains the
      model on diverse spatial windows.
    """

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
        in_grid: int = 16,
        train_mode: bool = True,
    ):
        assert supervision in ("A", "B"), supervision
        self.coco = coco
        self.dinov2_cache = dinov2_cache
        self.supervision = supervision
        self.clipseg_cache = clipseg_cache
        self.target_grid = target_grid
        self.in_grid = in_grid
        self.train_mode = train_mode
        self.native = _is_native_cache(dinov2_cache)

        if self.native and supervision == "A":
            raise NotImplementedError(
                "Recipe A (CLIPSeg distillation) on a native-aspect cache "
                "would require per-crop CLIPSeg recompute. Use Recipe B."
            )

        self.cat_names = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
        self.clip_text = torch.load(clip_text_path, map_location="cpu", weights_only=True)

        self.samples: list[tuple[int, int]] = []
        all_cat_ids = list(self.cat_names.keys())
        for img_id in img_ids:
            cache_path = Path(dinov2_cache) / f"{img_id}.pt"
            if not cache_path.exists():
                continue
            present = get_image_categories(coco, img_id)
            cat_ids = list(present.keys())[:max_cats_per_image]
            if supervision == "A":
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

    def _load_patches(self, img_id: int):
        """Return (patches_tensor, meta_or_None). meta is None for letterbox."""
        obj = torch.load(Path(self.dinov2_cache) / f"{img_id}.pt", map_location="cpu",
                         weights_only=True)
        if self.native:
            return obj["patches"].float(), obj
        return obj.float(), None

    def _crop_native(self, patches_full: torch.Tensor, meta: dict, mask_native: np.ndarray):
        """Patch-space crop: take an in_grid×in_grid sub-grid; project to
        original-image pixels; crop the (h, w) COCO mask to that
        rectangle; resize to target_grid.

        Returns (patches_crop, target_grid_tensor).
        """
        n_h, n_w = meta["grid"]
        ig = self.in_grid
        if n_h < ig or n_w < ig:
            raise ValueError(f"img has grid {n_h}x{n_w} < in_grid {ig}; resize edge too small")
        if self.train_mode:
            r0 = random.randint(0, n_h - ig)
            c0 = random.randint(0, n_w - ig)
        else:
            r0 = (n_h - ig) // 2
            c0 = (n_w - ig) // 2
        # Crop patches: (n_h*n_w, D) → reshape (n_h, n_w, D), slice, flatten.
        D = patches_full.shape[-1]
        grid = patches_full.view(n_h, n_w, D)
        patches = grid[r0:r0 + ig, c0:c0 + ig, :].reshape(ig * ig, D).contiguous()

        # Project the patch sub-grid back to original-image pixel coords.
        ps = meta["patch_size"]
        enc_h, enc_w = meta["encode_hw"]
        top_pad, left_pad = meta["crop_top_left"]
        # In post-resize, post-crop frame: pixel (top, left, bottom, right) of sub-grid.
        sub_top = r0 * ps
        sub_left = c0 * ps
        sub_bot = sub_top + ig * ps
        sub_right = sub_left + ig * ps
        # Add the resize-then-crop offset to land in resized-image coords:
        rs_top = sub_top + top_pad
        rs_left = sub_left + left_pad
        rs_bot = sub_bot + top_pad
        rs_right = sub_right + left_pad
        # Map to original-image coords via the per-axis scale stored in
        # the cache (avoids re-deriving it from h0/w0/shortest_edge).
        h0, w0 = meta["img_hw"]
        new_h, new_w = meta["resize_hw"]
        sy = h0 / float(new_h)
        sx = w0 / float(new_w)
        og_top = max(0, int(round(rs_top * sy)))
        og_left = max(0, int(round(rs_left * sx)))
        og_bot = min(h0, int(round(rs_bot * sy)))
        og_right = min(w0, int(round(rs_right * sx)))
        if og_bot <= og_top or og_right <= og_left:
            target = torch.zeros(self.target_grid, self.target_grid)
        else:
            sub_mask = mask_native[og_top:og_bot, og_left:og_right]
            t = torch.from_numpy(sub_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=(self.target_grid, self.target_grid),
                              mode="bilinear", align_corners=False)
            target = t.squeeze(0).squeeze(0).clamp(0, 1)
        return patches, target

    def __getitem__(self, idx):
        img_id, cat_id = self.samples[idx]
        patches_full, meta = self._load_patches(img_id)
        query = self.clip_text[str(cat_id)].float()

        if self.supervision == "A":
            # Letterbox-only (native+A is rejected in __init__).
            cs = torch.load(Path(self.clipseg_cache) / f"{img_id}_g{self.target_grid}.pt",
                            map_location="cpu", weights_only=True)
            target = cs.get(str(cat_id))
            if target is None:
                target = torch.zeros(self.target_grid, self.target_grid, dtype=torch.float16)
            target = target.float()
            return patches_full, query, target

        # Recipe B: COCO masks.
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=0))
        if not anns:
            target = torch.zeros(self.target_grid, self.target_grid)
            if self.native:
                # still need to crop patches to in_grid x in_grid
                m_native = np.zeros(meta["img_hw"], dtype=np.float32)
                patches, target = self._crop_native(patches_full, meta, m_native)
                return patches, query, target
            return patches_full, query, target

        m = build_category_mask(self.coco, img_id, cat_id, anns)
        if self.native:
            patches, target = self._crop_native(patches_full, meta, m)
            return patches, query, target
        # legacy letterbox path
        m_sq, _ = letterbox_mask(m)
        t = torch.from_numpy(m_sq.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(self.target_grid, self.target_grid),
                          mode="bilinear", align_corners=False)
        target = t.squeeze(0).squeeze(0).clamp(0, 1)
        return patches_full, query, target


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
    p.add_argument("--val-ann", default=None,
                   help="If set, val on this annotation file instead of "
                        "internal --val-frac split. Use to train on train2017 "
                        "and validate on val2017. Requires --val-img-subdir "
                        "and --val-dinov2-cache.")
    p.add_argument("--val-img-subdir", default=None)
    p.add_argument("--val-dinov2-cache", default=None)
    args = p.parse_args()
    if (args.val_ann or args.val_img_subdir or args.val_dinov2_cache) and not (
        args.val_ann and args.val_img_subdir and args.val_dinov2_cache
    ):
        raise SystemExit("--val-ann/--val-img-subdir/--val-dinov2-cache must be set together")

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

    # split: either external val ann (full train + full val from separate files)
    # or internal random split of img_ids by --val-frac.
    if args.val_ann is not None:
        train_ids = img_ids
        val_coco = COCO(os.path.join(args.data_dir, args.val_ann))
        val_ids = sorted(val_coco.getImgIds())
        if args.max_images is not None:
            val_ids = val_ids[: args.max_images]
        val_dinov2_cache = args.val_dinov2_cache
        print(f"[data] external val: train={len(train_ids)} ({args.ann}) "
              f"val={len(val_ids)} ({args.val_ann})")
    else:
        rng = random.Random(args.seed)
        shuffled = list(img_ids)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * args.val_frac))
        val_ids = sorted(shuffled[:n_val])
        train_ids = sorted(shuffled[n_val:])
        val_coco = coco
        val_dinov2_cache = args.dinov2_cache
        print(f"[data] internal split: train={len(train_ids)} val={len(val_ids)}")

    common_train = dict(
        coco=coco,
        dinov2_cache=args.dinov2_cache,
        clip_text_path=args.clip_text_cache,
        supervision=args.supervision,
        clipseg_cache=args.clipseg_cache,
    )
    common_val = dict(
        coco=val_coco,
        dinov2_cache=val_dinov2_cache,
        clip_text_path=args.clip_text_cache,
        supervision=args.supervision,
        clipseg_cache=args.clipseg_cache,
    )
    train_ds = _IconDataset(img_ids=train_ids, train_mode=True, **common_train)
    val_ds = _IconDataset(img_ids=val_ids, train_mode=False, **common_val)
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
