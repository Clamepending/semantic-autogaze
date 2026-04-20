"""Pre-cache DINOv2-small patch features for all images in a COCO split.

Run once per dataset; both recipe A and recipe B trainings reuse the cache.

Two modes:

- ``--mode letterbox`` (legacy): each image is letterbox-padded to a
  square, resized to 224×224, encoded → (256, 384) float16 tensor saved
  as a bare tensor at ``<cache_dir>/<img_id>.pt``. Patches in
  letterbox-square frame indexed by L = max(h, w).

- ``--mode native`` (autogaze-style, default): each image is resized so
  ``shortest_edge=224``, then center-cropped to the nearest multiple of
  the patch size (14) on each axis (so the result is exactly N_h×14 by
  N_w×14 with the shorter axis = 16 patches). Encoded → (N_h*N_w, 384)
  float16 patches; saved as a dict
  ``{"patches": tensor, "grid": (n_h, n_w), "img_hw": (h, w),
     "encode_hw": (n_h*14, n_w*14), "patch_size": 14}`` so the train /
  inference paths can map patches back to original-image pixel
  coordinates. Aspect ratio is preserved, no padding, no content loss
  beyond the centered crop-to-multiple-of-14.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from semantic_autogaze.letterbox import letterbox_image


class _LetterboxDataset(Dataset):
    def __init__(
        self,
        img_ids: list[int],
        img_dir: str,
        file_names: dict[int, str],
        image_mean: list[float],
        image_std: list[float],
        input_size: int = 224,
    ):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.file_names = file_names
        self.image_mean = image_mean
        self.image_std = image_std
        self.input_size = input_size

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        path = os.path.join(self.img_dir, self.file_names[img_id])
        img = Image.open(path).convert("RGB")
        # Letterbox to square first → resize to model's native input → normalize.
        # This replaces AutoImageProcessor's shortest-edge + center-crop, which
        # silently discards content for non-square images.
        square, _ = letterbox_image(img)
        if square.size != (self.input_size, self.input_size):
            square = square.resize((self.input_size, self.input_size), Image.BILINEAR)
        tensor = TF.to_tensor(square)  # (3, H, W) in [0, 1]
        tensor = TF.normalize(tensor, mean=self.image_mean, std=self.image_std)
        return img_id, tensor


def _native_aspect_resize_crop(
    img: Image.Image, shortest_edge: int = 224, patch_size: int = 14
) -> tuple[Image.Image, dict]:
    """Resize so shortest_edge==``shortest_edge``, then center-crop both
    axes to a multiple of ``patch_size``.

    Returns ``(processed_image, meta)`` where meta has the original
    ``img_hw`` and the post-crop ``encode_hw`` and ``grid``.
    """
    w0, h0 = img.size
    scale = shortest_edge / float(min(h0, w0))
    new_h = int(round(h0 * scale))
    new_w = int(round(w0 * scale))
    img_r = img.resize((new_w, new_h), Image.BILINEAR)

    # Center-crop each axis to a multiple of patch_size.
    n_h = new_h // patch_size
    n_w = new_w // patch_size
    enc_h = n_h * patch_size
    enc_w = n_w * patch_size
    top = (new_h - enc_h) // 2
    left = (new_w - enc_w) // 2
    img_c = img_r.crop((left, top, left + enc_w, top + enc_h))

    meta = {
        "img_hw": (h0, w0),
        "resize_hw": (new_h, new_w),
        "encode_hw": (enc_h, enc_w),
        "grid": (n_h, n_w),
        "crop_top_left": (top, left),
        "patch_size": patch_size,
    }
    return img_c, meta


class _NativeDataset(Dataset):
    """Yields (img_id, pixel_tensor, n_h, n_w, meta_pickle_bytes).

    Variable image sizes per sample → custom collate must keep a list,
    not stack. We use ``batch_size=1`` for simplicity (DINOv2-small at
    these resolutions is fast enough that batching's not a big win).
    """

    def __init__(
        self,
        img_ids: list[int],
        img_dir: str,
        file_names: dict[int, str],
        image_mean: list[float],
        image_std: list[float],
        shortest_edge: int = 224,
        patch_size: int = 14,
    ):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.file_names = file_names
        self.image_mean = image_mean
        self.image_std = image_std
        self.shortest_edge = shortest_edge
        self.patch_size = patch_size

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        path = os.path.join(self.img_dir, self.file_names[img_id])
        img = Image.open(path).convert("RGB")
        proc, meta = _native_aspect_resize_crop(img, self.shortest_edge, self.patch_size)
        tensor = TF.to_tensor(proc)
        tensor = TF.normalize(tensor, mean=self.image_mean, std=self.image_std)
        return img_id, tensor, meta


def _unwrap_single(batch):
    """DataLoader collate for batch_size=1 with non-stackable items."""
    return batch[0]


@torch.inference_mode()
def cache_dinov2(
    coco_ann: str,
    img_dir: str,
    cache_dir: str,
    mode: str = "native",
    device: str = "mps",
    batch_size: int = 16,
    num_workers: int = 2,
    max_images: int | None = None,
    model_name: str = "facebook/dinov2-small",
    shortest_edge: int = 224,
):
    assert mode in ("letterbox", "native"), mode
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    coco = COCO(coco_ann)
    img_ids = sorted(coco.getImgIds())
    if max_images is not None:
        img_ids = img_ids[:max_images]
    file_names = {i: coco.imgs[i]["file_name"] for i in img_ids}

    pending = [i for i in img_ids if not (Path(cache_dir) / f"{i}.pt").exists()]
    if not pending:
        print(f"[dinov2] all {len(img_ids)} images already cached at {cache_dir}")
        return
    print(f"[dinov2] mode={mode} need to cache {len(pending)}/{len(img_ids)} images → {cache_dir}")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    patch_size = model.config.patch_size

    if mode == "letterbox":
        input_size = (
            processor.crop_size["height"]
            if hasattr(processor, "crop_size") and processor.crop_size
            else processor.size.get("shortest_edge", 224)
        )
        print(
            f"[dinov2] model on {device}, hidden={model.config.hidden_size}, "
            f"patch={patch_size}, input={input_size}, frame=letterbox-square"
        )
        ds = _LetterboxDataset(
            pending, img_dir, file_names,
            image_mean=list(processor.image_mean),
            image_std=list(processor.image_std),
            input_size=input_size,
        )
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        for batch_ids, pixels in tqdm(loader, desc="dinov2 cache (letterbox)"):
            pixels = pixels.to(device)
            out = model(pixel_values=pixels)
            feats = out.last_hidden_state[:, 1:, :].cpu().to(torch.float16)
            for i, img_id in enumerate(batch_ids.tolist()):
                torch.save(feats[i].contiguous().clone(), Path(cache_dir) / f"{img_id}.pt")
    else:
        print(
            f"[dinov2] model on {device}, hidden={model.config.hidden_size}, "
            f"patch={patch_size}, shortest_edge={shortest_edge}, frame=native-aspect"
        )
        ds = _NativeDataset(
            pending, img_dir, file_names,
            image_mean=list(processor.image_mean),
            image_std=list(processor.image_std),
            shortest_edge=shortest_edge,
            patch_size=patch_size,
        )
        # Variable image shapes → batch_size=1 (no collate). DINOv2-small
        # at these resolutions is fast enough that batching is not the
        # bottleneck (disk I/O is). collate_fn must be a top-level
        # function so it pickles for multi-worker DataLoader.
        loader = DataLoader(ds, batch_size=1, num_workers=num_workers, shuffle=False,
                            collate_fn=_unwrap_single)
        for img_id, pixels, meta in tqdm(loader, desc="dinov2 cache (native)"):
            pixels = pixels.unsqueeze(0).to(device)  # (1, 3, H, W)
            out = model(pixel_values=pixels)
            feats = out.last_hidden_state[0, 1:, :].cpu().to(torch.float16).contiguous().clone()
            n_h, n_w = meta["grid"]
            assert feats.shape[0] == n_h * n_w, (
                f"img {img_id}: DINOv2 returned {feats.shape[0]} patches but "
                f"meta says {n_h}*{n_w}={n_h*n_w}"
            )
            torch.save({
                "patches": feats,
                "grid": (n_h, n_w),
                "img_hw": meta["img_hw"],
                "resize_hw": meta["resize_hw"],
                "encode_hw": meta["encode_hw"],
                "crop_top_left": meta["crop_top_left"],
                "patch_size": meta["patch_size"],
            }, Path(cache_dir) / f"{img_id}.pt")

    print(f"[dinov2] done. {len(list(Path(cache_dir).glob('*.pt')))} cached at {cache_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann", required=True, help="COCO annotation json (e.g. instances_val2017.json)")
    p.add_argument("--img-dir", required=True, help="image directory (e.g. data/coco/val2017)")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--mode", default="native", choices=["letterbox", "native"])
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=16,
                   help="ignored in --mode native (uses batch=1 due to variable shape)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--model", default="facebook/dinov2-small")
    p.add_argument("--shortest-edge", type=int, default=224,
                   help="native mode: target shortest-edge after resize (multiple of patch_size)")
    args = p.parse_args()

    cache_dinov2(
        coco_ann=args.ann,
        img_dir=args.img_dir,
        cache_dir=args.cache_dir,
        mode=args.mode,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images=args.max_images,
        model_name=args.model,
        shortest_edge=args.shortest_edge,
    )


if __name__ == "__main__":
    main()
