"""Pre-cache DINOv2-small patch features for all images in a COCO split.

Run once per dataset; both recipe A and recipe B trainings reuse the cache.

Each image is letterbox-padded to a square (no aspect distortion, no content
loss), then resized to 224×224, passed through the frozen DINOv2 encoder,
and the per-patch hidden state (no CLS) is saved as a (256, 384) float16
tensor at `<cache_dir>/<img_id>.pt`. The patches are in letterbox-square
frame indexed by L = max(h, w) of the original image.
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


class _ImageDataset(Dataset):
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


@torch.inference_mode()
def cache_dinov2(
    coco_ann: str,
    img_dir: str,
    cache_dir: str,
    device: str = "mps",
    batch_size: int = 16,
    num_workers: int = 2,
    max_images: int | None = None,
    model_name: str = "facebook/dinov2-small",
):
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
    print(f"[dinov2] need to cache {len(pending)}/{len(img_ids)} images → {cache_dir}")

    # We borrow only the normalization constants from the HF processor and
    # build our own letterbox-aware preprocessing pipeline. Input size is
    # taken from the processor's crop_size (224 for DINOv2-small), which is
    # what the original cache used and what IconStudent's pos_embed expects
    # (16x16 = 256 patches). The model itself supports variable resolution,
    # but downstream code assumes the 16x16 grid.
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    input_size = (
        processor.crop_size["height"]
        if hasattr(processor, "crop_size") and processor.crop_size
        else processor.size.get("shortest_edge", 224)
    )
    print(
        f"[dinov2] model on {device}, hidden={model.config.hidden_size}, "
        f"patch={model.config.patch_size}, input={input_size}, "
        f"frame=letterbox-square"
    )

    ds = _ImageDataset(
        pending,
        img_dir,
        file_names,
        image_mean=list(processor.image_mean),
        image_std=list(processor.image_std),
        input_size=input_size,
    )
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    for batch_ids, pixels in tqdm(loader, desc="dinov2 cache"):
        pixels = pixels.to(device)
        out = model(pixel_values=pixels)
        # Drop CLS token (first); take patch tokens only.
        feats = out.last_hidden_state[:, 1:, :].cpu().to(torch.float16)
        for i, img_id in enumerate(batch_ids.tolist()):
            torch.save(feats[i].contiguous().clone(), Path(cache_dir) / f"{img_id}.pt")

    print(f"[dinov2] done. {len(list(Path(cache_dir).glob('*.pt')))} cached at {cache_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann", required=True, help="COCO annotation json (e.g. instances_val2017.json)")
    p.add_argument("--img-dir", required=True, help="image directory (e.g. data/coco/val2017)")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--model", default="facebook/dinov2-small")
    args = p.parse_args()

    cache_dinov2(
        coco_ann=args.ann,
        img_dir=args.img_dir,
        cache_dir=args.cache_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images=args.max_images,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
