"""Pre-cache DINOv2-small patch features for all images in a COCO split.

Run once per dataset; both recipe A and recipe B trainings reuse the cache.

Each image is resized to 224×224 (DINOv2's native res), passed through the
frozen encoder, and the per-patch hidden state (no CLS) is saved as a
(256, 384) float16 tensor at `<cache_dir>/<img_id>.pt`.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


class _ImageDataset(Dataset):
    def __init__(self, img_ids: list[int], img_dir: str, file_names: dict[int, str], processor):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.file_names = file_names
        self.processor = processor

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        path = os.path.join(self.img_dir, self.file_names[img_id])
        img = Image.open(path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        return img_id, inputs["pixel_values"][0]


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

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    print(f"[dinov2] model on {device}, hidden={model.config.hidden_size}, patch={model.config.patch_size}")

    ds = _ImageDataset(pending, img_dir, file_names, processor)
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
