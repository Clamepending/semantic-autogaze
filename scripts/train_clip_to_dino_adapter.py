import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from pycocotools.coco import COCO
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor


@dataclass
class TrainConfig:
    coco_root: str
    ann_file: str
    output_dir: str
    dino_model: str = "facebook/dinov2-small"
    clip_model: str = "openai/clip-vit-base-patch16"
    dino_size: int = 224
    batch_size: int = 16
    steps: int = 1000
    lr: float = 3e-4
    num_workers: int = 4
    seed: int = 0
    device: str = "cuda"
    val_samples: int = 8
    log_every: int = 25
    val_every: int = 250
    hidden_dim: int | None = None
    wandb_project: str = "semantic-autogaze"
    wandb_run_name: str | None = None


class CocoInstanceTextDataset(Dataset):
    """Each sample is one (image, category) pair with ALL instances of that category merged."""

    def __init__(self, coco_root: str, ann_file: str, dino_size: int, seed: int = 0):
        self.coco_root = Path(coco_root)
        self.coco = COCO(ann_file)
        self.dino_size = dino_size
        self.categories = {cat["id"]: cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())}

        anns = self.coco.loadAnns(self.coco.getAnnIds())
        grouped: dict[tuple[int, int], list] = defaultdict(list)
        for ann in anns:
            if ann.get("iscrowd", 0) == 0 and ann.get("area", 0) > 64 and ann.get("segmentation"):
                grouped[(ann["image_id"], ann["category_id"])].append(ann)

        self.samples = list(grouped.values())
        rng = random.Random(seed)
        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        anns = self.samples[index]
        image_id = anns[0]["image_id"]
        cat_id = anns[0]["category_id"]
        image_info = self.coco.loadImgs([image_id])[0]
        image_path = self.coco_root / image_info["file_name"]
        image = Image.open(image_path).convert("RGB").resize((self.dino_size, self.dino_size), Image.BICUBIC)

        mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.float32)
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann).astype(np.float32))
        mask = cv2.resize(mask, (self.dino_size, self.dino_size), interpolation=cv2.INTER_AREA)
        mask = np.clip(mask, 0.0, 1.0)

        category = self.categories[cat_id]
        text = f"a photo of a {category}"
        return image, torch.from_numpy(mask).float(), text, image_info["file_name"], category


def collate(batch):
    images, masks, texts, names, categories = zip(*batch)
    return list(images), torch.stack(masks), list(texts), list(names), list(categories)


def infer_grid(num_patches: int) -> tuple[int, int]:
    side = int(math.sqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} patches")
    return side, side


def patch_targets(masks: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    pooled = F.adaptive_avg_pool2d(masks.unsqueeze(1), (grid_h, grid_w))
    return pooled.flatten(1)


def clip_text_features(clip_processor, clip_model, texts: list[str], device: str) -> torch.Tensor:
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = clip_model.text_model(**inputs)
        features = clip_model.text_projection(outputs.pooler_output)
    return F.normalize(features, dim=-1).detach()


def dino_patch_features(dino_processor, dino_model, images: list[Image.Image], device: str) -> torch.Tensor:
    inputs = dino_processor(images=images, return_tensors="pt", do_resize=False, do_center_crop=False).to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    return F.normalize(outputs.last_hidden_state[:, 1:], dim=-1).detach()


class ClipTextToDinoAdapter(nn.Module):
    def __init__(self, clip_dim: int, dino_dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            self.proj = nn.Sequential(
                nn.LayerNorm(clip_dim),
                nn.Linear(clip_dim, dino_dim),
            )
        else:
            self.proj = nn.Sequential(
                nn.LayerNorm(clip_dim),
                nn.Linear(clip_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dino_dim),
            )
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, text_features: torch.Tensor, patch_features: torch.Tensor) -> torch.Tensor:
        query = F.normalize(self.proj(text_features), dim=-1)
        logits = torch.einsum("bnd,bd->bn", patch_features, query)
        return logits * self.logit_scale.clamp(1.0, 100.0)


def render_examples(output_dir: Path, adapter, batch, dino_processor, dino_model, clip_processor, clip_model, device: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    images, masks, texts, names, categories = batch
    patch_features = dino_patch_features(dino_processor, dino_model, images, device)
    grid_h, grid_w = infer_grid(patch_features.shape[1])
    text_features = clip_text_features(clip_processor, clip_model, texts, device)
    with torch.no_grad():
        logits = adapter(text_features, patch_features)
        probs = torch.sigmoid(logits).reshape(-1, grid_h, grid_w).cpu().numpy()

    for i, image in enumerate(images[:8]):
        try:
            arr = np.array(image)
            heat = cv2.resize(probs[i], image.size, interpolation=cv2.INTER_NEAREST)
            target = cv2.resize(masks[i].numpy(), image.size, interpolation=cv2.INTER_AREA)

            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(arr)
            axes[0].set_title(categories[i])
            axes[1].imshow(arr, alpha=0.55)
            axes[1].imshow(target, alpha=0.55, vmin=0, vmax=1, cmap="viridis")
            axes[1].set_title("target")
            axes[2].imshow(arr, alpha=0.55)
            axes[2].imshow(heat, alpha=0.55, vmin=0, vmax=1, cmap="turbo")
            axes[2].set_title("prediction")
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
            fig.tight_layout()
            safe_name = Path(names[i]).stem
            fig.savefig(output_dir / f"{i:02d}_{safe_name}_{categories[i].replace(' ', '_')}.png", dpi=120)
            plt.close(fig)
        except Exception:
            plt.close("all")


def train(config: TrainConfig):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name or output_dir.name,
        config=asdict(config),
    )

    device = config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu"
    dataset = CocoInstanceTextDataset(config.coco_root, config.ann_file, config.dino_size, config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate,
        drop_last=True,
    )

    print(f"device={device} samples={len(dataset)} output={output_dir}", flush=True)
    dino_processor = AutoImageProcessor.from_pretrained(config.dino_model)
    dino_model = AutoModel.from_pretrained(config.dino_model).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(config.clip_model)
    clip_model = CLIPModel.from_pretrained(config.clip_model, use_safetensors=True).to(device).eval()

    for module in [dino_model, clip_model]:
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    images, masks, texts, _, _ = next(iter(loader))
    patch_features = dino_patch_features(dino_processor, dino_model, images, device)
    text_features = clip_text_features(clip_processor, clip_model, texts, device)
    grid_h, grid_w = infer_grid(patch_features.shape[1])

    adapter = ClipTextToDinoAdapter(text_features.shape[-1], patch_features.shape[-1], hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.lr, weight_decay=1e-4)

    val_loader = DataLoader(dataset, batch_size=config.val_samples, shuffle=True, num_workers=0, collate_fn=collate)
    val_batch = next(iter(val_loader))

    history = []
    step = 0
    pbar = tqdm(total=config.steps)
    while step < config.steps:
        for images, masks, texts, names, categories in loader:
            step += 1
            masks = masks.to(device)
            patch_features = dino_patch_features(dino_processor, dino_model, images, device)
            text_features = clip_text_features(clip_processor, clip_model, texts, device)

            targets = patch_targets(masks, grid_h, grid_w)
            logits = adapter(text_features, patch_features)
            loss = F.binary_cross_entropy_with_logits(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pred = probs > 0.5
                target_bin = targets > 0.25
                intersection = (pred & target_bin).float().sum()
                union = (pred | target_bin).float().sum().clamp_min(1.0)
                iou = intersection / union

            if step % config.log_every == 0 or step == 1:
                loss_val = float(loss.detach().cpu())
                iou_val = float(iou.cpu())
                record = {"step": step, "loss": loss_val, "patch_iou": iou_val}
                history.append(record)
                wandb.log({"train/loss": loss_val, "train/patch_iou": iou_val}, step=step)
                print(json.dumps(record), flush=True)

            if step % config.val_every == 0:
                examples_dir = output_dir / "examples" / f"step_{step:06d}"
                render_examples(
                    examples_dir,
                    adapter.eval(),
                    val_batch,
                    dino_processor,
                    dino_model,
                    clip_processor,
                    clip_model,
                    device,
                )
                adapter.train()
                images_to_log = []
                for img_path in sorted(examples_dir.glob("*.png"))[:8]:
                    try:
                        images_to_log.append(wandb.Image(str(img_path)))
                    except Exception:
                        pass
                if images_to_log:
                    wandb.log({"val/examples": images_to_log}, step=step)

            pbar.update(1)
            if step >= config.steps:
                break
    pbar.close()

    torch.save({"adapter": adapter.state_dict(), "config": asdict(config)}, output_dir / "adapter.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    render_examples(
        output_dir / "examples" / "final",
        adapter.eval(),
        val_batch,
        dino_processor,
        dino_model,
        clip_processor,
        clip_model,
        device,
    )
    wandb.finish()
    print(f"saved {output_dir / 'adapter.pt'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", default="/home/ogata/semantic-autogaze/data/coco_train2017/train2017")
    parser.add_argument("--ann-file", default="/home/ogata/semantic-autogaze/data/coco_train2017/annotations/instances_train2017.json")
    parser.add_argument("--output-dir", default="/home/ogata/semantic-autogaze/results/clip_text_to_dino_adapter")
    parser.add_argument("--dino-model", default="facebook/dinov2-small")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--dino-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--wandb-project", default="semantic-autogaze")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()
    train(TrainConfig(**vars(args)))


if __name__ == "__main__":
    main()
