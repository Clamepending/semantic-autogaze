"""r/independent-text-scorer-v1 cycle 1.

Train a small text-conditioned patch scorer on top of frozen CLIP ViT-B/16
patch features. Loss = BCE(pred, COCO_GT_at_14x14) * alpha + MSE(pred, CLIPSeg_at_14x14) * beta.

Output: /home/ogata/semantic-autogaze/results/independent_scorer/best_v1.pt
"""
from __future__ import annotations
import os, sys, time, json, argparse, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, "/home/ogata/semantic-autogaze")
from pycocotools.coco import COCO

GRID = 14
N_PATCH = GRID * GRID

CLIP_NAME = "ViT-B-16"; CLIP_PRETRAINED = "openai"
CLIPSEG_NAME = "CIDAS/clipseg-rd64-refined"

# 8 image_ids used in the qualitative grid — held out from training.
QUAL_GRID_HOLDOUT = {337987, 32861, 370208, 346638, 223747, 267300, 151962, 232489}

# Cap candidate categories per image (largest by area)
MAX_CATS_PER_IMAGE = 3
MIN_AREA_FRAC = 0.03  # skip tiny instances


class CocoCatScorerDataset(Dataset):
    """Each item = (image, query_text, GT_mask_14x14, CLIPSeg_target_14x14_indices_hint).
    The CLIPSeg target is computed on-the-fly in a worker (slow but simple)."""

    def __init__(self, coco_root: str, split: str = "train"):
        self.img_dir = Path(coco_root) / "val2017"
        self.coco = COCO(str(Path(coco_root) / "annotations" / "instances_val2017.json"))
        self.cat_id_to_name = {c["id"]: c["name"] for c in self.coco.loadCats(self.coco.getCatIds())}

        # Build per-image candidate (image, cat) pairs by selecting top-K largest instances per category
        items = []
        all_img_ids = sorted(self.coco.getImgIds())
        for img_id in all_img_ids:
            holdout = img_id in QUAL_GRID_HOLDOUT
            if (split == "train" and holdout) or (split == "val" and not holdout):
                continue
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            if not anns: continue
            info = self.coco.loadImgs([img_id])[0]
            H, W = info["height"], info["width"]

            # Group by category, pick the largest instance per category
            cat_to_largest = {}
            for ann in anns:
                area = ann.get("area", 0) / max(1, H * W)
                if area < MIN_AREA_FRAC: continue
                cid = ann["category_id"]
                if cid not in cat_to_largest or ann["area"] > cat_to_largest[cid]["area"]:
                    cat_to_largest[cid] = ann

            # Take the top MAX_CATS_PER_IMAGE by area
            top = sorted(cat_to_largest.values(), key=lambda a: -a["area"])[:MAX_CATS_PER_IMAGE]
            for ann in top:
                items.append({
                    "image_id": img_id,
                    "file_name": info["file_name"],
                    "ann_id": ann["id"],
                    "cat_id": ann["category_id"],
                    "cat_name": self.cat_id_to_name[ann["category_id"]],
                })
        self.items = items
        print(f"[{split}] {len(items)} (image, category) pairs "
              f"from {len(set(i['image_id'] for i in items))} images")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        ann = self.coco.loadAnns([item["ann_id"]])[0]
        info = self.coco.loadImgs([item["image_id"]])[0]
        H, W = info["height"], info["width"]
        img_path = self.img_dir / item["file_name"]

        # Load image as PIL, keep numpy too
        pil = Image.open(img_path).convert("RGB")
        gt_mask = self.coco.annToMask(ann).astype(np.float32)  # (H, W)
        # Downsample GT to 14x14 via adaptive max-pool (any-overlap = positive)
        gt_t = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0)
        gt14 = F.adaptive_max_pool2d(gt_t, (GRID, GRID)).squeeze().numpy() > 0.5

        return {
            "image_pil": pil,
            "image_path": str(img_path),
            "query": item["cat_name"],
            "gt14": gt14.astype(np.float32),  # (14, 14) bool→float
            "image_id": item["image_id"],
            "cat_id": item["cat_id"],
        }


def coco_collate(batch):
    return {
        "image_pil": [b["image_pil"] for b in batch],
        "query": [b["query"] for b in batch],
        "gt14": torch.stack([torch.from_numpy(b["gt14"]) for b in batch]),  # (B, 14, 14)
        "image_id": [b["image_id"] for b in batch],
        "cat_id": [b["cat_id"] for b in batch],
    }


# ---- Model ----

class TextScorerHead(nn.Module):
    """Small head over frozen CLIP patch features.

    Input:  patch_feats (B, 196, 768) — CLIP ViT-B/16 visual patches (post visual.proj is OK,
            but we use pre-projection for richer information; head learns the rest).
            text_emb   (B, 512) — CLIP text embedding
    Output: scores (B, 196) logits
    """
    def __init__(self, patch_dim=768, text_dim=512, hidden_dim=384,
                 n_attn_heads=6, n_attn_layers=2, grid_size=GRID):
        super().__init__()
        self.grid_size = grid_size

        self.patch_proj = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, grid_size * grid_size, hidden_dim) * 0.02)

        self.self_attn_layers = nn.ModuleList()
        for _ in range(n_attn_layers):
            self.self_attn_layers.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(hidden_dim, n_attn_heads, batch_first=True),
                "norm1": nn.LayerNorm(hidden_dim),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                ),
                "norm2": nn.LayerNorm(hidden_dim),
            }))

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_attn_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)

        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, patch_feats, text_emb):
        B = patch_feats.shape[0]
        G = self.grid_size

        x = self.patch_proj(patch_feats)  # (B, 196, hidden)
        x = x + self.pos_embed
        for layer in self.self_attn_layers:
            r = x; x = layer["norm1"](x)
            x_a, _ = layer["attn"](x, x, x); x = r + x_a
            r = x; x = layer["norm2"](x); x = r + layer["ffn"](x)

        # Cross-attention: patches attend to text query
        q = self.text_proj(text_emb).unsqueeze(1)  # (B, 1, hidden)
        # Make patch positions the queries, text the key/value (so each patch is modulated by text)
        cross_out, _ = self.cross_attn(x, q, q)
        x = self.cross_norm(x + cross_out)

        scores = self.score_mlp(x).squeeze(-1)  # (B, 196)
        grids = scores.reshape(B, 1, G, G)
        refined = grids + self.spatial(grids)
        return refined.reshape(B, G * G)


# ---- Backbones (frozen) ----

@torch.no_grad()
def encode_clip_patches_and_text(clip_model, clip_tok,
                                 pil_images, queries, device,
                                 mean, std):
    """Batch through CLIP visual + text. Returns patch_feats (B, 196, 768), text_emb (B, 512)."""
    # Image tensor batch
    imgs = []
    for pil in pil_images:
        arr = np.array(pil)  # H,W,3
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        t = F.interpolate(t.unsqueeze(0), size=(224, 224), mode="bicubic", align_corners=False).squeeze(0)
        t = (t - mean[:, None, None]) / std[:, None, None]
        imgs.append(t)
    img_batch = torch.stack(imgs, dim=0).to(device)  # (B, 3, 224, 224)

    clip_model.visual.output_tokens = True
    pooled, patch_tokens = clip_model.visual(img_batch)  # patch_tokens: (B, 196, 768)
    clip_model.visual.output_tokens = False

    toks = clip_tok(queries).to(device)
    text_emb = clip_model.encode_text(toks)
    text_emb = F.normalize(text_emb, dim=-1)
    return patch_tokens, text_emb


@torch.no_grad()
def clipseg_target_14x14(clipseg_model, clipseg_proc, pil_images, queries, device):
    """Compute CLIPSeg sigmoid-output downsampled to 14x14 for each (image, query) in batch."""
    inputs = clipseg_proc(text=list(queries), images=list(pil_images), return_tensors="pt", padding=True).to(device)
    out = clipseg_model(**inputs)
    logits = out.logits  # (B, H, W)  e.g. (B, 352, 352)
    if logits.dim() == 2: logits = logits.unsqueeze(0)
    if logits.dim() == 3: logits = logits.unsqueeze(1)
    probs = torch.sigmoid(logits).float()  # (B, 1, H, W)
    return F.adaptive_avg_pool2d(probs, (GRID, GRID)).squeeze(1)  # (B, 14, 14)


def iou14(pred_logits, gt14_bool):
    """IoU between top-K-binarized prediction and GT mask at 14x14, where K = sum(GT). Per-batch list."""
    B = pred_logits.shape[0]
    pred = pred_logits.detach().reshape(B, GRID * GRID)
    gt = gt14_bool.reshape(B, GRID * GRID).bool()
    ious = []
    for i in range(B):
        K = int(gt[i].sum().item())
        if K == 0 or K >= GRID * GRID:
            ious.append(float("nan")); continue
        topk = torch.topk(pred[i], K).indices
        m = torch.zeros(GRID * GRID, dtype=torch.bool, device=pred.device); m[topk] = True
        inter = (m & gt[i]).sum().item()
        union = (m | gt[i]).sum().item()
        ious.append(inter / max(1, union))
    return ious


def main(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("[setup] CLIP ViT-B/16 (frozen) ...", flush=True)
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms(CLIP_NAME, pretrained=CLIP_PRETRAINED)
    clip_tok = open_clip.get_tokenizer(CLIP_NAME)
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters(): p.requires_grad_(False)
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)

    print("[setup] CLIPSeg (frozen, target source) ...", flush=True)
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    clipseg_proc = CLIPSegProcessor.from_pretrained(CLIPSEG_NAME)
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_NAME).to(device).eval()
    for p in clipseg_model.parameters(): p.requires_grad_(False)

    print("[setup] head + datasets ...", flush=True)
    head = TextScorerHead(patch_dim=768, text_dim=512, hidden_dim=384,
                          n_attn_heads=6, n_attn_layers=2, grid_size=GRID).to(device)
    print(f"  head params: {sum(p.numel() for p in head.parameters())/1e6:.2f}M")

    train_ds = CocoCatScorerDataset(args.coco_root, split="train")
    val_ds = CocoCatScorerDataset(args.coco_root, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, collate_fn=coco_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, collate_fn=coco_collate)

    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    best_val_iou = -1.0; best_path = out_dir / "best_v1.pt"
    metrics = []
    for ep in range(args.epochs):
        head.train()
        t0 = time.time()
        loss_sum = 0.0; n_batches = 0
        for batch in train_loader:
            with torch.no_grad():
                patches, text = encode_clip_patches_and_text(
                    clip_model, clip_tok, batch["image_pil"], batch["query"],
                    device, CLIP_MEAN, CLIP_STD)
                clipseg_t = clipseg_target_14x14(
                    clipseg_model, clipseg_proc, batch["image_pil"], batch["query"], device)
            gt = batch["gt14"].to(device).float()  # (B, 14, 14)

            pred = head(patches, text)  # (B, 196) logits
            pred_grid = pred.reshape(-1, GRID, GRID)
            l_bce = bce(pred_grid, gt)
            l_mse = mse(torch.sigmoid(pred_grid), clipseg_t)
            loss = args.alpha * l_bce + args.beta * l_mse

            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_sum += loss.item(); n_batches += 1

        # Eval
        head.eval()
        ious = []
        with torch.no_grad():
            for batch in val_loader:
                patches, text = encode_clip_patches_and_text(
                    clip_model, clip_tok, batch["image_pil"], batch["query"],
                    device, CLIP_MEAN, CLIP_STD)
                pred = head(patches, text)
                gt = batch["gt14"].to(device).bool()
                ious.extend(iou14(pred, gt))
        val_iou = float(np.nanmean(ious))
        sched.step()

        line = (f"[ep {ep+1}/{args.epochs}] train_loss={loss_sum/max(1,n_batches):.4f} "
                f"val_mIoU={val_iou:.3f}  wall={time.time()-t0:.1f}s")
        print(line, flush=True)
        metrics.append({"epoch": ep+1, "train_loss": loss_sum/max(1,n_batches), "val_mIoU": val_iou})

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({"head": head.state_dict(),
                        "args": vars(args), "val_iou": val_iou, "epoch": ep+1},
                       best_path)
            print(f"  saved best to {best_path}  val_mIoU={val_iou:.3f}", flush=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"best_val_iou": best_val_iou, "epochs": metrics}, f, indent=2)
    print(f"[done] best val mIoU = {best_val_iou:.3f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--coco_root", default="/home/ogata/semantic-autogaze/data/coco_val2017")
    p.add_argument("--output_dir", default="/home/ogata/semantic-autogaze/results/independent_scorer")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.5, help="BCE on COCO GT weight")
    p.add_argument("--beta", type=float, default=0.5, help="MSE on CLIPSeg distillation weight")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
