"""Cycle 1 (light-finetune) audit: render 20-pair side-by-side panels for a
specific gaze_decoder layer head vs IconStudent-118K.

Like bench_and_audit.py but loads from `best_layer{K}.pt` (the per-layer
ckpts from train_layer_ablation.py) and runs AutoGaze with
`output_hidden_states=True`, picking the K-th layer's features for the head.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from einops import rearrange
from pycocotools.coco import COCO

from semantic_autogaze.icon_student import IconStudent
from semantic_autogaze.inference import load_autogaze
from semantic_autogaze.model import SemanticAutoGaze
from semantic_autogaze.train_coco_seg import build_category_mask, get_image_categories


FILENAME_RE = re.compile(r"^\d{3}_(\d+)_(.+)\.png$")


class BilinearCosineHead(nn.Module):
    def __init__(self, patch_dim=192, query_dim=512, proj_dim=128):
        super().__init__()
        self.proj_p = nn.Linear(patch_dim, proj_dim, bias=False)
        self.proj_q = nn.Linear(query_dim, proj_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, patches, query):
        zp = F.normalize(self.proj_p(patches), dim=-1)
        zq = F.normalize(self.proj_q(query), dim=-1).unsqueeze(1)
        return (zp * zq).sum(-1) * self.scale + self.bias


def load_head(ckpt: str, device: torch.device):
    raw = torch.load(ckpt, map_location=device, weights_only=False)
    head = BilinearCosineHead(**raw["config"]).to(device)
    head.load_state_dict(raw["state_dict"])
    head.eval()
    layer_idx = raw.get("layer_idx", 3)
    return head, layer_idx


def load_iconstudent(ckpt: str, device):
    raw = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = raw["config"]
    model = IconStudent(
        patch_dim=cfg["patch_dim"], query_dim=cfg["query_dim"],
        decoder_dim=cfg["decoder_dim"],
        in_grid=cfg["in_grid"], out_grid=cfg["out_grid"],
        n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        dropout=0.0,
    ).to(device)
    sd = raw["state_dict"]
    pe = sd.get("pos_embed")
    if pe is not None and pe.dim() == 3:
        ig = cfg["in_grid"]; D = cfg["decoder_dim"]
        sd["pos_embed"] = pe.transpose(1, 2).reshape(1, D, ig, ig).contiguous()
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.inference_mode()
def autogaze_layer_features(model: SemanticAutoGaze, video, layer_idx: int):
    """Returns the K-th layer's hidden states (1, 196, 192) for layer_idx in 0..4."""
    B, T = video.shape[:2]
    gm = model.autogaze.gazing_model
    video_resized = rearrange(video, "b t c h w -> (b t) c h w")
    video_resized = F.interpolate(video_resized,
                                  size=(gm.input_img_size, gm.input_img_size),
                                  mode="bicubic", align_corners=False)
    video_resized = rearrange(video_resized, "(b t) c h w -> b t c h w", b=B)
    vision_features, _ = gm.vision_model(video_resized)
    vision_features = vision_features.transpose(1, 2)
    vision_features = rearrange(vision_features, "b t c h w -> b t (h w) c")
    vision_features = gm.connector(vision_features)
    B_, T_sub, N, C = vision_features.shape
    inputs_embeds = vision_features.reshape(B_, T_sub * N, C)
    attention_mask = torch.ones(B_, T_sub * N, device=video.device, dtype=torch.long)
    outputs = gm.gaze_decoder.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=attention_mask.cumsum(dim=-1) - 1,
        output_hidden_states=True,
    )
    # outputs.hidden_states is a tuple of 5: [input_embed, after_L0, after_L1, after_L2, after_L3].
    # Cache uses 0..3 indexing into [after_L0..after_L3], so we add 1 here.
    return outputs.hidden_states[layer_idx + 1]  # (1, 196, 192)


@torch.inference_mode()
def predict_iconstudent_native(model, native_obj, query, device, h, w):
    patches = native_obj["patches"].float()
    n_h, n_w = native_obj["grid"]
    logits = model(patches.unsqueeze(0).to(device), query.unsqueeze(0).to(device),
                   grid_hw=(n_h, n_w))
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    enc_h, enc_w = native_obj["encode_hw"]
    new_h, new_w = native_obj["resize_hw"]
    top_pad, left_pad = native_obj["crop_top_left"]
    sy = h / float(new_h); sx = w / float(new_w)
    og_top = max(0, int(round(top_pad * sy)))
    og_left = max(0, int(round(left_pad * sx)))
    og_bot = min(h, int(round((top_pad + enc_h) * sy)))
    og_right = min(w, int(round((left_pad + enc_w) * sx)))
    if og_bot - og_top <= 0 or og_right - og_left <= 0:
        return np.zeros((h, w), dtype=np.float32)
    region = F.interpolate(
        torch.from_numpy(probs).float()[None, None],
        size=(og_bot - og_top, og_right - og_left),
        mode="bilinear", align_corners=False)[0, 0].numpy()
    out = np.zeros((h, w), dtype=np.float32)
    out[og_top:og_bot, og_left:og_right] = np.clip(region, 0, 1)
    return out


@torch.inference_mode()
def predict_layer_head(image_pil, query, autogaze_w, head, layer_idx, device, h, w):
    t = TF.to_tensor(image_pil.resize((224, 224))).unsqueeze(0).unsqueeze(0).to(device)
    hidden = autogaze_layer_features(autogaze_w, t, layer_idx)  # (1, 196, 192)
    logits = head(hidden, query.unsqueeze(0).to(device))   # (1, 196)
    probs = torch.sigmoid(logits)[0].view(14, 14).cpu().numpy()
    sq = F.interpolate(
        torch.from_numpy(probs).float()[None, None], size=(224, 224),
        mode="bilinear", align_corners=False)[0, 0].numpy()
    out = np.array(Image.fromarray((sq * 255).astype(np.uint8)).resize((w, h),
                                                                       Image.BILINEAR)) / 255.0
    return out.astype(np.float32)


def parse_qual_dir(qual_dir: Path):
    pairs = []
    for p in sorted(qual_dir.iterdir()):
        m = FILENAME_RE.match(p.name)
        if m:
            pairs.append((int(m.group(1)), m.group(2)))
    return pairs


def overlay(img_np, gray, cmap):
    rgb = (cmap(gray)[..., :3] * 255).astype(np.uint8)
    return (0.5 * img_np.astype(np.float32) + 0.5 * rgb).clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head-ckpt", required=True,
                   help="Path to best_layer{K}.pt from train_layer_ablation.py")
    p.add_argument("--iconstudent-ckpt", default="results/icon_student_B_native_train/best.pt")
    p.add_argument("--qual-dir", default="results/compare_native_5k_vs_118k")
    p.add_argument("--dinov2-cache-native", default="results/dinov2_val_native")
    p.add_argument("--clip-text", default="results/icon_student_B_native_train/clip_text_embeddings.pt")
    p.add_argument("--data-dir", default="data/coco")
    p.add_argument("--ann", default="annotations/instances_val2017.json")
    p.add_argument("--img-subdir", default="val2017")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    coco = COCO(os.path.join(args.data_dir, args.ann))
    img_dir = os.path.join(args.data_dir, args.img_subdir)
    name_to_id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    clip_text = torch.load(args.clip_text, map_location="cpu", weights_only=True)

    print("[diag] loading head ckpt...")
    head, layer_idx = load_head(args.head_ckpt, device)
    print(f"[diag] layer_idx = {layer_idx}")

    print("[diag] loading AutoGaze...")
    ag = load_autogaze(device=device)
    w = SemanticAutoGaze(ag).to(device).eval()

    print("[diag] loading IconStudent-118K...")
    icon = load_iconstudent(args.iconstudent_ckpt, device)

    pairs = parse_qual_dir(Path(args.qual_dir))
    print(f"[diag] {len(pairs)} audit pairs")

    cmap = matplotlib.colormaps["jet"]
    summary = []
    for idx, (img_id, cat_slug) in enumerate(pairs):
        cat_name = cat_slug.replace("_", " ")
        if cat_name not in name_to_id:
            continue
        cat_id = name_to_id[cat_name]
        info = coco.imgs[img_id]
        img = Image.open(os.path.join(img_dir, info["file_name"])).convert("RGB")
        img_np = np.array(img); h, w_im = img_np.shape[:2]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=0))
        if not anns:
            continue
        gt = build_category_mask(coco, img_id, cat_id, anns).astype(np.float32)
        gt = np.clip(gt, 0, 1)
        query = clip_text[str(cat_id)].float()

        p_layer = predict_layer_head(img, query, w, head, layer_idx, device, h, w_im)
        nv = Path(args.dinov2_cache_native) / f"{img_id}.pt"
        if nv.exists():
            native_obj = torch.load(nv, map_location="cpu", weights_only=True)
            p_icon = predict_iconstudent_native(icon, native_obj, query, device, h, w_im)
        else:
            p_icon = np.zeros_like(p_layer)

        other = [coco.cats[c]["name"] for c in get_image_categories(coco, img_id).keys() if c != cat_id]
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        axes[0].imshow(img_np); axes[0].set_title(f"image {img_id}"); axes[0].axis("off")
        axes[1].imshow(overlay(img_np, gt, cmap)); axes[1].set_title(f"COCO GT \"{cat_name}\""); axes[1].axis("off")
        axes[2].imshow(overlay(img_np, p_layer, cmap)); axes[2].set_title(f"AutoGaze L{layer_idx}+head (max={p_layer.max():.2f})"); axes[2].axis("off")
        axes[3].imshow(overlay(img_np, p_icon, cmap)); axes[3].set_title(f"IconStudent-118K (max={p_icon.max():.2f})"); axes[3].axis("off")
        others = ", ".join(other[:8]) + ("…" if len(other) > 8 else "")
        plt.suptitle(f"query: \"{cat_name}\" | also present: {others}", fontsize=12)
        plt.tight_layout()
        out_path = out_dir / f"{idx:03d}_{img_id}_{cat_slug}.png"
        plt.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close()
        summary.append({"idx": idx, "img_id": img_id, "cat": cat_name,
                        "out": str(out_path),
                        "max_layer": float(p_layer.max()),
                        "max_118k": float(p_icon.max())})
        print(f"  {idx:03d}: {cat_name:14s} on {img_id} -> {out_path.name}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[diag] wrote {len(summary)} audit panels to {out_dir}")


if __name__ == "__main__":
    main()
