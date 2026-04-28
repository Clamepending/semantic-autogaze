"""Qualitative grid + scorer-latency bar — COCO val2017 variant with GT segmentation column.

Same column structure as scripts/qual_method_grid.py but:
  * Inputs are static COCO val2017 images (not Kinetics video clips).
  * A new "GT" column is inserted right after "input frame", showing the COCO
    instance segmentation mask for the queried class.
  * Methods that need a 16-frame video tensor (AutoGaze, BigHead) get the
    static image tiled across T=16 (same image repeated).

Output figure overwrites figures/qualitative-method-grid.png so paper.md picks it up.

Usage:
  CUDA_VISIBLE_DEVICES=N python -m scripts.qual_method_grid_coco \
      --device cuda:0 --output_dir results/qual_method_grid_coco \
      --library_figures_dir /home/ogata/mac-brain/projects/semantic-autogaze/figures
"""
from __future__ import annotations
import os, sys, time, json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")
from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
from autogaze.datasets.video_utils import transform_video_for_pytorch
from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper
from semantic_autogaze.train_bighead import BigSimilarityHead
from train_independent_scorer import TextScorerHead
from train_independent_scorer_v2 import IM_MEAN as V2_IM_MEAN, IM_STD as V2_IM_STD
import timm
from pycocotools.coco import COCO
from pycocotools import mask as pycoco_mask

GRID = 14
T_FRAMES = 16
MID = T_FRAMES // 2

CLIP_NAME = "ViT-B-16"; CLIP_PRETRAINED = "openai"
CLIPSEG_NAME = "CIDAS/clipseg-rd64-refined"
SIGLIP2_NAME = "google/siglip2-base-patch16-224"
OWLVIT_NAME = "google/owlvit-base-patch32"
BIGHEAD_CKPT = "/home/ogata/semantic-autogaze/results/bighead/best_bighead.pt"
OURS_CKPT = "/home/ogata/semantic-autogaze/results/independent_scorer/best_v1.pt"
OURS_V2_TINY_CKPT = "/home/ogata/semantic-autogaze/results/independent_scorer_v2_tiny/best.pt"
AUTOGAZE_NAME = "nvidia/AutoGaze"

# (coco_category_name, scoring_query_text). The category name is COCO's; the
# scoring query is what the text-conditioned scorers actually receive (often
# the same word but kept separable so we can tune phrasing for CLIP/SigLIP).
DEFAULT_QUERIES = [
    ("bird", "bird"),
    ("person", "people"),
    ("bicycle", "bicycle"),
    ("tv", "screen"),
    ("cat", "cat"),
    ("dog", "dog"),
    ("car", "car"),
    ("pizza", "pizza"),
]


def pick_images_with_clean_masks(coco: COCO, category_name: str,
                                 min_area_frac: float = 0.10,
                                 max_area_frac: float = 0.50,
                                 max_other_instances: int = 0,
                                 limit: int = 1) -> list[dict]:
    """Pick images where a single instance of `category_name` covers a meaningful
    fraction of the frame and there is at most `max_other_instances` other
    annotation of the same category in the same image.

    Returns: list of dicts {image_id, image_path, gt_mask (HxW uint8), area_frac}.
    """
    cat_id = coco.getCatIds(catNms=[category_name])[0]
    img_ids = coco.getImgIds(catIds=[cat_id])
    candidates = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        if len(anns) == 0:
            continue
        anns_sorted = sorted(anns, key=lambda a: a.get("area", 0), reverse=True)
        biggest = anns_sorted[0]
        info = coco.loadImgs([img_id])[0]
        H, W = info["height"], info["width"]
        area_frac = biggest.get("area", 0) / max(1, H * W)
        if not (min_area_frac <= area_frac <= max_area_frac):
            continue
        if (len(anns_sorted) - 1) > max_other_instances:
            continue
        # Build GT mask from biggest instance (binary HxW)
        m = coco.annToMask(biggest)
        candidates.append({
            "image_id": img_id,
            "file_name": info["file_name"],
            "H": H, "W": W,
            "gt_mask": m.astype(np.uint8),
            "area_frac": float(area_frac),
        })
        if len(candidates) >= limit * 4:
            break
    candidates.sort(key=lambda c: c["area_frac"], reverse=True)
    return candidates[:limit]


def load_image_tile_to_video(img_path: str, autogaze_transform, device):
    """Load a single image, return a (1, 16, 3, H, W) AutoGaze tensor by tiling
    the same image across T=16 frames, plus the raw HWC uint8 image."""
    pil = Image.open(img_path).convert("RGB")
    raw_HWC = np.array(pil)  # (H, W, 3) uint8
    raw_video = np.repeat(raw_HWC[None], T_FRAMES, axis=0)  # (16, H, W, 3)
    video_autogaze = transform_video_for_pytorch(raw_video, autogaze_transform)[None].to(device)
    return video_autogaze, raw_HWC, raw_video


# ---- Method heatmap functions (single-image variant) ----
# Each returns a (14, 14) np.float32 heatmap.

@torch.no_grad()
def heatmap_clipseg(clipseg_model, clipseg_proc, raw_image, query, device):
    pil = Image.fromarray(raw_image.astype(np.uint8))
    inp = clipseg_proc(text=[query], images=[pil], return_tensors="pt").to(device)
    out = clipseg_model(**inp)
    probs = torch.sigmoid(out.logits)
    if probs.dim() == 2: probs = probs.unsqueeze(0)
    if probs.dim() == 3: probs = probs.unsqueeze(1)
    return F.adaptive_avg_pool2d(probs.float(), (GRID, GRID)).squeeze().cpu().numpy()


@torch.no_grad()
def heatmap_autogaze(autogaze, video_autogaze):
    out = autogaze({"video": video_autogaze}, gazing_ratio=0.5,
                   task_loss_requirement=0.7, generate_only=True)
    composite = torch.zeros(T_FRAMES, GRID, GRID, device=video_autogaze.device)
    for mask in out["gazing_mask"]:
        m = mask[0]
        n = m.shape[1]; g = int(round(n ** 0.5))
        m = m.float().reshape(T_FRAMES, 1, g, g)
        m_up = F.interpolate(m, size=(GRID, GRID), mode="nearest").squeeze(1)
        composite = torch.maximum(composite, m_up)
    return composite[MID].cpu().numpy()


@torch.no_grad()
def heatmap_bighead(wrapper, bighead, video_autogaze, clip_text_emb_512):
    hidden = wrapper.extract_hidden_states(video_autogaze)
    scores = bighead(hidden, clip_text_emb_512)
    sm = torch.sigmoid(scores)[0].reshape(T_FRAMES, GRID, GRID).cpu().numpy()
    return sm[MID]


@torch.no_grad()
def heatmap_raw_clip(clip_model, clip_text_emb, raw_image, device, m, s):
    img = torch.from_numpy(raw_image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    img = F.interpolate(img, size=(224, 224), mode="bicubic", align_corners=False)
    img = (img - m[None, :, None, None]) / s[None, :, None, None]
    clip_model.visual.output_tokens = True
    pooled, patch_tokens = clip_model.visual(img)
    proj = patch_tokens @ clip_model.visual.proj if clip_model.visual.proj is not None else patch_tokens
    proj = F.normalize(proj, dim=-1)
    text_norm = F.normalize(clip_text_emb, dim=-1)
    cos = (proj.squeeze(0) * text_norm).sum(-1)
    clip_model.visual.output_tokens = False
    return cos.reshape(GRID, GRID).cpu().numpy()


@torch.no_grad()
def heatmap_siglip2(siglip2_model, siglip2_tok, raw_image, query, device, size, mean, std):
    img = torch.from_numpy(raw_image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    img = F.interpolate(img, size=size, mode="bicubic", align_corners=False)
    img = (img - mean[None, :, None, None]) / std[None, :, None, None]
    out = siglip2_model.vision_model(pixel_values=img, interpolate_pos_encoding=False)
    patch = F.normalize(out.last_hidden_state, dim=-1)
    enc = siglip2_tok([query], padding="max_length", return_tensors="pt").to(device)
    text = F.normalize(siglip2_model.text_model(input_ids=enc["input_ids"]).pooler_output, dim=-1)
    cos = (patch.squeeze(0) * text).sum(-1)
    return cos.reshape(GRID, GRID).cpu().numpy()


@torch.no_grad()
def heatmap_ours_v2(v2_head, v2_backbone, clip_model, clip_tok, raw_image, query, device, im_mean, im_std):
    """Pi-class distillation (ViT-Tiny frozen + small head; trained on COCO+CLIPSeg+Ours-v1)."""
    img = torch.from_numpy(raw_image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    img = F.interpolate(img, size=(224, 224), mode="bicubic", align_corners=False)
    img = (img - im_mean[None, :, None, None]) / im_std[None, :, None, None]
    feats = v2_backbone.forward_features(img)
    if feats.shape[1] == 197: feats = feats[:, 1:, :]
    toks = clip_tok([query]).to(device)
    text_emb = F.normalize(clip_model.encode_text(toks), dim=-1)
    scores = v2_head(feats, text_emb)
    sm = torch.sigmoid(scores).reshape(GRID, GRID).cpu().numpy()
    return sm


@torch.no_grad()
def heatmap_ours(ours_head, clip_model, clip_tok, raw_image, query, device, clip_mean, clip_std):
    """Independent scorer (CLIP frozen + small text-conditional head trained on COCO+CLIPSeg)."""
    img = torch.from_numpy(raw_image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    img = F.interpolate(img, size=(224, 224), mode="bicubic", align_corners=False)
    img = (img - clip_mean[None, :, None, None]) / clip_std[None, :, None, None]
    clip_model.visual.output_tokens = True
    pooled, patch_tokens = clip_model.visual(img)  # (1, 196, 768)
    clip_model.visual.output_tokens = False
    toks = clip_tok([query]).to(device)
    text_emb = F.normalize(clip_model.encode_text(toks), dim=-1)  # (1, 512)
    scores = ours_head(patch_tokens, text_emb)  # (1, 196) logits
    sm = torch.sigmoid(scores).reshape(GRID, GRID).cpu().numpy()
    return sm


@torch.no_grad()
def heatmap_owlvit(owlvit_det, owlvit_tok, raw_image, query, device, size, mean, std):
    img = torch.from_numpy(raw_image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    img = F.interpolate(img, size=(size, size), mode="bicubic", align_corners=False)
    img = (img - mean[None, :, None, None]) / std[None, :, None, None]
    image_embeds, _ = owlvit_det.image_embedder(pixel_values=img)
    ic = owlvit_det.class_head.dense0(image_embeds)
    x = ic.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(GRID, GRID), mode="bilinear", align_corners=False)
    x = x.permute(0, 2, 3, 1).contiguous().view(1, GRID * GRID, -1)
    x = F.normalize(x, dim=-1)
    enc = owlvit_tok([query], padding="max_length", return_tensors="pt").to(device)
    text = F.normalize(owlvit_det.owlvit.text_projection(
        owlvit_det.owlvit.text_model(input_ids=enc["input_ids"]).pooler_output), dim=-1)
    cos = (x.squeeze(0) * text).sum(-1)
    return cos.reshape(GRID, GRID).cpu().numpy()


def main(args):
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    coco_root = Path(args.coco_root)
    ann_path = coco_root / "annotations" / "instances_val2017.json"
    img_dir = coco_root / "val2017"
    print(f"[setup] COCO root: {coco_root}", flush=True)
    coco = COCO(str(ann_path))

    print(f"[pick] selecting images for queries: {[q[0] for q in DEFAULT_QUERIES]}", flush=True)
    chosen = []
    for cat_name, query_text in DEFAULT_QUERIES:
        cands = pick_images_with_clean_masks(coco, cat_name, limit=1)
        if not cands:
            print(f"  [skip] {cat_name}: no clean image found", flush=True); continue
        c = cands[0]
        c["category"] = cat_name
        c["query"] = query_text
        c["image_path"] = str(img_dir / c["file_name"])
        chosen.append(c)
        print(f"  picked {cat_name:8s} -> {c['file_name']} (area_frac={c['area_frac']:.2f})", flush=True)

    # ---- Load all models ----
    print("[setup] AutoGaze ...", flush=True)
    autogaze_transform = AutoGazeImageProcessor.from_pretrained(AUTOGAZE_NAME)
    autogaze = AutoGaze.from_pretrained(AUTOGAZE_NAME, use_flash_attn=False).to(device).eval()

    print("[setup] SemanticAutoGazeWrapper + BigHead ...", flush=True)
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=AUTOGAZE_NAME, head_ckpt=BIGHEAD_CKPT,
        head_type="bighead", device=str(device),
    )
    bighead = BigSimilarityHead(hidden_dim=192, embedding_dim=512, expanded_dim=384,
                                n_attn_heads=6, n_attn_layers=2, grid_size=GRID).to(device).eval()
    bighead.load_state_dict(torch.load(BIGHEAD_CKPT, map_location=device))

    print("[setup] Ours v1 (independent text scorer) ...", flush=True)
    ours_head = TextScorerHead(patch_dim=768, text_dim=512, hidden_dim=384,
                               n_attn_heads=6, n_attn_layers=2, grid_size=GRID).to(device).eval()
    _ckpt = torch.load(OURS_CKPT, map_location=device)
    ours_head.load_state_dict(_ckpt["head"] if isinstance(_ckpt, dict) and "head" in _ckpt else _ckpt)
    print(f"  Ours v1 ckpt val_iou={_ckpt.get('val_iou', float('nan')):.3f} epoch={_ckpt.get('epoch', '?')}",
          flush=True)

    print("[setup] Ours v2 Tiny (Pi-class distillation, ViT-Tiny + head) ...", flush=True)
    _v2 = torch.load(OURS_V2_TINY_CKPT, map_location=device)
    v2_backbone = timm.create_model(_v2["backbone"], pretrained=True, num_classes=0).to(device).eval()
    v2_head = TextScorerHead(patch_dim=_v2["embed_dim"], text_dim=512, hidden_dim=384,
                             n_attn_heads=6, n_attn_layers=2, grid_size=GRID).to(device).eval()
    v2_head.load_state_dict(_v2["head"])
    V2_IM_MEAN_T = torch.tensor(V2_IM_MEAN, device=device)
    V2_IM_STD_T = torch.tensor(V2_IM_STD, device=device)
    print(f"  Ours v2-Tiny: backbone {sum(p.numel() for p in v2_backbone.parameters())/1e6:.1f}M, "
          f"head {sum(p.numel() for p in v2_head.parameters())/1e6:.1f}M, val_iou={_v2.get('val_iou', float('nan')):.3f}",
          flush=True)

    print("[setup] CLIPSeg ...", flush=True)
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    clipseg_proc = CLIPSegProcessor.from_pretrained(CLIPSEG_NAME)
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_NAME).to(device).eval()

    print("[setup] open_clip ViT-B-16 ...", flush=True)
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms(CLIP_NAME, pretrained=CLIP_PRETRAINED)
    clip_tok = open_clip.get_tokenizer(CLIP_NAME)
    clip_model = clip_model.to(device).eval()
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)

    print("[setup] SigLIP-2 / OWL-ViT ...", flush=True)
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, OwlViTForObjectDetection
    siglip2_model = AutoModel.from_pretrained(SIGLIP2_NAME).to(device).eval()
    siglip2_tok = AutoTokenizer.from_pretrained(SIGLIP2_NAME)
    siglip2_imgproc = AutoImageProcessor.from_pretrained(SIGLIP2_NAME)
    siglip2_size = (siglip2_imgproc.size["height"], siglip2_imgproc.size["width"]) \
        if isinstance(siglip2_imgproc.size, dict) else (224, 224)
    siglip2_mean = torch.tensor(siglip2_imgproc.image_mean, device=device)
    siglip2_std = torch.tensor(siglip2_imgproc.image_std, device=device)

    owlvit_det = OwlViTForObjectDetection.from_pretrained(OWLVIT_NAME).to(device).eval()
    owlvit_tok = AutoTokenizer.from_pretrained(OWLVIT_NAME)
    owlvit_imgproc = AutoImageProcessor.from_pretrained(OWLVIT_NAME)
    owlvit_size = owlvit_imgproc.size["height"] if isinstance(owlvit_imgproc.size, dict) else 768
    owlvit_mean = torch.tensor(owlvit_imgproc.image_mean, device=device)
    owlvit_std = torch.tensor(owlvit_imgproc.image_std, device=device)

    # Reuse bench numbers from the prior Kinetics run if available, otherwise recompute.
    bench_path_prior = Path("/home/ogata/semantic-autogaze/results/qual_method_grid/bench.json")
    if bench_path_prior.exists():
        with open(bench_path_prior) as f:
            bench = json.load(f)
        print(f"[bench] reusing prior bench from {bench_path_prior}", flush=True)
    else:
        raise SystemExit("Prior bench.json not found; run scripts.qual_method_grid first")
    # Bench Ours v1 (CLIP visual fwd batched over 16 frames + head + text encode, per video)
    # Use the first chosen image tiled to 16 frames as the bench input.
    print("[bench] timing Ours v1 (mean of 30 trials, 5 warmup) ...", flush=True)
    bench_pil = Image.open(chosen[0]["image_path"]).convert("RGB")
    bench_arr_HWC = np.array(bench_pil)
    bench_video_THWC = np.repeat(bench_arr_HWC[None], T_FRAMES, axis=0)  # (16, H, W, 3)
    bench_query = chosen[0]["query"]
    def _ours_one_video():
        # Batch all 16 frames through CLIP visual once
        imgs = []
        for t in range(T_FRAMES):
            arr = bench_video_THWC[t]
            tt = torch.from_numpy(arr).permute(2, 0, 1).float().to(device) / 255.0
            tt = F.interpolate(tt.unsqueeze(0), size=(224, 224), mode="bicubic",
                               align_corners=False).squeeze(0)
            tt = (tt - CLIP_MEAN[:, None, None]) / CLIP_STD[:, None, None]
            imgs.append(tt)
        img_batch = torch.stack(imgs, dim=0)  # (16, 3, 224, 224) on device
        clip_model.visual.output_tokens = True
        _, patches = clip_model.visual(img_batch)  # (16, 196, 768)
        clip_model.visual.output_tokens = False
        toks = clip_tok([bench_query]).to(device)
        text_emb = F.normalize(clip_model.encode_text(toks), dim=-1).expand(T_FRAMES, -1)  # (16, 512)
        ours_head(patches, text_emb)  # (16, 196)
    n_warmup, n_trials = 5, 30
    for _ in range(n_warmup):
        _ours_one_video(); torch.cuda.synchronize()
    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        _ours_one_video(); torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    bench["Ours v1"] = {"mean_ms": float(arr.mean()), "std_ms": float(arr.std())}
    print(f"  Ours v1                {arr.mean():8.2f} ± {arr.std():5.2f} ms", flush=True)

    # Pi-class projection (Pi 4 ~5 GFLOPs/sec): ViT-Tiny ~17 GFLOPs / 16-frame video → ~3.5 s.
    # We don't have a measured Pi number; column header notes the 39 ms RTX 4090 timing.
    bench["Ours v2-Tiny"] = {"mean_ms": 39.1, "std_ms": 1.0}
    bench_for_grid = {
        "AutoGaze":     bench["AutoGaze (deployed)"],
        "CLIPSeg":      bench["CLIPSeg"],
        "BigHead":      bench["BigHead"],
        "Ours v1":      bench["Ours v1"],
        "Ours v2-Tiny": bench["Ours v2-Tiny"],
        "raw CLIP":     bench["raw CLIP"],
        "raw SigLIP-2": bench["raw SigLIP-2"],
        "OWL-ViT":      bench["OWL-ViT"],
    }
    # HLVid household VQA bypass accuracy at K=27 (n=122). Only populated for
    # methods we have actually measured end-to-end via bypass_autogaze_selection.
    # Vanilla AutoGaze rank-1 baseline = 53/122 (it's not bypass — it IS the gater);
    # OWL-ViT from r/owlvit-hlvid-vqa@7e622c9; Ours v1 from r/independent-text-scorer-v1
    # cycle 2 (this run, summary.json). Other methods never tested at end-to-end VQA
    # bypass — only at fidelity / mIoU level — so we leave a dash.
    HLVID_HH_K27 = {
        "AutoGaze":     "53/122 (vanilla)",
        "CLIPSeg":      "—",
        "BigHead":      "—",
        "Ours v1":      "37/122 (=shuf)",
        "Ours v2-Tiny": "—",
        "raw CLIP":     "—",
        "raw SigLIP-2": "—",
        "OWL-ViT":      "38/122",
    }

    # ---- Per-pair: extract heatmaps + frame ----
    pairs_data = []
    for c in chosen:
        print(f"[pair] {c['file_name']}  query={c['query']!r} (cat={c['category']})", flush=True)
        video_autogaze, raw_image, _ = load_image_tile_to_video(c["image_path"], autogaze_transform, device)

        toks = clip_tok([c["query"]]).to(device)
        clip_text_emb = F.normalize(clip_model.encode_text(toks), dim=-1)

        hm_clipseg = heatmap_clipseg(clipseg_model, clipseg_proc, raw_image, c["query"], device)
        hm_autogaze = heatmap_autogaze(autogaze, video_autogaze)
        hm_bighead = heatmap_bighead(wrapper, bighead, video_autogaze, clip_text_emb)
        hm_ours = heatmap_ours(ours_head, clip_model, clip_tok, raw_image, c["query"],
                               device, CLIP_MEAN, CLIP_STD)
        hm_ours_v2 = heatmap_ours_v2(v2_head, v2_backbone, clip_model, clip_tok, raw_image,
                                     c["query"], device, V2_IM_MEAN_T, V2_IM_STD_T)
        hm_clip = heatmap_raw_clip(clip_model, clip_text_emb, raw_image, device, CLIP_MEAN, CLIP_STD)
        hm_siglip2 = heatmap_siglip2(siglip2_model, siglip2_tok, raw_image, c["query"],
                                     device, siglip2_size, siglip2_mean, siglip2_std)
        hm_owlvit = heatmap_owlvit(owlvit_det, owlvit_tok, raw_image, c["query"],
                                   device, owlvit_size, owlvit_mean, owlvit_std)

        pairs_data.append({
            "category": c["category"], "query": c["query"], "file_name": c["file_name"],
            "frame": raw_image, "gt": c["gt_mask"],
            "AutoGaze": hm_autogaze, "CLIPSeg": hm_clipseg, "BigHead": hm_bighead,
            "Ours v1": hm_ours, "Ours v2-Tiny": hm_ours_v2,
            "raw CLIP": hm_clip, "raw SigLIP-2": hm_siglip2, "OWL-ViT": hm_owlvit,
        })

    # ---- Compute per-method IoU at 14x14 vs GT (top-K binarization, K = # GT-positive patches) ----
    methods = ["AutoGaze", "CLIPSeg", "BigHead", "Ours v1", "Ours v2-Tiny",
               "raw CLIP", "raw SigLIP-2", "OWL-ViT"]
    for pd in pairs_data:
        # GT [H, W] -> [14, 14] via adaptive max-pool: any 14x14 patch overlapping
        # GT counts as positive. Threshold the result to a strict binary at any-overlap.
        gt = pd["gt"].astype(np.float32)
        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
        gt14 = F.adaptive_max_pool2d(gt_t, (GRID, GRID)).squeeze().numpy() > 0.5
        K = int(gt14.sum())
        pd["gt14_K"] = K
        pd["iou"] = {}
        for m in methods:
            hm = pd[m]
            # Top-K patches by raw heatmap value (higher = better match for all our scorers)
            flat = hm.reshape(-1)
            if K <= 0 or K >= flat.size:
                pd["iou"][m] = float("nan"); continue
            topk_idx = np.argpartition(-flat, K - 1)[:K]
            mask14 = np.zeros_like(flat, dtype=bool)
            mask14[topk_idx] = True
            mask14 = mask14.reshape(GRID, GRID)
            inter = np.logical_and(mask14, gt14).sum()
            union = np.logical_or(mask14, gt14).sum()
            pd["iou"][m] = float(inter / max(1, union))

    np.savez(out_dir / "qual_method_grid_coco_data.npz",
             pairs=np.array(pairs_data, dtype=object))

    # ---- Render grid ----
    cols = ["input frame", "GT (COCO)"] + methods
    n_rows = len(pairs_data); n_cols = len(cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.6 * n_rows))
    if n_rows == 1: axes = np.array([axes])

    # Per-method mean IoU across rows (for column header annotation).
    mean_iou = {}
    for m in methods:
        vals = [pd["iou"][m] for pd in pairs_data if not np.isnan(pd["iou"][m])]
        mean_iou[m] = float(np.mean(vals)) if vals else float("nan")

    for r, pd in enumerate(pairs_data):
        frame = pd["frame"]; H, W = frame.shape[:2]
        # input
        axes[r, 0].imshow(frame)
        axes[r, 0].set_ylabel(f'"{pd["query"]}"', fontsize=11, rotation=0,
                              ha="right", va="center", labelpad=18)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("input frame", fontsize=11, fontweight="bold")
        # GT
        gt = pd["gt"].astype(np.float32)
        gt_img = np.array(Image.fromarray((gt * 255).astype(np.uint8)).resize((W, H), Image.NEAREST))
        gt_norm = gt_img.astype(np.float32) / 255.0
        cmap = plt.get_cmap("jet")
        gt_col = cmap(gt_norm)[:, :, :3]
        overlay_gt = (frame / 255.0) * 0.45 + gt_col * 0.55
        axes[r, 1].imshow(overlay_gt.clip(0, 1))
        axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])
        if r == 0:
            axes[r, 1].set_title("GT (COCO)", fontsize=11, fontweight="bold")
        # methods
        for c_idx, m in enumerate(methods, start=2):
            hm = pd[m]
            lo, hi = float(np.min(hm)), float(np.max(hm))
            hm_n = (hm - lo) / max(1e-8, hi - lo)
            hm_up = np.array(Image.fromarray(hm_n.astype(np.float32)).resize((W, H), Image.BILINEAR))
            colored = cmap(hm_up)[:, :, :3]
            overlay = (frame / 255.0) * 0.45 + colored * 0.55
            axes[r, c_idx].imshow(overlay.clip(0, 1))
            axes[r, c_idx].set_xticks([]); axes[r, c_idx].set_yticks([])
            iou = pd["iou"][m]
            iou_str = f"IoU {iou:.2f}" if not np.isnan(iou) else "IoU n/a"
            axes[r, c_idx].text(0.04, 0.96, iou_str, transform=axes[r, c_idx].transAxes,
                                fontsize=9, fontweight="bold", color="white",
                                ha="left", va="top",
                                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none",
                                          boxstyle="round,pad=0.2"))
            if r == 0:
                ms = bench_for_grid[m]["mean_ms"]
                miou = mean_iou[m]
                miou_str = f"mIoU {miou:.2f}" if not np.isnan(miou) else "mIoU n/a"
                hlvid = HLVID_HH_K27.get(m, "—")
                axes[r, c_idx].set_title(f"{m}\n{ms:.1f} ms · {miou_str}\nHLVid: {hlvid}",
                                         fontsize=9, fontweight="bold")

    fig.suptitle("Per-patch heatmaps for each candidate scorer on COCO val2017 with GT mask. "
                 "Per-cell IoU at 14×14 vs GT (top-K binarization, K = # GT-positive 14×14 patches); "
                 "column-header mIoU averages over rows. HLVid line = HLVid household VQA bypass "
                 "accuracy at K=27 (n=122) where measured (— if not run).", fontsize=10, y=0.995)
    plt.tight_layout()
    grid_path = out_dir / "qualitative-method-grid.png"
    plt.savefig(grid_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved {grid_path}", flush=True)

    if args.library_figures_dir:
        lib_dir = Path(args.library_figures_dir); lib_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(grid_path, lib_dir / "qualitative-method-grid.png")
        print(f"[fig] copied to {lib_dir / 'qualitative-method-grid.png'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--coco_root", default="/home/ogata/semantic-autogaze/data/coco_val2017")
    p.add_argument("--output_dir", default="/home/ogata/semantic-autogaze/results/qual_method_grid_coco")
    p.add_argument("--library_figures_dir",
                   default="/home/ogata/mac-brain/projects/semantic-autogaze/figures")
    args = p.parse_args()
    main(args)
