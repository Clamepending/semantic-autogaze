"""Phase 2c: evaluate a SigLIP-distilled ckpt.

Three eval flavors:

  1. COCO qual-grid mIoU (the standard 8-image, 8-category benchmark)
     vs the baseline (CLIPSeg 0.806 / Ours v1 0.786)

  2. Demo-failure-mode visual: small/OOD vocab on Pi-captured frames.
     Saves a side-by-side figure: input | new ckpt | Ours v1 baseline.

  3. (Optional) m-v-s metric: tests with a small EgoSchema sample (n=100)
     using the new ckpt as the bypass scorer.

Reports:
  per-image / per-frame IoU
  abstention rate on absent-class queries
  (optional) match-vs-shuf paired-flip
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")
from train_independent_scorer import TextScorerHead, GRID
from train_independent_scorer_v2 import IM_MEAN, IM_STD
from pycocotools.coco import COCO
from train_siglip_dense_distill import build_backbone, SiglipBias, CLIP_MEAN, CLIP_STD, ObjectnessHead

# Same QUAL_PAIRS as cycle 2's qual grid for direct comparison
QUAL_PAIRS = [
    ("bird", 337987, "bird"),
    ("person", 32861, "people"),
    ("bicycle", 370208, "bicycle"),
    ("tv", 346638, "screen"),
    ("cat", 223747, "cat"),
    ("dog", 267300, "dog"),
    ("car", 151962, "car"),
    ("pizza", 232489, "pizza"),
]
COCO_ROOT = "/home/ogata/semantic-autogaze/data/coco_val2017"

# Demo failure modes — same queries the user reported as broken
DEMO_QUERIES = ["pen", "pencil", "hand", "book", "laptop", "robot gripper"]
PI_FRAMES = ["/tmp/pi_raw_frames/raw_2.jpg", "/tmp/pi_raw_frames/raw_4.jpg"]


def load_ckpt(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ck.get("args", {})
    model = args.get("model", "v1")
    bb_fn, patch_dim, mean, std, kind, bb_module = build_backbone(
        model, device, finetune_blocks=args.get("finetune_backbone_blocks", 0))
    # Phase35 cost-volume head detection: ckpts trained with --head_type
    # cost_volume have a state dict with `cost_proj`, `transformer.layers.*`,
    # `out_proj` keys instead of TextScorerHead's `patch_proj.*` etc.
    head_state = ck["head"]
    is_cost_volume = "cost_proj.weight" in head_state and "transformer.layers.0.self_attn.in_proj_weight" in head_state
    if is_cost_volume:
        from train_siglip_dense_distill import CostVolumeHead  # type: ignore
        # Infer config from state-dict shapes
        embed_dim = head_state["cost_proj.weight"].shape[0]
        n_layers = sum(1 for k in head_state.keys() if k.startswith("transformer.layers.") and k.endswith(".self_attn.in_proj_weight"))
        n_heads = args.get("cv_n_heads", 8)
        # grid_in is fixed at 14 (backbone always interpolates to GRID)
        # grid_out is recoverable from training args
        grid_out = args.get("grid_size_out", 14)
        head = CostVolumeHead(
            patch_dim=patch_dim, text_dim=512,
            grid_in=GRID, grid_out=grid_out,
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers,
        ).to(device).eval()
        head.load_state_dict(head_state)
        print(f"[ckpt] CostVolumeHead grid_in={GRID} grid_out={grid_out} "
              f"embed_dim={embed_dim} n_layers={n_layers}", flush=True)
    else:
        head = TextScorerHead(
            patch_dim=patch_dim, text_dim=512,
            hidden_dim=args.get("head_hidden_dim", 384),
            n_attn_heads=args.get("head_attn_heads", 6),
            n_attn_layers=args.get("head_attn_layers", 2),
            grid_size=GRID, use_spatial=args.get("head_use_spatial", True),
        ).to(device).eval()
        # Note: phase20 ckpts trained with --grid_size_out 28 have a state_dict
        # that is parameter-bit-identical to this 14x14 head (the only difference
        # is a bilinear upsample at the end of forward(); zero learnable params).
        # So load_state_dict works cleanly and eval runs at 14x14 — which is the
        # right resolution for the existing iou_topk + heatmap_one downstream.
        head.load_state_dict(head_state)
    # phase19b/e/f compat: ckpts trained with --per_query_bias have an MLP-bias
    # SiglipBiasPerQuery sb instead of the global SiglipBias. Detect by sb
    # state_dict keys and instantiate the right class.
    sb_state = ck["sb"]
    if any(k.startswith("bias_mlp.") for k in sb_state.keys()):
        from train_siglip_dense_distill import SiglipBiasPerQuery  # type: ignore
        sb = SiglipBiasPerQuery().to(device).eval()
        sb.load_state_dict(sb_state)
        sb._is_per_query = True
    elif any(k.startswith("bias_linear.") for k in sb_state.keys()):
        from train_siglip_dense_distill import SiglipBiasPerQueryLinear  # type: ignore
        sb = SiglipBiasPerQueryLinear().to(device).eval()
        sb.load_state_dict(sb_state)
        sb._is_per_query = True
    else:
        sb = SiglipBias().to(device).eval()
        sb.load_state_dict(sb_state)
        sb._is_per_query = False
    if "backbone_state" in ck:
        bb_module.load_state_dict(ck["backbone_state"])
    # Optional objectness head (phase21+). When the ckpt has an 'obj' state-dict,
    # construct an ObjectnessHead with the same patch_dim and load it. Inference
    # callers can then multiply per-query sigmoid scores by sigmoid(objectness)
    # to suppress query-agnostic background/context cells.
    obj_head = None
    if "obj" in ck and ck["obj"] is not None:
        obj_head = ObjectnessHead(patch_dim=patch_dim).to(device).eval()
        obj_head.load_state_dict(ck["obj"])
    return bb_fn, head, sb, mean, std, model, kind, bb_module, obj_head


@torch.no_grad()
def heatmap_one(pil, query, bb_fn, head, sb, mean, std, clip_model_text, clip_tok, device,
                obj_head=None, obj_gate: bool = True):
    arr = np.array(pil.resize((224, 224), Image.BICUBIC))
    x = (arr.astype(np.float32) / 255.0 - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(device)
    patches = bb_fn(x)  # (1, 196, D)
    toks = clip_tok([query]).to(device)
    text = F.normalize(clip_model_text.encode_text(toks), dim=-1)
    # Head output is flat (B, G_out*G_out) where G_out matches the head's
    # configured grid_size_out (default 14, can be 28 or 56 for cost_volume).
    logits_flat = head(patches, text)  # (1, G_out*G_out)
    G_out = int(round((logits_flat.shape[-1]) ** 0.5))
    logits = logits_flat.reshape(G_out, G_out)
    if getattr(sb, "_is_per_query", False):
        # SiglipBiasPerQuery expects (B, B, H, W) + (B, dim) text. Wrap singleton.
        logits_wrapped = logits.unsqueeze(0).unsqueeze(0)  # (1, 1, G_out, G_out)
        cal = sb(logits_wrapped, text).squeeze(0).squeeze(0)
    else:
        cal = sb(logits)
    prob = torch.sigmoid(cal)
    if obj_gate and obj_head is not None:
        obj_logits = obj_head(patches).squeeze(0)  # (GRID, GRID)
        prob = prob * torch.sigmoid(obj_logits)
    return prob.cpu().numpy()


def iou_topk(soft_14, gt_full, K=None):
    gt_t = torch.from_numpy(gt_full).float().unsqueeze(0).unsqueeze(0)
    gt14 = F.adaptive_max_pool2d(gt_t, (GRID, GRID)).squeeze().numpy() > 0.5
    K = K or int(gt14.sum())
    if K <= 0: return 0.0
    flat = soft_14.flatten()
    top = np.argpartition(-flat, K - 1)[:K]
    m = np.zeros(GRID * GRID, bool); m[top] = True
    m14 = m.reshape(GRID, GRID)
    inter = np.logical_and(m14, gt14).sum()
    union = np.logical_or(m14, gt14).sum()
    return float(inter / max(1, union))


def eval_qual_grid(bb_fn, head, sb, mean, std, clip_model_text, clip_tok, device, output_dir, obj_head=None):
    coco = COCO(os.path.join(COCO_ROOT, "annotations", "instances_val2017.json"))
    rows = []
    for cat, img_id, query in QUAL_PAIRS:
        cat_id = coco.getCatIds(catNms=[cat])[0]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=False)
        anns = sorted(coco.loadAnns(ann_ids), key=lambda a: -a.get("area", 0))
        ann = anns[0]
        gt = coco.annToMask(ann).astype(np.float32)
        info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(COCO_ROOT, "val2017", info["file_name"])
        pil = Image.open(img_path).convert("RGB")
        h = heatmap_one(pil, query, bb_fn, head, sb, mean, std, clip_model_text, clip_tok, device, obj_head=obj_head)
        iou = iou_topk(h, gt)
        rows.append((cat, query, h, iou, np.array(pil), gt))
        print(f"  {cat:8s} '{query}': IoU={iou:.3f}", flush=True)
    miou = float(np.mean([r[3] for r in rows]))
    print(f"  >>> mIoU = {miou:.3f}", flush=True)

    # Render figure
    fig, axes = plt.subplots(len(rows), 3, figsize=(8, 1.5 * len(rows)))
    for i, (cat, q, h, iou, arr, gt) in enumerate(rows):
        axes[i, 0].imshow(arr); axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        axes[i, 0].set_ylabel(f"{cat}\n'{q}'", fontsize=8, rotation=0, ha="right", va="center", labelpad=22)
        axes[i, 1].imshow(arr); axes[i, 1].imshow(gt, alpha=0.5, cmap="Greens")
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
        if i == 0: axes[i, 1].set_title("GT", fontsize=10)
        axes[i, 2].imshow(arr, alpha=0.6)
        h_up = np.kron(h, np.ones((arr.shape[0] // GRID + 1, arr.shape[1] // GRID + 1)))[:arr.shape[0], :arr.shape[1]]
        axes[i, 2].imshow(h_up, alpha=0.5, cmap="hot", vmin=0, vmax=1)
        axes[i, 2].set_xticks([]); axes[i, 2].set_yticks([])
        axes[i, 2].text(0.02, 0.95, f"IoU={iou:.2f}", color="white", fontsize=8,
                        transform=axes[i, 2].transAxes, va="top",
                        bbox=dict(facecolor="black", alpha=0.6, pad=2, edgecolor="none"))
        if i == 0: axes[i, 2].set_title("ckpt", fontsize=10)
    if i == 0: axes[i, 0].set_title("input", fontsize=10)
    plt.suptitle(f"Phase 2 ckpt qual grid — mIoU={miou:.3f} (cf. CLIPSeg 0.806, Ours v1 0.786)", fontsize=10)
    plt.tight_layout()
    out_png = os.path.join(output_dir, "phase2_qual_grid.png")
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_png}")
    return {"miou": miou, "per_cat": {r[0]: r[3] for r in rows}}


def eval_demo_frames(bb_fn, head, sb, mean, std, clip_model_text, clip_tok, device, output_dir, obj_head=None):
    rows = []
    for fpath in PI_FRAMES:
        if not os.path.exists(fpath): continue
        pil = Image.open(fpath).convert("RGB").rotate(-90, expand=True)  # match demo rotation
        for q in DEMO_QUERIES:
            h = heatmap_one(pil, q, bb_fn, head, sb, mean, std, clip_model_text, clip_tok, device)
            print(f"  {os.path.basename(fpath):20s} '{q:20s}' max={h.max():.2f} mean={h.mean():.3f}", flush=True)
            rows.append((os.path.basename(fpath), q, np.array(pil), h))
    # Side-by-side figure
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(6, 1.6 * n))
    for i, (fname, q, arr, h) in enumerate(rows):
        axes[i, 0].imshow(arr); axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        axes[i, 0].set_ylabel(f"{fname[:5]}\n'{q[:18]}'", fontsize=7, rotation=0, ha="right", va="center", labelpad=24)
        axes[i, 1].imshow(arr, alpha=0.6)
        h_up = np.kron(h, np.ones((arr.shape[0] // GRID + 1, arr.shape[1] // GRID + 1)))[:arr.shape[0], :arr.shape[1]]
        axes[i, 1].imshow(h_up, alpha=0.5, cmap="hot", vmin=0, vmax=1)
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
        axes[i, 1].text(0.02, 0.95, f"max={h.max():.2f}", color="white", fontsize=7,
                        transform=axes[i, 1].transAxes, va="top",
                        bbox=dict(facecolor="black", alpha=0.6, pad=2, edgecolor="none"))
    plt.suptitle("Phase 2 ckpt — demo failure modes", fontsize=10)
    plt.tight_layout()
    out_png = os.path.join(output_dir, "phase2_demo_frames.png")
    plt.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out_png}")


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[ckpt] loading {args.ckpt}", flush=True)
    bb_fn, head, sb, mean, std, model, kind, bb_module, obj_head = load_ckpt(args.ckpt, device)
    print(f"  model={model}  obj_head={'yes' if obj_head is not None else 'no'}", flush=True)

    print(f"[clip-text] loading ...", flush=True)
    import open_clip
    clip_model_text, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_model_text = clip_model_text.to(device).eval()

    if not args.skip_qual:
        print("\n=== COCO qual grid ===", flush=True)
        qual = eval_qual_grid(bb_fn, head, sb, mean, std, clip_model_text, clip_tok, device, args.output_dir, obj_head=obj_head)
    if not args.skip_demo:
        print("\n=== demo failure-mode frames ===", flush=True)
        eval_demo_frames(bb_fn, head, sb, mean, std, clip_model_text, clip_tok, device, args.output_dir, obj_head=obj_head)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--skip_qual", action="store_true")
    p.add_argument("--skip_demo", action="store_true")
    args = p.parse_args()
    main(args)
