"""Qualitative open-vocab validation for the user's `data/eval_openvocab_streets/`.

Renders a per-image grid of heatmaps for a curated keyword set covering:
  - in-distribution things  (car, person, traffic light, tree, sign)
  - in-distribution stuff   (road, sidewalk, building, sky)
  - OOD compositional       (jeep, crosswalk, bike rack, manhole)
  - color + object          (red sign, black car, white stripes)
  - clearly absent          (elephant, pizza) — abstention test

NEVER train on these images. They are pure validation signal.

Output: a single PNG per (ckpt × image) under
  /home/ogata/mac-brain/projects/semantic-autogaze/figures/openvocab_eval/<slug>/<image>.png

The panel is laid out: input frame as background, per-query heatmap
overlay tiled in a grid, with `max=` and `mean=` per query so
abstention-vs-firing can be read at a glance.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image as _PIL

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")
from scripts.eval_phase2_ckpt import load_ckpt, heatmap_one, GRID
import torch.nn.functional as F

# Test-time synonym expansion (Wysoczanska et al., NeurIPS 2024,
# arxiv 2407.05061 "Test-time contrastive concepts"). For each query at
# inference we encode bare + K synonym variants, average the (normalized)
# text embeddings, and pass the averaged emb to the head once. Cost: K
# extra text-encoder forwards per keyword change (one-time per session
# in the demo loop) -- ~0 ms per frame on Pi 5.
SYNONYM_TABLE = {
    "car":          ["car", "vehicle", "automobile", "passenger car"],
    "person":       ["person", "human", "people", "pedestrian"],
    "traffic light":["traffic light", "stoplight", "traffic signal", "signal light"],
    "tree":         ["tree", "tall tree", "leafy tree", "tree trunk"],
    "sign":         ["sign", "billboard", "store sign", "street sign"],
    "road":         ["road", "asphalt", "street", "paved road"],
    "sidewalk":     ["sidewalk", "pavement", "walkway", "footpath"],
    "building":     ["building", "structure", "office building", "storefront"],
    "sky":          ["sky", "night sky", "dark sky", "evening sky"],
    "jeep":         ["jeep", "Jeep Wrangler", "off-road vehicle", "SUV"],
    "crosswalk":    ["crosswalk", "zebra crossing", "pedestrian crossing", "marked crosswalk"],
    "bike rack":    ["bike rack", "bicycle rack", "bike stand", "bicycle parking"],
    "manhole":      ["manhole", "manhole cover", "round metal cover on road", "drain cover"],
    "red sign":     ["red sign", "red logo", "red shop sign", "red signage"],
    "black car":    ["black car", "black vehicle", "dark car", "black SUV"],
    "elephant":     ["elephant", "elephant animal"],
}


@torch.no_grad()
def heatmap_one_tta(pil, query, bb_fn, head, sb, mean, std, clip_model_text, clip_tok, device,
                    obj_head=None, tta_synonyms=False):
    """Variant of heatmap_one that averages text embeddings across synonyms
    when tta_synonyms is True. Falls back to bare-query when synonym list
    is missing for a query. Same forward signature as heatmap_one."""
    if not tta_synonyms or query not in SYNONYM_TABLE:
        return heatmap_one(pil, query, bb_fn, head, sb, mean, std,
                           clip_model_text, clip_tok, device, obj_head=obj_head)

    arr = np.array(pil.resize((224, 224), _PIL.BICUBIC))
    x = (arr.astype(np.float32) / 255.0 - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(device)
    patches = bb_fn(x)

    syns = SYNONYM_TABLE[query]
    toks = clip_tok(syns).to(device)
    text = F.normalize(clip_model_text.encode_text(toks), dim=-1)  # (K, 512)
    text_avg = F.normalize(text.mean(dim=0, keepdim=True), dim=-1)  # (1, 512)

    logits = head(patches, text_avg).reshape(GRID, GRID)
    if getattr(sb, "_is_per_query", False):
        logits_w = logits.unsqueeze(0).unsqueeze(0)
        cal = sb(logits_w, text_avg).squeeze(0).squeeze(0)
    else:
        cal = sb(logits)
    prob = torch.sigmoid(cal)
    if obj_head is not None:
        obj_logits = obj_head(patches).squeeze(0)
        prob = prob * torch.sigmoid(obj_logits)
    return prob.cpu().numpy()

# Curated keyword set. Order matters — laid out left-to-right, top-to-bottom
# in the panel for easy visual diffing across ckpts.
KEYWORDS = [
    # In-distribution things
    "car", "person", "traffic light", "tree", "sign",
    # In-distribution stuff
    "road", "sidewalk", "building", "sky",
    # OOD compositional / color
    "jeep", "crosswalk", "bike rack", "manhole",
    "red sign", "black car",
    # Abstention test
    "elephant",
]


def render_panel(pil, image_label, heats, save_path):
    """heats: list of (query, heat_14x14, hmax, hmean)."""
    n = len(heats)
    cols = 4
    rows = (n + cols - 1) // cols
    # +1 row at top for the input image alone
    fig = plt.figure(figsize=(cols * 3.0, (rows + 1) * 3.0))
    gs = fig.add_gridspec(rows + 1, cols)

    ax_in = fig.add_subplot(gs[0, :])
    ax_in.imshow(pil)
    ax_in.set_title(f"input: {image_label}", fontsize=10)
    ax_in.set_xticks([]); ax_in.set_yticks([])

    arr = np.array(pil)
    H, W = arr.shape[:2]
    for i, (q, heat, hmax, hmean) in enumerate(heats):
        r = 1 + (i // cols); c = i % cols
        ax = fig.add_subplot(gs[r, c])
        # Upsample heatmap to image resolution via nearest (preserve cells).
        heat_up = np.kron(heat, np.ones((H // GRID + 1, W // GRID + 1)))[:H, :W]
        ax.imshow(arr, alpha=0.55)
        ax.imshow(heat_up, alpha=0.55, cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"'{q}'  max={hmax:.2f} mean={hmean:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(args):
    device = torch.device(args.device)
    bb_fn, head, sb, mean, std, model, kind, bb_module, obj_head = load_ckpt(args.ckpt, device)
    if obj_head is not None and args.no_objectness:
        obj_head = None
    print(f"[ckpt] loaded {args.ckpt}  obj_gate={'on' if obj_head is not None else 'off'}", flush=True)

    import open_clip
    text_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    text_model = text_model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-B-16")

    img_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))

    aggregate = []  # for the optional summary csv
    for img_p in image_paths:
        pil = _PIL.open(img_p).convert("RGB")
        heats = []
        for q in KEYWORDS:
            heat = heatmap_one_tta(pil, q, bb_fn, head, sb, mean, std, text_model, tok, device,
                                   obj_head=obj_head, tta_synonyms=args.tta_synonyms)
            hmax = float(heat.max()); hmean = float(heat.mean())
            heats.append((q, heat, hmax, hmean))
            aggregate.append((img_p.stem, q, hmax, hmean))
            print(f"  [{img_p.stem:35s}] '{q:14s}' max={hmax:.3f} mean={hmean:.3f}", flush=True)
        save_path = out_dir / f"{img_p.stem}.png"
        render_panel(pil, img_p.name, heats, save_path)
        print(f"  [saved] {save_path}", flush=True)

    # Aggregate CSV
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("image,query,hmax,hmean\n")
        for img, q, hmax, hmean in aggregate:
            f.write(f"{img},{q},{hmax:.4f},{hmean:.4f}\n")
    print(f"[saved] {csv_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--image_dir", default="/home/ogata/semantic-autogaze/data/eval_openvocab_streets")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--no_objectness", action="store_true",
                   help="Disable obj-gate even if the ckpt has one. Used to A/B "
                        "compare gated vs ungated on the same ckpt.")
    p.add_argument("--tta_synonyms", action="store_true",
                   help="Enable test-time synonym expansion: average text-embs "
                        "across SYNONYM_TABLE[query] before running the head. "
                        "Fixes compositional-adjective failure mode (red sign / "
                        "night sky) at zero per-frame Pi cost. NeurIPS 2024 "
                        "Test-time Contrastive Concepts (arxiv 2407.05061).")
    main(p.parse_args())
