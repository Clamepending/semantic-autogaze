"""Quick eval of Ours v2 (ViT-Tiny + head) on the 8 qual-grid images + latency bench."""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image

sys.path.insert(0, "/home/ogata/semantic-autogaze")
sys.path.insert(0, "/home/ogata/semantic-autogaze/scripts")
from train_independent_scorer import TextScorerHead, GRID
from train_independent_scorer_v2 import IM_MEAN, IM_STD, BACKBONES

import pycocotools.coco as cc
COCO_ROOT = "/home/ogata/semantic-autogaze/data/coco_val2017"
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


def main():
    import sys as _sys
    device = torch.device("cuda:0")
    ckpt_path = _sys.argv[1] if len(_sys.argv) > 1 else \
        "/home/ogata/semantic-autogaze/results/independent_scorer_v2_tiny/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone_name = ckpt["backbone"]; embed_dim = ckpt["embed_dim"]
    print(f"loading v2 backbone: {backbone_name}, embed_dim={embed_dim}")
    bb = timm.create_model(backbone_name, pretrained=True, num_classes=0).to(device).eval()
    head = TextScorerHead(patch_dim=embed_dim, text_dim=512, hidden_dim=384,
                          n_attn_heads=6, n_attn_layers=2, grid_size=GRID).to(device).eval()
    head.load_state_dict(ckpt["head"])

    print("loading CLIP text encoder ...")
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()

    IM_MEAN_T = torch.tensor(IM_MEAN, device=device)
    IM_STD_T = torch.tensor(IM_STD, device=device)

    coco = cc.COCO(os.path.join(COCO_ROOT, "annotations", "instances_val2017.json"))

    # Per-row IoU
    print("\nPer-row IoU on 8 qual-grid images:")
    ious = {}
    for cat, img_id, query in QUAL_PAIRS:
        cat_id = coco.getCatIds(catNms=[cat])[0]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=False)
        anns = sorted(coco.loadAnns(ann_ids), key=lambda a: -a.get("area", 0))
        ann = anns[0]
        gt_mask = coco.annToMask(ann).astype(np.float32)
        gt14 = F.adaptive_max_pool2d(torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0),
                                      (GRID, GRID)).squeeze().numpy() > 0.5

        info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(COCO_ROOT, "val2017", info["file_name"])
        pil = Image.open(img_path).convert("RGB")

        with torch.no_grad():
            arr = np.array(pil)
            t = torch.from_numpy(arr).permute(2, 0, 1).float().to(device) / 255.0
            t = F.interpolate(t.unsqueeze(0), size=(224, 224), mode="bicubic",
                              align_corners=False).squeeze(0)
            t = (t - IM_MEAN_T[:, None, None]) / IM_STD_T[:, None, None]
            feats = bb.forward_features(t.unsqueeze(0))
            if feats.shape[1] == 197: feats = feats[:, 1:, :]
            toks = clip_tok([query]).to(device)
            text_emb = F.normalize(clip_model.encode_text(toks), dim=-1)
            scores = head(feats, text_emb)
            sm = torch.sigmoid(scores).reshape(GRID, GRID).cpu().numpy()
        # Top-K IoU
        K = int(gt14.sum())
        topk = np.argpartition(-sm.flatten(), K - 1)[:K]
        m = np.zeros(GRID * GRID, bool); m[topk] = True
        m = m.reshape(GRID, GRID)
        inter = np.logical_and(m, gt14).sum()
        union = np.logical_or(m, gt14).sum()
        iou = float(inter / max(1, union))
        ious[cat] = iou
        print(f"  {cat:8s}  K={K:3d}  IoU={iou:.2f}")
    miou = np.mean(list(ious.values()))
    print(f"\n[v2 ViT-Tiny] mIoU on 8 qual-grid pairs = {miou:.3f}")

    # Latency: 16-frame video as repeated image
    arr = np.array(Image.open(os.path.join(COCO_ROOT, "val2017",
                                            coco.loadImgs([QUAL_PAIRS[0][1]])[0]["file_name"])).convert("RGB"))
    video_THWC = np.repeat(arr[None], 16, axis=0)
    bench_query = "bird"
    def f_v2():
        imgs = []
        for t in range(16):
            tt = torch.from_numpy(video_THWC[t]).permute(2, 0, 1).float().to(device) / 255.0
            tt = F.interpolate(tt.unsqueeze(0), size=(224, 224), mode="bicubic", align_corners=False).squeeze(0)
            tt = (tt - IM_MEAN_T[:, None, None]) / IM_STD_T[:, None, None]
            imgs.append(tt)
        img_b = torch.stack(imgs, dim=0)
        feats = bb.forward_features(img_b)
        if feats.shape[1] == 197: feats = feats[:, 1:, :]
        toks = clip_tok([bench_query]).to(device)
        text_emb = F.normalize(clip_model.encode_text(toks), dim=-1).expand(16, -1)
        head(feats, text_emb)

    print("\n[bench] timing v2 (5 warmup + 30 trials, 16-frame video)")
    for _ in range(5): f_v2(); torch.cuda.synchronize()
    times = []
    for _ in range(30):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        f_v2(); torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    print(f"  v2 latency: {times.mean():.2f} +/- {times.std():.2f} ms / 16-frame video")
    print(f"  v2 backbone: {sum(p.numel() for p in bb.parameters())/1e6:.2f}M frozen params")
    print(f"  v2 head:     {sum(p.numel() for p in head.parameters())/1e6:.2f}M trainable params")

    out = {"mIoU": miou, "per_cat_iou": ious,
           "latency_ms_mean": float(times.mean()), "latency_ms_std": float(times.std()),
           "backbone": backbone_name,
           "backbone_params_M": sum(p.numel() for p in bb.parameters())/1e6,
           "head_params_M": sum(p.numel() for p in head.parameters())/1e6}
    eval_path = os.path.join(os.path.dirname(ckpt_path), "eval.json")
    with open(eval_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {eval_path}")


if __name__ == "__main__":
    main()
