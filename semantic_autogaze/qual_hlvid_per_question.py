"""
Per-question qualitative analysis on HLVid shard-1 subset.

Goal: understand *why* the semantic filter flipped q82/q85 from right→wrong
(AutoGaze got them, filter didn't) while flipping q81 wrong→right.

For each HLVid QA pair we render a 4×4 grid:
  row 0: raw frame
  row 1: AutoGaze-only kept patches (green)
  row 2: AG ∩ sem-30%, query = full question stem (blue)
  row 3: AG ∩ sem-30%, query = noun-phrase extraction (red)

If (row 3) keeps the tile containing the asked-about sign more often than
(row 2), the eval regression is a query-engineering problem, not a filter
quality problem. If neither keeps the text tile, the filter itself is the
bottleneck for OCR-style questions.

Usage:
    CUDA_VISIBLE_DEVICES=N python3 -m semantic_autogaze.qual_hlvid_per_question \\
        --device cuda:0 --output_dir results/qual_hlvid_per_q
"""

import os
import re
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open_clip
import av
import pandas as pd

from autogaze.models.autogaze import AutoGazeImageProcessor
from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper


T_FRAMES = 16
GRID = 14
N_PATCHES = GRID * GRID
TOTAL = T_FRAMES * N_PATCHES

PARQUET = "/home/ogata/.cache/huggingface/hub/datasets--bfshi--HLVid/snapshots/4956b601aec0bb9d455bb8f57552f78cbd3f0338/data/test-00000-of-00001.parquet"


def extract_noun_phrase(stem: str) -> str:
    """Strip HLVid's question scaffolding to recover the subject NP.

    Heuristic: prefer the first `on the X` match (HLVid almost always asks
    about text `on [some sign/surface]`). Fallback: drop the wh-prefix.
    """
    m = re.search(r"\bon\s+(the\s+[^?,.]+)", stem, flags=re.I)
    if m:
        # Trim trailing modifiers like "on the right", "in the center"
        np_ = m.group(1).strip()
        # Drop trailing locative clause
        np_ = re.sub(r"\s+on\s+the\s+(right|left|top|bottom|center|middle)$", "", np_, flags=re.I)
        return np_.strip(" ?.,")
    # Fallback: strip wh-prefix up to first copula
    m2 = re.sub(r"^what\s+\S+\s+(is|are|does|do|say)\s+", "", stem, flags=re.I)
    return m2.strip(" ?.,")


def clip_text_embed(text, model, tok, device):
    tokens = tok([text]).to(device)
    with torch.no_grad():
        f = model.encode_text(tokens)
        f = F.normalize(f, dim=-1)
    return f


def info_to_per_frame_mask(info):
    gp = info["gazing_pos"]
    pd_ = info["if_padded_gazing"]
    valid = (~pd_[0]).cpu()
    pos = gp[0].cpu()[valid].clamp(0, TOTAL - 1).numpy().astype(np.int64)
    flat = np.zeros(TOTAL, dtype=bool)
    flat[pos] = True
    return flat.reshape(T_FRAMES, GRID, GRID)


def pick_4_frames(mask_per_frame):
    counts = mask_per_frame.sum(axis=(1, 2))
    idx = []
    for lo, hi in [(0, 4), (4, 8), (8, 12), (12, 16)]:
        sub = counts[lo:hi]
        idx.append(int(lo + int(np.argmax(sub))))
    return idx


def overlay_mask(ax, frame_img, mask_2d, color_rgb, alpha=0.5):
    ax.imshow(frame_img)
    H, W = frame_img.shape[:2]
    ph, pw = H // GRID, W // GRID
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    for r in range(GRID):
        for c in range(GRID):
            if mask_2d[r, c]:
                overlay[r*ph:(r+1)*ph, c*pw:(c+1)*pw, :3] = color_rgb
                overlay[r*ph:(r+1)*ph, c*pw:(c+1)*pw, 3] = alpha
    ax.imshow(overlay)
    for i in range(1, GRID):
        ax.axhline(i * ph, color="white", lw=0.2, alpha=0.4)
        ax.axvline(i * pw, color="white", lw=0.2, alpha=0.4)
    ax.set_xticks([]); ax.set_yticks([])


def load_hlvid_subset(video_dir):
    df = pd.read_parquet(PARQUET)
    samples = []
    for _, r in df.iterrows():
        full = os.path.join(video_dir, r["video_path"])
        if not os.path.exists(full):
            continue
        q = r["question"]
        m = re.split(r"\n(?=A\.)", q, maxsplit=1)
        stem = m[0].strip() if len(m) == 2 else q
        samples.append({
            "qid": int(r["question_id"]),
            "video": full,
            "stem": stem,
            "answer": r["answer"],
        })
    return samples


def main(args):
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    os.makedirs(args.output_dir, exist_ok=True)

    print("[setup] Loading wrapper + CLIP ...")
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )
    ag_tf = AutoGazeImageProcessor.from_pretrained(args.autogaze_model)
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()

    from transformers import AutoImageProcessor
    siglip_tf = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")

    samples = load_hlvid_subset(args.video_dir)
    print(f"[data] {len(samples)} HLVid QA samples")

    report = []
    for s in samples:
        qid = s["qid"]
        stem = s["stem"]
        np_ = extract_noun_phrase(stem)
        print(f"\nq{qid} ({s['answer']}) {os.path.basename(s['video'])}")
        print(f"   stem:      {stem}")
        print(f"   extracted: '{np_}'")

        try:
            c = av.open(s["video"])
            # Uniform sample across the entire clip (matches NVILA's sampling).
            n_total = c.streams.video[0].frames
            indices = np.linspace(0, n_total - 1, T_FRAMES).astype(int).tolist()
            raw = read_video_pyav(container=c, indices=indices)
            c.close()
            if raw.shape[0] < T_FRAMES:
                print("   skip: short clip"); continue
            v_ag = transform_video_for_pytorch(raw, ag_tf)[None].to(device)
            v_sg = transform_video_for_pytorch(raw, siglip_tf)[None].to(device)
        except Exception as e:
            print(f"   skip: {e}"); continue

        q_full = clip_text_embed(stem, clip_model, clip_tok, device)
        q_np = clip_text_embed(np_, clip_model, clip_tok, device)

        info_g = wrapper.forward(v_ag, q_full, mode="gaze_only",
                                 gazing_ratio=args.gazing_ratio,
                                 task_loss_requirement=args.task_loss_req,
                                 semantic_keep_ratio=1.0)
        info_full = wrapper.forward(v_ag, q_full, mode="intersect",
                                    gazing_ratio=args.gazing_ratio,
                                    task_loss_requirement=args.task_loss_req,
                                    semantic_keep_ratio=args.keep)
        info_np = wrapper.forward(v_ag, q_np, mode="intersect",
                                  gazing_ratio=args.gazing_ratio,
                                  task_loss_requirement=args.task_loss_req,
                                  semantic_keep_ratio=args.keep)

        m_g = info_to_per_frame_mask(info_g)
        m_full = info_to_per_frame_mask(info_full)
        m_np = info_to_per_frame_mask(info_np)

        # Per-frame Jaccard between the two query strategies (on AG-kept patches)
        inter = (m_full & m_np).sum()
        union = (m_full | m_np).sum()
        jaccard = inter / max(union, 1)
        # How many patches differ?
        only_full = int((m_full & ~m_np).sum())
        only_np = int((m_np & ~m_full).sum())

        fidx = pick_4_frames(m_g)
        frames_np = v_sg[0].cpu().numpy().transpose(0, 2, 3, 1)
        frames_np = (frames_np - frames_np.min()) / (frames_np.max() - frames_np.min() + 1e-9)

        fig, axes = plt.subplots(4, 4, figsize=(13, 13))
        for col, t in enumerate(fidx):
            frame = frames_np[t]
            axes[0, col].imshow(frame)
            axes[0, col].set_title(f"frame {t}", fontsize=10)
            axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
            if col == 0: axes[0, col].set_ylabel("raw", fontsize=11)

            overlay_mask(axes[1, col], frame, m_g[t], (0.2, 0.9, 0.2), 0.45)
            axes[1, col].set_title(f"AG: {int(m_g[t].sum())}/196", fontsize=9)
            if col == 0: axes[1, col].set_ylabel("AutoGaze", fontsize=11)

            overlay_mask(axes[2, col], frame, m_full[t], (0.15, 0.5, 1.0), 0.55)
            axes[2, col].set_title(f"stem-q: {int(m_full[t].sum())}", fontsize=9)
            if col == 0: axes[2, col].set_ylabel("∩ full stem", fontsize=11)

            overlay_mask(axes[3, col], frame, m_np[t], (1.0, 0.3, 0.3), 0.55)
            axes[3, col].set_title(f"NP-q: {int(m_np[t].sum())}", fontsize=9)
            if col == 0: axes[3, col].set_ylabel("∩ noun-phrase", fontsize=11)

        fig.suptitle(
            f"q{qid} ans={s['answer']}   stem: {stem[:95]}\n"
            f"NP: '{np_}'   |   Jaccard(stem,NP)={jaccard:.2f}   "
            f"stem-only={only_full}   NP-only={only_np}",
            fontsize=10,
        )
        fig.tight_layout()
        out = os.path.join(args.output_dir, f"q{qid:03d}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"   AG={int(m_g.sum())}  stem={int(m_full.sum())}  NP={int(m_np.sum())}  "
              f"Jaccard={jaccard:.2f}  -> {out}")

        report.append({
            "qid": qid, "answer": s["answer"], "stem": stem, "np": np_,
            "n_ag": int(m_g.sum()), "n_stem": int(m_full.sum()), "n_np": int(m_np.sum()),
            "jaccard_stem_np": float(jaccard),
            "stem_only_patches": only_full, "np_only_patches": only_np,
            "png": out,
        })

    with open(os.path.join(args.output_dir, "per_question.json"), "w") as f:
        json.dump({"rows": report, "keep_ratio": args.keep}, f, indent=2)
    print(f"\n[save] {args.output_dir}/per_question.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", default="hlvid_videos/extracted/videos")
    p.add_argument("--keep", type=float, default=0.3)
    p.add_argument("--gazing_ratio", type=float, default=0.75)
    p.add_argument("--task_loss_req", type=float, default=0.7)
    p.add_argument("--ckpt", default="results/bighead_warmrestart/best_bighead_student.pt")
    p.add_argument("--head_type", default="bighead")
    p.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/qual_hlvid_per_q")
    args = p.parse_args()
    main(args)
