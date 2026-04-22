"""Cycle 2 of r/debug-score-distribution.

Visualize BigHead score maps on HLVid clips as a sanity check that cycle 1's
numbers (per-frame p99 up to 0.35) correspond to the head actually attending
to the right regions. For each clip, render a 2xT grid where top row is
the frame thumbnails and bottom row is the BigHead sigmoid heatmap (14x14,
bilinearly upsampled to 224).

Output: one PNG per (clip, preproc_mode) under results/debug_score_dist/.
"""
import os, argparse
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
import matplotlib.pyplot as plt

from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper
from semantic_autogaze.dump_bighead_scores import (
    HLVID_CLIPS, read_video_frames_fast, load_frames_and_preprocess,
    IMAGENET_STD_MEAN, IMAGENET_STD_STD,
)


@torch.no_grad()
def render_clip(wrapper, clip_model, clip_tokenizer, clip, mode, device,
                num_frames, size, out_dir):
    if not os.path.exists(clip["path"]):
        return None
    frames = read_video_frames_fast(clip["path"], num_frames=num_frames, size=size)
    if frames is None:
        return None

    q = clip["query_hint"]
    tok = clip_tokenizer([q]).to(device)
    emb = clip_model.encode_text(tok)
    emb = F.normalize(emb, dim=-1)  # (1, 512)

    # Preprocess per mode
    vid = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    if mode == "eval":
        vid = (vid - IMAGENET_STD_MEAN) / IMAGENET_STD_STD
    vid = vid.unsqueeze(0).to(device)  # (1, T, C, H, W)

    hidden = wrapper.extract_hidden_states(vid)
    scores = wrapper.semantic_filter.get_scores(hidden, emb)  # (1, T*196)
    T = num_frames
    G = 14
    per_frame = scores.reshape(1, T, G, G).squeeze(0).cpu().numpy()  # (T, 14, 14)

    # Render 2 x T grid
    fig, axes = plt.subplots(2, T, figsize=(2 * T, 4.2))
    for t in range(T):
        axes[0, t].imshow(frames[t])
        axes[0, t].set_title(f"t={t}", fontsize=7)
        axes[0, t].axis("off")

        hm = per_frame[t]
        axes[1, t].imshow(frames[t], alpha=0.5)
        axes[1, t].imshow(
            np.kron(hm, np.ones((size // G, size // G))),
            alpha=0.55, cmap="jet", vmin=0.0,
            vmax=max(0.05, float(per_frame.max())),
        )
        axes[1, t].set_title(f"p99={np.quantile(hm, 0.99):.2f} max={hm.max():.2f}", fontsize=7)
        axes[1, t].axis("off")

    fig.suptitle(
        f"clip={clip['tag']}  query={q!r}  preproc={mode}  "
        f"(sig_overall max={per_frame.max():.3f}, p99={np.quantile(per_frame, 0.99):.3f})",
        fontsize=10,
    )
    fig.tight_layout()
    out = os.path.join(out_dir, f"heatmap_{clip['tag']}_{mode}.png")
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/bighead_warmrestart/best_bighead_student.pt")
    ap.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--output_dir", default="results/debug_score_dist")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"[setup] loading wrapper on {device}")
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type="bighead",
        device=str(device),
    )
    print("[setup] loading CLIP")
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()

    for clip in HLVID_CLIPS:
        for mode in ("train", "eval"):
            out = render_clip(
                wrapper, clip_model, clip_tokenizer, clip, mode, device,
                args.num_frames, args.size, args.output_dir,
            )
            print(f"  {'saved' if out else 'SKIP'}: {out or clip['path']}")


if __name__ == "__main__":
    main()
