"""r/nvila-attention-distill Phase 2 step 1 — cache AutoGaze hidden states + gazing_info.

For each (qid, video) on HLVid household at the SMALLER config (matching cycle 1
extraction: max_tiles=1, num_frames=16, gazing_ratio=0.50), cache:
- hidden_states: (T*196, hidden_dim=192) — AutoGaze SigLIP-grid features (BigHead input)
- gazing_pos: (K_kept,) — patch positions kept by AutoGaze on the (T*196) flat grid
- num_gazing_each_frame: (T,) — per-frame kept counts

These will be used to build a NVILAAttentionDataset for Phase 2 distillation training.

Run:
  CUDA_VISIBLE_DEVICES=N python -m semantic_autogaze.extract_autogaze_for_distill \\
    --device cuda:0 --output_dir results/autogaze_hidden_for_distill
"""
from __future__ import annotations
import os, time, argparse
import numpy as np
import torch

from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper
from semantic_autogaze.eval_hlvid_subset import load_subset, PARQUET_PATH

from autogaze.models.autogaze import AutoGazeImageProcessor
from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
import av


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[setup] Loading SemanticAutoGazeWrapper...", flush=True)
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type=args.head_type,
        device=str(device),
    )
    autogaze_transform = AutoGazeImageProcessor.from_pretrained(args.autogaze_model)

    samples = load_subset(args.video_dir, parquet_path=args.parquet_path, query_mode="stem")
    if args.category:
        samples = [s for s in samples if s.get("category") == args.category]
    if args.n_samples is not None:
        samples = samples[:args.n_samples]
    print(f"[data] {len(samples)} samples (category={args.category})", flush=True)

    NUM_FRAMES = args.num_frames
    GAZING_RATIO = args.gazing_ratio

    # Use the first sample's question stem as a stub for AutoGaze.forward (which
    # requires a query_emb input but doesn't use it for hidden state extraction).
    # Actually extract_hidden_states doesn't need query_emb, so we use it directly.

    for i, s in enumerate(samples):
        qid = s["question_id"]
        out_path = os.path.join(args.output_dir, f"qid_{qid:04d}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"  [{i+1}/{len(samples)}] qid={qid} already cached, skipping", flush=True)
            continue

        try:
            t0 = time.perf_counter()
            container = av.open(s["video_path"])
            stream = container.streams.video[0]
            n_frames = stream.frames
            indices = list(range(min(NUM_FRAMES, n_frames or NUM_FRAMES)))
            raw_video = read_video_pyav(container=container, indices=indices)
            container.close()
            if raw_video.shape[0] < NUM_FRAMES:
                # Pad by repeating the last frame
                pad = NUM_FRAMES - raw_video.shape[0]
                raw_video = np.concatenate([raw_video, np.repeat(raw_video[-1:], pad, axis=0)], axis=0)

            video_autogaze = transform_video_for_pytorch(raw_video, autogaze_transform)[None].to(device)

            with torch.inference_mode():
                # Hidden states (B, T*196, hidden_dim)
                hidden = wrapper.extract_hidden_states(video_autogaze)  # (1, T*196, hidden_dim)

                # gazing_info via AutoGaze.forward gaze_only mode at the same gazing_ratio
                # We use a dummy 512-d query_emb; gazing_info doesn't depend on text
                dummy_query = torch.zeros(1, 512, device=device)
                gazing_info = wrapper.forward(
                    video_autogaze, dummy_query, mode="gaze_only",
                    gazing_ratio=GAZING_RATIO, task_loss_requirement=0.6,
                    semantic_keep_ratio=1.0,
                )

            hidden_cpu = hidden[0].detach().to(torch.float32).cpu().numpy()  # (T*196, hidden_dim)
            gazing_pos = gazing_info["gazing_pos"][0].detach().cpu().numpy()
            if_padded = gazing_info["if_padded_gazing"][0].detach().cpu().numpy()
            num_gaze_each_frame = gazing_info["num_gazing_each_frame"][0].detach().cpu().numpy()

            np.savez_compressed(
                out_path,
                hidden_states=hidden_cpu,                       # (T*196, hidden_dim)
                gazing_pos=gazing_pos,                           # (K_kept,) flat indices
                if_padded_gazing=if_padded,                      # (K_kept,) bool
                num_gazing_each_frame=num_gaze_each_frame,       # (T,)
                qid=int(qid),
                num_frames=NUM_FRAMES,
                gazing_ratio=GAZING_RATIO,
                video_path=s["video_path"],
                question_stem=s["question_stem"],
                answer=s["answer"],
            )
            wall = time.perf_counter() - t0
            print(f"  [{i+1}/{len(samples)}] qid={qid} hidden={hidden_cpu.shape} "
                  f"K_kept={(~if_padded.astype(bool)).sum()} wall={wall:.1f}s", flush=True)
        except Exception as e:
            print(f"  [error] qid={qid}: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    p.add_argument("--ckpt", default="results/bighead/best_bighead.pt",
                   help="(BigHead checkpoint kept for SemanticAutoGazeWrapper plumbing only)")
    p.add_argument("--head_type", default="bighead")
    p.add_argument("--parquet_path", default=PARQUET_PATH)
    p.add_argument("--video_dir", default="hlvid_videos/extracted_household/videos")
    p.add_argument("--category", default="household")
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--gazing_ratio", type=float, default=0.50)
    p.add_argument("--output_dir", default="results/autogaze_hidden_for_distill")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    main(args)
