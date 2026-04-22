"""Cycle 1 of r/debug-score-distribution.

Goal: determine whether BigHead's near-zero sigmoid scores on HLVid
(p99<=0.01, max<=0.07 in r/hlvid-expand-n) are a REAL calibration gap or a
preprocessing / loader bug.

Three probes:

  (A) Training-cache reference: load a random CLIPSeg-cache file, feed its
      stored (hidden_states, text_embedding) directly to the BigHead, compare
      to stored target_scores. If the head outputs a flat distribution here,
      the checkpoint/loader is broken. If it matches the teacher, the head
      works on its training distribution.

  (B) HLVid clip, TRAINING-matched preprocessing: read frames, normalize to
      [-1, 1] (matches how results/clipseg/hidden_cache was built — see
      train_clipseg.py:precompute_hidden_states). Run through AutoGaze
      encoder -> hidden states -> BigHead.

  (C) HLVid clip, EVAL-matched preprocessing: apply AutoGazeImageProcessor's
      full transform (rescale to [-1, 1] THEN normalize by IMAGENET_STANDARD
      mean/std = 0.5/0.5). This is what the NVILA processor feeds at eval
      time via transform_video_for_pytorch.

If (B) distributions look like the training reference (A) but (C) are flat,
we've identified a preprocessing-path mismatch between BigHead training and
eval. If (B) is also flat on HLVid, the teacher gap is real.

Writes one JSON at results/debug_score_dist/scores.json with per-probe
per-frame quantile stats.
"""
import os, re, json, glob, random, argparse, hashlib, fractions
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
import pandas as pd
import av
from PIL import Image
from einops import rearrange

from semantic_autogaze.semantic_autogaze_wrapper import SemanticAutoGazeWrapper


def read_video_frames_fast(video_path, num_frames=16, size=224):
    """Fast frame reader: seeks to 16 evenly-spaced timestamps instead of
    decoding every frame. Critical for 2-4 GB HLVid mp4s where sequential
    decode takes 5+ minutes.
    """
    try:
        container = av.open(video_path)
    except Exception:
        return None

    stream = container.streams.video[0]
    stream.codec_context.skip_frame = "NONKEY"  # ignored for precise seek fallback

    duration = float(stream.duration * stream.time_base) if stream.duration else None
    if duration is None or duration <= 0:
        container.close()
        return None

    # Target times: N evenly spaced points in [0, duration - 1/fps]
    times = np.linspace(0.1, max(0.2, duration - 0.1), num_frames)

    frames = []
    for t_sec in times:
        pts = int(t_sec / stream.time_base)
        try:
            container.seek(pts, any_frame=False, backward=True, stream=stream)
        except Exception:
            continue
        got = None
        for frame in container.decode(stream):
            # Take the first frame at or after the target pts
            if frame.pts is None:
                continue
            if frame.pts * stream.time_base >= t_sec - 0.05:
                got = frame
                break
        if got is None:
            continue
        img = got.to_image().resize((size, size), Image.BILINEAR)
        frames.append(np.array(img))

    container.close()
    if len(frames) == 0:
        return None
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.stack(frames[:num_frames])


IMAGENET_STD_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
IMAGENET_STD_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

# HLVid clips chosen to span the three question-flavor buckets in the
# resolved r/hlvid-expand-n eval. Picked by hand from the on-disk extracted
# shards; each matches a specific question from test-00000-of-00001.parquet.
HLVID_CLIPS = [
    {
        "tag": "ocr_sign",
        "path": "hlvid_videos/extracted/videos/clip_av_video_0_000.mp4",
        "query_hint": "text on the white signboard",
    },
    {
        "tag": "vehicle_dynamics",
        "path": "hlvid_videos/extracted/videos/clip_av_video_10_000.mp4",
        "query_hint": "the car on the road",
    },
    {
        "tag": "pedestrian",
        "path": "hlvid_videos/extracted/videos/clip_av_video_1_000.mp4",
        "query_hint": "person walking",
    },
]


def quant_stats(t: torch.Tensor) -> dict:
    """Scalar summary stats for a flat score tensor."""
    x = t.detach().float().cpu().flatten()
    if x.numel() == 0:
        return {}
    qs = torch.quantile(x, torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99]))
    # Shannon entropy of the distribution over 20 bins in [0, 1].
    hist = torch.histc(x.clamp(0.0, 1.0), bins=20, min=0.0, max=1.0) + 1e-9
    p = hist / hist.sum()
    ent = -(p * p.log()).sum().item()
    return {
        "n": int(x.numel()),
        "min": float(x.min()),
        "p01": float(qs[0]),
        "p10": float(qs[1]),
        "p50": float(qs[2]),
        "p90": float(qs[3]),
        "p99": float(qs[4]),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "entropy_bits": float(ent / np.log(2)),
    }


@torch.no_grad()
def probe_training_cache(wrapper, device, n_samples=8, seed=0):
    """Probe (A): BigHead on stored training triples."""
    rng = random.Random(seed)
    files = sorted(glob.glob("results/clipseg/clipseg_cache/*_clipseg_clip.pt"))
    hidden_dir = "results/clipseg/hidden_cache"
    if not files:
        return {"error": "no clipseg cache files found"}
    rng.shuffle(files)

    results = []
    for cf in files:
        if len(results) >= n_samples:
            break
        data = torch.load(cf, map_location="cpu", weights_only=False)
        vp = data["video_path"]
        key = hashlib.md5(vp.encode()).hexdigest()
        hp = os.path.join(hidden_dir, f"{key}_hidden.pt")
        if not os.path.exists(hp):
            continue
        hidden = torch.load(hp, map_location=device, weights_only=True)  # (T*196, 192)
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)  # (1, T*196, 192)
        q = rng.choice(data["queries"])
        emb = q["text_embedding"].to(device).unsqueeze(0)  # (1, 512)
        tgt = q["target_scores"].to(device).unsqueeze(0)   # (1, T*196)

        scores = wrapper.semantic_filter.get_scores(hidden, emb)  # (1, T*196)
        tgt_sig = torch.sigmoid(tgt)

        results.append({
            "video_path": vp,
            "query_text": q["text"],
            "bighead_sig": quant_stats(scores),
            "teacher_sig": quant_stats(tgt_sig),
            "corr": float(torch.corrcoef(
                torch.stack([scores.flatten(), tgt_sig.flatten()])
            )[0, 1]),
        })
    return results


@torch.no_grad()
def load_frames_and_preprocess(video_path, num_frames, size, mode, device):
    """Return (1, T, C, H, W) video tensor under the chosen preprocessing.

    mode:
      "train"  -> (pixel/127.5) - 1  (matches train_clipseg hidden_cache)
      "eval"   -> rescale to [-1, 1], then (x - 0.5) / 0.5
                  (matches AutoGazeImageProcessor defaults used by NVILA)
    """
    frames = read_video_frames_fast(video_path, num_frames=num_frames, size=size)
    if frames is None:
        return None
    vid = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # (T, C, H, W)
    vid = vid / 127.5 - 1.0
    if mode == "eval":
        vid = (vid - IMAGENET_STD_MEAN) / IMAGENET_STD_STD
    vid = vid.unsqueeze(0).to(device)  # (1, T, C, H, W)
    return vid


@torch.no_grad()
def probe_hlvid(wrapper, clip_model, clip_tokenizer, device, num_frames=16):
    """Probe (B) and (C): run BigHead on HLVid clips under two preproc paths."""
    results = []
    for clip in HLVID_CLIPS:
        if not os.path.exists(clip["path"]):
            results.append({**clip, "error": "video missing"})
            continue

        # Query embedding (matches eval's path: CLIP ViT-B/16, normalized)
        q = clip["query_hint"]
        tok = clip_tokenizer([q]).to(device)
        emb = clip_model.encode_text(tok)
        emb = F.normalize(emb, dim=-1)  # (1, 512)

        per_mode = {}
        for mode in ("train", "eval"):
            vid = load_frames_and_preprocess(
                clip["path"], num_frames=num_frames, size=224,
                mode=mode, device=device,
            )
            if vid is None:
                per_mode[mode] = {"error": "frame read failed"}
                continue
            hidden = wrapper.extract_hidden_states(vid)  # (1, T*196, 192)
            scores = wrapper.semantic_filter.get_scores(hidden, emb)  # (1, T*196)

            # per-frame breakdown: (1, T, 196)
            T = num_frames
            per_frame = scores.reshape(1, T, -1).squeeze(0)  # (T, 196)
            per_frame_stats = [quant_stats(per_frame[t]) for t in range(T)]

            per_mode[mode] = {
                "hidden_state_stats": quant_stats(hidden.flatten()),
                "logits": quant_stats(wrapper.semantic_filter.head(hidden, emb)),
                "sig_overall": quant_stats(scores),
                "sig_per_frame": per_frame_stats,
            }

        results.append({**clip, "per_mode": per_mode})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/bighead_warmrestart/best_bighead_student.pt")
    ap.add_argument("--autogaze_model", default="nvidia/AutoGaze")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--output_dir", default="results/debug_score_dist")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"[setup] Loading SemanticAutoGazeWrapper on {device}...")
    wrapper = SemanticAutoGazeWrapper(
        autogaze_model_name=args.autogaze_model,
        head_ckpt=args.ckpt,
        head_type="bighead",
        device=str(device),
    )

    print("[setup] Loading CLIP ViT-B/16...")
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.to(device).eval()

    print("\n[probe A] BigHead on training-cache triples (teacher comparison)")
    probe_a = probe_training_cache(wrapper, device, n_samples=8, seed=args.seed)
    if isinstance(probe_a, list):
        for r in probe_a:
            bh = r["bighead_sig"]
            th = r["teacher_sig"]
            print(f"  {os.path.basename(r['video_path'])[:30]:<30}  q={r['query_text']!r:<25}  "
                  f"BH p99={bh['p99']:.3f} max={bh['max']:.3f}  "
                  f"teacher p99={th['p99']:.3f} max={th['max']:.3f}  "
                  f"corr={r['corr']:.3f}")

    print("\n[probe B/C] BigHead on HLVid clips, train-matched vs eval-matched preproc")
    probe_bc = probe_hlvid(wrapper, clip_model, clip_tokenizer, device, num_frames=args.num_frames)
    for r in probe_bc:
        print(f"\n  clip={r['tag']}  q={r['query_hint']!r}  path={r['path']}")
        if "error" in r:
            print(f"    ERROR: {r['error']}")
            continue
        for mode in ("train", "eval"):
            pm = r["per_mode"][mode]
            if "error" in pm:
                print(f"    {mode:<5} ERROR: {pm['error']}")
                continue
            h = pm["hidden_state_stats"]
            lo = pm["logits"]
            si = pm["sig_overall"]
            print(f"    {mode:<5}  hidden mean={h['mean']:+.3f} std={h['std']:.3f}  "
                  f"logits p99={lo['p99']:+.2f} max={lo['max']:+.2f}  "
                  f"sig p50={si['p50']:.4f} p90={si['p90']:.4f} p99={si['p99']:.4f} "
                  f"max={si['max']:.4f} ent={si['entropy_bits']:.2f}")

    out = {
        "meta": {
            "ckpt": args.ckpt,
            "autogaze_model": args.autogaze_model,
            "device": str(device),
            "num_frames": args.num_frames,
            "seed": args.seed,
        },
        "probe_A_training_cache": probe_a,
        "probe_BC_hlvid": probe_bc,
    }
    out_path = os.path.join(args.output_dir, "scores.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
