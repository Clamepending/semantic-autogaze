"""Self-contained Pi throughput benchmark for D-Mobile / v2-Tiny / v1.

Measures forward time on a 16-frame 224×224 video tensor (the deployment
shape used throughout the project). CPU-only; uses pre-cached text emb
to match the §4.4 deployment-realistic measurement protocol.

Usage on Pi 4 / Pi 5:
  python3 -m venv ~/.venv-pi-bench && source ~/.venv-pi-bench/bin/activate
  pip install --extra-index-url https://download.pytorch.org/whl/cpu \\
      torch torchvision timm numpy open_clip_torch
  curl -L https://github.com/Clamepending/semantic-autogaze/releases/download/v0.1.0-demo/D_mobilenet_std_best.pt -o D_mobile.pt
  python3 pi_throughput_bench.py --ckpt D_mobile.pt --model d-mobile --n_trials 20

For v2-Tiny:
  curl -L https://github.com/Clamepending/semantic-autogaze/releases/download/v0.1.0-demo/v2_tiny_best.pt -o v2_tiny.pt
  python3 pi_throughput_bench.py --ckpt v2_tiny.pt --model v2-tiny

For v1 (large; expect 30-60s/video on Pi 4):
  curl -L https://github.com/Clamepending/semantic-autogaze/releases/download/v0.1.0-demo/v1_best.pt -o v1.pt
  python3 pi_throughput_bench.py --ckpt v1.pt --model v1

The script does NOT need the project repo cloned — it inlines the head
class. Only torch + timm + open_clip + numpy are required.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


GRID = 14


# ----- Inlined head class (verbatim from train_independent_scorer.TextScorerHead) -----
class TextScorerHead(nn.Module):
    def __init__(self, patch_dim=768, text_dim=512, hidden_dim=384,
                 n_attn_heads=6, n_attn_layers=2, grid_size=14, use_spatial=True):
        super().__init__()
        self.grid_size = grid_size
        self.use_spatial = use_spatial
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
        if use_spatial:
            self.spatial = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.GELU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.GELU(),
                nn.Conv2d(64, 1, kernel_size=3, padding=1),
            )
        else:
            self.spatial = None

    def forward(self, patch_feats, text_emb):
        B = patch_feats.shape[0]
        G = self.grid_size
        x = self.patch_proj(patch_feats) + self.pos_embed
        for layer in self.self_attn_layers:
            r = x; x = layer["norm1"](x)
            xa, _ = layer["attn"](x, x, x); x = r + xa
            r = x; x = layer["norm2"](x); x = r + layer["ffn"](x)
        q = self.text_proj(text_emb).unsqueeze(1)
        cross_out, _ = self.cross_attn(x, q, q)
        x = self.cross_norm(x + cross_out)
        scores = self.score_mlp(x).squeeze(-1)
        if self.spatial is None:
            return scores
        grids = scores.reshape(B, 1, G, G)
        return (grids + self.spatial(grids)).reshape(B, G * G)


def adapt(feats, grid=GRID):
    if feats.dim() == 4:
        feats = F.interpolate(feats, size=(grid, grid), mode="bilinear", align_corners=False)
        return feats.permute(0, 2, 3, 1).reshape(feats.shape[0], grid * grid, feats.shape[1])
    if feats.shape[1] == 197:
        return feats[:, 1:, :]
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True, choices=["d-mobile", "v2-tiny", "v1"])
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--n_warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU thread count; default = all cores. Try 4 on Pi 4 (4 cores), 4 on Pi 5.")
    ap.add_argument("--out", default="pi_bench_result.json")
    args = ap.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    print(f"[torch] threads={torch.get_num_threads()}", flush=True)

    device = torch.device("cpu")

    print(f"[ckpt] loading {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    ca = ck.get("args", {}) or {}
    head_kwargs = dict(
        patch_dim=ck.get("embed_dim", 768) if args.model != "v1" else 768,
        text_dim=512,
        hidden_dim=ca.get("head_hidden_dim", 384),
        n_attn_heads=ca.get("head_attn_heads", 6),
        n_attn_layers=ca.get("head_attn_layers", 2),
        grid_size=GRID,
        use_spatial=ca.get("head_use_spatial", True),
    )
    print(f"[ckpt] head_kwargs={head_kwargs}", flush=True)
    head = TextScorerHead(**head_kwargs).to(device).eval()
    head.load_state_dict(ck["head"])

    print(f"[backbone] building {args.model}", flush=True)
    if args.model == "v1":
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        clip_model = clip_model.to(device).eval()
        clip_model.visual.output_tokens = True
        backbone = clip_model.visual
        clip_for_text = clip_model
        clip_tok_name = "ViT-B-16"
    else:
        import timm
        backbone = timm.create_model(ck["backbone"], pretrained=True, num_classes=0).to(device).eval()
        # text encoder via open_clip (CLIP-B/16)
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        clip_model = clip_model.to(device).eval()
        clip_for_text = clip_model
        clip_tok_name = "ViT-B-16"

    import open_clip
    clip_tok = open_clip.get_tokenizer(clip_tok_name)
    toks = clip_tok(["hand"]).to(device)
    with torch.no_grad():
        text_emb_cached = F.normalize(clip_for_text.encode_text(toks), dim=-1).expand(16, -1)
    print(f"[text] cached text_emb shape={text_emb_cached.shape}", flush=True)

    # Synthetic 16-frame video tensor (no decode IO; isolates compute).
    bench_in = torch.randn(16, 3, 224, 224, dtype=torch.float32)

    def fn():
        with torch.no_grad():
            if args.model == "v1":
                _, patches = backbone(bench_in)  # (16, 196, 768)
                head(patches, text_emb_cached)
            else:
                feats = adapt(backbone.forward_features(bench_in))
                head(feats, text_emb_cached)

    print(f"[warmup] {args.n_warmup} ...", flush=True)
    for i in range(args.n_warmup):
        t0 = time.perf_counter(); fn(); dt = time.perf_counter() - t0
        print(f"  warmup {i+1}: {dt*1000:.0f} ms", flush=True)

    print(f"[bench] {args.n_trials} trials ...", flush=True)
    times = []
    for i in range(args.n_trials):
        t0 = time.perf_counter(); fn(); dt = time.perf_counter() - t0
        times.append(dt * 1000)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{args.n_trials}  last: {dt*1000:.0f} ms  running mean: {np.mean(times):.0f} ms", flush=True)
    times = np.array(times)
    mean, std = float(times.mean()), float(times.std())
    median = float(np.median(times))

    backbone_params = sum(p.numel() for p in backbone.parameters()) / 1e6
    head_params = sum(p.numel() for p in head.parameters()) / 1e6

    result = {
        "model": args.model,
        "ckpt": args.ckpt,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "n_trials": args.n_trials,
        "video_shape": "(16, 3, 224, 224)",
        "deployment_realistic": True,
        "pre_cached_text_emb": True,
        "latency_ms_mean": mean,
        "latency_ms_std": std,
        "latency_ms_median": median,
        "backbone_params_M": backbone_params,
        "head_params_M": head_params,
        "all_trials_ms": times.tolist(),
    }
    print()
    print(f"=== {args.model} on CPU ({torch.get_num_threads()} threads) ===")
    print(f"  latency: {mean:.0f} ± {std:.0f} ms  (median {median:.0f}, n={args.n_trials})")
    print(f"  per-frame: {mean/16:.0f} ms  (16-frame video)")
    print(f"  throughput: {16000/mean:.1f} fps  (sustained)")
    print(f"  backbone: {backbone_params:.2f} M params")
    print(f"  head:     {head_params:.2f} M params")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved {args.out}")


if __name__ == "__main__":
    main()
