"""Local CPU/MPS smoke test. Builds BigHead + SimilarityHead, runs a forward
pass on random inputs, prints param counts + shapes. Does NOT download the
AutoGaze backbone or CLIP weights (keeps the test fast and offline)."""

from __future__ import annotations

import torch

from semantic_autogaze.bighead import BigHead, TemporalBigHead
from semantic_autogaze.model import SimilarityHead


def main():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {dev}")

    configs = [
        ("BigHead e=512,L=3,h=8 (best run)", BigHead(192, 512, 512, 8, 3)),
        ("BigHead e=384,L=2,h=6", BigHead(192, 512, 384, 6, 2)),
        ("BigHead A4 (no spatial)", BigHead(192, 512, 384, 6, 2, use_spatial_attn=False)),
        ("TemporalBigHead e=384,sL=2,tL=1", TemporalBigHead(192, 512, 384, 6, 2, 1)),
        ("SimilarityHead (committed)", SimilarityHead(hidden_dim=192, embedding_dim=512)),
    ]

    B, T, N = 2, 16, 196
    patches = torch.randn(B, T * N, 192, device=dev)
    query = torch.randn(B, 512, device=dev)

    for name, m in configs:
        m = m.to(dev).eval()
        params = sum(p.numel() for p in m.parameters()) / 1000
        with torch.no_grad():
            out = m(patches, query)
        print(f"  {name:<45s}  params={params:>8.1f}K  out={tuple(out.shape)}")


if __name__ == "__main__":
    main()
