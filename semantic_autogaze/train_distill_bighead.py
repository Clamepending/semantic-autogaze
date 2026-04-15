"""
Student BigHead training (phase 2 of the distillation pipeline).

STUB — RECONSTRUCTED 2026-04-15 from wandb logs. See sibling
`train_distill.py` for provenance notes. Best observed result:

    run nkmyibxc (distill-bighead-e512-L3), val/best_bce=0.0683 @ 50 epochs.

Expected loss:

    L = BCE(student_logits, clipseg_target)
        + distill_alpha * T^2 * KL(softmax(teacher/T) || softmax(student/T))
    distill_alpha=0.5, T=2

Iteration math for verification: batch_size=48 × 675 steps/epoch = 32,400
samples/epoch = train split size. ✓
"""

from __future__ import annotations

import argparse


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/distill_bighead_xl")
    p.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    p.add_argument("--clip_visual_dir", default="results/distill/clip_visual_cache")
    p.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    p.add_argument("--teacher_ckpt", default="results/distill/best_teacher.pt")
    p.add_argument("--resume_ckpt", default=None)
    p.add_argument("--expanded_dim", type=int, default=512)
    p.add_argument("--n_attn_heads", type=int, default=8)
    p.add_argument("--n_attn_layers", type=int, default=3)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--distill_alpha", type=float, default=0.5)
    p.add_argument("--distill_temp", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", default="semantic-autogaze")
    return p


def main():  # pragma: no cover - stub
    raise NotImplementedError(
        "train_distill_bighead.py is a STUB. Replace with cluster code when "
        "cluster returns. For architecture, see `bighead.BigHead`."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
