"""
Teacher-head training (phase 1 of the distillation pipeline).

STUB — RECONSTRUCTED 2026-04-15 from wandb logs. Ground-truth code is on
the deactivated `cthulhu1` cluster. Do not consider this a faithful
replica; it captures the arguments and training shape so we can diff-and-
merge when the cluster returns.

Observed in `7bjfjj04` (distill-clip-teacher):

    === Phase 1: Training teacher on pre-decoder features ===
    Teacher: 2808.3K params
    Teacher 1/50: ... train=0.0819 val=0.0814
    ... Teacher 50/50: ...
    (saves results/distill/best_teacher.pt)

The teacher supervises a BigHead student in phase 2 (`train_distill_bighead.py`).
"""

from __future__ import annotations

import argparse

# NOTE: full data pipeline (cache loaders, CLIPSeg-target construction) was
# implemented on the cluster and has not been reproduced here. See
# ../../.../mac-brain/research/semantic-autogaze/architecture.md for the
# reconstructed description of the pipeline; reimplement only when needed.


def make_parser() -> argparse.ArgumentParser:
    """Argument spec inferred from wandb configs for `7bjfjj04`, `36huj7kn`."""
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/distill")
    p.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    p.add_argument("--clip_visual_dir", default="results/distill/clip_visual_cache")
    p.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", default="semantic-autogaze")
    return p


def main():  # pragma: no cover - stub
    raise NotImplementedError(
        "train_distill.py is a STUB. The full teacher training loop was "
        "implemented on the cluster. Reconstruct from output.log "
        "raw/wandb/semantic-autogaze/7bjfjj04_distill-clip-teacher/output.log "
        "or wait for cluster access to pull the real code."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
