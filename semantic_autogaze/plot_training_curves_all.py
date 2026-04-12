"""Generate a training-curves comparison figure across all active runs.

Pulls data from wandb and plots val BCE vs epoch for every experiment,
highlighting the baseline and the new best (warm restart).
"""

import os
import argparse
import matplotlib.pyplot as plt
import wandb


# Runs to include and their display config
RUNS = [
    # (run_id, label, color, linestyle)
    ("w7ly4c8w", "BigHead Distill (baseline, 0.0668)", "tab:blue",   "-"),
    ("lkrzsjm4", "Warm Restart (0.0666)",             "tab:red",    "-"),
    ("nkmyibxc", "XL BigHead (e512, L3, 8M, 0.0683)", "tab:brown",  "-"),
    ("evn4q4am", "Temporal BigHead (0.0700)",         "tab:green",  "-"),
    ("6kslwkls", "Deep Temporal (2T+3S)",             "tab:purple", "--"),
    ("36huj7kn", "Pre-decoder Student",               "tab:orange", "--"),
]

PROJECT = "semantic-autogaze"


def main(output_path: str) -> None:
    api = wandb.Api()
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5))

    for run_id, label, color, ls in RUNS:
        try:
            run = api.run(f"{PROJECT}/{run_id}")
        except Exception as e:
            print(f"Skipping {run_id}: {e}")
            continue

        epochs, vals = [], []
        for row in run.history(keys=["epoch", "val/epoch_bce"], samples=500, pandas=False):
            e, v = row.get("epoch"), row.get("val/epoch_bce")
            if e is not None and v is not None:
                epochs.append(e)
                vals.append(v)
        if not epochs:
            print(f"No data for {run_id}")
            continue

        best = min(vals)
        display_label = f"{label} [{run.state}, best={best:.4f}]"
        ax_full.plot(epochs, vals, label=display_label, color=color, linestyle=ls, linewidth=1.8)
        ax_zoom.plot(epochs, vals, label=display_label, color=color, linestyle=ls, linewidth=1.8)

    ax_full.set_xlabel("Epoch")
    ax_full.set_ylabel("Val BCE")
    ax_full.set_title("Training Convergence — All Variants")
    ax_full.grid(True, alpha=0.3)
    ax_full.legend(loc="upper right", fontsize=8)

    ax_zoom.set_xlabel("Epoch")
    ax_zoom.set_ylabel("Val BCE")
    ax_zoom.set_title("Zoom: 0.065 – 0.085")
    ax_zoom.set_ylim(0.065, 0.085)
    ax_zoom.axhline(0.0668, color="gray", linestyle=":", linewidth=1, label="baseline 0.0668")
    ax_zoom.axhline(0.0666, color="tab:red", linestyle=":", linewidth=1, label="warm restart 0.0666")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/paper_figures/training_curves_all.png")
    args = parser.parse_args()
    main(args.output)
