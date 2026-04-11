"""
Post-hoc temperature calibration for semantic filtering head.

Learns a single temperature parameter T such that sigmoid(logits/T)
is better calibrated against CLIPSeg GT probabilities.

Also implements Platt scaling (a*logit + b) for more flexible calibration.

Usage:
  python3 -m semantic_autogaze.calibrate \
    --hidden_dir results/distill/hidden_cache \
    --clipseg_dir results/distill/clipseg_cache \
    --ckpt results/distill_bighead/best_bighead_student.pt \
    --head_type bighead \
    --output_dir results/calibration
"""

import os
import glob
import hashlib
import random
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from semantic_autogaze.eval_filtering import EvalDataset, load_head


def collect_logits_and_targets(head, dataloader, device):
    """Collect raw logits and GT probabilities."""
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting logits"):
            hidden = batch["hidden_states"].to(device)
            query = batch["text_embedding"].to(device)
            target = batch["target_scores"].to(device)

            logits = head(hidden, query)
            gt_probs = torch.sigmoid(target)

            all_logits.append(logits.cpu())
            all_targets.append(gt_probs.cpu())

    return torch.cat(all_logits), torch.cat(all_targets)


def find_optimal_temperature(logits, targets, n_steps=1000):
    """Find T that minimizes BCE between sigmoid(logits/T) and targets."""
    best_T = 1.0
    best_loss = float("inf")

    temperatures = np.logspace(-1, 1, n_steps)  # 0.1 to 10.0
    losses = []

    for T in temperatures:
        preds = torch.sigmoid(logits / T)
        # BCE
        eps = 1e-7
        preds_clamped = preds.clamp(eps, 1 - eps)
        bce = -(targets * torch.log(preds_clamped) +
                (1 - targets) * torch.log(1 - preds_clamped)).mean().item()
        losses.append(bce)

        if bce < best_loss:
            best_loss = bce
            best_T = T

    return best_T, best_loss, temperatures, losses


def find_platt_scaling(logits, targets, lr=0.01, n_steps=2000):
    """Learn Platt scaling: sigmoid(a * logit + b)."""
    a = torch.nn.Parameter(torch.tensor(1.0))
    b = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.LBFGS([a, b], lr=lr)

    # Subsample for efficiency
    N = min(100000, logits.numel())
    idx = torch.randperm(logits.numel())[:N]
    logits_flat = logits.flatten()[idx]
    targets_flat = targets.flatten()[idx]

    def closure():
        optimizer.zero_grad()
        scaled = torch.sigmoid(a * logits_flat + b)
        eps = 1e-7
        scaled = scaled.clamp(eps, 1 - eps)
        loss = -(targets_flat * torch.log(scaled) +
                 (1 - targets_flat) * torch.log(1 - scaled)).mean()
        loss.backward()
        return loss

    for _ in range(50):
        optimizer.step(closure)

    return a.item(), b.item()


def evaluate_calibration(logits, targets, T=1.0, a=None, b_val=None):
    """Evaluate calibration quality with reliability diagram data."""
    if a is not None:
        preds = torch.sigmoid(a * logits + b_val)
    else:
        preds = torch.sigmoid(logits / T)

    preds_np = preds.numpy().flatten()
    targets_np = targets.numpy().flatten()
    targets_bin = (targets_np > 0.5).astype(np.float32)

    # Reliability diagram: bin predictions, compute mean target in each bin
    n_bins = 20
    bins = np.linspace(0, 1, n_bins + 1)
    bin_means = []
    bin_targets = []
    bin_counts = []

    for i in range(n_bins):
        mask = (preds_np >= bins[i]) & (preds_np < bins[i + 1])
        if mask.sum() > 0:
            bin_means.append(preds_np[mask].mean())
            bin_targets.append(targets_np[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_means.append((bins[i] + bins[i + 1]) / 2)
            bin_targets.append(0)
            bin_counts.append(0)

    # ECE (expected calibration error)
    total = sum(bin_counts)
    ece = sum(abs(m - t) * c / total for m, t, c in zip(bin_means, bin_targets, bin_counts))

    # mAP with calibrated scores
    if targets_bin.sum() > 0 and targets_bin.sum() < len(targets_bin):
        # Per-sample mAP
        B = logits.shape[0]
        aps = []
        for i in range(B):
            gt_i = (targets[i].numpy() > 0.5).astype(np.float32)
            if gt_i.sum() > 0 and gt_i.sum() < len(gt_i):
                aps.append(average_precision_score(gt_i, preds[i].numpy()))
        mAP = np.mean(aps) if aps else float("nan")
    else:
        mAP = float("nan")

    return {
        "ece": ece,
        "mAP": mAP,
        "bin_means": bin_means,
        "bin_targets": bin_targets,
        "bin_counts": bin_counts,
        "pred_mean": preds_np.mean(),
        "pred_std": preds_np.std(),
        "pred_max": preds_np.max(),
    }


def plot_calibration(results_dict, output_dir):
    """Plot reliability diagrams for different calibration methods."""
    fig, axes = plt.subplots(1, len(results_dict), figsize=(6 * len(results_dict), 5))
    if len(results_dict) == 1:
        axes = [axes]

    for ax, (label, res) in zip(axes, results_dict.items()):
        bins = np.array(res["bin_means"])
        targets = np.array(res["bin_targets"])
        counts = np.array(res["bin_counts"])

        # Plot bars
        width = 0.04
        ax.bar(bins, targets, width=width, alpha=0.6, color="#2196F3",
               edgecolor="#1565C0", label="Observed frequency")
        ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="Perfect calibration")

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"{label}\nECE={res['ece']:.4f}, mAP={res['mAP']:.4f}")
        ax.legend(fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "calibration_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/calibration_comparison.png")


def main(args):
    torch.manual_seed(42)
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load data
    clipseg_files = sorted(glob.glob(os.path.join(args.clipseg_dir, "*_clipseg_clip*.pt")))
    random.shuffle(clipseg_files)
    split = int(0.9 * len(clipseg_files))
    val_files = clipseg_files[split:]

    dataset = EvalDataset(val_files, args.hidden_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    head = load_head(args, device)

    # Collect logits
    print("Collecting logits and targets...")
    logits, targets = collect_logits_and_targets(head, dataloader, device)
    print(f"  Shape: {logits.shape}")

    # 1. Uncalibrated (T=1)
    print("\n1. Evaluating uncalibrated (T=1.0)...")
    uncal = evaluate_calibration(logits, targets, T=1.0)
    print(f"  ECE={uncal['ece']:.4f}, mAP={uncal['mAP']:.4f}")
    print(f"  Pred range: mean={uncal['pred_mean']:.4f}, std={uncal['pred_std']:.4f}, max={uncal['pred_max']:.4f}")

    # 2. Temperature scaling
    print("\n2. Finding optimal temperature...")
    best_T, best_loss, temps, losses = find_optimal_temperature(logits, targets)
    print(f"  Best T={best_T:.4f}, BCE={best_loss:.6f}")
    cal_T = evaluate_calibration(logits, targets, T=best_T)
    print(f"  ECE={cal_T['ece']:.4f}, mAP={cal_T['mAP']:.4f}")
    print(f"  Pred range: mean={cal_T['pred_mean']:.4f}, std={cal_T['pred_std']:.4f}, max={cal_T['pred_max']:.4f}")

    # 3. Platt scaling
    print("\n3. Learning Platt scaling (a*logit + b)...")
    a, b_val = find_platt_scaling(logits, targets)
    print(f"  a={a:.4f}, b={b_val:.4f}")
    cal_platt = evaluate_calibration(logits, targets, a=a, b_val=b_val)
    print(f"  ECE={cal_platt['ece']:.4f}, mAP={cal_platt['mAP']:.4f}")
    print(f"  Pred range: mean={cal_platt['pred_mean']:.4f}, std={cal_platt['pred_std']:.4f}, max={cal_platt['pred_max']:.4f}")

    # Plot
    results_dict = {
        f"Uncalibrated (T=1.0)": uncal,
        f"Temperature (T={best_T:.2f})": cal_T,
        f"Platt (a={a:.2f}, b={b_val:.2f})": cal_platt,
    }
    plot_calibration(results_dict, args.output_dir)

    # Temperature sweep plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(temps, losses, lw=2, color="#2196F3")
    ax.axvline(x=best_T, color="red", linestyle="--", label=f"Best T={best_T:.3f}")
    ax.axvline(x=1.0, color="gray", linestyle=":", label="T=1.0 (uncalibrated)")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Temperature Scaling Sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "temperature_sweep.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {args.output_dir}/temperature_sweep.png")

    # Save results
    save = {
        "head_type": args.head_type,
        "ckpt": args.ckpt,
        "uncalibrated": {"T": 1.0, "ece": uncal["ece"], "mAP": uncal["mAP"]},
        "temperature": {"T": best_T, "ece": cal_T["ece"], "mAP": cal_T["mAP"]},
        "platt": {"a": a, "b": b_val, "ece": cal_platt["ece"], "mAP": cal_platt["mAP"]},
    }
    with open(os.path.join(args.output_dir, "calibration_results.json"), "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved results to {args.output_dir}/calibration_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", default="results/distill/hidden_cache")
    parser.add_argument("--clipseg_dir", default="results/distill/clipseg_cache")
    parser.add_argument("--ckpt", default="results/distill_bighead/best_bighead_student.pt")
    parser.add_argument("--head_type", choices=["small", "bighead"], default="bighead")
    parser.add_argument("--output_dir", default="results/calibration")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--expanded_dim", type=int, default=384)
    parser.add_argument("--n_attn_heads", type=int, default=6)
    parser.add_argument("--n_attn_layers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args)
