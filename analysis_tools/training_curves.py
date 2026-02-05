import argparse
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from .common import ensure_outdir, load_history, safe_figsave
except ImportError:
    from common import ensure_outdir, load_history, safe_figsave


def plot_losses(hist, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist["train_loss"], label="train_loss")
    ax.plot(hist["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train/Val loss")
    ax.legend()
    safe_figsave(fig, out_dir / "loss.png")


def plot_f1(hist, out_dir: Path):
    if "val_f1_macro" not in hist:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist["val_f1_macro"], label="val_f1_macro", color="#4C78A8")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 macro")
    ax.set_title("Validation F1")
    safe_figsave(fig, out_dir / "val_f1_macro.png")


def plot_lr(hist, out_dir: Path):
    if "lr" not in hist:
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(hist["lr"], color="#54A24B")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.set_title("Learning rate")
    safe_figsave(fig, out_dir / "lr.png")


def plot_grad_norms(hist, out_dir: Path):
    keys = ["grad_norm_min", "grad_norm_mean", "grad_norm_max"]
    if not all(k in hist for k in keys):
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist["grad_norm_min"], label="min", alpha=0.6)
    ax.plot(hist["grad_norm_mean"], label="mean", alpha=0.8)
    ax.plot(hist["grad_norm_max"], label="max", alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Grad norm")
    ax.set_title("Gradient norms")
    ax.legend()
    safe_figsave(fig, out_dir / "grad_norms.png")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from history.json")
    parser.add_argument("--exp-dir", type=Path, required=True, help="Experiment directory with history.json")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = ensure_outdir(args.out_dir or args.exp_dir / "analysis" / "training_curves")
    hist = load_history(args.exp_dir)

    plot_losses(hist, out_dir)
    plot_f1(hist, out_dir)
    plot_lr(hist, out_dir)
    plot_grad_norms(hist, out_dir)
    print(f"Saved training curves to {out_dir}")


if __name__ == "__main__":
    main()
