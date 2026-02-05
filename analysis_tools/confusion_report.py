import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize

try:
    from .common import ensure_outdir, load_val_preds, safe_figsave
except ImportError:
    from common import ensure_outdir, load_val_preds, safe_figsave


def plot_confusion(y_true, y_pred, n_classes, out_dir: Path):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    for name, mat in [("counts", cm), ("normalized", cm_norm)]:
        fig, ax = plt.subplots(figsize=(1.6 * n_classes, 1.4 * n_classes))
        im = ax.imshow(mat, cmap="Blues")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_title(f"Confusion matrix ({name})")
        safe_figsave(fig, out_dir / f"confusion_{name}.png")


def plot_per_class_metrics(y_true, y_pred, n_classes, out_dir: Path):
    precision = metrics.precision_score(y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0)

    x = np.arange(n_classes)
    width = 0.25
    fig, ax = plt.subplots(figsize=(1.6 * n_classes, 4))
    ax.bar(x - width, precision, width, label="precision")
    ax.bar(x, recall, width, label="recall")
    ax.bar(x + width, f1, width, label="f1")
    ax.set_xticks(x)
    ax.set_xlabel("Class")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Per-class metrics")
    safe_figsave(fig, out_dir / "per_class_bars.png")


def plot_pr_roc(y_true, proba, n_classes, out_dir: Path):
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))

    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    pr_areas = []
    roc_areas = []
    for c in range(n_classes):
        precision, recall, _ = metrics.precision_recall_curve(y_true_bin[:, c], proba[:, c])
        pr_auc = metrics.auc(recall, precision)
        fpr, tpr, _ = metrics.roc_curve(y_true_bin[:, c], proba[:, c])
        roc_auc = metrics.auc(fpr, tpr)
        pr_areas.append(pr_auc)
        roc_areas.append(roc_auc)
        ax_pr.plot(recall, precision, alpha=0.6, label=f"class {c} (AP={pr_auc:.3f})")
        ax_roc.plot(fpr, tpr, alpha=0.6, label=f"class {c} (AUC={roc_auc:.3f})")

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"PR curves (macro AP={np.mean(pr_areas):.3f})")
    ax_pr.legend(fontsize=8, ncol=2)
    safe_figsave(fig_pr, out_dir / "pr_curves.png")

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title(f"ROC curves (macro AUC={np.mean(roc_areas):.3f})")
    ax_roc.legend(fontsize=8, ncol=2)
    safe_figsave(fig_roc, out_dir / "roc_curves.png")


def main():
    parser = argparse.ArgumentParser(description="Confusion matrix and PR/ROC from val_preds.npz")
    parser.add_argument("--exp-dir", type=Path, required=True, help="Experiment directory with val_preds.npz")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = ensure_outdir(args.out_dir or args.exp_dir / "analysis" / "confusion")
    preds = load_val_preds(args.exp_dir)
    y_true = preds["y_true"]
    y_pred = preds["y_pred"]
    proba = preds["proba"]
    n_classes = proba.shape[1]

    plot_confusion(y_true, y_pred, n_classes, out_dir)
    plot_per_class_metrics(y_true, y_pred, n_classes, out_dir)
    plot_pr_roc(y_true, proba, n_classes, out_dir)
    print(f"Saved confusion/PR/ROC plots to {out_dir}")


if __name__ == "__main__":
    main()
