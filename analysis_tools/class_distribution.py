import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .common import ensure_outdir, load_val_preds, safe_figsave
except ImportError:
    from common import ensure_outdir, load_val_preds, safe_figsave


def plot_class_counts(y_true: np.ndarray, n_classes: int, out_dir: Path):
    counts = np.bincount(y_true, minlength=n_classes)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(n_classes), counts, color="#4C78A8")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class distribution")
    safe_figsave(fig, out_dir / "class_counts.png")


def plot_subject_counts(subject_ids: np.ndarray, out_dir: Path):
    uniq, counts = np.unique(subject_ids, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(u) for u in uniq], counts, color="#F58518")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Count")
    ax.set_title("Samples per subject")
    safe_figsave(fig, out_dir / "subject_counts.png")


def plot_subject_class_heatmap(y_true: np.ndarray, subject_ids: np.ndarray, n_classes: int, out_dir: Path):
    subjects = np.unique(subject_ids)
    mat = np.zeros((len(subjects), n_classes), dtype=int)
    subj_index = {s: i for i, s in enumerate(subjects)}
    for label, subj in zip(y_true, subject_ids):
        mat[subj_index[subj], label] += 1

    fig, ax = plt.subplots(figsize=(1.6 * n_classes, 0.6 * len(subjects) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="Blues")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels([str(s) for s in subjects])
    ax.set_xlabel("Class")
    ax.set_ylabel("Subject")
    ax.set_title("Class counts per subject")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
    safe_figsave(fig, out_dir / "subject_class_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description="Plot class and subject distributions from val_preds.npz")
    parser.add_argument("--exp-dir", type=Path, required=True, help="Path to experiment directory with val_preds.npz")
    parser.add_argument("--metaclasses", type=Path, default=Path("json/metaclasses.json"),
                        help="Path to metaclasses mapping (used to infer n_classes if proba missing)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for figures")
    args = parser.parse_args()

    out_dir = ensure_outdir(args.out_dir or args.exp_dir / "analysis" / "class_distribution")
    preds = load_val_preds(args.exp_dir)
    y_true = preds["y_true"]
    subject_ids = preds.get("subject_id")
    n_classes = preds["proba"].shape[1] if "proba" in preds else len(json.load(args.metaclasses.open()))

    plot_class_counts(y_true, n_classes, out_dir)
    if subject_ids is not None:
        plot_subject_counts(subject_ids, out_dir)
        plot_subject_class_heatmap(y_true, subject_ids, n_classes, out_dir)

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
