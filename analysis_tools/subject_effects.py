import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

try:
    from .common import ensure_outdir, load_val_preds, optional_import_umap, safe_figsave
except ImportError:
    from common import ensure_outdir, load_val_preds, optional_import_umap, safe_figsave


def per_subject_metrics(y_true, y_pred, subject_ids, n_classes):
    subjects = np.unique(subject_ids)
    rows = []
    for s in subjects:
        mask = subject_ids == s
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "subject": s,
                "f1_macro": metrics.f1_score(y_true[mask], y_pred[mask], average="macro"),
                "accuracy": metrics.accuracy_score(y_true[mask], y_pred[mask]),
            }
        )
    return subjects, rows


def plot_box(subjects, values, title, out_file):
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(subjects)), 4))
    ax.boxplot(values, labels=[str(s) for s in subjects])
    ax.set_title(title)
    ax.set_xlabel("Subject")
    safe_figsave(fig, out_file)


def plot_umap(features: np.ndarray, subject_ids: np.ndarray, out_file: Path):
    umap = optional_import_umap()
    if umap is None:
        print("umap-learn not installed, skipping UMAP plot.")
        return
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    emb = reducer.fit_transform(features)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=subject_ids, cmap="tab10", s=10, alpha=0.7)
    fig.colorbar(sc, ax=ax, label="subject")
    ax.set_title("UMAP of pooled features by subject")
    safe_figsave(fig, out_file)


def main():
    parser = argparse.ArgumentParser(description="Subject-wise metrics and embeddings")
    parser.add_argument("--exp-dir", type=Path, required=True, help="Experiment directory with val_preds.npz")
    parser.add_argument("--features", type=Path, default=None,
                        help="Optional npy/npz with pooled features aligned to val_preds order")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = ensure_outdir(args.out_dir or args.exp_dir / "analysis" / "subject_effects")
    preds = load_val_preds(args.exp_dir)
    y_true, y_pred = preds["y_true"], preds["y_pred"]
    subject_ids = preds.get("subject_id")
    if subject_ids is None:
        raise ValueError("subject_id is missing in val_preds.npz")
    n_classes = preds["proba"].shape[1]

    subjects, rows = per_subject_metrics(y_true, y_pred, subject_ids, n_classes)
    f1_values = [r["f1_macro"] for r in rows]
    acc_values = [r["accuracy"] for r in rows]
    plot_box(subjects, f1_values, "F1 macro per subject", out_dir / "f1_per_subject.png")
    plot_box(subjects, acc_values, "Accuracy per subject", out_dir / "acc_per_subject.png")

    if args.features:
        feats = np.load(args.features)
        if isinstance(feats, np.lib.npyio.NpzFile):
            feats = feats[feats.files[0]]
        if feats.shape[0] != len(y_true):
            raise ValueError("features array length must match val_preds length")
        plot_umap(feats, subject_ids, out_dir / "features_umap.png")

    print(f"Saved subject-wise analysis to {out_dir}")


if __name__ == "__main__":
    main()
