import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

try:
    from .common import ensure_outdir, list_experiments, load_config, load_metrics, safe_figsave
except ImportError:
    from common import ensure_outdir, list_experiments, load_config, load_metrics, safe_figsave


def main():
    parser = argparse.ArgumentParser(description="SPD augmentation sensitivity: std/prob vs F1")
    parser.add_argument("--results-root", type=Path, default=Path("Train/results"),
                        help="Root with experiment subfolders")
    parser.add_argument("--out-dir", type=Path, default=Path("Train/analysis/spd_aug"),
                        help="Where to save plots")
    args = parser.parse_args()

    out_dir = ensure_outdir(args.out_dir)
    rows = []
    for exp in list_experiments(args.results_root):
        try:
            cfg = load_config(exp)
            m = load_metrics(exp)
        except FileNotFoundError:
            continue
        aug_cfg = cfg.get("model", {})
        std = aug_cfg.get("spd_jitter_std")
        prob = aug_cfg.get("spd_jitter_prob")
        use_aug = aug_cfg.get("use_spd_augment", False)
        if std is None or prob is None:
            continue
        rows.append({"exp": exp.name, "std": float(std), "prob": float(prob), "use": bool(use_aug),
                     "f1_macro": m.get("f1_macro")})

    if not rows:
        raise ValueError("No experiments with spd_jitter_std/prob found")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "spd_aug_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    for prob, sub in df.groupby("prob"):
        ax.plot(sub["std"], sub["f1_macro"], marker="o", label=f"prob={prob}")
    ax.set_xlabel("spd_jitter_std")
    ax.set_ylabel("F1 macro")
    ax.set_title("SPD augmentation sensitivity")
    ax.legend()
    safe_figsave(fig, out_dir / "std_prob_curve.png")

    pivot = df.pivot_table(index="prob", columns="std", values="f1_macro")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("std")
    ax.set_ylabel("prob")
    ax.set_title("F1 heatmap")
    fig.colorbar(im, ax=ax, label="F1 macro")
    safe_figsave(fig, out_dir / "std_prob_heatmap.png")

    print(f"Saved SPD augmentation sensitivity plots to {out_dir}")


if __name__ == "__main__":
    main()
