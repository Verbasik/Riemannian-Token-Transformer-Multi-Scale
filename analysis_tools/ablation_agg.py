import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .common import ensure_outdir, list_experiments, load_metrics, safe_figsave
except ImportError:
    from common import ensure_outdir, list_experiments, load_metrics, safe_figsave


def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics across experiments and compute deltas to baseline")
    parser.add_argument("--results-root", type=Path, default=Path("Train/results"),
                        help="Root directory with experiment subfolders")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline experiment folder name")
    parser.add_argument("--out-dir", type=Path, default=Path("Train/analysis/ablation"),
                        help="Where to save tables/plots")
    args = parser.parse_args()

    out_dir = ensure_outdir(args.out_dir)
    exps = list_experiments(args.results_root)
    if not exps:
        raise FileNotFoundError(f"No experiments found in {args.results_root}")

    rows = []
    baseline_metrics = None
    for exp in exps:
        try:
            m = load_metrics(exp)
        except FileNotFoundError:
            continue
        row = {
            "exp": exp.name,
            "f1_macro": m.get("f1_macro"),
            "accuracy": m.get("accuracy"),
            "loss": m.get("loss"),
        }
        rows.append(row)
        if exp.name == args.baseline:
            baseline_metrics = row

    if baseline_metrics is None:
        raise ValueError(f"Baseline {args.baseline} not found in {args.results_root}")

    df = pd.DataFrame(rows).dropna(subset=["f1_macro"])
    for col in ["f1_macro", "accuracy", "loss"]:
        if col == "loss":
            df[f"{col}_delta"] = df[col] - baseline_metrics[col]
        else:
            df[f"{col}_delta"] = df[col] - baseline_metrics[col]

    df.to_csv(out_dir / "ablation_metrics.csv", index=False)

    # Bar plot of F1 delta
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(df)), 4))
    ax.bar(df["exp"], df["f1_macro_delta"], color="#4C78A8")
    ax.set_ylabel("Δ F1 macro vs baseline")
    ax.set_title(f"Ablations vs {args.baseline}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticklabels(df["exp"], rotation=45, ha="right")
    safe_figsave(fig, out_dir / "f1_delta.png")

    print(f"Saved aggregation to {out_dir}")


if __name__ == "__main__":
    main()
