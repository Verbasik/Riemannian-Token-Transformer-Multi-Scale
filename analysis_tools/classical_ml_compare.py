import argparse
from pathlib import Path

import pandas as pd

try:
    from .common import ensure_outdir, list_experiments, load_metrics
except ImportError:
    from common import ensure_outdir, list_experiments, load_metrics


def main():
    parser = argparse.ArgumentParser(description="Compare classical ML and DL metrics across experiments")
    parser.add_argument("--results-root", type=Path, default=Path("Train/results"),
                        help="Root with experiment folders")
    parser.add_argument("--include", type=str, default=None,
                        help="Substring to filter experiment names (e.g., 'classical')")
    parser.add_argument("--out-dir", type=Path, default=Path("Train/analysis/classical_compare"),
                        help="Output directory")
    args = parser.parse_args()

    out_dir = ensure_outdir(args.out_dir)
    rows = []
    for exp in list_experiments(args.results_root):
        if args.include and args.include not in exp.name:
            continue
        try:
            m = load_metrics(exp)
        except FileNotFoundError:
            continue
        rows.append({"exp": exp.name, "f1_macro": m.get("f1_macro"), "accuracy": m.get("accuracy"), "loss": m.get("loss")})

    if not rows:
        raise ValueError("No experiments matched the filter.")

    df = pd.DataFrame(rows).sort_values(by="f1_macro", ascending=False)
    df.to_csv(out_dir / "classical_vs_dl.csv", index=False)
    print(f"Saved comparison table to {out_dir}")


if __name__ == "__main__":
    main()
