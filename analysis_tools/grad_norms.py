import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .common import ensure_outdir, load_history, safe_figsave
except ImportError:
    from common import ensure_outdir, load_history, safe_figsave


def main():
    parser = argparse.ArgumentParser(description="Gradient norm histograms from history.json")
    parser.add_argument("--exp-dir", type=Path, required=True, help="Experiment directory with history.json")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = ensure_outdir(args.out_dir or args.exp_dir / "analysis" / "grad_norms")
    hist = load_history(args.exp_dir)
    keys = [k for k in hist if k.startswith("grad_norm")]
    if not keys:
        raise ValueError("No grad_norm entries in history.json")

    values = []
    for k in keys:
        values.extend([v for v in hist[k] if v is not None])
    values = np.array(values)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=40, color="#4C78A8", alpha=0.85)
    ax.set_xlabel("Gradient norm")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of gradient norms")
    safe_figsave(fig, out_dir / "grad_norm_hist.png")

    print(f"Saved grad norm histogram to {out_dir}")


if __name__ == "__main__":
    main()
