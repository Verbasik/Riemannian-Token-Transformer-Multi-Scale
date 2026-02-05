import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .common import ensure_outdir, safe_figsave
except ImportError:
    from common import ensure_outdir, safe_figsave


def load_spd(path: Path) -> np.ndarray:
    data = np.load(path)
    if "covs" in data:
        return data["covs"]
    if "log_spectra" in data:
        return data["log_spectra"]
    # take first array
    first = data.files[0]
    return data[first]


def plot_eigenvalues(covs: np.ndarray, out_dir: Path):
    flat_eigs = []
    for c in covs:
        w, _ = np.linalg.eigh(c)
        flat_eigs.extend(w)
    flat_eigs = np.array(flat_eigs)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(flat_eigs, bins=60, color="#4C78A8", alpha=0.85)
    ax.set_title("Eigenvalue spectrum")
    ax.set_xlabel("Eigenvalue")
    safe_figsave(fig, out_dir / "eigenvalues_hist.png")


def plot_log_spectra(covs: np.ndarray, out_dir: Path):
    logs = []
    for c in covs:
        w, _ = np.linalg.eigh(c)
        logs.append(np.log(np.clip(w, 1e-8, None)))
    logs = np.array(logs)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(logs.mean(axis=0), marker="o")
    ax.set_title("Mean log-eigenvalues per channel")
    ax.set_xlabel("Channel index (ordered)")
    ax.set_ylabel("Mean log-eig")
    safe_figsave(fig, out_dir / "log_spectra_mean.png")


def plot_corr_of_logs(covs: np.ndarray, out_dir: Path):
    logs_flat = []
    for c in covs:
        w, _ = np.linalg.eigh(c)
        logs_flat.append(np.log(np.clip(w, 1e-8, None)))
    logs_flat = np.stack(logs_flat)
    corr = np.corrcoef(logs_flat, rowvar=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8, label="corr")
    ax.set_title("Correlation of log-eigenvalues")
    safe_figsave(fig, out_dir / "log_spectra_corr.png")


def main():
    parser = argparse.ArgumentParser(description="Plot SPD spectra and correlations")
    parser.add_argument("--covs", type=Path, required=True, help="npz with covs or log_spectra")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = ensure_outdir(args.out_dir or Path("Train/analysis/spd_spectra"))
    covs = load_spd(args.covs)
    if covs.ndim != 3 or covs.shape[1] != covs.shape[2]:
        raise ValueError("Expected covs shape (N, C, C)")

    plot_eigenvalues(covs, out_dir)
    plot_log_spectra(covs, out_dir)
    plot_corr_of_logs(covs, out_dir)
    print(f"Saved SPD spectra plots to {out_dir}")


if __name__ == "__main__":
    main()
