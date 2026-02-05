import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

try:
    from .common import ensure_outdir, optional_import_umap, safe_figsave
except ImportError:
    from common import ensure_outdir, optional_import_umap, safe_figsave


def load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        key = list(data.files)[0]
        return data[key]
    return np.load(path)


def apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std_safe = np.where(std == 0, 1.0, std)
    return (x - mean) / std_safe


def boxplot_channels(x: np.ndarray, title: str, out_file: Path):
    # x shape: N x C x T
    per_channel = x.reshape(x.shape[0], x.shape[1], -1).mean(axis=(0, 2))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(per_channel, vert=False)
    ax.set_title(title)
    ax.set_xlabel("Mean value per channel")
    safe_figsave(fig, out_file)


def project_embeddings(x: np.ndarray, title: str, out_file: Path, use_umap: bool):
    # flatten time per sample
    flat = x.reshape(x.shape[0], -1)
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(flat)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.4)
    ax.set_title(f"{title} (PCA)")
    safe_figsave(fig, out_file)

    if use_umap:
        umap = optional_import_umap()
        if umap is None:
            print("umap-learn is not installed; skipping UMAP plot.")
            return
        reducer = umap.UMAP(random_state=42, n_components=2, n_neighbors=15, min_dist=0.1)
        emb_u = reducer.fit_transform(flat)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(emb_u[:, 0], emb_u[:, 1], s=6, alpha=0.4, c="tab:orange")
        ax.set_title(f"{title} (UMAP)")
        safe_figsave(fig, out_file.with_name(out_file.stem + "_umap.png"))


def main():
    parser = argparse.ArgumentParser(description="Visualize effect of normalization on train/val arrays")
    parser.add_argument("--train", type=Path, required=True, help="Path to train array (.npy or .npz)")
    parser.add_argument("--val", type=Path, required=True, help="Path to val array (.npy or .npz)")
    parser.add_argument("--norm-stats", type=Path, required=True, help="npz with mean/std arrays")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--umap", action="store_true", help="Also build UMAP projections (requires umap-learn)")
    args = parser.parse_args()

    out_dir = ensure_outdir(args.out_dir or Path("Train/analysis/normalization_effect"))

    train = load_array(args.train)
    val = load_array(args.val)

    stats = np.load(args.norm_stats)
    mean = stats["mean"] if "mean" in stats else stats["mean_global"] if "mean_global" in stats else None
    std = stats["std"] if "std" in stats else stats["std_global"] if "std_global" in stats else None
    if mean is None or std is None:
        raise ValueError("norm-stats must contain mean/std arrays")

    train_norm = apply_norm(train, mean, std)
    val_norm = apply_norm(val, mean, std)

    boxplot_channels(train, "Train before norm (mean per channel)", out_dir / "train_box_before.png")
    boxplot_channels(train_norm, "Train after norm (mean per channel)", out_dir / "train_box_after.png")
    boxplot_channels(val, "Val before norm (mean per channel)", out_dir / "val_box_before.png")
    boxplot_channels(val_norm, "Val after norm (mean per channel)", out_dir / "val_box_after.png")

    project_embeddings(train, "Train before norm", out_dir / "train_pca.png", args.umap)
    project_embeddings(train_norm, "Train after norm", out_dir / "train_norm_pca.png", args.umap)
    project_embeddings(val_norm, "Val after norm", out_dir / "val_norm_pca.png", args.umap)

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
