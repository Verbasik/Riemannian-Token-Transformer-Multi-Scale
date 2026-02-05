import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .common import ensure_outdir, safe_figsave
except ImportError:
    from common import ensure_outdir, safe_figsave


def load_attn(path: Path) -> np.ndarray:
    data = np.load(path)
    if "attn_weights" in data:
        return data["attn_weights"]
    first = data.files[0]
    return data[first]


def plot_head_heatmaps(attn: np.ndarray, out_dir: Path):
    # expected shape: (B, heads, tokens, tokens) or (heads, tokens, tokens)
    if attn.ndim == 4:
        attn_mean = attn.mean(axis=0)
    elif attn.ndim == 3:
        attn_mean = attn
    else:
        raise ValueError("Unsupported attention shape")

    heads = attn_mean.shape[0]
    for h in range(heads):
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(attn_mean[h], cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.8, label="weight")
        ax.set_title(f"Head {h} mean attention")
        safe_figsave(fig, out_dir / f"head_{h}.png")


def plot_head_weights(attn: np.ndarray, out_dir: Path):
    # mean weight per head
    if attn.ndim == 4:
        head_mean = attn.mean(axis=(0, 2, 3))
    elif attn.ndim == 3:
        head_mean = attn.mean(axis=(1, 2))
    else:
        return
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(head_mean)), head_mean, color="#4C78A8")
    ax.set_xlabel("Head")
    ax.set_ylabel("Mean weight")
    ax.set_title("Head-wise mean attention weight")
    safe_figsave(fig, out_dir / "head_weights.png")


def main():
    parser = argparse.ArgumentParser(description="Attention statistics from attn_stats.npz")
    parser.add_argument("--attn", type=Path, required=True, help="Path to attn_stats.npz")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = ensure_outdir(args.out_dir or args.attn.parent / "analysis" / "attention")
    attn = load_attn(args.attn)
    plot_head_heatmaps(attn, out_dir)
    plot_head_weights(attn, out_dir)
    print(f"Saved attention plots to {out_dir}")


if __name__ == "__main__":
    main()
