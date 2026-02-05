import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from .common import ensure_outdir, safe_figsave
except ImportError:
    from common import ensure_outdir, safe_figsave


def load_array(path: Path) -> np.ndarray:
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        return data[data.files[0]]
    return data


def channel_variance(data: np.ndarray) -> np.ndarray:
    # data shape: N x C x T
    return data.var(axis=(0, 2))


def extract_channel_weights(ckpt_path: Path) -> np.ndarray:
    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    keys = [k for k in state.keys() if "channel_proj.weight" in k]
    if not keys:
        raise ValueError("channel_proj.weight not found in checkpoint")
    w = state[keys[0]]
    # shape: out_channels x in_channels; importance per input channel = L2 norm
    if w.ndim == 2:
        return torch.norm(w, dim=0).detach().cpu().numpy()
    if w.ndim == 3:  # conv1d weight (out,in,k)
        return torch.norm(w, dim=(0, 2)).detach().cpu().numpy()
    raise ValueError("Unsupported channel_proj weight shape")


def plot_importance(values: np.ndarray, title: str, out_file: Path):
    idx = np.argsort(values)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(values)), values[idx])
    ax.set_xlabel("Channel (sorted)")
    ax.set_ylabel("Importance")
    ax.set_title(title)
    safe_figsave(fig, out_file)


def plot_montage(values: np.ndarray, montage_path: Path, out_file: Path):
    df = pd.read_csv(montage_path)
    cols = {c.lower(): c for c in df.columns}
    xcol = cols.get("x")
    ycol = cols.get("y")
    vcol = cols.get("value") or cols.get("channel") or cols.get("ch") or cols.get("name")
    if xcol is None or ycol is None or vcol is None:
        raise ValueError("montage csv must have columns for x,y and channel index/name")
    df["value"] = values[df[vcol].astype(int)]
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(df[xcol], df[ycol], c=df["value"], cmap="viridis", s=80)
    fig.colorbar(sc, ax=ax, label="importance")
    ax.set_title("Channel importance projected on montage")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    safe_figsave(fig, out_file)


def main():
    parser = argparse.ArgumentParser(description="Channel importance via variance or channel_proj weights")
    parser.add_argument("--data", type=Path, help="npy/npz with EEG data (N x C x T) to compute variance/energy")
    parser.add_argument("--checkpoint", type=Path, help="Model checkpoint (.pt) with channel_proj weights")
    parser.add_argument("--out-dir", type=Path, default=Path("Train/analysis/channel_importance"))
    parser.add_argument("--montage", type=Path, help="CSV with channel,x,y to project importance on montage")
    args = parser.parse_args()

    out_dir = ensure_outdir(args.out_dir)

    if args.data:
        arr = load_array(args.data)
        var = channel_variance(arr)
        np.save(out_dir / "channel_variance.npy", var)
        plot_importance(var, "Channel variance (data)", out_dir / "variance.png")
        if args.montage:
            plot_montage(var, args.montage, out_dir / "variance_montage.png")

    if args.checkpoint:
        weights = extract_channel_weights(args.checkpoint)
        np.save(out_dir / "channel_weights.npy", weights)
        plot_importance(weights, "Channel importance from channel_proj weights", out_dir / "weights.png")
        if args.montage:
            plot_montage(weights, args.montage, out_dir / "weights_montage.png")

    print(f"Saved channel importance to {out_dir}")


if __name__ == "__main__":
    main()
