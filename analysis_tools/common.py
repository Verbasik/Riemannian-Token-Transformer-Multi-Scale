"""
Utility helpers for analysis scripts that work with training artifacts.
Dependencies: numpy, pandas, matplotlib, sklearn (used by some scripts).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_experiments(results_root: Path) -> List[Path]:
    if not results_root.exists():
        return []
    return sorted([p for p in results_root.iterdir() if p.is_dir()])


def load_metrics(exp_dir: Path) -> Dict:
    metrics_path = exp_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {exp_dir}")
    with metrics_path.open() as f:
        return json.load(f)


def load_history(exp_dir: Path) -> Dict:
    hist_path = exp_dir / "history.json"
    if not hist_path.exists():
        raise FileNotFoundError(f"history.json not found in {exp_dir}")
    with hist_path.open() as f:
        return json.load(f)


def load_config(exp_dir: Path) -> Dict:
    cfg_path = exp_dir / "config_run.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config_run.json not found in {exp_dir}")
    with cfg_path.open() as f:
        return json.load(f)


def load_val_preds(exp_dir: Path) -> Dict[str, np.ndarray]:
    npz_path = exp_dir / "val_preds.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"val_preds.npz not found in {exp_dir}")
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def optional_import_umap():
    try:
        import umap

        return umap
    except Exception:
        return None


def safe_figsave(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.clf()
