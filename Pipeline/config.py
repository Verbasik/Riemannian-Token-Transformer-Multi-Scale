# -*- coding: utf-8 -*-
"""
Configuration for the Phase 4B experiment.

This file centralizes:
    - Data paths
    - Model hyperparameters
    - Training settings
    - Cross-validation settings
    - Optimizer and scheduler parameters
"""

# =============================================================================
# Standard library
# =============================================================================
import os
from pathlib import Path
from typing import Any, Dict, Optional

# =============================================================================
# Third-party libraries
# =============================================================================
import torch

# =============================================================================
# Global constants
# =============================================================================

PROJECT_ROOT: Path = Path(__file__).parent.parent
# Project root directory (one level above the config folder)

DATA_ROOT: Path = PROJECT_ROOT
# Base data directory

JSON_DIR: Path = DATA_ROOT / "json"
# Directory with JSON dataset descriptions

PREPROCESSED_DIR: Path = Path("/mnt/data/EEG/preprocessed_pkl")
# Legacy path to preprocessed data

RANDOM_SEED: int = 42
# Fixed seed for reproducibility

EPSILON: float = 1e-4
# Numerical constant for computational stability


# =============================================================================
# Helper function
# =============================================================================
def _resolve_preprocessed_dir() -> Path:
    """
    Description:
    ---------------
        Resolves an existing directory with preprocessed PKL files.

    Returns:
    ---------------
        Path: Resolved data path.
    """
    env_dir: Optional[str] = os.getenv("EEG_PREPROCESSED_DIR")

    candidates = []

    if env_dir:
        candidates.append(Path(env_dir))  # Environment variable priority

    candidates.extend(
        [
            Path("/mnt/data/data/derivatives/preprocessed_pkl"),
            Path("/mnt/data/derivatives/preprocessed_pkl"),
            Path("/mnt/data/EEG/preprocessed_pkl"),
            PROJECT_ROOT / "derivatives" / "preprocessed_pkl",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


# =============================================================================
# Main configuration function
# =============================================================================
def default_config(device_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Description:
    ---------------
        Builds the configuration dictionary for the Phase 4B experiment.

    Args:
    ---------------
        device_hint: Device hint ('cuda' or 'cpu').

    Returns:
    ---------------
        Dict[str, Any]: Experiment configuration.
    """
    use_cuda: bool = torch.cuda.is_available() and (
        (device_hint or "cuda") == "cuda"
    )

    device: str = "cuda" if use_cuda else "cpu"
    data_dir: Path = _resolve_preprocessed_dir()

    return {
        "data": {
            "data_dir": data_dir,
            # Path to preprocessed data

            "subject_ids": [
                "sub-01", "sub-02", "sub-03", "sub-04", "sub-05"
            ],
            # Subject identifiers used for training/evaluation

            "task": "imagine",
            # Task type (for example: imagine / overt / rest)

            "normalize": "zscore_hybrid",
            # Input data normalization method

            "exclude_channels": [124],
            # EEG channel indices excluded from analysis
        },
        "model": {
            "n_channels": 125,
            # Total number of input EEG channels

            "n_classes": 8,
            # Number of target classification classes

            "proj_channels": 24,
            # Projection layer dimensionality before attention

            "window_size_small": 128,
            # Small temporal window size (in samples)

            "stride_small": 96,
            # Small-window stride

            "window_size_large": 256,
            # Large temporal window size

            "stride_large": 128,
            # Large-window stride

            "d_model": 128,
            # Transformer hidden representation dimensionality

            "n_heads": 4,
            # Number of attention heads in the Transformer

            "ff_dim": 256,
            # Feed-forward layer dimensionality

            "n_layers": 2,
            # Number of Transformer blocks

            "dropout": 0.1,
            # Dropout for model regularization

            "eps": EPSILON,
            # Constant for numerical stability

            "attn_heads": 1,
            # Number of attention heads in the covariance module

            "cov_type": "corr",
            # Covariance matrix type ('cov' or 'corr')

            "oas_min_alpha": 0.1,
            # Minimum OAS shrinkage value

            "use_subject_embed": True,
            # Whether to use subject embeddings

            "subject_embed_dim": 16,
            # Subject embedding vector dimensionality

            "subject_embed_dropout": 0.2,
            # Dropout for subject embedding

            "unknown_subject_policy": "auto",
            # Behavior for subject-held-out validation:
            # auto / error / zero / mean
        },
        "training": {
            "n_epochs": 50,
            # Maximum number of training epochs

            "batch_size": 16 if device == "cuda" else 8,
            # Batch size (larger on GPU)

            "learning_rate": 3e-4,
            # Initial learning rate

            "weight_decay": 1e-4,
            # L2 weight regularization

            "early_stopping_patience": 8,
            # Number of epochs without improvement before stopping

            "use_amp": device == "cuda",
            # Whether to use mixed precision

            "grad_clip": 1.0,
            # Gradient norm clipping limit

            "num_workers": 0,
            # Number of DataLoader worker processes.
            # The data is already loaded into memory; on Python 3.14,
            # forkserver tries to pickle the large dataset and may fail.

            "pin_memory": device == "cuda",
            # Whether to pin memory (speeds up GPU transfers)

            "persistent_workers": False,
            # Whether to keep worker processes alive between epochs

            "prefetch_factor": 4 if device == "cuda" else 2,
            # Number of prefetched batches

            "allow_multiprocessing_dataloader": False,
            # Allow num_workers > 0. Disabled by default on Python 3.14
            # because of forkserver and the large in-memory dataset.
        },
        "cv": {
            "protocol": "within_subject",
            # Evaluation protocol:
            # within_subject: each subject is present in train and val
            # subject_heldout: validation subjects are absent from train

            "n_splits": 5,
            # Number of cross-validation folds

            "random_state": RANDOM_SEED,
            # Seed for fold splitting

            "mode": "within_subject",
            # CV mode:
            # within_subject / stratified / stratified_group / loso

            "fold_index": 0,
            # Specific fold index
        },
        "evaluation": {
            "pipeline": "both",
            # Full evaluation pipeline:
            # si: Subject-Independent pooled personalized model
            # sd: Subject-Dependent per-subject models
            # both: run si and sd sequentially

            "si_use_subject_embed": True,
            # SI uses one shared model and subject embeddings.

            "sd_use_subject_embed": False,
            # SD trains a separate model for each subject; subject
            # embedding is disabled by default because personalization
            # is provided by the separate model itself.
        },
        "optimizer": {
            "name": "adamw",
            # Optimizer type

            "betas": [0.9, 0.999],
            # Momentum parameters for AdamW

            "subject_embed_weight_decay": 5e-4,
            # Separate regularization for subject embeddings
        },
        "scheduler": {
            "name": "cosine",
            # Scheduler type

            "T_max": 20,
            # Cosine scheduler period

            "warmup_epochs": 3,
            # Number of warmup epochs
        },
        "loss": {
            "type": "cb_focal",
            # Loss function type

            "beta": 0.999,
            # Class balancing parameter

            "gamma": 1.75,
            # Focusing parameter in Focal Loss
        },
        "logging": {
            "save_attn": False,
            # Whether to save attention weights during validation
        },
        "seed": RANDOM_SEED,
        # Global experiment seed

        "device": device,
        # Compute device ('cpu' or 'cuda')

        "checkpoint_dir": (
            f"Train/checkpoints/phase4b_5subjects_{device.upper()}"
        ),
        # Checkpoint output directory

        "results_dir": (
            f"Train/results/phase4b_5subjects_{device.upper()}"
        ),
        # Experiment results output directory
    }
