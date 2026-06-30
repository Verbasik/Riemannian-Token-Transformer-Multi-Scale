# file: utils.py
# -*- coding: utf-8 -*-
"""
Shared helper utilities.

Contains functions that are not tied to a specific domain, such as:
1. Setting seeds for experiment reproducibility (PyTorch, NumPy, CUDA).
2. Ensuring NumPy version compatibility when deserializing pickle files.
3. Formatted console output for metrics and run configuration.

These utilities are used across all training and testing scripts.
"""

# =============================================================================
# Standard Libraries
# =============================================================================
import sys
from typing import Any, Dict

# =============================================================================
# Third-Party Libraries
# =============================================================================
import numpy as np
import torch


def _ensure_numpy_pickle_compat() -> None:
    """
    Description:
    ---------------
        Ensures compatibility when deserializing NumPy pickle files
        created by older library versions.
        
        Problem: In newer NumPy versions (>=1.25), the `numpy.core`
        module was moved to `numpy._core`. Older pickle files may still
        reference the old path, which causes loading failures.
        
        Solution: Create an alias in `sys.modules` so the new module can
        be found by its old name.

    Args:
    ---------------
        No arguments.

    Returns:
    ---------------
        None

    Raises:
    ---------------
        No explicit exceptions (import errors are ignored).

    Examples:
    ---------------
        >>> _ensure_numpy_pickle_compat()
        >>> # Old pickle files with arrays can now be loaded safely.
    """
    try:
        # Try the new path first (NumPy >= 1.25).
        import numpy._core  # noqa: F401
    except ImportError:
        try:
            # If the new path is missing, try the old one and create an alias.
            import numpy.core as ncore
            sys.modules['numpy._core'] = ncore
        except ImportError:
            # If neither path exists, skip it (possibly a very old version).
            pass


def set_seed(seed: int) -> None:
    """
    Description:
    ---------------
        Sets a fixed seed for all random number generators in the
        project. This is critical for experiment reproducibility.
        
        Initializes:
        - PyTorch CPU RNG.
        - PyTorch CUDA RNG (for all GPUs).
        - NumPy RNG.
        - cuDNN settings for deterministic algorithms.

    Args:
    ---------------
        seed: int - Integer seed value.

    Returns:
    ---------------
        None

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> set_seed(42)
        >>> # All subsequent random operations will be deterministic.
    """
    torch.manual_seed(seed)
    
    # Set the seed for all GPUs.
    torch.cuda.manual_seed_all(seed)
    
    # Set the NumPy seed.
    np.random.seed(seed)
    
    # Configure cuDNN for determinism.
    # deterministic=True guarantees identical results, but may be slower.
    torch.backends.cudnn.deterministic = True
    
    # benchmark=False disables the search for the optimal convolution
    # algorithm, which may be nondeterministic.
    torch.backends.cudnn.benchmark = False


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Description:
    ---------------
        Prints a metrics dictionary to the console in "key: value" format.
        Values are formatted to 4 decimal places.

    Args:
    ---------------
        metrics: Dict[str, float] - Metrics dictionary (for example,
            {'accuracy': 0.95}).
        prefix: str - Prefix added to the output header (optional).

    Returns:
    ---------------
        None

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> metrics = {'loss': 0.12345, 'acc': 0.98765}
        >>> print_metrics(metrics, prefix="Val")
        Val Metrics:
          loss: 0.1235
          acc: 0.9877
    """
    print(f"\n{prefix} Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


def pretty_print_run(cfg: Dict[str, Any]) -> None:
    """
    Description:
    ---------------
        Prints key run configuration parameters in a readable format.
        Used for a quick check of experiment settings before startup.
        Groups parameters by category: data, model, training, optimization.

    Args:
    ---------------
        cfg: Dict[str, Any] - Full project configuration dictionary.

    Returns:
    ---------------
        None

    Raises:
    ---------------
        KeyError: If the configuration is missing expected keys
            ('data', 'model', etc.).

    Examples:
    ---------------
        >>> cfg = default_config()
        >>> pretty_print_run(cfg)
        ==================================================
        Phase 4B run configuration
        ==================================================
        Data: 5 subjects, normalization: zscore_hybrid
        ...
    """
    # Extract the main config sections for brevity.
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    train_cfg = cfg['training']
    sched_cfg = cfg['scheduler']
    opt_cfg = cfg['optimizer']
    loss_cfg = cfg['loss']

    print("\n" + "=" * 50)
    print("Phase 4B run configuration")
    print("=" * 50)
    
    # Data block.
    n_subjects = len(data_cfg['subject_ids'])
    norm_type = data_cfg['normalize']
    print(f"Data: {n_subjects} subjects, normalization: {norm_type}")
    
    # Model block.
    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    print(f"Model: rtt_ms, d_model={d_model}, layers={n_layers}")
    
    # Training block.
    n_epochs = train_cfg['n_epochs']
    batch_size = train_cfg['batch_size']
    lr = train_cfg['learning_rate']
    print(
        f"Training: {n_epochs} epochs, batch_size={batch_size}, "
        f"lr={lr:.0e}"
    )
    
    # Optimization block.
    sch_name = sched_cfg['name']
    opt_name = opt_cfg['name']
    loss_type = loss_cfg['type']
    print(
        f"Scheduler: {sch_name}, Optimizer: {opt_name}, "
        f"Loss: {loss_type}"
    )
    
    print("=" * 50)
