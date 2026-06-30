# file: trainer.py
# -*- coding: utf-8 -*-
"""
Training/evaluation loops and artifact logging.

The module contains:
1. Class-Balanced Focal Loss implementation for class imbalance.
2. Metric computation functions (Accuracy, F1, Precision, Recall).
3. Model evaluation loop with optional prediction and attention-stat collection.
4. Main training loop with AMP, gradient clipping, and Early Stopping support.
5. Utilities for saving artifacts (weights, metrics, history, config).
"""

# =============================================================================
# Standard Libraries
# =============================================================================
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Third-Party Libraries
# =============================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

# =============================================================================
# Local Imports
# =============================================================================
from utils import print_metrics


# =============================================================================
# Serialization and device-check utilities
# =============================================================================

def _to_serializable(obj: Any) -> Any:
    """
    Description:
    ---------------
        Recursively converts an object to a JSON-compatible format.
        Handles specific types: Path, torch.device, NumPy scalars,
        dictionaries, and lists.

    Args:
    ---------------
        obj: Any - Object to convert.

    Returns:
    ---------------
        Any: Serializable object (str, int, float, list, dict).

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> _to_serializable(Path('/tmp'))
        '/tmp'
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _is_cuda_device(device: Any) -> bool:
    """
    Description:
    ---------------
        Normalizes CUDA device checks. Works with both torch.device
        objects and strings.

    Args:
    ---------------
        device: Any - Device (torch.device or str).

    Returns:
    ---------------
        bool: True if the device is CUDA, otherwise False.

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> _is_cuda_device(torch.device('cuda:0'))
        True
    """
    if isinstance(device, torch.device):
        return device.type == 'cuda'
    return str(device) == 'cuda'


# =============================================================================
# Loss functions and metrics
# =============================================================================

class ClassBalancedFocalLoss(nn.Module):
    """
    Description:
    ---------------
        Class-Balanced Focal Loss.
        Combines class weighting (Class-Balanced) to handle imbalance
        and focusing on hard examples (Focal Loss).

        Formula:
        CB_weight = (1 - beta) / (1 - beta^n)
        FL_weight = (1 - p_t)^gamma
        Loss = - CB_weight * FL_weight * log(p_t)

    Args:
    ---------------
        class_counts: np.ndarray - Number of examples per class.
        beta: float - Effective-number parameter (usually 0.9999).
        gamma: float - Focusing parameter (usually 2.0).

    Returns:
    ---------------
        Tensor: Scalar loss value.

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> loss_fn = ClassBalancedFocalLoss(np.array([100, 10]), beta=0.99, gamma=2.0)
        >>> logits = torch.randn(10, 2)
        >>> targets = torch.randint(0, 2, (10,))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        class_counts: np.ndarray,
        beta: float,
        gamma: float
    ):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)

        # Compute the effective number of examples.
        effective_num = 1.0 - torch.pow(beta, counts)

        # Compute class weights (CB weights).
        alpha = (1.0 - beta) / effective_num.clamp(min=1e-8)

        # Normalize weights so their sum equals the number of classes.
        alpha = alpha / alpha.sum() * len(counts)

        self.register_buffer('alpha', alpha)
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Description:
        ---------------
            Computes the loss value.

        Args:
        ---------------
            logits: torch.Tensor [B, C] - Model logits.
            targets: torch.Tensor [B] - True class indices.

        Returns:
        ---------------
            torch.Tensor: Scalar loss value.
        """
        probs = F.softmax(logits, dim=-1)

        # Extract the correct-class probability (p_t).
        pt = torch.gather(
            probs,
            -1,
            targets.unsqueeze(-1)
        ).squeeze(-1)

        # Select the class weight for each example.
        alpha_t = self.alpha[targets]

        # Compute the focusing weight (1 - p_t)^gamma.
        focal_weight = torch.pow(1.0 - pt, self.gamma)

        # Final loss formula.
        loss = -alpha_t * focal_weight * torch.log(pt.clamp(min=1e-8))

        return loss.mean()


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str
) -> Dict[str, float]:
    """
    Description:
    ---------------
        Computes a set of classification metrics.

    Args:
    ---------------
        y_true: np.ndarray - True labels.
        y_pred: np.ndarray - Predicted labels.
        average: str - Averaging strategy ('macro', 'weighted', etc.).

    Returns:
    ---------------
        Dict[str, float]: Dictionary with metrics (accuracy, f1, precision, recall,
            balanced_accuracy).

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> metrics = compute_metrics(np.array([0, 1]), np.array([0, 1]), 'macro')
        >>> 'accuracy' in metrics
        True
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        f'f1_{average}': f1_score(
            y_true, y_pred, average=average, zero_division=0
        ),
        f'precision_{average}': precision_score(
            y_true, y_pred, average=average, zero_division=0
        ),
        f'recall_{average}': recall_score(
            y_true, y_pred, average=average, zero_division=0
        ),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
    }


# =============================================================================
# Model evaluation
# =============================================================================

@torch.no_grad()
def evaluate_with_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
    collect_outputs: bool = False,
    collect_attn: bool = False,
) -> Tuple[
    Dict[str, float],
    Optional[Dict[str, np.ndarray]],
    Optional[Dict[str, np.ndarray]]
]:
    """
    Description:
    ---------------
        Evaluates the model on the validation split. Optionally collects
        predictions (probabilities, labels) and attention mechanism
        statistics (attention weights) for subsequent interpretability
        analysis.

    Args:
    ---------------
        model: nn.Module - Model to evaluate.
        loader: DataLoader - Data loader.
        device: str - Compute device.
        criterion: nn.Module - Loss function (for val loss calculation).
        collect_outputs: bool - Whether to collect full predictions.
        collect_attn: bool - Whether to collect attention statistics.

    Returns:
    ---------------
        Tuple[Dict, Optional[Dict], Optional[Dict]]:
            - metrics: Metrics dictionary.
            - outputs: Dictionary with prediction arrays (if collect_outputs).
            - attn_stats: Dictionary with mean attention weights (if collect_attn).

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> metrics, outs, attn = evaluate_with_outputs(model, loader, 'cpu', criterion)
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_subj: List[np.ndarray] = []
    all_sample_ids: List[np.ndarray] = []

    attn_sum: Optional[torch.Tensor] = None
    attn_count = 0
    attn_meta: Dict[str, Any] = {}

    use_subject_embed = getattr(model, 'use_subject_embed', False)
    non_blocking = _is_cuda_device(device)

    for batch in loader:
        eeg = batch['eeg'].to(device, non_blocking=non_blocking)
        labels = batch['label'].to(device, non_blocking=non_blocking)

        subject_ids = None
        if use_subject_embed:
            subject_ids = batch['subject_id'].to(device, non_blocking=non_blocking)

        sample_ids = batch.get('sample_id')

        # Forward pass.
        out = model(
            eeg,
            subject_ids=subject_ids,
            return_attn=collect_attn
        )

        if collect_attn:
            logits, attn_stats = out
            if attn_stats is not None:
                # Average attention weights across batches.
                # attn_stats['weights_tok_mean']: [L, H]
                w = attn_stats['weights_tok_mean']
                if attn_sum is None:
                    attn_sum = w.detach().cpu()
                else:
                    attn_sum += w.detach().cpu()
                attn_count += 1

                # Store meta-information (batch-independent).
                attn_meta = {
                    'head_weights': (
                        attn_stats['head_weights'].detach().cpu().numpy()
                    ),
                    'scale_lengths': attn_stats['scale_lengths'],
                }
        else:
            logits = out

        probs = F.softmax(logits, dim=-1)
        total_loss += criterion(logits, labels).item()
        preds = logits.argmax(-1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        if collect_outputs:
            all_probs.append(probs.cpu().numpy())
            subj_np = (
                subject_ids.cpu().numpy()
                if subject_ids is not None
                else np.zeros_like(preds.cpu().numpy())
            )
            all_subj.append(subj_np)

            if sample_ids is not None:
                # sample_id may be a tensor or a list.
                sid_np = np.array(sample_ids)
                all_sample_ids.append(sid_np)

    # Concatenate results.
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    metrics = compute_metrics(y_true, y_pred, average='macro')
    metrics['loss'] = total_loss / len(loader)

    # Build outputs dictionary.
    outputs: Optional[Dict[str, np.ndarray]] = None
    if collect_outputs:
        outputs = {
            'y_true': y_true,
            'y_pred': y_pred,
            'proba': (
                np.concatenate(all_probs) if all_probs else None
            ),
            'subject_id': (
                np.concatenate(all_subj) if all_subj else None
            ),
            'sample_id': (
                np.concatenate(all_sample_ids) if all_sample_ids else None
            ),
        }

    # Build attention statistics.
    attn_stats_out: Optional[Dict[str, np.ndarray]] = None
    if collect_attn and attn_sum is not None and attn_count > 0:
        attn_stats_out = {
            'weights_tok_mean': (attn_sum / attn_count).numpy(),
            'head_weights': attn_meta['head_weights'],
            'scale_lengths': attn_meta['scale_lengths'],
        }

    return metrics, outputs, attn_stats_out


# =============================================================================
# Training loop
# =============================================================================

def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    cfg: Dict[str, Any],
    device: str
) -> Tuple[
    Dict[str, List[float]],
    Dict[str, float],
    Optional[Dict[str, np.ndarray]],
    Optional[Dict[str, np.ndarray]]
]:
    """
    Description:
    ---------------
        Main model training loop.
        Supports:
        - Automatic Mixed Precision (AMP) for GPU acceleration.
        - Gradient clipping for stability.
        - Early Stopping by F1-macro.
        - Training history and gradient norm logging.
        - Best weight restoration at the end of training.

    Args:
    ---------------
        model: nn.Module - Model to train.
        train_loader: DataLoader - Training data loader.
        val_loader: DataLoader - Validation data loader.
        criterion: nn.Module - Loss function.
        optimizer: Optimizer - Optimizer.
        scheduler: Any - Learning-rate scheduler.
        cfg: Dict[str, Any] - Training configuration.
        device: str - Device.

    Returns:
    ---------------
        Tuple[Dict, Dict, Optional[Dict], Optional[Dict]]:
            - history: Per-epoch metric history.
            - final_metrics: Final validation metrics.
            - final_outputs: Validation predictions.
            - final_attn: Attention statistics.

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> history, metrics, outs, attn = train_loop(...)
    """
    use_cuda = _is_cuda_device(device)
    use_amp = cfg['training']['use_amp'] and use_cuda

    # GradScaler for AMP.
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_f1 = 0.0
    patience = 0

    history: Dict[str, List[float]] = {
        'train_loss': [],
        'val_loss': [],
        'val_f1_macro': [],
        'lr': [],
        'grad_norm_min': [],
        'grad_norm_mean': [],
        'grad_norm_max': []
    }

    n_epochs = cfg['training']['n_epochs']
    grad_clip = cfg['training']['grad_clip']

    print(f"\nStarting training for {n_epochs} epochs...")
    print(f"Device: {device}, AMP: {use_amp}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    use_subject_embed = getattr(model, 'use_subject_embed', False)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        grad_norms: List[float] = []

        for batch in train_loader:
            eeg = batch['eeg'].to(device, non_blocking=use_cuda)
            labels = batch['label'].to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass with AMP.
            with torch.cuda.amp.autocast(enabled=use_amp):
                if use_subject_embed:
                    subject_ids = batch['subject_id'].to(
                        device, non_blocking=use_cuda
                    )
                    logits = model(eeg, subject_ids=subject_ids)
                else:
                    logits = model(eeg)
                loss = criterion(logits, labels)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Compute the gradient norm before clipping (for monitoring).
            total_norm_sq = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm_sq += param_norm * param_norm

            grad_norm = total_norm_sq ** 0.5 if total_norm_sq > 0 else 0.0
            grad_norms.append(grad_norm)

            # Gradient clipping.
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Optimizer step.
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # Validation.
        val_metrics, _, _ = evaluate_with_outputs(
            model,
            val_loader,
            device,
            criterion,
            collect_outputs=False,
            collect_attn=False
        )

        if scheduler:
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']

        # Update history.
        history['train_loss'].append(total_loss / len(train_loader))
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['lr'].append(lr)

        if grad_norms:
            history['grad_norm_min'].append(float(np.min(grad_norms)))
            history['grad_norm_mean'].append(float(np.mean(grad_norms)))
            history['grad_norm_max'].append(float(np.max(grad_norms)))
        else:
            history['grad_norm_min'].append(0.0)
            history['grad_norm_mean'].append(0.0)
            history['grad_norm_max'].append(0.0)

        # Log epoch.
        print(f"\nEpoch {epoch+1}/{n_epochs} | LR: {lr:.6f}")
        print(f"  Train Loss: {history['train_loss'][-1]:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Acc:    {val_metrics['accuracy']:.4f} | "
              f"Val F1:   {val_metrics['f1_macro']:.4f}")

        # Early stopping logic.
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            # Save model state on CPU.
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience = 0
            print(f"  ✅ New best F1: {best_f1:.4f}. Model saved.")
        else:
            patience += 1
            if patience >= cfg['training']['early_stopping_patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Restore best weights.
    if best_state:
        model.load_state_dict(best_state)
        print(f"\n✅ Restored best model (F1: {best_f1:.4f})")

    # Final evaluation with artifact collection.
    final_metrics, final_outputs, final_attn = evaluate_with_outputs(
        model,
        val_loader,
        device,
        criterion,
        collect_outputs=True,
        collect_attn=bool(
            cfg.get('logging', {}).get('save_attn', False)
        )
    )

    return history, final_metrics, final_outputs, final_attn


# =============================================================================
# Artifact saving
# =============================================================================

def save_artifacts(
    cfg: Dict[str, Any],
    metrics: Dict[str, float],
    history: Dict[str, List[float]],
    val_outputs: Optional[Dict[str, np.ndarray]],
    attn_stats: Optional[Dict[str, np.ndarray]],
    model: nn.Module
) -> None:
    """
    Description:
    ---------------
        Saves all experiment artifacts:
        - Best model weights (.pt).
        - Metrics and training history (.json).
        - Validation predictions (.npz).
        - Attention statistics (.npz).
        - Full run configuration (.json).

    Args:
    ---------------
        cfg: Dict[str, Any] - Configuration.
        metrics: Dict[str, float] - Final metrics.
        history: Dict[str, List[float]] - Training history.
        val_outputs: Optional[Dict] - Predictions.
        attn_stats: Optional[Dict] - Attention statistics.
        model: nn.Module - Model (for saving weights).

    Returns:
    ---------------
        None

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> save_artifacts(cfg, metrics, history, outs, attn, model)
    """
    ckpt_dir = Path(cfg['checkpoint_dir'])
    res_dir = Path(cfg['results_dir'])

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights.
    torch.save(model.state_dict(), ckpt_dir / 'best_model.pt')

    # Process outputs and per-class metrics.
    if val_outputs is not None:
        if (val_outputs.get('y_true') is not None and
                val_outputs.get('y_pred') is not None):
            prec, rec, f1, support = precision_recall_fscore_support(
                val_outputs['y_true'],
                val_outputs['y_pred'],
                average=None,
                zero_division=0
            )
            metrics['per_class'] = [
                {
                    'precision': float(p),
                    'recall': float(r),
                    'f1': float(f),
                    'support': int(s)
                }
                for p, r, f, s in zip(prec, rec, f1, support)
            ]

        # Save predictions.
        np.savez(
            res_dir / 'val_preds.npz',
            **{k: v for k, v in val_outputs.items() if v is not None}
        )

    # Save attention statistics.
    if attn_stats is not None:
        np.savez(res_dir / 'attn_stats.npz', **attn_stats)

    # Save metrics.
    with open(
        res_dir / 'metrics.json', 'w', encoding='utf-8'
    ) as f:
        json.dump(
            _to_serializable(metrics),
            f,
            indent=2,
            ensure_ascii=False
        )

    # Save history.
    with open(
        res_dir / 'history.json', 'w', encoding='utf-8'
    ) as f:
        json.dump(
            _to_serializable(history),
            f,
            indent=2,
            ensure_ascii=False
        )

    # Save config for reproducibility.
    with open(
        res_dir / 'config_run.json', 'w', encoding='utf-8'
    ) as f:
        json.dump(
            _to_serializable(cfg),
            f,
            indent=2,
            ensure_ascii=False
        )

    # Generate visualizations (plots + metric tables).
    try:
        from visualization import save_single_run_plots
        save_single_run_plots(
            history=history,
            val_outputs=val_outputs,
            attn_stats=attn_stats,
            res_dir=res_dir,
        )
    except Exception as exc:
        print(f'[viz] Visualization skipped: {exc}')

    print(f"Artifacts saved to {ckpt_dir} and {res_dir}")
