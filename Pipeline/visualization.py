# file: visualization.py
# -*- coding: utf-8 -*-
"""
Visualization module for EEG_to_Text pipeline.

Generates publication-quality figures and metric tables from training
history, validation predictions, attention statistics, and K-fold
cross-validation results.

High-level entry points
-----------------------
    save_single_run_plots(history, val_outputs, attn_stats, res_dir)
        Called automatically by save_artifacts() in trainer.py after
        each training run.

    save_full_eval_plots(results, analysis, res_dir)
        Called automatically after run_full_evaluation.py finishes all
        25 experiments and statistical analysis.

Individual plotting functions can also be called directly for custom use.

Output layout
-------------
    <res_dir>/
    ├── plots/
    │   ├── training_curves.png
    │   ├── confusion_matrix.png
    │   ├── per_class_metrics.png
    │   ├── roc_curves.png
    │   ├── precision_recall_curves.png
    │   ├── attention_heatmap.png        (if attn_stats available)
    │   ├── head_importance.png          (if attn_stats available)
    │   ├── cv_boxplot.png               (full eval only)
    │   ├── subject_fold_heatmap.png     (full eval only)
    │   ├── bootstrap_ci.png             (full eval only)
    │   └── overall_metrics_bar.png      (full eval only)
    └── tables/
        ├── metrics_summary.csv
        ├── per_class_metrics.csv
        ├── full_eval_summary.csv        (full eval only)
        ├── per_subject_metrics.csv      (full eval only)
        └── metrics_report.md            (full eval only)
"""

# ===========================================================================
# NOTE: 'Agg' backend must be set before any pyplot import.
# It is a non-interactive raster backend safe for headless GPU servers.
# ===========================================================================
import matplotlib
matplotlib.use('Agg')

import csv
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore', category=UserWarning)

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False


# ===========================================================================
# Module-level constants
# ===========================================================================

_DPI: int = 300
_FMT: str = 'png'

_DEFAULT_CLASS_NAMES: List[str] = [
    'Class 0', 'Class 1', 'Class 2', 'Class 3',
    'Class 4', 'Class 5', 'Class 6', 'Class 7',
]

_PALETTE: Tuple = plt.cm.tab10.colors   # 10 distinct colours (tab10)


# ===========================================================================
# Style setup (runs once on import)
# ===========================================================================

def _configure_style() -> None:
    if _HAS_SEABORN:
        sns.set_theme(
            style='whitegrid',
            palette='tab10',
            rc={
                'axes.spines.top': False,
                'axes.spines.right': False,
                'font.size': 11,
            },
        )
    else:
        plt.rcParams.update({
            'axes.grid': True,
            'grid.alpha': 0.35,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'font.size': 11,
        })


_configure_style()


# ===========================================================================
# Private helpers
# ===========================================================================

def _save(fig: plt.Figure, path: Path) -> None:
    """Tight-layout → save → close → log path."""
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'    [viz] {path.name}')


def _plots_dir(base: Union[str, Path]) -> Path:
    p = Path(base) / 'plots'
    p.mkdir(parents=True, exist_ok=True)
    return p


def _tables_dir(base: Union[str, Path]) -> Path:
    p = Path(base) / 'tables'
    p.mkdir(parents=True, exist_ok=True)
    return p


def _class_names_for(n: int) -> List[str]:
    if n <= len(_DEFAULT_CLASS_NAMES):
        return _DEFAULT_CLASS_NAMES[:n]
    return [f'C{i}' for i in range(n)]


def _best_epoch(history: Dict[str, List[float]]) -> int:
    """Return 0-based epoch index of best val_f1_macro. -1 if empty."""
    vals = history.get('val_f1_macro', [])
    return int(np.argmax(vals)) if vals else -1


# ===========================================================================
# Section 1 — Training dynamics
# ===========================================================================

def plot_training_curves(
    history: Dict[str, List[float]],
    save_dir: Union[str, Path],
) -> None:
    """
    Description:
    ---------------
        Saves a 2×2 dashboard of training dynamics:
        [0,0] Train vs Val Loss   [0,1] Val F1-macro
        [1,0] Learning Rate       [1,1] Gradient Norm (mean ± min/max)

        A vertical dashed line marks the best-F1 epoch on every subplot.

    Args:
    ---------------
        history: Dict returned by train_loop() with keys:
            'train_loss', 'val_loss', 'val_f1_macro', 'lr',
            'grad_norm_min', 'grad_norm_mean', 'grad_norm_max'.
        save_dir: Directory where 'training_curves.png' is written.
    """
    epochs_list = history.get('train_loss', [])
    if not epochs_list:
        return

    n_epochs = len(epochs_list)
    xs = np.arange(1, n_epochs + 1)
    best = _best_epoch(history) + 1   # convert to 1-based for display

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Training Dynamics', fontsize=14, fontweight='bold')

    # ------------------------------------------------------------------
    # [0, 0]  Loss curves
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(xs, history['train_loss'], color=_PALETTE[0],
            linewidth=1.8, label='Train Loss')
    ax.plot(xs, history['val_loss'], color=_PALETTE[1],
            linewidth=1.8, linestyle='--', label='Val Loss')
    if best > 0:
        ax.axvline(best, color='grey', linewidth=1.0,
                   linestyle=':', alpha=0.7, label=f'Best epoch ({best})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train / Validation Loss')
    ax.legend(fontsize=9)

    # ------------------------------------------------------------------
    # [0, 1]  Val F1-macro
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    f1s = history.get('val_f1_macro', [])
    if f1s:
        ax.plot(xs, f1s, color=_PALETTE[2], linewidth=1.8)
        if best > 0:
            best_f1 = f1s[best - 1]
            ax.axvline(best, color='grey', linewidth=1.0,
                       linestyle=':', alpha=0.7)
            ax.scatter([best], [best_f1], color=_PALETTE[3],
                       zorder=5, s=60,
                       label=f'Best F1 = {best_f1:.4f}')
            ax.legend(fontsize=9)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Macro')
        ax.set_title('Validation F1-Macro')
        ax.set_ylim(bottom=0.0)

    # ------------------------------------------------------------------
    # [1, 0]  Learning rate
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    lrs = history.get('lr', [])
    if lrs:
        ax.plot(xs, lrs, color=_PALETTE[4], linewidth=1.8)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{v:.2e}')
        )
        if best > 0:
            ax.axvline(best, color='grey', linewidth=1.0,
                       linestyle=':', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate (log)')
        ax.set_title('LR Schedule (Warmup + Cosine)')

    # ------------------------------------------------------------------
    # [1, 1]  Gradient norms
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    g_mean = history.get('grad_norm_mean', [])
    g_min = history.get('grad_norm_min', [])
    g_max = history.get('grad_norm_max', [])
    if g_mean:
        ax.plot(xs, g_mean, color=_PALETTE[5],
                linewidth=1.8, label='Mean Grad Norm')
        if g_min and g_max:
            ax.fill_between(xs, g_min, g_max,
                            alpha=0.2, color=_PALETTE[5],
                            label='Min–Max range')
        ax.axhline(1.0, color='red', linewidth=1.0,
                   linestyle='--', alpha=0.6, label='Clip = 1.0')
        if best > 0:
            ax.axvline(best, color='grey', linewidth=1.0,
                       linestyle=':', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient L2 Norm')
        ax.set_title('Gradient Norm Evolution')
        ax.legend(fontsize=9)

    out = _plots_dir(save_dir) / f'training_curves.{_FMT}'
    _save(fig, out)


# ===========================================================================
# Section 2 — Classification results
# ===========================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: Union[str, Path],
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = 'Confusion Matrix',
    fname: str = 'confusion_matrix',
) -> None:
    """
    Description:
    ---------------
        Saves a normalised (row-wise) confusion matrix heatmap.
        Cell values show recall per class; raw counts appear in
        parentheses when normalize=True.

    Args:
    ---------------
        y_true: Ground-truth integer labels [N].
        y_pred: Predicted integer labels [N].
        save_dir: Output directory.
        class_names: Optional list of class label strings.
        normalize: If True, normalise rows to [0, 1] (recall per class).
        title: Figure title string.
        fname: Filename stem (without extension).
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    names = class_names if class_names else _class_names_for(n)

    cm_raw = confusion_matrix(y_true, y_pred, labels=classes)

    if normalize:
        row_sums = cm_raw.sum(axis=1, keepdims=True).astype(float)
        cm_plot = np.where(row_sums > 0, cm_raw / row_sums, 0.0)
        fmt_str = '.2f'
    else:
        cm_plot = cm_raw
        fmt_str = 'd'

    fig_sz = max(6, n * 0.9)
    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz * 0.9))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    if _HAS_SEABORN:
        sns.heatmap(
            cm_plot,
            annot=True,
            fmt=fmt_str,
            cmap='Blues',
            xticklabels=names[:n],
            yticklabels=names[:n],
            linewidths=0.4,
            linecolor='white',
            ax=ax,
            vmin=0.0,
            vmax=1.0 if normalize else None,
            annot_kws={'size': max(7, 11 - n // 2)},
        )
    else:
        im = ax.imshow(cm_plot, interpolation='nearest', cmap='Blues',
                       vmin=0.0, vmax=1.0 if normalize else None)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        tick_marks = np.arange(n)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(names[:n], rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(names[:n], fontsize=9)
        thresh = cm_plot.max() / 2.0
        for i in range(n):
            for j in range(n):
                val = cm_plot[i, j]
                txt = f'{val:{fmt_str}}'
                ax.text(j, i, txt, ha='center', va='center',
                        color='white' if val > thresh else 'black',
                        fontsize=max(7, 11 - n // 2))

    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)

    if normalize:
        # Add raw counts as secondary annotation below fraction
        for i in range(n):
            for j in range(n):
                raw = cm_raw[i, j]
                ax.text(
                    j + 0.5 if not _HAS_SEABORN else j,
                    i + 0.35 if not _HAS_SEABORN else i + 0.35,
                    f'({raw})',
                    ha='center', va='center',
                    fontsize=max(5, 8 - n // 2),
                    color='grey',
                )

    out = _plots_dir(save_dir) / f'{fname}.{_FMT}'
    _save(fig, out)


def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: Union[str, Path],
    class_names: Optional[List[str]] = None,
    fname: str = 'per_class_metrics',
) -> None:
    """
    Description:
    ---------------
        Saves a grouped bar chart with Precision, Recall, and F1
        for each class, plus a horizontal line at the macro average.

    Args:
    ---------------
        y_true: Ground-truth labels [N].
        y_pred: Predicted labels [N].
        save_dir: Output directory.
        class_names: Optional list of class label strings.
        fname: Filename stem.
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    names = class_names if class_names else _class_names_for(n)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    macro_f1 = float(np.mean(f1))

    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 5))
    fig.suptitle('Per-Class Classification Metrics', fontsize=13, fontweight='bold')

    ax.bar(x - width, prec, width, label='Precision',
           color=_PALETTE[0], alpha=0.85)
    ax.bar(x,         rec,  width, label='Recall',
           color=_PALETTE[1], alpha=0.85)
    ax.bar(x + width, f1,   width, label='F1',
           color=_PALETTE[2], alpha=0.85)

    ax.axhline(macro_f1, color='black', linewidth=1.2,
               linestyle='--', label=f'Macro F1 = {macro_f1:.3f}')

    ax.set_xticks(x)
    ax.set_xticklabels(names[:n], rotation=30, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')

    # Annotate support counts above each group
    for i, sup in enumerate(support):
        ax.text(x[i], 1.02, f'n={sup}', ha='center',
                fontsize=8, color='grey')

    ax.legend(fontsize=9, loc='lower right')

    out = _plots_dir(save_dir) / f'{fname}.{_FMT}'
    _save(fig, out)


def plot_roc_curves(
    y_true: np.ndarray,
    proba: np.ndarray,
    save_dir: Union[str, Path],
    class_names: Optional[List[str]] = None,
    fname: str = 'roc_curves',
) -> None:
    """
    Description:
    ---------------
        Saves One-vs-Rest ROC curves for each class plus the
        macro-average ROC. Requires probability estimates (proba).

    Args:
    ---------------
        y_true: Ground-truth integer labels [N].
        proba:  Softmax probabilities [N, C].
        save_dir: Output directory.
        class_names: Optional class label strings.
        fname: Filename stem.
    """
    classes = np.unique(y_true)
    n = len(classes)
    names = class_names if class_names else _class_names_for(n)

    y_bin = label_binarize(y_true, classes=list(range(proba.shape[1])))
    if y_bin.ndim == 1:          # binary fallback
        y_bin = np.column_stack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle('ROC Curves (One-vs-Rest)', fontsize=13, fontweight='bold')

    mean_fpr = np.linspace(0, 1, 300)
    tpr_list: List[np.ndarray] = []

    for idx, cls in enumerate(classes):
        if cls >= y_bin.shape[1]:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, cls], proba[:, cls])
        roc_auc = auc(fpr, tpr)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_list.append(tpr_interp)
        ax.plot(fpr, tpr, linewidth=1.3, alpha=0.7,
                color=_PALETTE[idx % 10],
                label=f'{names[idx]} (AUC={roc_auc:.3f})')

    # Macro-average ROC
    if tpr_list:
        mean_tpr = np.mean(tpr_list, axis=0)
        mean_tpr[-1] = 1.0
        macro_auc = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, color='black',
                linewidth=2.2, linestyle='--',
                label=f'Macro avg (AUC={macro_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k:', linewidth=1.0, alpha=0.5, label='Chance')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=8, loc='lower right',
              ncol=2 if n > 5 else 1)

    out = _plots_dir(save_dir) / f'{fname}.{_FMT}'
    _save(fig, out)


def plot_precision_recall_curves(
    y_true: np.ndarray,
    proba: np.ndarray,
    save_dir: Union[str, Path],
    class_names: Optional[List[str]] = None,
    fname: str = 'precision_recall_curves',
) -> None:
    """
    Description:
    ---------------
        Saves One-vs-Rest Precision-Recall curves for each class.
        Especially informative under class imbalance (CB-Focal loss).

    Args:
    ---------------
        y_true: Ground-truth integer labels [N].
        proba:  Softmax probabilities [N, C].
        save_dir: Output directory.
        class_names: Optional class label strings.
        fname: Filename stem.
    """
    classes = np.unique(y_true)
    n = len(classes)
    names = class_names if class_names else _class_names_for(n)

    y_bin = label_binarize(y_true, classes=list(range(proba.shape[1])))
    if y_bin.ndim == 1:
        y_bin = np.column_stack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle('Precision–Recall Curves (One-vs-Rest)',
                 fontsize=13, fontweight='bold')

    ap_scores: List[float] = []
    for idx, cls in enumerate(classes):
        if cls >= y_bin.shape[1]:
            continue
        prec_c, rec_c, _ = precision_recall_curve(
            y_bin[:, cls], proba[:, cls]
        )
        ap = average_precision_score(y_bin[:, cls], proba[:, cls])
        ap_scores.append(ap)
        ax.plot(rec_c, prec_c, linewidth=1.3, alpha=0.75,
                color=_PALETTE[idx % 10],
                label=f'{names[idx]} (AP={ap:.3f})')

    if ap_scores:
        macro_ap = float(np.mean(ap_scores))
        ax.text(0.02, 0.04, f'Macro AP = {macro_ap:.3f}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(fontsize=8, loc='upper right',
              ncol=2 if n > 5 else 1)

    out = _plots_dir(save_dir) / f'{fname}.{_FMT}'
    _save(fig, out)


# ===========================================================================
# Section 3 — Attention analysis
# ===========================================================================

def plot_attention_heatmap(
    attn_stats: Dict[str, Any],
    save_dir: Union[str, Path],
    fname: str = 'attention_heatmap',
) -> None:
    """
    Description:
    ---------------
        Saves a heatmap of per-token attention weights averaged over
        the validation set, with small-scale and large-scale token
        regions visually separated by a vertical line.

        Rows = attention heads, Columns = token positions.

    Args:
    ---------------
        attn_stats: Dict with keys:
            'weights_tok_mean' [L, H] - mean attention per (token, head).
            'scale_lengths'    Tuple (L_small, L_large).
        save_dir: Output directory.
        fname: Filename stem.
    """
    weights = attn_stats.get('weights_tok_mean')
    scale_lengths = attn_stats.get('scale_lengths', (None, None))

    if weights is None or weights.ndim != 2:
        return

    L, H = weights.shape        # L tokens, H heads

    fig, ax = plt.subplots(figsize=(max(8, L // 3), max(3, H * 0.7 + 1)))
    fig.suptitle('Transformer Attention Weights\n'
                 '(averaged over validation set)',
                 fontsize=12, fontweight='bold')

    # weights.T → shape (H, L): rows=heads, cols=tokens
    data = weights.T
    if _HAS_SEABORN:
        sns.heatmap(
            data,
            ax=ax,
            cmap='YlOrRd',
            annot=(L <= 20),
            fmt='.2f' if L <= 20 else '',
            xticklabels=np.arange(1, L + 1),
            yticklabels=[f'Head {h + 1}' for h in range(H)],
            linewidths=0.3 if L <= 20 else 0,
            cbar_kws={'label': 'Attention Weight'},
        )
    else:
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        fig.colorbar(im, ax=ax, label='Attention Weight')
        ax.set_yticks(range(H))
        ax.set_yticklabels([f'Head {h + 1}' for h in range(H)])
        ax.set_xlabel('Token position')

    # Separator between small-scale and large-scale token regions
    L_small = scale_lengths[0]
    if L_small is not None and 0 < L_small < L:
        ax.axvline(x=L_small - 0.5, color='white',
                   linewidth=2.0, linestyle='-')
        ax.text(L_small / 2 - 0.5, -0.6, 'Small-scale',
                ha='center', fontsize=8, color='dimgrey',
                transform=ax.get_xaxis_transform())
        ax.text(L_small + (L - L_small) / 2 - 0.5, -0.6, 'Large-scale',
                ha='center', fontsize=8, color='dimgrey',
                transform=ax.get_xaxis_transform())

    out = _plots_dir(save_dir) / f'{fname}.{_FMT}'
    _save(fig, out)


def plot_head_importance(
    attn_stats: Dict[str, Any],
    save_dir: Union[str, Path],
    fname: str = 'head_importance',
) -> None:
    """
    Description:
    ---------------
        Saves a bar chart of learned attention-pooling head weights,
        showing which heads contribute most to the final representation.

    Args:
    ---------------
        attn_stats: Dict with key 'head_weights' [H].
        save_dir: Output directory.
        fname: Filename stem.
    """
    head_w = attn_stats.get('head_weights')
    if head_w is None:
        return

    head_w = np.asarray(head_w).ravel()
    H = len(head_w)

    fig, ax = plt.subplots(figsize=(max(4, H * 0.8), 4))
    fig.suptitle('Learned Attention-Pooling Head Weights',
                 fontsize=12, fontweight='bold')

    bars = ax.bar(range(H), head_w,
                  color=[_PALETTE[h % 10] for h in range(H)],
                  alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, head_w):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(range(H))
    ax.set_xticklabels([f'Head {h + 1}' for h in range(H)])
    ax.set_xlabel('Attention Head')
    ax.set_ylabel('Weight')
    ax.set_ylim(0, head_w.max() * 1.2)

    out = _plots_dir(save_dir) / f'{fname}.{_FMT}'
    _save(fig, out)


# ===========================================================================
# Section 4 — Metric tables
# ===========================================================================

def generate_metrics_table(
    metrics: Dict[str, Any],
    save_dir: Union[str, Path],
    fname: str = 'metrics_summary',
) -> None:
    """
    Description:
    ---------------
        Saves a CSV with scalar classification metrics
        (accuracy, F1, precision, recall, balanced_accuracy, loss).

    Args:
    ---------------
        metrics: Dict of {metric_name: float_value}.
        save_dir: Output directory.
        fname: Filename stem.
    """
    scalar_keys = [
        'accuracy', 'f1_macro', 'precision_macro',
        'recall_macro', 'balanced_accuracy', 'loss',
    ]
    rows = [
        (k, f'{metrics[k]:.6f}')
        for k in scalar_keys
        if k in metrics and isinstance(metrics[k], (int, float))
    ]

    out = _tables_dir(save_dir) / f'{fname}.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerows(rows)

    print(f'    [viz] {out.name}')


def generate_per_class_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: Union[str, Path],
    class_names: Optional[List[str]] = None,
    fname: str = 'per_class_metrics',
) -> None:
    """
    Description:
    ---------------
        Saves a CSV table with Precision, Recall, F1, and Support
        for each class plus macro/weighted averages.

    Args:
    ---------------
        y_true: Ground-truth labels [N].
        y_pred: Predicted labels [N].
        save_dir: Output directory.
        class_names: Optional class label strings.
        fname: Filename stem.
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    names = class_names if class_names else _class_names_for(n)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    # Macro and weighted averages
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    out = _tables_dir(save_dir) / f'{fname}.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'precision', 'recall', 'f1', 'support'])
        for i, cls in enumerate(classes):
            writer.writerow([
                names[i] if i < len(names) else str(cls),
                f'{prec[i]:.6f}', f'{rec[i]:.6f}',
                f'{f1[i]:.6f}', int(support[i]),
            ])
        writer.writerow([
            'macro avg', f'{prec_m:.6f}', f'{rec_m:.6f}',
            f'{f1_m:.6f}', int(np.sum(support))
        ])
        writer.writerow([
            'weighted avg', f'{prec_w:.6f}', f'{rec_w:.6f}',
            f'{f1_w:.6f}', int(np.sum(support))
        ])

    print(f'    [viz] {out.name}')


# ===========================================================================
# Section 5 — Full K-fold cross-validation visualizations
# ===========================================================================

def plot_cv_boxplot(
    results: Dict[str, Any],
    save_dir: Union[str, Path],
    metrics: Optional[List[str]] = None,
    fname: str = 'cv_boxplot',
) -> None:
    """
    Description:
    ---------------
        Saves a box-plot of each metric across all 25 experiments
        (5 subjects × 5 folds), with individual data points overlaid.

    Args:
    ---------------
        results: Dict from run_full_evaluation() with 'experiments' key.
        save_dir: Output directory.
        metrics: Which metric keys to show. Defaults to the 5 main ones.
        fname: Filename stem.
    """
    if metrics is None:
        metrics = [
            'accuracy', 'f1_macro', 'precision_macro',
            'recall_macro', 'balanced_accuracy',
        ]

    data_per_metric: Dict[str, List[float]] = {m: [] for m in metrics}
    for exp in results.get('experiments', []):
        if exp['status'] != 'success' or exp['metrics'] is None:
            continue
        for m in metrics:
            if m in exp['metrics']:
                data_per_metric[m].append(exp['metrics'][m])

    # Drop metrics with no data
    metrics = [m for m in metrics if data_per_metric[m]]
    if not metrics:
        return

    nice_labels = {
        'accuracy': 'Accuracy',
        'f1_macro': 'F1 Macro',
        'precision_macro': 'Precision',
        'recall_macro': 'Recall',
        'balanced_accuracy': 'Balanced Acc',
    }

    fig, ax = plt.subplots(figsize=(max(6, len(metrics) * 1.6), 5))
    fig.suptitle('Performance Distribution across All Experiments\n'
                 f'(N={len(list(data_per_metric.values())[0])} per metric)',
                 fontsize=12, fontweight='bold')

    plot_data = [data_per_metric[m] for m in metrics]
    labels = [nice_labels.get(m, m) for m in metrics]

    bp = ax.boxplot(
        plot_data,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2.0),
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=4, alpha=0.6),
        widths=0.5,
    )

    for patch, color in zip(bp['boxes'], _PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    # Overlay individual points with jitter
    rng = np.random.default_rng(42)
    for i, vals in enumerate(plot_data, 1):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals, s=18, alpha=0.55,
            color=_PALETTE[(i - 1) % 10], zorder=3,
        )

    ax.set_xticks(range(1, len(metrics) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Score')
    ax.set_ylim(0.0, 1.05)

    out = _plots_dir(save_dir) / f'{fname}.{_FMT}'
    _save(fig, out)


def plot_subject_heatmap(
    results: Dict[str, Any],
    save_dir: Union[str, Path],
    metric: str = 'f1_macro',
    fname: str = 'subject_fold_heatmap',
) -> None:
    """
    Description:
    ---------------
        Saves a Subject × Fold heatmap of a chosen metric.
        Rows = subjects, Columns = folds.
        Colour encodes the metric value; each cell also shows the number.

    Args:
    ---------------
        results: Dict from run_full_evaluation() with 'experiments' key.
        save_dir: Output directory.
        metric: Metric to visualise (default 'f1_macro').
        fname: Filename stem.
    """
    subjects = sorted({
        exp['subject']
        for exp in results.get('experiments', [])
        if exp['status'] == 'success'
    })
    if not subjects:
        return

    max_fold = max(
        exp['fold']
        for exp in results['experiments']
        if exp['status'] == 'success'
    )
    n_folds = max_fold

    # Build matrix: rows=subjects, cols=folds
    mat = np.full((len(subjects), n_folds), np.nan)
    for exp in results['experiments']:
        if exp['status'] != 'success' or exp['metrics'] is None:
            continue
        row = subjects.index(exp['subject'])
        col = exp['fold'] - 1
        val = exp['metrics'].get(metric)
        if val is not None:
            mat[row, col] = val

    fig, ax = plt.subplots(figsize=(max(5, n_folds * 1.2), max(3, len(subjects) * 0.9)))
    nice = metric.replace('_', ' ').title()
    fig.suptitle(f'Per-Subject × Fold Heatmap: {nice}',
                 fontsize=12, fontweight='bold')

    fold_labels = [f'Fold {f + 1}' for f in range(n_folds)]

    if _HAS_SEABORN:
        sns.heatmap(
            mat,
            ax=ax,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=fold_labels,
            yticklabels=subjects,
            vmin=0.0, vmax=1.0,
            linewidths=0.6,
            cbar_kws={'label': nice},
        )
    else:
        im = ax.imshow(mat, cmap='RdYlGn', vmin=0.0, vmax=1.0, aspect='auto')
        fig.colorbar(im, ax=ax, label=nice)
        ax.set_xticks(range(n_folds))
        ax.set_xticklabels(fold_labels)
        ax.set_yticks(range(len(subjects)))
        ax.set_yticklabels(subjects)
        for i in range(len(subjects)):
            for j in range(n_folds):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            fontsize=9, color='black')

    ax.set_xlabel('Fold')
    ax.set_ylabel('Subject')

    out = _plots_dir(save_dir) / f'{fname}.{_FMT}'
    _save(fig, out)


def plot_bootstrap_ci(
    analysis: Dict[str, Any],
    save_dir: Union[str, Path],
    metric: str = 'f1_macro',
    fname: str = 'bootstrap_ci',
) -> None:
    """
    Description:
    ---------------
        Saves a point-plot with 95% Bootstrap Confidence Intervals
        for each subject and the overall mean.

    Args:
    ---------------
        analysis: Dict from perform_statistical_analysis() with
            'per_subject' and 'summary' keys.
        save_dir: Output directory.
        metric: Metric to plot (default 'f1_macro').
        fname: Filename stem.
    """
    per_subject = analysis.get('per_subject', {})
    summary = analysis.get('summary', {})
    subjects = sorted(per_subject.keys())
    if not subjects:
        return

    means, ci_lo, ci_hi, labels = [], [], [], []

    for subj in subjects:
        subj_data = per_subject[subj].get(metric)
        if subj_data is None:
            continue
        m = subj_data['mean']
        lo = subj_data['ci_lower']
        hi = subj_data['ci_upper']
        means.append(m)
        ci_lo.append(m - lo)   # error bar extent (lower)
        ci_hi.append(hi - m)   # error bar extent (upper)
        labels.append(subj)

    # Overall
    if metric in summary:
        s = summary[metric]
        means.append(s['mean'])
        ci_lo.append(s['mean'] - s['ci_lower'])
        ci_hi.append(s['ci_upper'] - s['mean'])
        labels.append('Overall')

    x = np.arange(len(means))
    colors = [_PALETTE[i % 10] for i in range(len(means) - 1)] + ['black']

    fig, ax = plt.subplots(figsize=(max(5, len(means) * 1.0 + 1), 5))
    nice = metric.replace('_', ' ').title()
    fig.suptitle(f'{nice} — 95% Bootstrap Confidence Intervals',
                 fontsize=12, fontweight='bold')

    for xi, (m, lo, hi, col) in enumerate(zip(means, ci_lo, ci_hi, colors)):
        ax.errorbar(
            xi, m,
            yerr=[[lo], [hi]],
            fmt='o',
            markersize=8,
            color=col,
            capsize=6,
            capthick=2.0,
            linewidth=1.8,
        )
        ax.text(xi, m + hi + 0.01, f'{m:.3f}',
                ha='center', fontsize=8, color=col)

    # Separate the "Overall" marker visually
    if len(means) > 1:
        ax.axvline(len(means) - 1.5, color='grey',
                   linewidth=0.8, linestyle='--', alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel(nice)
    ax.set_ylim(
        max(0.0, min(means) - max(ci_lo) - 0.1),
        min(1.05, max(means) + max(ci_hi) + 0.08)
    )

    out = _plots_dir(save_dir) / f'{fname}.{_FMT}'
    _save(fig, out)


def plot_overall_metrics_bar(
    analysis: Dict[str, Any],
    save_dir: Union[str, Path],
    fname: str = 'overall_metrics_bar',
) -> None:
    """
    Description:
    ---------------
        Saves a horizontal bar chart of overall mean metric values
        with ± std error bars, across all experiments.

    Args:
    ---------------
        analysis: Dict from perform_statistical_analysis().
        save_dir: Output directory.
        fname: Filename stem.
    """
    summary = analysis.get('summary', {})
    ordered = [
        'accuracy', 'balanced_accuracy',
        'f1_macro', 'precision_macro', 'recall_macro',
    ]
    keys = [k for k in ordered if k in summary]
    if not keys:
        return

    nice = {
        'accuracy': 'Accuracy',
        'balanced_accuracy': 'Balanced Accuracy',
        'f1_macro': 'F1 Macro',
        'precision_macro': 'Precision Macro',
        'recall_macro': 'Recall Macro',
    }
    means = [summary[k]['mean'] for k in keys]
    stds  = [summary[k]['std']  for k in keys]
    ci_lo = [summary[k]['mean'] - summary[k]['ci_lower'] for k in keys]
    ci_hi = [summary[k]['ci_upper'] - summary[k]['mean'] for k in keys]
    labels = [nice.get(k, k) for k in keys]

    fig, ax = plt.subplots(figsize=(7, max(3, len(keys) * 0.8)))
    fig.suptitle('Overall Metric Summary (Mean ± Std)',
                 fontsize=12, fontweight='bold')

    y = np.arange(len(keys))
    ax.barh(y, means, xerr=stds,
            color=[_PALETTE[i % 10] for i in range(len(keys))],
            alpha=0.8, capsize=5, edgecolor='white',
            error_kw=dict(elinewidth=1.5, ecolor='black'))

    # Overlay 95% CI tick marks
    for yi, (lo, hi) in enumerate(zip(ci_lo, ci_hi)):
        m = means[yi]
        ax.plot([m - lo, m + hi], [yi, yi],
                color='black', linewidth=2.5, solid_capstyle='round')

    for yi, m in enumerate(means):
        ax.text(m + 0.002, yi, f'{m:.3f}', va='center', fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Score')
    ax.set_xlim(0.0, min(1.15, max(means) + 0.15))
    ax.invert_yaxis()

    out = _plots_dir(save_dir) / f'{fname}.{_FMT}'
    _save(fig, out)


# ===========================================================================
# Section 6 — Full-eval tables
# ===========================================================================

def generate_full_eval_table(
    analysis: Dict[str, Any],
    save_dir: Union[str, Path],
) -> None:
    """
    Description:
    ---------------
        Saves three CSV files and one Markdown report:

        full_eval_summary.csv     — one row with mean ± std per metric
        per_subject_metrics.csv   — one row per subject (F1 mean ± std, CI)
        metrics_report.md         — human-readable markdown tables

    Args:
    ---------------
        analysis: Dict from perform_statistical_analysis().
        save_dir: Output directory.
    """
    tdir = _tables_dir(save_dir)
    summary = analysis.get('summary', {})
    per_subj = analysis.get('per_subject', {})

    metric_keys = [
        'accuracy', 'balanced_accuracy',
        'f1_macro', 'precision_macro', 'recall_macro', 'loss',
    ]
    nice_names = {
        'accuracy': 'Accuracy',
        'balanced_accuracy': 'Balanced Acc',
        'f1_macro': 'F1 Macro',
        'precision_macro': 'Precision Macro',
        'recall_macro': 'Recall Macro',
        'loss': 'Loss',
    }

    # ------------------------------------------------------------------
    # 1. full_eval_summary.csv
    # ------------------------------------------------------------------
    path1 = tdir / 'full_eval_summary.csv'
    with open(path1, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'metric', 'mean', 'std', 'min', 'max', 'ci_lower', 'ci_upper'
        ])
        for k in metric_keys:
            if k not in summary:
                continue
            s = summary[k]
            writer.writerow([
                nice_names.get(k, k),
                f'{s["mean"]:.6f}', f'{s["std"]:.6f}',
                f'{s["min"]:.6f}', f'{s["max"]:.6f}',
                f'{s["ci_lower"]:.6f}', f'{s["ci_upper"]:.6f}',
            ])
    print(f'    [viz] {path1.name}')

    # ------------------------------------------------------------------
    # 2. per_subject_metrics.csv
    # ------------------------------------------------------------------
    subjects = sorted(per_subj.keys())
    path2 = tdir / 'per_subject_metrics.csv'
    with open(path2, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'subject', 'f1_mean', 'f1_std', 'f1_ci_lower', 'f1_ci_upper'
        ])
        for subj in subjects:
            d = per_subj[subj].get('f1_macro', {})
            writer.writerow([
                subj,
                f'{d.get("mean", float("nan")):.6f}',
                f'{d.get("std", float("nan")):.6f}',
                f'{d.get("ci_lower", float("nan")):.6f}',
                f'{d.get("ci_upper", float("nan")):.6f}',
            ])
    print(f'    [viz] {path2.name}')

    # ------------------------------------------------------------------
    # 3. metrics_report.md — Markdown tables
    # ------------------------------------------------------------------
    path3 = tdir / 'metrics_report.md'
    with open(path3, 'w', encoding='utf-8') as f:
        f.write('# Evaluation Results Report\n\n')
        f.write('## Table 1 — Overall Metric Summary\n\n')
        f.write('| Metric | Mean | Std | 95% CI | Min | Max |\n')
        f.write('|--------|------|-----|--------|-----|-----|\n')
        for k in metric_keys:
            if k not in summary:
                continue
            s = summary[k]
            ci = f'[{s["ci_lower"]:.4f}, {s["ci_upper"]:.4f}]'
            f.write(
                f'| {nice_names.get(k, k)} '
                f'| {s["mean"]:.4f} '
                f'| {s["std"]:.4f} '
                f'| {ci} '
                f'| {s["min"]:.4f} '
                f'| {s["max"]:.4f} |\n'
            )

        f.write('\n## Table 2 — Per-Subject F1 Macro\n\n')
        f.write('| Subject | Mean | Std | 95% CI |\n')
        f.write('|---------|------|-----|--------|\n')
        for subj in subjects:
            d = per_subj[subj].get('f1_macro', {})
            ci = f'[{d.get("ci_lower", 0):.4f}, {d.get("ci_upper", 0):.4f}]'
            f.write(
                f'| {subj} '
                f'| {d.get("mean", float("nan")):.4f} '
                f'| {d.get("std", float("nan")):.4f} '
                f'| {ci} |\n'
            )

        # Wilcoxon comparisons table if available
        comparisons = analysis.get('comparisons', [])
        if comparisons:
            f.write('\n## Table 3 — Pairwise Wilcoxon Tests (F1 Macro)\n\n')
            f.write('| Pair | Statistic | p-value | Significant (FDR 0.05) |\n')
            f.write('|------|-----------|---------|------------------------|\n')
            for comp in comparisons:
                sig = 'Yes' if comp.get('significant_fdr05') else 'No'
                f.write(
                    f'| {comp["pair"]} '
                    f'| {comp["statistic"]:.3f} '
                    f'| {comp["p_value"]:.4f} '
                    f'| {sig} |\n'
                )

    print(f'    [viz] {path3.name}')


# ===========================================================================
# Section 7 — High-level wrappers (main entry points)
# ===========================================================================

def save_single_run_plots(
    history: Dict[str, List[float]],
    val_outputs: Optional[Dict[str, np.ndarray]],
    attn_stats: Optional[Dict[str, Any]],
    res_dir: Union[str, Path],
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Description:
    ---------------
        Generates all visualisations and tables for a single training run.
        Called automatically from save_artifacts() in trainer.py.

        Outputs written to:
            <res_dir>/plots/   — PNG figures
            <res_dir>/tables/  — CSV tables

    Args:
    ---------------
        history:     Training history dict from train_loop().
        val_outputs: Validation predictions dict from evaluate_with_outputs().
                     Expected keys: 'y_true', 'y_pred', 'proba'.
        attn_stats:  Attention statistics dict (or None).
        res_dir:     Base results directory (cfg['results_dir']).
        class_names: Optional override for class label strings.
    """
    print('\n[viz] Generating single-run visualisations...')

    # 1. Training dynamics
    if history and history.get('train_loss'):
        plot_training_curves(history, res_dir)

    if val_outputs is None:
        print('[viz] val_outputs is None — skipping classification plots.')
        return

    y_true = val_outputs.get('y_true')
    y_pred = val_outputs.get('y_pred')
    proba  = val_outputs.get('proba')

    if y_true is None or y_pred is None:
        print('[viz] y_true / y_pred missing — skipping classification plots.')
        return

    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    names = class_names if class_names else _class_names_for(n_classes)

    # 2. Confusion matrix
    plot_confusion_matrix(y_true, y_pred, res_dir,
                          class_names=names, normalize=True)

    # 3. Per-class bar chart
    plot_per_class_metrics(y_true, y_pred, res_dir, class_names=names)

    # 4. ROC / PR curves (require probability estimates)
    if proba is not None and proba.ndim == 2:
        plot_roc_curves(y_true, proba, res_dir, class_names=names)
        plot_precision_recall_curves(y_true, proba, res_dir,
                                     class_names=names)

    # 5. Attention plots
    if attn_stats is not None:
        plot_attention_heatmap(attn_stats, res_dir)
        plot_head_importance(attn_stats, res_dir)

    # 6. Tables
    generate_per_class_table(y_true, y_pred, res_dir, class_names=names)

    print('[viz] Done — single-run visualisations saved.\n')


def save_full_eval_plots(
    results: Dict[str, Any],
    analysis: Dict[str, Any],
    res_dir: Union[str, Path],
) -> None:
    """
    Description:
    ---------------
        Generates all visualisations and tables for the full K-fold
        cross-validation run.
        Called automatically from main() in run_full_evaluation.py.

        Outputs written to:
            <res_dir>/plots/   — PNG figures
            <res_dir>/tables/  — CSV and Markdown tables

    Args:
    ---------------
        results:  Dict from run_full_evaluation() with 'experiments' key.
        analysis: Dict from perform_statistical_analysis().
        res_dir:  Base results directory (RESULTS_DIR constant).
    """
    print('\n[viz] Generating full-evaluation visualisations...')

    # 1. Box-plot of metric distributions
    plot_cv_boxplot(results, res_dir)

    # 2. Subject × Fold heatmap
    plot_subject_heatmap(results, res_dir, metric='f1_macro')

    # 3. Bootstrap CI per subject
    plot_bootstrap_ci(analysis, res_dir, metric='f1_macro')

    # 4. Overall metrics bar chart
    plot_overall_metrics_bar(analysis, res_dir)

    # 5. Tables: summary CSV, per-subject CSV, markdown report
    generate_full_eval_table(analysis, res_dir)

    print('[viz] Done — full-evaluation visualisations saved.\n')
