# file: trainer.py
# -*- coding: utf-8 -*-
"""
Циклы обучения/оценки и логгинг артефактов.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from utils import print_metrics


def _to_serializable(obj: Any):
    """Рекурсивно конвертирует объект в JSON-совместимый формат."""
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
    """Нормализует проверку CUDA для torch.device и строк."""
    if isinstance(device, torch.device):
        return device.type == 'cuda'
    return str(device) == 'cuda'

# =============================================================================
# Функции потерь и метрики
# =============================================================================

class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss."""
    def __init__(self, class_counts: np.ndarray, beta: float, gamma: float):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        effective_num = 1.0 - torch.pow(beta, counts)
        alpha = (1.0 - beta) / effective_num.clamp(min=1e-8)
        alpha = alpha / alpha.sum() * len(counts)
        self.register_buffer('alpha', alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        pt = torch.gather(probs, -1, targets.unsqueeze(-1)).squeeze(-1)
        alpha_t = self.alpha[targets]
        focal_weight = torch.pow(1.0 - pt, self.gamma)
        loss = -alpha_t * focal_weight * torch.log(pt.clamp(min=1e-8))
        return loss.mean()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str) -> Dict[str, float]:
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        f'f1_{average}': f1_score(y_true, y_pred, average=average, zero_division=0),
        f'precision_{average}': precision_score(y_true, y_pred, average=average, zero_division=0),
        f'recall_{average}': recall_score(y_true, y_pred, average=average, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
    }


# =============================================================================
# Оценка
# =============================================================================

@torch.no_grad()
def evaluate_with_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
    collect_outputs: bool = False,
    collect_attn: bool = False,
) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]:
    """Оценка + опциональный сбор предсказаний и attention статистик."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    all_probs, all_subj, all_sample_ids = [], [], []

    attn_sum = None
    attn_count = 0

    use_subject_embed = getattr(model, 'use_subject_embed', False)
    non_blocking = _is_cuda_device(device)

    for batch in loader:
        eeg = batch['eeg'].to(device, non_blocking=non_blocking)
        labels = batch['label'].to(device, non_blocking=non_blocking)
        subject_ids = batch['subject_id'].to(device, non_blocking=non_blocking) if use_subject_embed else None
        sample_ids = batch.get('sample_id')

        out = model(eeg, subject_ids=subject_ids, return_attn=collect_attn)
        if collect_attn:
            logits, attn_stats = out
            if attn_stats is not None:
                # attn_stats: {'weights_tok_mean': [L,H], 'head_weights': [H], 'scale_lengths': (Ls, Ll)}
                w = attn_stats['weights_tok_mean']  # tensor
                if attn_sum is None:
                    attn_sum = w.detach().cpu()
                else:
                    attn_sum += w.detach().cpu()
                attn_count += 1
                attn_meta = {
                    'head_weights': attn_stats['head_weights'].detach().cpu().numpy(),
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
            subj_np = subject_ids.cpu().numpy() if subject_ids is not None else np.zeros_like(preds.cpu().numpy())
            all_subj.append(subj_np)
            if sample_ids is not None:
                # sample_id может быть тензором или list
                sid_np = np.array(sample_ids)
                all_sample_ids.append(sid_np)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    metrics = compute_metrics(y_true, y_pred, average='macro')
    metrics['loss'] = total_loss / len(loader)

    outputs = None
    if collect_outputs:
        outputs = {
            'y_true': y_true,
            'y_pred': y_pred,
            'proba': np.concatenate(all_probs) if all_probs else None,
            'subject_id': np.concatenate(all_subj) if all_subj else None,
            'sample_id': np.concatenate(all_sample_ids) if all_sample_ids else None,
        }

    attn_stats_out = None
    if collect_attn and attn_sum is not None and attn_count > 0:
        attn_stats_out = {
            'weights_tok_mean': (attn_sum / attn_count).numpy(),
            'head_weights': attn_meta['head_weights'],
            'scale_lengths': attn_meta['scale_lengths'],
        }

    return metrics, outputs, attn_stats_out


# =============================================================================
# Обучение
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
) -> Tuple[Dict[str, List], Dict[str, float]]:
    """Основной цикл обучения модели."""
    use_cuda = _is_cuda_device(device)
    use_amp = cfg['training']['use_amp'] and use_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_state, best_f1, patience = None, 0.0, 0
    history: Dict[str, List] = {
        'train_loss': [], 'val_loss': [], 'val_f1_macro': [], 'lr': [],
        'grad_norm_min': [], 'grad_norm_mean': [], 'grad_norm_max': []
    }
    n_epochs = cfg['training']['n_epochs']
    grad_clip = cfg['training']['grad_clip']

    print(f"\nНачало обучения на {n_epochs} эпох...")
    print(f"Устройство: {device}, AMP: {use_amp}")
    print(f"Параметров в модели: {sum(p.numel() for p in model.parameters()):,}")

    use_subject_embed = getattr(model, 'use_subject_embed', False)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        grad_norms = []
        for batch in train_loader:
            eeg = batch['eeg'].to(device, non_blocking=use_cuda)
            labels = batch['label'].to(device, non_blocking=use_cuda)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if use_subject_embed:
                    subject_ids = batch['subject_id'].to(device, non_blocking=use_cuda)
                    logits = model(eeg, subject_ids=subject_ids)
                else:
                    logits = model(eeg)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # градиентные нормы до клиппинга
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm_sq += param_norm * param_norm
            grad_norms.append(total_norm_sq ** 0.5 if total_norm_sq > 0 else 0.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        val_metrics, _, _ = evaluate_with_outputs(model, val_loader, device, criterion, collect_outputs=False, collect_attn=False)
        if scheduler:
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(total_loss / len(train_loader))
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['lr'].append(lr)
        history['grad_norm_min'].append(float(np.min(grad_norms)) if grad_norms else 0.0)
        history['grad_norm_mean'].append(float(np.mean(grad_norms)) if grad_norms else 0.0)
        history['grad_norm_max'].append(float(np.max(grad_norms)) if grad_norms else 0.0)

        print(f"\nЭпоха {epoch+1}/{n_epochs} | LR: {lr:.6f}")
        print(f"  Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Acc:    {val_metrics['accuracy']:.4f} | Val F1:   {val_metrics['f1_macro']:.4f}")

        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            print(f"  ✅ Новая лучшая F1: {best_f1:.4f}. Модель сохранена.")
        else:
            patience += 1
            if patience >= cfg['training']['early_stopping_patience']:
                print(f"\nРанняя остановка на эпохе {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
        print(f"\n✅ Восстановлена лучшая модель (F1: {best_f1:.4f})")

    final_metrics, final_outputs, final_attn = evaluate_with_outputs(
        model,
        val_loader,
        device,
        criterion,
        collect_outputs=True,
        collect_attn=bool(cfg.get('logging', {}).get('save_attn', False))
    )
    return history, final_metrics, final_outputs, final_attn


# =============================================================================
# Сохранение артефактов
# =============================================================================

def save_artifacts(
    cfg: Dict[str, Any],
    metrics: Dict[str, float],
    history: Dict[str, List],
    val_outputs: Optional[Dict[str, np.ndarray]],
    attn_stats: Optional[Dict[str, np.ndarray]],
    model: nn.Module
) -> None:
    """Сохраняет веса, метрики, историю, предсказания и конфиг."""
    ckpt_dir = Path(cfg['checkpoint_dir'])
    res_dir = Path(cfg['results_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), ckpt_dir / 'best_model.pt')
    if val_outputs is not None:
        # per-class метрики
        if val_outputs.get('y_true') is not None and val_outputs.get('y_pred') is not None:
            prec, rec, f1, support = precision_recall_fscore_support(
                val_outputs['y_true'], val_outputs['y_pred'], average=None, zero_division=0
            )
            metrics['per_class'] = [
                {'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(s)}
                for p, r, f, s in zip(prec, rec, f1, support)
            ]
        np.savez(res_dir / 'val_preds.npz', **{k: v for k, v in val_outputs.items() if v is not None})

    if attn_stats is not None:
        np.savez(res_dir / 'attn_stats.npz', **attn_stats)

    with open(res_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(_to_serializable(metrics), f, indent=2, ensure_ascii=False)

    with open(res_dir / 'history.json', 'w', encoding='utf-8') as f:
        json.dump(_to_serializable(history), f, indent=2, ensure_ascii=False)

    # Dump full config for воспроизводимости
    with open(res_dir / 'config_run.json', 'w', encoding='utf-8') as f:
        json.dump(_to_serializable(cfg), f, indent=2, ensure_ascii=False)

    print(f"Артефакты сохранены в {ckpt_dir} и {res_dir}")
