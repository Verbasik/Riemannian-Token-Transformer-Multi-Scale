# file: trainer.py
# -*- coding: utf-8 -*-
"""
Домен обучения: циклы тренировки и оценки, функции потерь, метрики.

Содержит основной цикл обучения (train_loop), функцию оценки (evaluate),
реализацию ClassBalancedFocalLoss и функции для расчета и вывода метрик.
"""
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)
from torch.utils.data import DataLoader

from utils import print_metrics

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
    """Вычисляет набор стандартных метрик для задачи классификации."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        f'f1_{average}': f1_score(y_true, y_pred, average=average, zero_division=0),
        f'precision_{average}': precision_score(y_true, y_pred, average=average, zero_division=0),
        f'recall_{average}': recall_score(y_true, y_pred, average=average, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
    }

# =============================================================================
# Циклы обучения и оценки
# =============================================================================

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, criterion: nn.Module) -> Dict[str, float]:
    """Выполняет оценку модели на предоставленном наборе данных."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    use_subject_embed = hasattr(model, 'use_subject_embed') and model.use_subject_embed

    for batch in loader:
        eeg, labels = batch['eeg'].to(device), batch['label'].to(device)

        if use_subject_embed:
            subject_ids = batch['subject_id'].to(device)
            logits = model(eeg, subject_ids=subject_ids)
        else:
            logits = model(eeg)

        total_loss += criterion(logits, labels).item()
        all_preds.append(logits.argmax(-1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    y_pred, y_true = np.concatenate(all_preds), np.concatenate(all_labels)
    metrics = compute_metrics(y_true, y_pred, average='macro')
    metrics['loss'] = total_loss / len(loader)
    return metrics


def train_loop(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
    criterion: nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: Any, cfg: Dict[str, Any], device: str
) -> Tuple[Dict[str, List], Dict[str, float]]:
    """Основной цикл обучения модели."""
    use_amp = cfg['training']['use_amp'] and device == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_state, best_f1, patience = None, 0.0, 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1_macro': [], 'lr': []}
    n_epochs = cfg['training']['n_epochs']
    grad_clip = cfg['training']['grad_clip']

    print(f"\nНачало обучения на {n_epochs} эпох...")
    print(f"Устройство: {device}, AMP: {use_amp}")
    print(f"Параметров в модели: {sum(p.numel() for p in model.parameters()):,}")

    use_subject_embed = hasattr(model, 'use_subject_embed') and model.use_subject_embed

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            eeg, labels = batch['eeg'].to(device), batch['label'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if use_subject_embed:
                    subject_ids = batch['subject_id'].to(device)
                    logits = model(eeg, subject_ids=subject_ids)
                else:
                    logits = model(eeg)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        val_metrics = evaluate(model, val_loader, device, criterion)
        if scheduler: scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(total_loss / len(train_loader))
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['lr'].append(lr)

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
    
    final_metrics = evaluate(model, val_loader, device, criterion)
    return history, final_metrics


def save_artifacts(cfg: Dict[str, Any], metrics: Dict[str, float], model: nn.Module) -> None:
    """Сохраняет артефакты обучения: веса модели и метрики."""
    ckpt_dir = Path(cfg['checkpoint_dir'])
    res_dir = Path(cfg['results_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / 'best_model.pt')
    with open(res_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Артефакты сохранены в {ckpt_dir} и {res_dir}")