# file: trainer.py
# -*- coding: utf-8 -*-
"""
Циклы обучения/оценки и логгинг артефактов.

Модуль содержит:
1. Реализацию Class-Balanced Focal Loss для работы с дисбалансом классов.
2. Функции вычисления метрик (Accuracy, F1, Precision, Recall).
3. Цикл оценки модели с опциональным сбором предсказаний и attention статистик.
4. Основной цикл обучения с поддержкой AMP, градиентного клиппинга и Early Stopping.
5. Утилиты для сохранения артефактов (веса, метрики, история, конфиг).
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
# Утилиты сериализации и проверки устройств
# =============================================================================

def _to_serializable(obj: Any) -> Any:
    """
    Description:
    ---------------
        Рекурсивно конвертирует объект в JSON-совместимый формат.
        Обрабатывает специфичные типы: Path, torch.device, numpy скаляры,
        словари и списки.

    Args:
    ---------------
        obj: Any - Объект для конвертации.

    Returns:
    ---------------
        Any: Сериализуемый объект (str, int, float, list, dict).

    Raises:
    ---------------
        Нет явных исключений.

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
        Нормализует проверку устройства на CUDA.
        Работает как с объектами torch.device, так и со строками.

    Args:
    ---------------
        device: Any - Устройство (torch.device или str).

    Returns:
    ---------------
        bool: True, если устройство CUDA, иначе False.

    Raises:
    ---------------
        Нет явных исключений.

    Examples:
    ---------------
        >>> _is_cuda_device(torch.device('cuda:0'))
        True
    """
    if isinstance(device, torch.device):
        return device.type == 'cuda'
    return str(device) == 'cuda'


# =============================================================================
# Функции потерь и метрики (Loss Functions & Metrics)
# =============================================================================

class ClassBalancedFocalLoss(nn.Module):
    """
    Description:
    ---------------
        Class-Balanced Focal Loss.
        Комбинирует взвешивание классов (Class-Balanced) для учета дисбаланса
        и фокусировку на сложных примерах (Focal Loss).

        Формула:
        CB_weight = (1 - beta) / (1 - beta^n)
        FL_weight = (1 - p_t)^gamma
        Loss = - CB_weight * FL_weight * log(p_t)

    Args:
    ---------------
        class_counts: np.ndarray - Количество примеров для каждого класса.
        beta: float - Параметр эффективного числа примеров (обычно 0.9999).
        gamma: float - Параметр фокусировки (обычно 2.0).

    Returns:
    ---------------
        Tensor: Скалярное значение потерь.

    Raises:
    ---------------
        Нет явных исключений.

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

        # Вычисление эффективного числа примеров
        effective_num = 1.0 - torch.pow(beta, counts)

        # Вычисление весов классов (CB weights)
        alpha = (1.0 - beta) / effective_num.clamp(min=1e-8)

        # Нормализация весов так, чтобы их сумма равнялась числу классов
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
            Вычисляет значение функции потерь.

        Args:
        ---------------
            logits: torch.Tensor [B, C] - Логиты модели.
            targets: torch.Tensor [B] - Индексы истинных классов.

        Returns:
        ---------------
            torch.Tensor: Скалярное значение потерь.
        """
        probs = F.softmax(logits, dim=-1)

        # Извлечение вероятности правильного класса (p_t)
        pt = torch.gather(
            probs,
            -1,
            targets.unsqueeze(-1)
        ).squeeze(-1)

        # Выбор веса класса для каждого примера
        alpha_t = self.alpha[targets]

        # Вычисление фокусирующего веса (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - pt, self.gamma)

        # Финальная формула потерь
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
        Вычисляет набор метрик классификации.

    Args:
    ---------------
        y_true: np.ndarray - Истинные метки.
        y_pred: np.ndarray - Предсказанные метки.
        average: str - Стратегия усреднения ('macro', 'weighted', etc.).

    Returns:
    ---------------
        Dict[str, float]: Словарь с метриками (accuracy, f1, precision, recall,
            balanced_accuracy).

    Raises:
    ---------------
        Нет явных исключений.

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
# Оценка модели (Evaluation)
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
        Проводит оценку модели на валидационной выборке.
        Опционально собирает предсказания (probabilities, labels) и
        статистики механизма внимания (attention weights) для последующего
        анализа интерпретируемости.

    Args:
    ---------------
        model: nn.Module - Модель для оценки.
        loader: DataLoader - Загрузчик данных.
        device: str - Устройство для вычислений.
        criterion: nn.Module - Функция потерь (для расчета val loss).
        collect_outputs: bool - Собирать ли полные предсказания.
        collect_attn: bool - Собирать ли attention статистики.

    Returns:
    ---------------
        Tuple[Dict, Optional[Dict], Optional[Dict]]:
            - metrics: Словарь метрик.
            - outputs: Словарь с массивами предсказаний (если collect_outputs).
            - attn_stats: Словарь со средними attention весами (если collect_attn).

    Raises:
    ---------------
        Нет явных исключений.

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

        # Прямой проход
        out = model(
            eeg,
            subject_ids=subject_ids,
            return_attn=collect_attn
        )

        if collect_attn:
            logits, attn_stats = out
            if attn_stats is not None:
                # Усреднение весов внимания по батчам
                # attn_stats['weights_tok_mean']: [L, H]
                w = attn_stats['weights_tok_mean']
                if attn_sum is None:
                    attn_sum = w.detach().cpu()
                else:
                    attn_sum += w.detach().cpu()
                attn_count += 1

                # Сохранение мета-информации (не зависит от батча)
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
                # sample_id может быть тензором или списком
                sid_np = np.array(sample_ids)
                all_sample_ids.append(sid_np)

    # Конкатенация результатов
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    metrics = compute_metrics(y_true, y_pred, average='macro')
    metrics['loss'] = total_loss / len(loader)

    # Формирование словаря выводов
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

    # Формирование статистик внимания
    attn_stats_out: Optional[Dict[str, np.ndarray]] = None
    if collect_attn and attn_sum is not None and attn_count > 0:
        attn_stats_out = {
            'weights_tok_mean': (attn_sum / attn_count).numpy(),
            'head_weights': attn_meta['head_weights'],
            'scale_lengths': attn_meta['scale_lengths'],
        }

    return metrics, outputs, attn_stats_out


# =============================================================================
# Обучение (Training Loop)
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
        Основной цикл обучения модели.
        Поддерживает:
        - Автоматическое смешанное точность (AMP) для ускорения на GPU.
        - Градиентный клиппинг для стабильности.
        - Early Stopping по метрике F1-macro.
        - Логирование истории обучения и норм градиентов.
        - Восстановление лучших весов в конце обучения.

    Args:
    ---------------
        model: nn.Module - Модель для обучения.
        train_loader: DataLoader - Загрузчик тренировочных данных.
        val_loader: DataLoader - Загрузчик валидационных данных.
        criterion: nn.Module - Функция потерь.
        optimizer: Optimizer - Оптимизатор.
        scheduler: Any - Планировщик скорости обучения.
        cfg: Dict[str, Any] - Конфигурация обучения.
        device: str - Устройство.

    Returns:
    ---------------
        Tuple[Dict, Dict, Optional[Dict], Optional[Dict]]:
            - history: История метрик по эпохам.
            - final_metrics: Финальные метрики на валидации.
            - final_outputs: Предсказания на валидации.
            - final_attn: Attention статистики.

    Raises:
    ---------------
        Нет явных исключений.

    Examples:
    ---------------
        >>> history, metrics, outs, attn = train_loop(...)
    """
    use_cuda = _is_cuda_device(device)
    use_amp = cfg['training']['use_amp'] and use_cuda

    # GradScaler для AMP
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

    print(f"\nНачало обучения на {n_epochs} эпох...")
    print(f"Устройство: {device}, AMP: {use_amp}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров в модели: {n_params:,}")

    use_subject_embed = getattr(model, 'use_subject_embed', False)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        grad_norms: List[float] = []

        for batch in train_loader:
            eeg = batch['eeg'].to(device, non_blocking=use_cuda)
            labels = batch['label'].to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass с AMP
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

            # Вычисление нормы градиента до клиппинга (для мониторинга)
            total_norm_sq = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm_sq += param_norm * param_norm

            grad_norm = total_norm_sq ** 0.5 if total_norm_sq > 0 else 0.0
            grad_norms.append(grad_norm)

            # Градиентный клиппинг
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Шаг оптимизатора
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # Валидация
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

        # Обновление истории
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

        # Логирование эпохи
        print(f"\nЭпоха {epoch+1}/{n_epochs} | LR: {lr:.6f}")
        print(f"  Train Loss: {history['train_loss'][-1]:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Acc:    {val_metrics['accuracy']:.4f} | "
              f"Val F1:   {val_metrics['f1_macro']:.4f}")

        # Early Stopping логика
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            # Сохранение состояния модели на CPU
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience = 0
            print(f"  ✅ Новая лучшая F1: {best_f1:.4f}. Модель сохранена.")
        else:
            patience += 1
            if patience >= cfg['training']['early_stopping_patience']:
                print(f"\nРанняя остановка на эпохе {epoch+1}")
                break

    # Восстановление лучших весов
    if best_state:
        model.load_state_dict(best_state)
        print(f"\n✅ Восстановлена лучшая модель (F1: {best_f1:.4f})")

    # Финальная оценка с сбором артефактов
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
# Сохранение артефактов (Artifact Saving)
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
        Сохраняет все артефакты эксперимента:
        - Веса лучшей модели (.pt).
        - Метрики и историю обучения (.json).
        - Предсказания на валидации (.npz).
        - Attention статистики (.npz).
        - Полную конфигурацию запуска (.json).

    Args:
    ---------------
        cfg: Dict[str, Any] - Конфигурация.
        metrics: Dict[str, float] - Финальные метрики.
        history: Dict[str, List[float]] - История обучения.
        val_outputs: Optional[Dict] - Предсказания.
        attn_stats: Optional[Dict] - Attention статистики.
        model: nn.Module - Модель (для сохранения весов).

    Returns:
    ---------------
        None

    Raises:
    ---------------
        Нет явных исключений.

    Examples:
    ---------------
        >>> save_artifacts(cfg, metrics, history, outs, attn, model)
    """
    ckpt_dir = Path(cfg['checkpoint_dir'])
    res_dir = Path(cfg['results_dir'])

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Сохранение весов модели
    torch.save(model.state_dict(), ckpt_dir / 'best_model.pt')

    # Обработка выводов и per-class метрик
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

        # Сохранение предсказаний
        np.savez(
            res_dir / 'val_preds.npz',
            **{k: v for k, v in val_outputs.items() if v is not None}
        )

    # Сохранение attention статистик
    if attn_stats is not None:
        np.savez(res_dir / 'attn_stats.npz', **attn_stats)

    # Сохранение метрик
    with open(
        res_dir / 'metrics.json', 'w', encoding='utf-8'
    ) as f:
        json.dump(
            _to_serializable(metrics),
            f,
            indent=2,
            ensure_ascii=False
        )

    # Сохранение истории
    with open(
        res_dir / 'history.json', 'w', encoding='utf-8'
    ) as f:
        json.dump(
            _to_serializable(history),
            f,
            indent=2,
            ensure_ascii=False
        )

    # Сохранение конфига для воспроизводимости
    with open(
        res_dir / 'config_run.json', 'w', encoding='utf-8'
    ) as f:
        json.dump(
            _to_serializable(cfg),
            f,
            indent=2,
            ensure_ascii=False
        )

    # Генерация визуализаций (графики + таблицы метрик)
    try:
        from visualization import save_single_run_plots
        save_single_run_plots(
            history=history,
            val_outputs=val_outputs,
            attn_stats=attn_stats,
            res_dir=res_dir,
        )
    except Exception as exc:
        print(f'[viz] Визуализация пропущена: {exc}')

    print(f"Артефакты сохранены в {ckpt_dir} и {res_dir}")