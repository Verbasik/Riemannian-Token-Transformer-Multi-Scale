# file: train.py
# -*- coding: utf-8 -*-
"""
Точка входа обучения Phase 4B с логгингом артефактов для анализа.

Этот скрипт orchestrates весь процесс обучения:
1. Загрузка и предобработка данных (Subject-Wise Normalization).
2. Настройка кросс-валидации (Stratified, Grouped, LOSO).
3. Инициализация модели, критерия потерь и оптимизатора.
4. Запуск тренировочного цикла с сохранением метрик и артефактов.

Использование:
    python train.py [--save-attn]
"""

# =============================================================================
# Standard Libraries
# =============================================================================
import argparse
import sys
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Third-Party Libraries
# =============================================================================
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# =============================================================================
# Local Imports
# =============================================================================
from config import default_config
from data_loader import (
    ChiscoDataset,
    ChiscoSubset,
    compute_channelwise_stats,
    compute_hybrid_stats,
    compute_subjectwise_stats,
    create_subject_mapping,
    get_stratified_cv_splits,
    get_stratified_group_cv_splits,
    get_within_subject_cv_splits,
    get_loso_splits,
    load_all_data_metaclass,
)
from model import RTTMultiScale
from trainer import (
    ClassBalancedFocalLoss,
    save_artifacts,
    train_loop,
)
from utils import pretty_print_run, print_metrics, set_seed


# =============================================================================
# Builders (Компоненты сборки пайплайна)
# =============================================================================

def _subjects_for_indices(
    samples: List[Dict[str, Any]],
    indices: np.ndarray
) -> set:
    """Возвращает множество subject IDs для заданных индексов."""
    return {samples[int(idx)]['subject'] for idx in indices}


def _resolve_cv_protocol(cfg: Dict[str, Any]) -> Tuple[str, str]:
    """
    Description:
    ---------------
        Нормализует пару (protocol, mode) и валидирует методологический
        смысл выбранной оценки.

    Returns:
    ---------------
        Tuple[str, str]: (protocol, mode).
    """
    cv_cfg = cfg.setdefault('cv', {})
    mode = cv_cfg.get('mode', 'within_subject')
    protocol = cv_cfg.get('protocol')

    if protocol is None:
        if mode in ('within_subject', 'stratified'):
            protocol = 'within_subject'
        elif mode in ('stratified_group', 'loso'):
            protocol = 'subject_heldout'
        else:
            raise ValueError(f"Unknown CV mode: {mode}.")
        cv_cfg['protocol'] = protocol

    allowed_modes = {
        'within_subject': {'within_subject', 'stratified'},
        'subject_heldout': {'stratified_group', 'loso'},
    }

    if protocol not in allowed_modes:
        raise ValueError(
            f"Unknown CV protocol: {protocol}. Expected one of "
            f"{sorted(allowed_modes)}."
        )
    if mode not in allowed_modes[protocol]:
        raise ValueError(
            f"CV mode '{mode}' is incompatible with protocol '{protocol}'. "
            f"Allowed modes: {sorted(allowed_modes[protocol])}."
        )

    return protocol, mode


def _resolve_unknown_subject_policy(
    cfg: Dict[str, Any],
    protocol: str
) -> str:
    """
    Description:
    ---------------
        Преобразует policy='auto' в явное поведение:
        - within_subject: error, потому что unseen subject является ошибкой.
        - subject_heldout: zero, потому что val subject отсутствует в train.

    Returns:
    ---------------
        str: Разрешенная policy ('error', 'zero' или 'mean').
    """
    model_cfg = cfg.setdefault('model', {})
    policy = model_cfg.get('unknown_subject_policy', 'auto')

    if policy == 'auto':
        policy = 'error' if protocol == 'within_subject' else 'zero'

    allowed = {'error', 'zero', 'mean'}
    if policy not in allowed:
        raise ValueError(
            f"Unknown subject policy '{policy}'. Expected one of "
            f"{sorted(allowed)} or 'auto'."
        )

    model_cfg['unknown_subject_policy_resolved'] = policy
    return policy


def build_loaders(
    cfg: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, np.ndarray, int, int]:
    """
    Description:
    ---------------
        Создает DataLoaders для обучения и валидации с учетом выбранной
        стратегии кросс-валидации и нормализации.
        Поддерживает протоколы:
        - within_subject: субъект присутствует и в train, и в val.
        - subject_heldout: val-субъекты полностью отсутствуют в train.
        Вычисляет статистику нормализации только на train-части (preventing leakage).

    Args:
    ---------------
        cfg: Dict[str, Any] - Конфигурационный словарь.

    Returns:
    ---------------
        Tuple[DataLoader, DataLoader, np.ndarray, int, int]:
            - train_loader: DataLoader для обучающей выборки.
            - val_loader: DataLoader для валидационной выборки.
            - train_labels: Метки обучающей выборки (для взвешивания потерь).
            - effective_channels: Количество каналов после исключения.
            - n_subjects: Количество train-субъектов в embedding table.

    Raises:
    ---------------
        ValueError: Если режим CV неизвестен или если получен пустой fold.

    Examples:
    ---------------
        >>> cfg = default_config()
        >>> loaders = build_loaders(cfg)
        >>> len(loaders)
        5
    """
    # Загрузка данных с конвертацией в мета-классы
    samples = load_all_data_metaclass(
        data_dir=cfg['data']['data_dir'],
        subject_ids=cfg['data']['subject_ids'],
        task=cfg['data']['task'],
    )

    labels = np.array([s['label'] for s in samples])
    subject_mapping_all = create_subject_mapping(samples)

    # =========================================================================
    # Выбор стратегии кросс-валидации (Subject-Aware CV)
    # =========================================================================
    cv_protocol, cv_mode = _resolve_cv_protocol(cfg)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    if cv_protocol == 'within_subject':
        if cv_mode == 'within_subject':
            splits = get_within_subject_cv_splits(
                samples,
                labels,
                cfg['cv']['n_splits'],
                cfg['cv']['random_state']
            )
            print(
                "✅ CV Protocol: within_subject "
                "(каждый субъект есть в train и val)"
            )
        else:
            # Backward-compatible mixed-subject split. Это не LOSO:
            # subject embeddings обучаются на train-части тех же субъектов.
            splits = get_stratified_cv_splits(
                labels,
                cfg['cv']['n_splits'],
                cfg['cv']['random_state']
            )
            print(
                "⚠️  CV Protocol: within_subject + stratified "
                "(mixed-subject split)"
            )

    elif cv_mode == 'stratified_group':
        groups = np.array([
            subject_mapping_all[s['subject']] for s in samples
        ])
        n_unique_groups = np.unique(groups).size

        if n_unique_groups < 2:
            raise ValueError(
                "subject_heldout protocol requires at least 2 subjects. "
                f"Got {n_unique_groups}."
            )

        splits = get_stratified_group_cv_splits(
            labels,
            groups,
            n_splits=cfg['cv']['n_splits'],
            random_state=cfg['cv']['random_state']
        )
        print(
            "🔐 CV Protocol: subject_heldout + stratified_group "
            "(val subjects absent from train)"
        )

    elif cv_mode == 'loso':
        splits = get_loso_splits(samples, subject_mapping_all)
        print(
            f"🔐 CV Protocol: subject_heldout + LOSO "
            f"{len(splits)} разбиений)"
        )

    else:
        raise ValueError(f"Unknown CV mode: {cv_mode}.")

    # Выбор конкретного fold для текущего запуска
    fold_index = int(cfg.get('cv', {}).get('fold_index', 0))
    fold_index = max(0, min(fold_index, len(splits) - 1))
    train_idx, val_idx = splits[fold_index]

    # Валидация непустоты разрезов
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"Пустой fold: train={len(train_idx)}, val={len(val_idx)}. "
            f"cv_mode={cv_mode}, n_samples={len(labels)}. "
            "Проверьте режим CV и число субъектов в запуске."
        )

    print(
        f"   Fold {fold_index + 1}/{len(splits)}: "
        f"train={len(train_idx)}, val={len(val_idx)}"
    )

    train_subjects = _subjects_for_indices(samples, train_idx)
    val_subjects = _subjects_for_indices(samples, val_idx)
    unknown_val_subjects = val_subjects - train_subjects
    overlapping_subjects = train_subjects & val_subjects

    if cv_protocol == 'within_subject':
        if unknown_val_subjects:
            raise ValueError(
                "within_subject protocol requires every validation subject "
                f"to appear in train. Missing: {sorted(unknown_val_subjects)}."
            )
    elif overlapping_subjects:
        raise ValueError(
            "subject_heldout protocol requires disjoint train/val subjects. "
            f"Overlap: {sorted(overlapping_subjects)}."
        )

    unknown_policy = _resolve_unknown_subject_policy(cfg, cv_protocol)
    use_subject_embed = cfg['model'].get('use_subject_embed', False)
    if cv_protocol == 'subject_heldout' and use_subject_embed:
        if unknown_policy == 'error':
            raise ValueError(
                "subject_heldout with subject embeddings requires an unknown "
                "subject policy. Set model.unknown_subject_policy to 'zero' "
                "or 'mean', or disable model.use_subject_embed."
            )
        print(
            "   Unknown validation subjects use "
            f"subject embedding policy: {unknown_policy}"
        )

    subject_mapping = create_subject_mapping(samples, train_idx)
    n_subjects = len(subject_mapping)

    # Mapping строится только по train, поэтому held-out subjects получают -1.
    dataset = ChiscoDataset(
        samples=samples,
        normalize=cfg['data']['normalize'],
        exclude_channels=cfg['data'].get('exclude_channels'),
        subject_mapping=subject_mapping,
        unknown_subject_index=-1
    )

    # =========================================================================
    # Вычисление статистики нормализации (только на Train!)
    # =========================================================================
    norm_mode = cfg['data']['normalize']
    exclude_ch = cfg['data'].get('exclude_channels')

    if cv_protocol == 'subject_heldout' and norm_mode == 'zscore_subject_channel':
        raise ValueError(
            "zscore_subject_channel is incompatible with subject_heldout: "
            "held-out subjects have no train statistics. Use zscore_hybrid "
            "or zscore_dataset_channel."
        )

    if norm_mode == 'zscore_hybrid':
        # Гибридная нормализация: центрирование по субъекту, скейлинг глобальный
        dataset.norm_stats = compute_hybrid_stats(
            samples, train_idx, exclude_ch
        )
    elif norm_mode == 'zscore_subject_channel':
        # Полная нормализация по каждому субъекту отдельно
        dataset.norm_stats = compute_subjectwise_stats(
            samples, train_idx, exclude_ch
        )
    elif norm_mode == 'zscore_dataset_channel':
        # Глобальная нормализация по всему датасету
        mean_c, std_c = compute_channelwise_stats(
            samples, train_idx, exclude_ch
        )
        dataset.norm_stats = {'mean': mean_c, 'std': std_c}

    # =========================================================================
    # Конфигурация DataLoader
    # =========================================================================
    num_workers = int(cfg['training']['num_workers'])
    if (
        num_workers > 0 and
        sys.version_info >= (3, 14) and
        not cfg['training'].get('allow_multiprocessing_dataloader', False)
    ):
        print(
            "⚠️  DataLoader multiprocessing отключён: Python 3.14 "
            "использует forkserver, который pickle-ит большой in-memory "
            "dataset и может падать с pickle data was truncated. "
            "Set training.allow_multiprocessing_dataloader=True to override."
        )
        num_workers = 0

    loader_common: Dict[str, Any] = {
        'batch_size': cfg['training']['batch_size'],
        'num_workers': num_workers,
        'pin_memory': cfg['training']['pin_memory'],
    }

    # Дополнительные параметры для многопроцессной загрузки
    if num_workers > 0:
        loader_common['persistent_workers'] = bool(
            cfg['training'].get('persistent_workers', True)
        )
        prefetch_factor = int(cfg['training'].get('prefetch_factor', 2))
        if prefetch_factor > 0:
            loader_common['prefetch_factor'] = prefetch_factor

    # Создание подвыборок и DataLoader
    train_loader = DataLoader(
        ChiscoSubset(dataset, train_idx),
        shuffle=True,
        **loader_common,
    )
    val_loader = DataLoader(
        ChiscoSubset(dataset, val_idx),
        shuffle=False,
        **loader_common,
    )

    # Определение эффективного количества каналов
    eeg_shape = dataset[int(train_idx[0])]['eeg'].shape
    effective_channels = eeg_shape[0]

    return train_loader, val_loader, labels[train_idx], effective_channels, n_subjects


def build_model(
    cfg: Dict[str, Any],
    n_channels: int,
    n_subjects: int
) -> nn.Module:
    """
    Description:
    ---------------
        Инициализирует модель RTTMultiScale на основе конфигурации.
        Автоматически подставляет размеры входных данных и количество субъектов.

    Args:
    ---------------
        cfg: Dict[str, Any] - Конфигурация модели.
        n_channels: int - Количество каналов EEG.
        n_subjects: int - Количество субъектов (для Embedding слоя).

    Returns:
    ---------------
        nn.Module: Инициализированная модель.

    Raises:
    ---------------
        Нет явных исключений.
    """
    m = cfg['model']
    unknown_subject_policy = m.get('unknown_subject_policy_resolved')
    if unknown_subject_policy is None:
        protocol, _ = _resolve_cv_protocol(cfg)
        unknown_subject_policy = _resolve_unknown_subject_policy(cfg, protocol)

    return RTTMultiScale(
        n_channels=n_channels,
        n_classes=m['n_classes'],
        proj_channels=m['proj_channels'],
        window_size_small=m['window_size_small'],
        stride_small=m['stride_small'],
        window_size_large=m['window_size_large'],
        stride_large=m['stride_large'],
        d_model=m['d_model'],
        n_heads=m['n_heads'],
        ff_dim=m['ff_dim'],
        n_layers=m['n_layers'],
        dropout=m['dropout'],
        eps=m['eps'],
        attn_heads=m.get('attn_heads', 1),
        cov_type=m.get('cov_type', 'corr'),
        oas_min_alpha=m.get('oas_min_alpha', 0.1),
        use_subject_embed=m.get('use_subject_embed', False),
        n_subjects=n_subjects,
        subject_embed_dim=m.get('subject_embed_dim', 16),
        subject_embed_dropout=m.get('subject_embed_dropout', 0.0),
        unknown_subject_policy=unknown_subject_policy
    )


def build_criterion(
    cfg: Dict[str, Any],
    train_labels: np.ndarray
) -> nn.Module:
    """
    Description:
    ---------------
        Создает функцию потерь.
        Поддерживает Class-Balanced Focal Loss для работы с дисбалансом классов.

    Args:
    ---------------
        cfg: Dict[str, Any] - Конфигурация функции потерь.
        train_labels: np.ndarray - Метки обучающей выборки (для подсчета весов).

    Returns:
    ---------------
        nn.Module: Функция потерь.

    Raises:
    ---------------
        Нет явных исключений.
    """
    loss_cfg = cfg['loss']
    if loss_cfg['type'] == 'cb_focal':
        counts = np.bincount(
            train_labels,
            minlength=cfg['model']['n_classes']
        )
        return ClassBalancedFocalLoss(
            counts,
            beta=loss_cfg['beta'],
            gamma=loss_cfg['gamma']
        )
    return nn.CrossEntropyLoss()


def build_optimizer_and_scheduler(
    model: nn.Module,
    cfg: Dict[str, Any]
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """
    Description:
    ---------------
        Настраивает оптимизатор и планировщик скорости обучения.
        Реализует дифференцированный Weight Decay для эмбеддингов субъектов
        (обычно меньший WD для эмбеддингов улучшает обобщаемость).
        Поддерживает Warmup + Cosine Annealing.

    Args:
    ---------------
        model: nn.Module - Модель для оптимизации.
        cfg: Dict[str, Any] - Конфигурация оптимизатора и scheduler.

    Returns:
    ---------------
        Tuple[Optimizer, Scheduler]:
            - Optimizer: Настроенный оптимизатор (Adam/AdamW).
            - Scheduler: Планировщик LR (или None).

    Raises:
    ---------------
        Нет явных исключений.
    """
    opt_cfg = cfg['optimizer']
    train_cfg = cfg['training']

    params_subject: List[nn.Parameter] = []
    params_other: List[nn.Parameter] = []

    # Разделение параметров: эмбеддинги субъектов vs остальные веса
    if getattr(model, 'use_subject_embed', False):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('subject_embed.'):
                params_subject.append(param)
            else:
                params_other.append(param)
    else:
        params_other = [
            p for p in model.parameters() if p.requires_grad
        ]

    # Получение коэффициента WD для эмбеддингов (может отличаться от основного)
    subject_wd = opt_cfg.get(
        'subject_embed_weight_decay',
        train_cfg['weight_decay']
    )

    param_groups = []
    if params_other:
        param_groups.append({
            'params': params_other,
            'weight_decay': train_cfg['weight_decay']
        })
    if params_subject:
        param_groups.append({
            'params': params_subject,
            'weight_decay': subject_wd
        })

    # Инициализация оптимизатора
    if opt_cfg['name'] == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=train_cfg['learning_rate'],
            betas=tuple(opt_cfg['betas'])
        )
    else:
        optimizer = torch.optim.Adam(
            param_groups,
            lr=train_cfg['learning_rate']
        )

    # Инициализация планировщика (Scheduler)
    sched_cfg = cfg['scheduler']
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

    if sched_cfg.get('name') == 'cosine':
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR,
            LinearLR,
            SequentialLR,
        )

        warmup_epochs = int(sched_cfg.get('warmup_epochs', 0))

        if warmup_epochs > 0:
            # Линейный разогрев (Warmup)
            sched_warmup = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs
            )
            # Косинусное затухание (Cosine Decay)
            t_max = max(sched_cfg['T_max'] - warmup_epochs, 1)
            sched_cosine = CosineAnnealingLR(
                optimizer,
                T_max=t_max
            )
            # Последовательное объединение
            scheduler = SequentialLR(
                optimizer,
                schedulers=[sched_warmup, sched_cosine],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=sched_cfg['T_max']
            )

    return optimizer, scheduler


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Description:
    ---------------
        Парсит аргументы командной строки.

    Returns:
    ---------------
        argparse.Namespace: Объект с аргументами.
    """
    parser = argparse.ArgumentParser(description="Train Phase 4B")
    parser.add_argument(
        '--save-attn',
        action='store_true',
        help='Сохранять усреднённые attention веса на валидации'
    )
    return parser.parse_args()


def main(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Description:
    ---------------
        Главная функция запуска обучения.
        Координирует сборку всех компонентов, запуск цикла обучения и
        сохранение результатов. Может принимать конфиг программно или
        использовать CLI аргументы.

    Args:
    ---------------
        cfg: Optional[Dict[str, Any]] - Внешняя конфигурация.
            Если None, используется default_config + CLI аргументы.

    Returns:
    ---------------
        Dict[str, Any]: Финальные метрики валидации.

    Raises:
    ---------------
        Exception: Любые ошибки в процессе обучения.
    """
    # Обработка конфигурации
    if cfg is None:
        args = parse_args()
        cfg = default_config()
        cfg['logging']['save_attn'] = bool(args.save_attn)
    else:
        cfg = dict(cfg)  # Создаем копию, чтобы не менять оригинал
        cfg.setdefault('logging', {})
        cfg['logging'].setdefault('save_attn', False)

    pretty_print_run(cfg)
    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])

    # Сборка пайплайна
    train_loader, val_loader, train_labels, n_channels, n_subjects = (
        build_loaders(cfg)
    )
    model = build_model(cfg, n_channels, n_subjects).to(device)
    criterion = build_criterion(cfg, train_labels).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    # Запуск обучения
    history, final_metrics, val_outputs, attn_stats = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=device
    )

    # Вывод результатов
    print("\nИтоговые метрики валидации (Fold 1):")
    print_metrics(final_metrics)

    # Сохранение артефактов (модель, логи, графики)
    save_artifacts(
        cfg,
        final_metrics,
        history,
        val_outputs,
        attn_stats,
        model
    )

    print("\nОБУЧЕНИЕ PHASE 4B ЗАВЕРШЕНО")
    return final_metrics


if __name__ == '__main__':
    main()
