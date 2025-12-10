# file: train.py
# -*- coding: utf-8 -*-
"""
Главный исполняемый файл для запуска обучения модели Phase 4B.

Этот скрипт выполняет следующие шаги:
1. Загружает конфигурацию.
2. Инициализирует окружение (seed, устройство).
3. Создает ("строит") загрузчики данных, модель, функцию потерь и оптимизатор.
4. Запускает основной цикл обучения из модуля `trainer`.
5. Сохраняет итоговые результаты.
"""
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# Импорт компонентов из доменных модулей
from config import default_config
from data_loader import (ChiscoDataset, ChiscoSubset,
                         compute_channelwise_stats, compute_subjectwise_stats,
                         compute_hybrid_stats, create_subject_mapping,
                         get_stratified_cv_splits, load_all_data_metaclass)
from model import RTTMultiScale
from trainer import (ClassBalancedFocalLoss, save_artifacts, train_loop)
from utils import pretty_print_run, print_metrics, set_seed

# =============================================================================
# Функции-конструкторы (Builders)
# =============================================================================

def build_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, np.ndarray, int, int]:
    """Создает и настраивает загрузчики данных."""
    samples = load_all_data_metaclass(
        data_dir=cfg['data']['data_dir'],
        subject_ids=cfg['data']['subject_ids'],
        task=cfg['data']['task'],
    )

    # Create subject mapping for embeddings
    subject_mapping = create_subject_mapping(samples)
    n_subjects = len(subject_mapping)

    dataset = ChiscoDataset(
        samples=samples,
        normalize=cfg['data']['normalize'],
        exclude_channels=cfg['data'].get('exclude_channels'),
        subject_mapping=subject_mapping
    )
    labels = np.array([s['label'] for s in samples])
    splits = get_stratified_cv_splits(
        labels, cfg['cv']['n_splits'], cfg['cv']['random_state']
    )
    train_idx, val_idx = splits[0]

    # Compute normalization statistics based on mode
    if cfg['data']['normalize'] == 'zscore_hybrid':
        # NEW: Hybrid normalization (subject-wise centering + global scaling)
        stats_dict = compute_hybrid_stats(
            samples, train_idx, cfg['data'].get('exclude_channels')
        )
        dataset.norm_stats = stats_dict
    elif cfg['data']['normalize'] == 'zscore_subject_channel':
        # Subject-wise normalization (DEPRECATED: causes overfitting)
        stats_dict = compute_subjectwise_stats(
            samples, train_idx, cfg['data'].get('exclude_channels')
        )
        dataset.norm_stats = stats_dict
    elif cfg['data']['normalize'] == 'zscore_dataset_channel':
        # Global normalization (baseline)
        mean_c, std_c = compute_channelwise_stats(
            samples, train_idx, cfg['data'].get('exclude_channels')
        )
        dataset.norm_stats = {'mean': mean_c, 'std': std_c}
    
    # Train loader: optional class-balanced sampling without dropping data
    train_subset = ChiscoSubset(dataset, train_idx)
    use_sampler = cfg['training'].get('use_weighted_sampler', False)
    # Disable sampler when using CB-Focal unless explicitly allowed
    if cfg['loss'].get('type') == 'cb_focal' and not cfg['training'].get('allow_sampler_with_cb_focal', False):
        use_sampler = False

    if use_sampler:
        n_classes = cfg['model']['n_classes']
        train_labels_arr = labels[train_idx]
        class_counts = np.bincount(train_labels_arr, minlength=n_classes)
        # w_c = total / (num_classes * count_c)
        weights_per_class = (len(train_labels_arr) / (n_classes * np.clip(class_counts, 1, None))).astype(np.float64)
        sample_weights = weights_per_class[train_labels_arr]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_idx),
            replacement=True
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=cfg['training']['batch_size'],
            sampler=sampler,
            shuffle=False,
            num_workers=cfg['training']['num_workers'],
            pin_memory=cfg['training']['pin_memory']
        )
    else:
        train_loader = DataLoader(
            train_subset, batch_size=cfg['training']['batch_size'],
            shuffle=True, num_workers=cfg['training']['num_workers'],
            pin_memory=cfg['training']['pin_memory']
        )
    val_loader = DataLoader(
        ChiscoSubset(dataset, val_idx), batch_size=cfg['training']['batch_size'],
        shuffle=False, num_workers=cfg['training']['num_workers'],
        pin_memory=cfg['training']['pin_memory']
    )
    
    # Определяем эффективное число каналов
    eeg_shape = dataset[0]['eeg'].shape
    effective_channels = eeg_shape[0]

    return train_loader, val_loader, labels[train_idx], effective_channels, n_subjects


def build_model(cfg: Dict[str, Any], n_channels: int, n_subjects: int) -> nn.Module:
    """Создает экземпляр модели на основе конфигурации."""
    m_cfg = cfg['model']
    return RTTMultiScale(
        n_channels=n_channels, n_classes=m_cfg['n_classes'],
        proj_channels=m_cfg['proj_channels'],
        window_size_small=m_cfg['window_size_small'], stride_small=m_cfg['stride_small'],
        window_size_large=m_cfg['window_size_large'], stride_large=m_cfg['stride_large'],
        d_model=m_cfg['d_model'], n_heads=m_cfg['n_heads'], ff_dim=m_cfg['ff_dim'],
        n_layers=m_cfg['n_layers'], dropout=m_cfg['dropout'], eps=m_cfg['eps'],
        attn_heads=m_cfg.get('attn_heads', 1), gating=m_cfg.get('gating', False),
        cov_type=m_cfg.get('cov_type', 'corr'),
        use_subject_embed=m_cfg.get('use_subject_embed', False),
        n_subjects=n_subjects,
        subject_embed_dim=m_cfg.get('subject_embed_dim', 16)
    )


def build_criterion(cfg: Dict[str, Any], train_labels: np.ndarray) -> nn.Module:
    """Создает экземпляр функции потерь."""
    loss_cfg = cfg['loss']
    if loss_cfg['type'] == 'cb_focal':
        counts = np.bincount(train_labels, minlength=cfg['model']['n_classes'])
        return ClassBalancedFocalLoss(
            class_counts=counts, beta=loss_cfg['beta'], gamma=loss_cfg['gamma']
        )
    return nn.CrossEntropyLoss()


def build_optimizer_and_scheduler(model: nn.Module, cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    """Создает оптимизатор и планировщик скорости обучения."""
    opt_cfg = cfg['optimizer']
    train_cfg = cfg['training']
    if opt_cfg['name'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay'], betas=tuple(opt_cfg['betas'])
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay']
        )
    
    sched_cfg = cfg['scheduler']
    scheduler = None
    if sched_cfg.get('name') == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup = int(sched_cfg.get('warmup_epochs', 0))
        if warmup > 0:
            sched1 = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
            sched2 = CosineAnnealingLR(optimizer, T_max=max(sched_cfg['T_max'] - warmup, 1))
            scheduler = SequentialLR(optimizer, schedulers=[sched1, sched2], milestones=[warmup])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=sched_cfg['T_max'])
            
    return optimizer, scheduler

# =============================================================================
# Основная точка входа
# =============================================================================

def main() -> None:
    """Основная функция, запускающая весь процесс обучения."""
    # 1. Конфигурация и инициализация
    cfg = default_config()
    pretty_print_run(cfg)
    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])

    # 2. Создание компонентов
    train_loader, val_loader, train_labels, n_channels, n_subjects = build_loaders(cfg)
    model = build_model(cfg, n_channels, n_subjects).to(device)
    criterion = build_criterion(cfg, train_labels).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    # 3. Запуск обучения
    _, final_metrics = train_loop(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        cfg=cfg, device=device
    )

    # 4. Вывод и сохранение результатов
    print("\nИтоговые метрики валидации (Fold 1):")
    print_metrics(final_metrics)
    save_artifacts(cfg, final_metrics, model)
    print("\nОБУЧЕНИЕ PHASE 4B ЗАВЕРШЕНО")


if __name__ == '__main__':
    main()
