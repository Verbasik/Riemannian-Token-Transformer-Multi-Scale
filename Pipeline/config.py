# file: config.py
# -*- coding: utf-8 -*-
"""
Конфигурация для эксперимента Phase 4B.

Содержит глобальные константы и функцию для создания словаря с параметрами
запуска, включая настройки данных, модели и процесса обучения.
"""
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# =============================================================================
# Глобальные константы
# =============================================================================
# Определяем константы для путей, чтобы избежать "магических строк" в коде.
# Работаем из директории Best/, поэтому корень проекта - родительская директория
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT
JSON_DIR = DATA_ROOT / "json"
PREPROCESSED_DIR = Path("/mnt/data/EEG/preprocessed_pkl")

# Константы для конфигурации модели и обучения
RANDOM_SEED = 42
EPSILON = 1e-4  # Малая константа для численной стабильности


def default_config(device_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Description:
    ---------------
        Создает конфигурационный словарь по умолчанию для эксперимента.
        Автоматически определяет доступность CUDA и настраивает параметры
        соответствующим образом.

    Args:
    ---------------
        device_hint: Подсказка для выбора устройства ('cuda' или 'cpu').

    Returns:
    ---------------
        Словарь с полной конфигурацией.
    """
    use_cuda = torch.cuda.is_available() and (device_hint or 'cuda') == 'cuda'
    device = 'cuda' if use_cuda else 'cpu'
    return {
        'data': {
            'data_dir': PREPROCESSED_DIR,
            # Текущий дефолтный сценарий: одиночный субъект sub-03
            'subject_ids': ['sub-03'],
            'task': 'imagine',
            'normalize': 'zscore_hybrid',  # Phase 4B-6: Hybrid normalization (subject centering + global scaling)
            'exclude_channels': [124],
        },
        'model': {
            'n_channels': 125,
            'n_classes': 8,
            'proj_channels': 24,
            'window_size_small': 128,
            'stride_small': 96,
            'window_size_large': 256,
            'stride_large': 128,
            'd_model': 128,
            'n_heads': 4,
            'ff_dim': 256,
            'n_layers': 2,
            'dropout': 0.1,
            'eps': EPSILON,
            'attn_heads': 1,  # A2/A5: лучший результат на DS1 при 1 голове
            'cov_type': 'corr',
            'oas_min_alpha': 0.1,  # B2 база: clamp 0.1
            'use_subject_embed': True,     # Phase 4B-6: Subject embeddings ENABLED
            'subject_embed_dim': 16,       # Embedding dimension (A6 best on DS1)
            'subject_embed_dropout': 0.2,  # A6 best: dropout 0.2
        },
        'training': {
            'n_epochs': 50,
            'batch_size': 16 if device == 'cuda' else 8,
            'learning_rate': 3e-4,
            'weight_decay': 1e-4,
            'early_stopping_patience': 8,
            'use_amp': device == 'cuda',
            'grad_clip': 1.0,
            'num_workers': 4 if device == 'cuda' else 0,
            'pin_memory': device == 'cuda',
            # Для CUDA-режима ускоряем подачу батчей.
            'persistent_workers': device == 'cuda',
            'prefetch_factor': 4 if device == 'cuda' else 2,
        },
        'cv': {
            'n_splits': 5,
            'random_state': RANDOM_SEED,
            'mode': 'stratified_group',  # 'stratified' (default behavior), 'stratified_group' (subject-aware), 'loso' (leave-one-subject-out)
            'fold_index': 0,  # Используется для специфичной fold индекса
        },
        'optimizer': {
            'name': 'adamw', 'betas': [0.9, 0.999],
            # A6 best: отдельный weight_decay для subject embeddings
            'subject_embed_weight_decay': 5e-4,
        },
        # Дефолт: настройки из A2 (лучшие на DS1)
        'scheduler': {'name': 'cosine', 'T_max': 20, 'warmup_epochs': 3},
        # D1: фиксируем лучшие гиперпараметры A2
        'loss': {'type': 'cb_focal', 'beta': 0.999, 'gamma': 1.75},
        'logging': {
            'save_attn': False  # сохранять ли усреднённые attention веса на валидации
        },
        'seed': RANDOM_SEED,
        'device': device,
        'checkpoint_dir': f'Train/checkpoints/phase4b_5subjects_{device.upper()}',
        'results_dir': f'Train/results/phase4b_5subjects_{device.upper()}',
    }
