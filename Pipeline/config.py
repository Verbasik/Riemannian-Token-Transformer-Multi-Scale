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
PREPROCESSED_DIR = DATA_ROOT / "derivatives/preprocessed_pkl"

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
            'subject_ids': ['sub-04'],  # 'sub-03', 'sub-04', 'sub-05'
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
            'attn_heads': 1,
            'gating': False,
            'cov_type': 'corr',
            'use_subject_embed': True,     # Phase 4B-6: Subject embeddings ENABLED
            'subject_embed_dim': 16,       # Embedding dimension
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
        },
        'cv': {'n_splits': 5, 'random_state': RANDOM_SEED},
        'optimizer': {'name': 'adamw', 'betas': [0.9, 0.999]},
        'scheduler': {'name': 'cosine', 'T_max': 20, 'warmup_epochs': 3},
        'loss': {'type': 'cb_focal', 'beta': 0.9999, 'gamma': 1.5},
        'seed': RANDOM_SEED,
        'device': device,
        'checkpoint_dir': f'Train/checkpoints/phase4b_5subjects_{device.upper()}',
        'results_dir': f'Train/results/phase4b_5subjects_{device.upper()}',
    }