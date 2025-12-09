# file: utils.py
# -*- coding: utf-8 -*-
"""
Общие вспомогательные утилиты.

Содержит функции, не привязанные к конкретному домену, такие как
установка seed для воспроизводимости, обеспечение совместимости NumPy
и функции для красивого вывода в консоль.
"""
import sys
from typing import Any, Dict

import numpy as np
import torch


def _ensure_numpy_pickle_compat() -> None:
    """Обеспечивает совместимость при десериализации pickle-файлов NumPy."""
    try:
        import numpy._core
    except ImportError:
        try:
            import numpy.core as ncore
            sys.modules['numpy._core'] = ncore
        except ImportError:
            pass


def set_seed(seed: int) -> None:
    """Устанавливает seed для всех генераторов случайных чисел."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """Красиво выводит словарь с метриками в консоль."""
    print(f"\n{prefix} Метрики:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


def pretty_print_run(cfg: Dict[str, Any]) -> None:
    """Выводит ключевые параметры конфигурации запуска в читаемом виде."""
    d, m, t = cfg['data'], cfg['model'], cfg['training']
    sch, opt, ls = cfg['scheduler'], cfg['optimizer'], cfg['loss']
    print("\n" + "="*50)
    print("Конфигурация запуска Phase 4B")
    print("="*50)
    print(f"Данные: {len(d['subject_ids'])} испытуемых, нормализация: {d['normalize']}")
    print(f"Модель: rtt_ms, d_model={m['d_model']}, слоев={m['n_layers']}")
    print(f"Обучение: {t['n_epochs']} эпох, batch_size={t['batch_size']}, lr={t['learning_rate']:.0e}")
    print(f"Планировщик: {sch['name']}, Оптимизатор: {opt['name']}, Потери: {ls['type']}")
    print("="*50)