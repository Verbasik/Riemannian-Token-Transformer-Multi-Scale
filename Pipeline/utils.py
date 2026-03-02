# file: utils.py
# -*- coding: utf-8 -*-
"""
Общие вспомогательные утилиты.

Содержит функции, не привязанные к конкретному домену, такие как:
1. Установка seed для воспроизводимости экспериментов (PyTorch, NumPy, CUDA).
2. Обеспечение совместимости версий NumPy при десериализации pickle-файлов.
3. Форматированный вывод метрик и конфигурации запуска в консоль.

Эти утилиты используются во всех скриптах обучения и тестирования.
"""

# =============================================================================
# Standard Libraries
# =============================================================================
import sys
from typing import Any, Dict

# =============================================================================
# Third-Party Libraries
# =============================================================================
import numpy as np
import torch


def _ensure_numpy_pickle_compat() -> None:
    """
    Description:
    ---------------
        Обеспечивает совместимость при десериализации pickle-файлов NumPy,
        созданных в старых версиях библиотеки.
        
        Проблема: В новых версиях NumPy (>=1.25) модуль `numpy.core` был
        перемещен в `numpy._core`. Старые pickle-файлы могут ссылаться на
        старый путь, что вызывает ошибку при загрузке.
        
        Решение: Создается алиас в `sys.modules`, позволяющий найти новый
        модуль по старому имени.

    Args:
    ---------------
        Нет аргументов.

    Returns:
    ---------------
        None

    Raises:
    ---------------
        Нет явных исключений (ошибки импорта игнорируются).

    Examples:
    ---------------
        >>> _ensure_numpy_pickle_compat()
        >>> # Теперь можно безопасно загружать старые pickle с массивами
    """
    try:
        # Попытка импорта нового пути (NumPy >= 1.25)
        import numpy._core  # noqa: F401
    except ImportError:
        try:
            # Если новый путь не найден, пробуем старый и создаем алиас
            import numpy.core as ncore
            sys.modules['numpy._core'] = ncore
        except ImportError:
            # Если ничего не найдено, пропускаем (возможно, очень старая версия)
            pass


def set_seed(seed: int) -> None:
    """
    Description:
    ---------------
        Устанавливает фиксированный seed (зерно) для всех генераторов
        случайных чисел в проекте. Это критически важно для обеспечения
        воспроизводимости результатов экспериментов.
        
        Инициализирует:
        - PyTorch CPU RNG.
        - PyTorch CUDA RNG (для всех GPU).
        - NumPy RNG.
        - Настройки cuDNN для детерминированных алгоритмов.

    Args:
    ---------------
        seed: int - Целочисленное значение зерна.

    Returns:
    ---------------
        None

    Raises:
    ---------------
        Нет явных исключений.

    Examples:
    ---------------
        >>> set_seed(42)
        >>> # Все последующие случайные операции будут детерминированы
    """
    torch.manual_seed(seed)
    
    # Установка seed для всех GPU
    torch.cuda.manual_seed_all(seed)
    
    # Установка seed для NumPy
    np.random.seed(seed)
    
    # Настройка cuDNN для детерминизма
    # deterministic=True гарантирует одинаковые результаты, но может быть медленнее
    torch.backends.cudnn.deterministic = True
    
    # benchmark=False отключает поиск оптимального алгоритма свертки
    # (который может быть недетерминированным)
    torch.backends.cudnn.benchmark = False


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Description:
    ---------------
        Красиво выводит словарь с метриками в консоль в формате "ключ: значение".
        Значения форматируются до 4 знаков после запятой.

    Args:
    ---------------
        metrics: Dict[str, float] - Словарь метрик (например, {'accuracy': 0.95}).
        prefix: str - Префикс, добавляемый к заголовку вывода (опционально).

    Returns:
    ---------------
        None

    Raises:
    ---------------
        Нет явных исключений.

    Examples:
    ---------------
        >>> metrics = {'loss': 0.12345, 'acc': 0.98765}
        >>> print_metrics(metrics, prefix="Val")
        Val Метрики:
          loss: 0.1235
          acc: 0.9877
    """
    print(f"\n{prefix} Метрики:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


def pretty_print_run(cfg: Dict[str, Any]) -> None:
    """
    Description:
    ---------------
        Выводит ключевые параметры конфигурации запуска в читаемом виде.
        Используется для быстрой проверки настроек эксперимента перед стартом.
        Группирует параметры по категориям: Данные, Модель, Обучение, Оптимизация.

    Args:
    ---------------
        cfg: Dict[str, Any] - Полный словарь конфигурации проекта.

    Returns:
    ---------------
        None

    Raises:
    ---------------
        KeyError: Если в конфиге отсутствуют ожидаемые ключи ('data', 'model' и т.д.).

    Examples:
    ---------------
        >>> cfg = default_config()
        >>> pretty_print_run(cfg)
        ==================================================
        Конфигурация запуска Phase 4B
        ==================================================
        Данные: 5 испытуемых, нормализация: zscore_hybrid
        ...
    """
    # Извлечение основных секций конфига для краткости
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    train_cfg = cfg['training']
    sched_cfg = cfg['scheduler']
    opt_cfg = cfg['optimizer']
    loss_cfg = cfg['loss']

    print("\n" + "=" * 50)
    print("Конфигурация запуска Phase 4B")
    print("=" * 50)
    
    # Блок данных
    n_subjects = len(data_cfg['subject_ids'])
    norm_type = data_cfg['normalize']
    print(f"Данные: {n_subjects} испытуемых, нормализация: {norm_type}")
    
    # Блок модели
    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    print(f"Модель: rtt_ms, d_model={d_model}, слоев={n_layers}")
    
    # Блок обучения
    n_epochs = train_cfg['n_epochs']
    batch_size = train_cfg['batch_size']
    lr = train_cfg['learning_rate']
    print(
        f"Обучение: {n_epochs} эпох, batch_size={batch_size}, "
        f"lr={lr:.0e}"
    )
    
    # Блок оптимизации
    sch_name = sched_cfg['name']
    opt_name = opt_cfg['name']
    loss_type = loss_cfg['type']
    print(
        f"Планировщик: {sch_name}, Оптимизатор: {opt_name}, "
        f"Потери: {loss_type}"
    )
    
    print("=" * 50)