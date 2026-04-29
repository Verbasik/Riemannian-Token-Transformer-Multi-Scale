# -*- coding: utf-8 -*-
"""
Конфигурация для эксперимента Phase 4B.

Файл централизует:
    - Пути к данным
    - Гиперпараметры модели
    - Настройки обучения
    - Настройки кросс-валидации
    - Параметры оптимизатора и scheduler
"""

# =============================================================================
# Стандартная библиотека
# =============================================================================
import os
from pathlib import Path
from typing import Any, Dict, Optional

# =============================================================================
# Сторонние библиотеки
# =============================================================================
import torch

# =============================================================================
# Глобальные константы
# =============================================================================

PROJECT_ROOT: Path = Path(__file__).parent.parent
# Корневая директория проекта (на уровень выше папки config)

DATA_ROOT: Path = PROJECT_ROOT
# Базовая директория данных

JSON_DIR: Path = DATA_ROOT / "json"
# Папка с JSON-описаниями датасетов

PREPROCESSED_DIR: Path = Path("/mnt/data/EEG/preprocessed_pkl")
# Legacy-путь к предварительно обработанным данным

RANDOM_SEED: int = 42
# Фиксация seed для воспроизводимости

EPSILON: float = 1e-4
# Численная константа для стабильности вычислений


# =============================================================================
# Вспомогательная функция
# =============================================================================
def _resolve_preprocessed_dir() -> Path:
    """
    Description:
    ---------------
        Определяет существующую директорию с preprocessed pkl-файлами.

    Returns:
    ---------------
        Path: Найденный путь к данным.
    """
    env_dir: Optional[str] = os.getenv("EEG_PREPROCESSED_DIR")

    candidates = []

    if env_dir:
        candidates.append(Path(env_dir))  # Приоритет переменной окружения

    candidates.extend(
        [
            Path("/mnt/data/data/derivatives/preprocessed_pkl"),
            Path("/mnt/data/derivatives/preprocessed_pkl"),
            Path("/mnt/data/EEG/preprocessed_pkl"),
            PROJECT_ROOT / "derivatives" / "preprocessed_pkl",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


# =============================================================================
# Основная функция конфигурации
# =============================================================================
def default_config(device_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Description:
    ---------------
        Формирует словарь конфигурации эксперимента Phase 4B.

    Args:
    ---------------
        device_hint: Подсказка устройства ('cuda' или 'cpu').

    Returns:
    ---------------
        Dict[str, Any]: Конфигурация эксперимента.
    """
    use_cuda: bool = torch.cuda.is_available() and (
        (device_hint or "cuda") == "cuda"
    )

    device: str = "cuda" if use_cuda else "cpu"
    data_dir: Path = _resolve_preprocessed_dir()

    return {
        "data": {
            "data_dir": data_dir,
            # Путь к preprocessed данным

            "subject_ids": [
                "sub-01", "sub-02", "sub-03", "sub-04", "sub-05"
            ],
            # Список идентификаторов субъектов для обучения/оценки

            "task": "imagine",
            # Тип задачи (например: imagine / overt / rest)

            "normalize": "zscore_hybrid",
            # Метод нормализации входных данных

            "exclude_channels": [124],
            # Индексы каналов EEG, исключаемых из анализа
        },
        "model": {
            "n_channels": 125,
            # Общее количество входных EEG-каналов

            "n_classes": 8,
            # Количество целевых классов классификации

            "proj_channels": 24,
            # Размерность проекционного слоя перед attention

            "window_size_small": 128,
            # Размер малого временного окна (в сэмплах)

            "stride_small": 96,
            # Шаг скольжения малого окна

            "window_size_large": 256,
            # Размер большого временного окна

            "stride_large": 128,
            # Шаг скольжения большого окна

            "d_model": 128,
            # Размерность скрытого представления Transformer

            "n_heads": 4,
            # Количество attention-голов в Transformer

            "ff_dim": 256,
            # Размерность feed-forward слоя

            "n_layers": 2,
            # Количество Transformer-блоков

            "dropout": 0.1,
            # Dropout для регуляризации модели

            "eps": EPSILON,
            # Константа для численной стабильности

            "attn_heads": 1,
            # Количество attention-голов в covariance-модуле

            "cov_type": "corr",
            # Тип ковариационной матрицы ('cov' или 'corr')

            "oas_min_alpha": 0.1,
            # Минимальное значение shrinkage для OAS

            "use_subject_embed": True,
            # Использовать ли embedding субъекта

            "subject_embed_dim": 16,
            # Размерность embedding-вектора субъекта

            "subject_embed_dropout": 0.2,
            # Dropout для subject embedding

            "unknown_subject_policy": "auto",
            # Поведение для subject-held-out validation:
            # auto / error / zero / mean
        },
        "training": {
            "n_epochs": 50,
            # Максимальное количество эпох обучения

            "batch_size": 16 if device == "cuda" else 8,
            # Размер батча (увеличен для GPU)

            "learning_rate": 3e-4,
            # Начальная скорость обучения

            "weight_decay": 1e-4,
            # L2-регуляризация весов

            "early_stopping_patience": 8,
            # Количество эпох без улучшения до остановки

            "use_amp": device == "cuda",
            # Использовать ли mixed precision

            "grad_clip": 1.0,
            # Ограничение нормы градиента

            "num_workers": 0,
            # Количество worker-процессов DataLoader.
            # Данные уже загружены в память; на Python 3.14 forkserver
            # пытается pickle-ить большой dataset и может падать.

            "pin_memory": device == "cuda",
            # Закреплять ли память (ускоряет передачу на GPU)

            "persistent_workers": False,
            # Сохранять worker-процессы между эпохами

            "prefetch_factor": 4 if device == "cuda" else 2,
            # Количество предварительно загружаемых батчей

            "allow_multiprocessing_dataloader": False,
            # Разрешить num_workers > 0. Для Python 3.14 по умолчанию
            # выключено из-за forkserver + большого in-memory dataset.
        },
        "cv": {
            "protocol": "within_subject",
            # Протокол оценки:
            # within_subject: каждый субъект есть в train и val
            # subject_heldout: val-субъекты полностью отсутствуют в train

            "n_splits": 5,
            # Количество фолдов в кросс-валидации

            "random_state": RANDOM_SEED,
            # Seed для разбиения на фолды

            "mode": "within_subject",
            # Режим CV:
            # within_subject / stratified / stratified_group / loso

            "fold_index": 0,
            # Индекс конкретного фолда
        },
        "evaluation": {
            "pipeline": "both",
            # Полный evaluation pipeline:
            # si: Subject-Independent pooled personalized model
            # sd: Subject-Dependent per-subject models
            # both: последовательно запустить si и sd

            "si_use_subject_embed": True,
            # SI использует одну общую модель и subject embeddings.

            "sd_use_subject_embed": False,
            # SD обучает отдельную модель на каждого субъекта; embedding
            # субъекта по умолчанию отключен, т.к. персонализация задается
            # самой отдельной моделью.
        },
        "optimizer": {
            "name": "adamw",
            # Тип оптимизатора

            "betas": [0.9, 0.999],
            # Параметры momentum для AdamW

            "subject_embed_weight_decay": 5e-4,
            # Отдельная регуляризация для embedding субъекта
        },
        "scheduler": {
            "name": "cosine",
            # Тип scheduler

            "T_max": 20,
            # Период косинусного scheduler

            "warmup_epochs": 3,
            # Количество эпох warmup
        },
        "loss": {
            "type": "cb_focal",
            # Тип функции потерь

            "beta": 0.999,
            # Параметр балансировки классов

            "gamma": 1.75,
            # Параметр фокусировки в Focal Loss
        },
        "logging": {
            "save_attn": False,
            # Сохранять ли attention-веса при валидации
        },
        "seed": RANDOM_SEED,
        # Общий seed эксперимента

        "device": device,
        # Устройство вычислений ('cpu' или 'cuda')

        "checkpoint_dir": (
            f"Train/checkpoints/phase4b_5subjects_{device.upper()}"
        ),
        # Директория сохранения чекпоинтов

        "results_dir": (
            f"Train/results/phase4b_5subjects_{device.upper()}"
        ),
        # Директория сохранения результатов эксперимента
    }
