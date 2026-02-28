# file: data_loader.py
# -*- coding: utf-8 -*-
"""
Домен данных: загрузка, предобработка и создание датасетов.

Содержит классы ChiscoDataset и ChiscoSubset, а также все утилиты для
загрузки данных из pkl-файлов, преобразования меток в мета-классы и
создания стратифицированных разрезов для кросс-валидации.
"""
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from torch.utils.data import Dataset

from config import EPSILON, JSON_DIR
from utils import _ensure_numpy_pickle_compat

# =============================================================================
# Subject ID Mapping (for embeddings)
# =============================================================================

def create_subject_mapping(samples: List[Dict]) -> Dict[str, int]:
    """
    Создаёт маппинг subject_id -> integer index для embedding layer.

    Args:
        samples: Список всех образцов данных.

    Returns:
        Dict[subject_id -> int]
    """
    unique_subjects = sorted(set(s['subject'] for s in samples))
    return {subject_id: idx for idx, subject_id in enumerate(unique_subjects)}

# =============================================================================
# Классы Dataset
# =============================================================================

class ChiscoDataset(Dataset):
    """
    Description:
    ---------------
        Класс Dataset для PyTorch, предназначенный для работы с данными
        ЭЭГ воображаемой речи из набора данных Chisco.
    """
    def __init__(
        self,
        samples: List[Dict],
        transform: Optional[Callable] = None,
        normalize: Optional[str] = 'zscore',
        augment_prob: float = 0.5,
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
        exclude_channels: Optional[List[int]] = None,
        subject_mapping: Optional[Dict[str, int]] = None,
    ):
        self.samples = samples
        self.transform = transform
        self.normalize = normalize
        self.augment_prob = augment_prob
        self.norm_stats = norm_stats
        self.exclude_channels = exclude_channels if exclude_channels else []
        self.subject_mapping = subject_mapping if subject_mapping else {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        eeg = sample['eeg']
        label = sample['label']

        if eeg.ndim == 3 and eeg.shape[0] == 1:
            eeg = eeg.squeeze(0)

        eeg = eeg.astype(np.float32)

        if self.exclude_channels:
            keep_mask = np.ones(eeg.shape[0], dtype=bool)
            keep_mask[self.exclude_channels] = False
            eeg = eeg[keep_mask, :]

        if self.normalize == 'zscore':
            mean = eeg.mean(axis=-1, keepdims=True)
            std = eeg.std(axis=-1, keepdims=True) + EPSILON
            eeg = (eeg - mean) / std
        elif self.normalize == 'minmax':
            min_val = eeg.min(axis=-1, keepdims=True)
            max_val = eeg.max(axis=-1, keepdims=True)
            eeg = (eeg - min_val) / (max_val - min_val + EPSILON)
        elif self.normalize == 'zscore_dataset_channel':
            if self.norm_stats is None:
                raise ValueError("Для 'zscore_dataset_channel' требуются `norm_stats`.")
            mean_c = self.norm_stats['mean'].reshape(-1, 1)
            std_c = (self.norm_stats['std'].reshape(-1, 1) + EPSILON)
            eeg = (eeg - mean_c) / std_c
        elif self.normalize == 'zscore_subject_channel':
            # Subject-wise normalization (DEPRECATED: causes overfitting)
            if self.norm_stats is None:
                raise ValueError("Для 'zscore_subject_channel' требуются `norm_stats`.")

            subject_id = sample['subject']
            if subject_id not in self.norm_stats:
                raise ValueError(f"Subject '{subject_id}' не найден в norm_stats.")

            mean_c, std_c = self.norm_stats[subject_id]
            mean_c = mean_c.reshape(-1, 1)
            std_c = (std_c.reshape(-1, 1) + EPSILON)
            eeg = (eeg - mean_c) / std_c
        elif self.normalize == 'zscore_hybrid':
            # NEW: Hybrid normalization (subject-wise centering + global scaling)
            if self.norm_stats is None:
                raise ValueError("Для 'zscore_hybrid' требуются `norm_stats`.")

            subject_id = sample['subject']
            if 'mean_per_subject' not in self.norm_stats or 'std_global' not in self.norm_stats:
                raise ValueError("norm_stats должны содержать 'mean_per_subject' и 'std_global'.")

            # Если subject отсутствует в train (LOSO/novel), используем нулевое центрирование
            if subject_id in self.norm_stats['mean_per_subject']:
                mean_c = self.norm_stats['mean_per_subject'][subject_id]
            else:
                mean_c = np.zeros(eeg.shape[0], dtype=np.float32)
            std_global = self.norm_stats['std_global']

            mean_c = mean_c.reshape(-1, 1)
            std_global = (std_global.reshape(-1, 1) + EPSILON)

            # Step 1: Subject-wise centering
            eeg_centered = eeg - mean_c
            # Step 2: Global scaling
            eeg = eeg_centered / std_global

        if self.transform and np.random.rand() < self.augment_prob:
            eeg = self.transform(eeg)

        eeg_tensor = torch.from_numpy(eeg.copy())
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Convert subject_id to integer for embedding
        subject_id = sample['subject']
        subject_id_int = self.subject_mapping.get(subject_id, 0)  # default to 0 if not found

        return {
            'eeg': eeg_tensor,
            'label': label_tensor,
            'subject': subject_id,
            'subject_id': torch.tensor(subject_id_int, dtype=torch.long),
            'text': sample['text']
        }


class ChiscoSubset(Dataset):
    """
    Description:
    ---------------
        Представляет собой подвыборку из основного `ChiscoDataset`.
    """
    def __init__(self, dataset: ChiscoDataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = indices
        self.samples = [dataset.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[self.indices[idx]]
        # Добавляем идентификатор исходного образца для последующего логгинга предсказаний
        item['sample_id'] = int(self.indices[idx])
        return item

# =============================================================================
# Функции загрузки и обработки данных
# =============================================================================

def load_class_mapping(json_path: Path = JSON_DIR) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Загружает сопоставления классов из JSON-файлов."""
    with open(json_path / "classnumber.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)
    with open(json_path / "textmaps.json", "r", encoding="utf-8") as f:
        text_to_class = json.load(f)
    return class_names, text_to_class


def load_all_data(
    data_dir: Path, subject_ids: List[str], task: str = "imagine", verbose: bool = True
) -> List[Dict[str, Any]]:
    """Загружает все pkl-файлы для указанных испытуемых и задачи."""
    if verbose:
        print(f"Загрузка данных для испытуемых: {subject_ids}, задача: {task}")
    _, text_to_class = load_class_mapping()
    all_samples = []
    for subject_id in subject_ids:
        subject_dir = data_dir / subject_id / "eeg"
        if not subject_dir.exists():
            if verbose: print(f"⚠️  {subject_id}: директория не найдена, пропуск.")
            continue
        pkl_files = sorted(subject_dir.glob(f"*task-{task}*.pkl"))
        if verbose: print(f"  {subject_id}: найдено {len(pkl_files)} файлов.")
        for pkl_file in pkl_files:
            run_id = pkl_file.stem.split("_run-")[1].split("_")[0]
            _ensure_numpy_pickle_compat()
            with open(pkl_file, "rb") as f:
                data_list = pickle.load(f)
            for item in data_list:
                text, eeg = item['text'], item['input_features']
                label = text_to_class.get(text, -1)
                if label == -1:
                    if verbose: print(f"⚠️  Текст '{text}' не найден в textmaps.json.")
                    continue
                all_samples.append({
                    'subject': subject_id, 'run': run_id, 'text': text,
                    'label': int(label), 'eeg': eeg
                })
    if verbose: print(f"✅ Всего загружено сэмплов: {len(all_samples)}")
    return all_samples


def load_metaclass_mapping(json_path: Path = JSON_DIR) -> Tuple[Dict[str, str], Dict[int, int]]:
    """Загружает сопоставление из 39 исходных классов в 8 мета-классов."""
    metaclass_file = json_path / "metaclasses.json"
    with open(metaclass_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    metaclass_names = data['metaclass_names']
    class_to_meta = {int(k): v for k, v in data['metaclass_mapping'].items()}
    return metaclass_names, class_to_meta


def load_all_data_metaclass(
    data_dir: Path, subject_ids: List[str], task: str = "imagine", verbose: bool = True
) -> List[Dict[str, Any]]:
    """Загружает все данные и преобразует метки в 8 мета-классов."""
    samples = load_all_data(data_dir, subject_ids, task, verbose)
    _, class_to_meta = load_metaclass_mapping()
    converted_samples = []
    for sample in samples:
        orig_label = sample['label']
        meta_label = class_to_meta.get(orig_label, -1)
        if meta_label != -1:
            converted_sample = sample.copy()
            converted_sample['label'] = meta_label
            converted_sample['original_label'] = orig_label
            converted_samples.append(converted_sample)
    if verbose:
        print("\n" + "="*80)
        print(f"Конвертация в мета-классы: {len(converted_samples)} сэмплов.")
        print("="*80 + "\n")
    return converted_samples


def get_stratified_cv_splits(
    labels: np.ndarray, n_splits: int, random_state: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Создает стратифицированные K-Fold разрезы для кросс-валидации."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(labels)), labels))


def get_stratified_group_cv_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Stratified K-Fold с группировкой по субъектам.

    Гарантирует, что одна и та же группа (субъект) не будет одновременно в train и val.
    Это критично для оценки cross-subject обобщаемости.

    Description:
    ---------------
        Используется для контроля утечки по субъектам в кросс-валидации.
        Сохраняет распределение классов в каждом fold при группировке по субъектам.

    Args:
    ---------------
        labels: np.ndarray [N] - мета-классы (0..7)
        groups: np.ndarray [N] - subject_id indices (целые числа для группировки)
        n_splits: int - число folds (по умолчанию 5)
        random_state: int - seed для воспроизводимости

    Returns:
    ---------------
        List[(train_idx, val_idx)] - стратифицированные разбиения с группировкой

    Example:
    ---------------
        >>> samples = load_all_data_metaclass(...)
        >>> labels = np.array([s['label'] for s in samples])
        >>> subject_mapping = create_subject_mapping(samples)
        >>> groups = np.array([subject_mapping[s['subject']] for s in samples])
        >>> splits = get_stratified_group_cv_splits(labels, groups, n_splits=5)
    """
    sgf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    return list(sgf.split(X=np.arange(len(labels)), y=labels, groups=groups))


def get_loso_splits(
    samples: List[Dict],
    subject_mapping: Dict[str, int]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Leave-One-Subject-Out кросс-валидация.

    Для каждого субъекта:
    - Test: все сэмплы этого субъекта
    - Train: все сэмплы остальных субъектов

    Это максимально строгая оценка cross-subject обобщаемости.

    Description:
    ---------------
        Используется для оценки способности модели обобщаться на совершенно новых субъектах.
        Каждый субъект по очереди становится тестовым набором.

    Args:
    ---------------
        samples: List[Dict] - список всех образцов данных
        subject_mapping: Dict[str, int] - маппинг subject_id -> integer index

    Returns:
    ---------------
        List[(train_idx, val_idx)] - LOSO разбиения

    Example:
    ---------------
        >>> splits = get_loso_splits(samples, subject_mapping)
        >>> print(f"LOSO: {len(splits)} разбиений для {len(subject_mapping)} субъектов")
    """
    unique_subjects = sorted(subject_mapping.keys())
    splits = []

    for test_subject in unique_subjects:
        test_mask = np.array([s['subject'] == test_subject for s in samples])
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        splits.append((train_idx, test_idx))

    return splits


def compute_channelwise_stats(
    samples: List[Dict], indices: np.ndarray, exclude_channels: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Вычисляет среднее и стандартное отклонение для каждого канала."""
    exclude_channels = exclude_channels or []
    if len(indices) == 0: raise ValueError("Массив индексов не может быть пустым.")

    eeg0 = samples[indices[0]]['eeg'].squeeze(0)
    if exclude_channels:
        keep_mask = np.ones(eeg0.shape[0], dtype=bool)
        keep_mask[exclude_channels] = False
        eeg0 = eeg0[keep_mask, :]
    C, _ = eeg0.shape

    sum_c, sumsq_c, count = np.zeros(C, np.float64), np.zeros(C, np.float64), 0
    for idx in indices:
        eeg = samples[idx]['eeg'].squeeze(0)
        if exclude_channels:
            keep_mask = np.ones(eeg.shape[0], dtype=bool)
            keep_mask[exclude_channels] = False
            eeg = eeg[keep_mask, :]
        x = eeg.astype(np.float64)
        sum_c += x.sum(axis=-1)
        sumsq_c += (x ** 2).sum(axis=-1)
        count += x.shape[1]

    mean_c = sum_c / count
    var_c = (sumsq_c / count) - (mean_c ** 2)
    std_c = np.sqrt(np.maximum(var_c, 1e-12))
    return mean_c.astype(np.float32), std_c.astype(np.float32)


def compute_subjectwise_stats(
    samples: List[Dict],
    indices: np.ndarray,
    exclude_channels: Optional[List[int]] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Вычисляет mean/std для каждого subject ОТДЕЛЬНО.

    Description:
    ---------------
        Группирует samples по subject_id и вычисляет channelwise статистики
        для каждого subject независимо. Это устраняет subject baseline shifts
        (различия в electrode impedance, skull thickness, etc.).

    Args:
    ---------------
        samples: Список всех образцов данных.
        indices: Индексы образцов для обучающей выборки.
        exclude_channels: Список индексов каналов для исключения.

    Returns:
    ---------------
        Dict[subject_id -> (mean_c, std_c)]
        где mean_c, std_c — массивы размера (n_channels,)

    Mathematical Justification:
    ---------------
        Текущий подход (WRONG):
            μ_global = E[X | all subjects]
            X̂ = (X - μ_global) / σ_global

        Правильный подход:
            Для каждого subject i:
                X̂ᵢ = (Xᵢ - μᵢ) / σᵢ

            Затем pool: X̂_pooled = [X̂₁, X̂₂, ..., X̂ₙ]

        Это устраняет subject-specific baseline shifts и позволяет модели
        учиться на "чистых" discriminative patterns.
    """
    exclude_channels = exclude_channels or []

    # Group indices by subject
    subject_groups = {}
    for idx in indices:
        subject_id = samples[idx]['subject']
        if subject_id not in subject_groups:
            subject_groups[subject_id] = []
        subject_groups[subject_id].append(idx)

    # Compute stats per subject
    stats = {}
    for subject_id, subject_indices in subject_groups.items():
        subject_indices_np = np.array(subject_indices)
        mean_c, std_c = compute_channelwise_stats(
            samples, subject_indices_np, exclude_channels
        )
        stats[subject_id] = (mean_c, std_c)

    return stats


def compute_hybrid_stats(
    samples: List[Dict],
    indices: np.ndarray,
    exclude_channels: Optional[List[int]] = None
) -> Dict[str, any]:
    """
    HYBRID NORMALIZATION: Subject-wise centering + Global scaling.

    Description:
    ---------------
        Комбинация subject-wise и global normalization для максимизации
        discriminative signal при минимизации subject baseline shifts.

    Mathematical Justification:
    ---------------
        Step 1: Subject-wise CENTERING (устраняет baseline shifts):
            X̂ᵢ_centered = Xᵢ - μᵢ  для каждого subject i

        Step 2: GLOBAL SCALING (сохраняет inter-subject variance):
            X̂_pooled = [X̂₁_centered, X̂₂_centered, ...]
            σ_global = std(X̂_pooled)
            X̂_final = X̂_pooled / σ_global

    Benefits:
    ---------------
        ✅ Устраняет subject baseline shifts (μᵢ removed)
        ✅ Сохраняет inter-subject variance structure (σᵢ relative to σ_global)
        ✅ Effective sample size остаётся полным (N = total samples)
        ✅ Compatible с class-balanced loss на pooled distribution

    Args:
    ---------------
        samples: Список всех образцов данных.
        indices: Индексы образцов для обучающей выборки.
        exclude_channels: Список индексов каналов для исключения.

    Returns:
    ---------------
        Dict with:
            'mean_per_subject': Dict[subject_id -> mean_c]
            'std_global': np.ndarray (n_channels,)
    """
    exclude_channels = exclude_channels or []

    # Group indices by subject
    subject_groups = {}
    for idx in indices:
        subject_id = samples[idx]['subject']
        if subject_id not in subject_groups:
            subject_groups[subject_id] = []
        subject_groups[subject_id].append(idx)

    # Step 1: Compute per-subject means (for centering)
    subject_means = {}
    for subject_id, subject_indices in subject_groups.items():
        subject_indices_np = np.array(subject_indices)
        mean_c, _ = compute_channelwise_stats(
            samples, subject_indices_np, exclude_channels
        )
        subject_means[subject_id] = mean_c

    # Step 2: Compute GLOBAL std on centered data
    # Accumulate centered data statistics
    eeg0 = samples[indices[0]]['eeg'].squeeze(0)
    if exclude_channels:
        keep_mask = np.ones(eeg0.shape[0], dtype=bool)
        keep_mask[exclude_channels] = False
        eeg0 = eeg0[keep_mask, :]
    C, _ = eeg0.shape

    sumsq_c_global = np.zeros(C, np.float64)
    count_global = 0

    for idx in indices:
        subject_id = samples[idx]['subject']
        eeg = samples[idx]['eeg'].squeeze(0)
        if exclude_channels:
            keep_mask = np.ones(eeg.shape[0], dtype=bool)
            keep_mask[exclude_channels] = False
            eeg = eeg[keep_mask, :]

        # Center by subject mean
        x_centered = eeg.astype(np.float64) - subject_means[subject_id].reshape(-1, 1)

        sumsq_c_global += (x_centered ** 2).sum(axis=-1)
        count_global += x_centered.shape[1]

    # Global std (on centered data)
    var_c_global = sumsq_c_global / count_global
    std_c_global = np.sqrt(np.maximum(var_c_global, 1e-12))

    return {
        'mean_per_subject': subject_means,
        'std_global': std_c_global.astype(np.float32)
    }
