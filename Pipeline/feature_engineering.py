# file: feature_engineering.py
# -*- coding: utf-8 -*-
"""
Feature Engineering для EEG-сигналов.

Реализует извлечение признаков для классических ML алгоритмов:
1. Спектральные признаки (PSD в частотных диапазонах)
2. Статистические признаки (mean, std, skewness, kurtosis)
3. Вейвлет-признаки (энергия в масштабах)
4. Временные признаки (Hjorth parameters)

Основано на литературе (Hossain et al. 2025, Torres-García et al. 2013).
"""

# =============================================================================
# Standard Libraries
# =============================================================================
from typing import Dict, List, Optional, Tuple

# =============================================================================
# Third-Party Libraries
# =============================================================================
import numpy as np
import pywt
from scipy import signal, stats
from scipy.integrate import simpson


# =============================================================================
# Спектральные признаки (Spectral Features)
# =============================================================================

def compute_psd_features(
    eeg: np.ndarray,
    fs: float = 500.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Description:
    ---------------
        Вычисляет мощность спектральной плотности (PSD) в заданных
        частотных диапазонах для каждого канала EEG.
        Использует метод Уэлча для оценки спектра и интегрирование
        Симпсона для расчета мощности полосы.

    Args:
    ---------------
        eeg: ndarray [n_channels, n_samples] - Входной сигнал EEG.
        fs: float - Частота дискретизации в Гц (по умолчанию 500.0).
        bands: Dict[str, Tuple[float, float]] - Словарь частотных
            диапазонов вида {'band_name': (low_freq, high_freq)}.
            Если None, используются стандартные диапазоны EEG.

    Returns:
    ---------------
        ndarray [n_channels * n_bands]: Массив признаков мощности.

    Raises:
    ---------------
        Нет явных исключений.

    Examples:
    ---------------
        >>> signal = np.random.randn(2, 1000)
        >>> feats = compute_psd_features(signal, fs=500.0)
        >>> feats.shape[0] == 2 * 5  # 2 канала, 5 стандартных полос
        True
    """
    if bands is None:
        # Стандартные EEG диапазоны согласно литературе
        bands = {
            'delta': (0.5, 4.0),    # 0.5-4 Hz: глубокий сон
            'theta': (4.0, 8.0),    # 4-8 Hz: релаксация, медитация
            'alpha': (8.0, 13.0),   # 8-13 Hz: спокойное бодрствование
            'beta': (13.0, 30.0),   # 13-30 Hz: активное мышление
            'gamma': (30.0, 45.0),  # 30-45 Hz: когнитивная обработка
        }

    n_channels = eeg.shape[0]
    features = []

    for ch_idx in range(n_channels):
        # Welch's method для оценки PSD
        # nperseg ограничен длиной сигнала или 256 отсчетами
        freqs, psd = signal.welch(
            eeg[ch_idx, :],
            fs=fs,
            nperseg=min(256, eeg.shape[1]),
            noverlap=None
        )

        # Извлекаем мощность в каждом диапазоне
        for band_name, (low, high) in bands.items():
            # Логическая маска для выбора частот в диапазоне
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            # Интегрируем PSD в диапазоне (метод Симпсона)
            # Это дает полную энергию в полосе частот
            band_power = simpson(psd[idx_band], x=freqs[idx_band])
            features.append(band_power)

    return np.array(features, dtype=np.float32)


def compute_band_ratios(
    eeg: np.ndarray,
    fs: float = 500.0
) -> np.ndarray:
    """
    Description:
    ---------------
        Вычисляет соотношения мощностей между ключевыми частотными
        диапазонами. Эти отношения часто более информативны для
        классификации состояний мозга, чем абсолютная мощность.

        Рассчитываемые ratios:
        - theta/alpha: индикатор релаксации vs активности.
        - beta/alpha: индикатор когнитивной активации.
        - (theta+alpha)/(alpha+beta): индекс утомления.

    Args:
    ---------------
        eeg: ndarray [n_channels, n_samples] - Входной сигнал EEG.
        fs: float - Частота дискретизации в Гц.

    Returns:
    ---------------
        ndarray [n_channels * 3]: Массив соотношений.

    Raises:
    ---------------
        Нет явных исключений.

    Examples:
    ---------------
        >>> signal = np.random.randn(1, 1000)
        >>> ratios = compute_band_ratios(signal)
        >>> len(ratios)
        3
    """
    bands = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 45.0),
    }

    n_channels = eeg.shape[0]
    features = []
    # Малое число для защиты от деления на ноль
    epsilon = 1e-8

    for ch_idx in range(n_channels):
        freqs, psd = signal.welch(
            eeg[ch_idx, :],
            fs=fs,
            nperseg=min(256, eeg.shape[1])
        )

        # Вычисляем мощность в каждом диапазоне
        powers = {}
        for band_name, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            powers[band_name] = simpson(psd[idx], x=freqs[idx])

        # Соотношения с защитой от деления на ноль
        theta_alpha = powers['theta'] / (powers['alpha'] + epsilon)
        beta_alpha = powers['beta'] / (powers['alpha'] + epsilon)
        fatigue_idx = (
            (powers['theta'] + powers['alpha']) /
            (powers['alpha'] + powers['beta'] + epsilon)
        )

        features.extend([theta_alpha, beta_alpha, fatigue_idx])

    return np.array(features, dtype=np.float32)


# =============================================================================
# Статистические признаки (Statistical Features)
# =============================================================================

def compute_statistical_features(eeg: np.ndarray) -> np.ndarray:
    """
    Description:
    ---------------
        Вычисляет набор статистических признаков временного ряда
        для каждого канала EEG. Описывает форму распределения
        амплитуд сигнала.

        Признаки:
        - Mean: Среднее значение (смещение сигнала).
        - Std: Стандартное отклонение (разброс).
        - Skewness: Асимметрия распределения.
        - Kurtosis: Эксцесс (островершинность).
        - Min, Max: Экстремумы.
        - Range: Размах (Max - Min).
        - RMS: Среднеквадратичное значение (энергия сигнала).

    Args:
    ---------------
        eeg: ndarray [n_channels, n_samples] - Входной сигнал EEG.

    Returns:
    ---------------
        ndarray [n_channels * 8]: Массив статистических признаков.

    Raises:
    ---------------
        Нет явных исключений.

    Examples:
    ---------------
        >>> signal = np.random.randn(2, 1000)
        >>> stats = compute_statistical_features(signal)
        >>> stats.shape[0] == 2 * 8
        True
    """
    n_channels = eeg.shape[0]
    features = []

    for ch_idx in range(n_channels):
        signal_ch = eeg[ch_idx, :]

        mean_val = np.mean(signal_ch)
        std_val = np.std(signal_ch)
        skew_val = stats.skew(signal_ch)
        kurt_val = stats.kurtosis(signal_ch)
        min_val = np.min(signal_ch)
        max_val = np.max(signal_ch)
        range_val = max_val - min_val
        # RMS коррелирует с энергией сигнала
        rms_val = np.sqrt(np.mean(signal_ch ** 2))

        features.extend([
            mean_val, std_val, skew_val, kurt_val,
            min_val, max_val, range_val, rms_val
        ])

    return np.array(features, dtype=np.float32)


# =============================================================================
# Вейвлет-признаки (Wavelet Features)
# =============================================================================

def compute_wavelet_features(
    eeg: np.ndarray,
    wavelet: str = 'db4',
    level: int = 5
) -> np.ndarray:
    """
    Description:
    ---------------
        Вычисляет энергию вейвлет-коэффициентов на разных масштабах
        с помощью дискретного вейвлет-преобразования (DWT).
        Позволяет анализировать сигнал одновременно во времени и
        частоте, выявляя локальные особенности.

        Используется вейвлет Добеши (db4) по умолчанию, так как он
        хорошо подходит для анализа биосигналов.

    Args:
    ---------------
        eeg: ndarray [n_channels, n_samples] - Входной сигнал EEG.
        wavelet: str - Тип материнского вейвлета (по умолчанию 'db4').
        level: int - Уровень декомпозиции (глубина разложения).

    Returns:
    ---------------
        ndarray [n_channels * (level + 1)]: Энергия коэффициентов
            на каждом уровне декомпозиции (детали + аппроксимация).

    Raises:
    ---------------
        Нет явных исключений.

    Examples:
    ---------------
        >>> signal = np.random.randn(1, 1000)
        >>> w_feats = compute_wavelet_features(signal, level=4)
        >>> len(w_feats) == 5  # 4 уровня деталей + 1 аппроксимация
        True
    """
    n_channels = eeg.shape[0]
    features = []

    for ch_idx in range(n_channels):
        signal_ch = eeg[ch_idx, :]

        # Discrete Wavelet Transform (DWT)
        # Возвращает список: [cA_n, cD_n, cD_n-1, ..., cD_1]
        coeffs = pywt.wavedec(signal_ch, wavelet, level=level)

        # Энергия на каждом уровне (сумма квадратов коэффициентов)
        for coeff in coeffs:
            energy = np.sum(coeff ** 2)
            features.append(energy)

    return np.array(features, dtype=np.float32)


# =============================================================================
# Временные признаки (Temporal Features - Hjorth Parameters)
# =============================================================================

def compute_hjorth_parameters(eeg: np.ndarray) -> np.ndarray:
    """
    Description:
    ---------------
        Вычисляет параметры Хьорта (Hjorth Parameters) для каждого
        канала. Это простые метрики сложности сигнала во временной
        области, вычисляемые через производные.

        Параметры:
        - Activity (Активность): Дисперсия сигнала (мощность).
        - Mobility (Подвижность): Средняя частота/ширина полосы.
          sqrt(var(derivative) / var(signal)).
        - Complexity (Сложность): Изменение частоты во времени.
          Показывает, насколько сигнал отличается от чистого синуса.
          Mobility(derivative) / Mobility(signal).

    Args:
    ---------------
        eeg: ndarray [n_channels, n_samples] - Входной сигнал EEG.

    Returns:
    ---------------
        ndarray [n_channels * 3]: Массив параметров Хьорта.

    Raises:
    ---------------
        Нет явных исключений.

    Examples:
    ---------------
        >>> signal = np.random.randn(1, 1000)
        >>> hjorth = compute_hjorth_parameters(signal)
        >>> len(hjorth)
        3
    """
    n_channels = eeg.shape[0]
    features = []
    epsilon = 1e-8  # Защита от деления на ноль

    for ch_idx in range(n_channels):
        signal_ch = eeg[ch_idx, :]

        # Activity: Variance of the signal
        activity = np.var(signal_ch)

        # Первая производная (скорость изменения)
        deriv1 = np.diff(signal_ch)

        # Mobility: sqrt(variance(deriv1) / variance(signal))
        # Характеризует среднюю частоту спектра
        mobility = np.sqrt(np.var(deriv1) / (activity + epsilon))

        # Вторая производная (ускорение изменения)
        deriv2 = np.diff(deriv1)

        # Mobility первой производной
        mobility_deriv = np.sqrt(
            np.var(deriv2) / (np.var(deriv1) + epsilon)
        )

        # Complexity: отношение подвижности производной к подвижности сигнала
        # Значение ~1 означает простой сигнал (синусоида), >1 - сложный
        complexity = mobility_deriv / (mobility + epsilon)

        features.extend([activity, mobility, complexity])

    return np.array(features, dtype=np.float32)


# =============================================================================
# Главная функция извлечения признаков
# =============================================================================

def extract_all_features(
    eeg: np.ndarray,
    fs: float = 500.0,
    include: Optional[List[str]] = None
) -> np.ndarray:
    """
    Description:
    ---------------
        Извлекает все доступные признаки из EEG сигнала, объединяя
        результаты различных методов.feature engineering в один вектор.
        Позволяет гибко выбирать типы признаков через параметр include.

    Args:
    ---------------
        eeg: ndarray [n_channels, n_samples] - Входной сигнал EEG.
        fs: float - Частота дискретизации в Гц.
        include: List[str] - Список типов признаков для включения.
            Допустимые значения: 'psd', 'ratios', 'stats', 'wavelet',
            'hjorth'. Если None, включаются все.

    Returns:
    ---------------
        ndarray [total_features]: Объединенный вектор признаков.

    Raises:
    ---------------
        Нет явных исключений.

    Examples:
    ---------------
        >>> signal = np.random.randn(2, 1000)
        >>> feats = extract_all_features(signal, include=['stats', 'hjorth'])
        >>> feats.ndim
        1
    """
    if include is None:
        include = ['psd', 'ratios', 'stats', 'wavelet', 'hjorth']

    all_features = []

    if 'psd' in include:
        psd_feats = compute_psd_features(eeg, fs=fs)
        all_features.append(psd_feats)

    if 'ratios' in include:
        ratio_feats = compute_band_ratios(eeg, fs=fs)
        all_features.append(ratio_feats)

    if 'stats' in include:
        stat_feats = compute_statistical_features(eeg)
        all_features.append(stat_feats)

    if 'wavelet' in include:
        wavelet_feats = compute_wavelet_features(eeg)
        all_features.append(wavelet_feats)

    if 'hjorth' in include:
        hjorth_feats = compute_hjorth_parameters(eeg)
        all_features.append(hjorth_feats)

    # Объединяем все признаки в один плоский вектор
    return np.concatenate(all_features)


def batch_extract_features(
    eeg_batch: np.ndarray,
    fs: float = 500.0,
    include: Optional[List[str]] = None
) -> np.ndarray:
    """
    Description:
    ---------------
        Извлекает признаки для батча EEG сигналов.
        Итерируется по первому измерению (batch_size) и применяет
        extract_all_features к каждому сэмплу.

    Args:
    ---------------
        eeg_batch: ndarray [batch_size, n_channels, n_samples] -
            Батч сигналов EEG.
        fs: float - Частота дискретизации в Гц.
        include: List[str] - Список типов признаков (см. extract_all).

    Returns:
    ---------------
        ndarray [batch_size, total_features]: Матрица признаков.

    Raises:
    ---------------
        Нет явных исключений.

    Examples:
    ---------------
        >>> batch = np.random.randn(10, 2, 1000)
        >>> feat_matrix = batch_extract_features(batch)
        >>> feat_matrix.shape[0] == 10
        True
    """
    batch_size = eeg_batch.shape[0]
    feature_vectors = []

    for i in range(batch_size):
        feats = extract_all_features(
            eeg_batch[i],
            fs=fs,
            include=include
        )
        feature_vectors.append(feats)

    # Stack vertically to create 2D matrix [samples, features]
    return np.vstack(feature_vectors)