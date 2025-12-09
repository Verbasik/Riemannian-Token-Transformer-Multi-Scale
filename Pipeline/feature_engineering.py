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
from typing import Dict, List, Tuple

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
    bands: Dict[str, Tuple[float, float]] = None
) -> np.ndarray:
    """
    Вычисляет мощность спектральной плотности (PSD) в частотных диапазонах.

    Args:
        eeg: EEG сигнал [n_channels, n_samples]
        fs: Частота дискретизации (Hz)
        bands: Словарь частотных диапазонов {'band_name': (low_freq, high_freq)}

    Returns:
        features: Массив признаков [n_channels * n_bands]
    """
    if bands is None:
        # Стандартные EEG диапазоны
        bands = {
            'delta': (0.5, 4.0),    # 0.5-4 Hz
            'theta': (4.0, 8.0),    # 4-8 Hz
            'alpha': (8.0, 13.0),   # 8-13 Hz
            'beta': (13.0, 30.0),   # 13-30 Hz
            'gamma': (30.0, 45.0),  # 30-45 Hz
        }

    n_channels = eeg.shape[0]
    features = []

    for ch_idx in range(n_channels):
        # Welch's method для оценки PSD
        freqs, psd = signal.welch(
            eeg[ch_idx, :],
            fs=fs,
            nperseg=min(256, eeg.shape[1]),
            noverlap=None
        )

        # Извлекаем мощность в каждом диапазоне
        for band_name, (low, high) in bands.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            # Интегрируем PSD в диапазоне (используя метод Симпсона)
            band_power = simpson(psd[idx_band], x=freqs[idx_band])
            features.append(band_power)

    return np.array(features, dtype=np.float32)


def compute_band_ratios(
    eeg: np.ndarray,
    fs: float = 500.0
) -> np.ndarray:
    """
    Вычисляет соотношения мощностей между частотными диапазонами.

    Ratios типа:
    - theta/alpha (релаксация vs активность)
    - beta/alpha (активация)
    - (theta+alpha)/(alpha+beta) (индекс утомления)

    Args:
        eeg: EEG сигнал [n_channels, n_samples]
        fs: Частота дискретизации

    Returns:
        features: Массив соотношений [n_channels * n_ratios]
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

        # Соотношения (с защитой от деления на ноль)
        epsilon = 1e-8
        theta_alpha = powers['theta'] / (powers['alpha'] + epsilon)
        beta_alpha = powers['beta'] / (powers['alpha'] + epsilon)
        fatigue_idx = (powers['theta'] + powers['alpha']) / (powers['alpha'] + powers['beta'] + epsilon)

        features.extend([theta_alpha, beta_alpha, fatigue_idx])

    return np.array(features, dtype=np.float32)


# =============================================================================
# Статистические признаки (Statistical Features)
# =============================================================================

def compute_statistical_features(eeg: np.ndarray) -> np.ndarray:
    """
    Вычисляет статистические признаки для каждого канала.

    Признаки:
    - Mean (среднее)
    - Std (стандартное отклонение)
    - Skewness (асимметрия)
    - Kurtosis (эксцесс)
    - Min, Max
    - Range (max - min)
    - RMS (Root Mean Square)

    Args:
        eeg: EEG сигнал [n_channels, n_samples]

    Returns:
        features: Массив статистических признаков [n_channels * 8]
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
    Вычисляет энергию вейвлет-коэффициентов на разных масштабах.

    Args:
        eeg: EEG сигнал [n_channels, n_samples]
        wavelet: Тип вейвлета (по умолчанию Daubechies 4)
        level: Уровень декомпозиции

    Returns:
        features: Энергия вейвлет-коэффициентов [n_channels * (level + 1)]
    """
    n_channels = eeg.shape[0]
    features = []

    for ch_idx in range(n_channels):
        signal_ch = eeg[ch_idx, :]

        # Discrete Wavelet Transform (DWT)
        coeffs = pywt.wavedec(signal_ch, wavelet, level=level)

        # Энергия на каждом уровне
        for coeff in coeffs:
            energy = np.sum(coeff ** 2)
            features.append(energy)

    return np.array(features, dtype=np.float32)


# =============================================================================
# Временные признаки (Temporal Features - Hjorth Parameters)
# =============================================================================

def compute_hjorth_parameters(eeg: np.ndarray) -> np.ndarray:
    """
    Вычисляет параметры Хьорта (Hjorth Parameters) для каждого канала.

    Параметры:
    - Activity (активность): variance сигнала
    - Mobility (подвижность): sqrt(variance(derivative) / variance(signal))
    - Complexity (сложность): Mobility(derivative) / Mobility(signal)

    Args:
        eeg: EEG сигнал [n_channels, n_samples]

    Returns:
        features: Параметры Хьорта [n_channels * 3]
    """
    n_channels = eeg.shape[0]
    features = []

    for ch_idx in range(n_channels):
        signal_ch = eeg[ch_idx, :]

        # Activity
        activity = np.var(signal_ch)

        # Первая производная
        deriv1 = np.diff(signal_ch)

        # Mobility
        mobility = np.sqrt(np.var(deriv1) / (activity + 1e-8))

        # Вторая производная (для Complexity)
        deriv2 = np.diff(deriv1)
        mobility_deriv = np.sqrt(np.var(deriv2) / (np.var(deriv1) + 1e-8))

        # Complexity
        complexity = mobility_deriv / (mobility + 1e-8)

        features.extend([activity, mobility, complexity])

    return np.array(features, dtype=np.float32)


# =============================================================================
# Главная функция извлечения признаков
# =============================================================================

def extract_all_features(
    eeg: np.ndarray,
    fs: float = 500.0,
    include: List[str] = None
) -> np.ndarray:
    """
    Извлекает все доступные признаки из EEG сигнала.

    Args:
        eeg: EEG сигнал [n_channels, n_samples]
        fs: Частота дискретизации
        include: Список типов признаков для включения.
                 По умолчанию: ['psd', 'ratios', 'stats', 'wavelet', 'hjorth']

    Returns:
        features: Объединенный вектор признаков [total_features]
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

    # Объединяем все признаки в один вектор
    return np.concatenate(all_features)


def batch_extract_features(
    eeg_batch: np.ndarray,
    fs: float = 500.0,
    include: List[str] = None
) -> np.ndarray:
    """
    Извлекает признаки для батча EEG сигналов.

    Args:
        eeg_batch: Батч EEG сигналов [batch_size, n_channels, n_samples]
        fs: Частота дискретизации
        include: Список типов признаков

    Returns:
        features: Матрица признаков [batch_size, total_features]
    """
    batch_size = eeg_batch.shape[0]
    feature_vectors = []

    for i in range(batch_size):
        feats = extract_all_features(eeg_batch[i], fs=fs, include=include)
        feature_vectors.append(feats)

    return np.vstack(feature_vectors)
