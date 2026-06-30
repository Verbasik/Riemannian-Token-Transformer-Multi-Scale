# file: feature_engineering.py
# -*- coding: utf-8 -*-
"""
Feature engineering for EEG signals.

Implements feature extraction for classic ML algorithms:
1. Spectral features (PSD in frequency bands)
2. Statistical features (mean, std, skewness, kurtosis)
3. Wavelet features (energy across scales)
4. Temporal features (Hjorth parameters)

Based on the literature (Hossain et al. 2025, Torres-García et al. 2013).
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
# Spectral features
# =============================================================================

def compute_psd_features(
    eeg: np.ndarray,
    fs: float = 500.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Description:
    ---------------
        Computes power spectral density (PSD) power in specified
        frequency bands for each EEG channel.
        Uses Welch's method for spectrum estimation and Simpson
        integration to compute band power.

    Args:
    ---------------
        eeg: ndarray [n_channels, n_samples] - Input EEG signal.
        fs: float - Sampling frequency in Hz (default: 500.0).
        bands: Dict[str, Tuple[float, float]] - Frequency-band dictionary
            in the form {'band_name': (low_freq, high_freq)}. If None,
            standard EEG bands are used.

    Returns:
    ---------------
        ndarray [n_channels * n_bands]: Power feature array.

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> signal = np.random.randn(2, 1000)
        >>> feats = compute_psd_features(signal, fs=500.0)
        >>> feats.shape[0] == 2 * 5  # 2 channels, 5 standard bands
        True
    """
    if bands is None:
        # Standard EEG bands according to the literature.
        bands = {
            'delta': (0.5, 4.0),    # 0.5-4 Hz: deep sleep
            'theta': (4.0, 8.0),    # 4-8 Hz: relaxation, meditation
            'alpha': (8.0, 13.0),   # 8-13 Hz: relaxed wakefulness
            'beta': (13.0, 30.0),   # 13-30 Hz: active thinking
            'gamma': (30.0, 45.0),  # 30-45 Hz: cognitive processing
        }

    n_channels = eeg.shape[0]
    features = []

    for ch_idx in range(n_channels):
        # Welch's method for PSD estimation.
        # nperseg is capped by the signal length or 256 samples.
        freqs, psd = signal.welch(
            eeg[ch_idx, :],
            fs=fs,
            nperseg=min(256, eeg.shape[1]),
            noverlap=None
        )

        # Extract power in each band.
        for band_name, (low, high) in bands.items():
            # Boolean mask for selecting frequencies in the band.
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            # Integrate PSD within the band (Simpson's method).
            # This gives the total energy in the frequency band.
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
        Computes power ratios between key frequency bands. These ratios
        are often more informative than absolute power for classifying
        brain states.

        Computed ratios:
        - theta/alpha: relaxation vs activity indicator.
        - beta/alpha: cognitive activation indicator.
        - (theta+alpha)/(alpha+beta): fatigue index.

    Args:
    ---------------
        eeg: ndarray [n_channels, n_samples] - Input EEG signal.
        fs: float - Sampling frequency in Hz.

    Returns:
    ---------------
        ndarray [n_channels * 3]: Ratio array.

    Raises:
    ---------------
        No explicit exceptions.

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
    # Small value to protect against division by zero.
    epsilon = 1e-8

    for ch_idx in range(n_channels):
        freqs, psd = signal.welch(
            eeg[ch_idx, :],
            fs=fs,
            nperseg=min(256, eeg.shape[1])
        )

        # Compute power in each band.
        powers = {}
        for band_name, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            powers[band_name] = simpson(psd[idx], x=freqs[idx])

        # Ratios protected against division by zero.
        theta_alpha = powers['theta'] / (powers['alpha'] + epsilon)
        beta_alpha = powers['beta'] / (powers['alpha'] + epsilon)
        fatigue_idx = (
            (powers['theta'] + powers['alpha']) /
            (powers['alpha'] + powers['beta'] + epsilon)
        )

        features.extend([theta_alpha, beta_alpha, fatigue_idx])

    return np.array(features, dtype=np.float32)


# =============================================================================
# Statistical features
# =============================================================================

def compute_statistical_features(eeg: np.ndarray) -> np.ndarray:
    """
    Description:
    ---------------
        Computes a set of time-series statistical features for each EEG
        channel. Describes the shape of the signal amplitude distribution.

        Features:
        - Mean: Average value (signal offset).
        - Std: Standard deviation (spread).
        - Skewness: Distribution asymmetry.
        - Kurtosis: Distribution peakedness.
        - Min, Max: Extremes.
        - Range: Spread (Max - Min).
        - RMS: Root mean square value (signal energy).

    Args:
    ---------------
        eeg: ndarray [n_channels, n_samples] - Input EEG signal.

    Returns:
    ---------------
        ndarray [n_channels * 8]: Statistical feature array.

    Raises:
    ---------------
        No explicit exceptions.

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
        # RMS correlates with signal energy.
        rms_val = np.sqrt(np.mean(signal_ch ** 2))

        features.extend([
            mean_val, std_val, skew_val, kurt_val,
            min_val, max_val, range_val, rms_val
        ])

    return np.array(features, dtype=np.float32)


# =============================================================================
# Wavelet features
# =============================================================================

def compute_wavelet_features(
    eeg: np.ndarray,
    wavelet: str = 'db4',
    level: int = 5
) -> np.ndarray:
    """
    Description:
    ---------------
        Computes wavelet coefficient energy at different scales using
        the discrete wavelet transform (DWT). Enables simultaneous
        time-frequency signal analysis and detection of local features.

        The Daubechies wavelet (db4) is used by default because it is
        well suited for biosignal analysis.

    Args:
    ---------------
        eeg: ndarray [n_channels, n_samples] - Input EEG signal.
        wavelet: str - Mother wavelet type (default: 'db4').
        level: int - Decomposition level (decomposition depth).

    Returns:
    ---------------
        ndarray [n_channels * (level + 1)]: Coefficient energy at each
            decomposition level (details + approximation).

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> signal = np.random.randn(1, 1000)
        >>> w_feats = compute_wavelet_features(signal, level=4)
        >>> len(w_feats) == 5  # 4 detail levels + 1 approximation
        True
    """
    n_channels = eeg.shape[0]
    features = []

    for ch_idx in range(n_channels):
        signal_ch = eeg[ch_idx, :]

        # Discrete Wavelet Transform (DWT)
        # Returns a list: [cA_n, cD_n, cD_n-1, ..., cD_1].
        coeffs = pywt.wavedec(signal_ch, wavelet, level=level)

        # Energy at each level (sum of squared coefficients).
        for coeff in coeffs:
            energy = np.sum(coeff ** 2)
            features.append(energy)

    return np.array(features, dtype=np.float32)


# =============================================================================
# Temporal features (Hjorth parameters)
# =============================================================================

def compute_hjorth_parameters(eeg: np.ndarray) -> np.ndarray:
    """
    Description:
    ---------------
        Computes Hjorth parameters for each channel. These are simple
        time-domain signal complexity metrics computed from derivatives.

        Parameters:
        - Activity: Signal variance (power).
        - Mobility: Mean frequency/bandwidth.
          sqrt(var(derivative) / var(signal)).
        - Complexity: Frequency variation over time. Indicates how much
          the signal differs from a pure sine wave.
          Mobility(derivative) / Mobility(signal).

    Args:
    ---------------
        eeg: ndarray [n_channels, n_samples] - Input EEG signal.

    Returns:
    ---------------
        ndarray [n_channels * 3]: Hjorth parameter array.

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> signal = np.random.randn(1, 1000)
        >>> hjorth = compute_hjorth_parameters(signal)
        >>> len(hjorth)
        3
    """
    n_channels = eeg.shape[0]
    features = []
    epsilon = 1e-8  # Protection against division by zero

    for ch_idx in range(n_channels):
        signal_ch = eeg[ch_idx, :]

        # Activity: Variance of the signal
        activity = np.var(signal_ch)

        # First derivative (rate of change).
        deriv1 = np.diff(signal_ch)

        # Mobility: sqrt(variance(deriv1) / variance(signal))
        # Characterizes the average spectral frequency.
        mobility = np.sqrt(np.var(deriv1) / (activity + epsilon))

        # Second derivative (acceleration of change).
        deriv2 = np.diff(deriv1)

        # First-derivative mobility.
        mobility_deriv = np.sqrt(
            np.var(deriv2) / (np.var(deriv1) + epsilon)
        )

        # Complexity: derivative mobility divided by signal mobility.
        # A value near 1 means a simple signal (sine wave), >1 means complex.
        complexity = mobility_deriv / (mobility + epsilon)

        features.extend([activity, mobility, complexity])

    return np.array(features, dtype=np.float32)


# =============================================================================
# Main feature extraction function
# =============================================================================

def extract_all_features(
    eeg: np.ndarray,
    fs: float = 500.0,
    include: Optional[List[str]] = None
) -> np.ndarray:
    """
    Description:
    ---------------
        Extracts all available features from an EEG signal by combining
        the results of different feature engineering methods into one
        vector. The `include` parameter allows flexible feature selection.

    Args:
    ---------------
        eeg: ndarray [n_channels, n_samples] - Input EEG signal.
        fs: float - Sampling frequency in Hz.
        include: List[str] - List of feature types to include.
            Accepted values: 'psd', 'ratios', 'stats', 'wavelet',
            'hjorth'. If None, all are included.

    Returns:
    ---------------
        ndarray [total_features]: Combined feature vector.

    Raises:
    ---------------
        No explicit exceptions.

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

    # Combine all features into one flat vector.
    return np.concatenate(all_features)


def batch_extract_features(
    eeg_batch: np.ndarray,
    fs: float = 500.0,
    include: Optional[List[str]] = None
) -> np.ndarray:
    """
    Description:
    ---------------
        Extracts features for a batch of EEG signals. Iterates over the
        first dimension (batch_size) and applies extract_all_features to
        each sample.

    Args:
    ---------------
        eeg_batch: ndarray [batch_size, n_channels, n_samples] -
            Batch of EEG signals.
        fs: float - Sampling frequency in Hz.
        include: List[str] - List of feature types (see extract_all).

    Returns:
    ---------------
        ndarray [batch_size, total_features]: Feature matrix.

    Raises:
    ---------------
        No explicit exceptions.

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
