# file: train_classical_ml.py
# -*- coding: utf-8 -*-
"""
Обучение классических ML алгоритмов на извлеченных признаках.

Реализует подход Feature Engineering + Random Forest/XGBoost
на основе литературы (Hossain et al. 2025).

Использует те же данные, что и Phase 4B-2 (2 subjects).
"""
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost недоступен. Используем только Random Forest.")

from config import default_config
from data_loader import (load_all_data_metaclass, get_stratified_cv_splits,
                         compute_channelwise_stats)
from feature_engineering import batch_extract_features
from utils import set_seed, print_metrics


# =============================================================================
# Функции для извлечения признаков из датасета
# =============================================================================

def load_and_extract_features(
    cfg: Dict[str, Any],
    feature_types: list = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает данные и извлекает признаки для обучения классических ML.

    Args:
        cfg: Конфигурационный словарь
        feature_types: Типы признаков для извлечения

    Returns:
        X_train, X_val, y_train, y_val: Наборы данных и меток
    """
    if feature_types is None:
        feature_types = ['psd', 'ratios', 'stats', 'wavelet', 'hjorth']

    print(f"\n{'='*80}")
    print("ЗАГРУЗКА ДАННЫХ И ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ")
    print(f"{'='*80}")

    # 1. Загружаем данные (те же, что для Phase 4B-2)
    samples = load_all_data_metaclass(
        data_dir=cfg['data']['data_dir'],
        subject_ids=cfg['data']['subject_ids'],
        task=cfg['data']['task'],
    )
    print(f"Загружено образцов: {len(samples)}")
    print(f"Испытуемые: {cfg['data']['subject_ids']}")

    # 2. Получаем train/val splits (те же, что в Phase 4B-2)
    labels = np.array([s['label'] for s in samples])
    splits = get_stratified_cv_splits(
        labels, cfg['cv']['n_splits'], cfg['cv']['random_state']
    )
    train_idx, val_idx = splits[0]  # Используем первый фолд

    print(f"\nРазбиение данных:")
    print(f"  Train: {len(train_idx)} образцов")
    print(f"  Val:   {len(val_idx)} образцов")

    # 3. Применяем нормализацию (используем ГЛОБАЛЬНУЮ из Phase 4B-2)
    mean_c, std_c = compute_channelwise_stats(
        samples, train_idx, cfg['data'].get('exclude_channels')
    )
    exclude_channels = cfg['data'].get('exclude_channels', [])

    def normalize_eeg(eeg_raw: np.ndarray) -> np.ndarray:
        """Применяет z-score нормализацию по каналам."""
        # Исключаем каналы (если указано)
        if exclude_channels:
            keep_mask = np.ones(eeg_raw.shape[0], dtype=bool)
            keep_mask[exclude_channels] = False
            eeg = eeg_raw[keep_mask, :]
        else:
            eeg = eeg_raw

        # Нормализуем
        mean = mean_c.reshape(-1, 1)
        std = (std_c.reshape(-1, 1) + 1e-4)
        return (eeg - mean) / std

    # 4. Извлекаем признаки для train и val
    print(f"\nИзвлечение признаков ({', '.join(feature_types)})...")

    X_train_list = []
    y_train_list = []
    for idx in train_idx:
        eeg_raw = samples[idx]['eeg']  # Shape: (1, n_channels, n_samples)
        eeg_raw = eeg_raw.squeeze(0)   # Remove batch dim: (n_channels, n_samples)
        eeg_norm = normalize_eeg(eeg_raw)
        features = batch_extract_features(
            eeg_norm[np.newaxis, :, :],
            fs=500.0,
            include=feature_types
        )
        X_train_list.append(features[0])
        y_train_list.append(samples[idx]['label'])

    X_val_list = []
    y_val_list = []
    for idx in val_idx:
        eeg_raw = samples[idx]['eeg']  # Shape: (1, n_channels, n_samples)
        eeg_raw = eeg_raw.squeeze(0)   # Remove batch dim: (n_channels, n_samples)
        eeg_norm = normalize_eeg(eeg_raw)
        features = batch_extract_features(
            eeg_norm[np.newaxis, :, :],
            fs=500.0,
            include=feature_types
        )
        X_val_list.append(features[0])
        y_val_list.append(samples[idx]['label'])

    X_train = np.vstack(X_train_list)
    X_val = np.vstack(X_val_list)
    y_train = np.array(y_train_list)
    y_val = np.array(y_val_list)

    print(f"  Размерность признаков: {X_train.shape[1]}")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")

    # 5. Стандартизация признаков (важно для классических ML)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    print("✅ Извлечение признаков завершено")

    return X_train, X_val, y_train, y_val, scaler


# =============================================================================
# Функции обучения моделей
# =============================================================================

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 500,
    max_depth: int = 20,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Обучает Random Forest классификатор.

    Args:
        X_train: Признаки обучающей выборки
        y_train: Метки обучающей выборки
        n_estimators: Количество деревьев
        max_depth: Максимальная глубина дерева
        random_state: Random seed

    Returns:
        Обученная модель Random Forest
    """
    print(f"\n{'='*80}")
    print(f"ОБУЧЕНИЕ RANDOM FOREST")
    print(f"{'='*80}")
    print(f"Параметры:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  random_state: {random_state}")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Для учета дисбаланса классов
        random_state=random_state,
        n_jobs=-1,  # Используем все ядра
        verbose=1
    )

    model.fit(X_train, y_train)
    print("✅ Обучение Random Forest завершено")

    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42
) -> Any:
    """
    Обучает XGBoost классификатор.

    Args:
        X_train: Признаки обучающей выборки
        y_train: Метки обучающей выборки
        n_estimators: Количество деревьев
        max_depth: Максимальная глубина дерева
        learning_rate: Learning rate
        random_state: Random seed

    Returns:
        Обученная модель XGBoost
    """
    if not XGBOOST_AVAILABLE:
        print("⚠️ XGBoost недоступен.")
        return None

    print(f"\n{'='*80}")
    print(f"ОБУЧЕНИЕ XGBOOST")
    print(f"{'='*80}")
    print(f"Параметры:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  learning_rate: {learning_rate}")

    # Вычисляем веса классов для балансировки
    class_counts = np.bincount(y_train)
    total = len(y_train)
    weights = total / (len(class_counts) * class_counts)
    sample_weights = weights[y_train]

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective='multi:softmax',
        num_class=len(class_counts),
        random_state=random_state,
        n_jobs=-1,
        verbosity=1
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)
    print("✅ Обучение XGBoost завершено")

    return model


# =============================================================================
# Функции оценки
# =============================================================================

def evaluate_model(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Оценивает модель на валидационной выборке.

    Args:
        model: Обученная модель
        X_val: Признаки валидационной выборки
        y_val: Метки валидационной выборки
        model_name: Название модели для вывода

    Returns:
        Словарь с метриками
    """
    print(f"\n{'='*80}")
    print(f"ОЦЕНКА МОДЕЛИ: {model_name}")
    print(f"{'='*80}")

    # Предсказания
    y_pred = model.predict(X_val)

    # Метрики
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'f1_macro': f1_score(y_val, y_pred, average='macro', zero_division=0),
        'precision_macro': precision_score(y_val, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_val, y_pred, average='macro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_val, y_pred)
    }

    print_metrics(metrics)

    # Детальный отчет
    print("\nКлассификационный отчет:")
    print(classification_report(y_val, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print("\nMatрица ошибок:")
    print(cm)

    return metrics


# =============================================================================
# Главная функция
# =============================================================================

def main() -> None:
    """Основная функция для запуска классических ML алгоритмов."""
    # 1. Конфигурация
    cfg = default_config()
    set_seed(cfg['seed'])

    # Изменяем конфигурацию на Phase 4B-2 (2 subjects, global norm)
    cfg['data']['subject_ids'] = ['sub-01', 'sub-02']
    cfg['data']['normalize'] = 'zscore_dataset_channel'  # Используем ГЛОБАЛЬНУЮ нормализацию

    print(f"\n{'='*80}")
    print("CLASSICAL ML: FEATURE ENGINEERING APPROACH")
    print(f"{'='*80}")
    print(f"Основание: Hossain et al. 2025 (Random Forest > Deep Learning)")
    print(f"Baseline для сравнения: Phase 4B-2 (27.72% accuracy, 25.69% F1)")

    # 2. Загрузка данных и извлечение признаков
    X_train, X_val, y_train, y_val, scaler = load_and_extract_features(cfg)

    # 3. Обучение Random Forest
    rf_model = train_random_forest(
        X_train, y_train,
        n_estimators=500,
        max_depth=20,
        random_state=cfg['seed']
    )
    rf_metrics = evaluate_model(rf_model, X_val, y_val, model_name="Random Forest")

    # 4. Обучение XGBoost (если доступен)
    xgb_metrics = None
    if XGBOOST_AVAILABLE:
        xgb_model = train_xgboost(
            X_train, y_train,
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            random_state=cfg['seed']
        )
        xgb_metrics = evaluate_model(xgb_model, X_val, y_val, model_name="XGBoost")

    # 5. Сравнение с baseline
    print(f"\n{'='*80}")
    print("СРАВНЕНИЕ С BASELINE (Phase 4B-2)")
    print(f"{'='*80}")
    baseline_acc = 27.72
    baseline_f1 = 25.69

    print(f"\nBaseline (RTT-MultiScale Deep Learning):")
    print(f"  Accuracy: {baseline_acc:.2f}%")
    print(f"  F1-macro: {baseline_f1:.2f}%")

    print(f"\nRandom Forest (Feature Engineering):")
    print(f"  Accuracy: {rf_metrics['accuracy']*100:.2f}% ({rf_metrics['accuracy']*100 - baseline_acc:+.2f}%)")
    print(f"  F1-macro: {rf_metrics['f1_macro']*100:.2f}% ({rf_metrics['f1_macro']*100 - baseline_f1:+.2f}%)")

    if xgb_metrics:
        print(f"\nXGBoost (Feature Engineering):")
        print(f"  Accuracy: {xgb_metrics['accuracy']*100:.2f}% ({xgb_metrics['accuracy']*100 - baseline_acc:+.2f}%)")
        print(f"  F1-macro: {xgb_metrics['f1_macro']*100:.2f}% ({xgb_metrics['f1_macro']*100 - baseline_f1:+.2f}%)")

    # 6. Сохранение результатов
    results_dir = Path('Train/results/classical_ml_phase4b')
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'random_forest': rf_metrics,
        'xgboost': xgb_metrics,
        'baseline_phase4b2': {
            'accuracy': baseline_acc / 100,
            'f1_macro': baseline_f1 / 100
        }
    }

    with open(results_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Результаты сохранены в {results_dir}")
    print(f"\n{'='*80}")
    print("ОБУЧЕНИЕ CLASSICAL ML ЗАВЕРШЕНО")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
