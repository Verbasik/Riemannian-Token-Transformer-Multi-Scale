# Technical Context — EEG_to_Text

## Технологический стек

### Ядро
| Компонент | Технология | Версия | Назначение |
|-----------|-----------|--------|------------|
| Язык | Python | 3.12+ | Основной язык разработки |
| Фреймворк | PyTorch | 2.2+ | Глубокое обучение, автодифференцирование |
| Numerical | NumPy | 1.26+ | Векторные операции, линейная алгебра |
| Scientific | SciPy | 1.11+ | Научные вычисления |
| ML | scikit-learn | 1.3+ | Кросс-валидация, метрики, preprocessing |

### EEG Preprocessing
| Библиотека | Версия | Назначение |
|-----------|--------|------------|
| MNE | 1.6+ | Обработка ЭЭГ, чтение форматов |
| PyPREP | 0.4+ | Автоматический preprocessing pipeline |
| mne-icalabel | 0.5+ | ICA компонент classification |
| h5py | 3.9+ | HDF5 файлы |
| mat73 | 0.62+ | MATLAB v7.3+ файлы |

### Анализ и визуализация
| Библиотека | Версия | Назначение |
|-----------|--------|------------|
| Pandas | 2.0+ | Манипуляция табличными данными |
| Matplotlib | 3.7+ | Визуализация результатов |
| transformers | 4.40+ | Hugging Face модели (опционально) |

### Утилиты
| Библиотека | Назначение |
|-----------|------------|
| openpyxl | Excel файлы |
| tqdm | Progress bars |

## Аппаратные требования

### Минимальные
- **CPU**: 4+ cores
- **RAM**: 16 GB
- **Storage**: 50 GB (для данных)

### Рекомендуемые
- **GPU**: NVIDIA CUDA (8+ GB VRAM)
- **CPU**: 8+ cores
- **RAM**: 32 GB

## Структура данных

### Входные данные (pkl)
```python
{
    'text': str,           # Текст стимула
    'input_features': np.ndarray,  # ЭЭГ [125, T]
    'subject': str,        # ID субъекта (e.g., 'sub-03')
    'run': str,            # ID запуска
    'label': int           # Original class (0-38)
}
```

### Маппинги классов
```
39 исходных классов → 8 мета-классов:
  0: SOCIAL_INTERACTION
  1: DAILY_LIFE
  2: HEALTH_WELLNESS
  3: FOOD_DINING
  4: TRAVEL_TRANSPORT
  5: WORK_EDUCATION
  6: COMMERCE_SERVICES
  7: ENTERTAINMENT_LEISURE
```

### Формат файлов
```
derivatives/preprocessed_pkl/<subject>/eeg/*task-imagine*_run-<run_id>.pkl
```

## Конфигурация (Pipeline/config.py)

### Параметры данных
```python
'data': {
    'data_dir': Path,              # Корень данных
    'subject_ids': ['sub-03'],     # Список субъектов
    'task': 'imagine',             # Тип задачи
    'normalize': 'zscore_hybrid',  # Тип нормализации
    'exclude_channels': [124],     # Исключаемые каналы
}
```

### Параметры модели
```python
'model': {
    'n_channels': 125,             # Входные каналы
    'n_classes': 8,                # Выходные классы
    'proj_channels': 24,           # Каналы после projection
    'window_size_small': 128,      # Малое окно
    'stride_small': 96,            # Шаг малого окна
    'window_size_large': 256,      # Большое окно
    'stride_large': 128,           # Шаг большого окна
    'd_model': 128,                # Embedding dimension
    'n_heads': 4,                  # Attention heads
    'ff_dim': 256,                 # FFN dimension
    'n_layers': 2,                 # Transformer layers
    'dropout': 0.1,                # Dropout rate
    'cov_type': 'corr',            # 'corr' | 'trace'
    'oas_min_alpha': 0.1,          # Мин. коэффициент shrinkage
    'use_subject_embed': True,     # Subject embeddings
    'subject_embed_dim': 16,       # Dimension embeddings
    'subject_embed_dropout': 0.2,  # Dropout embeddings
}
```

### Параметры обучения
```python
'training': {
    'n_epochs': 50,
    'batch_size': 16,              # CUDA | 8 (CPU)
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'early_stopping_patience': 8,
    'use_amp': True,               # Automatic Mixed Precision
    'grad_clip': 1.0,
    'num_workers': 4,              # CUDA | 0 (CPU)
    'pin_memory': True,            # CUDA
    'persistent_workers': True,    # CUDA
    'prefetch_factor': 4,          # CUDA
}
```

### Оптимизация и scheduler
```python
'optimizer': {
    'name': 'adamw',
    'betas': [0.9, 0.999],
    'subject_embed_weight_decay': 5e-4,  # Отдельный WD для embeddings
}
'scheduler': {
    'name': 'cosine',
    'T_max': 20,
    'warmup_epochs': 3,
}
'loss': {
    'type': 'cb_focal',
    'beta': 0.999,
    'gamma': 1.75,
}
```

## Вычислительные особенности

### SPD операции с fallback
```python
# MPS (Apple Silicon) → CPU fallback
# CUDA FP16 → FP32 conversion
# Nan/Inf защита с clamp
```

### AMP (Automatic Mixed Precision)
- Включается только для CUDA
- GradScaler для stability
- Fallback на FP32 при overflow

### DataLoader оптимизация
```python
# CUDA режим
num_workers=4, pin_memory=True, 
persistent_workers=True, prefetch_factor=4

# CPU режим
num_workers=0, pin_memory=False
```

## Запуск

### Обучение
```bash
python3 Pipeline/train.py
```

### Быстрая проверка (1 эпоха)
```bash
python3 Pipeline/test_dryrun.py
```

### Классический ML (baseline)
```bash
python3 Pipeline/train_classical_ml.py
```

### С сохранением attention статистик
```bash
python3 Pipeline/train.py --save-attn
```

## Артефакты

### Сохраняемые файлы
```
Train/checkpoints/<exp>/best_model.pt    # Веса модели
Train/results/<exp>/
    ├── metrics.json                      # Итоговые метрики
    ├── history.json                      # История обучения
    ├── config_run.json                   # Конфиг запуска
    └── val_preds.npz                     # Предсказания + subject_id
```

### Формат metrics.json
```json
{
    "accuracy": 0.2399,
    "f1_macro": 0.2182,
    "precision_macro": 0.23,
    "recall_macro": 0.21,
    "balanced_accuracy": 0.2297,
    "loss": 1.3933,
    "per_class": [...]
}
```
