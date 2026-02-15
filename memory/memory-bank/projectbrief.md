# Проект: EEG_to_Text — Phase 4B

## Цель
- Распознавание воображаемой речи по ЭЭГ-сигналам с классификацией в 8 метаклассов (сопоставление 39→8 через `json/metaclasses.json`).
- Исследовать и улучшать устойчивость к межсубъектным сдвигам и дисбалансу классов.

## Основные требования
- Данные: предобработанные pkl-файлы в `/mnt/data/EEG/preprocessed_pkl/<subject>/eeg/*task-imagine*.pkl` (обновлено 2026-02-15).
- Субъекты: конфигурацией (по умолчанию `['sub-04']` — лучший исторический результат).
- Метрики: приоритет `f1_macro`, также `accuracy`, `balanced_accuracy`, `precision_macro`, `recall_macro`, `loss`.
- Воспроизводимость: фиксированный seed, сохранение `best_model.pt` и `metrics.json`.
- Производительность: поддержка CPU/GPU (CUDA/MPS), стабильность линейной алгебры на SPD.

## Подход
- Модель RTTMultiScale (Riemannian Token Transformer):
  - Формирование токенов через SPD-представления окон EEG (логарифмическая карта, векторизация).
  - Трансформер-энкодер с двухмасштабными окнами, attention pooling.
  - Необязательные subject embeddings для персонализации.
- Предобработка/нормализация:
  - Гибридная нормализация: subject-wise центрирование + глобальное масштабирование (Phase 4B-6).
  - Исключение шумного канала (`exclude_channels: [124]`).
- Обучение:
  - Class-Balanced Focal Loss, AdamW, CosineAnnealingLR (+warmup), ранняя остановка.
  - Стратифицированные K-Fold разбиения; в текущем коде используется первый фолд.

## Архитектура кода (слои)
- `Pipeline/config.py` — константы и `default_config()`.
- `Pipeline/data_loader.py` — загрузка, маппинги классов/метаклассов, нормализация, датасеты.
- `Pipeline/riemannian_utils.py` — операции на SPD, устойчивые разложения.
- `Pipeline/model.py` — архитектура RTTMultiScale, subject embeddings.
- `Pipeline/trainer.py` — цикл обучения, метрики, сохранение артефактов.
- `Pipeline/train.py` — точка входа, сборка компонентов и запуск.
- Дополнительно: `Pipeline/feature_engineering.py`, `Pipeline/train_classical_ml.py` — классические ML-бейзлайны.

## Текущий статус
- Текущая лучшая метрика для `python3 Pipeline/train.py` на пациенте `sub-04` (Fold 1):
  - f1_macro: 0.2737, accuracy: 0.2874, balanced_accuracy: 0.2929, loss: 1.4607
  - Артефакты: `Train/checkpoints/phase4b_5subjects_CPU/best_model.pt`, метрики: `Train/results/phase4b_5subjects_CPU/metrics.json`.

### Новые результаты (2025-12-10)
- Идеально сбалансированный датасет по метаклассам из `sub-01..sub-04` (11,464 примеров; 1,433/класс; 51.9% удалено) дал худшую валидацию, чем обучение только на `sub-04`.
- Далее приоритет: subject-aware CV (StratifiedGroupKFold/LOSO), отказ от сильного undersampling в пользу re-weighting/WeightedRandomSampler.

### Результаты абляций (2025-12-11)
- A2: лучший `CB-Focal(gamma=1.75, beta=0.999)` — Fold1 f1=0.2908 (+1.71% абс. к базовому), 5-fold mean f1=0.2686.
- A3: расписание LR — вернуть режим A2 (`T_max=20`, `warmup=3`).
- A4/A5: по умолчанию `gating=False`, `attn_heads=1`.
- A6: subject embeddings — `dim=16`, `dropout=0.2`, `embed_wd=5e-4` (кандидат дефолта).
- A7: `stride_small=96` (база) лучше, чем 80/64.

## Ограничения и риски
- Межсубъектные сдвиги (subject shift), дисбаланс классов, высокая вариативность сигналов.
- Ограниченный набор субъектов; доступность/объём данных.
- Стоимость SPD-операций и устойчивость разложений на разных бэкендах (CPU/CUDA/MPS).

## Критерии приемки
- Конвейер обучения воспроизводим, сохраняет артефакты и метрики.
- Улучшение `f1_macro` относительно текущего бейзлайна или прохождение оговорённых абляций.
