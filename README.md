# EEG_to_Text — Phase 4B

Репозиторий для обучения моделей распознавания воображаемой речи по ЭЭГ.
Основа — двухмасштабный Riemannian Token Transformer (RTTMultiScale) с
гибридной нормализацией и subject embeddings (включены по умолчанию).

## Быстрый старт
- Python 3.12+
- Установить зависимости: `pip install -r requirements.txt`
- Подготовить данные:
  - Предобработанные pkl-файлы ожидаются в `derivatives/preprocessed_pkl/<subject>/eeg/*task-imagine*.pkl`
  - Маппинги находятся в `json/` (`classnumber.json`, `textmaps.json`, `metaclasses.json`)

## Запуск
- Глубокое обучение:
  - `python3 Pipeline/train.py`
  - Артефакты: чекпойнт — `Train/checkpoints/<exp>/best_model.pt`, метрики — `Train/results/<exp>/metrics.json`
- Быстрая проверка (1 эпоха):
  - `python3 Pipeline/test_dryrun.py`
- Классический ML (признаки + RF/XGBoost):
  - `python3 Pipeline/train_classical_ml.py`

## Конфигурация (Pipeline/config.py)
- Ключевые параметры:
  - `data`: `data_dir`, `subject_ids` (дефолт `['sub-01']`), `task`, `normalize='zscore_hybrid'`, `exclude_channels=[124]`
  - `model`: окна/шаги (128/96 и 256/128), `d_model`, `n_layers`, `cov_type='corr'`, `use_subject_embed=True`, `subject_embed_dim=16`, `subject_embed_dropout=0.2`
  - `training`: `n_epochs`, `batch_size`, `learning_rate`, `weight_decay`, `early_stopping_patience`, `use_amp`, `grad_clip`, `num_workers`, `pin_memory`
  - `cv`: `n_splits=5`, `random_state`
  - `optimizer`, `scheduler`, `loss` (Class-Balanced Focal Loss)
- Устройство выбирается автоматически (`cuda` при доступности, иначе `cpu`).

## Как это работает (схема)
1) Загрузка pkl → 2) маппинг 39→8 метаклассов → 3) Stratified K-Fold →
4) Нормстаты (по train): гибридная нормализация (центрирование по субъекту + глобальный std) →
5) Dataset/DataLoader (исключение каналов) → 6) RTTMultiScale (SPD-токены, Transformer) →
7) Обучение/валидация (Class-Balanced Focal, Cosine LR, ранняя остановка) → 8) сохранение артефактов.

## Ключевые особенности
- Римановы признаки: окно EEG → ковариация (OAS) → корреляция → logm → векторизация (верхний треугольник).
- Двухмасштабные токены (малые/большие окна), attention pooling.
- Subject embeddings (опция) для персонализации в классификаторе.
- Устойчивость к платформам: SPD-операции с fallback для MPS/FP16.

## Текущие метрики
- Последний сохранённый прогон: `Train/results/phase4b_5subjects_CUDA` (sub-04, Fold 1):
  - f1_macro: 0.2182, accuracy: 0.2399, balanced_accuracy: 0.2297, loss: 1.3933.
  - Артефакты: `config_run.json`, `history.json`, `metrics.json`, `val_preds.npz` (с subject_id/sample_id).
- Исторически лучший сетап (A6 на базе A2/A3, sub-04 Fold 1) давал f1_macro≈0.2915; он соответствует текущим дефолтным гиперпараметрам. Текущий дефолт `subject_ids=['sub-01']`; для воспроизведения лучших метрик переключите на `sub-04`.

## Советы по исследованию
- Развернуть 5-fold оценку и усреднить метрики.
- Абляции: `use_subject_embed` on/off, `cov_type`=`corr|trace`, окна/шаги, `proj_channels`, `d_model`, `n_layers`.
- Подбор гиперпараметров: `lr`, `T_max`, `weight_decay`, CB-Focal (`beta`, `gamma`).
- Эксперименты на нескольких субъектах и переносимость.

## Структура
- `Pipeline/` — код пайплайна: config, data_loader, model, riemannian_utils, trainer, train, feature_engineering, train_classical_ml, test_dryrun
- `json/` — маппинги классов и текстов
- `derivatives/preprocessed_pkl/` — предобработанные EEG (внешняя папка данных)
- `Train/` — артефакты обучения (`results/**`)
- `analysis_tools/` — скрипты постобработки (конфьюжн, PR/ROC, кривые обучения, абляции, subject effects и др.)

## Лицензия
См. `COPYRIGHT.md`.
