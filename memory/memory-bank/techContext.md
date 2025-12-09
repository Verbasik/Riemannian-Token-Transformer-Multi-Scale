# Технологический контекст

## Стек и зависимости
- Язык: Python 3.10+
- ML/DL: PyTorch 2.x, scikit‑learn, NumPy, SciPy, PyWavelets, XGBoost (опционально)
- Файлы данных: `Chisco/derivatives/preprocessed_pkl/` (pkl со списками образцов), `Chisco/json/` (classnumber/textmaps/metaclasses)

## Модули и ответственность
- `config.py`: пути/константы и `default_config()` — единая точка конфигурации (данные, модель, обучение, CV, оптимизатор/шедулер, лосс, каталоги артефактов).
- `data_loader.py`: загрузка данных, маппинги классов→мета‑классов, Dataset/Subset, стратифицированные фолды, статистики нормализации (global/subject/hybrid), маппинг субъектов для embeddings.
- `riemannian_utils.py`: SPD‑операции (OAS‑ковариации, корреляции, log‑map, векторизация), аугментация в касательном пространстве, разбиение на окна.
- `model.py`: RTTMultiScale — multi‑scale окна → SPD признаки → TransformerEncoder → attention pooling → классификатор; опциональные subject embeddings.
- `trainer.py`: ClassBalancedFocalLoss, метрики, цикл обучения, оценка, сохранение артефактов (веса/метрики), ранняя остановка.
- `train.py`: оркестрация (build_* фабрики, запуск train_loop, печать/сохранение результатов).
- `feature_engineering.py`: извлечение признаков (PSD/частотные соотношения/статистика/вейвлет/Hjorth) и батч‑обработка.
- `train_classical_ml.py`: baseline для классического ML (RF/XGB), стандартизация и отчёт.
- `utils.py`: сиды, печать метрик, совместимость pickle NumPy.
- `test_dryrun.py`: быстрый тест (1 эпоха) — требует синхронизации сигнатур с `train.py` (см. Ниже).

## Потоки данных и формы тензоров
1) Загрузка: pkl → список словарей `{subject, run, text, label, eeg}`; `label` преобразуется в мета‑класс.
2) Dataset: нормализация по выбранному режиму (`zscore_hybrid` по умолчанию) и возврат тензоров: `eeg: FloatTensor [C,T]`, `label: LongTensor []`, `subject_id: LongTensor []`.
3) DataLoader: батчи формируются как `eeg [B,C,T]`, `label [B]`, `subject_id [B]`.
4) Модель: оконное разбиение → SPD ковариации/корреляции → log‑map → векторизация → линейная проекция → TransformerEncoder → attention pooling. При включённых subject embeddings — конкатенация эмбеддинга субъекта к пулам и классификация.

## Конфигурация (основные ключи)
- `data`: `data_dir`, `subject_ids`, `task`, `normalize`, `exclude_channels`
- `model`: `n_channels`, `n_classes`, `proj_channels`, `window_size_*`, `stride_*`, `d_model/n_heads/ff_dim/n_layers`, `dropout`, `eps`, `attn_heads`, `gating`, `cov_type`, `use_subject_embed`, `subject_embed_dim`
- `training`: `n_epochs`, `batch_size`, `learning_rate`, `weight_decay`, `early_stopping_patience`, `use_amp`, `grad_clip`, `num_workers`, `pin_memory`
- `cv/optimizer/scheduler/loss/seed/device/checkpoint_dir/results_dir`

## Запуск
- DL тренировка: `python train.py`
- Классический baseline: `python train_classical_ml.py`
- Dry‑run: `python test_dryrun.py` (в текущей версии скрипт требует обновления: `build_loaders` теперь возвращает `(train_loader, val_loader, train_labels, effective_channels, n_subjects)`, а `build_model` требует `n_subjects`).

## Артефакты
- Весы модели: `Train/checkpoints/.../best_model.pt`
- Метрики: `Train/results/.../metrics.json`

## Воспроизводимость
- Функция `set_seed` фиксирует сиды PyTorch/NumPy и отключает nondeterministic режимы CuDNN.

