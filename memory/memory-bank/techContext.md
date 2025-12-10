# Технический контекст

## Язык и версии
- Python ≥3.12
- PyTorch ≥2.2, NumPy ≥1.26, SciPy ≥1.11, scikit-learn ≥1.3, Transformers ≥4.40
- Доп.: pandas, matplotlib, openpyxl, tqdm, mne, pyprep, mne-icalabel, h5py, mat73

## Структура проекта
- `Pipeline/` — код обучения (config, data_loader, model, riemannian_utils, trainer, train, feature_engineering, train_classical_ml, test_dryrun)
- `json/` — маппинги классов и текстов (`classnumber.json`, `textmaps.json`, `metaclasses.json`)
- `derivatives/preprocessed_pkl/` — предобработанные pkl-данные (ожидается на диске)
- `Train/` — артефакты обучения: чекпойнты и метрики

## Запуск
- DL-пайплайн: `python3 Pipeline/train.py`
- Классический ML: `python3 Pipeline/train_classical_ml.py`

## Оценка и семплинг (обновление)
- Для межсубъектной оценки: `StratifiedGroupKFold` (группа = subject) или LOSO (Leave-One-Subject-Out) из scikit-learn.
- Для балансировки без потери данных: `WeightedRandomSampler` в PyTorch вместо агрессивного undersampling.

## Конфигурация (`Pipeline/config.py`)
- Ключевые поля:
  - `data`: `data_dir`, `subject_ids`, `task`, `normalize`, `exclude_channels`
  - `model`: `n_classes`, окна/шаги (малый/большой), `d_model`, `n_layers`, `cov_type`, `use_subject_embed`, `subject_embed_dim`
  - `training`: `n_epochs`, `batch_size`, `learning_rate`, `weight_decay`, `use_amp`, `grad_clip`, `num_workers`, `pin_memory`
  - `cv`: `n_splits`, `random_state`
  - `optimizer`, `scheduler`, `loss`, `seed`, `device`, `checkpoint_dir`, `results_dir`

## Данные и маппинги
- `load_all_data()` загружает pkl для субъектов/задачи; `load_all_data_metaclass()` конвертирует метки в 8 метаклассов.
- Маппинги читаются из `json/*.json`. Исключение канала(ов) через `exclude_channels`.

## Устойчивость/совместимость
- SPD-операции с бэкенд-фолбэками (CPU/MPS/FP16→FP32) для стабильности eigendecomposition.
- Автоматика устройства: CUDA при доступности, иначе CPU.
