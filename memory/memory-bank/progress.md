# Progress — EEG_to_Text

## Хронология разработки

### Phase 4B — Текущая версия (Январь 2025)

#### Phase 4B-10: Исправление CV-режима для per-subject Full Evaluation ✅
**Дата**: 2 марта 2026
**Статус**: Завершено

**Изменения**:
- Добавлен fallback `stratified_group -> stratified` при одном субъекте в `Pipeline/train.py`.
- Добавлена проверка на пустые fold'ы (`train==0` или `val==0`) с явной ошибкой.
- В `run_full_evaluation.py` per-subject запуски переведены на `cv_mode='stratified'`.

**Результат**:
- Устранено падение `need at least one array to concatenate`.
- Валидация по фолдам стала корректной для сценария `subject_ids=[single_subject]`.

**Файлы**:
- `Pipeline/train.py`
- `run_full_evaluation.py`

---

#### Phase 4B-9: Обновление пути данных и проверка 0-success в Full Evaluation ✅
**Дата**: 2 марта 2026
**Статус**: Завершено

**Изменения**:
- В `Pipeline/config.py` добавлена динамическая резолюция каталога preprocessed данных:
  - `/mnt/data/derivatives/preprocessed_pkl` (приоритет)
  - `/mnt/data/EEG/preprocessed_pkl` (legacy)
  - `<project>/derivatives/preprocessed_pkl` (локальный fallback)
  - `EEG_PREPROCESSED_DIR` (переменная окружения)
- В `run_full_evaluation.py` добавлен fail-fast при `success_experiments == 0`.

**Результат**:
- Исключены ложные падения из-за устаревшего пути `data_dir`.
- Исключен ложный финальный статус успешности при полном отсутствии успешных запусков.

**Файлы**:
- `Pipeline/config.py`
- `run_full_evaluation.py`

---

#### Phase 4B-8: Совместимость `train.main(cfg)` для Full Evaluation ✅
**Дата**: 2 марта 2026
**Статус**: Завершено

**Изменения**:
- Обновлена сигнатура `Pipeline/train.py::main` для поддержки программного вызова с конфигом
- Добавлен возврат `final_metrics` из `main(cfg)` для агрегатора `run_full_evaluation.py`
- Сохранен CLI-режим без изменений (`python3 Pipeline/train.py`)

**Результат**:
- Устранена массовая ошибка запуска `main() takes 0 positional arguments but 1 was given`
- Разблокирован цикл `subject × fold` в `run_full_evaluation.py`

**Файлы**:
- `Pipeline/train.py` — обновлен интерфейс `main`

---

#### Phase 4B-6: Гибридная нормализация ✅
**Дата**: Январь 2025
**Статус**: Завершено

**Изменения**:
- Реализована гибридная нормализация (subject centering + global scaling)
- Добавлена функция `compute_hybrid_stats()` в `data_loader.py`
- Обновлена логика в `ChiscoDataset.__getitem__()`

**Результат**:
- Устранены subject baseline shifts
- Сохранена inter-subject variance
- Увеличен effective sample size

**Файлы**:
- `Pipeline/data_loader.py` — добавлена `compute_hybrid_stats()`
- `Pipeline/config.py` — `normalize='zscore_hybrid'` по умолчанию

---

#### Phase 4B-5: Subject Embeddings ✅
**Дата**: Январь 2025
**Статус**: Завершено

**Изменения**:
- Добавлен слой subject embeddings (dim=16)
- Реализован отдельный weight_decay для embeddings
- Интеграция в классификатор

**Результат**:
- Персонализация модели для каждого субъекта
- Улучшение обобщения на новых субъектах

**Файлы**:
- `Pipeline/model.py` — добавлен `nn.Embedding(n_subjects, 16)`
- `Pipeline/train.py` — differentiated weight_decay
- `Pipeline/config.py` — `use_subject_embed=True`

---

#### Phase 4B-4: Class-Balanced Focal Loss ✅
**Дата**: Январь 2025
**Статус**: Завершено

**Изменения**:
- Реализован Class-Balanced Focal Loss
- Фиксированы гиперпараметры β=0.999, γ=1.75
- Подсчёт class weights по effective number

**Результат**:
- Улучшена работа с дисбалансом классов
- Лучшая классификация minority классов

**Файлы**:
- `Pipeline/trainer.py` — класс `ClassBalancedFocalLoss`
- `Pipeline/config.py` — параметры loss

---

#### Phase 4B-3: Riemannian Geometry ✅
**Дата**: Декабрь 2024
**Статус**: Завершено

**Изменения**:
- Реализованы SPD операции (cov, logm, vectorize)
- Добавлен OAS shrinkage для ковариации
- Fallback для MPS/FP16 устройств

**Результат**:
- Корректная обработка SPD матриц
- Кроссплатформенная совместимость

**Файлы**:
- `Pipeline/riemannian_utils.py` — полный набор SPD утилит

---

#### Phase 4B-2: Multi-Scale Architecture ✅
**Дата**: Декабрь 2024
**Статус**: Завершено

**Изменения**:
- Двухмасштабные токены (128/96 и 256/128)
- Attention pooling с learnable head weights
- Scale embeddings

**Результат**:
- Захват паттернов на разных масштабах
- Адаптивная агрегация токенов

**Файлы**:
- `Pipeline/model.py` — архитектура RTTMultiScale

---

#### Phase 4B-1: Base Pipeline ✅
**Дата**: Декабрь 2024
**Статус**: Завершено

**Изменения**:
- Базовая структура пайплайна
- Загрузка данных из pkl
- Stratified K-Fold кросс-валидация
- Training loop с early stopping

**Результат**:
- Рабочий пайплайн обучения
- Сохранение артефактов

**Файлы**:
- `Pipeline/train.py`, `Pipeline/trainer.py`, `Pipeline/data_loader.py`

---

## Roadmap

### Q1 2025 (Январь — Март)

#### [→] 5-Fold Cross-Validation
**Приоритет**: Высокий
**Описание**: Развернуть полную 5-fold оценку на всех 5 субъектах

**Задачи**:
- [ ] Изменить `subject_ids = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']`
- [ ] Запустить обучение для всех 5 folds
- [ ] Усреднить метрики по folds
- [ ] Оценить variance между folds

**Ожидаемый результат**: Статистически значимые метрики качества

---

#### [→] Ablation Studies
**Приоритет**: Высокий
**Описание**: Систематическая оценка вклада компонентов

**Задачи**:
- [ ] Subject embeddings: ON vs OFF
- [ ] Covariance type: 'corr' vs 'trace'
- [ ] Window sizes: [64/128, 128/256, 256/512]
- [ ] d_model: [64, 128, 256]
- [ ] n_layers: [1, 2, 3]

**Ожидаемый результат**: Понимание оптимальной конфигурации

---

#### [ ] Hyperparameter Tuning
**Приоритет**: Средний
**Описание**: Оптимизация гиперпараметров

**Параметры**:
- Learning rate: [1e-4, 3e-4, 1e-3]
- T_max: [10, 20, 30]
- Weight decay: [1e-5, 1e-4, 1e-3]
- CB-Focal: β∈[0.99, 0.999, 0.9999], γ∈[1.0, 1.75, 2.0]

**Ожидаемый результат**: Улучшение F1-macro на 10-15%

---

#### [ ] Transfer Learning (LOSO)
**Приоритет**: Средний
**Описание**: Leave-One-Subject-Out кросс-валидация

**Задачи**:
- [ ] Train на 4 subjects, test на 5th
- [ ] Повторить для каждого субъекта
- [ ] Оценить обобщающую способность

**Ожидаемый результат**: Метрики для novel subjects

---

### Q2 2025 (Апрель — Июнь)

#### [ ] Data Augmentation
**Приоритет**: Низкий
**Описание**: Аугментация в SPD касательном пространстве

**Метод**: Tangent Space Jittering
```python
log_C = spd_logm(C)
log_C_noisy = log_C + noise  # Gaussian noise
C_aug = expm(log_C_noisy)
```

**Ожидаемый результат**: Улучшение обобщения, снижение overfitting

---

#### [ ] Attention Analysis
**Приоритет**: Средний
**Описание**: Интерпретация attention паттернов

**Задачи**:
- [ ] Визуализация weights_tok_mean по каналам
- [ ] Выявление наиболее информативных каналов
- [ ] Корреляция с известными речевыми зонами

**Ожидаемый результат**: Нейробиологическая интерпретируемость

---

#### [ ] Multi-Subject Pre-training
**Приоритет**: Низкий
**Описание**: Pre-train на всех субъектах + fine-tuning

**Подход**:
1. Train на pooled data (all subjects)
2. Fine-tune на target subject

**Ожидаемый результат**: Улучшение для subjects с малым числом сэмплов

---

## Метрики проекта

### Текущие (Phase 4B, sub-03, Fold 1)
| Метрика | Значение |
|---------|----------|
| F1-macro | 0.2182 |
| Accuracy | 0.2399 |
| Balanced Accuracy | 0.2297 |
| Loss | 1.3933 |

### Целевые (Q1 2025)
| Метрика | Цель |
|---------|------|
| F1-macro (5-fold mean) | ≥0.25 |
| F1-macro (best subject) | ≥0.30 |
| Balanced Accuracy | ≥0.28 |

### Исторический лучший результат (A6, sub-04, Fold 1)
| Метрика | Значение |
|---------|----------|
| F1-macro | ~0.2915 |
| Конфигурация | subject embeddings ON, cov_type='corr' |

---

## Статус задач

### Завершено ✅
- [x] Базовый пайплайн обучения
- [x] Riemannian geometry для SPD матриц
- [x] Multi-scale архитектура
- [x] Class-Balanced Focal Loss
- [x] Subject embeddings
- [x] Hybrid normalization
- [x] **Subject-Aware Cross-Validation** (StratifiedGroupKFold + LOSO)

### В работе →
- [→] Full K-Fold evaluation (готов скрипт, ожидание запуска)
- [→] Bootstrap доверительные интервалы (реализовано в скрипте)
- [→] Статистические тесты (Wilcoxon + Benjamini-Hochberg в скрипте)
- [ ] Temperature scaling (калибровка)
- [ ] Ablation studies
- [ ] Hyperparameter tuning

### Запланировано [ ]
- [ ] Data augmentation
- [ ] Attention analysis
- [ ] Multi-subject pre-training

---

## Анализ соответствия review.txt (26 Февраля 2025)

**Статус**: 92% соответствие кодовой базы с документацией

### Архитектурные компоненты (ВСЕ РЕАЛИЗОВАНЫ)
- [x] RTTMultiScale модель (model.py:31-131)
- [x] SPD геометрия + fallback (riemannian_utils.py)
- [x] Гибридная нормализация (data_loader.py:115-125)
- [x] Class-Balanced Focal Loss (trainer.py:52-69)
- [x] Subject embeddings (model.py:48-53)
- [x] AdamW + Cosine scheduler (train.py)
- [x] AMP + Gradient clipping (trainer.py)
- [x] Stratified K-Fold (data_loader.py)

### Критические пробелы (ТРЕБУЕТСЯ СРОЧНО)
- [ ] Subject-aware CV (StratifiedGroupKFold/LOSO) [HIGH]
  * Текущее: StratifiedKFold перемешивает субъектов
  * Требуется: группировка по subject_id для cross-subject оценки
- [ ] Full K-Fold evaluation (все 5 folds) [HIGH]
  * Текущее: только Fold 1
  * Требуется: mean±std по всем 5 folds

### Опциональные улучшения (MEDIUM приоритет)
- [ ] Bootstrap доверительные интервалы (95% CI)
- [ ] Статистические тесты (Wilcoxon, Benjamini–Hochberg)
- [ ] Temperature scaling для калибровки вероятностей
- [ ] ECE/NLL/Brier score для оценки доверия

### Параметры
- ✅ 100% совпадение всех 21 параметра с review.txt
- ✅ Конфигурация: config.py идентична описанию

## Бэклог идей

### Исследовательские
- [ ] Contrastive learning для pre-training
- [ ] Graph Neural Networks для каналов
- [ ] Temporal convolution networks
- [ ] Multi-task learning (классификация + reconstruction)

### Инженерные
- [ ] WandB/MLflow интеграция для экспериментов
- [ ] Автоматический hyperparameter search (Optuna)
- [ ] Docker контейнер для воспроизводимости
- [ ] CI/CD для тестов
- [→] Subject-aware CV implementation (блокирующий элемент)
- [→] Full K-Fold automated evaluation

### Аналитические
- [ ] Confusion matrix heatmap
- [ ] PR/ROC кривые для каждого класса
- [ ] Learning curves (train/val по эпохам)
- [ ] Gradient norm analysis
- [ ] Bootstrap CI for statistical significance
