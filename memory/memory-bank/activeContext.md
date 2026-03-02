# Active Context — EEG_to_Text

## Текущий фокус работы

### Статус проекта: Phase 4B — Реализация Subject-Aware CV

**Последняя активность**: Синхронизированы документы `review/*` с фактическими full-evaluation результатами (2 марта 2026)

### Текущая конфигурация по умолчанию
```python
subject_ids = ['sub-03']  # Одиночный субъект для отладки
n_epochs = 50
batch_size = 16 (CUDA) / 8 (CPU)
use_subject_embed = True
subject_embed_dim = 16
normalize = 'zscore_hybrid'
```

### Последние изменения

#### Phase 4B-11: Синхронизация review-документации с full evaluation ✅ [2 марта 2026]
**Изменение**:
- Обновлен `review/review.txt` в разделах 10.7, 11.1, 11.7 и 13:
  - заменены устаревшие Fold-1 метрики на результаты 25/25 запусков;
  - добавлены агрегированные full-evaluation метрики и bootstrap CI;
  - обновлены ограничения с учётом выполненного полнофолдового этапа.
- Обновлен путь к данным в разделе воспроизводимости:
  `/mnt/data/derivatives/preprocessed_pkl/...`.
- Актуализированы `review/COMPARISON_SUMMARY.txt`, `review/analysis_report.md`,
  `review/NEXT_STEPS_TEMPLATE.py` блоками «что сделано / что дальше».

**Результат**:
- Документация и фактическое состояние кодовой базы согласованы.
- Убраны противоречия вида «только Fold 1 / subject-aware не выполнен» в ключевых разделах отчёта.

---

#### Phase 4B-10: Фикс CV-режима для per-subject evaluation ✅ [2 марта 2026]
**Проблема**: При запуске одного субъекта `stratified_group` приводил к пустой валидации (`val=0`) и падению на `need at least one array to concatenate`.

**Изменение**:
- `Pipeline/train.py`: если в данных <2 уникальных групп, `stratified_group` автоматически переключается на `stratified`.
- `Pipeline/train.py`: добавлена явная проверка на пустой `train/val` fold с понятной ошибкой.
- `run_full_evaluation.py`: для per-subject запусков явно используется `cv_mode='stratified'`.

**Результат**:
- Устранена ошибка `need at least one array to concatenate` из-за `val=0`.
- Full evaluation корректно обучается по фолдам внутри каждого субъекта.

---

#### Phase 4B-9: Исправление пути данных + fail-fast в full evaluation ✅ [2 марта 2026]
**Проблема**: Данные перемещены в `/mnt/data/derivatives/preprocessed_pkl`, а конфиг использовал legacy путь `/mnt/data/EEG/preprocessed_pkl`.

**Изменение**:
- `Pipeline/config.py`: добавлен `_resolve_preprocessed_dir()` с fallback по кандидатам путей и поддержкой `EEG_PREPROCESSED_DIR`.
- `run_full_evaluation.py`: добавлен вывод `Data dir`, счетчики `success_experiments/failed_experiments`.
- `run_full_evaluation.py`: при 0 успешных экспериментов скрипт теперь завершает работу с понятной ошибкой.

**Результат**:
- Устранена причина загрузки `0` сэмплов при корректно смонтированных данных.
- Исключен ложный статус "pipeline completed successfully" при полном провале обучения.

---

#### Phase 4B-8: Фикс программного запуска train.main ✅ [2 марта 2026]
**Проблема**: `run_full_evaluation.py` вызывал `train_main(cfg)`, но `Pipeline/train.py::main()` не принимал аргументы.

**Изменение**:
```python
# Pipeline/train.py
def main(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if cfg is None:
        # CLI режим
    else:
        # Программный режим из run_full_evaluation.py
```

**Результат**:
- Устранена ошибка `main() takes 0 positional arguments but 1 was given`
- Сохранена обратная совместимость CLI (`python3 Pipeline/train.py`)
- `main(cfg)` теперь возвращает `final_metrics` для агрегации в full evaluation

---

#### Phase 4B-7: Subject-Aware Cross-Validation ✅ [28 февраля 2026]
**Изменение**: Внедрены функции StratifiedGroupKFold и LOSO для правильной кросс-валидации

**Реализация**:
```python
# data_loader.py
- get_stratified_group_cv_splits(labels, groups, n_splits=5)
  └─ Гарантирует что субъекты не пересекаются между train/val
- get_loso_splits(samples, subject_mapping)
  └─ Leave-One-Subject-Out для максимально строгой оценки

# config.py
- cv['mode']: 'stratified_group' (default), 'stratified' (old), 'loso'

# train.py (build_loaders)
- Поддержка всех трёх CV режимов
- Логирование выбранного режима
- Backward compatible
```

**Статус**: ✅ Реализовано и протестировано
- Stratified Group CV: субъекты не пересекаются
- LOSO: каждый субъект по очереди в тесте
- Backward compatible: старый код продолжает работать

**Следующие шаги**:
1. Запустить 5-fold evaluation со всеми субъектами
2. Собрать mean±std метрики
3. Добавить статистические тесты

---

#### Phase 4B-6: Гибридная нормализация
**Изменение**: Переход от `zscore_subject_channel` к `zscore_hybrid`

**Мотивация**:
- Subject-wise centering устраняет baseline shifts
- Global scaling сохраняет inter-subject variance
- Увеличивает effective sample size для обучения

**Реализация**:
```python
compute_hybrid_stats():
    # Step 1: Per-subject means
    for each subject: μᵢ = mean(Xᵢ)
    
    # Step 2: Global std on centered data
    X_centered = [Xᵢ - μᵢ for all i]
    σ_global = std(concat(X_centered))
    
    # Apply: X̂ = (X - μᵢ) / σ_global
```

#### Phase 4B-5: Subject Embeddings
**Изменение**: Добавлены learnable subject embeddings

**Параметры**:
- Dimension: 16
- Dropout: 0.2
- Separate weight_decay: 5e-4

**Интеграция**:
```python
combined = cat([h_cls, h_attn, subject_emb], dim=-1)
```

#### Phase 4B-4: Class-Balanced Focal Loss
**Изменение**: Фиксированы гиперпараметры loss функции

**Параметры**:
- β = 0.999 (для class weights)
- γ = 1.75 (для focal weighting)

### Следующие шаги (ПО ПРИОРИТЕТУ)

#### ✅ ЗАВЕРШЕНО
1. **Subject-aware Cross-Validation** ✅ ЗАВЕРШЕНО
   - Внедрена StratifiedGroupKFold и LOSO
   - Группировка по subject_id
   - Код: Pipeline/data_loader.py, train.py, config.py

2. **Full K-Fold Evaluation Script** ✅ ГОТОВ К ЗАПУСКУ
   - Создан скрипт run_full_evaluation.py
   - 25 экспериментов (5 subj × 5 folds)
   - Встроена статистика: Bootstrap CI, Wilcoxon тесты, Benjamini-Hochberg
   - Запуск: `python3 run_full_evaluation.py`
   - Время: 2-4 часа (на GPU)

#### 🟡 ВЫСОКИЕ (Желательны)
3. **Bootstrap Доверительные Интервалы** [1-2 дня]
   - 95% CI по фолдам
   - Использовать np.percentile для bootstrap
   - Файл: analysis_tools/

4. **Статистические Тесты** [1-2 дня]
   - Wilcoxon signed-rank для попарных сравнений
   - Benjamini–Hochberg коррекция для множественных сравнений
   - Файл: analysis_tools/

5. **Temperature Scaling (Калибровка)** [1 день]
   - Post-hoc на validation set
   - Вычисление ECE, NLL, Brier score
   - Файл: trainer.py

#### 🟢 ОПЦИОНАЛЬНЫЕ
6. **5-fold кросс-валидация** на всех 5 субъектах (ПОСЛЕ Subject-aware CV)
   - Развернуть `subject_ids = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']`
   - Усреднить метрики по folds
   - Оценить variance между folds

7. **Абляция subject embeddings**
   - Запуск с `use_subject_embed = False`
   - Сравнение с текущей версией
   - Количественная оценка вклада

8. **Анализ attention паттернов**
   - Запуск с `--save-attn`
   - Визуализация `weights_tok_mean` по каналам
   - Выявление наиболее информативных каналов

#### Среднесрочные
4. **Гиперпараметры tuning**
   - Learning rate: [1e-4, 3e-4, 1e-3]
   - T_max: [10, 20, 30]
   - Weight decay: [1e-5, 1e-4, 1e-3]

5. **Абляция архитектуры**
   - cov_type: 'corr' vs 'trace'
   - Window sizes: [64/128, 128/256, 256/512]
   - d_model: [64, 128, 256]
   - n_layers: [1, 2, 3]

6. **Transfer learning (LOSO)**
   - Train на 4 subjects, test на 5th
   - Оценка обобщающей способности

### Известные проблемы

#### ⚠️ Низкие метрики
**Симптом**: F1-macro ~0.22 при целевом ~0.29

**Гипотезы**:
- Недостаточно данных для одного субъекта
- Субоптимальные гиперпараметры
- Требуется больше эпох обучения

**План диагностики**:
1. Проверить learning curves на overfitting/underfitting
2. Анализ confusion matrix для выявления проблемных классов
3. Проверка градиентов на vanishing/exploding

#### ⚠️ Межсубъектная вариабельность
**Симптом**: Метрики сильно различаются между субъектами

**Текущее решение**: Subject embeddings + hybrid normalization

**Мониторинг**: Per-subject метрики в analysis_tools/subject_effects.py

### Активные эксперименты

| Эксперимент | Статус | Конфигурация | Метрики |
|------------|--------|--------------|---------|
| phase4b_5subjects_CUDA | Completed | sub-03, Fold 1 | F1=0.2182 |
| A6 (historical) | Completed | sub-04, Fold 1 | F1≈0.2915 |

### Ресурсы

#### Данные
- **Расположение**: `derivatives/preprocessed_pkl/<subject>/eeg/`
- **Формат**: pkl файлы с ЭЭГ [125, T]
- **Задача**: imagine (воображаемая речь)
- **Классы**: 39 → 8 метаклассов

#### Код
- **Обучение**: `Pipeline/train.py`
- **Модель**: `Pipeline/model.py` (RTTMultiScale)
- **Данные**: `Pipeline/data_loader.py`
- **Анализ**: `analysis_tools/`

#### Артефакты
- **Checkpoints**: `Train/checkpoints/phase4b_5subjects_CUDA/`
- **Результаты**: `Train/results/phase4b_5subjects_CUDA/`

### Запуск

```bash
# Обучение (CUDA)
python3 Pipeline/train.py

# Обучение (CPU, отладка)
python3 Pipeline/train.py  # device определится автоматически

# Быстрая проверка (1 эпоха)
python3 Pipeline/test_dryrun.py

# С сохранением attention статистик
python3 Pipeline/train.py --save-attn

# Анализ результатов
python3 analysis_tools/run_analysis_suite.py
```
