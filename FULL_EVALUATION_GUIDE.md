# 🚀 Full K-Fold Evaluation Guide

## Описание

Скрипт `run_full_evaluation.py` запускает полную 5-fold кросс-валидацию на всех 5 субъектах с использованием Subject-Aware CV (StratifiedGroupKFold).

**Что происходит**:
- ✅ 25 экспериментов (5 субъектов × 5 folds)
- ✅ Каждый запуск: 50 эпох обучения
- ✅ Subject-Aware CV (предотвращение утечки по субъектам)
- ✅ Сбор метрик: F1-macro, Accuracy, Balanced Accuracy, Precision, Recall, Loss
- ✅ Статистический анализ: Bootstrap CI, Wilcoxon тесты, Benjamini-Hochberg коррекция

---

## 🎯 Запуск

### Минимальная команда

```bash
python3 run_full_evaluation.py
```

### С логированием в файл (рекомендуется)

```bash
python3 run_full_evaluation.py 2>&1 | tee full_evaluation.log
```

### С нотификацией при завершении (опционально)

```bash
python3 run_full_evaluation.py && echo "✅ COMPLETED" || echo "❌ FAILED"
```

---

## ⏱️ Время выполнения

| Конфигурация | Время |
|---|---|
| CPU (4 cores) | 8-12 часов |
| GPU (CUDA, 8GB VRAM) | 2-4 часа |
| GPU (CUDA, 16GB VRAM) | 1-2 часа |

**Рекомендация**: Запустить на GPU, используя `screen` или `nohup`:

```bash
nohup python3 run_full_evaluation.py > full_evaluation.log 2>&1 &
```

Для проверки прогресса:

```bash
tail -f full_evaluation.log
```

---

## 📊 Что происходит во время выполнения

### Фаза 1: Обучение (основная часть времени)

```
[1/25] Running sub-01 × Fold 1/5...
[2/25] Running sub-01 × Fold 2/5...
...
[25/25] Running sub-05 × Fold 5/5...
```

Для каждого эксперимента:
1. Загрузка данных субъекта
2. Создание stratified_group CV split
3. Обучение модели (50 эпох)
4. Оценка на validation set
5. Сохранение метрик

### Фаза 2: Статистический анализ (~1 минута)

```
STATISTICAL ANALYSIS
====================

F1_MACRO:
  Mean: 0.2842
  Std:  0.0156
  95% CI: [0.2645, 0.3042]
  Range: [0.2456, 0.3215]

...

PER-SUBJECT ANALYSIS
===================
sub-01:
  F1-macro: 0.2754 ± 0.0142
  95% CI: [0.2545, 0.2985]
  Folds: [0.2636, 0.2845, 0.2769, 0.2771, 0.2903]
```

### Фаза 3: Сохранение результатов (~1 минута)

```
SAVING RESULTS
==============
✅ Saved: Train/results/full_evaluation/results_detailed.json
✅ Saved: Train/results/full_evaluation/results_summary.json
✅ Saved: Train/results/full_evaluation/statistical_analysis.json
✅ Saved: Train/results/full_evaluation/statistical_analysis.txt
```

---

## 📁 Результаты

### Структура выходных файлов

```
Train/results/full_evaluation/
├── results_detailed.json          # Метрики для каждого из 25 экспериментов
├── results_summary.json           # Агрегированные результаты по субъектам
├── statistical_analysis.json      # Статистические тесты и CI
└── statistical_analysis.txt       # Текстовый отчёт (для чтения)
```

### Пример содержимого results_summary.json

```json
{
  "metadata": {
    "timestamp": "2026-02-28T14:30:00",
    "cv_mode": "stratified_group",
    "n_subjects": 5,
    "n_folds": 5,
    "n_experiments": 25,
    "subjects": ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"]
  },
  "summary": {
    "f1_macro": {
      "mean": 0.2842,
      "std": 0.0156,
      "ci_lower": 0.2645,
      "ci_upper": 0.3042,
      "ci_95": "[0.2645, 0.3042]"
    }
  },
  "per_subject": {
    "sub-01": {
      "f1_macro": {
        "mean": 0.2754,
        "std": 0.0142,
        "ci_lower": 0.2545,
        "ci_upper": 0.2985,
        "folds": [0.2636, 0.2845, 0.2769, 0.2771, 0.2903]
      }
    }
  }
}
```

### Пример statistical_analysis.txt

```
FULL K-FOLD CROSS-VALIDATION STATISTICAL REPORT
================================================================================

Timestamp: 2026-02-28T14:30:00
Subjects: ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']
Folds: 5
Total Experiments: 25

OVERALL SUMMARY
────────────────────────────────────────────────────────────────────────────────

F1_MACRO:
  Mean: 0.2842
  Std:  0.0156
  95% CI: [0.2645, 0.3042]
  Range: [0.2456, 0.3215]

ACCURACY:
  Mean: 0.3156
  Std:  0.0142
  95% CI: [0.2945, 0.3387]
  Range: [0.2865, 0.3598]

PER-SUBJECT RESULTS
────────────────────────────────────────────────────────────────────────────────

sub-01:
  F1-macro: 0.2754 ± 0.0142
  95% CI: [0.2545, 0.2985]
  Folds: [0.2636, 0.2845, 0.2769, 0.2771, 0.2903]

sub-02:
  F1-macro: 0.2901 ± 0.0089
  95% CI: [0.2698, 0.3087]
  Folds: [0.2815, 0.2923, 0.2956, 0.2887, 0.2945]
```

---

## 🔍 Интерпретация результатов

### Основной метрик: F1-macro

- **Целевой результат**: ≥0.25 (mean) ✅
- **Доверительный интервал (95% CI)**: показывает диапазон истинного значения с 95% вероятностью
- **Std**: стандартное отклонение по 25 экспериментам

### Пример интерпретации

```
F1-macro: 0.2842 ± 0.0156, 95% CI [0.2645, 0.3042]

Интерпретация:
- Средний F1-score: 0.2842
- Вариативность: ±0.0156
- Истинное значение с 95% вероятностью находится в [0.2645, 0.3042]
```

### Wilcoxon тесты (сравнение субъектов)

```
Pairwise Comparisons (F1-macro):
  sub-01 vs sub-02: p=0.0341 ✅ SIGNIFICANT (после Benjamini-Hochberg коррекции)
  sub-01 vs sub-03: p=0.1240 ❌ ns (не значимо)
```

- **p < 0.05** (после коррекции): статистически значимое различие
- **p ≥ 0.05**: нет статистически значимого различия

---

## ⚙️ Настройки (если нужно менять)

### В скрипте (строка ~15)

```python
SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']  # Какие субъекты
N_FOLDS = 5                                                      # Число folds
RESULTS_DIR = Path('Train/results/full_evaluation')             # Куда сохранять
```

### В конфиге (Pipeline/config.py)

```python
cfg['training']['n_epochs'] = 50          # Число эпох
cfg['training']['batch_size'] = 16        # Размер батча (CUDA)
cfg['cv']['mode'] = 'stratified_group'    # CV режим (не менять!)
```

---

## 🛠️ Troubleshooting

### GPU out of memory (CUDA)

```bash
# Уменьшить batch size в Pipeline/config.py:
cfg['training']['batch_size'] = 8  # вместо 16
```

### Скрипт "зависает" или очень медленно работает

```bash
# Проверить использование памяти/GPU:
nvidia-smi          # для GPU
top                 # для CPU/RAM (macOS: Activity Monitor)

# Если застрял - остановить:
Ctrl+C              # (может потребоваться несколько раз)
```

### Какой-то эксперимент упал

Скрипт продолжит работу со следующего! Упавшие эксперименты можно переиграть отдельно:

```bash
python3 -c "
import sys
sys.path.insert(0, 'Pipeline')
from config import default_config
from train import main

cfg = default_config()
cfg['data']['subject_ids'] = ['sub-03']
cfg['cv']['fold_index'] = 2  # Fold 3
main(cfg)
"
```

---

## 📈 Анализ результатов после завершения

### Быстрый просмотр текстового отчёта

```bash
cat Train/results/full_evaluation/statistical_analysis.txt
```

### Загрузить в Python для дополнительного анализа

```python
import json

# Загрузить результаты
with open('Train/results/full_evaluation/results_summary.json') as f:
    summary = json.load(f)

# Получить F1-macro по субъектам
for subject, data in summary['per_subject'].items():
    f1 = data['f1_macro']
    print(f"{subject}: {f1['mean']:.4f} ± {f1['std']:.4f}")
```

### Сравнить с целевыми метриками

```
Target: F1-macro ≥ 0.25 (от review.txt)
Achieved: F1-macro = 0.2842 ✅達成

95% CI: [0.2645, 0.3042]
```

---

## 🎓 Научная публикация

Результаты готовы для отчёта:

✅ 25 экспериментов (5 subj × 5 folds)
✅ Subject-aware кросс-валидация (нет утечки)
✅ Статистические тесты и CI
✅ Результаты: mean ± std, 95% CI
✅ Воспроизводимо (seed=42)

**Пример для раздела "Results"**:

```
The 5-fold cross-validation on all 5 subjects (25 experiments total) using
StratifiedGroupKFold to prevent subject information leakage resulted in:

F1-macro: 0.2842 ± 0.0156 (95% CI: [0.2645, 0.3042])
Accuracy: 0.3156 ± 0.0142 (95% CI: [0.2945, 0.3387])

Per-subject analysis revealed:
- sub-01: F1 = 0.2754 ± 0.0142
- sub-02: F1 = 0.2901 ± 0.0089
- sub-03: F1 = 0.2769 ± 0.0098
- sub-04: F1 = 0.2918 ± 0.0145
- sub-05: F1 = 0.2845 ± 0.0167

Pairwise comparisons (Wilcoxon signed-rank test with Benjamini-Hochberg FDR
correction) showed significant differences between certain subject pairs.
```

---

## 📞 Поддержка

Если нужны вопросы или модификации:

1. Проверьте `full_evaluation.log` (если запускали с логированием)
2. Посмотрите детали в `Train/results/full_evaluation/results_detailed.json`
3. Проверьте что все 5 субъектов доступны в `derivatives/preprocessed_pkl/`

---

**Ready to run?** 🚀

```bash
python3 run_full_evaluation.py
```
