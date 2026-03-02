<div align="center">

# 🧠 EEG to Text
## Imagined-Speech EEG Decoder

Дeкодирование воображаемой речи с помощью **Римановой геометрии** + **мультимасштабный Transformer**

```
╔═══════════════════════════════════════════════════════════════╗
║  Классификация EEG сигналов в 8 мета-классов                  ║
║  Riemannian SPD-токены + MultiScale Attention Pooling         ║
╚═══════════════════════════════════════════════════════════════╝
```

---

<a href="#-быстрый-старт"><img src="https://img.shields.io/badge/Quickstart-ready-00b894?style=for-the-badge" alt="Quickstart"></a>
<a href="#-требования"><img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge" alt="Python"></a>
<a href="#-требования"><img src="https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c?style=for-the-badge" alt="PyTorch"></a>
<a href="#-данные"><img src="https://img.shields.io/badge/Data-Chisco-informational?style=for-the-badge" alt="Data"></a>
<a href="#-бенчмарки"><img src="https://img.shields.io/badge/Status-Benchmark-6f42c1?style=for-the-badge" alt="Status"></a>
<a href="#-лицензия"><img src="https://img.shields.io/badge/License-CUSTOM-lightgrey?style=for-the-badge" alt="License"></a>

<sub>📧 **Автор:** Вербецкий Эдуард Игоревич · МАИ (НИУ) · [verbasik@gmail.com](mailto:verbasik@gmail.com)</sub>

</div>

---

## 📑 Навигация

| 🚀 Быстрый старт | 📚 Полная документация | 🔧 Конфигурация |
|:---:|:---:|:---:|
| [Установка](#-быстрый-старт) | [Использование](#-использование) | [Параметры](#-конфигурация) |
| [Первый запуск](#проверка-1-эпоха) | [Архитектура](#-архитектура) | [Структура](#-структура-проекта) |

---

## ✨ Ключевые особенности

<table>
<tr>
<td width="50%">

### 🎯 Точность & Устойчивость
- ✅ **F1-macro: 0.265 ± 0.011**
- ✅ **Accuracy: 0.288 ± 0.013**
- ✅ **Balanced Accuracy: 0.285 ± 0.014**
- ✅ **Выше случайного (~0.125) в 2.3x**

</td>
<td width="50%">

### 🧮 Технологический стек
- 🔷 **Риманова геометрия** (SPD матрицы)
- 🔶 **Мультимасштабные токены** (128/96 + 256/128)
- 🟦 **Transformer архитектура** (2 слоя, 4 heads)
- 🟩 **Гибридная нормализация** + Subject embeddings

</td>
</tr>
</table>

---

## 🚀 Быстрый старт

### 📋 Требования

```
Python 3.12+
PyTorch 2.2+
CUDA 11.8+ (опционально, для ускорения)
```

### 💻 Установка

```bash
# Создание виртуального окружения
python -m venv .venv
source .venv/bin/activate  # На Windows: .venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt
```

> **💡 Совет:** Для GPU-ускорения установите `torch` с поддержкой CUDA:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

### 📁 Данные

```
Предобработанные EEG сигналы:
├── /mnt/data/derivatives/preprocessed_pkl/
│   └── <subject>/eeg/*task-imagine*.pkl
│
Словари и маппинги:
├── json/
│   ├── classnumber.json      (39 исходных классов)
│   ├── textmaps.json         (текстовые описания)
│   └── metaclasses.json      (мэппинг 39→8)
```

### ⚡ Первый запуск (Test DryRun)

```bash
# Проверка на 1 эпоху (быстро!)
python3 Pipeline/test_dryrun.py
```

✅ Если завершилось успешно — окружение готово!

---

## 📖 Использование

### 🎓 Обучение Deep Learning модели

```bash
python3 Pipeline/train.py
```

**Выходные файлы:**
```
Train/
├── checkpoints/<exp>/
│   └── best_model.pt                    # Лучший чекпойнт
└── results/<exp>/
    ├── metrics.json                     # Финальные метрики
    ├── history.json                     # История обучения
    ├── config_run.json                  # Конфигурация запуска
    └── val_preds.npz                    # Предсказания валидации
```

### 📊 Полная 5×5 оценка

```bash
# Полный cross-validation (5 фолдов × 5 субъектов)
python3 Pipeline/run_full_evaluation.py
```

Выведет:
- Метрики по каждому фолду и субъекту
- Статистику с доверительными интервалами
- Confusion matrices

### 🤖 Классический ML baseline

```bash
# Обучение моделей SVM/RF/LogReg на инженерных признаках
python3 Pipeline/train_classical_ml.py
```

### 📈 Сохранение attention карт

```bash
# Визуализация внимания трансформера
python3 Pipeline/train.py --save-attn
```

---

## ⚙️ Конфигурация

**Основной файл:** `Pipeline/config.py` → функция `default_config()`

<details>
<summary><b>📂 Параметры данных</b></summary>

| Параметр | Значение | Описание |
|----------|----------|---------|
| `data_dir` | `/mnt/data/derivatives/preprocessed_pkl` | Путь к предобработанным данным |
| `subject_ids` | `[1-5]` | ID субъектов для обучения |
| `task` | `'imagine'` | Тип задачи (воображаемая речь) |
| `normalize` | `'zscore_hybrid'` | Subject-centering + global std |
| `exclude_channels` | `[124]` | Исключить артефактные каналы |

</details>

<details>
<summary><b>🧠 Параметры модели</b></summary>

| Параметр | Значение | Описание |
|----------|----------|---------|
| `proj_channels` | 24 | Количество выходных каналов после projection |
| **Малое окно** | 128 / 96 | Размер окна / Шаг (ms) |
| **Большое окно** | 256 / 128 | Размер окна / Шаг (ms) |
| `d_model` | 128 | Размер embedding'а |
| `n_layers` | 2 | Количество слоев Transformer |
| `n_heads` | 4 | Количество attention heads |
| `cov_type` | `'corr'` | Тип ковариационной матрицы |
| `use_subject_embed` | `True` | Использовать subject embeddings |
| `subject_embed_dim` | 16 | Размер subject embedding |

</details>

<details>
<summary><b>🎯 Параметры обучения</b></summary>

| Параметр | GPU | CPU | Описание |
|----------|-----|-----|---------|
| `batch_size` | 16 | 8 | Размер батча |
| `lr` | 3e-4 | 3e-4 | Learning rate (AdamW) |
| `weight_decay` | 1e-4 | 1e-4 | L2 регуляризация |
| `early_stopping_patience` | 8 | 8 | Эпох без улучшения |
| `use_amp` | ✅ | ❌ | Automatic Mixed Precision |
| `grad_clip` | 1.0 | 1.0 | Gradient clipping |
| `num_workers` | 4 | 0 | Data loader workers |

</details>

<details>
<summary><b>🔥 Параметры loss & optimizer</b></summary>

| Параметр | Значение | Описание |
|----------|----------|---------|
| **Loss** | Focal (β=0.999, γ=1.75) | Class-Balanced Focal Loss |
| **Optimizer** | AdamW | С отдельным weight decay для embeddings |
| **Scheduler** | Cosine + Warmup | Линейный warmup → косинусный decay |

</details>

---

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                    EEG Input [B, 125, T]                    │
│                                                             │
│                    ↓ Channel Projection ↓                   │
│                        → 24 channels                        │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
     ┌──▼──┐         ┌──▼──┐        ┌───▼────┐
     │Small│         │Large│        │Scale   │
     │128/96│        │256/128│      │Embedding│
     └──┬──┘         └──┬──┘        └───┬────┘
        │                │                │
    ┌───▼────────────────▼────────────────▼──┐
    │  SPD Covariance (OAS) → Correlation    │
    │  ↓ Riemannian Log-Euclidean Map        │
    │  ↓ Vectorization                       │
    │          → Tokens                      │
    └───┬─────────────────────────────────────┘
        │
    ┌───▼──────────────────────────────────────────┐
    │  + Positional Encoding + Scale Embeddings   │
    │  ↓ Transformer Encoder (2 layers, 4 heads)  │
    │  ↓ Attention-Pooling + CLS Token            │
    │          → Global Representation            │
    └───┬──────────────────────────────────────────┘
        │
    ┌───▼────────────────────────────────────┐
    │  + Subject Embedding (dim=16)          │
    │  ↓ Classification Head                 │
    │          → 8 Meta-Classes              │
    └────────────────────────────────────────┘
```

### 🔑 Ключевые компоненты

| Компонент | Функция | Особенность |
|-----------|---------|-------------|
| **SPD Tokens** | Представление сигнала | Риманова геометрия, устойчивость |
| **Multi-Scale** | Многоуровневый анализ | 128/96 + 256/128 ms окна |
| **Transformer** | Контекстное обучение | 2 слоя × 4 heads |
| **Attention Pool** | Агрегация токенов | Адаптивные веса |
| **Subject Embed** | Персонализация | Отдельный weight decay |

---

## 📊 Бенчмарки

### Результаты на базовой конфигурации

```
┌─────────────────────────────────────────────────────────┐
│         Полная серия: 5 субъектов × 5 фолдов            │
│                (25 успешных запусков)                   │
├─────────────────────────────────────────────────────────┤
│ Метрика              │ Значение      │ Доверие (95%)    │
├──────────────────────┼───────────────┼──────────────────┤
│ F1-macro             │ 0.265 ± 0.011 │ [0.261; 0.269]   │
│ Accuracy             │ 0.288 ± 0.013 │ [0.283; 0.293]   │
│ Balanced Accuracy    │ 0.285 ± 0.014 │ [0.280; 0.290]   │
│ Loss (val)           │ 1.376 ± 0.049 │ [1.371; 1.381]   │
└──────────────────────┴───────────────┴──────────────────┘
```

### Сравнение с baseline

| Модель | Accuracy | Примечание |
|--------|----------|-----------|
| 🎲 **Random Chance** | **0.125** | 8 классов (1/8) |
| 🎯 **Наша модель** | **0.288** | ↑ 2.3x выше random |

### Per-Subject (Fold-1)

| Субъект | F1-macro | Accuracy | BA |
|---------|----------|----------|-----|
| sub-01 | 0.278 ± 0.005 | 0.303 ± 0.007 | 0.299 |
| sub-02 | 0.279 ± 0.008 | 0.298 ± 0.010 | 0.294 |
| sub-03 | 0.265 ± 0.010 | 0.290 ± 0.013 | 0.285 |
| sub-04 | 0.263 ± 0.015 | 0.293 ± 0.018 | 0.288 |
| **Avg** | **0.276 ± 0.009** | **0.299 ± 0.012** | **0.291** |

---

## 📂 Структура проекта

```
EEG_to_Text/
│
├── 📜 README.md                          # Этот файл
├── 📄 COPYRIGHT.md                       # Лицензия
├── 📋 requirements.txt                   # Зависимости
│
├── Pipeline/                             # ⭐ ОСНОВНОЙ КОД
│   ├── config.py                         # Конфигурация
│   ├── data_loader.py                    # Загрузчик данных
│   ├── model.py                          # RTTMultiScale модель
│   ├── riemannian_utils.py               # SPD операции
│   ├── trainer.py                        # Логика обучения
│   ├── train.py                          # Основной скрипт обучения
│   ├── run_full_evaluation.py            # 5×5 cross-validation
│   ├── feature_engineering.py            # Инженерные признаки
│   ├── train_classical_ml.py             # ML baseline
│   └── test_dryrun.py                    # Быстрая проверка
│
├── json/                                 # 📖 СЛОВАРИ
│   ├── classnumber.json                  # 39 классов
│   ├── textmaps.json                     # Описания
│   └── metaclasses.json                  # 39→8 мэппинг
│
├── Train/                                # 📊 РЕЗУЛЬТАТЫ
│   ├── checkpoints/
│   │   └── <exp_id>/
│   │       └── best_model.pt
│   └── results/
│       └── <exp_id>/
│           ├── metrics.json
│           ├── history.json
│           ├── config_run.json
│           └── val_preds.npz
│
├── analysis_tools/                       # 🔬 АНАЛИЗ
│   ├── confusion_matrix_plotter.py
│   ├── pr_roc_analyzer.py
│   ├── training_curves.py
│   ├── ablation_study.py
│   └── subject_effects_analysis.py
│
└── derivatives/                          # 💾 ЛОКАЛЬНЫЕ ДАННЫЕ
    └── preprocessed_pkl/                 # (опционально)
        └── <subject>/eeg/
```

---

## 🗺️ Дорожная карта

### Phase 1
- ✅ Многомасштабная Transformer архитектура
- ✅ Subject embeddings + гибридная нормализация
- ✅ Полная 5×5 оценка с доверительными интервалами
- ✅ Сохранение артефактов и confusion matrices

### Phase 2 (Планируется)
- ⬜ Subject-aware переносимость (LOSO / StratifiedGroupKFold)
- ⬜ Температурная калибровка (ECE/NLL/Brier score)
- ⬜ Системные абляции (subject_embed, cov_type, окна)
- ⬜ Визуализация attention карт
- ⬜ Анализ важности каналов (gradients, SHAP)

### Phase 3 (Дальний горизонт)
- ⬜ Online обучение и адаптация к новым субъектам
- ⬜ Ансамблевые методы
- ⬜ Дистилляция и оптимизация для deployment

---

## 🤝 Вклад

Мы приветствуем вклад в проект! 

### Как помочь:
1. **Баг-фиксы:** Откройте Issue с описанием
2. **Улучшения:** Fork → Branch → Commit → PR
3. **Документация:** Уточнения и примеры всегда приветствуются
4. **Абляции:** Результаты новых конфигураций интересны

### Требования к PR:
- ✅ Следуйте стилю проекта (PEP 8)
- ✅ Добавьте минимальные тесты
- ✅ Обновите документацию
- ✅ Опишите изменения в описании PR

---

## 📜 Лицензия

Этот проект распределяется под **собственной лицензией**. См. [`COPYRIGHT.md`](./COPYRIGHT.md) для полных условий использования.

> ⚠️ **Ограничение:** Коммерческое использование требует письменного согласия автора.

---

## 📚 Цитирование

Если вы используете этот проект в исследованиях, пожалуйста, цитируйте:

```bibtex
@software{Verbetskiy_EEG_to_Text_Phase4B_2026,
  author      = {Вербецкий, Эдуард Игоревич},
  title       = {EEG\_to\_Text: Imagined-Speech EEG Decoder (Phase 4B)},
  year        = {2026},
  institution = {МАИ (НИУ)},
  url         = {https://github.com/<org>/EEG_to_Text}
}
```

---

## 📞 Контакты

- 👨‍💼 **Автор:** Эдуард Вербецкий
- 📧 **Email:** [verbasik@gmail.com](mailto:verbasik@gmail.com)
- 🏛️ **Учреждение:** МАИ (НИУ)
- 📍 **Проект:** Imagined Speech Decoding

---

<div align="center">

### ⭐ Если проект вам нравится, поставьте звезду на GitHub!

**Made with ❤️ for Brain-Computer Interface research**

*Последнее обновление: Март 2026*

</div>