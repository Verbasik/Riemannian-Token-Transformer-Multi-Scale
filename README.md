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

| 🚀 Быстрый старт                     | 📚 Полная документация             | 🔧 Конфигурация                  |
|:------------------------------------ |:---------------------------------- |:-------------------------------- |
| [Установка](#-быстрый-старт)         | [Использование](#-использование)   | [Параметры](#-конфигурация)      |
| [Первый запуск](#проверка-1-эпоха)   | [Архитектура](#-архитектура)       | [Структура](#-структура-проекта) |

---

## ✨ Ключевые особенности

<table>
<tr>
<td width="50%">

### 🎯 Точность & Устойчивость
- ✅ **SI F1-macro: 0.253 ± 0.002**
- ✅ **SD F1-macro: 0.266 ± 0.010**
- ✅ **SD Balanced Accuracy: 0.285 ± 0.013**
- ✅ **Выше случайного (~0.125) в 2.3x**

</td>
<td width="50%">

### 🧮 Технологический стек
- 🔷 **Риманова геометрия** (SPD матрицы)
- 🔶 **Мультимасштабные токены** (128/96 + 256/128 samples)
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
├── $EEG_PREPROCESSED_DIR
├── /mnt/data/data/derivatives/preprocessed_pkl/
├── /mnt/data/derivatives/preprocessed_pkl/
├── /mnt/data/EEG/preprocessed_pkl/
├── ./derivatives/preprocessed_pkl/
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

### 🔢 Фактическая форма входа

Текущий loader после исключения одного канала отдаёт батчи формы:

```text
batch: [B, 124, 1651]
sample: [124, 1651]
```

При 500 Гц `T=1651` соответствует примерно 3.3 секунды воображаемой речи.

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

### 📊 Полная SI/SD оценка

```bash
# SI pooled personalized + SD per-subject evaluation
python3 Pipeline/run_full_evaluation.py --pipeline both
```

Выведет:
- Метрики по каждому pipeline, фолду и субъекту
- Bootstrap 95% доверительные интервалы
- Таблицы и графики в `Train/results/full_evaluation/`

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

| Параметр               | Значение                                                                       | Описание                               |
|:-----------------------|:-------------------------------------------------------------------------------|:---------------------------------------|
| `data_dir`             | `$EEG_PREPROCESSED_DIR` или `/mnt/data/data/derivatives/preprocessed_pkl`      | Путь к предобработанным данным         |
| `subject_ids`          | `["sub-01", ..., "sub-05"]`                                                    | ID субъектов для обучения              |
| `task`                 | `'imagine'`                                                                    | Тип задачи (воображаемая речь)         |
| `normalize`            | `'zscore_hybrid'`                                                              | Subject-centering + global std         |
| `exclude_channels`     | `[124]`                                                                        | Исключить артефактные каналы           |

</details>

<details>
<summary><b>🧠 Параметры модели</b></summary>

| Параметр               | Значение             | Описание                                                            |
|:-----------------------|:---------------------|:--------------------------------------------------------------------|
| n_classes              | 8                    | Количество мета-классов                                             |
| proj_channels          | 24                   | Количество выходных каналов после projection                        |
| **Малое окно**         | 128 / 96             | Размер окна / шаг в samples                                         |
| **Большое окно**       | 256 / 128            | Размер окна / шаг в samples                                         |
| spd_vec_dim            | 300                  | `24 * 25 / 2`, верхний треугольник SPD-матрицы                      |
| feature_proj           | `Linear(300, 128)`   | Проекция SPD-вектора в пространство токенов                         |
| d_model                | 128                  | Размер embedding'а                                                  |
| n_layers               | 2                    | Количество слоев Transformer                                        |
| n_heads                | 4                    | Количество attention heads                                          |
| attn_heads             | 1                    | Количество голов attention pooling                                  |
| cov_type               | `'corr'`             | Тип ковариационной матрицы                                          |
| use_subject_embed      | `True`               | Использовать subject embeddings                                     |
| subject_embed_dim      | 16                   | Размер subject embedding                                            |
| subject_embed_dropout  | 0.2                  | Dropout для subject embedding                                       |
| SI classifier input    | 272                  | `CLS(128) + attention pooled(128) + subject embedding(16)`          |
| SD classifier input    | 256                  | `CLS(128) + attention pooled(128)`                                  |

</details>

<details>
<summary><b>🧪 Параметры оценки</b></summary>

| Параметр                           | Значение             | Описание                                                                |
|:-----------------------------------|:---------------------|:------------------------------------------------------------------------|
| `evaluation.pipeline`              | `'both'`               | Запустить SI, затем SD                                                |
| `evaluation.si_use_subject_embed`  | `True`                 | SI использует subject embeddings                                      |
| `evaluation.sd_use_subject_embed`  | `False`                | SD обучает отдельную модель на субъекта                               |
| `cv.protocol`                      | `'within_subject'`     | Каждый субъект есть и в train, и в validation                         |
| `cv.mode`                          | `'within_subject'`     | Фолды строятся внутри каждого субъекта                                |
| `model.unknown_subject_policy`     | `'auto'`               | `error` для within-subject, `zero` для subject-held-out               |

Текущий SI baseline оценивает known-subject generalization. Он не является
строгой проверкой переноса на полностью нового субъекта; для этого нужен
отдельный `subject_heldout`/LOSO запуск.

</details>

<details>
<summary><b>🎯 Параметры обучения</b></summary>

| Параметр                     | GPU  | CPU  | Описание                                         |
|:-----------------------------|:-----|:-----|:-------------------------------------------------|
| `batch_size`                 | 16   | 8    | Размер батча                                     |
| `lr`                         | 3e-4 | 3e-4 | Learning rate (AdamW)                            |
| `weight_decay`               | 1e-4 | 1e-4 | L2 регуляризация                                 |
| `early_stopping_patience`    | 8    | 8    | Эпох без улучшения                               |
| `use_amp`                    | ✅   | ❌   | Automatic Mixed Precision                        |
| `grad_clip`                  | 1.0  | 1.0  | Gradient clipping                                |
| `num_workers`                | 0    | 0    | DataLoader multiprocessing отключён по умолчанию |

</details>

<details>
<summary><b>🔥 Параметры loss & optimizer</b></summary>

| Параметр | Значение | Описание |
|----------|----------|---------|
| **Loss** | CB-Focal (β=0.999, γ=1.75) | Class-Balanced Focal Loss |
| **Optimizer** | AdamW | С отдельным weight decay для embeddings |
| **Scheduler** | Cosine + Warmup | Линейный warmup → косинусный decay |

</details>

---

## 🏗️ Архитектура

![Архитектура](/EEG_TO_TEXT/assets/main.png)

### 🔑 Ключевые компоненты

| Компонент          | Функция                | Особенность                      |
|--------------------|-----------------------|-----------------------------------|
| **SPD Tokens**     | Представление сигнала | Риманова геометрия, устойчивость  |
| **Multi-Scale**    | Многоуровневый анализ | 128/96 + 256/128 samples          |
| **Transformer**    | Контекстное обучение  | 2 слоя × 4 heads                  |
| **Attention Pool** | Агрегация токенов     | Адаптивные веса                   |
| **Subject Embed**  | Персонализация        | Отдельный weight decay            |

---

## 📊 Бенчмарки

### Результаты текущего full evaluation

```
┌─────────────────────────────────────────────────────────┐
│         Полная серия: SI + SD pipelines                 │
│                (30 успешных запусков)                   │
├─────────────────────────────────────────────────────────┤
│ Pipeline / метрика   │ Значение      │ Доверие (95%)    │
├──────────────────────┼───────────────┼──────────────────┤
│ SI F1-macro          │ 0.253 ± 0.002 │ [0.251; 0.255]   │
│ SI Accuracy          │ 0.283 ± 0.003 │ [0.281; 0.285]   │
│ SI Balanced Accuracy │ 0.266 ± 0.003 │ [0.264; 0.269]   │
│ SD F1-macro          │ 0.266 ± 0.010 │ [0.262; 0.271]   │
│ SD Accuracy          │ 0.285 ± 0.013 │ [0.280; 0.290]   │
│ SD Balanced Accuracy │ 0.285 ± 0.013 │ [0.281; 0.290]   │
└──────────────────────┴───────────────┴──────────────────┘
```

> Не используйте mixed overall mean как главный headline metric: общий отчёт
> смешивает SI и SD эксперименты.

### Сравнение со случайным уровнем

| Модель               | Accuracy    | Примечание          |
|----------------------|-------------|---------------------|
| 🎲 **Random Chance** | **0.125**   | 8 классов (1/8)     |
| 🎯 **SI baseline**   | **0.283**   | ↑ 2.26x выше random |
| 🎯 **SD baseline**   | **0.285**   | ↑ 2.28x выше random |

### Per-subject SD F1-macro

| Субъект | F1-macro      | 95% CI         |
|---------|---------------|----------------|
| sub-01 | 0.257 ± 0.009 | [0.249; 0.264] |
| sub-02 | 0.270 ± 0.011 | [0.261; 0.280] |
| sub-03 | 0.270 ± 0.012 | [0.260; 0.280] |
| sub-04 | 0.268 ± 0.006 | [0.262; 0.273] |
| sub-05 | 0.267 ± 0.005 | [0.262; 0.271] |

---

## 📂 Структура проекта

```
EEG_to_Text/
│
├── 📜 README.md                          # Этот файл
├── 📄 LICENSE                            # Лицензия MIT
├── 📋 requirements.txt                   # Зависимости
│
├── Pipeline/                             # ⭐ ОСНОВНОЙ КОД
│   ├── config.py                         # Конфигурация
│   ├── data_loader.py                    # Загрузчик данных
│   ├── model.py                          # RTTMultiScale модель
│   ├── riemannian_utils.py               # SPD операции
│   ├── trainer.py                        # Логика обучения
│   ├── train.py                          # Основной скрипт обучения
│   ├── run_full_evaluation.py            # SI/SD full evaluation
│   ├── feature_engineering.py            # Инженерные признаки
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
└── derivatives/                          # 💾 ЛОКАЛЬНЫЕ ДАННЫЕ
    └── preprocessed_pkl/                 # (опционально)
        └── <subject>/eeg/
```

---

## 🗺️ Дорожная карта

### Phase 1
- ✅ Многомасштабная Transformer архитектура
- ✅ Subject embeddings + гибридная нормализация
- ✅ Полная SI/SD оценка с доверительными интервалами
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

Этот проект распространяется под лицензией **MIT**. См. [`LICENSE`](./LICENSE) для полных условий использования.

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

## Citation

```bibtex
@thesis{verbetskii2026rttmultiscale,
title       = {Riemannian geometric features and transformer for decoding imagined speech from EEG},
author      = {Verbetskii, Eduard Igorevich},
institution = {Moscow Aviation Institute (National Research University)},
location    = {Moscow, Russia},
year        = {2026},
type        = {Master of Science},
note        = {Institute No. 8 `Computer Science and Applied Mathematics''; educational program `Machine Learning and Data Analysis''},
langid      = {russian}
}
```

---

<div align="center">

### ⭐ Если проект вам нравится, поставьте звезду на GitHub!

**Made with ❤️ for Brain-Computer Interface research**

*Последнее обновление: Апрель 2026*

</div>
