---
title: "ARCHITECTURE — архитектура системы"
purpose: "Фиксировать верхнеуровневую структуру системы, границы модулей, потоки данных и внешние зависимости"
entrypoint: "AGENTS.md -> INDEX.md -> ARCHITECTURE.md"
authority: "controlled"
scope: "Архитектурные решения, карта компонентов, публичные границы, интеграции"
canonical_owner: "Все вопросы об устройстве системы и связях между модулями"
reads: []
writes: []
depends_on:
  - "PROJECT.md"
  - "DECISIONS.md"
provides:
  - "component_map"
  - "module_boundaries"
  - "data_flows"
  - "integration_points"
last_verified: "2026-03-14"
max_lines: 120
---

# ARCHITECTURE

Канонический источник истины об устройстве системы.
Используется для архитектурных задач, изменения границ модулей и добавления новых сервисов.

## Contract
- when: архитектурные задачи; изменение модулей; добавление сервисов; анализ потоков данных; интеграции
- prereq: понятна цель задачи и её связь с системой
- reads: при необходимости `PROJECT.md`, `DECISIONS.md`, `ADR/`, `AREAS/<имя>.md`
- writes: none
- success: определены релевантные компоненты, их связи, границы и точки изменения
- on_fail: если структуры недостаточно -> открыть `AREAS/<имя>.md`; если причина решения неясна -> открыть `DECISIONS.md` и `ADR/`

## System overview

Пайплайн классификации EEG-сигналов воображаемой речи на основе датасета Chisco.
Сырые EEG-данные загружаются из pkl-файлов, проходят нормализацию и разбиение на фолды кросс-валидации.
Модель RTTMultiScale извлекает многомасштабные Riemannian признаки из скользящих окон (SPD ковариации + Log-Euclidean mapping) и классифицирует их с помощью Transformer-энкодера.
Subject embeddings позволяют модели адаптироваться к конкретному субъекту.

## Component map

```text
Pipeline/
├── config.py           — централизованная конфигурация (пути, гиперпараметры)
│       │
├── data_loader.py      — загрузка pkl, создание ChiscoDataset, кросс-валидация
│       │
├── feature_engineering.py — предобработка и нормализация (zscore_hybrid и др.)
│       │
├── riemannian_utils.py — SPD операции: cov_shrinkage_oas, spd_logm, spd_vectorize
│       │
├── model.py            — RTTMultiScale (SinusoidalPE + Transformer + AttentionPooling)
│       │
├── trainer.py          — train_epoch / evaluate (forward + backward + метрики)
│       │
├── train.py            — build_loaders / build_model / build_criterion / run
│       │
├── run_full_evaluation.py — оркестрация полного запуска кросс-валидации
│
├── utils.py            — вспомогательные функции (set_seed, pretty_print_run, ...)
└── test_dryrun.py      — dry-run проверка целостности пайплайна (1 epoch)

Данные (внешние):
  /mnt/data/EEG/preprocessed_pkl/  — pkl-файлы субъектов
  json/                             — JSON-описания датасетов
  participants.tsv                  — метаданные субъектов
  montage.csv                       — монтаж электродов EEG

Выход:
  Train/checkpoints/                — чекпоинты модели
  Train/results/                    — результаты экспериментов
```

## Modules

| Module | Responsibility | Public entrypoint | Depends on | Verified |
|--------|----------------|-------------------|------------|----------|
| config.py | Конфигурация путей и гиперпараметров | `default_config()` | torch | 2026-03-14 |
| data_loader.py | Загрузка данных, ChiscoDataset, CV разбиение | `ChiscoDataset`, `create_subject_mapping()` | config, utils, sklearn | 2026-03-14 |
| riemannian_utils.py | SPD операции (OAS, logm, vectorize, window) | `cov_shrinkage_oas`, `spd_logm`, `spd_vectorize`, `window_signal` | torch | 2026-03-14 |
| model.py | RTTMultiScale архитектура | `RTTMultiScale` | riemannian_utils, torch | 2026-03-14 |
| trainer.py | Цикл обучения и оценки | `evaluate()` | torch, sklearn | 2026-03-14 |
| train.py | Оркестрация: сборка и запуск | `build_loaders`, `build_model`, `build_criterion` | все остальные | 2026-03-14 |
| feature_engineering.py | Нормализация EEG | [требует верификации] | numpy | 2026-03-14 |
| utils.py | Утилиты: seed, print, pickle compat | `set_seed`, `pretty_print_run` | — | 2026-03-14 |
| test_dryrun.py | Dry-run интеграционный тест | `main()` | все Pipeline модули | 2026-03-14 |

## Data flows

### Flow: EEG Classification Training

* trigger: запуск `train.py` или `run_full_evaluation.py`
* input: pkl-файлы субъектов из `$EEG_PREPROCESSED_DIR` (или `/mnt/data/EEG/preprocessed_pkl`)
* path: `data_loader.py` (load pkl) → `feature_engineering.py` (normalize) → `StratifiedKFold` → `DataLoader` → `RTTMultiScale.forward()` → `cb_focal` loss → AdamW optimizer
* output: чекпоинты (`.pt`) в `Train/checkpoints/`, метрики в `Train/results/`
* failure_mode: если данные не найдены — `_resolve_preprocessed_dir()` пробует несколько путей; при отсутствии всех — использует первый кандидат без проверки существования

### Flow: Dry-Run Validation

* trigger: `python test_dryrun.py`
* input: та же конфигурация, n_epochs=1
* path: build_loaders → build_model → build_criterion → 1 epoch train → evaluate
* output: вывод метрик в stdout; проверка всех компонентов без сохранения чекпоинтов
* failure_mode: любое исключение прерывает тест

## External dependencies

| Dependency | Purpose | Version | Integration point | Notes |
|------------|---------|---------|-------------------|-------|
| torch | Deep learning framework | >=2.2 | model.py, trainer.py, train.py | CUDA optional |
| numpy | Числовые операции | >=1.26 | data_loader.py, feature_engineering.py | |
| scipy | Научные вычисления | >=1.11 | riemannian_utils.py | |
| scikit-learn | CV, метрики | >=1.3 | data_loader.py, trainer.py | StratifiedGroupKFold |
| transformers | [требует верификации] | >=4.40 | [требует верификации] | В requirements, но использование не верифицировано |
| mne | EEG preprocessing | >=1.6 | [требует верификации] | Возможно в preprocessing pipeline |
| pandas | Метаданные, IO | >=2.0 | participants.tsv | |
| tqdm | Прогресс-бары | >=4.66 | trainer.py | |
| h5py / mat73 | Загрузка EEG данных | >=3.9 / >=0.62 | [требует верификации] | Для raw data |

## Infrastructure

* hosting: локальная машина исследователя (CPU/CUDA)
* database: none — данные в pkl-файлах на диске
* messaging: none
* storage: файловая система: `/mnt/data/EEG/preprocessed_pkl/` (данные); `Train/` (результаты)
* deployment: none (исследовательский скрипт)
* observability: stdout logging; метрики сохраняются в json/csv в `Train/results/`
* device resolution: автоматически через `$EEG_PREPROCESSED_DIR` или поиск по candidate paths

## Module boundaries

### Public APIs

* `config.default_config()`: возвращает Dict с полной конфигурацией эксперимента
* `model.RTTMultiScale`: основная модель; вход `[B, C, T]`, выход logits `[B, n_classes]`
* `train.build_loaders`, `build_model`, `build_criterion`: фабрики компонентов

### Invariants

* Один факт конфигурации → только в `config.py`
* SPD операции (logm, vectorize, OAS) → только в `riemannian_utils.py`
* Путь к данным определяется через `_resolve_preprocessed_dir()` или `$EEG_PREPROCESSED_DIR`
* seed фиксируется через `utils.set_seed()` перед любым обучением

## Change policy

* Новые модули добавляются только с явной ответственностью и точкой входа.
* Если меняется публичная граница, обнови связанные `AREAS/`, `DECISIONS.md` и при необходимости `ADR/`.
* Если модуль нарушает существующие инварианты, изменение требует human review.
* Неподтверждённые архитектурные предположения не фиксируются как канонические.

## Failure routes

* Если модуль не удаётся классифицировать -> уточнить через `PROJECT.md` или `AREAS/<имя>.md`
* Если поток данных не подтверждён -> пометить как open question
* Если архитектурное решение спорное -> зафиксировать в `DECISIONS.md` и вынести в `ADR/`
