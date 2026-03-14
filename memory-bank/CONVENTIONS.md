---
title: "CONVENTIONS — инженерные соглашения проекта"
purpose: "Фиксировать правила написания кода, именования и workflow, не очевидные из конфигов"
entrypoint: "AGENTS.md -> INDEX.md -> CONVENTIONS.md"
authority: "controlled"
status: "active"
reads: []
writes: []
depends_on:
  - "PROJECT.md"
provides:
  - "code_style"
  - "naming_conventions"
  - "git_workflow"
  - "error_handling_patterns"
canonical_owner: "Все вопросы о стиле кода, именовании и workflow"
last_verified: "2026-03-14"
max_lines: 100
---

# CONVENTIONS

Канонический источник истины об инженерных соглашениях проекта.
Содержит только правила, НЕ очевидные из конфигов линтера/форматтера.

## Contract

- when: написание или ревью кода; создание нового модуля; настройка инструментов
- prereq: задача предполагает создание или изменение кода
- reads: только этот файл; при необходимости `ARCHITECTURE.md` для контекста модуля
- writes: none
- success: понятны правила именования, стиль, workflow и неочевидные паттерны проекта
- on_fail: если соглашение отсутствует -> зафиксировать в `OPEN_QUESTIONS.md`; не изобретать самостоятельно

## Canonical scope

- contains: структура файлов, стиль кода, naming-конвенции, git workflow, обработка ошибок
- excludes:
  - тестовая стратегия -> `TESTING.md`
  - архитектурные решения -> `DECISIONS.md`
  - бизнес-цели и scope -> `PROJECT.md`
  - конфиги линтера/форматтера (они самодокументированы)

## File organization

- Исходный код: `Pipeline/` — все модули ML пайплайна
- Данные: `/mnt/data/EEG/preprocessed_pkl/` (внешний том, не в репо)
- Метаданные: `json/`, `participants.tsv`, `montage.csv` — в корне проекта
- Результаты: `Train/checkpoints/`, `Train/results/` — генерируются при запуске
- Именование файлов: snake_case (напр. `data_loader.py`, `riemannian_utils.py`)
- Верифицировано: 2026-03-14

## Code style

- Язык: Python 3.12
- Форматтер: [требует верификации] — конфигов .prettierrc/.black не обнаружено
- Линтер: [требует верификации]
- Кодировка файлов: UTF-8 (явное `# -*- coding: utf-8 -*-` в заголовке каждого файла)
- Верифицировано: 2026-03-14

### Non-obvious rules

- Заголовок каждого файла: `# file: <имя>.py` + `# -*- coding: utf-8 -*-` + docstring модуля
- Docstring функций: структурированный формат с разделами `Description`, `Args`, `Returns`, `Raises`, `Examples`
- Комментарии к конфигурации: каждый параметр в `default_config()` сопровождается inline-комментарием
- Type hints: обязательны для всех публичных функций; используется `typing` (Dict, Optional, Tuple, Union)
- Секции кода разделяются блоками `# === ... ===`; стандартные разделы: Standard Libraries, Third-Party Libraries, Local Imports
- Верифицировано: 2026-03-14

## Naming

| Что | Конвенция | Пример |
|-----|-----------|--------|
| Переменные/функции | snake_case | `build_loaders`, `n_channels`, `data_dir` |
| Классы | PascalCase | `RTTMultiScale`, `ChiscoDataset`, `SinusoidalPE` |
| Константы модуля | UPPER_SNAKE_CASE | `PROJECT_ROOT`, `RANDOM_SEED`, `EPSILON` |
| Параметры конфига | snake_case в dict | `"learning_rate"`, `"n_epochs"`, `"weight_decay"` |
| Приватные функции | `_prefix` | `_resolve_preprocessed_dir`, `_reset_parameters` |
| Верифицировано: 2026-03-14 | | |

## Git workflow

- Ветки: [требует верификации] — единственная ветка `main` обнаружена
- Коммиты: нет явной конвенции; последние коммиты используют свободный стиль
- PR: [требует верификации]
- Верифицировано: 2026-03-14

## Error handling

- Паттерн: исключения Python (ValueError, RuntimeError); нет Result types
- Логирование: print() в stdout; нет структурированного logging
- Файл данных не найден: `_resolve_preprocessed_dir()` возвращает первый кандидат без raise
- Верифицировано: 2026-03-14

## Failure routes

- Если конвенция конфликтует с `CONSTITUTION.md` -> приоритет у конституции; зафиксировать в `DECISIONS.md`
- Если конвенция устарела -> пометить для ревью; добавить в `OPEN_QUESTIONS.md`
- Если нужна новая конвенция -> предложить и зафиксировать в `DECISIONS.md` до применения
