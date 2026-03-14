---
title: "TESTING — стратегия тестирования и валидации"
purpose: "Фиксировать команды запуска, структуру тестов и quality gates"
entrypoint: "AGENTS.md -> INDEX.md -> TESTING.md"
authority: "controlled"
status: "active"
reads: []
writes: []
depends_on:
  - "PROJECT.md"
  - "CONVENTIONS.md"
provides:
  - "test_commands"
  - "quality_gates"
  - "ci_pipeline"
canonical_owner: "Все вопросы о тестировании, CI и проверках перед коммитом"
last_verified: "2026-03-14"
max_lines: 60
---

# TESTING

Канонический источник истины о тестировании и валидации.
Используется при запуске тестов, написании новых и настройке CI.

## Contract

- when: запуск тестов; написание новых тестов; настройка CI; проверка перед коммитом
- prereq: задача затрагивает поведение кода или его валидацию
- reads: только этот файл
- writes: none
- success: понятны команды, расположение тестов и обязательные проверки
- on_fail: если команды устарели -> пометить; добавить в `OPEN_QUESTIONS.md`

## Commands

```bash
# Dry-run тест (интеграционная проверка, 1 epoch):
cd Pipeline && python test_dryrun.py

# Полная кросс-валидация:
cd Pipeline && python run_full_evaluation.py

# Обучение одного запуска:
cd Pipeline && python train.py
```

- Верифицировано: 2026-03-14

## Before commit

Минимально рекомендуется: `python test_dryrun.py` — проверяет загрузку данных, инициализацию модели, один шаг forward/backward и валидацию.
CI не настроен — [требует верификации].

## Test structure

- Dry-run (интеграционный): `Pipeline/test_dryrun.py` — проверяет весь пайплайн за 1 epoch
- Unit-тесты: не обнаружены; нет отдельной директории `tests/`
- E2E: `run_full_evaluation.py` — полный прогон кросс-валидации (не автоматизирован как тест)
- Верифицировано: 2026-03-14

## Coverage

- Цель: [требует верификации] — инструмент coverage не обнаружен
- Покрытие сейчас: только интеграционный dry-run

## Heavy tests

- `run_full_evaluation.py` — полная кросс-валидация (50 epochs × 5 folds); требует данных на `/mnt/data/`
- Когда запускать: вручную при значимых изменениях модели

## CI pipeline

- На PR: не настроен
- На merge в main: не настроен
- Верифицировано: 2026-03-14

## Failure routes

- Если тест не проходит в CI -> не мержить; зафиксировать причину в `CURRENT.md`
- Если тестового фреймворка нет -> добавить вопрос в `OPEN_QUESTIONS.md`
- Если команды устарели -> верифицировать, обновить дату `last_verified`
