---
title: "OPEN_QUESTIONS — нерешённые вопросы и пробелы"
purpose: "Фиксировать неопределённости, чтобы не принимать решений на основе неверифицированных предположений"
entrypoint: "AGENTS.md -> INDEX.md -> OPEN_QUESTIONS.md"
authority: "free"
status: "active"
reads: []
writes: []
depends_on: []
provides:
  - "open_questions_registry"
canonical_owner: "Все нерешённые вопросы, пробелы в знаниях и неоднозначности"
last_verified: "2026-03-14"
max_lines: 60
---

# OPEN_QUESTIONS

Реестр нерешённых вопросов проекта.
Вопрос живёт здесь до тех пор, пока не получен ответ — затем переносится в канонический файл.

## Contract

- when: планирование; обнаружена неопределённость; audit выявил пробел; нет подтверждённого факта
- prereq: агент не может ответить достоверно без внешней информации
- reads: только этот файл
- writes: новая строка при обнаружении пробела; статус `→ Решён` при закрытии
- success: вопрос зафиксирован с контекстом; не принято решение на основе предположения
- on_fail: если вопросов накопилось много -> запустить `memory-clarify`

## Registry

| # | Вопрос | Контекст | Поднят | Владелец | Статус |
|---|--------|----------|--------|----------|--------|
| 1 | Используется ли библиотека `transformers` (HuggingFace) непосредственно в коде Pipeline? | memory-bootstrap: есть в requirements.txt, но в исследованных файлах не импортируется | 2026-03-14 | — | Открыт |
| 2 | Есть ли линтер или форматтер (black, ruff, flake8)? Конфиги не обнаружены | memory-bootstrap: нет .prettierrc, pyproject.toml [tool.ruff], .flake8 | 2026-03-14 | — | Открыт |
| 3 | Какой git workflow принят? Есть ли конвенция коммитов? | memory-bootstrap: только ветка main; коммиты в свободном стиле | 2026-03-14 | — | Открыт |
| 4 | Используются ли mne, pyprep, mne-icalabel для preprocessing внутри Pipeline? | memory-bootstrap: есть в requirements, но preprocessing файлы не найдены в Pipeline/ | 2026-03-14 | — | Открыт |
| 5 | Каковы целевые метрики (accuracy, F1) для успешного эксперимента? | memory-bootstrap: 8 классов, но baseline или target не задокументированы | 2026-03-14 | — | Открыт |
| 6 | Настроен ли CI (GitHub Actions или иной)? | memory-bootstrap: нет .github/workflows/; флаг Status=InProgress в README | 2026-03-14 | — | Открыт |
| 7 | Где располагается preprocessing pipeline (mne-based)? Отдельный репо или директория? | memory-bootstrap: упоминается в README как "EEG preprocessing pipeline" в requirements | 2026-03-14 | — | Открыт |

## Resolution policy

- Решён -> перенести ответ в канонический файл-владелец.
- Пометить строку: `→ Решён: см. [файл § секция]`.
- Нерешённые вопросы старше 30 дней -> эскалировать или закрыть.
