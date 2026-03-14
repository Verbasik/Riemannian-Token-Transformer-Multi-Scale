#!/usr/bin/env bash
set -euo pipefail

echo "Инициализация Memory Bank v3..."

D=$(date +%Y-%m-%d)

# Структура
mkdir -p memory-bank/{ADR,PATTERNS,AREAS,ARCHIVE}
mkdir -p memory-bank/.local/SESSIONS
mkdir -p memory-bank/scripts
mkdir -p .claude/{skills/update-memory,skills/memory-audit,skills/memory-consolidate,skills/memory-gc,agents,rules}
mkdir -p .codex/skills/{update-memory,memory-audit}
mkdir -p .qwen/agents
mkdir -p .opencode/commands

# Helper
mk() { [ -f "$1" ] || printf '%s\n' "$2" > "$1"; }

mk memory-bank/INDEX.md "# INDEX — см. AGENTS.md"
mk memory-bank/CONSTITUTION.md "# КОНСТИТУЦИЯ ПРОЕКТА

> Загружай когда: архитектурные решения, выбор технологий, ревью.
> Макс: 60 строк. Верифицировано: $D.
> УРОВЕНЬ ЗАЩИТЫ: неприкосновенный.

## Назначение

Этот файл содержит ненарушимые принципы проекта.

## Принципы

### I. [Заполни первый принцип]

### II. [Заполни второй принцип]

## Управление

- Конституция имеет приоритет над всеми остальными файлами.
- Изменения требуют одобрения человека."

mk memory-bank/PROJECT.md "# ПРОЕКТ

> Загружай когда: первый раз, scope, домен. Макс: 80 строк. Верифицировано: $D.

## Идентичность

- **Название**:
- **Тип**:
- **Стадия**:"

mk memory-bank/ARCHITECTURE.md "# АРХИТЕКТУРА

> Загружай когда: архитектура, модули, сервисы. Макс: 120 строк. Верифицировано: $D.

## Обзор системы

"

mk memory-bank/CONVENTIONS.md "# КОНВЕНЦИИ

> Загружай когда: код, ревью. Макс: 100 строк. Верифицировано: $D.
"

mk memory-bank/TESTING.md "# ТЕСТИРОВАНИЕ

> Загружай когда: тесты, CI. Макс: 60 строк. Верифицировано: $D.

## Команды

\`\`\`bash

\`\`\`"

mk memory-bank/DECISIONS.md "# РЕШЕНИЯ

> Макс: 80 строк. Верифицировано: $D.

| # | Решение | Статус | Дата | ADR | Влияние |
|---|---------|--------|------|-----|---------|"

mk memory-bank/OPEN_QUESTIONS.md "# ОТКРЫТЫЕ ВОПРОСЫ

> Макс: 60 строк. Верифицировано: $D.

| # | Вопрос | Контекст | Поднят | Владелец | Статус |
|---|--------|----------|--------|----------|--------|"

mk memory-bank/CHANGELOG.md "# CHANGELOG

## [Unreleased]

-"

mk memory-bank/.local/CURRENT.md "# ТЕКУЩЕЕ СОСТОЯНИЕ

> Последнее обновление: $D от init

## Активная цель

[Установи первую цель]

## Статус

- [ ] Заполнить PROJECT.md
- [ ] Заполнить ARCHITECTURE.md
- [ ] Заполнить TESTING.md
- [ ] Установить реальную цель

## Checkpoint compaction

> Timestamp: $D"

mk memory-bank/.local/HANDOFF.md "# ПЕРЕДАЧА КОНТЕКСТА

> Написано: $D от init

## Что сделано

Memory bank инициализирован.

## Что делать дальше

1. Заполнить PROJECT.md
2. Заполнить ARCHITECTURE.md
3. Заполнить TESTING.md
4. Установить цель в CURRENT.md"

# Qwen settings
mk .qwen/settings.json '{"context":{"fileName":["AGENTS.md","QWEN.md"]}}'

# Gitignore
if [ -f .gitignore ]; then
  grep -q "memory-bank/.local/" .gitignore 2>/dev/null || \
    printf '\n# Memory bank — локальные файлы сессий\nmemory-bank/.local/\n.claude/memory/\n.qwen/memory/\n' >> .gitignore
else
  printf '# Memory bank — локальные файлы сессий\nmemory-bank/.local/\n.claude/memory/\n.qwen/memory/\n' > .gitignore
fi

# Claudeignore
[ -f .claudeignore ] || printf '.env\n.env.*\n*.key\n*.pem\n*.p12\n**/secrets/\n**/credentials/\n' > .claudeignore

echo ""
echo "✓ Memory Bank v3 инициализирован."
echo ""
echo "Следующие шаги:"
echo "  1. Скопируй AGENTS.md и CLAUDE.md в корень проекта"
echo "  2. Скопируй skill-файлы в .claude/, .codex/, .qwen/, .opencode/"
echo "  3. Заполни memory-bank/PROJECT.md"
echo "  4. Заполни memory-bank/ARCHITECTURE.md"
echo "  5. Заполни memory-bank/TESTING.md"
echo "  6. git add AGENTS.md CLAUDE.md memory-bank/*.md memory-bank/ADR/"
echo "  7. Скопируй скрипты в memory-bank/scripts/"
