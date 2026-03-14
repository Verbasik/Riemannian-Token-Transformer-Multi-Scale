# Codex CLI — подключение памяти

Инструкция по подключению memory bank в Codex CLI (консольный агент).

## Обзор

- Канон: `AGENTS.md` (нативно читается Codex CLI).
- Маршрутизация: `memory-bank/INDEX.md`.
- Skills: `.codex/skills/*/SKILL.md` (update-memory, memory-audit, memory-consolidate, memory-gc, memory-explorer).
- Конфиг проекта: `.codex/config.toml` (sandbox, fallbacks).

## Требования

- Установлен Codex CLI и разрешён доступ к проектным настройкам (`codex --trust-project`).
- Репозиторий с `AGENTS.md` и `memory-bank/`.

## Настройка

1) Запустите в корне: `codex --trust-project` — активирует `.codex/config.toml`.
2) Проверьте sandbox: в `config.toml` задано `sandbox = "workspace-write"`.
3) Fallback (на случай специфики конфигураций): `project_doc_fallback_filenames = ["CLAUDE.md"]` — пригодится, если Codex ищет альтернативный вход.
4) Убедитесь, что `.gitignore` исключает `memory-bank/.local/`.

## Как агент читает память

1) Агент открывает `AGENTS.md`.
2) Ссылается на `memory-bank/INDEX.md` и загружает только релевантные файлы.

## Skills (операции с памятью)

Откройте и выполните шаги из SKILL-файлов:

- Обновление: `.codex/skills/update-memory/SKILL.md`
- Аудит: `.codex/skills/memory-audit/SKILL.md`
- Консолидация: `.codex/skills/memory-consolidate/SKILL.md`
- Очистка: `.codex/skills/memory-gc/SKILL.md`
- Обзор памяти: `.codex/skills/memory-explorer/SKILL.md`

При необходимости оберните эти процедуры в ваши shell-команды/скрипты.

## Верификация установки

- Команда поиска: `rg -n "Верифицировано:" memory-bank` — убедитесь, что даты расставлены.
- Быстрая проверка секретов: см. раздел "Сканирование безопасности" в `.codex/skills/memory-audit/SKILL.md`.

## Триггеры обслуживания памяти

- После завершения задачи — `update-memory`.
- Раз в неделю — `memory-consolidate` и `memory-audit`.
- Раз в месяц — `memory-gc`.

## Безопасность

- Режимы sandbox в Codex ограничивают запись вне рабочей папки.
- Следуйте правилам репозитория: не хранить секреты, вместо значений использовать имена переменных окружения.

## Troubleshooting

- Агент загружает слишком много: сверяйтесь с `INDEX.md`, избегайте полного чтения `memory-bank/`.
- Навигация не работает: убедитесь, что `AGENTS.md` в корне, а `.codex/config.toml` активирован через `--trust-project`.

