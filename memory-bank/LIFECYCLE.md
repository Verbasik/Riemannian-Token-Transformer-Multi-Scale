---
title: "Lifecycle операций с памятью"
purpose: "Поддерживать память компактной, актуальной и проверяемой"
entrypoint: "AGENTS.md -> INDEX.md -> LIFECYCLE.md -> SKILL.md"
flow: ["/mb.bootstrap", "/mb.update", "/mb.consolidate", "/mb.gc", "/mb.audit", "/mb.clarify"]
aliases:
  /mb.bootstrap: "memory-bootstrap"
  /mb.update: "update-memory"
  /mb.consolidate: "memory-consolidate"
  /mb.gc: "memory-gc"
  /mb.audit: "memory-audit"
  /mb.clarify: "memory-clarify"
invariants:
  - "CURRENT.md <= 80 lines"
  - "HANDOFF.md <= 40 lines"
  - "Стабильные знания пишутся только в канонические файлы"
  - "Конфликты с CONSTITUTION.md не промотируются автоматически"
  - "Секреты, токены, пароли, PII не записываются в память"
failure_routes:
  secret_like: "stop_promotion -> remediate -> /mb.audit"
  constitution_conflict: "human_review -> /mb.clarify"
  drift_detected: "/mb.audit -> /mb.update"
---

# Lifecycle операций с памятью

Подробные шаги каждой команды описаны в соответствующем `SKILL.md`.
Этот файл задаёт порядок, контракты и ветки реакции.

## /mb.bootstrap
- when: первый запуск агента в проекте; memory bank содержит только шаблонные плейсхолдеры; после `memora init`
- prereq: memory bank создан, но не заполнен (PROJECT.md содержит `[Название проекта]`)
- reads: кодовая база, README, package manifest, конфиги
- writes: `PROJECT.md`, `ARCHITECTURE.md`, `CONVENTIONS.md` (если определяемо), `TESTING.md` (если определяемо), `.local/CURRENT.md`, `.local/HANDOFF.md`, `OPEN_QUESTIONS.md`
- success: базовый контекст заполнен; предложены принципы для CONSTITUTION.md; зафиксированы open questions
- on_fail: если проект нераспознаваем -> записать всё что найдено, остальное в OPEN_QUESTIONS.md

## /mb.update
- when: после значимой задачи; перед завершением сессии
- prereq: есть новые факты, решения, изменения состояния
- writes: `.local/CURRENT.md`, `.local/HANDOFF.md`, при необходимости канонические файлы
- success: текущий контекст обновлён; лимиты соблюдены; устойчивые знания промотированы
- on_fail: при conflict или secret-like контенте остановить промоцию

## /mb.consolidate
- when: после нескольких сессий; еженедельно
- prereq: есть непромотированные знания в `SESSIONS/`
- writes: `DECISIONS.md`, `ADR/`, `PATTERNS/`, `AREAS/`, другие канонические файлы
- success: знания перенесены; дубли и дрейф уменьшены
- on_fail: при `CONSTITUTION_CONFLICT` не перезаписывать, отправить на review

## /mb.gc
- when: ежемесячно; или если `SESSIONS/` > 20 файлов
- prereq: `/mb.consolidate` уже выполнен
- writes: архив/очистка `SESSIONS/`
- success: старые сессии архивированы; непромотированные знания не потеряны
- on_fail: если есть необработанные знания, вернуть управление в `consolidate`

## /mb.audit
- when: перед крупными задачами; еженедельно
- prereq: память доступна для проверки
- reads: вся структура memory-bank и связанные артефакты проекта
- success: выявлены дрейф, устаревание, дубли, secret-like записи, пробелы
- on_fail: при критических проблемах остановить промоцию и вызвать `clarify`

## /mb.clarify
- when: если `audit` нашёл пробелы, конфликты или неоднозначность
- prereq: есть конкретные точки неопределённости
- writes: список целевых вопросов или open issues
- success: сформированы вопросы, без которых нельзя надёжно продолжать
- on_fail: если вопросов нет, вернуть управление в `audit`