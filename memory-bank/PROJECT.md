---
title: "PROJECT — идентичность и границы проекта"
purpose: "Определить название, миссию, scope и словарь домена"
entrypoint: "AGENTS.md -> INDEX.md -> PROJECT.md"
authority: "controlled"
status: "active"
reads: []
writes: []
depends_on: []
provides:
  - "project_identity"
  - "domain_vocabulary"
  - "scope_boundaries"
canonical_owner: "Все вопросы о том, что делает проект и для кого"
last_verified: "2026-03-14"
max_lines: 80
---

# PROJECT

Канонический источник истины об идентичности проекта.
Используется при первом знакомстве с проектом, вопросах scope и доменной терминологии.

## Contract

- when: первый запуск в проекте; вопросы scope; уточнение доменных терминов; определение границ задачи
- prereq: агент ещё не знаком с проектом или задача требует понимания контекста
- reads: только этот файл
- writes: none
- success: понятны цель, аудитория, границы и ключевые термины проекта
- on_fail: если scope неясен -> открыть `OPEN_QUESTIONS.md`; если архитектура нужна -> открыть `ARCHITECTURE.md`

## Canonical scope

- contains: название, тип, стадия, миссия, границы в scope / вне scope, словарь домена, стейкхолдеры
- excludes:
  - устройство системы -> `ARCHITECTURE.md`
  - инженерные соглашения -> `CONVENTIONS.md`
  - архитектурные решения -> `DECISIONS.md`
  - текущее состояние работы -> `.local/CURRENT.md`

## Identity

- **Название**: EEG_to_Text (репозиторий); содержит исследование на основе датасета Chisco
- **Тип**: исследовательский ML-пайплайн (Python, PyTorch)
- **Стадия**: beta (есть dry-run тест, нет CI)
- **Репозиторий**: git@github.com:Verbasik/EEG_to_Text.git
- Верифицировано: 2026-03-14

## Mission

PhD-исследование по классификации EEG-сигналов: расшифровка воображаемой речи по сигналам электроэнцефалограммы.
Проект реализует пайплайн от предобработки сырых EEG-данных до классификации с помощью Riemannian Transformer (RTTMultiScale).
Цель — научиться различать 8 мета-классов воображаемых слов/звуков на основе ковариационных признаков EEG.

## Boundaries

- **В scope**:
  - Загрузка и предобработка EEG-данных (pkl-формат, датасет Chisco)
  - Извлечение Riemannian признаков (SPD матрицы, Log-Euclidean метрика)
  - Обучение и валидация RTTMultiScale с кросс-валидацией
  - Классификация 8 мета-классов воображаемой речи
  - Subject embeddings для обобщаемости между субъектами
  - Экспериментальные скрипты (Phase 4B и выше)
- **Вне scope**:
  - BCI (Brain-Computer Interface) в реальном времени
  - Сбор и аппаратная предобработка EEG-данных
  - Продуктовая инфраструктура (API, деплой, веб-интерфейс)

## Domain vocabulary

| Термин | Значение в контексте проекта |
|--------|------------------------------|
| EEG | Электроэнцефалограмма; многоканальный временной сигнал мозговой активности |
| SPD | Symmetric Positive Definite — симметричная положительно определённая матрица ковариации |
| Riemannian features | Признаки, извлекаемые через геометрию Риманова пространства SPD матриц |
| OAS shrinkage | Oracle Approximating Shrinkage — регуляризация ковариационной матрицы |
| RTTMultiScale | Многомасштабный Riemannian Transformer — основная архитектура модели |
| Chisco | Название датасета EEG (imagined speech); субъекты обозначаются sub-XX |
| subject embedding | Векторное представление идентичности субъекта для персонализации модели |
| мета-класс | Агрегированная категория воображаемой речи (8 классов) |
| Log-Euclidean | Метрика на пространстве SPD матриц; логарифмическое отображение |
| Phase 4B | Текущая экспериментальная фаза; конфигурация multi-subject |

## Stakeholders

- Владелец продукта: Edward (PhD-исследователь, автор репозитория)
- Основные пользователи: автор; научный руководитель; потенциально — читатели публикации

## Phase 4B Results (2026-03-14)

**Experiment:** Full K-fold cross-validation (5 subjects × 5 folds = 25 experiments)

| Metric | Mean ± Std | 95% CI | Range |
|--------|-----------|--------|-------|
| F1 Macro | 0.265 ± 0.011 | [0.261, 0.269] | [0.242, 0.285] |
| Accuracy | 0.288 ± 0.013 | [0.283, 0.293] | [0.258, 0.311] |
| Balanced Acc | 0.285 ± 0.014 | [0.279, 0.290] | [0.252, 0.317] |
| Precision | 0.275 ± 0.018 | [0.269, 0.282] | [0.243, 0.322] |
| Recall | 0.285 ± 0.014 | [0.279, 0.290] | [0.252, 0.317] |

**Interpretation:** Model is 2.3× better than random (12.5%) but indicates need for hyperparameter tuning. Low inter-fold variance (std 0.006–0.011 within subjects) shows stable but sub-optimal performance.

**Visualizations:** 11 PNG plots + 3 CSV/Markdown tables auto-generated in `Train/results/full_evaluation/`.

**Verification:** Верифицировано 2026-03-14.

## Failure routes

- Если scope задачи выходит за границы проекта -> уточнить у пользователя или зафиксировать в `OPEN_QUESTIONS.md`
- Если термин не определён в словаре -> не предполагать значение; добавить вопрос в `OPEN_QUESTIONS.md`
