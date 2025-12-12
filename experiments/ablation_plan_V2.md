# Ablation Plan V2 — Phase 4B (EEG_to_Text)

Цели
- Основная: устойчивый прирост macro‑F1 ≥ +0.33 п.п. относительно A2 (mean по 5‑fold DS1), подтверждённый статистически (paired t‑test, p<0.05).
- Дополнительные: не ухудшить balanced_accuracy >0.3 п.п.; сохранить/снизить val loss; обеспечить перенос на DS2 (subject‑aware CV).

Бейзлайн и контекст
- A2 (лучший на DS1/Fold1): CB‑Focal (γ=1.75, β=0.999) + Cosine(T=20,warmup=3), stride_small=96, subject embeddings включены.
- A6 (регуляция эмбеддингов) даёт +0.0007 п.п. к A2 на Fold1 — статистически незначимо без 5‑fold.
- A8 (SPD‑jitter) снижает loss/растит acc, но macro‑F1 ниже A2 (лучший: std=0.03, p=0.2).

Протокол оценки
- DS1: 5‑fold cross‑validation (фикс. seed=42). Отчёт: mean±std по (macro‑F1, balanced_acc, accuracy, loss).
- DS2: subject‑aware CV (StratifiedGroupKFold(group=subject)) или LOSO; отчёт по тем же метрикам.
- Статистика: парный t‑test по фолдам против A2 (mean DS1). Порог значимости: p<0.05.

Серия B — План экспериментов (приоритет)

1) B0 — Контрольная точка A2/A6
- Задача: 5‑fold DS1 для текущего «лучшего» набора (A6 на базе A2/A3).
- Результат: зафиксировать mean/std; использовать для последующих сравнений.

2) B1 — Потери, согласованные с дисбалансом/метрикой
- Гипотеза: логит‑коррекция по частотам улучшит macro‑F1 на дисбалансных классах vs CB‑Focal.
- Кандидаты:
  - Logit‑Adjusted Cross‑Entropy (LA‑CE): z'_k = z_k + log(π_k), где π_k — частота класса на train‑фолде.
  - Balanced Softmax (BSCE): p_k ∝ exp(z_k)·π_k.
  - LDAM‑DRW (margins per class): увеличивает margin редких классов.
- Сетка (DS1, 5‑fold):
  - LA‑CE: λ∈{1.0} (логит‑сдвиг без микса) vs CB‑Focal (γ=1.75, β=0.999).
  - BSCE: сравнить 1:1 с CB‑Focal.
  - LDAM‑DRW: margin scale s∈{16,32}; DRW включён после 10 эпох.

3) B2 — Оценка ковариации (SPD) и shrinkage
- Гипотеза: сохранение тонких корреляций повысит дискриминацию.
- Варианты:
  - OAS с ослабленным clamp: α∈[0.01,1.0] (min=0.01 вместо 0.1).
  - Ledoit‑Wolf (LW) вместо/вдобавок OAS.
- Сетка (DS1, 5‑fold): OAS(minα=0.01) vs OAS(minα=0.1, база) vs LW.

4) B3 — Регуляризация токенов (ёмкость vs число токенов)
- Гипотеза: DropToken до энкодера снижает переобучение на избыточных токенах лучше, чем gating/heads.
- Изменение: случайный дроп p_tok ∈ {0.1, 0.2} для токенов (кроме CLS), одинаково на обеих шкалах.
- Сетка (DS1, 5‑fold): p_tok ∈ {0.1, 0.2}; оставить stride=96, d_model=128.

5) B4 — SPD‑аугментации с аннейлингом
- Гипотеза: поэпохная калибровка вероятности шума даёт прирост macro‑F1 vs фиксированный jitter.
- Варианты:
  - Anneal p: 0→0.2 (эпохи 1–10), затем 0.2→0.1 (эпохи 11–50) при std=0.02/0.03.
  - Off‑diag jitter: шум только во внедиагональных элементах logC.
- Сетка (DS1, 5‑fold): std∈{0.02,0.03} × схемы p(·) ∈ {anneal_updown, const0.2}, тип ∈ {full, offdiag}.

6) B5 — EMA параметров (Polyak averaging)
- Гипотеза: EMA снижает вариативность и повышает макро‑F1 на малых данных.
- Сетка (DS1, 5‑fold): α ∈ {0.99, 0.995}; валидировать EMA‑снимок vs обычный.

Критерии успеха
- Δmacro‑F1 ≥ +0.33 п.п. (абс.) на DS1 (mean 5‑fold) против A2 (контроль), p<0.05.
- balanced_accuracy не ниже контроля >0.3 п.п.; val loss не выше контроля >0.03 (мягкий порог).
- Репликация на DS2 (subject‑aware CV): сохраняется положительная дельта (допускается меньшее значение из‑за shift).

Логирование и артефакты
- Путь: `Train/results/ablations/B{idx}_<ключи>/metrics.json`, рядом `config.json`, чекпойнт — `Train/checkpoints/ablations/B{idx}_<ключи>/best_model.pt`.
- Итоговые сводки: `Train/results/ablations/B{idx}_summary.json` и общий `ablation_summary_V2.json`.
- Доп. отчёты: confusion matrix, per‑class F1, (опц.) ECE для калибровки.

Риски и меры
- Несогласованная метрика: при ухудшении macro‑F1 и улучшении loss/acc — возвращаться к loss, согласованным с дисбалансом (B1).
- Избыточный шум/регуляризация: мониторить recall миноритарных классов; снижать p/std или p_tok.
- Переносимость: проверять DS2 subject‑aware; избегать решений, завязанных на «под субъекта».

Реализация — чек‑лист (код)
- [ ] B1: добавить LA‑CE/BSCE/LDAM‑DRW в `trainer.py`/`loss` с переключателем `cfg['loss']['type']`.
- [ ] B2: ослабить clamp OAS (minα=0.01) в `riemannian_utils.py`; добавить Ledoit‑Wolf и флаг `cfg['model']['cov_estimator'] ∈ {oas,lw}`.
- [ ] B3: модуль DropToken в `model.RTTMultiScale` (флаг/вероятность в конфиге).
- [ ] B4: планировщик вероятности SPD‑jitter (по эпохам); off‑diag режим в `TangentSpaceJittering`.
- [ ] B5: EMA в `trainer.train_loop` (хранить EMA‑копию, валидировать обе версии).
- [ ] Скрипты `experiments/run_b{idx}_*.py` для каждой серии, сводки JSON.

Ресурсы и график (оценка)
- Каждая серия на DS1 5‑fold: ~N часов в зависимости от GPU/CPU; запускать батчами по сериям.
- Порядок: B0 → B1 → B2 → B3 → B4 → B5; перенос на DS2 для топ‑2 решений.

