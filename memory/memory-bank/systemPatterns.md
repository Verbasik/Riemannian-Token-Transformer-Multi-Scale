# Архитектура и паттерны

## Слои и ответственность
- Конфигурация: `config.default_config()` формирует единый словарь параметров запуска.
- Данные: `data_loader` загружает pkl, формирует датасеты, маппинги и нормстаты.
- Модель: `model.RTTMultiScale` строит токены из SPD-представлений и применяет Transformer.
- Геометрия: `riemannian_utils` — устойчивые SPD-операции (OAS, logm, vectorize, corr).
- Обучение: `trainer` реализует цикл, метрики, ClassBalancedFocalLoss и сохранение артефактов.
- Сборка и запуск: `train.py` — builders компонентов и `main()`.

## Ключевые паттерны
- Builder-функции: `build_loaders`, `build_model`, `build_criterion`, `build_optimizer_and_scheduler` упрощают конфигурирование.
- Гибридная нормализация: subject-wise центрирование + глобальный std, снижает subject shift (Phase 4B-6).
- Subject embeddings: персонализация в классификаторе (конкатенация к pooled признакам).
- Стабильность SPD: eigendecomposition с фолбэками (MPS/FP16→FP32), clamp/jitter для SPD.
- Обучение и контроль: ранняя остановка по `f1_macro`, сохранение лучшего состояния; cosine LR с warmup.
- Валидация: Stratified K-Fold, в коде используется первый фолд (`splits[0]`).

### Балансировка данных (обновление)
- Антипаттерн: «идеальный баланс» через сильный undersampling в мультисубъектном сценарии — приводит к потере информативных примеров и деградации метрик.
- Рекомендуемый паттерн: сохранять весь объём данных и применять re-weighting (Class-Balanced Focal Loss уже используется) и/или `WeightedRandomSampler` для батчей.
- Оценка должна быть subject-aware: использовать `StratifiedGroupKFold` (группа = subject) или LOSO для честной проверки переносимости.

### Инсайты абляций A1–A7
- A2 (CB-Focal): тонкая настройка `gamma=1.75`, `beta=0.999` даёт лучший баланс precision/recall; дефолт.
- A3 (Cosine): короткий период T=20 и небольшой warmup=3 стабильно лучше длинных периодов на DS1.
- A4/A5 (Gating/heads): `gating=False` и `attn_heads=1` показывают лучший f1_macro на DS1; многоголовый pooling улучшает loss, но не f1.
- A6 (Subject embeddings): лёгкая регуляция эмбеддинга (Dropout≈0.2, отдельный L2≈5e-4) помогает; увеличение dim>16 не гарантирует выигрыш по f1.
- A7 (stride_small): уменьшение шага (80/64) ухудшает f1 и accuracy; дефолт — 96.

### A8 — SPD-аугментация (tangent jitter)
- Идея: малый симметричный гауссов шум в касательном пространстве (после `spd_logm`, до `spd_vectorize`) для повышения робастности.
- Реализация: `TangentSpaceJittering(noise_std, prob)`; включается флагом `use_spd_augment=True` в модели.
- Параметры свипа: `spd_jitter_std ∈ {0.02, 0.03}`, `spd_jitter_prob ∈ {0.2, 0.3}`.
- Применение: только в обучении (`model.training=True`), на обеих шкалах токенов.
- Результаты (DS1, Fold 1): лучший сетап `std=0.03`, `prob=0.2` — f1_macro=0.2761, accuracy=0.2978, loss=1.3897. Тренд: `prob=0.2` > `0.3` по f1; при `prob=0.2` умеренный шум (`std=0.03`) > `0.02`.

### Изменения в архитектуре/конфиге
- Добавлен `subject_embed_dropout` в модель; отдельная группа параметров оптимизатора для `subject_embed.*` с собственным weight_decay.
- В загрузчике: поддержка `cv.fold_index`; опциональный WeightedRandomSampler отключается при `loss=cb_focal`.
- В модельный билдер добавлена прокладка параметров A8: `use_spd_augment`, `spd_jitter_std`, `spd_jitter_prob`.

## Поток данных
`pkl → samples → (метаклассы) → train/val split → norm stats (по train) → Dataset(нормализация, исключения каналов) → DataLoader → RTTMultiScale → trainer/evaluate → метрики/чекпойнты`

## Артефакты
- Чекпойнты: `Train/checkpoints/<exp>/best_model.pt`
- Метрики: `Train/results/<exp>/metrics.json`

## Дополнительно
- Классический ML пайплайн: экстракция спектральных/статистических/вейвлет/Хьорт признаков и обучение RF/XGBoost для сравнения с DL-бейзлайном.
