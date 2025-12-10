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

## Поток данных
`pkl → samples → (метаклассы) → train/val split → norm stats (по train) → Dataset(нормализация, исключения каналов) → DataLoader → RTTMultiScale → trainer/evaluate → метрики/чекпойнты`

## Артефакты
- Чекпойнты: `Train/checkpoints/<exp>/best_model.pt`
- Метрики: `Train/results/<exp>/metrics.json`

## Дополнительно
- Классический ML пайплайн: экстракция спектральных/статистических/вейвлет/Хьорт признаков и обучение RF/XGBoost для сравнения с DL-бейзлайном.
