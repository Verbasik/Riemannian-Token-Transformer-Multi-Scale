# Архитектура и паттерны

## Высокоуровневая архитектура
- Слоение по доменам: `data` (загрузка/нормализация/сплиты) → `features/SPD` → `model` → `training` → `artifacts`.
- Builder‑фабрики в `train.py`: `build_loaders`, `build_model`, `build_criterion`, `build_optimizer_and_scheduler` изолируют создание компонентов от логики тренировки.
- Конфигурация как источник правды: `config.default_config()` задаёт поведение всех доменов (пути/режимы/параметры).

## Ключевые паттерны
- Hybrid Normalization: subject‑wise центрирование + global масштабирование (устраняет baseline, сохраняет межсубъектную вариативность).
- SPD Manifold Features: ковариации с OAS‑сжатием → корреляционные матрицы → log‑map → векторизация верхнего треугольника.
- Multi‑Scale Tokenization: два масштаба окон (малый/большой) + токены со специальным `scale_emb` и `cls` токеном.
- TransformerEncoder + Attention Pooling: свёртка по токенам через softmax‑веса голов и обучаемые веса голов.
- Subject Embeddings (optional): эмбеддинг субъекта конкатенируется к агрегированным признакам перед классификатором.
- Early Stopping + Best‑state restore: модель восстанавливается к лучшему состоянию по `f1_macro`.
- Class‑Balanced Focal Loss: компенсация дисбаланса классов.

## Диаграмма потока (текстовая)
```
Pickle/JSON → load_all_data_metaclass → Dataset(norm=hybrid) → DataLoader
    → RTTMultiScale(window→SPD→log→vec→proj→Transformer→attn pool [+ subj emb])
    → ClassBalancedFocalLoss + Optimizer + Scheduler → train_loop
    → evaluate → save_artifacts (best_model.pt, metrics.json)
```

## Известные ограничения
- Стоимость eigendecomposition для SPD на больших батчах/окнах.
- Требования к консистентности путей данных (вне репозитория).
- Несоответствие сигнатур в `test_dryrun.py` относительно текущих билд-функций.

