"""
TEMPLATE: Subject-Aware Cross-Validation Implementation
================================================================
Шаблон для внедрения критического компонента StratifiedGroupKFold.

Это 🔴 БЛОКИРУЮЩИЙ элемент для финализации проекта.
Требуется 1-2 дня для реализации и тестирования.

ФАЙЛЫ ДЛЯ ИЗМЕНЕНИЯ:
  1. Pipeline/data_loader.py - добавить StratifiedGroupKFold функцию
  2. Pipeline/train.py - использовать group CV splits
  3. Pipeline/config.py - добавить флаг для mode='subject_aware'

================================================================
"""

# ============================================================================
# ШАГ 1: В data_loader.py добавить функцию
# ============================================================================

# ДОБАВИТЬ после функции get_stratified_cv_splits():

def get_stratified_group_cv_splits(
    labels: np.ndarray,
    groups: np.ndarray,  # subject_ids as integers
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Stratified K-Fold с группировкой по субъектам.

    Гарантирует, что одна и та же группа (субъект) не будет в train и val одновременно.
    Это важно для cross-subject обобщаемости.

    Args:
        labels: np.ndarray [N] - мета-классы (0..7)
        groups: np.ndarray [N] - subject_id indices
        n_splits: int - число folds (по умолчанию 5)
        random_state: int - seed для воспроизводимости

    Returns:
        List[(train_idx, val_idx)] - стратифицированные разбиения с группировкой

    Использование:
        >>> samples = load_all_data_metaclass(...)
        >>> labels = np.array([s['label'] for s in samples])
        >>> groups = np.array([subject_mapping[s['subject']] for s in samples])
        >>> splits = get_stratified_group_cv_splits(labels, groups)
    """
    from sklearn.model_selection import StratifiedGroupKFold

    sgf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    return list(sgf.split(X=np.arange(len(labels)), y=labels, groups=groups))


# ============================================================================
# ШАГ 2: В config.py добавить флаг
# ============================================================================

# МОДИФИЦИРОВАТЬ default_config():

def default_config(device_hint: Optional[str] = None) -> Dict[str, Any]:
    # ... (существующий код)

    return {
        # ... (существующие секции)

        'cv': {
            'n_splits': 5,
            'random_state': RANDOM_SEED,
            'fold_index': 0,
            'mode': 'stratified_group',  # ← НОВЫЙ ПАРАМЕТР
            # 'mode' может быть:
            #   'stratified': StratifiedKFold (текущее, перемешивает субъектов)
            #   'stratified_group': StratifiedGroupKFold (GROUP-aware, FIX!)
            #   'loso': Leave-One-Subject-Out (максимально строгое)
        },

        # ... (остальное)
    }


# ============================================================================
# ШАГ 3: В train.py модифицировать build_loaders()
# ============================================================================

# ЗАМЕНИТЬ текущий вызов get_stratified_cv_splits() на:

def build_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, np.ndarray, int, int]:
    samples = load_all_data_metaclass(
        data_dir=cfg['data']['data_dir'],
        subject_ids=cfg['data']['subject_ids'],
        task=cfg['data']['task'],
    )

    subject_mapping = create_subject_mapping(samples)
    n_subjects = len(subject_mapping)

    dataset = ChiscoDataset(
        samples=samples,
        normalize=cfg['data']['normalize'],
        exclude_channels=cfg['data'].get('exclude_channels'),
        subject_mapping=subject_mapping
    )

    labels = np.array([s['label'] for s in samples])

    # ← НОВАЯ ЛОГИКА: Выбор режима CV
    cv_mode = cfg['cv'].get('mode', 'stratified_group')

    if cv_mode == 'stratified':
        # Старый режим (текущее, перемешивает субъектов)
        splits = get_stratified_cv_splits(
            labels,
            cfg['cv']['n_splits'],
            cfg['cv']['random_state']
        )
        print("⚠️  CV Mode: Stratified (перемешивает субъектов)")

    elif cv_mode == 'stratified_group':
        # НОВЫЙ режим (группировка по субъектам)
        groups = np.array([subject_mapping[s['subject']] for s in samples])
        splits = get_stratified_group_cv_splits(
            labels,
            groups,
            cfg['cv']['n_splits'],
            cfg['cv']['random_state']
        )
        print("✅ CV Mode: Stratified Group (группировка по субъектам)")

    elif cv_mode == 'loso':
        # МАКСИМАЛЬНО СТРОГИЙ: Leave-One-Subject-Out
        splits = get_loso_splits(labels, subject_mapping, samples)
        print("🔐 CV Mode: LOSO (Leave-One-Subject-Out, максимально строгое)")

    else:
        raise ValueError(f"Unknown CV mode: {cv_mode}")

    fold_index = int(cfg.get('cv', {}).get('fold_index', 0))
    fold_index = max(0, min(fold_index, len(splits) - 1))
    train_idx, val_idx = splits[fold_index]

    # ... (остальное как раньше)


# ============================================================================
# ШАГ 4 (ОПЦИОНАЛЬНО): Реализовать LOSO
# ============================================================================

def get_loso_splits(
    labels: np.ndarray,
    subject_mapping: Dict[str, int],
    samples: List[Dict]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Leave-One-Subject-Out кросс-валидация.

    Для каждого субъекта:
    - Test: все сэмплы этого субъекта
    - Train: все сэмплы остальных субъектов

    Это максимально строгая оценка cross-subject обобщаемости.
    """
    unique_subjects = sorted(subject_mapping.keys())
    splits = []

    for test_subject in unique_subjects:
        test_mask = np.array([s['subject'] == test_subject for s in samples])
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        splits.append((train_idx, test_idx))

    print(f"✅ LOSO: {len(splits)} разбиений (по субъектам: {unique_subjects})")
    return splits


# ============================================================================
# ШАГ 5: ТЕСТИРОВАНИЕ
# ============================================================================

"""
ТЕСТОВЫЙ СЦЕНАРИЙ:

from config import default_config
from data_loader import get_stratified_group_cv_splits

# Текущий конфиг
cfg = default_config()
cfg['cv']['mode'] = 'stratified_group'  # ← НОВОЕ

# Загрузить данные
samples = load_all_data_metaclass(...)
labels = np.array([s['label'] for s in samples])
groups = np.array([subject_mapping[s['subject']] for s in samples])

# Проверить разбиения
splits = get_stratified_group_cv_splits(labels, groups)

# Для первого fold проверить что subject_ids не пересекаются
train_idx, val_idx = splits[0]
train_subjects = set(samples[i]['subject'] for i in train_idx)
val_subjects = set(samples[i]['subject'] for i in val_idx)

assert train_subjects.isdisjoint(val_subjects), \
    "ERROR: Субъекты пересекаются между train и val!"

print(f"✅ Fold 1 валидирован:")
print(f"   Train subjects: {train_subjects}")
print(f"   Val subjects: {val_subjects}")
"""


# ============================================================================
# РЕКОМЕНДАЦИЯ: ПОРЯДОК РЕАЛИЗАЦИИ
# ============================================================================

"""
День 1:
  1. Добавить get_stratified_group_cv_splits() в data_loader.py
  2. Добавить флаг 'mode' в config.py
  3. Модифицировать build_loaders() для поддержки обоих режимов
  4. Тестировать на Fold 1, Fold 2

День 2:
  1. Запустить полный 5-fold на одном субъекте с новым режимом
  2. Сравнить результаты: stratified vs stratified_group
  3. Документировать различия в performance
  4. Commit с сообщением: "Add subject-aware CV (StratifiedGroupKFold)"

День 3:
  1. (Опционально) Реализовать LOSO для максимально строгой оценки
  2. Запустить полный LOSO × 5 folds
  3. Финальный report
"""


# ============================================================================
# ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ
# ============================================================================

"""
После внедрения Subject-Aware CV:

1. ✅ Соответствие review.txt улучшится с 92% до 95%+
2. ✅ Будет возможна оценка cross-subject обобщаемости
3. ✅ Результаты станут более доверительными
4. ✅ Раскроются скрытые проблемы (если есть overfitting на субъектов)

ВОЗМОЖНЫЕ ИСХОДЫ:
  - Сценарий A: Performance улучшится (улучшена generalization)
  - Сценарий B: Performance упадет (овerfitting на субъектов, нужны доработки)
  - Сценарий C: Performance останется стабильным (хорошая generalization)

В любом случае, результат будет ДОВЕРИТЕЛЬНЫМ и ВОСПРОИЗВОДИМЫМ.
"""


# ============================================================================
# ССЫЛКИ И РЕСУРСЫ
# ============================================================================

"""
Документация:
  - sklearn.model_selection.StratifiedGroupKFold
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

  - review.txt Раздел 10.1, 13

  - memory-bank: activeContext.md (следующие шаги)

Аналогичные работы:
  - Carvalho et al. 2024 - использовали subject-aware протокол
  - Wang & Ji 2022 - контролировали утечку по субъектам
"""

print(__doc__)
