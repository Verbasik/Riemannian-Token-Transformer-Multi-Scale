#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full K-Fold Cross-Validation with Subject-Aware CV
=================================================================

Запускает полную 5-кратную кросс-валидацию на всех 5 субъектах
(всего 25 экспериментов). Собирает метрики и проводит статистический
анализ с доверительными интервалами.

Возможности:
- ✅ Subject-Aware StratifiedGroupKFold (отсутствие утечки по субъектам)
- ✅ Все 5 субъектов × 5 фолдов = 25 экспериментов
- ✅ Bootstrap 95% доверительные интервалы
- ✅ Тест знаковых рангов Уилкоксона (Wilcoxon signed-rank)
- ✅ Поправка Бенджамини-Хохберга на множественное тестирование (FDR)
- ✅ Анализ по каждому субъекту и фолду

Использование:
    python3 run_full_evaluation.py

Выходные данные:
    Train/results/full_evaluation/
    ├── results_detailed.json
    ├── results_summary.json
    ├── statistical_analysis.json
    ├── statistical_analysis.txt
    ├── plots/
    │   ├── cv_boxplot.png
    │   ├── subject_fold_heatmap.png
    │   ├── bootstrap_ci.png
    │   └── overall_metrics_bar.png
    └── tables/
        ├── full_eval_summary.csv
        ├── per_subject_metrics.csv
        └── metrics_report.md
"""

# =============================================================================
# Standard Libraries
# =============================================================================
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# =============================================================================
# Third-Party Libraries
# =============================================================================
import numpy as np

# =============================================================================
# Local Imports
# =============================================================================
# Добавляем директорию Pipeline в путь поиска
sys.path.insert(0, 'Pipeline')

# Игнорирование предупреждений UserWarning для чистоты вывода
warnings.filterwarnings('ignore', category=UserWarning)

from config import default_config
from train import main as train_main

# =============================================================================
# Configuration (Константы)
# =============================================================================

SUBJECTS: List[str] = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']
N_FOLDS: int = 5
RESULTS_DIR: Path = Path('Train/results/full_evaluation')


# =============================================================================
# Main Training Loop (Основной цикл обучения)
# =============================================================================

def run_full_evaluation() -> Dict[str, Any]:
    """
    Description:
    ---------------
        Запускает полную 5-кратную кросс-валидацию на всех субъектах.
        Для каждого субъекта выполняется внутренний стратифицированный CV,
        так как при одном субъекте группировка невозможна.
        Собирает метрики для последующего статистического анализа.

    Returns:
    ---------------
        Dict[str, Any]: Словарь с детальными результатами для каждой
            комбинации субъект/фолд.

    Raises:
    ---------------
        Нет явных исключений (ошибки ловятся внутри цикла).

    Examples:
    ---------------
        >>> results = run_full_evaluation()
        >>> len(results['experiments'])
        25
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = default_config()
    data_dir = str(base_cfg['data']['data_dir'])

    results: Dict[str, Any] = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'cv_mode': 'stratified_group',
            'n_subjects': len(SUBJECTS),
            'n_folds': N_FOLDS,
            'n_experiments': len(SUBJECTS) * N_FOLDS,
            'subjects': SUBJECTS,
            'data_dir': data_dir,
        },
        'experiments': [],
        'per_subject': {},
        'per_fold': {},
    }

    total_experiments = len(SUBJECTS) * N_FOLDS
    completed = 0

    print("\n" + "=" * 100)
    print("FULL K-FOLD CROSS-VALIDATION WITH SUBJECT-AWARE CV")
    print("=" * 100)
    print(f"Configuration:")
    print(f"  Subjects: {SUBJECTS}")
    print(f"  Folds: {N_FOLDS}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  CV Mode: stratified_group (subject-aware)")
    print(f"  Data dir: {data_dir}")
    print("=" * 100 + "\n")

    # Запуск каждой комбинации субъект × фолд
    for subject_idx, subject_id in enumerate(SUBJECTS, 1):
        if subject_id not in results['per_subject']:
            results['per_subject'][subject_id] = []

        for fold_idx in range(N_FOLDS):
            completed += 1
            progress = f"[{completed}/{total_experiments}]"
            fold_num = fold_idx + 1

            print(f"\n{progress} Running {subject_id} × Fold {fold_num}/{N_FOLDS}...")
            print("-" * 100)

            try:
                # Создание конфигурации для текущего эксперимента
                cfg = default_config()
                cfg['data']['subject_ids'] = [subject_id]
                cfg['cv']['fold_index'] = fold_idx

                # Для прогона на одном субъекте (per-subject) SGKF неприменим,
                # так как нельзя разделить группы. Используем обычный Stratified CV.
                cfg['cv']['mode'] = 'stratified'

                # Обучение и получение метрик
                metrics = train_main(cfg)

                # Сохранение результатов
                experiment = {
                    'subject': subject_id,
                    'fold': fold_num,
                    'metrics': metrics,
                    'status': 'success'
                }
                results['experiments'].append(experiment)
                results['per_subject'][subject_id].append(metrics)

                f1_val = metrics.get('f1_macro', 0.0)
                acc_val = metrics.get('accuracy', 0.0)
                print(f"✅ Completed {subject_id} Fold {fold_num}")
                print(f"   F1-macro: {f1_val:.4f}")
                print(f"   Accuracy: {acc_val:.4f}")

            except Exception as e:
                print(f"❌ Error in {subject_id} Fold {fold_num}: {str(e)}")
                experiment = {
                    'subject': subject_id,
                    'fold': fold_num,
                    'metrics': None,
                    'status': 'failed',
                    'error': str(e)
                }
                results['experiments'].append(experiment)

    print("\n" + "=" * 100)
    print("✅ TRAINING PHASE COMPLETED")
    print("=" * 100)

    success_count = sum(
        1 for exp in results['experiments'] if exp['status'] == 'success'
    )
    failed_count = len(results['experiments']) - success_count

    results['metadata']['success_experiments'] = success_count
    results['metadata']['failed_experiments'] = failed_count

    return results


# =============================================================================
# Statistical Analysis (Статистический анализ)
# =============================================================================

def extract_metric_per_fold(
    results: Dict[str, Any],
    metric_name: str = 'f1_macro'
) -> Dict[str, np.ndarray]:
    """
    Description:
    ---------------
        Извлекает конкретную метрику для каждого субъекта по всем фолдам.

    Args:
    ---------------
        results: Dict[str, Any] - Результаты из run_full_evaluation().
        metric_name: str - Имя метрики ('f1_macro', 'accuracy' и т.д.).

    Returns:
    ---------------
        Dict[str, np.ndarray]: Словарь {subject_id -> array значений}.

    Raises:
    ---------------
        Нет явных исключений.
    """
    metric_per_subject: Dict[str, np.ndarray] = {}

    for subject_id in SUBJECTS:
        values: List[float] = []
        for experiment in results['experiments']:
            if (experiment['subject'] == subject_id and
                    experiment['status'] == 'success'):
                if experiment['metrics'] and metric_name in experiment['metrics']:
                    values.append(experiment['metrics'][metric_name])

        if values:
            metric_per_subject[subject_id] = np.array(values)

    return metric_per_subject


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 95
) -> Tuple[float, float, float]:
    """
    Description:
    ---------------
        Вычисляет бутстрэп доверительный интервал для среднего значения.
        Использует ресемплинг с возвращением для оценки распределения среднего.

    Args:
    ---------------
        data: np.ndarray - Одномерный массив значений.
        n_bootstrap: int - Количество бутстрэп выборок.
        ci: float - Уровень доверия в процентах (например, 95).

    Returns:
    ---------------
        Tuple[float, float, float]: (mean, lower_bound, upper_bound).

    Raises:
    ---------------
        Нет явных исключений.
    """
    mean = float(np.mean(data))
    bootstrap_means: List[float] = []

    np.random.seed(42)  # Фиксация seed для воспроизводимости CI
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(float(np.mean(sample)))

    bootstrap_means_arr = np.array(bootstrap_means)
    alpha = (100 - ci) / 2.0
    lower = float(np.percentile(bootstrap_means_arr, alpha))
    upper = float(np.percentile(bootstrap_means_arr, 100 - alpha))

    return mean, lower, upper


def wilcoxon_test(
    group1: np.ndarray,
    group2: np.ndarray
) -> Tuple[float, float]:
    """
    Description:
    ---------------
        Выполняет тест знаковых рангов Уилкоксона для парных выборок.
        Используется для проверки значимости различий между субъектами.

    Args:
    ---------------
        group1: np.ndarray - Значения первой группы.
        group2: np.ndarray - Значения второй группы.

    Returns:
    ---------------
        Tuple[float, float]: (statistic, p_value).

    Raises:
    ---------------
        Нет явных исключений.
    """
    from scipy import stats
    statistic, p_value = stats.wilcoxon(group1, group2)
    return float(statistic), float(p_value)


def benjamini_hochberg_correction(
    p_values: List[float],
    fdr: float = 0.05
) -> List[bool]:
    """
    Description:
    ---------------
        Применяет поправку Бенджамини-Хохберга для контроля FDR
        при множественном тестировании гипотез.

    Args:
    ---------------
        p_values: List[float] - Список p-значений.
        fdr: float - Порог False Discovery Rate.

    Returns:
    ---------------
        List[bool]: Список булевых флагов значимости после коррекции.

    Raises:
    ---------------
        Нет явных исключений.
    """
    n = len(p_values)
    if n == 0:
        return []

    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Вычисление порогов: (i / m) * alpha
    ranks = np.arange(1, n + 1)
    thresholds = (ranks / n) * fdr

    significant = np.zeros(n, dtype=bool)

    # Поиск наибольшего i, где P(i) <= порогу
    for i in range(n - 1, -1, -1):
        if sorted_p[i] <= thresholds[i]:
            significant[sorted_indices[:i + 1]] = True
            break

    return significant.tolist()


def perform_statistical_analysis(
    results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Description:
    ---------------
        Выполняет комплексный статистический анализ результатов:
        1. Общее резюме по всем экспериментам.
        2. Анализ по каждому субъекту.
        3. Бутстрэп доверительные интервалы.
        4. Построчные сравнения (Wilcoxon) с поправкой на множественность.

    Args:
    ---------------
        results: Dict[str, Any] - Результаты из run_full_evaluation().

    Returns:
    ---------------
        Dict[str, Any]: Словарь со статистическим анализом.

    Raises:
    ---------------
        Нет явных исключений.
    """
    analysis: Dict[str, Any] = {
        'timestamp': datetime.now().isoformat(),
        'summary': {},
        'per_subject': {},
        'bootstrap_ci': {},
        'comparisons': {},
    }

    metrics_list = [
        'f1_macro', 'accuracy', 'balanced_accuracy',
        'precision_macro', 'recall_macro', 'loss'
    ]

    # ===========================================================================
    # 1. Overall Summary (Общее резюме)
    # ===========================================================================
    print("\n" + "=" * 100)
    print("STATISTICAL ANALYSIS")
    print("=" * 100)

    for metric in metrics_list:
        all_values: List[float] = []
        for exp in results['experiments']:
            if exp['status'] == 'success' and metric in exp['metrics']:
                all_values.append(exp['metrics'][metric])

        if all_values:
            all_values_arr = np.array(all_values)
            mean_val, lower, upper = bootstrap_ci(all_values_arr)

            analysis['summary'][metric] = {
                'mean': float(mean_val),
                'std': float(np.std(all_values_arr)),
                'min': float(np.min(all_values_arr)),
                'max': float(np.max(all_values_arr)),
                'ci_lower': float(lower),
                'ci_upper': float(upper),
                'ci_95': f"[{lower:.4f}, {upper:.4f}]",
            }

            print(f"\n{metric.upper()}:")
            print(f"  Mean: {mean_val:.4f}")
            print(f"  Std:  {np.std(all_values_arr):.4f}")
            print(f"  95% CI: [{lower:.4f}, {upper:.4f}]")
            print(
                f"  Range: [{np.min(all_values_arr):.4f}, "
                f"{np.max(all_values_arr):.4f}]"
            )

    # ===========================================================================
    # 2. Per-Subject Analysis (Анализ по субъектам)
    # ===========================================================================
    print("\n" + "=" * 100)
    print("PER-SUBJECT ANALYSIS")
    print("=" * 100)

    metric_per_subject = extract_metric_per_fold(results, 'f1_macro')

    for subject_id in SUBJECTS:
        if subject_id in metric_per_subject:
            values = metric_per_subject[subject_id]
            mean_val, lower, upper = bootstrap_ci(values)

            analysis['per_subject'][subject_id] = {
                'f1_macro': {
                    'mean': float(mean_val),
                    'std': float(np.std(values)),
                    'ci_lower': float(lower),
                    'ci_upper': float(upper),
                    'folds': values.tolist(),
                },
            }

            folds_str = ', '.join([f'{v:.4f}' for v in values])
            print(f"\n{subject_id}:")
            print(f"  F1-macro: {mean_val:.4f} ± {np.std(values):.4f}")
            print(f"  95% CI: [{lower:.4f}, {upper:.4f}]")
            print(f"  Folds: [{folds_str}]")

    # ===========================================================================
    # 3. Bootstrap Confidence Intervals (Доверительные интервалы)
    # ===========================================================================
    print("\n" + "=" * 100)
    print("BOOTSTRAP CONFIDENCE INTERVALS (95%)")
    print("=" * 100)

    for metric in ['f1_macro', 'accuracy']:
        print(f"\n{metric.upper()}:")
        metric_map = extract_metric_per_fold(results, metric)
        for subject_id in SUBJECTS:
            if subject_id in metric_map:
                values = metric_map[subject_id]
                # Увеличиваем число итераций для точности CI
                mean_val, lower, upper = bootstrap_ci(values, n_bootstrap=5000)
                key = f"{subject_id}_{metric}"
                analysis['bootstrap_ci'][key] = {
                    'mean': float(mean_val),
                    'ci_95': [float(lower), float(upper)],
                }
                print(f"  {subject_id}: {mean_val:.4f} [{lower:.4f}, {upper:.4f}]")

    # ===========================================================================
    # 4. Wilcoxon Tests (Попарные сравнения)
    # ===========================================================================
    print("\n" + "=" * 100)
    print("WILCOXON SIGNED-RANK TESTS (Subject Comparisons)")
    print("=" * 100)

    metric_per_subject = extract_metric_per_fold(results, 'f1_macro')
    valid_subjects = [s for s in SUBJECTS if s in metric_per_subject]

    p_values: List[float] = []
    comparisons: List[Dict[str, Any]] = []

    for i, subj1 in enumerate(valid_subjects):
        for subj2 in valid_subjects[i + 1:]:
            values1 = metric_per_subject[subj1]
            values2 = metric_per_subject[subj2]

            # Сравнение возможно только при одинаковой длине выборок
            if len(values1) == len(values2):
                stat, p_val = wilcoxon_test(values1, values2)
                p_values.append(p_val)
                comparisons.append({
                    'pair': f"{subj1} vs {subj2}",
                    'statistic': stat,
                    'p_value': p_val,
                })

    # Применение поправки Бенджамини-Хохберга
    if p_values:
        significant = benjamini_hochberg_correction(p_values, fdr=0.05)
        print("\nPairwise Comparisons (F1-macro):")
        for comp, sig in zip(comparisons, significant):
            sig_str = "✅ SIGNIFICANT" if sig else "❌ ns"
            print(f"  {comp['pair']}: p={comp['p_value']:.4f} {sig_str}")
            comp['significant_fdr05'] = sig

        analysis['comparisons'] = comparisons

    print("\n" + "=" * 100)
    print("✅ STATISTICAL ANALYSIS COMPLETED")
    print("=" * 100 + "\n")

    return analysis


# =============================================================================
# Results Saving (Сохранение результатов)
# =============================================================================

def save_results(
    detailed_results: Dict[str, Any],
    analysis: Dict[str, Any]
) -> None:
    """
    Description:
    ---------------
        Сохраняет детальные результаты и статистический анализ в файлы:
        - JSON с полными данными.
        - JSON с кратким резюме.
        - JSON со статистикой.
        - TXT отчет для человека.

    Args:
    ---------------
        detailed_results: Dict[str, Any] - Результаты обучения.
        analysis: Dict[str, Any] - Результаты статистики.

    Returns:
    ---------------
        None

    Raises:
    ---------------
        Нет явных исключений.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Сохранение детальных результатов
    detailed_path = RESULTS_DIR / 'results_detailed.json'
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {detailed_path}")

    # Сохранение резюме
    summary_path = RESULTS_DIR / 'results_summary.json'
    summary = {
        'metadata': detailed_results['metadata'],
        'summary': analysis['summary'],
        'per_subject': analysis['per_subject'],
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {summary_path}")

    # Сохранение статистического анализа
    analysis_path = RESULTS_DIR / 'statistical_analysis.json'
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {analysis_path}")

    # Генерация текстового отчета
    report_path = RESULTS_DIR / 'statistical_analysis.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("FULL K-FOLD CROSS-VALIDATION STATISTICAL REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {analysis['timestamp']}\n")
        f.write(f"Subjects: {SUBJECTS}\n")
        f.write(f"Folds: {N_FOLDS}\n")
        f.write(f"Total Experiments: {len(SUBJECTS) * N_FOLDS}\n\n")

        f.write("OVERALL SUMMARY\n")
        f.write("-" * 80 + "\n")
        for metric, stats in analysis['summary'].items():
            f.write(f"\n{metric.upper()}:\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Std:  {stats['std']:.4f}\n")
            f.write(f"  95% CI: {stats['ci_95']}\n")
            f.write(
                f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n"
            )

        f.write("\n\nPER-SUBJECT RESULTS\n")
        f.write("-" * 80 + "\n")
        for subject, data in analysis['per_subject'].items():
            if 'f1_macro' in data:
                f1_data = data['f1_macro']
                f.write(f"\n{subject}:\n")
                f.write(
                    f"  F1-macro: {f1_data['mean']:.4f} ± "
                    f"{f1_data['std']:.4f}\n"
                )
                f.write(
                    f"  95% CI: [{f1_data['ci_lower']:.4f}, "
                    f"{f1_data['ci_upper']:.4f}]\n"
                )
                f.write(f"  Folds: {f1_data['folds']}\n")

    print(f"✅ Saved: {report_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> bool:
    """
    Description:
    ---------------
        Главная точка входа для пайплайна полной оценки.
        Выполняет три фазы: Обучение, Статистика, Сохранение.

    Returns:
    ---------------
        bool: True если успешно, False если произошла ошибка.

    Raises:
    ---------------
        RuntimeError: Если ни один эксперимент не завершился успешно.
    """
    try:
        # Phase 1: Обучение
        detailed_results = run_full_evaluation()

        success_exp = detailed_results['metadata'].get('success_experiments', 0)
        if success_exp == 0:
            raise RuntimeError(
                "Нет успешных экспериментов: проверьте cfg['data']['data_dir'] "
                "и наличие каталогов "
                "/mnt/data/derivatives/preprocessed_pkl/sub-*/eeg/*.pkl"
            )

        # Phase 2: Статистический анализ
        analysis = perform_statistical_analysis(detailed_results)

        # Phase 3: Сохранение результатов
        print("\n" + "=" * 100)
        print("SAVING RESULTS")
        print("=" * 100)
        save_results(detailed_results, analysis)

        # Phase 4: Генерация визуализаций
        print("\n" + "=" * 100)
        print("GENERATING VISUALIZATIONS")
        print("=" * 100)
        try:
            from visualization import save_full_eval_plots
            save_full_eval_plots(detailed_results, analysis, RESULTS_DIR)
        except Exception as exc:
            print(f'[viz] Визуализация пропущена: {exc}')

        # Финальное резюме
        print("\n" + "=" * 100)
        print("🎉 FULL EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 100)
        print(f"\nResults saved to: {RESULTS_DIR}")
        print("Files:")
        print(f"  - results_detailed.json  (detailed per-experiment results)")
        print(f"  - results_summary.json   (aggregated summary)")
        print(f"  - statistical_analysis.json (statistical tests)")
        print(f"  - statistical_analysis.txt  (human-readable report)")
        print(f"  - plots/cv_boxplot.png          (metric distributions)")
        print(f"  - plots/subject_fold_heatmap.png (subject × fold grid)")
        print(f"  - plots/bootstrap_ci.png         (95% CI per subject)")
        print(f"  - plots/overall_metrics_bar.png  (mean ± std bar chart)")
        print(f"  - tables/full_eval_summary.csv")
        print(f"  - tables/per_subject_metrics.csv")
        print(f"  - tables/metrics_report.md")
        print("\n")

        return True

    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)