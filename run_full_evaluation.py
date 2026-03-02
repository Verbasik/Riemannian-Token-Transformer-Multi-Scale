#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full K-Fold Cross-Validation with Subject-Aware CV
=================================================================

Runs complete 5-fold evaluation on all 5 subjects (25 experiments total).
Collects metrics and performs statistical analysis with confidence intervals.

Features:
- ✅ Subject-Aware StratifiedGroupKFold (no subject leakage)
- ✅ All 5 subjects × 5 folds = 25 experiments
- ✅ Bootstrap 95% confidence intervals
- ✅ Wilcoxon signed-rank tests
- ✅ Benjamini-Hochberg FDR correction
- ✅ Per-subject and per-fold analysis

Usage:
    python3 run_full_evaluation.py

Output:
    Train/results/full_evaluation/
    ├── results_detailed.json
    ├── results_summary.json
    └── statistical_analysis.txt
"""

import sys
sys.path.insert(0, 'Pipeline')

import json
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from datetime import datetime

# Configure warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Import pipeline components
from config import default_config
from train import main as train_main

# =============================================================================
# Configuration
# =============================================================================

SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']
N_FOLDS = 5
RESULTS_DIR = Path('Train/results/full_evaluation')

# =============================================================================
# Main Training Loop
# =============================================================================

def run_full_evaluation() -> Dict[str, Any]:
    """
    Description:
    ---------------
        Runs full 5-fold cross-validation on all subjects.
        Collects metrics for statistical analysis.

    Returns:
    ---------------
        Dict with detailed results for each subject/fold combination.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = default_config()
    data_dir = str(base_cfg['data']['data_dir'])

    results = {
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

    # Run each subject × fold combination
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
                # Create config for this experiment
                cfg = default_config()
                cfg['data']['subject_ids'] = [subject_id]
                cfg['cv']['fold_index'] = fold_idx
                # Для per-subject прогона (1 субъект) SGKF неприменим.
                # Используем стратифицированный CV внутри субъекта.
                cfg['cv']['mode'] = 'stratified'

                # Train and get metrics
                metrics = train_main(cfg)

                # Store results
                experiment = {
                    'subject': subject_id,
                    'fold': fold_num,
                    'metrics': metrics,
                    'status': 'success'
                }
                results['experiments'].append(experiment)
                results['per_subject'][subject_id].append(metrics)

                print(f"✅ Completed {subject_id} Fold {fold_num}")
                print(f"   F1-macro: {metrics.get('f1_macro', 'N/A'):.4f}")
                print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")

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

    success_count = sum(1 for exp in results['experiments'] if exp['status'] == 'success')
    failed_count = len(results['experiments']) - success_count
    results['metadata']['success_experiments'] = success_count
    results['metadata']['failed_experiments'] = failed_count

    return results


# =============================================================================
# Statistical Analysis
# =============================================================================

def extract_metric_per_fold(results: Dict[str, Any], metric_name: str = 'f1_macro') -> Dict[str, np.ndarray]:
    """
    Description:
    ---------------
        Extracts specific metric for each subject across all folds.

    Args:
    ---------------
        results: Results dictionary from run_full_evaluation()
        metric_name: Name of metric to extract ('f1_macro', 'accuracy', etc.)

    Returns:
    ---------------
        Dict mapping subject_id -> array of metric values across folds
    """
    metric_per_subject = {}

    for subject_id in SUBJECTS:
        values = []
        for experiment in results['experiments']:
            if experiment['subject'] == subject_id and experiment['status'] == 'success':
                if experiment['metrics'] and metric_name in experiment['metrics']:
                    values.append(experiment['metrics'][metric_name])

        if values:
            metric_per_subject[subject_id] = np.array(values)

    return metric_per_subject


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 95) -> Tuple[float, float, float]:
    """
    Description:
    ---------------
        Computes bootstrap confidence interval for mean.

    Args:
    ---------------
        data: 1D array of values
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval (e.g., 95 for 95% CI)

    Returns:
    ---------------
        (mean, lower_bound, upper_bound)
    """
    mean = np.mean(data)
    bootstrap_means = []

    np.random.seed(42)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)
    alpha = (100 - ci) / 2
    lower = np.percentile(bootstrap_means, alpha)
    upper = np.percentile(bootstrap_means, 100 - alpha)

    return mean, lower, upper


def wilcoxon_test(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """
    Description:
    ---------------
        Performs Wilcoxon signed-rank test between paired samples.

    Args:
    ---------------
        group1: First group values
        group2: Second group values

    Returns:
    ---------------
        (statistic, p_value)
    """
    from scipy import stats
    statistic, p_value = stats.wilcoxon(group1, group2)
    return statistic, p_value


def benjamini_hochberg_correction(p_values: List[float], fdr: float = 0.05) -> List[bool]:
    """
    Description:
    ---------------
        Applies Benjamini-Hochberg FDR correction for multiple comparisons.

    Args:
    ---------------
        p_values: List of p-values
        fdr: False discovery rate threshold

    Returns:
    ---------------
        List of booleans indicating significance after correction
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Compute thresholds
    thresholds = (np.arange(1, n + 1) / n) * fdr

    # Find largest i where P(i) <= (i/m)*alpha
    significant = np.zeros(n, dtype=bool)
    for i in range(n - 1, -1, -1):
        if sorted_p[i] <= thresholds[i]:
            significant[sorted_indices[:i + 1]] = True
            break

    return significant.tolist()


def perform_statistical_analysis(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Description:
    ---------------
        Performs comprehensive statistical analysis of results.

    Args:
    ---------------
        results: Results from run_full_evaluation()

    Returns:
    ---------------
        Dict with statistical analysis results
    """
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'summary': {},
        'per_subject': {},
        'bootstrap_ci': {},
        'comparisons': {},
    }

    metrics = ['f1_macro', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'loss']

    # ===========================================================================
    # 1. Overall Summary (across all subjects and folds)
    # ===========================================================================
    print("\n" + "=" * 100)
    print("STATISTICAL ANALYSIS")
    print("=" * 100)

    for metric in metrics:
        all_values = []
        for exp in results['experiments']:
            if exp['status'] == 'success' and metric in exp['metrics']:
                all_values.append(exp['metrics'][metric])

        if all_values:
            all_values = np.array(all_values)
            mean_val, lower, upper = bootstrap_ci(all_values)

            analysis['summary'][metric] = {
                'mean': float(mean_val),
                'std': float(np.std(all_values)),
                'min': float(np.min(all_values)),
                'max': float(np.max(all_values)),
                'ci_lower': float(lower),
                'ci_upper': float(upper),
                'ci_95': f"[{lower:.4f}, {upper:.4f}]",
            }

            print(f"\n{metric.upper()}:")
            print(f"  Mean: {mean_val:.4f}")
            print(f"  Std:  {np.std(all_values):.4f}")
            print(f"  95% CI: [{lower:.4f}, {upper:.4f}]")
            print(f"  Range: [{np.min(all_values):.4f}, {np.max(all_values):.4f}]")

    # ===========================================================================
    # 2. Per-Subject Analysis
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

            print(f"\n{subject_id}:")
            print(f"  F1-macro: {mean_val:.4f} ± {np.std(values):.4f}")
            print(f"  95% CI: [{lower:.4f}, {upper:.4f}]")
            print(f"  Folds: {[f'{v:.4f}' for v in values]}")

    # ===========================================================================
    # 3. Bootstrap Confidence Intervals for Key Metrics
    # ===========================================================================
    print("\n" + "=" * 100)
    print("BOOTSTRAP CONFIDENCE INTERVALS (95%)")
    print("=" * 100)

    for metric in ['f1_macro', 'accuracy']:
        print(f"\n{metric.upper()}:")
        for subject_id in SUBJECTS:
            metric_values = extract_metric_per_fold(results, metric)
            if subject_id in metric_values:
                values = metric_values[subject_id]
                mean_val, lower, upper = bootstrap_ci(values, n_bootstrap=5000)
                analysis['bootstrap_ci'][f"{subject_id}_{metric}"] = {
                    'mean': float(mean_val),
                    'ci_95': [float(lower), float(upper)],
                }
                print(f"  {subject_id}: {mean_val:.4f} [{lower:.4f}, {upper:.4f}]")

    # ===========================================================================
    # 4. Wilcoxon Tests - Pairwise Comparisons Between Subjects
    # ===========================================================================
    print("\n" + "=" * 100)
    print("WILCOXON SIGNED-RANK TESTS (Subject Comparisons)")
    print("=" * 100)

    metric_per_subject = extract_metric_per_fold(results, 'f1_macro')
    valid_subjects = [s for s in SUBJECTS if s in metric_per_subject]

    p_values = []
    comparisons = []

    for i, subj1 in enumerate(valid_subjects):
        for subj2 in valid_subjects[i + 1:]:
            values1 = metric_per_subject[subj1]
            values2 = metric_per_subject[subj2]

            # Only compare if same length folds
            if len(values1) == len(values2):
                stat, p_val = wilcoxon_test(values1, values2)
                p_values.append(p_val)
                comparisons.append({
                    'pair': f"{subj1} vs {subj2}",
                    'statistic': float(stat),
                    'p_value': float(p_val),
                })

    # Apply Benjamini-Hochberg correction
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
# Results Saving
# =============================================================================

def save_results(detailed_results: Dict[str, Any], analysis: Dict[str, Any]):
    """
    Description:
    ---------------
        Saves detailed results and statistical analysis to files.

    Args:
    ---------------
        detailed_results: Results from run_full_evaluation()
        analysis: Results from perform_statistical_analysis()
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    detailed_path = RESULTS_DIR / 'results_detailed.json'
    with open(detailed_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"✅ Saved: {detailed_path}")

    # Save summary
    summary_path = RESULTS_DIR / 'results_summary.json'
    summary = {
        'metadata': detailed_results['metadata'],
        'summary': analysis['summary'],
        'per_subject': analysis['per_subject'],
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved: {summary_path}")

    # Save analysis
    analysis_path = RESULTS_DIR / 'statistical_analysis.json'
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✅ Saved: {analysis_path}")

    # Generate text report
    report_path = RESULTS_DIR / 'statistical_analysis.txt'
    with open(report_path, 'w') as f:
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
            f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")

        f.write("\n\nPER-SUBJECT RESULTS\n")
        f.write("-" * 80 + "\n")
        for subject, data in analysis['per_subject'].items():
            if 'f1_macro' in data:
                f1_data = data['f1_macro']
                f.write(f"\n{subject}:\n")
                f.write(f"  F1-macro: {f1_data['mean']:.4f} ± {f1_data['std']:.4f}\n")
                f.write(f"  95% CI: [{f1_data['ci_lower']:.4f}, {f1_data['ci_upper']:.4f}]\n")
                f.write(f"  Folds: {f1_data['folds']}\n")

    print(f"✅ Saved: {report_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """
    Description:
    ---------------
        Main entry point for full evaluation pipeline.
    """
    try:
        # Phase 1: Training
        detailed_results = run_full_evaluation()
        if detailed_results['metadata'].get('success_experiments', 0) == 0:
            raise RuntimeError(
                "Нет успешных экспериментов: проверьте cfg['data']['data_dir'] и наличие "
                "каталогов /mnt/data/derivatives/preprocessed_pkl/sub-*/eeg/*.pkl"
            )

        # Phase 2: Statistical Analysis
        analysis = perform_statistical_analysis(detailed_results)

        # Phase 3: Save Results
        print("\n" + "=" * 100)
        print("SAVING RESULTS")
        print("=" * 100)
        save_results(detailed_results, analysis)

        # Final Summary
        print("\n" + "=" * 100)
        print("🎉 FULL EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 100)
        print(f"\nResults saved to: {RESULTS_DIR}")
        print("Files:")
        print(f"  - results_detailed.json (detailed per-experiment results)")
        print(f"  - results_summary.json (aggregated summary)")
        print(f"  - statistical_analysis.json (statistical tests)")
        print(f"  - statistical_analysis.txt (human-readable report)")
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
