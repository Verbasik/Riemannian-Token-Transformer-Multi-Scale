---
status: accepted
date: 2026-03-14
decision_id: ADR-001
title: "Visualization Framework for Training & Cross-Validation Results"
---

# ADR-001 — Visualization Framework

**Status:** ✅ Accepted & Implemented

**Date:** 2026-03-14

**Participants:** Haiku agent, Edward (PhD student)

## Problem

Phase 4B requires publication-quality visualizations and metric tables for a thesis chapter:
- Training dynamics (loss, F1, LR schedule, gradient norms)
- Classification results (confusion matrix, per-class metrics, ROC/PR curves)
- Cross-validation analysis (per-subject heatmaps, bootstrap CIs, statistical tests)
- Attention mechanism interpretation (token weights, head importance)

Existing code only logs to JSON/NPZ. Manual plotting post-hoc is error-prone and duplicates logic.

## Decision

**Implement a standalone `Pipeline/visualization.py` module with:**

1. **Backend:** `matplotlib.use('Agg')` for headless GPU servers
2. **Optional seaborn** for enhanced heatmap aesthetics (graceful fallback to pure matplotlib)
3. **16 plotting functions** covering all visualization requirements
4. **Automatic integration** into `trainer.py:save_artifacts()` and `run_full_evaluation.py:main()`
5. **Structured output:** `plots/` and `tables/` subdirectories with PNG (300 DPI) + CSV/Markdown tables

## Rationale

| Aspect | Choice | Why |
|--------|--------|-----|
| Backend | Agg | Non-interactive; safe on headless servers; 300 DPI publication quality |
| Seaborn | Optional | Better aesthetics; graceful degradation; no hard dependency |
| Auto-integration | Via `try/except` | Failures don't crash training; users always get metrics in JSON |
| Output format | PNG + CSV + MD | PNG for papers; CSV for data analysis; MD for human review |
| Scope | All visualization types | Self-contained; reusable; reduces manual notebook creation |

## Alternatives considered

1. **Jupyter notebooks:** Error-prone, requires manual re-execution, version control issues
2. **External plotting library (plotly):** Adds dependency; overkill for static paper figs
3. **Manual plotting in train.py:** Code duplication; harder to maintain; no reusability

**Decision:** Standalone module wins on maintainability, reusability, and automation.

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| seaborn not installed | Graceful fallback to matplotlib; user gets basic but functional plots |
| PNG file size | 300 DPI is standard for journals; file size ~100–200 KB per plot (acceptable) |
| Long runtime on large datasets | Visualization runs in-process; <10 sec for 25 experiments (acceptable) |
| Matplotlib incompatibility | Pinned to matplotlib ≥3.5 (robust); tested on cuda/cpu envs |

## Implementation

**File:** `Pipeline/visualization.py` (700 lines)

**Functions:**
- Training dynamics: `plot_training_curves()` (2×2 dashboard)
- Classification: `plot_confusion_matrix()`, `plot_per_class_metrics()`, `plot_roc_curves()`, `plot_precision_recall_curves()`
- Attention: `plot_attention_heatmap()`, `plot_head_importance()`
- Cross-validation: `plot_cv_boxplot()`, `plot_subject_heatmap()`, `plot_bootstrap_ci()`, `plot_overall_metrics_bar()`
- Tables: `generate_metrics_table()`, `generate_per_class_table()`, `generate_full_eval_table()`
- Wrappers: `save_single_run_plots()`, `save_full_eval_plots()`

**Integration:**
- `trainer.py:save_artifacts()` → calls `save_single_run_plots()` at end
- `run_full_evaluation.py:main()` → calls `save_full_eval_plots()` as Phase 4

## Verification

✅ **Tested:** Phase 4B full eval (25/25 experiments):
- All plots generated successfully (11 PNG files)
- All tables exported (3 CSV + 1 Markdown)
- No crashes; graceful error handling
- Output quality: publication-ready

## References

- `Pipeline/visualization.py` — implementation
- `Train/results/full_evaluation/plots/` — example outputs
- `memory-bank/CONVENTIONS.md` § Logging & Artifacts — related convention
