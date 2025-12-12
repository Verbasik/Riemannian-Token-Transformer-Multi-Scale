# -*- coding: utf-8 -*-
"""
B2 — SPD ковариации и shrinkage (DS1, 5-fold)

Гипотеза: сохранение тонких корреляций повышает дискриминацию.
Варианты для сравнения:
- OAS с min_alpha=0.1 (база)
- OAS с min_alpha=0.01 (ослабленный clamp)
- Ledoit–Wolf (LW)

Скрипт не трогает baseline-пайплайн, все изменения — через конфиг.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from statistics import mean

# Подключаем Pipeline/ как модульный путь
ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = ROOT / 'Pipeline'
sys.path.insert(0, str(PIPELINE_DIR))

import torch
from config import default_config
from train import build_loaders, build_model, build_criterion, build_optimizer_and_scheduler
from trainer import train_loop, save_artifacts
from utils import set_seed, pretty_print_run


def _to_jsonable(o):
    from pathlib import Path as _P
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, _P):
        return str(o)
    return o


def run_fold(cfg: dict, fold_index: int, tag: str) -> dict:
    cfg['cv']['fold_index'] = fold_index
    cfg['checkpoint_dir'] = f"Train/checkpoints/ablations/{tag}/fold{fold_index+1}"
    cfg['results_dir'] = f"Train/results/ablations/{tag}/fold{fold_index+1}"

    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])
    print("\n" + "=" * 80)
    print(f"B2 RUN: {tag} | Fold {fold_index+1}")
    pretty_print_run(cfg)

    train_loader, val_loader, train_labels, n_channels, n_subjects = build_loaders(cfg)
    model = build_model(cfg, n_channels, n_subjects).to(device)
    criterion = build_criterion(cfg, train_labels).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    _, final_metrics = train_loop(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        cfg=cfg, device=device
    )

    save_artifacts(cfg, final_metrics, model)

    res_dir = Path(cfg['results_dir'])
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(_to_jsonable(cfg), f, indent=2, ensure_ascii=False)
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description='B2 — SPD covariance estimators (OAS vs LW), 5-fold DS1')
    parser.add_argument('--subjects', type=str, default='sub-04', help='comma-separated subjects (DS1 default)')
    parser.add_argument('--folds', type=int, default=5, help='number of CV folds (for split generation)')
    parser.add_argument('--fold_index', type=int, default=None, help='run only this fold index (0-based); keeps n_splits')
    parser.add_argument('--epochs', type=int, default=None, help='override epochs')
    parser.add_argument('--device', type=str, default=None, help='cuda|cpu')
    args = parser.parse_args()

    subject_ids = [s.strip() for s in args.subjects.split(',') if s.strip()]
    folds = int(args.folds)

    # Базовый конфиг (A2/A6/A7 дефолты)
    cfg_base = default_config(device_hint=args.device)
    cfg_base['data']['subject_ids'] = subject_ids
    cfg_base['cv']['n_splits'] = folds
    if args.epochs is not None:
        cfg_base['training']['n_epochs'] = int(args.epochs)
    # Убеждаемся, что sampler отключен при CB-Focal
    cfg_base['training']['use_weighted_sampler'] = False
    cfg_base['training']['allow_sampler_with_cb_focal'] = False
    # Фиксируем лосс/шедулер как у лидера A2
    cfg_base['loss'] = {'type': 'cb_focal', 'beta': 0.999, 'gamma': 1.75}
    cfg_base['scheduler'] = {'name': 'cosine', 'T_max': 20, 'warmup_epochs': 3}

    modes = [
        ('oas_base', {'cov_estimator': 'oas', 'oas_min_alpha': 0.1}),
        ('oas_min0p01', {'cov_estimator': 'oas', 'oas_min_alpha': 0.01}),
        ('lw', {'cov_estimator': 'lw'}),
    ]

    subject_tag = '_'.join(subject_ids) + f"_k{folds}"
    summaries = {}
    for mode_name, params in modes:
        results = []
        run_folds = [args.fold_index] if args.fold_index is not None else list(range(folds))
        for fi in run_folds:
            cfg = cfg_base.copy()
            cfg['model']['cov_estimator'] = params['cov_estimator']
            if params.get('cov_estimator') == 'oas':
                cfg['model']['oas_min_alpha'] = params.get('oas_min_alpha', 0.1)
            tag = f"B2_{mode_name}_{subject_tag}"
            m = run_fold(cfg, fi, tag)
            results.append(m)
        keys = list(results[0].keys())
        mean_metrics = {k: mean([r[k] for r in results]) for k in keys}
        summaries[mode_name] = {
            'mean_metrics': mean_metrics,
            'per_fold': results,
            'fold_indices': run_folds,
            'n_splits': folds
        }

    out_dir = Path('Train/results/ablations')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'B2_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print("\nB2 SUMMARY:")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
