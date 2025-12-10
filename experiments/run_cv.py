# -*- coding: utf-8 -*-
"""
5-fold репликация результатов на DS1 (sub-04) и DS2 (sub-01..sub-04).

Функции:
- Прогнать k фолдов с конфигом (по умолчанию k=5) и собрать средние метрики.
- Параметры: список субъектов, число эпох (опционально), устройство.
- Сохраняет per-fold метрики/чекпойнты и summary.json с усреднением.

Примеры:
- DS1 (sub-04):
  python3 experiments/run_cv.py --subjects sub-04
- DS2 (sub-01..sub-04):
  python3 experiments/run_cv.py --subjects sub-01,sub-02,sub-03,sub-04
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
    # Папки per-fold
    cfg['checkpoint_dir'] = f"Train/checkpoints/cv/{tag}/fold{fold_index+1}"
    cfg['results_dir'] = f"Train/results/cv/{tag}/fold{fold_index+1}"

    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])

    print("\n" + "=" * 80)
    print(f"CV RUN: {tag} | Fold {fold_index+1}")
    pretty_print_run(cfg)

    # Build
    train_loader, val_loader, train_labels, n_channels, n_subjects = build_loaders(cfg)
    model = build_model(cfg, n_channels, n_subjects).to(device)
    criterion = build_criterion(cfg, train_labels).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    # Train
    _, final_metrics = train_loop(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        cfg=cfg, device=device
    )

    # Save artifacts
    save_artifacts(cfg, final_metrics, model)

    # Dump cfg
    res_dir = Path(cfg['results_dir'])
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(_to_jsonable(cfg), f, indent=2, ensure_ascii=False)

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description='5-fold CV for DS1/DS2')
    parser.add_argument('--subjects', type=str, default='sub-04', help='comma-separated list of subject IDs')
    parser.add_argument('--folds', type=int, default=5, help='number of CV folds')
    parser.add_argument('--epochs', type=int, default=None, help='override number of epochs')
    parser.add_argument('--device', type=str, default=None, help='device hint: cuda|cpu')
    args = parser.parse_args()

    subject_ids = [s.strip() for s in args.subjects.split(',') if s.strip()]
    folds = int(args.folds)

    # Конфиг по умолчанию + настройки
    cfg = default_config(device_hint=args.device)
    cfg['data']['subject_ids'] = subject_ids
    # Убеждаемся, что sampler отключен при CB-Focal
    cfg['training']['use_weighted_sampler'] = False
    cfg['training']['allow_sampler_with_cb_focal'] = False
    if args.epochs is not None:
        cfg['training']['n_epochs'] = int(args.epochs)
    cfg['cv']['n_splits'] = folds

    # Тег набора
    tag = f"{'_'.join(subject_ids)}_k{folds}"

    # Запуски по фолдам
    metrics_list = []
    for fi in range(folds):
        m = run_fold(cfg.copy(), fi, tag)
        metrics_list.append(m)

    # Агрегация
    keys = list(metrics_list[0].keys())
    mean_metrics = {k: mean([m[k] for m in metrics_list]) for k in keys}

    summary_dir = Path(f"Train/results/cv/{tag}")
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'subjects': subject_ids,
            'folds': folds,
            'mean_metrics': mean_metrics,
            'per_fold': metrics_list
        }, f, indent=2, ensure_ascii=False)

    print("\nCV SUMMARY:")
    print(json.dumps(mean_metrics, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

