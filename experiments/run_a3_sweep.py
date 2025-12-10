# -*- coding: utf-8 -*-
"""
A3' — Сетка расписания LR (Cosine + warmup) для DS1 (sub-04)

Сетка:
- T_max ∈ {20, 40, 50}
- warmup_epochs ∈ {3, 5, 10}

Фиксированные параметры:
- CB-Focal: gamma=1.75, beta=0.999 (лучшие из A2)
- WeightedRandomSampler: отключен при CB-Focal

Артефакты для каждой комбинации:
  Train/results/ablations/A3_cosine_T{T}_warm{W}/metrics.json
  Train/results/ablations/A3_cosine_T{T}_warm{W}/config.json
  Train/checkpoints/ablations/A3_cosine_T{T}_warm{W}/best_model.pt

Сводка по всем прогоном:
  Train/results/ablations/A3_sweep_summary.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

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


def run_one(t_max: int, warmup: int, epochs: int | None = None, device_hint: str | None = None) -> dict:
    cfg = default_config(device_hint=device_hint)
    # DS1 фиксируем явно
    cfg['data']['subject_ids'] = ['sub-04']
    # A2-параметры loss (на всякий случай зафиксировать явно)
    cfg['loss']['type'] = 'cb_focal'
    cfg['loss']['gamma'] = 1.75
    cfg['loss']['beta'] = 0.999
    # A1 выключен при CB-Focal
    cfg['training']['use_weighted_sampler'] = False
    cfg['training']['allow_sampler_with_cb_focal'] = False
    # A3 варианты расписания
    cfg['scheduler'] = {'name': 'cosine', 'T_max': int(t_max), 'warmup_epochs': int(warmup)}
    if epochs is not None:
        cfg['training']['n_epochs'] = int(epochs)

    tag = f"A3_cosine_T{t_max}_warm{warmup}"
    cfg['checkpoint_dir'] = f"Train/checkpoints/ablations/{tag}"
    cfg['results_dir'] = f"Train/results/ablations/{tag}"

    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])

    print("\n" + "=" * 80)
    print(f"A3 SWEEP: T_max={t_max}, warmup={warmup}")
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

    print("\nFINISHED:")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description='A3 sweep: Cosine(T_max) × Warmup on DS1')
    parser.add_argument('--tmax', type=str, default='20,40,50', help='comma-separated list of T_max values')
    parser.add_argument('--warmup', type=str, default='3,5,10', help='comma-separated list of warmup epochs')
    parser.add_argument('--epochs', type=int, default=None, help='override number of epochs')
    parser.add_argument('--device', type=str, default=None, help='device hint: cuda|cpu')
    args = parser.parse_args()

    t_list = [int(x.strip()) for x in args.tmax.split(',') if x.strip()]
    w_list = [int(x.strip()) for x in args.warmup.split(',') if x.strip()]

    results = []
    for t, w in itertools.product(t_list, w_list):
        m = run_one(t, w, epochs=args.epochs, device_hint=args.device)
        results.append({'T_max': t, 'warmup_epochs': w, 'metrics': m})

    # Сводка по f1_macro
    best = max(results, key=lambda r: r['metrics'].get('f1_macro', float('-inf')))
    summary = {
        'grid': {'T_max': t_list, 'warmup_epochs': w_list},
        'results': results,
        'best': best,
    }
    summary_path = Path('Train/results/ablations/A3_sweep_summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nA3 SWEEP SUMMARY:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

