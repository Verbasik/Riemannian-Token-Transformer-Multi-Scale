# -*- coding: utf-8 -*-
"""
A3 — Расписание LR (Cosine + warmup) на DS1 (sub-04).

Гипотеза: согласование периода с длиной обучения и мягкий старт повышают итоговую метрику.
Изменение: scheduler={'name': 'cosine', 'T_max': 50, 'warmup_epochs': 5}

Сохраняет результаты в:
  Train/results/ablations/A3_cosine_T50_warm5/metrics.json
  Train/checkpoints/ablations/A3_cosine_T50_warm5/best_model.pt
  + config.json (дамп итогового cfg)
"""
from __future__ import annotations

import argparse
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


def main():
    parser = argparse.ArgumentParser(description='A3 — CosineLR (T_max=50, warmup=5) eval on DS1')
    parser.add_argument('--epochs', type=int, default=None, help='override number of epochs')
    parser.add_argument('--device', type=str, default=None, help='device hint: cuda|cpu')
    args = parser.parse_args()

    cfg = default_config(device_hint=args.device)
    # DS1: sub-04
    cfg['data']['subject_ids'] = ['sub-04']
    # A3: убедимся, что расписание задано как в задаче
    cfg['scheduler'] = {'name': 'cosine', 'T_max': 50, 'warmup_epochs': 5}
    # Хор. гиперпараметры A2 уже зафиксированы в default_config (gamma=1.75, beta=0.999)
    if args.epochs is not None:
        cfg['training']['n_epochs'] = int(args.epochs)

    tag = 'A3_cosine_T50_warm5'
    cfg['checkpoint_dir'] = f'Train/checkpoints/ablations/{tag}'
    cfg['results_dir'] = f'Train/results/ablations/{tag}'

    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])

    print('\n' + '=' * 80)
    print(f'A3 EVAL: Cosine(T_max=50,warmup=5) on DS1 (sub-04)')
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

    print('\nA3 FINISHED:')
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

