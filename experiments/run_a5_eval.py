# -*- coding: utf-8 -*-
"""
A4+A5 — Gating + attn_heads=2 (DS1: sub-04)

Гипотеза: комбинация адаптивного взвешивания масштабов и более выразительного
мультиголовного pooling улучшит разделимость и повысит f1_macro.

Изменения относительно A2:
- model.gating = True
- model.attn_heads = 2

Остальное — лучшие найденные настройки:
- Loss: CB-Focal (gamma=1.75, beta=0.999)
- Scheduler: Cosine (T_max=20, warmup_epochs=3)
- Weighted sampler: отключен при CB-Focal

Артефакты:
  Train/results/ablations/A4A5_gating_true_heads2/metrics.json
  Train/results/ablations/A4A5_gating_true_heads2/config.json
  Train/checkpoints/ablations/A4A5_gating_true_heads2/best_model.pt
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
    parser = argparse.ArgumentParser(description='A4+A5 — Gating + attn_heads on DS1 (sub-04)')
    parser.add_argument('--epochs', type=int, default=None, help='override number of epochs')
    parser.add_argument('--heads', type=int, default=2, help='number of attn_heads for pooling (e.g., 2 or 4)')
    parser.add_argument('--device', type=str, default=None, help='device hint: cuda|cpu')
    args = parser.parse_args()

    cfg = default_config(device_hint=args.device)
    # DS1
    cfg['data']['subject_ids'] = ['sub-04']
    # A4 + A5
    cfg['model']['gating'] = True
    cfg['model']['attn_heads'] = int(args.heads)
    # A2 best loss
    cfg['loss']['type'] = 'cb_focal'
    cfg['loss']['gamma'] = 1.75
    cfg['loss']['beta'] = 0.999
    # Scheduler (A2 best)
    cfg['scheduler'] = {'name': 'cosine', 'T_max': 20, 'warmup_epochs': 3}
    # Sampler off
    cfg['training']['use_weighted_sampler'] = False
    cfg['training']['allow_sampler_with_cb_focal'] = False
    if args.epochs is not None:
        cfg['training']['n_epochs'] = int(args.epochs)

    tag = f"A4A5_gating_true_heads{cfg['model']['attn_heads']}"
    cfg['checkpoint_dir'] = f'Train/checkpoints/ablations/{tag}'
    cfg['results_dir'] = f'Train/results/ablations/{tag}'

    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])

    print('\n' + '=' * 80)
    print('A4+A5 EVAL: Gating=True, attn_heads=2 on DS1 (sub-04)')
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

    print('\nA4+A5 FINISHED:')
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
