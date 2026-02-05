# -*- coding: utf-8 -*-
"""
C1 — SPDNet-вставка до logm (DS1, Fold 1 по умолчанию)

Идея: обучаемые SPD-преобразования (BiMap + ReEig) до логарифмической карты.
Сетка dims:
- [16]
- [20,16]

База: лучшие настройки A2/A3 (+A6), остальное неизменно. Baseline не трогаем.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List
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


def parse_dims(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def dims_tag(dims: List[int]) -> str:
    return 'x'.join(str(d) for d in dims) if dims else 'none'


def run_one(dims: List[int], epochs: int | None, device_hint: str | None, alpha: float | None = None) -> dict:
    cfg = default_config(device_hint=device_hint)
    # DS1
    cfg['data']['subject_ids'] = ['sub-04']
    # A2
    cfg['loss'] = {'type': 'cb_focal', 'beta': 0.999, 'gamma': 1.75}
    # A3
    cfg['scheduler'] = {'name': 'cosine', 'T_max': 20, 'warmup_epochs': 3}
    # A6 baseline
    cfg['model']['subject_embed_dim'] = 16
    cfg['model']['subject_embed_dropout'] = 0.2
    # C1 enable
    cfg['model']['use_spdnet'] = True
    cfg['model']['spdnet_dims'] = dims
    if alpha is not None:
        cfg['model']['spdnet_alpha'] = float(alpha)
    # Ensure proj_channels >= first SPDNet dim (BiMap reduces or keeps dim only)
    if dims and dims[0] > cfg['model']['proj_channels']:
        cfg['model']['proj_channels'] = int(dims[0])

    if epochs is not None:
        cfg['training']['n_epochs'] = int(epochs)

    tag = f"C1_spdnet_dims{dims_tag(dims)}"
    cfg['checkpoint_dir'] = f"Train/checkpoints/ablations/{tag}"
    cfg['results_dir'] = f"Train/results/ablations/{tag}"

    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])

    print("\n" + "=" * 80)
    print(f"C1 RUN: dims={dims}")
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
    parser = argparse.ArgumentParser(description='C1 — SPDNet insertion before logm (DS1, Fold1)')
    parser.add_argument('--dims', type=str, default='16;20,16', help='semicolon-separated configs, each dims as comma list e.g. "16;20,16"')
    parser.add_argument('--epochs', type=int, default=None, help='override epochs')
    parser.add_argument('--device', type=str, default=None, help='cuda|cpu')
    parser.add_argument('--alpha', type=float, default=None, help='single residual alpha (overrides config)')
    parser.add_argument('--alphas', type=str, default=None, help='comma-separated list of alphas for sweep (e.g., "0.05,0.1")')
    args = parser.parse_args()

    configs = [parse_dims(cfg) for cfg in args.dims.split(';') if cfg.strip()]
    if args.alphas is not None:
        alphas = [float(x.strip()) for x in args.alphas.split(',') if x.strip()]
    else:
        alphas = [args.alpha] if args.alpha is not None else [None]
    results = []
    for dims in configs:
        for a in alphas:
            m = run_one(dims, args.epochs, args.device, a)
            results.append({'dims': dims, 'alpha': a, 'metrics': m})

    # Summary
    summary = {'results': results}
    out = Path('Train/results/ablations/C1_sweep_summary.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nC1 SWEEP SUMMARY:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
