# -*- coding: utf-8 -*-
"""
C2a — Learnable GraphConv over electrodes (DS1, Fold 1)

Свип по параметрам: K, layers, nonlinearity, norm, dropout, k, sigma.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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


def run_one(params: dict, epochs: int | None, device_hint: str | None) -> dict:
    cfg = default_config(device_hint=device_hint)
    cfg['data']['subject_ids'] = ['sub-04']
    cfg['loss'] = {'type': 'cb_focal', 'beta': 0.999, 'gamma': 1.75}
    cfg['scheduler'] = {'name': 'cosine', 'T_max': 20, 'warmup_epochs': 3}
    # Enable C2a
    m = cfg['model']
    m['use_gcn'] = True
    m['use_spdnet'] = False
    m['use_tangent_ortho'] = False
    m['gcn_k'] = int(params['k'])
    m['gcn_sigma'] = float(params['sigma'])
    m['gcn_K'] = int(params['K'])
    m['gcn_layers'] = int(params['layers'])
    m['gcn_nonlinearity'] = params['nl']
    m['gcn_norm'] = params['norm']
    m['gcn_filter'] = params['filter']
    m['gcn_dropout'] = float(params['dropout'])
    # Epochs override
    if epochs is not None:
        cfg['training']['n_epochs'] = int(epochs)

    tag = (
        f"C2a_k{params['k']}_sig{str(params['sigma']).replace('.', 'p')}_"
        f"K{params['K']}_L{params['layers']}_nl{params['nl']}_norm{params['norm']}_"
        f"f{params['filter']}_do{str(params['dropout']).replace('.', 'p')}"
    )
    cfg['checkpoint_dir'] = f"Train/checkpoints/ablations/{tag}"
    cfg['results_dir'] = f"Train/results/ablations/{tag}"

    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])
    print("\n" + "=" * 80)
    print(f"C2a RUN: {tag}")
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
    parser = argparse.ArgumentParser(description='C2a — Learnable GraphConv over electrodes (DS1, Fold1)')
    parser.add_argument('--k', type=str, default='8', help='kNN')
    parser.add_argument('--sigma', type=str, default='0.05', help='gaussian sigma')
    parser.add_argument('--K', type=str, default='1', help='hop order (chebyshev/poly)')
    parser.add_argument('--layers', type=str, default='2', help='number of GraphConv layers')
    parser.add_argument('--nl', type=str, default='relu', help='nonlinearity: relu|tanh|none')
    parser.add_argument('--norm', type=str, default='layer', help='norm: layer|batch|none')
    parser.add_argument('--dropout', type=str, default='0.1', help='channel dropout prob')
    parser.add_argument('--filter', type=str, default='cheby', help='filter: cheby|poly')
    parser.add_argument('--epochs', type=int, default=None, help='override epochs')
    parser.add_argument('--device', type=str, default=None, help='cuda|cpu')
    args = parser.parse_args()

    params = {
        'k': int(args.k), 'sigma': float(args.sigma), 'K': int(args.K), 'layers': int(args.layers),
        'nl': args.nl, 'norm': args.norm, 'dropout': float(args.dropout), 'filter': args.filter
    }
    m = run_one(params, args.epochs, args.device)
    summary = {'params': params, 'metrics': m}
    out = Path('Train/results/ablations/C2a_single_summary.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nC2a SINGLE SUMMARY:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

