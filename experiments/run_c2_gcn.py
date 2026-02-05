# -*- coding: utf-8 -*-
"""
C2 — Пространственный граф (GCN) по электродам (DS1, Fold 1)

Применяет графовое сглаживание по электродам перед `channel_proj`:
X' = (1−α) X + α Â X, где Â — нормированная матрица смежности (kNN по montage.csv + self-loop).

Сетка: k ∈ {6, 8}, α ∈ {0.2, 0.4}
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


def run_one(k: int, alpha: float, epochs: int | None, device_hint: str | None) -> dict:
    cfg = default_config(device_hint=device_hint)
    cfg['data']['subject_ids'] = ['sub-04']
    # Baseline best (A2/A3/A6)
    cfg['loss'] = {'type': 'cb_focal', 'beta': 0.999, 'gamma': 1.75}
    cfg['scheduler'] = {'name': 'cosine', 'T_max': 20, 'warmup_epochs': 3}
    # Enable C2; disable SPDNet/tangent_ortho
    cfg['model']['use_spdnet'] = False
    cfg['model']['use_tangent_ortho'] = False
    cfg['model']['use_gcn'] = True
    cfg['model']['gcn_k'] = int(k)
    cfg['model']['gcn_alpha'] = float(alpha)
    if epochs is not None:
        cfg['training']['n_epochs'] = int(epochs)

    tag = f"C2_gcn_k{k}_alpha{str(alpha).replace('.', 'p')}"
    cfg['checkpoint_dir'] = f"Train/checkpoints/ablations/{tag}"
    cfg['results_dir'] = f"Train/results/ablations/{tag}"

    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])
    print("\n" + "=" * 80)
    print(f"C2 RUN: k={k}, alpha={alpha}")
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
    parser = argparse.ArgumentParser(description='C2 — GCN over electrodes (DS1, Fold1)')
    parser.add_argument('--ks', type=str, default='6,8', help='comma-separated kNN sizes')
    parser.add_argument('--alphas', type=str, default='0.2,0.4', help='comma-separated alphas')
    parser.add_argument('--nl', type=str, default='tanh', help="nonlinearity: tanh|relu|none")
    parser.add_argument('--epochs', type=int, default=None, help='override number of epochs')
    parser.add_argument('--device', type=str, default=None, help='cuda|cpu')
    args = parser.parse_args()

    ks = [int(x.strip()) for x in args.ks.split(',') if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(',') if x.strip()]

    results = []
    for k in ks:
        for a in alphas:
            # inject nonlinearity via env config
            # (we reuse default_config; override at model level)
            cfg = default_config(device_hint=args.device)
            cfg['data']['subject_ids'] = ['sub-04']
            cfg['loss'] = {'type': 'cb_focal', 'beta': 0.999, 'gamma': 1.75}
            cfg['scheduler'] = {'name': 'cosine', 'T_max': 20, 'warmup_epochs': 3}
            cfg['model']['use_spdnet'] = False
            cfg['model']['use_tangent_ortho'] = False
            cfg['model']['use_gcn'] = True
            cfg['model']['gcn_k'] = int(k)
            cfg['model']['gcn_alpha'] = float(a)
            cfg['model']['gcn_nonlinearity'] = args.nl
            if args.epochs is not None:
                cfg['training']['n_epochs'] = int(args.epochs)

            tag = f"C2_gcn_k{k}_alpha{str(a).replace('.', 'p')}_nl{args.nl}"
            cfg['checkpoint_dir'] = f"Train/checkpoints/ablations/{tag}"
            cfg['results_dir'] = f"Train/results/ablations/{tag}"

            set_seed(cfg['seed'])
            device = torch.device(cfg['device'])
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

            results.append({'k': k, 'alpha': a, 'nl': args.nl, 'metrics': final_metrics})

    summary = {'grid': {'ks': ks, 'alphas': alphas}, 'results': results}
    out = Path('Train/results/ablations/C2_sweep_summary.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nC2 SWEEP SUMMARY:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
