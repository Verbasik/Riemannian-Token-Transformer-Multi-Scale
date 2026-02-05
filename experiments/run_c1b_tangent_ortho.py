# -*- coding: utf-8 -*-
"""
C1b — Ортонормальная проекция в тангенциальном пространстве (после logm→vectorize)

Идея: применить обучаемую ортонормальную линейную проекцию к векторизованным лог‑SPD признакам,
что согласовано с эвклидовой геометрией tangent space и менее склонно к изотропизации.

Сетка размеров проекции (по умолчанию): 192,160,128 (входная размерность ≈ 300 при proj_channels=24).
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


def run_one(tan_dim: int, epochs: int | None, device_hint: str | None) -> dict:
    cfg = default_config(device_hint=device_hint)
    # DS1: фиксируем субъекта
    cfg['data']['subject_ids'] = ['sub-04']
    # A2/A3 базовые параметры
    cfg['loss'] = {'type': 'cb_focal', 'beta': 0.999, 'gamma': 1.75}
    cfg['scheduler'] = {'name': 'cosine', 'T_max': 20, 'warmup_epochs': 3}
    # Отключаем SPDNet и включаем C1b
    cfg['model']['use_spdnet'] = False
    cfg['model']['use_tangent_ortho'] = True
    cfg['model']['tangent_ortho_dim'] = int(tan_dim)
    # Эпохи (опционально)
    if epochs is not None:
        cfg['training']['n_epochs'] = int(epochs)

    tag = f"C1b_tangent_ortho_dim{tan_dim}"
    cfg['checkpoint_dir'] = f"Train/checkpoints/ablations/{tag}"
    cfg['results_dir'] = f"Train/results/ablations/{tag}"

    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])
    print("\n" + "=" * 80)
    print(f"C1b RUN: tan_dim={tan_dim}")
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
    parser = argparse.ArgumentParser(description='C1b — Orthonormal projection in tangent space (DS1, Fold1)')
    parser.add_argument('--dims', type=str, default='192,160,128', help='comma-separated projection dims')
    parser.add_argument('--epochs', type=int, default=None, help='override epochs')
    parser.add_argument('--device', type=str, default=None, help='cuda|cpu')
    args = parser.parse_args()

    dims = [int(x.strip()) for x in args.dims.split(',') if x.strip()]
    results = []
    for d in dims:
        m = run_one(d, args.epochs, args.device)
        results.append({'tan_dim': d, 'metrics': m})

    summary = {'results': results}
    out = Path('Train/results/ablations/C1b_sweep_summary.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nC1b SWEEP SUMMARY:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

