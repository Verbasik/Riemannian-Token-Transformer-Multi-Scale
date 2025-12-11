# -*- coding: utf-8 -*-
"""
A6 — Subject embeddings: размерность и регуляция (DS1: sub-04)

Гипотеза: Увеличение размерности до 32/64 + лёгкая регуляция (Dropout/L2)
повышают переносимость и итоговый f1_macro.

Параметры сетки:
- subject_embed_dim ∈ {16, 32, 64}
- subject_embed_dropout ∈ {0.0, 0.1, 0.2}
- subject_embed_weight_decay ∈ {same_as_base, 5e-4, 1e-3} (опционально)

Фиксируем лучшие найденные:
- Loss: CB-Focal (gamma=1.75, beta=0.999)
- Scheduler: Cosine (T_max=20, warmup_epochs=3)
- gating=False, attn_heads=1 (лучшее по A2/A5)

Артефакты:
  Train/results/ablations/A6_dim{D}_drop{DR}_wd{WD}/metrics.json
  Train/results/ablations/A6_dim{D}_drop{DR}_wd{WD}/config.json
  Train/checkpoints/ablations/A6_dim{D}_drop{DR}_wd{WD}/best_model.pt

Сводка:
  Train/results/ablations/A6_sweep_summary.json
"""
from __future__ import annotations

import argparse
import itertools
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


def run_one(dim: int, drop: float, wd: float | None, epochs: int | None, device_hint: str | None) -> dict:
    cfg = default_config(device_hint=device_hint)
    # DS1
    cfg['data']['subject_ids'] = ['sub-04']
    # Model: best base + A6 variants
    cfg['model']['gating'] = False
    cfg['model']['attn_heads'] = 1
    cfg['model']['subject_embed_dim'] = int(dim)
    cfg['model']['subject_embed_dropout'] = float(drop)
    # Loss
    cfg['loss']['type'] = 'cb_focal'
    cfg['loss']['gamma'] = 1.75
    cfg['loss']['beta'] = 0.999
    # Scheduler
    cfg['scheduler'] = {'name': 'cosine', 'T_max': 20, 'warmup_epochs': 3}
    # Optimizer WD for subject embeddings
    if wd is not None:
        cfg['optimizer']['subject_embed_weight_decay'] = float(wd)
    # Sampler off
    cfg['training']['use_weighted_sampler'] = False
    cfg['training']['allow_sampler_with_cb_focal'] = False
    if epochs is not None:
        cfg['training']['n_epochs'] = int(epochs)

    tag = f"A6_dim{dim}_drop{str(drop).replace('.', 'p')}_wd{('same' if wd is None else str(wd).replace('.', 'p'))}"
    cfg['checkpoint_dir'] = f"Train/checkpoints/ablations/{tag}"
    cfg['results_dir'] = f"Train/results/ablations/{tag}"

    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])
    print("\n" + "=" * 80)
    print(f"A6 SWEEP: dim={dim}, drop={drop}, wd={wd}")
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
    parser = argparse.ArgumentParser(description='A6 — Subject embeddings (dim, dropout, L2) sweep')
    parser.add_argument('--dims', type=str, default='16,32,64', help='comma-separated dims')
    parser.add_argument('--dropouts', type=str, default='0.0,0.1,0.2', help='comma-separated dropouts')
    parser.add_argument('--embed_wd', type=str, default='same,0.0005,0.001', help='comma-separated embed weight decay; use "same" to keep base')
    parser.add_argument('--epochs', type=int, default=None, help='override number of epochs')
    parser.add_argument('--device', type=str, default=None, help='device hint: cuda|cpu')
    args = parser.parse_args()

    dims = [int(x.strip()) for x in args.dims.split(',') if x.strip()]
    drops = [float(x.strip()) for x in args.dropouts.split(',') if x.strip()]
    wds = []
    for x in args.embed_wd.split(','):
        x = x.strip()
        if not x:
            continue
        if x.lower() == 'same':
            wds.append(None)
        else:
            wds.append(float(x))

    results = []
    for d, dr, wd in itertools.product(dims, drops, wds):
        m = run_one(d, dr, wd, epochs=args.epochs, device_hint=args.device)
        results.append({'dim': d, 'dropout': dr, 'embed_wd': ('same' if wd is None else wd), 'metrics': m})

    best = max(results, key=lambda r: r['metrics'].get('f1_macro', float('-inf')))
    summary = {'results': results, 'best': best}
    summary_path = Path('Train/results/ablations/A6_sweep_summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nA6 SWEEP SUMMARY:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

