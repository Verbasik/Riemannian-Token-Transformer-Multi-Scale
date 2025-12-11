# -*- coding: utf-8 -*-
"""
A7 — Плотность окон (stride_small) при лучшей конфигурации (DS1: sub-04)

Гипотеза: более частая дискретизация окна повышает полноту признаков.
Сетка: stride_small ∈ {96 (base), 80, 64}

Фиксируем лучшие найденные ранее параметры:
- Loss: CB-Focal (gamma=1.75, beta=0.999)
- Scheduler: Cosine (T_max=20, warmup_epochs=3)
- Model: gating=False, attn_heads=1, subject_embed_dim=16, subject_embed_dropout=0.2
- Optimizer: AdamW, subject_embed_weight_decay=5e-4
- Самплирование: Weighted sampler отключён при CB-Focal

Артефакты по каждому варианту stride:
  Train/results/ablations/A7_stride{S}/metrics.json
  Train/results/ablations/A7_stride{S}/config.json
  Train/checkpoints/ablations/A7_stride{S}/best_model.pt

Сводка:
  Train/results/ablations/A7_sweep_summary.json
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


def run_one(stride: int, epochs: int | None = None, device_hint: str | None = None) -> dict:
    cfg = default_config(device_hint=device_hint)
    # DS1: sub-04
    cfg['data']['subject_ids'] = ['sub-04']
    # Лучшие параметры (на случай отличий от дефолта)
    cfg['loss']['type'] = 'cb_focal'
    cfg['loss']['gamma'] = 1.75
    cfg['loss']['beta'] = 0.999
    cfg['scheduler'] = {'name': 'cosine', 'T_max': 20, 'warmup_epochs': 3}
    cfg['model']['gating'] = False
    cfg['model']['attn_heads'] = 1
    cfg['model']['subject_embed_dim'] = 16
    cfg['model']['subject_embed_dropout'] = 0.2
    cfg['optimizer']['subject_embed_weight_decay'] = 5e-4
    # Меняем только stride_small
    cfg['model']['stride_small'] = int(stride)
    # Отключаем sampler при CB-Focal
    cfg['training']['use_weighted_sampler'] = False
    cfg['training']['allow_sampler_with_cb_focal'] = False
    # Эпохи
    if epochs is not None:
        cfg['training']['n_epochs'] = int(epochs)

    tag = f"A7_stride{stride}"
    cfg['checkpoint_dir'] = f"Train/checkpoints/ablations/{tag}"
    cfg['results_dir'] = f"Train/results/ablations/{tag}"

    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])

    print("\n" + "=" * 80)
    print(f"A7 SWEEP: stride_small={stride}")
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
    parser = argparse.ArgumentParser(description='A7 — stride_small sweep (DS1)')
    parser.add_argument('--strides', type=str, default='96,80,64', help='comma-separated stride_small values')
    parser.add_argument('--epochs', type=int, default=None, help='override number of epochs')
    parser.add_argument('--device', type=str, default=None, help='device hint: cuda|cpu')
    args = parser.parse_args()

    strides = [int(x.strip()) for x in args.strides.split(',') if x.strip()]
    results = []
    for s in strides:
        m = run_one(s, epochs=args.epochs, device_hint=args.device)
        results.append({'stride_small': s, 'metrics': m})

    best = max(results, key=lambda r: r['metrics'].get('f1_macro', float('-inf')))
    summary = {'results': results, 'best': best}
    summary_path = Path('Train/results/ablations/A7_sweep_summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nA7 SWEEP SUMMARY:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

