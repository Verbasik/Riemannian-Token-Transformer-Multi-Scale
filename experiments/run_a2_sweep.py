# -*- coding: utf-8 -*-
"""
A2 — CB-Focal настройка (gamma, beta) на DS1 (sub-04, Fold 1).

- Отключает WeightedRandomSampler при CB-Focal.
- Перебирает сетку gamma ∈ {1.25, 1.5, 1.75}, beta ∈ {0.999, 0.9995, 0.9999}.
- Для каждой комбинации запускает тренировку и сохраняет артефакты в
  Train/results/ablations/A2_focal_gamma{g}_beta{b}/metrics.json
  Train/checkpoints/ablations/A2_focal_gamma{g}_beta{b}/best_model.pt
- Дополнительно дампит итоговый cfg как config.json рядом с метриками.
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


def fmt_float(x: float) -> str:
    s = ("%.4f" % x).rstrip('0').rstrip('.')
    return s.replace('.', 'p')


def run_one(gamma: float, beta: float, epochs: int | None = None, device_hint: str | None = None) -> dict:
    cfg = default_config(device_hint=device_hint)
    # A2: отключаем sampler при CB-Focal
    cfg['training']['use_weighted_sampler'] = False
    cfg['training']['allow_sampler_with_cb_focal'] = False
    # Обновляем loss
    cfg['loss']['type'] = 'cb_focal'
    cfg['loss']['gamma'] = float(gamma)
    cfg['loss']['beta'] = float(beta)
    # Эпохи (опционально)
    if epochs is not None:
        cfg['training']['n_epochs'] = int(epochs)
    # Настраиваем уникальные папки результатов
    tag = f"A2_focal_gamma{fmt_float(gamma)}_beta{fmt_float(beta)}"
    cfg['checkpoint_dir'] = f"Train/checkpoints/ablations/{tag}"
    cfg['results_dir'] = f"Train/results/ablations/{tag}"

    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])

    print("\n" + "=" * 80)
    print(f"A2 SWEEP: gamma={gamma}, beta={beta}")
    pretty_print_run(cfg)

    # Build
    train_loader, val_loader, train_labels, n_channels, n_subjects = build_loaders(cfg)
    model = build_model(cfg, n_channels, n_subjects).to(device)
    criterion = build_criterion(cfg, train_labels).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    # Train
    history, final_metrics = train_loop(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        cfg=cfg, device=device
    )

    # Save artifacts (best model is already restored inside train_loop)
    save_artifacts(cfg, final_metrics, model)

    # Dump cfg (sanitize non-JSON types like pathlib.Path)
    def _to_jsonable(o):
        from pathlib import Path as _P
        if isinstance(o, dict):
            return {k: _to_jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_jsonable(v) for v in o]
        if isinstance(o, _P):
            return str(o)
        return o

    res_dir = Path(cfg['results_dir'])
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(_to_jsonable(cfg), f, indent=2, ensure_ascii=False)

    print("\nFINISHED:")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description='A2 — CB-Focal (gamma, beta) sweep')
    parser.add_argument('--gamma', type=float, help='single gamma value')
    parser.add_argument('--beta', type=float, help='single beta value')
    parser.add_argument('--epochs', type=int, default=None, help='override number of epochs')
    parser.add_argument('--device', type=str, default=None, help='device hint: cuda|cpu')
    args = parser.parse_args()

    if args.gamma is not None and args.beta is not None:
        run_one(args.gamma, args.beta, epochs=args.epochs, device_hint=args.device)
        return

    gammas = [1.25, 1.5, 1.75]
    betas = [0.999, 0.9995, 0.9999]
    for g in gammas:
        for b in betas:
            run_one(g, b, epochs=args.epochs, device_hint=args.device)


if __name__ == '__main__':
    main()
