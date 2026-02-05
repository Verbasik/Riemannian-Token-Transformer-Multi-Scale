# -*- coding: utf-8 -*-
"""
C3 — GRL + CORAL для борьбы с subject shift (DS2, subject-aware CV)

Запуск:
- LOSO: по одному субъекту валидационный, остальные train
- Сетка: da_lambda ∈ {0.05, 0.1, 0.2}, coral_lambda ∈ {0.0, 0.01, 0.05}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from statistics import mean
import copy

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = ROOT / 'Pipeline'
sys.path.insert(0, str(PIPELINE_DIR))

import numpy as np
import torch
from config import default_config
from data_loader import load_all_data_metaclass
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


def loso_splits(subjects: list[str], samples: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    subs = np.array([s['subject'] for s in samples])
    splits = []
    for sv in subjects:
        val_idx = np.nonzero(subs == sv)[0]
        train_idx = np.nonzero(subs != sv)[0]
        if len(val_idx) == 0 or len(train_idx) == 0:
            continue
        splits.append((train_idx, val_idx))
    return splits


def run_fold(cfg: dict, train_idx: np.ndarray, val_idx: np.ndarray, tag: str) -> dict:
    cfg['cv']['predefined_split'] = (train_idx, val_idx)
    cfg['checkpoint_dir'] = f"Train/checkpoints/ablations/{tag}"
    cfg['results_dir'] = f"Train/results/ablations/{tag}"
    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])
    pretty_print_run(cfg)

    train_loader, val_loader, train_labels, n_channels, n_subjects = build_loaders(cfg)
    model = build_model(cfg, n_channels, n_subjects).to(device)
    criterion = build_criterion(cfg, train_labels).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    _, final_metrics = train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, cfg, device)
    save_artifacts(cfg, final_metrics, model)
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description='C3 — GRL+CORAL (DS2 subject-aware CV)')
    parser.add_argument('--subjects', type=str, default='sub-01,sub-04', help='comma-separated subjects (DS2)')
    parser.add_argument('--mode', type=str, default='loso', help='loso')
    parser.add_argument('--da', type=str, default='0.05,0.1,0.2', help='comma-separated da_lambda values')
    parser.add_argument('--coral', type=str, default='0.0,0.01,0.05', help='comma-separated coral_lambda values')
    parser.add_argument('--epochs', type=int, default=None, help='override epochs')
    parser.add_argument('--device', type=str, default=None, help='cuda|cpu')
    args = parser.parse_args()

    subjects = [s.strip() for s in args.subjects.split(',') if s.strip()]
    da_vals = [float(x.strip()) for x in args.da.split(',') if x.strip()]
    coral_vals = [float(x.strip()) for x in args.coral.split(',') if x.strip()]

    # Load samples to compute splits
    cfg_base = default_config(device_hint=args.device)
    cfg_base['data']['subject_ids'] = subjects
    samples = load_all_data_metaclass(cfg_base['data']['data_dir'], subjects, cfg_base['data']['task'], verbose=True)
    if args.epochs is not None:
        cfg_base['training']['n_epochs'] = int(args.epochs)
    # Enable C3
    cfg_base['model']['use_c3'] = True
    # Ensure subject embeddings on to provide subject_ids and personalization
    cfg_base['model']['use_subject_embed'] = True

    # Build LOSO splits
    if args.mode.lower() == 'loso':
        splits = loso_splits(subjects, samples)
    else:
        splits = loso_splits(subjects, samples)

    summaries = {}
    for da in da_vals:
        for coral in coral_vals:
            cfg = copy.deepcopy(cfg_base)
            cfg['model']['c3']['da_lambda'] = float(da)
            cfg['model']['c3']['coral_lambda'] = float(coral)
            fold_metrics = []
            for fi, (train_idx, val_idx) in enumerate(splits):
                tag = f"C3_loso_{subjects[fi]}_da{str(da).replace('.', 'p')}_cor{str(coral).replace('.', 'p')}"
                m = run_fold(cfg.copy(), train_idx, val_idx, tag)
                fold_metrics.append(m)
            # Aggregate
            keys = list(fold_metrics[0].keys()) if fold_metrics else []
            mean_metrics = {k: mean([m[k] for m in fold_metrics]) for k in keys} if fold_metrics else {}
            summaries[f"da{da}_cor{coral}"] = {'mean_metrics': mean_metrics, 'per_fold': fold_metrics}

    out = Path('Train/results/ablations/C3_sweep_summary.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'grid': {'da': da_vals, 'coral': coral_vals, 'mode': args.mode, 'subjects': subjects}, 'results': summaries}, f, indent=2, ensure_ascii=False)
    print("\nC3 SWEEP SUMMARY:")
    print(json.dumps({'results': summaries}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
