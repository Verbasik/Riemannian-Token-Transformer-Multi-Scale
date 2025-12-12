# -*- coding: utf-8 -*-
"""
B1 — Потери, согласованные с дисбалансом/метрикой (DS1, 5-fold)

Реализация в отдельном скрипте, чтобы не затрагивать baseline:
- Logit-Adjusted Cross-Entropy (LA-CE): logits' = logits + log(pi)
- Balanced Softmax (BSCE): softmax(exp(z)*pi) (эквивалентно z+log(pi))
- LDAM-DRW: margin per class m_c = m_max / n_c^(1/4); CE(s*(z - m_y)) c весами DRW после 10 эпох

Сравнение с CB-Focal (gamma=1.75, beta=0.999) в тех же условиях.
Запуск формирует отдельные артефакты и сводки per-mode, baseline остаётся неизменным.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from statistics import mean

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Подключаем Pipeline как модуль
ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = ROOT / 'Pipeline'
sys.path.insert(0, str(PIPELINE_DIR))

from config import default_config
from train import build_loaders, build_model, build_optimizer_and_scheduler
from trainer import evaluate, save_artifacts
from utils import set_seed, pretty_print_run


# =============================
# Losses for B1
# =============================

class LogitAdjustedCrossEntropy(nn.Module):
    """LA-CE: CE(logits + λ log π). Here λ=1.0 by default."""
    def __init__(self, class_counts: np.ndarray, lambda_: float = 1.0):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        priors = (counts / counts.sum()).clamp(min=1e-12)
        self.register_buffer('log_prior', torch.log(priors) * lambda_)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_adj = logits + self.log_prior
        return F.cross_entropy(logits_adj, targets, reduction='mean')


class BalancedSoftmaxCE(nn.Module):
    """Balanced Softmax is equivalent to LA-CE with λ=1.0 in logits space."""
    def __init__(self, class_counts: np.ndarray):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        priors = (counts / counts.sum()).clamp(min=1e-12)
        self.register_buffer('log_prior', torch.log(priors))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_adj = logits + self.log_prior
        return F.cross_entropy(logits_adj, targets, reduction='mean')


class LDAMDRW(nn.Module):
    """LDAM with Deferred Re-Weighting.

    - margins: m_c = m_max / n_c^(1/4)
    - loss: CE(s * (z - m_y))
    - DRW: class weights switch at milestone epoch to effective-number weights
    """
    def __init__(self, class_counts: np.ndarray, s: float = 30.0, m_max: float = 0.5,
                 drw_milestone: int = 10, beta: float = 0.9999):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        self.s = float(s)
        self.milestone = int(drw_milestone)
        # margins
        margins = m_max / (counts.clamp(min=1.0) ** 0.25)
        self.register_buffer('margins', margins)
        # DRW weights
        C = len(counts)
        weights_before = torch.ones(C, dtype=torch.float32)
        eff_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), counts)
        weights_after = (1.0 - beta) / eff_num.clamp(min=1e-8)
        weights_after = weights_after / weights_after.mean()  # normalize
        self.register_buffer('w_before', weights_before)
        self.register_buffer('w_after', weights_after)
        # current epoch
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # subtract margin for true class
        margins_y = self.margins[targets]
        logits_adj = logits.clone()
        logits_adj[torch.arange(logits.size(0), device=logits.device), targets] -= margins_y
        logits_adj = logits_adj * self.s
        # choose class weights according to DRW schedule
        if self.current_epoch >= self.milestone:
            weight = self.w_after
        else:
            weight = self.w_before
        return F.cross_entropy(logits_adj, targets, weight=weight, reduction='mean')


# =============================
# Training loop (custom for dynamic losses)
# =============================

def train_loop_b1(model: nn.Module, train_loader, val_loader, criterion: nn.Module,
                  optimizer, scheduler, cfg: dict, device: torch.device):
    use_amp = cfg['training']['use_amp'] and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_state, best_f1, patience = None, 0.0, 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1_macro': [], 'lr': []}
    n_epochs = cfg['training']['n_epochs']
    grad_clip = cfg['training']['grad_clip']

    print(f"\nНачало обучения (B1) на {n_epochs} эпох | AMP: {use_amp}")
    use_subject_embed = hasattr(model, 'use_subject_embed') and model.use_subject_embed

    for epoch in range(n_epochs):
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch + 1)
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            eeg, labels = batch['eeg'].to(device), batch['label'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if use_subject_embed:
                    subject_ids = batch['subject_id'].to(device)
                    logits = model(eeg, subject_ids=subject_ids)
                else:
                    logits = model(eeg)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        val_metrics = evaluate(model, val_loader, device, criterion)
        if scheduler: scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(total_loss / len(train_loader))
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['lr'].append(lr)

        print(f"\n[B1] Эпоха {epoch+1}/{n_epochs} | LR: {lr:.6f}")
        print(f"  Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Acc:    {val_metrics['accuracy']:.4f} | Val F1:   {val_metrics['f1_macro']:.4f}")

        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            print(f"  ✅ Новая лучшая F1: {best_f1:.4f}. Модель сохранена.")
        else:
            patience += 1
            if patience >= cfg['training']['early_stopping_patience']:
                print(f"\n[B1] Ранняя остановка на эпохе {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
        print(f"\n✅ Восстановлена лучшая модель (F1: {best_f1:.4f})")
    final_metrics = evaluate(model, val_loader, device, criterion)
    return history, final_metrics


# =============================
# Runner per-fold and per-mode
# =============================

def _to_jsonable(o):
    from pathlib import Path as _P
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, _P):
        return str(o)
    return o


@dataclass
class ModeConfig:
    name: str
    params: dict


def build_criterion_b1(mode: ModeConfig, train_labels: np.ndarray, n_classes: int) -> nn.Module:
    counts = np.bincount(train_labels, minlength=n_classes)
    if mode.name == 'lace':
        return LogitAdjustedCrossEntropy(counts, lambda_=float(mode.params.get('lambda', 1.0)))
    if mode.name == 'bsce':
        return BalancedSoftmaxCE(counts)
    if mode.name == 'ldam':
        return LDAMDRW(counts, s=float(mode.params.get('s', 30.0)),
                       m_max=float(mode.params.get('m_max', 0.5)),
                       drw_milestone=int(mode.params.get('drw_milestone', 10)),
                       beta=float(mode.params.get('beta', 0.9999)))
    raise ValueError(f"Unknown mode: {mode.name}")


def run_fold_mode(cfg_base: dict, fold_index: int, subject_tag: str, mode: ModeConfig) -> dict:
    cfg = cfg_base.copy()
    cfg['cv']['fold_index'] = fold_index
    tag = f"B1_{mode.name}_{subject_tag}"
    if mode.name == 'ldam' and 's' in mode.params:
        tag += f"_s{mode.params['s']}"
    cfg['checkpoint_dir'] = f"Train/checkpoints/ablations/{tag}/fold{fold_index+1}"
    cfg['results_dir'] = f"Train/results/ablations/{tag}/fold{fold_index+1}"

    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])
    print("\n" + "=" * 80)
    print(f"B1 MODE: {mode.name} | Fold {fold_index+1}")
    pretty_print_run(cfg)

    # Build components
    train_loader, val_loader, train_labels, n_channels, n_subjects = build_loaders(cfg)
    model = build_model(cfg, n_channels, n_subjects).to(device)
    # Select criterion
    if mode.name == 'cbfocal':
        # baseline CB-Focal via default build_criterion in Pipeline/train.py
        from train import build_criterion
        criterion = build_criterion(cfg, train_labels).to(device)
    else:
        criterion = build_criterion_b1(mode, train_labels, n_classes=cfg['model']['n_classes']).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    # Train
    if mode.name == 'cbfocal':
        from trainer import train_loop as baseline_train_loop
        _, final_metrics = baseline_train_loop(
            model=model, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            cfg=cfg, device=device
        )
    else:
        _, final_metrics = train_loop_b1(
            model=model, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            cfg=cfg, device=device
        )

    # Save artifacts and cfg dump
    save_artifacts(cfg, final_metrics, model)
    res_dir = Path(cfg['results_dir'])
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(_to_jsonable(cfg), f, indent=2, ensure_ascii=False)
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description='B1 — Losses aligned with imbalance/metric (5-fold DS1)')
    parser.add_argument('--subjects', type=str, default='sub-04', help='comma-separated subjects (default DS1)')
    parser.add_argument('--folds', type=int, default=5, help='number of CV folds')
    parser.add_argument('--epochs', type=int, default=None, help='override epochs')
    parser.add_argument('--device', type=str, default=None, help='cuda|cpu')
    parser.add_argument('--modes', type=str, default='cbfocal,lace,bsce,ldam', help='comma list: cbfocal,lace,bsce,ldam')
    parser.add_argument('--ldam_s', type=str, default='16,32', help='s values for LDAM')
    parser.add_argument('--drw_milestone', type=int, default=10, help='epoch to enable DRW for LDAM')
    args = parser.parse_args()

    subject_ids = [s.strip() for s in args.subjects.split(',') if s.strip()]
    folds = int(args.folds)

    cfg = default_config(device_hint=args.device)
    cfg['data']['subject_ids'] = subject_ids
    cfg['cv']['n_splits'] = folds
    if args.epochs is not None:
        cfg['training']['n_epochs'] = int(args.epochs)
    # Общие правила как для A2: отключить sampler при CB-Focal
    cfg['training']['use_weighted_sampler'] = False
    cfg['training']['allow_sampler_with_cb_focal'] = False
    # Зафиксировать A2/A3 параметры, чтобы менять только loss
    cfg['loss'] = {'type': 'cb_focal', 'beta': 0.999, 'gamma': 1.75}
    cfg['scheduler'] = {'name': 'cosine', 'T_max': 20, 'warmup_epochs': 3}

    modes_req = [m.strip().lower() for m in args.modes.split(',') if m.strip()]
    mode_list: list[ModeConfig] = []
    for m in modes_req:
        if m == 'cbfocal':
            mode_list.append(ModeConfig('cbfocal', {}))
        elif m == 'lace':
            mode_list.append(ModeConfig('lace', {'lambda': 1.0}))
        elif m == 'bsce':
            mode_list.append(ModeConfig('bsce', {}))
        elif m == 'ldam':
            for s in [int(x.strip()) for x in args.ldam_s.split(',') if x.strip()]:
                mode_list.append(ModeConfig('ldam', {'s': s, 'drw_milestone': args.drw_milestone}))
        else:
            raise ValueError(f"Unknown mode: {m}")

    subject_tag = '_'.join(subject_ids) + f"_k{folds}"
    summaries = {}
    for mode in mode_list:
        results = []
        for fi in range(folds):
            m = run_fold_mode(cfg.copy(), fi, subject_tag, mode)
            results.append(m)
        keys = list(results[0].keys())
        mean_metrics = {k: mean([r[k] for r in results]) for k in keys}
        summaries[mode.name + (f"_s{mode.params['s']}" if 's' in mode.params else '')] = {
            'mean_metrics': mean_metrics,
            'per_fold': results
        }

    # Сводка по всем режимам
    out_dir = Path('Train/results/ablations')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'B1_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print("\nB1 SUMMARY:")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

