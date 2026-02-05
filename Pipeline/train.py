# file: train.py
# -*- coding: utf-8 -*-
"""
Точка входа обучения Phase 4B с логгингом артефактов для анализа.
"""
import argparse
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import default_config
from data_loader import (
    ChiscoDataset,
    ChiscoSubset,
    compute_channelwise_stats,
    compute_hybrid_stats,
    compute_subjectwise_stats,
    create_subject_mapping,
    get_stratified_cv_splits,
    load_all_data_metaclass,
)
from model import RTTMultiScale
from trainer import ClassBalancedFocalLoss, save_artifacts, train_loop
from utils import pretty_print_run, print_metrics, set_seed

# =============================================================================
# Builders
# =============================================================================

def build_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, np.ndarray, int, int]:
    samples = load_all_data_metaclass(
        data_dir=cfg['data']['data_dir'],
        subject_ids=cfg['data']['subject_ids'],
        task=cfg['data']['task'],
    )

    subject_mapping = create_subject_mapping(samples)
    n_subjects = len(subject_mapping)

    dataset = ChiscoDataset(
        samples=samples,
        normalize=cfg['data']['normalize'],
        exclude_channels=cfg['data'].get('exclude_channels'),
        subject_mapping=subject_mapping
    )
    labels = np.array([s['label'] for s in samples])
    splits = get_stratified_cv_splits(labels, cfg['cv']['n_splits'], cfg['cv']['random_state'])
    fold_index = int(cfg.get('cv', {}).get('fold_index', 0))
    fold_index = max(0, min(fold_index, len(splits) - 1))
    train_idx, val_idx = splits[fold_index]

    # Нормстаты
    if cfg['data']['normalize'] == 'zscore_hybrid':
        dataset.norm_stats = compute_hybrid_stats(samples, train_idx, cfg['data'].get('exclude_channels'))
    elif cfg['data']['normalize'] == 'zscore_subject_channel':
        dataset.norm_stats = compute_subjectwise_stats(samples, train_idx, cfg['data'].get('exclude_channels'))
    elif cfg['data']['normalize'] == 'zscore_dataset_channel':
        mean_c, std_c = compute_channelwise_stats(samples, train_idx, cfg['data'].get('exclude_channels'))
        dataset.norm_stats = {'mean': mean_c, 'std': std_c}

    train_loader = DataLoader(
        ChiscoSubset(dataset, train_idx),
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        pin_memory=cfg['training']['pin_memory']
    )
    val_loader = DataLoader(
        ChiscoSubset(dataset, val_idx),
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=cfg['training']['pin_memory']
    )

    eeg_shape = dataset[0]['eeg'].shape
    effective_channels = eeg_shape[0]

    return train_loader, val_loader, labels[train_idx], effective_channels, n_subjects


def build_model(cfg: Dict[str, Any], n_channels: int, n_subjects: int) -> nn.Module:
    m = cfg['model']
    return RTTMultiScale(
        n_channels=n_channels,
        n_classes=m['n_classes'],
        proj_channels=m['proj_channels'],
        window_size_small=m['window_size_small'], stride_small=m['stride_small'],
        window_size_large=m['window_size_large'], stride_large=m['stride_large'],
        d_model=m['d_model'], n_heads=m['n_heads'], ff_dim=m['ff_dim'],
        n_layers=m['n_layers'], dropout=m['dropout'], eps=m['eps'],
        attn_heads=m.get('attn_heads', 1),
        cov_type=m.get('cov_type', 'corr'), oas_min_alpha=m.get('oas_min_alpha', 0.1),
        use_subject_embed=m.get('use_subject_embed', False), n_subjects=n_subjects,
        subject_embed_dim=m.get('subject_embed_dim', 16),
        subject_embed_dropout=m.get('subject_embed_dropout', 0.0)
    )


def build_criterion(cfg: Dict[str, Any], train_labels: np.ndarray) -> nn.Module:
    loss_cfg = cfg['loss']
    if loss_cfg['type'] == 'cb_focal':
        counts = np.bincount(train_labels, minlength=cfg['model']['n_classes'])
        return ClassBalancedFocalLoss(counts, beta=loss_cfg['beta'], gamma=loss_cfg['gamma'])
    return nn.CrossEntropyLoss()


def build_optimizer_and_scheduler(model: nn.Module, cfg: Dict[str, Any]):
    opt_cfg = cfg['optimizer']
    train_cfg = cfg['training']

    params_subject, params_other = [], []
    if getattr(model, 'use_subject_embed', False):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (params_subject if name.startswith('subject_embed.') else params_other).append(p)
    else:
        params_other = [p for p in model.parameters() if p.requires_grad]

    subject_wd = opt_cfg.get('subject_embed_weight_decay', train_cfg['weight_decay'])
    param_groups = []
    if params_other:
        param_groups.append({'params': params_other, 'weight_decay': train_cfg['weight_decay']})
    if params_subject:
        param_groups.append({'params': params_subject, 'weight_decay': subject_wd})

    if opt_cfg['name'] == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=train_cfg['learning_rate'], betas=tuple(opt_cfg['betas']))
    else:
        optimizer = torch.optim.Adam(param_groups, lr=train_cfg['learning_rate'])

    sched_cfg = cfg['scheduler']
    scheduler = None
    if sched_cfg.get('name') == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup = int(sched_cfg.get('warmup_epochs', 0))
        if warmup > 0:
            sched1 = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
            sched2 = CosineAnnealingLR(optimizer, T_max=max(sched_cfg['T_max'] - warmup, 1))
            scheduler = SequentialLR(optimizer, schedulers=[sched1, sched2], milestones=[warmup])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=sched_cfg['T_max'])

    return optimizer, scheduler


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Phase 4B")
    parser.add_argument('--save-attn', action='store_true', help='Сохранять усреднённые attention веса на валидации')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config()
    cfg['logging']['save_attn'] = bool(args.save_attn)

    pretty_print_run(cfg)
    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])

    train_loader, val_loader, train_labels, n_channels, n_subjects = build_loaders(cfg)
    model = build_model(cfg, n_channels, n_subjects).to(device)
    criterion = build_criterion(cfg, train_labels).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    history, final_metrics, val_outputs, attn_stats = train_loop(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        cfg=cfg, device=device
    )

    print("\nИтоговые метрики валидации (Fold 1):")
    print_metrics(final_metrics)
    save_artifacts(cfg, final_metrics, history, val_outputs, attn_stats, model)
    print("\nОБУЧЕНИЕ PHASE 4B ЗАВЕРШЕНО")


if __name__ == '__main__':
    main()
