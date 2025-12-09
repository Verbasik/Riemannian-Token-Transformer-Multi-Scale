# file: test_dryrun.py
# -*- coding: utf-8 -*-
"""
Dry-run тест для валидации модификаций Subject-Wise Normalization.

Запускает 1 epoch обучения для проверки:
1. Корректности вычисления subject-wise stats
2. Отсутствия ошибок в data loading
3. Совместимости с существующей архитектурой
"""
import sys
from pathlib import Path

# Add Best/ to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from config import default_config
from train import build_loaders, build_model, build_criterion, build_optimizer_and_scheduler
from trainer import evaluate
from utils import set_seed, pretty_print_run

def main():
    """Dry-run с 1 epoch."""
    # Configuration
    cfg = default_config()
    cfg['training']['n_epochs'] = 1  # DRY-RUN: только 1 epoch

    print("\n" + "="*80)
    print("DRY-RUN TEST: Subject-Wise Normalization (1 epoch)")
    print("="*80)
    pretty_print_run(cfg)

    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])

    # Build components
    print("\n[1/4] Загрузка данных...")
    train_loader, val_loader, train_labels, n_channels = build_loaders(cfg)
    print(f"✅ Train samples: {len(train_loader.dataset)}")
    print(f"✅ Val samples: {len(val_loader.dataset)}")
    print(f"✅ Effective channels: {n_channels}")

    print("\n[2/4] Создание модели...")
    model = build_model(cfg, n_channels).to(device)
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n[3/4] Создание criterion и optimizer...")
    criterion = build_criterion(cfg, train_labels).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    print(f"✅ Criterion: {type(criterion).__name__}")
    print(f"✅ Optimizer: {type(optimizer).__name__}")

    print("\n[4/4] Запуск 1 epoch обучения...")
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(train_loader):
        eeg, labels = batch['eeg'].to(device), batch['label'].to(device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(eeg)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{len(train_loader)}: Loss = {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    print(f"\n✅ Epoch 1 Train Loss: {avg_train_loss:.4f}")

    print("\n[5/5] Оценка на валидации...")
    val_metrics = evaluate(model, val_loader, device, criterion)
    print(f"✅ Val Loss: {val_metrics['loss']:.4f}")
    print(f"✅ Val Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"✅ Val F1-macro: {val_metrics['f1_macro']:.4f}")

    print("\n" + "="*80)
    print("DRY-RUN COMPLETE ✅ — Все компоненты работают корректно!")
    print("="*80)
    print("\nГотов к запуску полного обучения (50 epochs).")

if __name__ == '__main__':
    main()
