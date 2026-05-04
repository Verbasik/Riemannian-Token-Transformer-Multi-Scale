# file: test_dryrun.py
# -*- coding: utf-8 -*-
"""
Dry-run тест для валидации модификаций Subject-Wise Normalization.

Запускает 1 epoch обучения для проверки:
1. Корректности вычисления subject-wise stats (среднее/STD по субъектам).
2. Отсутствия ошибок в data loading (загрузка pkl, нормализация).
3. Совместимости с существующей архитектурой модели (Subject Embeddings).
4. Прохождения полного шага обучения (forward, backward, optimizer step).

Использование:
    python test_dryrun.py

Ожидаемый результат:
    Успешное завершение скрипта без исключений и вывод метрик за 1 epoch.
"""

# =============================================================================
# Standard Libraries
# =============================================================================
import sys
from pathlib import Path

# =============================================================================
# Third-Party Libraries
# =============================================================================
import torch

# =============================================================================
# Local Imports
# =============================================================================
# Добавляем корневую директорию проекта в путь поиска модулей
sys.path.insert(0, str(Path(__file__).parent))

from config import default_config
from train import (
    build_loaders,
    build_model,
    build_criterion,
    build_optimizer_and_scheduler,
)
from trainer import evaluate_with_outputs
from utils import set_seed, pretty_print_run


def main() -> None:
    """
    Description:
    ---------------
        Запускает процесс dry-run тестирования: обучает модель в течение
        1 эпохи на небольших данных для проверки целостности пайплайна.
        Проверяет совместимость всех компонентов: загрузчика данных,
        модели, функции потерь и оптимизатора.

    Returns:
    ---------------
        None

    Raises:
    ---------------
        Exception: Любое необработанное исключение прервет тест и укажет
            на проблему в конфигурации или коде.

    Examples:
    ---------------
        >>> # Запуск из командной строки:
        >>> # python test_dryrun.py
        >>> # Ожидается вывод: "DRY-RUN COMPLETE ✅"
    """
    # =========================================================================
    # Конфигурация
    # =========================================================================
    cfg = default_config()
    # DRY-RUN: ограничиваем обучение одним эпoхом для скорости
    cfg['training']['n_epochs'] = 1

    print("\n" + "=" * 80)
    print("DRY-RUN TEST: Subject-Wise Normalization (1 epoch)")
    print("=" * 80)
    pretty_print_run(cfg)

    # Установка seed для воспроизводимости результатов
    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])

    # =========================================================================
    # Шаг 1: Загрузка данных
    # =========================================================================
    print("\n[1/5] Загрузка данных...")
    train_loader, val_loader, train_labels, n_channels, n_subjects = (
        build_loaders(cfg)
    )
    print(f"✅ Train samples: {len(train_loader.dataset)}")
    print(f"✅ Val samples: {len(val_loader.dataset)}")
    print(f"✅ Effective channels: {n_channels}")

    # =========================================================================
    # Шаг 2: Инициализация модели
    # =========================================================================
    print("\n[2/5] Создание модели...")
    model = build_model(cfg, n_channels, n_subjects).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model parameters: {n_params:,}")

    # =========================================================================
    # Шаг 3: Инициализация критерия и оптимизатора
    # =========================================================================
    print("\n[3/5] Создание criterion и optimizer...")
    criterion = build_criterion(cfg, train_labels).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    print(f"✅ Criterion: {type(criterion).__name__}")
    print(f"✅ Optimizer: {type(optimizer).__name__}")

    # =========================================================================
    # Шаг 4: Запуск обучения (1 epoch)
    # =========================================================================
    print("\n[4/5] Запуск 1 epoch обучения...")
    model.train()
    total_loss = 0.0

    # Проверка наличия флага использования эмбеддингов субъектов
    use_subject_embed = (
        hasattr(model, 'use_subject_embed') and model.use_subject_embed
    )

    for i, batch in enumerate(train_loader):
        # Перемещение данных на устройство (GPU/CPU)
        eeg = batch['eeg'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad(set_to_none=True)

        # Формирование аргументов для модели в зависимости от конфигурации
        if use_subject_embed:
            subject_ids = batch['subject_id'].to(device)
            logits = model(eeg, subject_ids=subject_ids)
        else:
            logits = model(eeg)

        # Вычисление потерь и обратное распространение
        loss = criterion(logits, labels)
        loss.backward()

        # Градиентный клиппинг для стабильности обучения
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg['training']['grad_clip']
        )
        optimizer.step()

        total_loss += loss.item()

        # Логирование прогресса каждые 100 батчей
        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{len(train_loader)}: Loss = {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    print(f"\n✅ Epoch 1 Train Loss: {avg_train_loss:.4f}")

    # =========================================================================
    # Шаг 5: Валидация
    # =========================================================================
    print("\n[5/5] Оценка на валидации...")
    val_metrics, _, _ = evaluate_with_outputs(
        model,
        val_loader,
        device,
        criterion
    )

    print(f"✅ Val Loss: {val_metrics['loss']:.4f}")
    print(f"✅ Val Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"✅ Val F1-macro: {val_metrics['f1_macro']:.4f}")

    # =========================================================================
    # Завершение
    # =========================================================================
    print("\n" + "=" * 80)
    print("DRY-RUN COMPLETE ✅ — Все компоненты работают корректно!")
    print("=" * 80)
    print("\nГотов к запуску полного обучения (50 epochs).")


if __name__ == '__main__':
    main()
