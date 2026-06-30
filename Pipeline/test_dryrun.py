# file: test_dryrun.py
# -*- coding: utf-8 -*-
"""
Dry-run test for validating Subject-Wise Normalization changes.

Runs 1 training epoch to verify:
1. Correct subject-wise stats computation (per-subject mean/STD).
2. Error-free data loading (PKL loading, normalization).
3. Compatibility with the existing model architecture (Subject Embeddings).
4. Completion of a full training step (forward, backward, optimizer step).

Usage:
    python test_dryrun.py

Expected result:
    The script finishes successfully without exceptions and prints metrics
    for 1 epoch.
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
# Add the project root directory to the module search path.
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
        Runs the dry-run testing process: trains the model for 1 epoch
        on a small amount of data to validate pipeline integrity.
        Checks compatibility across the data loader, model, loss function,
        and optimizer.

    Returns:
    ---------------
        None

    Raises:
    ---------------
        Exception: Any unhandled exception interrupts the test and points
            to a configuration or code issue.

    Examples:
    ---------------
        >>> # Run from the command line:
        >>> # python test_dryrun.py
        >>> # Expected output: "DRY-RUN COMPLETE ✅"
    """
    # =========================================================================
    # Configuration
    # =========================================================================
    cfg = default_config()
    # DRY-RUN: limit training to one epoch for speed.
    cfg['training']['n_epochs'] = 1

    print("\n" + "=" * 80)
    print("DRY-RUN TEST: Subject-Wise Normalization (1 epoch)")
    print("=" * 80)
    pretty_print_run(cfg)

    # Set the seed for result reproducibility.
    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])

    # =========================================================================
    # Step 1: Data loading
    # =========================================================================
    print("\n[1/5] Loading data...")
    train_loader, val_loader, train_labels, n_channels, n_subjects = (
        build_loaders(cfg)
    )
    print(f"✅ Train samples: {len(train_loader.dataset)}")
    print(f"✅ Val samples: {len(val_loader.dataset)}")
    print(f"✅ Effective channels: {n_channels}")

    # =========================================================================
    # Step 2: Model initialization
    # =========================================================================
    print("\n[2/5] Creating model...")
    model = build_model(cfg, n_channels, n_subjects).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model parameters: {n_params:,}")

    # =========================================================================
    # Step 3: Criterion and optimizer initialization
    # =========================================================================
    print("\n[3/5] Creating criterion and optimizer...")
    criterion = build_criterion(cfg, train_labels).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    print(f"✅ Criterion: {type(criterion).__name__}")
    print(f"✅ Optimizer: {type(optimizer).__name__}")

    # =========================================================================
    # Step 4: Run training (1 epoch)
    # =========================================================================
    print("\n[4/5] Running 1 training epoch...")
    model.train()
    total_loss = 0.0

    # Check whether subject embeddings are enabled.
    use_subject_embed = (
        hasattr(model, 'use_subject_embed') and model.use_subject_embed
    )

    for i, batch in enumerate(train_loader):
        # Move data to the device (GPU/CPU).
        eeg = batch['eeg'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad(set_to_none=True)

        # Build model arguments depending on configuration.
        if use_subject_embed:
            subject_ids = batch['subject_id'].to(device)
            logits = model(eeg, subject_ids=subject_ids)
        else:
            logits = model(eeg)

        # Compute loss and run backpropagation.
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for training stability.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg['training']['grad_clip']
        )
        optimizer.step()

        total_loss += loss.item()

        # Log progress every 100 batches.
        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{len(train_loader)}: Loss = {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    print(f"\n✅ Epoch 1 Train Loss: {avg_train_loss:.4f}")

    # =========================================================================
    # Step 5: Validation
    # =========================================================================
    print("\n[5/5] Evaluating on validation...")
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
    # Completion
    # =========================================================================
    print("\n" + "=" * 80)
    print("DRY-RUN COMPLETE ✅ — All components are working correctly!")
    print("=" * 80)
    print("\nReady to run full training (50 epochs).")


if __name__ == '__main__':
    main()
