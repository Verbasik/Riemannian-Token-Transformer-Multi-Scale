# Лучшая конфигурация (DS1: sub-04, Fold 1)

- Loss: Class-Balanced Focal Loss
  - `gamma = 1.75`
  - `beta = 0.999`
- Scheduler: CosineAnnealingLR
  - `T_max = 20`
  - `warmup_epochs = 3`
- Model:
  - `gating = false`
  - `attn_heads = 1`
  - `subject_embed_dim = 16`
  - `subject_embed_dropout = 0.2`
- Optimizer:
  - `name = adamw`
  - `betas = (0.9, 0.999)`
  - `weight_decay (base) = 1e-4`
  - `subject_embed_weight_decay = 5e-4`
- Data/Train:
  - `normalize = zscore_hybrid`
  - `exclude_channels = [124]`
  - `n_epochs = 50`
  - `batch_size = 16 (cuda) / 8 (cpu)`
  - `use_weighted_sampler = false` (при CB-Focal)

Опция (A8 — кандидат): включить SPD-аугментацию в касательном пространстве
  - `use_spd_augment = true`
  - `spd_jitter_std = 0.03`
  - `spd_jitter_prob = 0.2`

Примечание: базовая комбинация и A8-опция подтверждены на Fold 1; перед фиксацией в дефолтах рекомендуется 5-fold проверка на DS1 и проверка на DS2.
