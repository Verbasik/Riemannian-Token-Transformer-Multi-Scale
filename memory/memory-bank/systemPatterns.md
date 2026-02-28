# System Patterns — EEG_to_Text

## Архитектурные паттерны

### 1. Modular Pipeline Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    Pipeline/                            │
├─────────────┬─────────────┬─────────────┬──────────────┤
│  config.py  │ data_loader │   model.py  │  trainer.py  │
│  (Config)   │  (Data)     │ (Model)     │ (Training)   │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬───────┘
       │             │             │             │
       └─────────────┴─────────────┴─────────────┘
                         │
                    train.py
                (Orchestration)
```

**Принципы**:
- Single Responsibility: каждый модуль отвечает за одну доменную область
- Dependency Injection: конфигурация передаётся явно
- Separation of Concerns: данные/модель/обучение разделены

### 2. Builder Pattern для конфигурации
```python
# config.py
def default_config(device_hint: Optional[str] = None) -> Dict[str, Any]:
    """Создаёт конфигурационный словарь по умолчанию."""
    use_cuda = torch.cuda.is_available() and (device_hint or 'cuda') == 'cuda'
    device = 'cuda' if use_cuda else 'cpu'
    
    return {
        'data': {...},
        'model': {...},
        'training': {...},
        'cv': {...},
        'optimizer': {...},
        'scheduler': {...},
        'loss': {...},
        'device': device,
        ...
    }
```

**Преимущества**:
- Централизованное управление параметрами
- Автоматическая адаптация под устройство
- Версионность конфигураций

### 3. Strategy Pattern для нормализации
```python
# data_loader.py
if self.normalize == 'zscore':
    # Standard z-score
elif self.normalize == 'minmax':
    # Min-Max scaling
elif self.normalize == 'zscore_subject_channel':
    # Subject-wise normalization
elif self.normalize == 'zscore_hybrid':
    # Hybrid: subject centering + global scaling
```

**Преимущества**:
- Легко добавлять новые стратегии
- Явный выбор стратегии в конфиге
- Изоляция логики каждой стратегии

### 4. Factory Pattern для компонентов обучения
```python
# train.py
def build_loaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, ...]:
    """Factory для DataLoader"""
    
def build_model(cfg: Dict, n_channels: int, n_subjects: int) -> nn.Module:
    """Factory для модели"""
    
def build_criterion(cfg: Dict, train_labels: np.ndarray) -> nn.Module:
    """Factory для loss функции"""
    
def build_optimizer_and_scheduler(model, cfg) -> Tuple[Optimizer, Scheduler]:
    """Factory для оптимизатора и scheduler"""
```

### 5. Callback Pattern для early stopping
```python
# trainer.py
if val_metrics['f1_macro'] > best_f1:
    best_f1 = val_metrics['f1_macro']
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience = 0
else:
    patience += 1
    if patience >= cfg['training']['early_stopping_patience']:
        print(f"\nРанняя остановка на эпохе {epoch+1}")
        break
```

## Паттерны обработки данных

### 1. SPD Manifold Processing
```
EEG Signal → Covariance (OAS) → Correlation → Logarithm → Vectorize
   [C,T]        [C,C] SPD         [C,C] SPD      [C,C]       [C(C+1)/2]
```

**Ключевые операции**:
```python
# cov_shrinkage_oas: Оценка ковариации с регуляризацией
Σ_OAS = (1-α) * Σ_sample + α * μ * I
где α = clip((φ + μ²) / ((T-1)(φ - tr(Σ)²/C) + 1), 0.1, 1.0)

# spd_correlation_from_cov: Нормализация к корреляции
D = diag(Σ)^(-1/2)
R = D @ Σ @ D

# spd_logm: Логарифмическое отображение
Σ = V @ diag(λ) @ V^T
log(Σ) = V @ diag(log(λ)) @ V^T

# spd_vectorize: Извлечение верхнего треугольника
vec(Σ) = [Σ_ij для i≤j]
```

### 2. Multi-Scale Token Generation
```python
def _tokens_for_scale(x, w, s, scale_id):
    # Channel projection
    x_pc = channel_proj(x.transpose(1,2)).transpose(1,2)
    
    # Windowing
    x_win = window_signal(x_pc, w, s)  # [B, L, C, T_w]
    
    # SPD processing
    cov = cov_shrinkage_oas(x_win.reshape(B*L, C, T_w))
    vec = spd_vectorize(spd_logm(cov))
    
    # Feature projection + scale embedding
    tok = feature_proj(vec).view(B, L, -1)
    return tok + scale_emb[scale_id]
```

### 3. Attention Pooling
```python
# Weighted aggregation токенов
scores = attn_pool_W(toks)           # [B, L, H]
weights_tok = softmax(scores, dim=1) # [B, L, H]
h_heads = einsum('blh,bld->bhd')     # [B, H, D]

# Head aggregation
head_alpha = softmax(head_weights)   # [H]
h_attn = einsum('h,bhd->bd')         # [B, D]
```

### 4. Subject Embedding Integration
```python
# Embedding lookup
subject_emb = subject_embed(subject_ids)  # [B, dim]
subject_emb = subject_embed_drop(subject_emb)

# Concatenation
combined = cat([h_cls, h_attn, subject_emb], dim=-1)
logits = head(combined)
```

## Паттерны обучения

### 1. Class-Balanced Focal Loss
```python
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_counts, beta, gamma):
        # Effective number sampling
        effective_num = 1 - β^counts
        α = (1-β) / effective_num  # Class weights
        
    def forward(self, logits, targets):
        probs = softmax(logits)
        pt = probs[targets]
        focal_weight = (1-pt)^γ
        loss = -α[targets] * focal_weight * log(pt)
        return loss.mean()
```

### 2. Differentiated Weight Decay
```python
# Раздельный weight_decay для subject embeddings
params_subject = [p for n,p in model.named_parameters() 
                  if n.startswith('subject_embed.')]
params_other = [p for n,p in model.named_parameters() 
                if not n.startswith('subject_embed.')]

optimizer = AdamW([
    {'params': params_other, 'weight_decay': 1e-4},
    {'params': params_subject, 'weight_decay': 5e-4}
])
```

### 3. Warmup + Cosine Annealing
```python
# Linear warmup
scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=3)

# Cosine annealing
scheduler2 = CosineAnnealingLR(optimizer, T_max=17)

# Sequential composition
scheduler = SequentialLR(optimizer, [sched1, sched2], milestones=[3])
```

### 4. Gradient Clipping + AMP
```python
scaler = GradScaler(enabled=use_amp)

for batch in loader:
    with autocast(enabled=use_amp):
        logits = model(eeg, subject_ids)
        loss = criterion(logits, labels)
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    
    # Gradient norms logging
    grad_norms = [p.grad.norm() for p in model.parameters()]
    
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

## Паттерны анализа

### 1. Artifact Collection
```python
def save_artifacts(cfg, metrics, history, val_outputs, attn_stats, model):
    # Model checkpoint
    torch.save(model.state_dict(), 'best_model.pt')
    
    # Predictions with metadata
    np.savez('val_preds.npz', 
             y_true, y_pred, proba, 
             subject_id, sample_id)
    
    # Attention statistics
    np.savez('attn_stats.npz', 
             weights_tok_mean, head_weights, scale_lengths)
    
    # Metrics and config
    json.dump(metrics, 'metrics.json')
    json.dump(history, 'history.json')
    json.dump(cfg, 'config_run.json')
```

### 2. Per-Subject Analysis
```python
# analysis_tools/subject_effects.py
# Вычисление per-subject метрик для анализа обобщения
for subject_id in unique_subjects:
    mask = subject_ids == subject_id
    subject_metrics = compute_metrics(y_true[mask], y_pred[mask])
```

### 3. Attention Visualization
```python
# analysis_tools/attention_stats.py
# Визуализация attention весов по каналам и масштабам
weights_tok_mean: [L, H]  # Средние веса по токенам
head_weights: [H]         # Веса heads
scale_lengths: (L_s, L_l) # Число токенов по масштабам
```

## Конвенции кода

### Именование
```python
# Модули: snake_case (data_loader.py)
# Классы: PascalCase (RTTMultiScale, ChiscoDataset)
# Функции: snake_case (compute_metrics, build_loaders)
# Константы: UPPER_CASE (RANDOM_SEED, EPSILON)
# Приватные: _prefix ( _eigh_cpu_fallback)
```

### Типизация
```python
from typing import Dict, List, Optional, Tuple, Any, Callable

def func(
    arg1: int,
    arg2: Optional[str] = None,
    arg3: Dict[str, Any] = {}
) -> Tuple[float, Dict]:
    ...
```

### Документирование
```python
"""
Description:
---------------
    Краткое описание назначения функции.

Args:
---------------
    arg1: Описание аргумента.
    arg2: Описание аргумента.

Returns:
---------------
    Описание возвращаемого значения.
"""
```
