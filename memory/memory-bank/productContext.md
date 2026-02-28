# Product Context — EEG_to_Text

## Проблемная область

### Научный контекст
Проект решает задачу **декодирования воображаемой речи** — одной из сложнейших проблем в области brain-computer interfaces (BCI).

### Ключевые вызовы
1. **Низкое отношение сигнал/шум**: ЭЭГ-сигналы при воображаемой речи имеют амплитуду ~1-10 μV
2. **Межсубъектная вариабельность**: Различия в анатомии, impedance электродов, cognitive strategies
3. **Классовый дисбаланс**: Неравномерное распределение сэмплов между классами
4. **Высокая размерность**: 125 каналов × временные точки при ограниченном числе сэмплов

## Решения

### 1. Риманова геометрия для SPD матриц
**Проблема**: Ковариационные матрицы ЭЭГ лежат на SPD многообразии (не Евклидово пространство)

**Решение**:
- Оценка ковариации с OAS (Oracle Approximating Shrinkage)
- Логарифмическое отображение в касательное пространство (Log-Euclidean map)
- Векторизация верхнего треугольника SPD матриц

### 2. Двухмасштабные токены
**Проблема**: Речевые паттерны существуют на разных временных масштабах

**Решение**:
- Малые окна: 128 samples / stride 96 (высокое временное разрешение)
- Большие окна: 256 samples / stride 128 (контекстная информация)
- Attention pooling для агрегации токенов

### 3. Subject Embeddings
**Проблема**: Межсубъектная вариабельность снижает качество обобщения

**Решение**:
- Learnable embeddings для каждого субъекта (dim=16)
- Dropout 0.2 для регуляризации
- Отдельный weight_decay (5e-4) для subject embeddings

### 4. Гибридная нормализация
**Проблема**: Subject baseline shifts при сохранении discriminative signal

**Решение** (Phase 4B-6):
```
Step 1: Subject-wise centering
  X̂ᵢ_centered = Xᵢ - μᵢ  (устраняет baseline shifts)

Step 2: Global scaling
  X̂_final = X̂_pooled / σ_global  (сохраняет inter-subject variance)
```

### 5. Class-Balanced Focal Loss
**Проблема**: Дисбаланс классов + hard examples

**Решение**:
```
CB-Focal(β=0.999, γ=1.75):
  - α balancing по effective number of samples
  - Focal weighting для hard examples
```

## Архитектурные решения

### RTTMultiScale Architecture
```
EEG [B, 125, T]
    ↓
Channel Projection → [B, T, 24]
    ↓
┌─────────────────────────────────┐
│  Small Scale Tokens (128/96)    │ → SPD → logm → vectorize → [B, L_s, 128]
│  Large Scale Tokens (256/128)   │ → SPD → logm → vectorize → [B, L_l, 128]
└─────────────────────────────────┘
    ↓
Scale Embedding + Positional Encoding
    ↓
Transformer Encoder (2 layers, 4 heads)
    ↓
[CLS] Token + Attention Pooling (1 head)
    ↓
Concat + Subject Embedding
    ↓
Classification Head → 8 классов
```

### SPD Processing Pipeline
```
Windowed EEG [B, L, C, T_w]
    ↓
Covariance (OAS shrinkage, α∈[0.1, 1.0])
    ↓
Correlation normalization (если cov_type='corr')
    ↓
Matrix Logarithm (eigendecomposition с fallback)
    ↓
Vectorization (upper triangular) → [B, L, d_model]
```

## Метрики и результаты

### Текущие (Phase 4B, sub-03, Fold 1)
- **F1-macro**: 0.2182
- **Accuracy**: 0.2399
- **Balanced Accuracy**: 0.2297
- **Loss**: 1.3933

### Исторический лучший результат (A6, sub-04, Fold 1)
- **F1-macro**: ~0.2915
- Конфигурация: subject embeddings ON, cov_type='corr', окна 128/256

## Направления исследований

### Приоритетные
1. **5-fold кросс-валидация** на всех 5 субъектах
2. **Абляции**: subject embeddings on/off, cov_type, window sizes
3. **Гиперпараметры**: lr, T_max, weight_decay, β/γ для CB-Focal
4. **Transfer learning**: обобщение на новых субъектов (LOSO)

### Перспективные
- Data augmentation в SPD касательном пространстве
- Multi-subject pre-training + fine-tuning
- Attention visualization для интерпретируемости
