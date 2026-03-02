# file: model.py
# -*- coding: utf-8 -*-
"""
Базовая модель RTTMultiScale с optional subject embeddings и возможностью
вернуть attention статистики.

Модель использует многомасштабное разложение сигнала, римановы признаки
(SPD матрицы) и трансформер для классификации EEG сигналов.
Поддерживает внедрение эмбеддингов субъектов для улучшения обобщаемости.
"""

# =============================================================================
# Standard Libraries
# =============================================================================
import math
from typing import Any, Dict, Optional, Tuple, Union

# =============================================================================
# Third-Party Libraries
# =============================================================================
import torch
import torch.nn as nn

# =============================================================================
# Local Imports
# =============================================================================
from riemannian_utils import (
    cov_shrinkage_oas,
    spd_correlation_from_cov,
    spd_logm,
    spd_vectorize,
    window_signal,
)


# =============================================================================
# Позиционное кодирование (Positional Encoding)
# =============================================================================

class SinusoidalPE(nn.Module):
    """
    Description:
    ---------------
        Синусоидальное позиционное кодирование для добавления информации
        о порядке временных шагов во входную последовательность.
        Использует фиксированные частоты синусов и косинусов.

    Args:
    ---------------
        d_model: int - Размерность модели (глубина эмбеддинга).
        dropout: float - Вероятность обнуления элементов (по умолчанию 0.1).
        max_len: int - Максимальная длина последовательности (по умолчанию 1024).

    Returns:
    ---------------
        Tensor с добавленным позиционным кодированием.

    Raises:
    ---------------
        Нет явных исключений.

    Examples:
    ---------------
        >>> pe = SinusoidalPE(d_model=64)
        >>> x = torch.randn(2, 10, 64)  # [batch, seq_len, d_model]
        >>> out = pe(x)
        >>> out.shape
        torch.Size([2, 10, 64])
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 1024
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Создаем матрицу позиционных кодирований
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Вычисляем логарифмические шаги частот
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # Заполняем четные индексы синусом, нечетные - косинусом
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Регистрируем как буфер (не обновляется градиентом)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
        ---------------
            Добавляет позиционное кодирование к входному тензору и применяет
            dropout.

        Args:
        ---------------
            x: torch.Tensor [B, L, d_model] - Входная последовательность.

        Returns:
        ---------------
            torch.Tensor [B, L, d_model] - Тензор с позиционным кодированием.
        """
        L = x.size(1)
        return self.dropout(x + self.pe[:L, :])


# =============================================================================
# Основная модель (RTTMultiScale)
# =============================================================================

class RTTMultiScale(nn.Module):
    """
    Description:
    ---------------
        Многомасштабная модель на основе трансформера для классификации EEG.
        Использует риманову геометрию (SPD матрицы) для извлечения признаков
        из скользящих окон разного размера.

        Архитектура:
        1. Проекция каналов.
        2. Разбиение на окна (мелкие и крупные).
        3. Вычисление ковариационных матриц и их логарифмическое отображение.
        4. Векторизация и проекция в пространство трансформера.
        5. Кодирование масштаба и позиционное кодирование.
        6. Трансформер-энкодер с механизмом внимания.
        7. Attention pooling и классификация.
        8. Опционально: эмбеддинги субъектов.

    Args:
    ---------------
        n_channels: int - Количество каналов EEG.
        n_classes: int - Количество классов для классификации.
        proj_channels: int - Размерность после проекции каналов.
        window_size_small: int - Размер малого окна (высокое разрешение).
        stride_small: int - Шаг малого окна.
        window_size_large: int - Размер большого окна (низкое разрешение).
        stride_large: int - Шаг большого окна.
        d_model: int - Размерность пространства трансформера.
        n_heads: int - Количество голов внимания.
        ff_dim: int - Размерность feed-forward сети внутри трансформера.
        n_layers: int - Количество слоев трансформера.
        dropout: float - Коэффициент dropout.
        eps: float - Малое число для численной стабильности.
        attn_heads: int - Количество голов для attention pooling.
        cov_type: str - Тип нормализации ковариации ('corr' или 'trace').
        oas_min_alpha: float - Минимальный коэффициент сжатия OAS.
        use_subject_embed: bool - Использовать ли эмбеддинги субъектов.
        n_subjects: int - Общее количество субъектов (для Embedding слоя).
        subject_embed_dim: int - Размерность эмбеддинга субъекта.
        subject_embed_dropout: float - Dropout для эмбеддинга субъекта.

    Returns:
    ---------------
        logits: torch.Tensor - Логиты классов.
        attn_stats: Dict (опционально) - Статистики внимания.

    Raises:
    ---------------
        ValueError: Если use_subject_embed=True, но subject_ids не переданы.

    Examples:
    ---------------
        >>> model = RTTMultiScale(
        ...     n_channels=22, n_classes=4, proj_channels=16,
        ...     window_size_small=50, stride_small=25,
        ...     window_size_large=200, stride_large=100,
        ...     d_model=64, n_heads=4, ff_dim=128, n_layers=2,
        ...     dropout=0.1, eps=1e-6, attn_heads=4, cov_type='corr'
        ... )
        >>> x = torch.randn(2, 22, 1000)  # [batch, channels, time]
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([2, 4])
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        proj_channels: int,
        window_size_small: int,
        stride_small: int,
        window_size_large: int,
        stride_large: int,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        n_layers: int,
        dropout: float,
        eps: float,
        attn_heads: int,
        cov_type: str,
        oas_min_alpha: float = 0.1,
        use_subject_embed: bool = False,
        n_subjects: int = 5,
        subject_embed_dim: int = 16,
        subject_embed_dropout: float = 0.0,
    ):
        super().__init__()

        # Сохранение параметров окон и нормализации
        self.ws_s = window_size_small
        self.st_s = stride_small
        self.ws_l = window_size_large
        self.st_l = stride_large
        self.eps = eps
        self.cov_type = cov_type
        self.oas_min_alpha = oas_min_alpha
        self.use_subject_embed = use_subject_embed

        # Инициализация эмбеддингов субъектов (опционально)
        if self.use_subject_embed:
            self.subject_embed = nn.Embedding(n_subjects, subject_embed_dim)
            # Используем Identity, если dropout <= 0, чтобы избежать лишних вычислений
            if subject_embed_dropout and subject_embed_dropout > 0:
                self.subject_embed_drop = nn.Dropout(subject_embed_dropout)
            else:
                self.subject_embed_drop = nn.Identity()
            classifier_input_dim = d_model * 2 + subject_embed_dim
        else:
            classifier_input_dim = d_model * 2

        # Проекция каналов (линейное преобразование без смещения)
        self.channel_proj = nn.Linear(
            n_channels, proj_channels, bias=False
        )

        # Размерность вектора SPD матрицы: N*(N+1)/2
        spd_vec_dim = proj_channels * (proj_channels + 1) // 2
        self.feature_proj = nn.Linear(spd_vec_dim, d_model)

        # Обучаемые эмбеддинги для различения масштабов (малый/большой)
        self.scale_emb = nn.Parameter(torch.zeros(2, d_model))

        # Токен классификации [CLS]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Позиционное кодирование
        self.pos_enc = SinusoidalPE(d_model, dropout)

        # Слои трансформера
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN архитектура для стабильности
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Механизм Attention Pooling
        self.attn_pool_W = nn.Linear(d_model, attn_heads)
        self.head_weights = nn.Parameter(torch.zeros(attn_heads))

        # Классификатор (MLP Head)
        self.head = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )

        # Инициализация весов
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Description:
        ---------------
            Инициализирует веса модели.
            - Линейные слои: инициализация Xavier Uniform.
            - Смещения (bias): нули.
            - Learnable параметры (cls_token, scale_emb): Normal(0, 0.02).

        Returns:
        ---------------
            None
        """
        for name, param in self.named_parameters():
            if param.dim() > 1:
                # Для матриц весов используем Xavier
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Смещения обнуляем
                nn.init.zeros_(param)

        # Специальная инициализация для токенов и эмбеддингов масштаба
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.scale_emb, std=0.02)

    def _tokens_for_scale(
        self,
        x: torch.Tensor,
        w: int,
        s: int,
        scale_id: int
    ) -> torch.Tensor:
        """
        Description:
        ---------------
            Генерирует токены признаков для заданного масштаба окна.
            Процесс:
            1. Проекция каналов.
            2. Нарезка на перекрывающиеся окна.
            3. Вычисление SPD ковариационной матрицы для каждого окна.
            4. Нормализация (корреляция или след).
            5. Логарифмическое отображение (Log-Euclidean metric).
            6. Векторизация верхней треугольной части.
            7. Проекция в d_model и добавление эмбеддинга масштаба.

        Args:
        ---------------
            x: torch.Tensor [B, C, T] - Входной сигнал.
            w: int - Размер окна.
            s: int - Шаг окна.
            scale_id: int - Индекс масштаба (0 для малого, 1 для большого).

        Returns:
        ---------------
            torch.Tensor [B, L, d_model] - Последовательность токенов.
        """
        # Проекция каналов: [B, C, T] -> [B, Proj, T]
        x_pc = self.channel_proj(x.transpose(1, 2)).transpose(1, 2)

        # Нарезка на окна: [B, Proj, T] -> [B, L, Proj, W]
        x_win = window_signal(x_pc, w, s)

        B, L, c, _ = x_win.shape

        # Flatten для вычисления ковариации: [B*L, Proj, W]
        x_flat = x_win.reshape(B * L, c, -1)

        # Вычисление ковариации с OAS shrinkage
        cov = cov_shrinkage_oas(
            x_flat,
            eps=self.eps,
            min_alpha=self.oas_min_alpha
        )

        # Нормализация ковариационной матрицы
        if self.cov_type == 'corr':
            # Преобразование в матрицу корреляций
            cov = spd_correlation_from_cov(cov, eps=self.eps)
        else:
            # Нормализация по следу (Trace Normalization)
            tr = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
            cov = cov / tr.clamp(min=self.eps).unsqueeze(-1)

        # Log-Euclidean mapping: Log(Cov)
        log_cov = spd_logm(cov, eps=self.eps)

        # Векторизация верхней треугольной части
        vec = spd_vectorize(log_cov)

        # Проекция в пространство трансформера
        tok = self.feature_proj(vec).view(B, L, -1)

        # Добавление эмбеддинга масштаба
        return tok + self.scale_emb[scale_id]

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Description:
        ---------------
            Прямой проход модели.
            1. Извлечение признаков для двух масштабов.
            2. Конкатенация токенов и добавление [CLS].
            3. Кодирование позиции и прогон через трансформер.
            4. Attention pooling токенов.
            5. Объединение [CLS] и pooled вектора.
            6. Добавление эмбеддинга субъекта (если включено).
            7. Классификация.

        Args:
        ---------------
            x: torch.Tensor [B, C, T] - Входной EEG сигнал.
            subject_ids: Optional[torch.Tensor] [B] - ID субъектов.
                Требуется, если use_subject_embed=True.
            return_attn: bool - Если True, возвращает также статистики внимания.

        Returns:
        ---------------
            logits: torch.Tensor [B, n_classes] - Предсказания модели.
            attn_stats: Dict (только если return_attn=True):
                - 'weights_tok_mean': Средние веса внимания по токенам.
                - 'head_weights': Веса голов внимания.
                - 'scale_lengths': Длины последовательностей масштабов.

        Raises:
        ---------------
            ValueError: Если use_subject_embed=True, а subject_ids=None.
        """
        # Извлечение токенов для малого масштаба
        t_s = self._tokens_for_scale(x, self.ws_s, self.st_s, 0)

        # Извлечение токенов для большого масштаба
        t_l = self._tokens_for_scale(x, self.ws_l, self.st_l, 1)

        # Конкатенация последовательностей масштабов
        tokens = torch.cat([t_s, t_l], dim=1)

        # Подготовка [CLS] токена для батча
        cls = self.cls_token.expand(x.size(0), -1, -1)

        # Добавление позиции и прогон через энкодер
        seq = self.pos_enc(torch.cat([cls, tokens], dim=1))
        h = self.encoder(seq)

        # Разделение [CLS] и токенов последовательности
        h_cls = h[:, 0]       # [B, d_model]
        toks = h[:, 1:]       # [B, L, d_model]

        # Attention Pooling: вычисление весов для каждого токена
        scores = self.attn_pool_W(toks)  # [B, L, H]
        weights_tok = torch.softmax(scores, dim=1)

        # Взвешенная сумма токенов для каждой головы внимания
        # einsum: (batch, len, heads) x (batch, len, dim) -> (batch, heads, dim)
        h_heads = torch.einsum('blh,bld->bhd', weights_tok, toks)

        # Глобальные веса для объединения голов
        head_alpha = torch.softmax(self.head_weights, dim=0)

        # Финальное объединение голов: (heads) x (batch, heads, dim) -> (batch, dim)
        h_attn = torch.einsum('h,bhd->bd', head_alpha, h_heads)

        # Объединение представления [CLS] и Attention-pooled вектора
        combined = torch.cat([h_cls, h_attn], dim=-1)

        # Добавление эмбеддинга субъекта, если требуется
        if self.use_subject_embed:
            if subject_ids is None:
                raise ValueError(
                    "subject_ids required when use_subject_embed=True"
                )
            subject_emb = self.subject_embed(subject_ids)
            subject_emb = self.subject_embed_drop(subject_emb)
            combined = torch.cat([combined, subject_emb], dim=-1)

        # Классификация
        logits = self.head(combined)

        # Возврат статистик внимания, если запрошено
        if return_attn:
            attn_stats = {
                'weights_tok_mean': weights_tok.mean(dim=0),  # [L, H]
                'head_weights': head_alpha,
                'scale_lengths': (t_s.size(1), t_l.size(1)),
            }
            return logits, attn_stats

        return logits