# file: model.py
# -*- coding: utf-8 -*-
"""
Домен модели: определение архитектуры нейронной сети.

Содержит класс RTTMultiScale, который является основной моделью для
классификации ЭЭГ, а также вспомогательные модули, такие как
синусоидальное позиционное кодирование (SinusoidalPE).
"""
import math

import torch
import torch.nn as nn

from riemannian_utils import (TangentSpaceJittering, cov_shrinkage_oas,
                              cov_shrinkage_ledoit_wolf,
                              spd_correlation_from_cov, spd_logm,
                              spd_vectorize, window_signal)

class SinusoidalPE(nn.Module):
    """Модуль синусоидального позиционного кодирования (Positional Encoding)."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return self.dropout(x + self.pe[:L, :])


class RTTMultiScale(nn.Module):
    """Двухмасштабный риманов токен-трансформер (RTT-MultiScale) с Subject Embeddings."""
    def __init__(
        self, n_channels: int, n_classes: int, proj_channels: int,
        window_size_small: int, stride_small: int, window_size_large: int,
        stride_large: int, d_model: int, n_heads: int, ff_dim: int,
        n_layers: int, dropout: float, eps: float, attn_heads: int,
        gating: bool, cov_type: str,
        cov_estimator: str = 'oas', oas_min_alpha: float = 0.1,
        use_spd_augment: bool = False,
        spd_jitter_std: float = 0.05, spd_jitter_prob: float = 0.5,
        use_subject_embed: bool = False, n_subjects: int = 5, subject_embed_dim: int = 16,
        subject_embed_dropout: float = 0.0,
    ):
        super().__init__()
        self.ws_s, self.st_s = window_size_small, stride_small
        self.ws_l, self.st_l = window_size_large, stride_large
        self.eps, self.gating, self.cov_type = eps, gating, cov_type
        self.cov_estimator = cov_estimator
        self.oas_min_alpha = oas_min_alpha
        self.use_spd_augment = use_spd_augment
        self.use_subject_embed = use_subject_embed

        if self.use_spd_augment:
            self.spd_augmenter = TangentSpaceJittering(spd_jitter_std, spd_jitter_prob, eps)

        # Subject embedding layer (optional)
        if self.use_subject_embed:
            self.subject_embed = nn.Embedding(n_subjects, subject_embed_dim)
            self.subject_embed_drop = nn.Dropout(subject_embed_dropout) if subject_embed_dropout and subject_embed_dropout > 0 else nn.Identity()
            classifier_input_dim = d_model * 2 + subject_embed_dim
        else:
            classifier_input_dim = d_model * 2

        self.channel_proj = nn.Linear(n_channels, proj_channels, bias=False)
        spd_dim = proj_channels * (proj_channels + 1) // 2
        self.feature_proj = nn.Linear(spd_dim, d_model)
        self.scale_emb = nn.Parameter(torch.zeros(2, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = SinusoidalPE(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ff_dim, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.attn_pool_W = nn.Linear(d_model, attn_heads)
        self.head_weights = nn.Parameter(torch.zeros(attn_heads))
        self.gate_mlp_pair = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 2), nn.ReLU(True), nn.Linear(d_model // 2, 2)
        )

        # Modified classifier head with optional subject embedding
        self.head = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Инициализирует веса модели."""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.scale_emb, std=0.02)

    def _tokens_for_scale(self, x: torch.Tensor, w: int, s: int, scale_id: int) -> torch.Tensor:
        """Извлекает токены для одного масштаба."""
        x_pc = self.channel_proj(x.transpose(1, 2)).transpose(1, 2)
        x_win = window_signal(x_pc, w, s)
        B, L, c, _ = x_win.shape
        x_flat = x_win.reshape(B * L, c, -1)
        if self.cov_estimator == 'lw':
            cov = cov_shrinkage_ledoit_wolf(x_flat, eps=self.eps)
        else:
            cov = cov_shrinkage_oas(x_flat, eps=self.eps, min_alpha=self.oas_min_alpha)
        if self.cov_type == 'corr':
            cov = spd_correlation_from_cov(cov, eps=self.eps)
        else:
            tr = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
            cov = cov / tr.clamp(min=self.eps).unsqueeze(-1)
        if self.use_spd_augment and self.training:
            cov = self.spd_augmenter(cov)
        vec = spd_vectorize(spd_logm(cov, eps=self.eps))
        tok = self.feature_proj(vec).view(B, L, -1)
        return tok + self.scale_emb[scale_id]

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Прямой проход модели с опциональными subject embeddings.

        Args:
            x: EEG data [B, C, T]
            subject_ids: Subject IDs [B] (optional, required if use_subject_embed=True)

        Returns:
            logits [B, n_classes]
        """
        t_s = self._tokens_for_scale(x, self.ws_s, self.st_s, 0)
        t_l = self._tokens_for_scale(x, self.ws_l, self.st_l, 1)
        if self.gating:
            summary = torch.cat([t_s.mean(1), t_l.mean(1)], -1)
            gates = torch.softmax(self.gate_mlp_pair(summary), dim=-1)
            t_s, t_l = t_s * gates[:, 0:1, None], t_l * gates[:, 1:2, None]
        tokens = torch.cat([t_s, t_l], dim=1)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        seq = self.pos_enc(torch.cat([cls, tokens], dim=1))
        h = self.encoder(seq)
        h_cls, toks = h[:, 0], h[:, 1:]
        scores = self.attn_pool_W(toks)
        weights_tok = torch.softmax(scores, dim=1)
        h_heads = torch.einsum('blh,bld->bhd', weights_tok, toks)
        h_attn = torch.einsum('h,bhd->bd', torch.softmax(self.head_weights, 0), h_heads)

        # Concatenate with subject embedding if enabled
        combined = torch.cat([h_cls, h_attn], dim=-1)
        if self.use_subject_embed:
            if subject_ids is None:
                raise ValueError("subject_ids required when use_subject_embed=True")
            subject_emb = self.subject_embed(subject_ids)  # [B, subject_embed_dim]
            subject_emb = self.subject_embed_drop(subject_emb)
            combined = torch.cat([combined, subject_emb], dim=-1)

        return self.head(combined)
