# file: model.py
# -*- coding: utf-8 -*-
"""
Домен модели: определение архитектуры нейронной сети.

Содержит класс RTTMultiScale, который является основной моделью для
классификации ЭЭГ, а также вспомогательные модули, такие как
синусоидальное позиционное кодирование (SinusoidalPE).
"""
import math
from typing import Optional, List

import torch
import torch.nn as nn

from riemannian_utils import (TangentSpaceJittering, cov_shrinkage_oas,
                              cov_shrinkage_ledoit_wolf, _spd_eig_clamp_robust,
                              spd_correlation_from_cov, spd_logm,
                              spd_vectorize, window_signal)
from config import PROJECT_ROOT
import csv
import math as _math

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
        # C1: SPDNet insertion before logm
        use_spdnet: bool = False, spdnet_dims: Optional[List[int]] = None,
        spdnet_alpha: float = 0.3,
        # C1b: Orthonormal projection in tangent space (after spd_logm → vectorize)
        use_tangent_ortho: bool = False, tangent_ortho_dim: Optional[int] = None,
        # C2: Graph convolution over electrodes (pre channel_proj)
        use_gcn: bool = False, gcn_k: int = 8, gcn_alpha: float = 0.3, gcn_nonlinearity: str = 'tanh',
        gcn_K: int = 0, gcn_layers: int = 1, gcn_norm: str = 'batch', gcn_sigma: float = 0.05,
        gcn_filter: str = 'poly', gcn_dropout: float = 0.0,
        spd_jitter_std: float = 0.05, spd_jitter_prob: float = 0.5,
        use_subject_embed: bool = False, n_subjects: int = 5, subject_embed_dim: int = 16,
        subject_embed_dropout: float = 0.0,
        # C3: Domain adversarial flags
        use_domain_adv: bool = False, domain_hidden: int = 64,
    ):
        super().__init__()
        self.ws_s, self.st_s = window_size_small, stride_small
        self.ws_l, self.st_l = window_size_large, stride_large
        self.eps, self.gating, self.cov_type = eps, gating, cov_type
        self.cov_estimator = cov_estimator
        self.oas_min_alpha = oas_min_alpha
        self.use_spd_augment = use_spd_augment
        self.use_subject_embed = use_subject_embed
        # C1 flags
        self.use_spdnet = bool(use_spdnet)
        self.spdnet_dims = list(spdnet_dims) if (spdnet_dims is not None) else []
        self.spdnet_alpha = float(spdnet_alpha)
        # C1b flags
        self.use_tangent_ortho = bool(use_tangent_ortho)
        self.tangent_ortho_dim = int(tangent_ortho_dim) if tangent_ortho_dim is not None else None
        # C2 flags
        self.use_gcn = bool(use_gcn)
        self.gcn_k = int(gcn_k)
        self.gcn_alpha = float(gcn_alpha)
        self.gcn_nonlinearity = 'tanh'
        if isinstance(gcn_alpha, (int, float)):
            pass
        # pick nonlinearity
        try:
            self.gcn_nonlinearity = gcn_nonlinearity if gcn_nonlinearity in ('tanh', 'relu', 'none') else 'tanh'
        except Exception:
            self.gcn_nonlinearity = 'tanh'
        self.gcn_K = int(max(0, gcn_K))
        self.gcn_layers = int(max(1, gcn_layers))
        self.gcn_norm_type = gcn_norm if gcn_norm in ('batch', 'none') else 'batch'
        self.gcn_sigma = float(gcn_sigma)
        self.gcn_filter = gcn_filter if gcn_filter in ('poly', 'cheby') else 'poly'
        self.gcn_dropout = float(max(0.0, min(1.0, gcn_dropout)))
        # C3 flags
        self.use_domain_adv = bool(use_domain_adv)
        self.n_subjects = int(n_subjects)
        self.domain_hidden = int(domain_hidden)

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
        final_spd_dim = proj_channels
        if self.use_spdnet and len(self.spdnet_dims) > 0:
            final_spd_dim = int(self.spdnet_dims[-1])
        spd_vec_dim = final_spd_dim * (final_spd_dim + 1) // 2
        # C1b: if enabled, insert orthonormal projection layer in tangent space
        feat_in_dim = spd_vec_dim
        if self.use_tangent_ortho and self.tangent_ortho_dim is not None:
            self.tangent_proj = _OrthoProj(in_dim=spd_vec_dim, out_dim=self.tangent_ortho_dim)
            feat_in_dim = self.tangent_ortho_dim
        self.feature_proj = nn.Linear(feat_in_dim, d_model)
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
        # Domain adversarial head over pooled base features (before subject embed)
        if self.use_domain_adv and self.n_subjects > 1:
            dh = max(16, self.domain_hidden)
            self.grl = _GradReverse()
            self.domain_head = nn.Sequential(
                nn.LayerNorm(d_model * 2),
                nn.Linear(d_model * 2, dh),
                nn.ReLU(True),
                nn.Linear(dh, self.n_subjects)
            )
        self._reset_parameters()

        # Initialize SPDNet layers if enabled
        if self.use_spdnet and len(self.spdnet_dims) > 0:
            self._init_spdnet_layers(proj_channels, self.spdnet_dims)
        # Initialize GCN adjacency if enabled
        if self.use_gcn:
            self._init_gcn_adjacency(n_channels, k=self.gcn_k, sigma=self.gcn_sigma)
            if self.gcn_K > 0 or self.gcn_layers >= 1:
                self._init_graph_conv_layers(n_channels)

    def _reset_parameters(self) -> None:
        """Инициализирует веса модели."""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.scale_emb, std=0.02)

    # -------------------------
    # C1: SPDNet components
    # -------------------------
    def _init_spdnet_layers(self, in_dim: int, dims: List[int]) -> None:
        layers = []
        prev = in_dim
        for i, out in enumerate(dims):
            if out <= 0 or out > prev:
                raise ValueError(f"SPDNet dims must be >0 and <= prev. Got out={out}, prev={prev}")
            # residual alpha только в последнем слое; прочие слои без residual
            alpha = self.spdnet_alpha if (i == len(dims) - 1) else 0.0
            layer = _BiMapLayer(prev, out, eps=self.eps, alpha=alpha)
            layers.append(layer)
            prev = out
        self.spdnet = nn.ModuleList(layers)

    def _apply_spdnet(self, C: torch.Tensor) -> torch.Tensor:
        if not (self.use_spdnet and hasattr(self, 'spdnet')):
            return C
        Y = C
        for layer in self.spdnet:
            Y = layer(Y)
        return Y

    # -------------------------
    # C2: Graph over electrodes
    # -------------------------
    def _init_gcn_adjacency(self, n_channels: int, k: int = 8, sigma: float = 0.05) -> None:
        coords = []
        path = PROJECT_ROOT / 'montage.csv'
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= n_channels:
                        break
                    try:
                        x = float(row['x']); y = float(row['y']); z = float(row['z'])
                    except Exception:
                        x = y = z = 0.0
                    coords.append((x, y, z))
        except Exception:
            # Fallback: ring topology
            coords = [(float(i), 0.0, 0.0) for i in range(n_channels)]
        # build distance matrix
        import math as _m
        dmat = [[0.0]*n_channels for _ in range(n_channels)]
        for i in range(n_channels):
            xi, yi, zi = coords[i]
            for j in range(n_channels):
                xj, yj, zj = coords[j]
                dx = xi - xj; dy = yi - yj; dz = zi - zj
                d = _m.sqrt(dx*dx + dy*dy + dz*dz)
                dmat[i][j] = d
        # kNN adjacency with distance-weighted Gaussian weights (exclude self)
        A = torch.zeros(n_channels, n_channels, dtype=torch.float32)
        sig2 = max(1e-6, sigma*sigma)
        for i in range(n_channels):
            # get k nearest excluding self
            pairs = sorted([(dmat[i][j], j) for j in range(n_channels) if j != i], key=lambda p: p[0])
            for d, j in pairs[:max(1, k)]:
                w = _math.exp(-(d*d)/(2.0*sig2))
                A[i, j] = max(A[i, j].item(), w)
                A[j, i] = max(A[j, i].item(), w)
        A = A + torch.eye(n_channels, dtype=torch.float32)
        # normalize: D^{-1/2} A D^{-1/2}
        deg = A.sum(dim=1).clamp(min=1e-6)
        Dm12 = torch.diag_embed(deg.pow(-0.5))
        A_norm = Dm12 @ A @ Dm12
        self.register_buffer('gcn_A', A_norm)

    def _init_graph_conv_layers(self, n_channels: int) -> None:
        layers = []
        for _ in range(self.gcn_layers):
            layers.append(_GraphHopFilter(K=self.gcn_K, nonlin=self.gcn_nonlinearity, norm=self.gcn_norm_type, residual=True, channels=n_channels, filt=self.gcn_filter, dropout=self.gcn_dropout))
        self.graph_layers = nn.ModuleList(layers)

    def _tokens_for_scale(self, x: torch.Tensor, w: int, s: int, scale_id: int) -> torch.Tensor:
        """Извлекает токены для одного масштаба."""
        # C2: optional GCN smoothing over channels before projection
        if self.use_gcn and hasattr(self, 'gcn_A'):
            A = self.gcn_A.to(device=x.device, dtype=x.dtype)
            if hasattr(self, 'graph_layers'):
                # Learnable multi-hop filter layers
                for gl in self.graph_layers:
                    x = gl(x, A)
            else:
                # Fixed smoothing
                x_mix = (1.0 - self.gcn_alpha) * x + self.gcn_alpha * torch.einsum('cd,bdt->bct', A, x)
                if self.gcn_nonlinearity == 'relu':
                    x = torch.relu(x_mix)
                elif self.gcn_nonlinearity == 'tanh':
                    x = torch.tanh(x_mix)
                else:
                    x = x_mix
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
        # C1: SPDNet mapping before logm
        if self.use_spdnet and len(self.spdnet_dims) > 0:
            cov = self._apply_spdnet(cov)
        vec = spd_vectorize(spd_logm(cov, eps=self.eps))
        # C1b: orthonormal projection in tangent space
        if self.use_tangent_ortho and hasattr(self, 'tangent_proj'):
            vec = self.tangent_proj(vec)
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

        # Base pooled feature (for C3)
        base_feat = torch.cat([h_cls, h_attn], dim=-1)
        # Concatenate with subject embedding if enabled
        combined = base_feat
        if self.use_subject_embed:
            if subject_ids is None:
                raise ValueError("subject_ids required when use_subject_embed=True")
            subject_emb = self.subject_embed(subject_ids)  # [B, subject_embed_dim]
            subject_emb = self.subject_embed_drop(subject_emb)
            combined = torch.cat([combined, subject_emb], dim=-1)
        logits = self.head(combined)
        if self.use_domain_adv and hasattr(self, 'domain_head') and subject_ids is not None:
            domain_logits = self.domain_head(self.grl(base_feat))
            return logits, base_feat, domain_logits
        return logits, base_feat, None


class _GradReverse(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return _GradReverseFunc.apply(x, self.lambd)


class _GradReverseFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class _BiMapLayer(nn.Module):
    """BiMap layer: Y = W^T X W for SPD matrices with QR-orthogonalization and residual mix.

    - Orthonormal columns enforced via QR each forward (Stiefel projection)
    - Residual mix to identity: Y <- (1-α)·Y + α·(tr(Y)/out_dim)·I to stabilize
    - ReEig clamp for SPD robustness
    """
    def __init__(self, in_dim: int, out_dim: int, eps: float = 1e-4, alpha: float = 0.3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = eps
        self.alpha = float(alpha)
        # Weight maps from in_dim to out_dim
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.orthogonal_(self.W)

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        # C: [B, in_dim, in_dim]
        # Stiefel projection via QR to ensure orthonormal columns in W
        with torch.no_grad():
            # Small epsilon to avoid numerical issues in QR for near-singular matrices is not required here
            pass
        Q, _ = torch.linalg.qr(self.W, mode='reduced')  # [in_dim, out_dim]
        W = Q
        Wt = W.transpose(0, 1)  # [out_dim, in_dim]
        Y = Wt.unsqueeze(0) @ C @ W.unsqueeze(0)  # [B, out_dim, out_dim]
        Y = 0.5 * (Y + Y.transpose(-1, -2))
        # Residual mixing towards identity (stabilization)
        if self.alpha and self.alpha > 0.0:
            eye = torch.eye(self.out_dim, device=Y.device, dtype=Y.dtype)
            if Y.dim() == 3:
                eye = eye.unsqueeze(0)
            mu = Y.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / self.out_dim  # [B]
            Y = (1.0 - self.alpha) * Y + self.alpha * mu.view(-1, 1, 1) * eye
        # ReEig/robust SPD clamp
        Y = _spd_eig_clamp_robust(Y, eps=self.eps)
        return Y


class _OrthoProj(nn.Module):
    """Orthonormal linear projection in tangent (vector) space.

    Projects features x ∈ R^{D} to y ∈ R^{d} via y = x @ Q, where columns of Q are orthonormal.
    Orthonormality enforced by QR of a learnable weight W each forward.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        if out_dim <= 0 or out_dim > in_dim:
            raise ValueError(f"OrthoProj requires 0<out_dim<=in_dim, got out_dim={out_dim}, in_dim={in_dim}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.orthogonal_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, in_dim]
        Q, _ = torch.linalg.qr(self.W, mode='reduced')  # [in_dim, out_dim]
        return x @ Q


class _GraphHopFilter(nn.Module):
    """Learnable hop-based graph filter layer on (B,C,T).

    y = Σ_{k=0..K} θ_k · (Â^k X), with nonlinearity and optional residual + normalization.
    """
    def __init__(self, K: int, nonlin: str, norm: str, residual: bool, channels: int, filt: str = 'poly', dropout: float = 0.0):
        super().__init__()
        self.K = int(max(0, K))
        self.theta = nn.Parameter(torch.zeros(self.K + 1))
        nn.init.normal_(self.theta, mean=0.0, std=0.05)
        self.nonlin = nonlin
        self.residual = residual
        self.norm_type = norm
        self.bn = nn.BatchNorm1d(channels) if norm == 'batch' else nn.Identity()
        self.ln = _ChannelLayerNorm(channels) if norm == 'layer' else nn.Identity()
        self.filt = filt
        self.do = nn.Dropout2d(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T], A: [C,C]
        # Filter response
        if self.filt == 'cheby' and self.K >= 1:
            T0 = x
            T1 = torch.einsum('cd,bdt->bct', A, x)
            y = self.theta[0] * T0 + self.theta[1] * T1
            Tkm2, Tkm1 = T0, T1
            for k in range(2, self.K + 1):
                T_k = 2.0 * torch.einsum('cd,bdt->bct', A, Tkm1) - Tkm2
                y = y + self.theta[k] * T_k
                Tkm2, Tkm1 = Tkm1, T_k
        else:
            y = self.theta[0] * x
            if self.K > 0:
                Ax = torch.einsum('cd,bdt->bct', A, x)
                y = y + self.theta[1] * Ax
                Akx = Ax
                for k in range(2, self.K + 1):
                    Akx = torch.einsum('cd,bdt->bct', A, Akx)
                    y = y + self.theta[k] * Akx
        # Norm → Nonlinearity → Dropout → Residual → (optional second norm)
        if isinstance(self.bn, nn.BatchNorm1d):
            y = self.bn(y)
        if isinstance(self.ln, _ChannelLayerNorm):
            y = self.ln(y)
        if self.nonlin == 'relu':
            y = torch.relu(y)
        elif self.nonlin == 'tanh':
            y = torch.tanh(y)
        # Channel dropout via Dropout2d
        if isinstance(self.do, nn.Dropout2d):
            y = self.do(y.unsqueeze(-1)).squeeze(-1)
        if self.residual:
            y = y + x
        return y


class _ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T] -> [B,T,C] -> LN(C) -> [B,C,T]
        x_perm = x.permute(0, 2, 1)
        x_norm = self.ln(x_perm)
        return x_norm.permute(0, 2, 1)
