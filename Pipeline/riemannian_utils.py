# file: riemannian_utils.py
# -*- coding: utf-8 -*-
"""
Домен римановой геометрии: утилиты для обработки SPD матриц.

Содержит функции для вычисления ковариаций (OAS), логарифмического
отображения, векторизации и других операций на многообразии
симметричных положительно определенных (SPD) матриц. Также включает
модуль для аугментации данных в касательном пространстве.
"""
from typing import Tuple

import torch
import torch.nn as nn

from config import EPSILON

# =============================================================================
# Базовые математические операции
# =============================================================================

def _eigh_cpu_fallback(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Надежная функция для eigendecomposition (eigh) с обработкой устройств."""
    dev_type = A.device.type
    if dev_type == 'mps':
        A_cpu = A.detach().to('cpu', dtype=torch.float32)
        w, V = torch.linalg.eigh(A_cpu)
        return w.to(A.device, dtype=A.dtype), V.to(A.device, dtype=A.dtype)
    if dev_type == 'cuda' and A.dtype in (torch.float16, torch.bfloat16):
        A32 = A.to(dtype=torch.float32)
        w, V = torch.linalg.eigh(A32)
        return w.to(dtype=A.dtype), V.to(dtype=A.dtype)
    return torch.linalg.eigh(A)


def _spd_eig_clamp_robust(A: torch.Tensor, eps: float, max_tries: int = 6) -> torch.Tensor:
    """Надежное обеспечение SPD через ограничение собственных значений с эскалацией джиттера и float64-фолбэком."""
    # Санитизация и симметризация
    A = torch.nan_to_num(A, nan=0.0, posinf=1e6, neginf=-1e6)
    A = 0.5 * (A + A.transpose(-1, -2))
    n = A.size(-1)
    eye = torch.eye(n, device=A.device, dtype=A.dtype)
    if A.dim() == 3:
        eye = eye.unsqueeze(0)
    # Стартовый джиттер как число (не Tensor), чтобы избежать ошибок типов
    jitter = max(10.0 * eps, 1e-9)
    B = A + jitter * eye
    for i in range(max_tries):
        try:
            # Основной путь: считать на текущем устройстве.
            w, V = _eigh_cpu_fallback(B)
            w = torch.clamp(w, min=float(eps))
            return (V * w.unsqueeze(-2)) @ V.transpose(-1, -2)
        except Exception:
            # Fallback на CPU только при ошибке на основном пути.
            if A.device.type != 'cpu':
                try:
                    B64 = B.detach().to('cpu', dtype=torch.float64)
                    w64, V64 = torch.linalg.eigh(B64)
                    w64 = torch.clamp(w64, min=float(eps))
                    V = V64.to(device=A.device, dtype=A.dtype)
                    w = w64.to(device=A.device, dtype=A.dtype)
                    return (V * w.unsqueeze(-2)) @ V.transpose(-1, -2)
                except Exception:
                    pass
            # Эскалация джиттера
            scale = (10.0 ** (i + 1)) * eps
            B = A + scale * eye
    print("⚠️  ПРЕДУПРЕЖДЕНИЕ: _spd_eig_clamp_robust не сошелся. Используется диагональ.")
    diag = torch.diagonal(A, dim1=-2, dim2=-1).abs().clamp(min=eps)
    return torch.diag_embed(diag)

# =============================================================================
# Основные римановы утилиты
# =============================================================================

def window_signal(x: torch.Tensor, window: int, stride: int) -> torch.Tensor:
    """Разбивает сигнал [B, C, T] на перекрывающиеся окна."""
    B, C, T = x.shape
    if window > T: raise ValueError(f"Окно ({window}) > сигнала ({T})")
    L = 1 + (T - window) // stride
    xs = [x[:, :, i * stride:i * stride + window] for i in range(L)]
    return torch.stack(xs, dim=1)


def cov_shrinkage_oas(x: torch.Tensor, eps: float = EPSILON, min_alpha: float = 0.1) -> torch.Tensor:
    """Оценка ковариации с Oracle Approximating Shrinkage (OAS).

    Параметры:
      - min_alpha: нижняя граница зажима коэффициента усадки (по умолчанию 0.1; для B2 допускаем 0.01).
    """
    B, C, T = x.shape
    x = x - x.mean(dim=-1, keepdim=True)
    cov = (x @ x.transpose(-1, -2)) / (T - 1)
    eye = torch.eye(C, device=x.device, dtype=x.dtype).unsqueeze(0)
    cov = cov + eps * eye
    trace_cov = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    trace_cov2 = (cov ** 2).sum(dim=(-1, -2))
    mu = trace_cov / C
    num = trace_cov2 + mu ** 2
    den = (T - 1) * (trace_cov2 - (trace_cov ** 2) / C) + 1e-6
    alpha = (num / den).clamp(min_alpha, 1.0).unsqueeze(-1).unsqueeze(-1)
    shrunk = (1 - alpha) * cov + alpha * mu.view(B, 1, 1) * eye
    return _spd_eig_clamp_robust(shrunk, eps=eps)


def cov_shrinkage_ledoit_wolf(x: torch.Tensor, eps: float = EPSILON) -> torch.Tensor:
    """Оценка ковариации методом Ledoit–Wolf (закрытая форма).

    Формулы:
      S = (1/T) X X^T, mu = tr(S)/C, T — длина окна, C — число каналов.
      phi_hat = (1/T) * sum_ij ( E[x_i^2 x_j^2] - S_ij^2 ), где E[·]≈(1/T)∑ x_ti^2 x_tj^2
      gamma_hat = ||S - mu I||_F^2
      alpha = clip(phi_hat/gamma_hat, 0, 1)
      Σ = (1 - alpha) S + alpha * mu I
    """
    B, C, T = x.shape
    x = x - x.mean(dim=-1, keepdim=True)
    eye = torch.eye(C, device=x.device, dtype=x.dtype).unsqueeze(0)
    # sample covariance (1/T)
    S = (x @ x.transpose(-1, -2)) / T
    S = 0.5 * (S + S.transpose(-1, -2))
    mu = S.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / C  # [B]
    # E[x_i^2 x_j^2]
    X2 = x.pow(2)
    E_mat = (X2 @ X2.transpose(-1, -2)) / T  # [B,C,C]
    S_sq = S.pow(2)
    phi_mat = E_mat - S_sq
    phi_hat = (phi_mat.sum(dim=(-1, -2)) / T).clamp(min=0.0)  # [B]
    # gamma_hat = ||S - mu I||_F^2
    S_muI = S - mu.view(B, 1, 1) * eye
    gamma_hat = S_muI.pow(2).sum(dim=(-1, -2)).clamp(min=1e-12)
    alpha = (phi_hat / gamma_hat).clamp(0.0, 1.0).view(B, 1, 1)
    Sigma = (1 - alpha) * S + alpha * mu.view(B, 1, 1) * eye
    Sigma = Sigma + eps * eye
    return _spd_eig_clamp_robust(Sigma, eps=eps)


def spd_logm(A: torch.Tensor, eps: float = EPSILON) -> torch.Tensor:
    """Логарифмическое отображение для SPD матриц (Log-Euclidean map)."""
    try:
        w, V = _eigh_cpu_fallback(A)
        w = torch.clamp(w, min=eps)
        logw = torch.log(w)
        return (V * logw.unsqueeze(-2)) @ V.transpose(-1, -2)
    except Exception:
        print("⚠️  ПРЕДУПРЕЖДЕНИЕ: spd_logm не сошелся. Используется диагональ.")
        diag = torch.diagonal(A, dim1=-2, dim2=-1).clamp(min=eps)
        return torch.diag_embed(torch.log(diag))


def spd_vectorize(A: torch.Tensor) -> torch.Tensor:
    """Векторизует симметричную матрицу, извлекая верхний треугольник."""
    B, C, _ = A.shape
    iu = torch.triu_indices(C, C, offset=0, device=A.device)
    return A[:, iu[0], iu[1]]


def spd_correlation_from_cov(cov: torch.Tensor, eps: float = EPSILON) -> torch.Tensor:
    """Преобразует ковариационную матрицу в корреляционную."""
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)
    inv_sqrt = diag.clamp(min=10.0 * eps).rsqrt()
    Dm12 = torch.diag_embed(inv_sqrt)
    corr = Dm12 @ cov @ Dm12
    corr = 0.5 * (corr + corr.transpose(-1, -2))
    eye = torch.eye(corr.size(-1), device=corr.device, dtype=corr.dtype)
    if corr.dim() == 3: eye = eye.unsqueeze(0)
    corr = corr + 10.0 * eps * eye
    return _spd_eig_clamp_robust(corr, eps=eps)

# =============================================================================
# Аугментация на SPD многообразии
# =============================================================================

class TangentSpaceJittering(nn.Module):
    """Аугментация путем добавления гауссовского шума в касательном пространстве."""
    def __init__(self, noise_std: float = 0.05, prob: float = 0.5, eps: float = EPSILON):
        super().__init__()
        self.noise_std = noise_std
        self.prob = prob
        self.eps = eps

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.prob:
            return C
        log_C = spd_logm(C, eps=self.eps)
        noise = torch.randn_like(log_C) * self.noise_std
        noise = (noise + noise.transpose(-2, -1)) / 2.0
        log_C_noisy = log_C + noise
        eigval, eigvec = _eigh_cpu_fallback(log_C_noisy)
        return eigvec @ torch.diag_embed(torch.exp(eigval)) @ eigvec.transpose(-2, -1)
