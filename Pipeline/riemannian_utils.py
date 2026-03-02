# file: riemannian_utils.py
# -*- coding: utf-8 -*-
"""
Домен римановой геометрии: утилиты для обработки SPD матриц.

Содержит функции для вычисления ковариаций (OAS, Ledoit-Wolf),
логарифмического отображения (Log-Euclidean map), векторизации и других
операций на многообразии симметричных положительно определенных (SPD)
матриц. Также включает модуль для аугментации данных в касательном
пространстве.

Особенности реализации:
- Полная поддержка GPU (CUDA) и CPU, включая fallback для MPS (Apple Silicon).
- Робастная обработка численной нестабильности (регуляризация собственных значений).
- Векторизованные операции для пакетной обработки (Batch processing).
"""

# =============================================================================
# Standard Libraries
# =============================================================================
from typing import Tuple

# =============================================================================
# Third-Party Libraries
# =============================================================================
import torch
import torch.nn as nn

# =============================================================================
# Local Imports
# =============================================================================
from config import EPSILON


# =============================================================================
# Базовые математические операции (Low-level Math)
# =============================================================================

def _eigh_cpu_fallback(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
    ---------------
        Надежная функция для собственного разложения (eigendecomposition).
        Обрабатывает случаи, когда стандартный `torch.linalg.eigh` падает
        на специфичных устройствах (MPS) или при низких точностях (float16).

        Логика работы:
        1. Для MPS (Mac): переводит тензор на CPU, считает, возвращает обратно.
        2. Для CUDA с float16/bfloat16: повышает точность до float32, считает,
           приводит результат к исходному типу.
        3. В остальных случаях использует нативный метод.

    Args:
    ---------------
        A: torch.Tensor - Симметричная матрица (или батч матриц).

    Returns:
    ---------------
        Tuple[torch.Tensor, torch.Tensor]:
            - w: Собственные значения.
            - V: Собственные векторы.

    Raises:
    ---------------
        Нет явных исключений (ошибки делегируются вызывающему коду).

    Examples:
    ---------------
        >>> A = torch.randn(4, 4)
        >>> A = A @ A.t()  # Делаем симметричной
        >>> w, V = _eigh_cpu_fallback(A)
    """
    dev_type = A.device.type

    # Обработка Apple Silicon (MPS)
    if dev_type == 'mps':
        A_cpu = A.detach().to('cpu', dtype=torch.float32)
        w, V = torch.linalg.eigh(A_cpu)
        return (
            w.to(A.device, dtype=A.dtype),
            V.to(A.device, dtype=A.dtype)
        )

    # Обработка низких точностей на CUDA
    if dev_type == 'cuda' and A.dtype in (torch.float16, torch.bfloat16):
        A32 = A.to(dtype=torch.float32)
        w, V = torch.linalg.eigh(A32)
        return w.to(dtype=A.dtype), V.to(dtype=A.dtype)

    # Стандартный путь
    return torch.linalg.eigh(A)


def _spd_eig_clamp_robust(
    A: torch.Tensor,
    eps: float,
    max_tries: int = 6
) -> torch.Tensor:
    """
    Description:
    ---------------
        Гарантирует положительную определенность матрицы путем ограничения
        собственных значений. Использует адаптивную стратегию с увеличением
        регуляризации (jitter) и fallback на CPU с float64 в случае ошибок.

        Алгоритм:
        1. Санитизация входных данных (замена NaN/Inf).
        2. Симметризация.
        3. Попытка разложения с начальным jitter.
        4. При ошибке: попытка на CPU в float64.
        5. При неудаче: увеличение jitter в 10 раз и повтор.
        6. Полный провал: возврат диагональной матрицы.

    Args:
    ---------------
        A: torch.Tensor - Входная симметричная матрица.
        eps: float - Минимальное собственное значение (порог отсечки).
        max_tries: int - Максимальное количество попыток увеличения jitter.

    Returns:
    ---------------
        torch.Tensor: SPD матрица той же размерности.

    Raises:
    ---------------
        Нет явных исключений (возвращает аппроксимацию в худшем случае).

    Examples:
    ---------------
        >>> A = torch.randn(3, 3)
        >>> A_spd = _spd_eig_clamp_robust(A @ A.t(), eps=1e-6)
    """
    # Санитизация: замена NaN и бесконечностей
    A = torch.nan_to_num(A, nan=0.0, posinf=1e6, neginf=-1e6)

    # Принудительная симметризация (A = (A + A^T) / 2)
    A = 0.5 * (A + A.transpose(-1, -2))

    n = A.size(-1)
    eye = torch.eye(n, device=A.device, dtype=A.dtype)

    # Добавляем размерность батча к единичной матрице, если нужно
    if A.dim() == 3:
        eye = eye.unsqueeze(0)

    # Начальный jitter (число, не тензор, для избежания проблем типов)
    jitter = max(10.0 * eps, 1e-9)
    B = A + jitter * eye

    for i in range(max_tries):
        try:
            # Основной путь: разложение на текущем устройстве
            w, V = _eigh_cpu_fallback(B)

            # Ограничение собственных значений снизу (clamp)
            w = torch.clamp(w, min=float(eps))

            # Реконструкция матрицы: V * diag(w) * V^T
            # Используем broadcasting для умножения V на w
            return (V * w.unsqueeze(-2)) @ V.transpose(-1, -2)

        except Exception:
            # Fallback на CPU только при критической ошибке
            if A.device.type != 'cpu':
                try:
                    # Попытка в двойной точности на CPU
                    B64 = B.detach().to('cpu', dtype=torch.float64)
                    w64, V64 = torch.linalg.eigh(B64)
                    w64 = torch.clamp(w64, min=float(eps))

                    # Возврат в исходный формат
                    V = V64.to(device=A.device, dtype=A.dtype)
                    w = w64.to(device=A.device, dtype=A.dtype)
                    return (V * w.unsqueeze(-2)) @ V.transpose(-1, -2)
                except Exception:
                    pass

            # Эскалация: увеличение jitter в 10 раз
            scale = (10.0 ** (i + 1)) * eps
            B = A + scale * eye

    # Критический случай: возврат диагональной матрицы
    print(
        "⚠️  ПРЕДУПРЕЖДЕНИЕ: _spd_eig_clamp_robust не сошелся. "
        "Используется диагональ."
    )
    diag = torch.diagonal(A, dim1=-2, dim2=-1).abs().clamp(min=eps)
    return torch.diag_embed(diag)


# =============================================================================
# Основные римановы утилиты (Core Riemannian Utilities)
# =============================================================================

def window_signal(
    x: torch.Tensor,
    window: int,
    stride: int
) -> torch.Tensor:
    """
    Description:
    ---------------
        Разбивает сигнал на перекрывающиеся окна (sliding window).
        Преобразует временной ряд в последовательность сегментов для
        последующего вычисления ковариационных матриц.

    Args:
    ---------------
        x: torch.Tensor [B, C, T] - Входной сигнал (Batch, Channels, Time).
        window: int - Размер окна в отсчетах.
        stride: int - Шаг скольжения окна.

    Returns:
    ---------------
        torch.Tensor [B, L, C, window] - Тензор окон, где L - число окон.

    Raises:
    ---------------
        ValueError: Если размер окна больше длины сигнала.

    Examples:
    ---------------
        >>> x = torch.randn(2, 22, 1000)
        >>> windows = window_signal(x, window=200, stride=100)
        >>> windows.shape
        torch.Size([2, 9, 22, 200])
    """
    B, C, T = x.shape

    if window > T:
        raise ValueError(
            f"Окно ({window}) больше длины сигнала ({T})."
        )

    # Вычисление количества окон
    L = 1 + (T - window) // stride

    # Генерация окон через list comprehension (эффективно для GPU)
    xs = [
        x[:, :, i * stride:i * stride + window]
        for i in range(L)
    ]

    return torch.stack(xs, dim=1)


def cov_shrinkage_oas(
    x: torch.Tensor,
    eps: float = EPSILON,
    min_alpha: float = 0.1
) -> torch.Tensor:
    """
    Description:
    ---------------
        Оценка ковариационной матрицы с использованием метода
        Oracle Approximating Shrinkage (OAS).
        OAS обеспечивает лучшую оценку ковариации для малых выборок
        по сравнению с выборочной ковариацией, уменьшая дисперсию оценки
        за счет смещения к целевой матрице (масштабированной единичной).

        Формула:
        Sigma_OAS = (1 - alpha) * S + alpha * mu * I
        где alpha вычисляется аналитически для минимизации MSE.

    Args:
    ---------------
        x: torch.Tensor [B, C, T] - Центрированный сигнал.
        eps: float - Регуляризатор для численной стабильности.
        min_alpha: float - Нижняя граница коэффициента сжатия (alpha).

    Returns:
    ---------------
        torch.Tensor [B, C, C] - Робастная SPD ковариационная матрица.

    Raises:
    ---------------
        Нет явных исключений.

    Examples:
    ---------------
        >>> x = torch.randn(4, 22, 200)
        >>> cov = cov_shrinkage_oas(x)
        >>> cov.shape
        torch.Size([4, 22, 22])
    """
    B, C, T = x.shape

    # Централизация данных (вычитание среднего по времени)
    x = x - x.mean(dim=-1, keepdim=True)

    # Выборочная ковариация: S = (1/(T-1)) * X * X^T
    cov = (x @ x.transpose(-1, -2)) / (T - 1)

    # Добавление небольшой регуляризации перед вычислением весов
    eye = torch.eye(C, device=x.device, dtype=x.dtype).unsqueeze(0)
    cov = cov + eps * eye

    # Вычисление следов для формулы OAS
    trace_cov = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    trace_cov2 = (cov ** 2).sum(dim=(-1, -2))

    mu = trace_cov / C  # Среднее значение собственных значений

    # Числитель и знаменатель для оптимального alpha (формула OAS)
    num = trace_cov2 + mu ** 2
    den = (T - 1) * (trace_cov2 - (trace_cov ** 2) / C) + 1e-6

    # Вычисление alpha с ограничением диапазона [min_alpha, 1.0]
    alpha = (num / den).clamp(min_alpha, 1.0).unsqueeze(-1).unsqueeze(-1)

    # Формирование сжатой оценки
    shrunk = (1 - alpha) * cov + alpha * mu.view(B, 1, 1) * eye

    # Финальная гарантия SPD через ограничение собственных значений
    return _spd_eig_clamp_robust(shrunk, eps=eps)


def cov_shrinkage_ledoit_wolf(
    x: torch.Tensor,
    eps: float = EPSILON
) -> torch.Tensor:
    """
    Description:
    ---------------
        Оценка ковариации методом Ledoit–Wolf (закрытая форма).
        Альтернатива OAS, использующая другую эвристику для подбора
        коэффициента сжатия. Хорошо работает, когда истинная ковариация
        близка к скалярной матрице.

        Формулы:
        - S = (1/T) X X^T
        - mu = tr(S)/C
        - alpha = clip(phi_hat / gamma_hat, 0, 1)
        - Sigma = (1 - alpha) S + alpha * mu I

    Args:
    ---------------
        x: torch.Tensor [B, C, T] - Входной сигнал.
        eps: float - Регуляризатор.

    Returns:
    ---------------
        torch.Tensor [B, C, C] - SPD ковариационная матрица.

    Raises:
    ---------------
        Нет явных исключений.

    Examples:
    ---------------
        >>> x = torch.randn(4, 22, 200)
        >>> cov = cov_shrinkage_ledoit_wolf(x)
    """
    B, C, T = x.shape

    # Централизация
    x = x - x.mean(dim=-1, keepdim=True)

    eye = torch.eye(C, device=x.device, dtype=x.dtype).unsqueeze(0)

    # Sample covariance (нормировка на T, а не T-1, как в классике LW)
    S = (x @ x.transpose(-1, -2)) / T
    S = 0.5 * (S + S.transpose(-1, -2))  # Симметризация

    mu = S.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / C  # [B]

    # Вычисление phi_hat (оценка дисперсии ошибок)
    # E[x_i^2 x_j^2] приближается через выборочное среднее произведений квадратов
    X2 = x.pow(2)
    E_mat = (X2 @ X2.transpose(-1, -2)) / T  # [B, C, C]
    S_sq = S.pow(2)
    phi_mat = E_mat - S_sq
    phi_hat = (phi_mat.sum(dim=(-1, -2)) / T).clamp(min=0.0)  # [B]

    # Вычисление gamma_hat (квадрат нормы отклонения от целевой матрицы)
    # gamma_hat = ||S - mu I||_F^2
    S_muI = S - mu.view(B, 1, 1) * eye
    gamma_hat = S_muI.pow(2).sum(dim=(-1, -2)).clamp(min=1e-12)

    # Коэффициент сжатия
    alpha = (phi_hat / gamma_hat).clamp(0.0, 1.0).view(B, 1, 1)

    # Итоговая оценка
    Sigma = (1 - alpha) * S + alpha * mu.view(B, 1, 1) * eye
    Sigma = Sigma + eps * eye

    return _spd_eig_clamp_robust(Sigma, eps=eps)


def spd_logm(A: torch.Tensor, eps: float = EPSILON) -> torch.Tensor:
    """
    Description:
    ---------------
        Логарифмическое отображение для SPD матриц (Log-Euclidean map).
        Проецирует матрицу из риманова многообразия в касательное
        пространство (евклидово пространство симметричных матриц).

        Вычисляется через спектральное разложение:
        Log(A) = V * diag(log(w_i)) * V^T

    Args:
    ---------------
        A: torch.Tensor [B, C, C] - SPD матрица.
        eps: float - Минимальное собственное значение для логарифма.

    Returns:
    ---------------
        torch.Tensor [B, C, C] - Матрица в касательном пространстве.

    Raises:
    ---------------
        Нет явных исключений (fallback на диагональ при ошибке).

    Examples:
    ---------------
        >>> A = torch.eye(3).unsqueeze(0)
        >>> logA = spd_logm(A)
        >>> torch.allclose(logA, torch.zeros_like(logA))
        True
    """
    try:
        w, V = _eigh_cpu_fallback(A)
        w = torch.clamp(w, min=eps)
        logw = torch.log(w)
        return (V * logw.unsqueeze(-2)) @ V.transpose(-1, -2)
    except Exception:
        print(
            "⚠️  ПРЕДУПРЕЖДЕНИЕ: spd_logm не сошелся. "
            "Используется диагональ."
        )
        diag = torch.diagonal(A, dim1=-2, dim2=-1).clamp(min=eps)
        return torch.diag_embed(torch.log(diag))


def spd_vectorize(A: torch.Tensor) -> torch.Tensor:
    """
    Description:
    ---------------
        Векторизует симметричную матрицу, извлекая элементы
        верхнего треугольника (включая диагональ).
        Уменьшает размерность с C*C до C*(C+1)/2.

    Args:
    ---------------
        A: torch.Tensor [B, C, C] - Симметричная матрица.

    Returns:
    ---------------
        torch.Tensor [B, C*(C+1)/2] - Вектор признаков.

    Raises:
    ---------------
        Нет явных исключений.

    Examples:
    ---------------
        >>> A = torch.eye(3).unsqueeze(0)
        >>> v = spd_vectorize(A)
        >>> v.shape
        torch.Size([1, 6])
    """
    B, C, _ = A.shape
    # Получаем индексы верхнего треугольника
    iu = torch.triu_indices(C, C, offset=0, device=A.device)
    return A[:, iu[0], iu[1]]


def spd_correlation_from_cov(
    cov: torch.Tensor,
    eps: float = EPSILON
) -> torch.Tensor:
    """
    Description:
    ---------------
        Преобразует ковариационную матрицу в корреляционную.
        Нормализует ковариацию стандартными отклонениями:
        Corr = D^{-1/2} * Cov * D^{-1/2}, где D = diag(Cov).

    Args:
    ---------------
        cov: torch.Tensor [B, C, C] - Ковариационная матрица.
        eps: float - Регуляризатор для избежания деления на ноль.

    Returns:
    ---------------
        torch.Tensor [B, C, C] - Корреляционная матрица (SPD).

    Raises:
    ---------------
        Нет явных исключений.

    Examples:
    ---------------
        >>> cov = torch.randn(2, 3, 3)
        >>> cov = cov @ cov.transpose(-1, -2)  # Делаем SPD
        >>> corr = spd_correlation_from_cov(cov)
    """
    # Извлечение диагонали (дисперсии)
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)

    # Вычисление обратного квадратного корня (1 / std)
    # Защита от малых значений через 10*eps
    inv_sqrt = diag.clamp(min=10.0 * eps).rsqrt()

    # Формирование диагональной матрицы D^{-1/2}
    Dm12 = torch.diag_embed(inv_sqrt)

    # Преобразование: D^{-1/2} * Cov * D^{-1/2}
    corr = Dm12 @ cov @ Dm12

    # Принудительная симметризация (на случай численных ошибок)
    corr = 0.5 * (corr + corr.transpose(-1, -2))

    # Регуляризация
    eye = torch.eye(corr.size(-1), device=corr.device, dtype=corr.dtype)
    if corr.dim() == 3:
        eye = eye.unsqueeze(0)
    corr = corr + 10.0 * eps * eye

    return _spd_eig_clamp_robust(corr, eps=eps)


# =============================================================================
# Аугментация на SPD многообразии (SPD Augmentation)
# =============================================================================

class TangentSpaceJittering(nn.Module):
    """
    Description:
    ---------------
        Аугментация данных путем добавления гауссовского шума в
        касательном пространстве (Log-Euclidean domain).

        Алгоритм:
        1. Логарифмическое отображение: C -> Log(C).
        2. Добавление симметричного гауссовского шума.
        3. Экспоненциальное отображение: Log(C_noisy) -> C_augmented.

        Это позволяет генерировать валидные SPD матрицы, сохраняя
        геометрическую структуру многообразия.

    Args:
    ---------------
        noise_std: float - Стандартное отклонение шума.
        prob: float - Вероятность применения аугментации.
        eps: float - Регуляризатор для логарифма.

    Returns:
    ---------------
        torch.Tensor: Аугментированная SPD матрица.

    Raises:
    ---------------
        Нет явных исключений.

    Examples:
    ---------------
        >>> aug = TangentSpaceJittering(noise_std=0.1)
        >>> C = torch.eye(3).unsqueeze(0)
        >>> C_aug = aug(C)
    """

    def __init__(
        self,
        noise_std: float = 0.05,
        prob: float = 0.5,
        eps: float = EPSILON
    ):
        super().__init__()
        self.noise_std = noise_std
        self.prob = prob
        self.eps = eps

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """
        Description:
        ---------------
            Применяет аугментацию к входной ковариационной матрице.
            Работает только в режиме обучения (self.training).

        Args:
        ---------------
            C: torch.Tensor [B, C, C] - Входная SPD матрица.

        Returns:
        ---------------
            torch.Tensor [B, C, C] - Аугментированная матрица.
        """
        # Пропуск, если не режим обучения или выпало случайное число > prob
        if not self.training or torch.rand(1).item() > self.prob:
            return C

        # Переход в касательное пространство
        log_C = spd_logm(C, eps=self.eps)

        # Генерация симметричного шума
        noise = torch.randn_like(log_C) * self.noise_std
        noise = (noise + noise.transpose(-2, -1)) / 2.0

        # Добавление шума
        log_C_noisy = log_C + noise

        # Экспоненциальное отображение обратно в многообразие
        # Через спектральное разложение: exp(V * diag(lambdas) * V^T)
        eigval, eigvec = _eigh_cpu_fallback(log_C_noisy)
        return (
            eigvec @
            torch.diag_embed(torch.exp(eigval)) @
            eigvec.transpose(-2, -1)
        )