# file: riemannian_utils.py
# -*- coding: utf-8 -*-
"""
Riemannian geometry domain: utilities for processing SPD matrices.

Contains functions for covariance computation (OAS, Ledoit-Wolf),
logarithmic mapping (Log-Euclidean map), vectorization, and other
operations on the manifold of symmetric positive definite (SPD) matrices.
Also includes a module for data augmentation in tangent space.

Implementation features:
- Full GPU (CUDA) and CPU support, including fallback for MPS (Apple Silicon).
- Robust numerical instability handling (eigenvalue regularization).
- Vectorized operations for batch processing.
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
# Low-level math operations
# =============================================================================

def _eigh_cpu_fallback(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
    ---------------
        Robust eigendecomposition helper. Handles cases where standard
        `torch.linalg.eigh` fails on specific devices (MPS) or at low
        precision (float16).

        Logic:
        1. For MPS (Mac): move the tensor to CPU, compute, then move back.
        2. For CUDA with float16/bfloat16: promote to float32, compute,
           then cast the result back to the original dtype.
        3. Otherwise, use the native method.

    Args:
    ---------------
        A: torch.Tensor - Symmetric matrix (or batch of matrices).

    Returns:
    ---------------
        Tuple[torch.Tensor, torch.Tensor]:
            - w: Eigenvalues.
            - V: Eigenvectors.

    Raises:
    ---------------
        No explicit exceptions (errors are delegated to the caller).

    Examples:
    ---------------
        >>> A = torch.randn(4, 4)
        >>> A = A @ A.t()  # Make it symmetric
        >>> w, V = _eigh_cpu_fallback(A)
    """
    dev_type = A.device.type

    # Apple Silicon (MPS) handling.
    if dev_type == 'mps':
        A_cpu = A.detach().to('cpu', dtype=torch.float32)
        w, V = torch.linalg.eigh(A_cpu)
        return (
            w.to(A.device, dtype=A.dtype),
            V.to(A.device, dtype=A.dtype)
        )

    # Low-precision CUDA handling.
    if dev_type == 'cuda' and A.dtype in (torch.float16, torch.bfloat16):
        A32 = A.to(dtype=torch.float32)
        w, V = torch.linalg.eigh(A32)
        return w.to(dtype=A.dtype), V.to(dtype=A.dtype)

    # Standard path.
    return torch.linalg.eigh(A)


def _spd_eig_clamp_robust(
    A: torch.Tensor,
    eps: float,
    max_tries: int = 6
) -> torch.Tensor:
    """
    Description:
    ---------------
        Guarantees positive definiteness by clamping eigenvalues. Uses an
        adaptive strategy with increasing regularization (jitter) and a
        CPU float64 fallback on errors.

        Algorithm:
        1. Sanitize input data (replace NaN/Inf).
        2. Symmetrize.
        3. Try decomposition with initial jitter.
        4. On error: try CPU float64.
        5. On failure: increase jitter by 10x and retry.
        6. Complete failure: return a diagonal matrix.

    Args:
    ---------------
        A: torch.Tensor - Input symmetric matrix.
        eps: float - Minimum eigenvalue (clamping threshold).
        max_tries: int - Maximum number of jitter escalation attempts.

    Returns:
    ---------------
        torch.Tensor: SPD matrix with the same dimensionality.

    Raises:
    ---------------
        No explicit exceptions (returns an approximation in the worst case).

    Examples:
    ---------------
        >>> A = torch.randn(3, 3)
        >>> A_spd = _spd_eig_clamp_robust(A @ A.t(), eps=1e-6)
    """
    # Sanitization: replace NaN and infinities.
    A = torch.nan_to_num(A, nan=0.0, posinf=1e6, neginf=-1e6)

    # Force symmetrization (A = (A + A^T) / 2).
    A = 0.5 * (A + A.transpose(-1, -2))

    n = A.size(-1)
    eye = torch.eye(n, device=A.device, dtype=A.dtype)

    # Add a batch dimension to the identity matrix if needed.
    if A.dim() == 3:
        eye = eye.unsqueeze(0)

    # Initial jitter (number, not tensor, to avoid dtype issues).
    jitter = max(10.0 * eps, 1e-9)
    B = A + jitter * eye

    for i in range(max_tries):
        try:
            # Main path: decomposition on the current device.
            w, V = _eigh_cpu_fallback(B)

            # Clamp eigenvalues from below.
            w = torch.clamp(w, min=float(eps))

            # Matrix reconstruction: V * diag(w) * V^T.
            # Use broadcasting to multiply V by w.
            return (V * w.unsqueeze(-2)) @ V.transpose(-1, -2)

        except Exception:
            # CPU fallback only on critical error.
            if A.device.type != 'cpu':
                try:
                    # Try double precision on CPU.
                    B64 = B.detach().to('cpu', dtype=torch.float64)
                    w64, V64 = torch.linalg.eigh(B64)
                    w64 = torch.clamp(w64, min=float(eps))

                    # Return to the original format.
                    V = V64.to(device=A.device, dtype=A.dtype)
                    w = w64.to(device=A.device, dtype=A.dtype)
                    return (V * w.unsqueeze(-2)) @ V.transpose(-1, -2)
                except Exception:
                    pass

            # Escalation: increase jitter by 10x.
            scale = (10.0 ** (i + 1)) * eps
            B = A + scale * eye

    # Critical case: return a diagonal matrix.
    print(
        "⚠️  WARNING: _spd_eig_clamp_robust did not converge. "
        "Using the diagonal."
    )
    diag = torch.diagonal(A, dim1=-2, dim2=-1).abs().clamp(min=eps)
    return torch.diag_embed(diag)


# =============================================================================
# Core Riemannian utilities
# =============================================================================

def window_signal(
    x: torch.Tensor,
    window: int,
    stride: int
) -> torch.Tensor:
    """
    Description:
    ---------------
        Splits the signal into overlapping windows (sliding window).
        Converts the time series into a segment sequence for subsequent
        covariance matrix computation.

    Args:
    ---------------
        x: torch.Tensor [B, C, T] - Input signal (Batch, Channels, Time).
        window: int - Window size in samples.
        stride: int - Window stride.

    Returns:
    ---------------
        torch.Tensor [B, L, C, window] - Window tensor, where L is the
            number of windows.

    Raises:
    ---------------
        ValueError: If the window size is larger than the signal length.

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
            f"Window ({window}) is larger than the signal length ({T})."
        )

    # Compute the number of windows.
    L = 1 + (T - window) // stride

    # Generate windows through list comprehension (efficient on GPU).
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
        Covariance matrix estimation using the Oracle Approximating
        Shrinkage (OAS) method. OAS provides a better covariance estimate
        for small samples than sample covariance by reducing estimate
        variance through shrinkage toward the target matrix (scaled identity).

        Formula:
        Sigma_OAS = (1 - alpha) * S + alpha * mu * I
        where alpha is computed analytically to minimize MSE.

    Args:
    ---------------
        x: torch.Tensor [B, C, T] - Centered signal.
        eps: float - Regularizer for numerical stability.
        min_alpha: float - Lower bound for the shrinkage coefficient (alpha).

    Returns:
    ---------------
        torch.Tensor [B, C, C] - Robust SPD covariance matrix.

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> x = torch.randn(4, 22, 200)
        >>> cov = cov_shrinkage_oas(x)
        >>> cov.shape
        torch.Size([4, 22, 22])
    """
    B, C, T = x.shape

    # Center data (subtract the time mean).
    x = x - x.mean(dim=-1, keepdim=True)

    # Sample covariance: S = (1/(T-1)) * X * X^T.
    cov = (x @ x.transpose(-1, -2)) / (T - 1)

    # Add small regularization before computing weights.
    eye = torch.eye(C, device=x.device, dtype=x.dtype).unsqueeze(0)
    cov = cov + eps * eye

    # Compute traces for the OAS formula.
    trace_cov = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    trace_cov2 = (cov ** 2).sum(dim=(-1, -2))

    mu = trace_cov / C  # Mean eigenvalue.

    # Numerator and denominator for the optimal alpha (OAS formula).
    num = trace_cov2 + mu ** 2
    den = (T - 1) * (trace_cov2 - (trace_cov ** 2) / C) + 1e-6

    # Compute alpha with the [min_alpha, 1.0] range constraint.
    alpha = (num / den).clamp(min_alpha, 1.0).unsqueeze(-1).unsqueeze(-1)

    # Build the shrunk estimate.
    shrunk = (1 - alpha) * cov + alpha * mu.view(B, 1, 1) * eye

    # Final SPD guarantee through eigenvalue clamping.
    return _spd_eig_clamp_robust(shrunk, eps=eps)


def cov_shrinkage_ledoit_wolf(
    x: torch.Tensor,
    eps: float = EPSILON
) -> torch.Tensor:
    """
    Description:
    ---------------
        Covariance estimation with the Ledoit-Wolf method (closed form).
        Alternative to OAS that uses a different heuristic for selecting
        the shrinkage coefficient. Works well when the true covariance is
        close to a scalar matrix.

        Formulas:
        - S = (1/T) X X^T
        - mu = tr(S)/C
        - alpha = clip(phi_hat / gamma_hat, 0, 1)
        - Sigma = (1 - alpha) S + alpha * mu I

    Args:
    ---------------
        x: torch.Tensor [B, C, T] - Input signal.
        eps: float - Regularizer.

    Returns:
    ---------------
        torch.Tensor [B, C, C] - SPD covariance matrix.

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> x = torch.randn(4, 22, 200)
        >>> cov = cov_shrinkage_ledoit_wolf(x)
    """
    B, C, T = x.shape

    # Centering.
    x = x - x.mean(dim=-1, keepdim=True)

    eye = torch.eye(C, device=x.device, dtype=x.dtype).unsqueeze(0)

    # Sample covariance (normalized by T, not T-1 as in classic LW).
    S = (x @ x.transpose(-1, -2)) / T
    S = 0.5 * (S + S.transpose(-1, -2))  # Symmetrization

    mu = S.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / C  # [B]

    # Compute phi_hat (error variance estimate).
    # E[x_i^2 x_j^2] is approximated through the sample mean of square products.
    X2 = x.pow(2)
    E_mat = (X2 @ X2.transpose(-1, -2)) / T  # [B, C, C]
    S_sq = S.pow(2)
    phi_mat = E_mat - S_sq
    phi_hat = (phi_mat.sum(dim=(-1, -2)) / T).clamp(min=0.0)  # [B]

    # Compute gamma_hat (squared norm of deviation from the target matrix).
    # gamma_hat = ||S - mu I||_F^2
    S_muI = S - mu.view(B, 1, 1) * eye
    gamma_hat = S_muI.pow(2).sum(dim=(-1, -2)).clamp(min=1e-12)

    # Shrinkage coefficient.
    alpha = (phi_hat / gamma_hat).clamp(0.0, 1.0).view(B, 1, 1)

    # Final estimate.
    Sigma = (1 - alpha) * S + alpha * mu.view(B, 1, 1) * eye
    Sigma = Sigma + eps * eye

    return _spd_eig_clamp_robust(Sigma, eps=eps)


def spd_logm(A: torch.Tensor, eps: float = EPSILON) -> torch.Tensor:
    """
    Description:
    ---------------
        Logarithmic mapping for SPD matrices (Log-Euclidean map).
        Projects a matrix from the Riemannian manifold into tangent space
        (Euclidean space of symmetric matrices).

        Computed through spectral decomposition:
        Log(A) = V * diag(log(w_i)) * V^T

    Args:
    ---------------
        A: torch.Tensor [B, C, C] - SPD matrix.
        eps: float - Minimum eigenvalue for the logarithm.

    Returns:
    ---------------
        torch.Tensor [B, C, C] - Matrix in tangent space.

    Raises:
    ---------------
        No explicit exceptions (diagonal fallback on error).

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
            "⚠️  WARNING: spd_logm did not converge. "
            "Using the diagonal."
        )
        diag = torch.diagonal(A, dim1=-2, dim2=-1).clamp(min=eps)
        return torch.diag_embed(torch.log(diag))


def spd_vectorize(A: torch.Tensor) -> torch.Tensor:
    """
    Description:
    ---------------
        Vectorizes a symmetric matrix by extracting upper-triangular
        elements (including the diagonal). Reduces dimensionality from
        C*C to C*(C+1)/2.

    Args:
    ---------------
        A: torch.Tensor [B, C, C] - Symmetric matrix.

    Returns:
    ---------------
        torch.Tensor [B, C*(C+1)/2] - Feature vector.

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> A = torch.eye(3).unsqueeze(0)
        >>> v = spd_vectorize(A)
        >>> v.shape
        torch.Size([1, 6])
    """
    B, C, _ = A.shape
    # Get upper-triangular indices.
    iu = torch.triu_indices(C, C, offset=0, device=A.device)
    return A[:, iu[0], iu[1]]


def spd_correlation_from_cov(
    cov: torch.Tensor,
    eps: float = EPSILON
) -> torch.Tensor:
    """
    Description:
    ---------------
        Converts a covariance matrix to a correlation matrix. Normalizes
        covariance by standard deviations:
        Corr = D^{-1/2} * Cov * D^{-1/2}, where D = diag(Cov).

    Args:
    ---------------
        cov: torch.Tensor [B, C, C] - Covariance matrix.
        eps: float - Regularizer to avoid division by zero.

    Returns:
    ---------------
        torch.Tensor [B, C, C] - Correlation matrix (SPD).

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> cov = torch.randn(2, 3, 3)
        >>> cov = cov @ cov.transpose(-1, -2)  # Make SPD
        >>> corr = spd_correlation_from_cov(cov)
    """
    # Extract the diagonal (variances).
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)

    # Compute inverse square root (1 / std).
    # Protect against small values through 10*eps.
    inv_sqrt = diag.clamp(min=10.0 * eps).rsqrt()

    # Build diagonal matrix D^{-1/2}.
    Dm12 = torch.diag_embed(inv_sqrt)

    # Transform: D^{-1/2} * Cov * D^{-1/2}.
    corr = Dm12 @ cov @ Dm12

    # Force symmetrization in case of numerical errors.
    corr = 0.5 * (corr + corr.transpose(-1, -2))

    # Regularization.
    eye = torch.eye(corr.size(-1), device=corr.device, dtype=corr.dtype)
    if corr.dim() == 3:
        eye = eye.unsqueeze(0)
    corr = corr + 10.0 * eps * eye

    return _spd_eig_clamp_robust(corr, eps=eps)


# =============================================================================
# SPD manifold augmentation
# =============================================================================

class TangentSpaceJittering(nn.Module):
    """
    Description:
    ---------------
        Data augmentation by adding Gaussian noise in tangent space
        (Log-Euclidean domain).

        Algorithm:
        1. Logarithmic mapping: C -> Log(C).
        2. Add symmetric Gaussian noise.
        3. Exponential mapping: Log(C_noisy) -> C_augmented.

        This enables generation of valid SPD matrices while preserving the
        geometric structure of the manifold.

    Args:
    ---------------
        noise_std: float - Noise standard deviation.
        prob: float - Probability of applying augmentation.
        eps: float - Regularizer for the logarithm.

    Returns:
    ---------------
        torch.Tensor: Augmented SPD matrix.

    Raises:
    ---------------
        No explicit exceptions.

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
            Applies augmentation to the input covariance matrix.
            Works only in training mode (self.training).

        Args:
        ---------------
            C: torch.Tensor [B, C, C] - Input SPD matrix.

        Returns:
        ---------------
            torch.Tensor [B, C, C] - Augmented matrix.
        """
        # Skip if not in training mode or if the random draw is > prob.
        if not self.training or torch.rand(1).item() > self.prob:
            return C

        # Move into tangent space.
        log_C = spd_logm(C, eps=self.eps)

        # Generate symmetric noise.
        noise = torch.randn_like(log_C) * self.noise_std
        noise = (noise + noise.transpose(-2, -1)) / 2.0

        # Add noise.
        log_C_noisy = log_C + noise

        # Exponential mapping back to the manifold.
        # Through spectral decomposition: exp(V * diag(lambdas) * V^T).
        eigval, eigvec = _eigh_cpu_fallback(log_C_noisy)
        return (
            eigvec @
            torch.diag_embed(torch.exp(eigval)) @
            eigvec.transpose(-2, -1)
        )
