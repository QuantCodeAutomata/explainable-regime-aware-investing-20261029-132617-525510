"""Closed-form 2-Wasserstein distance between multivariate Gaussians.

Custom implementation (paper Section 5.4 "Template-Based Regime Identity via
Wasserstein Distance"):

    W_2^2(N(mu1, Sigma1), N(mu2, Sigma2))
        = ||mu1 - mu2||_2^2 + Tr(Sigma1 + Sigma2 - 2 (Sigma2^{1/2} Sigma1 Sigma2^{1/2})^{1/2})

No scikit-learn, scipy, skfolio or copulae routine exposes this
closed-form Bures-Wasserstein distance between Gaussian measures directly, so
it is implemented from the formula above using ``scipy.linalg.sqrtm``.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm


def _matrix_sqrt_psd(mat: np.ndarray) -> np.ndarray:
    """Principal square root of a symmetric PSD matrix, clipped against
    numerical noise so eigenvalues cannot become spuriously negative."""
    mat = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def gaussian_wasserstein2_sq(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """Squared 2-Wasserstein distance between N(mu1, sigma1) and N(mu2, sigma2)."""
    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    diff = mu1 - mu2
    mean_term = float(diff @ diff)

    sqrt_s1 = _matrix_sqrt_psd(sigma1)
    inner = sqrt_s1 @ sigma2 @ sqrt_s1
    sqrt_inner = _matrix_sqrt_psd(inner)

    trace_term = float(np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(sqrt_inner))
    trace_term = max(trace_term, 0.0)  # guard against negative numerical noise
    return mean_term + trace_term


def gaussian_wasserstein2(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """2-Wasserstein distance (square root of ``gaussian_wasserstein2_sq``)."""
    return float(np.sqrt(max(gaussian_wasserstein2_sq(mu1, sigma1, mu2, sigma2), 0.0)))
