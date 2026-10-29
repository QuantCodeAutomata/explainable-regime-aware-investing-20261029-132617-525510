"""Unit tests for the closed-form 2-Wasserstein distance between Gaussians
(paper Section 5.4)."""
from __future__ import annotations

import numpy as np
import pytest

from src.wasserstein import gaussian_wasserstein2, gaussian_wasserstein2_sq


def test_zero_distance_for_identical_gaussians():
    mu = np.array([0.01, -0.02, 0.0])
    sigma = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.02]])
    d2 = gaussian_wasserstein2_sq(mu, sigma, mu, sigma)
    assert d2 == pytest.approx(0.0, abs=1e-8)


def test_symmetry():
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.normal(size=(3, 3))
        sigma1 = a @ a.T + np.eye(3)
        b = rng.normal(size=(3, 3))
        sigma2 = b @ b.T + np.eye(3)
        mu1, mu2 = rng.normal(size=3), rng.normal(size=3)
        d_12 = gaussian_wasserstein2_sq(mu1, sigma1, mu2, sigma2)
        d_21 = gaussian_wasserstein2_sq(mu2, sigma2, mu1, sigma1)
        assert d_12 == pytest.approx(d_21, rel=1e-6)


def test_mean_only_shift_reduces_to_euclidean_distance():
    """When Sigma1 == Sigma2, the trace term vanishes and W_2^2 collapses to
    the squared Euclidean distance between means."""
    sigma = np.eye(2) * 0.05
    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([0.03, -0.04])
    d2 = gaussian_wasserstein2_sq(mu1, sigma, mu2, sigma)
    assert d2 == pytest.approx(float(np.sum((mu1 - mu2) ** 2)), abs=1e-6)


def test_univariate_matches_scalar_formula():
    """For 1-D Gaussians, W_2^2 = (mu1-mu2)^2 + (sigma1-sigma2)^2."""
    mu1, mu2 = 0.1, -0.2
    s1, s2 = 0.09, 0.25
    d2 = gaussian_wasserstein2_sq(np.array([mu1]), np.array([[s1]]), np.array([mu2]), np.array([[s2]]))
    expected = (mu1 - mu2) ** 2 + (np.sqrt(s1) - np.sqrt(s2)) ** 2
    assert d2 == pytest.approx(expected, abs=1e-6)


def test_nonnegative_and_sqrt_consistency():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(4, 4))
    sigma1 = a @ a.T + np.eye(4)
    b = rng.normal(size=(4, 4))
    sigma2 = b @ b.T + np.eye(4)
    mu1, mu2 = rng.normal(size=4), rng.normal(size=4)
    d2 = gaussian_wasserstein2_sq(mu1, sigma1, mu2, sigma2)
    d = gaussian_wasserstein2(mu1, sigma1, mu2, sigma2)
    assert d2 >= 0.0
    assert d == pytest.approx(np.sqrt(d2), abs=1e-8)
