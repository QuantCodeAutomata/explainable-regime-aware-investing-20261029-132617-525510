"""Unit tests for the Wasserstein HMM regime model (paper Section 5)."""
from __future__ import annotations

import numpy as np
import pytest

from src.hmm_regime import (
    RegimeTemplates,
    WassersteinHMMRegimeModel,
    _forward_log_predictive,
    _hmm_free_params,
    _fit_gaussian_hmm,
)


def _two_regime_series(n: int = 400, n_features: int = 3, seed: int = 0) -> np.ndarray:
    """Synthetic feature series with two well-separated Gaussian regimes and
    a known regime-switch point, used to check the model's basic sanity."""
    rng = np.random.default_rng(seed)
    regime1 = rng.normal(loc=0.0, scale=0.5, size=(n // 2, n_features))
    regime2 = rng.normal(loc=5.0, scale=0.5, size=(n - n // 2, n_features))
    return np.vstack([regime1, regime2])


def test_hmm_free_params_grows_with_k_and_dim():
    assert _hmm_free_params(2, 3) < _hmm_free_params(3, 3)
    assert _hmm_free_params(2, 3) < _hmm_free_params(2, 6)


def test_forward_log_predictive_matches_total_loglik():
    """sum_t log p(x_t|x_1:t-1) must equal the HMM's total log-likelihood of
    the sequence (telescoping identity of the forward recursion)."""
    X = _two_regime_series(200)
    model = _fit_gaussian_hmm(X, n_components=2, n_iter=20, tol=1e-3, min_covar=1e-3, seed=0)
    log_pred = _forward_log_predictive(model, X)
    assert log_pred.sum() == pytest.approx(model.score(X), rel=1e-6)


def test_regime_templates_nearest_assigns_identical_component_to_itself():
    templates = RegimeTemplates(n_templates=3, n_features=2, eta=0.1)
    templates.initialize(
        comp_mus=np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]]),
        comp_sigmas=np.array([np.eye(2)] * 3),
    )
    g = templates.nearest(np.array([5.0, 5.0]), np.eye(2))
    assert g == 1


def test_regime_templates_update_moves_toward_assigned_components():
    templates = RegimeTemplates(n_templates=2, n_features=1, eta=0.5)
    templates.initialize(comp_mus=np.array([[0.0], [10.0]]), comp_sigmas=np.array([np.eye(1), np.eye(1)]))
    templates.update({0: [(np.array([2.0]), np.eye(1), 1.0)]})
    assert templates.mu[0, 0] == pytest.approx(1.0)  # halfway between 0 and 2 at eta=0.5
    assert templates.mu[1, 0] == pytest.approx(10.0)  # untouched template unchanged


def test_conditional_moments_returns_none_before_min_train_obs():
    model = WassersteinHMMRegimeModel(
        n_assets=2, k_min=2, k_max=3, order_selection_freq=5, validation_slice_len=20,
        complexity_lambda=1.0, n_iter=10, tol=1e-2, min_covar=1e-3, min_train_obs=100,
        template_count=3, eta=0.1, calibration_window=60,
    )
    history = _two_regime_series(50, n_features=6)
    mu, sigma, diag = model.conditional_moments(history, t=49)
    assert mu is None and sigma is None


def test_conditional_moments_shapes_and_psd_after_warmup():
    n_assets = 2
    model = WassersteinHMMRegimeModel(
        n_assets=n_assets, k_min=2, k_max=3, order_selection_freq=10, validation_slice_len=20,
        complexity_lambda=1.0, n_iter=10, tol=1e-2, min_covar=1e-3, min_train_obs=80,
        template_count=3, eta=0.2, calibration_window=60,
    )
    history = _two_regime_series(150, n_features=3 * n_assets)
    mu, sigma, diag = model.conditional_moments(history, t=149)
    assert mu.shape == (n_assets,)
    assert sigma.shape == (n_assets, n_assets)
    eigvals = np.linalg.eigvalsh(sigma)
    assert eigvals.min() >= -1e-8
    assert np.allclose(sigma, sigma.T)
    assert "K_t" in diag and model.k_min <= diag["K_t"] <= model.k_max
