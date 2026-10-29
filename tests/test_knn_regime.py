"""Unit tests for the non-parametric KNN regime model (paper Section 6)."""
from __future__ import annotations

import numpy as np

from src.knn_regime import KNNRegimeModel


def test_excludes_current_day_from_neighbor_search():
    """Strict causality: the neighbor search at day t must only see rows
    [0, t), never day t itself, even if x_t is an exact duplicate of an
    earlier row."""
    n, n_assets = 50, 2
    features = np.zeros((n, 3 * n_assets))
    returns = np.zeros((n, n_assets))
    # Row 10 is an identical duplicate of the query at row 40; row 40 itself
    # carries an extreme return that must never be selected as its own neighbor.
    features[10] = 1.0
    returns[10] = 0.01
    features[40] = 1.0
    returns[40] = 999.0

    model = KNNRegimeModel(n_assets=n_assets, n_neighbors=1, lookback_l=5)
    mu, sigma, diag = model.conditional_moments(features, returns, t=40)
    assert mu is not None
    assert np.allclose(mu, 0.01)  # must pick row 10, not the day-40 return itself


def test_returns_none_before_lookback():
    n_assets = 2
    features = np.random.default_rng(0).normal(size=(10, 3 * n_assets))
    returns = np.random.default_rng(1).normal(size=(10, n_assets))
    model = KNNRegimeModel(n_assets=n_assets, n_neighbors=3, lookback_l=20)
    mu, sigma, diag = model.conditional_moments(features, returns, t=9)
    assert mu is None and sigma is None


def test_conditional_mean_equals_simple_average_of_neighbor_returns():
    n_assets = 1
    n = 60
    rng = np.random.default_rng(3)
    features = rng.normal(size=(n, 3 * n_assets))
    returns = rng.normal(size=(n, n_assets))
    model = KNNRegimeModel(n_assets=n_assets, n_neighbors=5, lookback_l=10)
    t = 50
    mu, sigma, diag = model.conditional_moments(features, returns, t)

    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=5).fit(features[:t])
    _, idx = nn.kneighbors(features[t : t + 1])
    expected_mu = returns[idx[0]].mean(axis=0)
    assert np.allclose(mu, expected_mu)
    assert sigma.shape == (n_assets, n_assets)
    assert diag["k_neighbors"] == 5
