"""Unit tests for the transaction-cost-aware MVO layer (paper Section 5.6)."""
from __future__ import annotations

import numpy as np
import pytest

from src.optimizer import TransactionCostAwareMVO


def test_weights_satisfy_simplex_and_box_constraints():
    n = 4
    opt = TransactionCostAwareMVO(n_assets=n, gamma=5.0, tau=0.001, w_max=0.6)
    mu = np.array([0.01, 0.02, -0.01, 0.0])
    sigma = 0.01 * np.eye(n)
    w_prev = np.full(n, 1.0 / n)
    w = opt.solve(mu, sigma, w_prev)
    assert w.sum() == pytest.approx(1.0, abs=1e-4)
    assert (w >= -1e-6).all()
    assert (w <= 0.6 + 1e-6).all()


def test_higher_transaction_cost_reduces_turnover():
    n = 4
    mu = np.array([0.05, -0.02, 0.0, 0.01])
    sigma = 0.02 * np.eye(n)
    w_prev = np.full(n, 1.0 / n)

    opt_low_tau = TransactionCostAwareMVO(n_assets=n, gamma=5.0, tau=0.0, w_max=0.6)
    opt_high_tau = TransactionCostAwareMVO(n_assets=n, gamma=5.0, tau=0.5, w_max=0.6)

    w_low = opt_low_tau.solve(mu, sigma, w_prev)
    w_high = opt_high_tau.solve(mu, sigma, w_prev)

    turnover_low = np.abs(w_low - w_prev).sum()
    turnover_high = np.abs(w_high - w_prev).sum()
    assert turnover_high <= turnover_low + 1e-6


def test_higher_risk_aversion_favors_more_diversified_weights():
    n = 3
    mu = np.array([0.05, 0.05, 0.05])
    sigma = np.array([[0.04, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.09]])
    w_prev = np.full(n, 1.0 / n)

    opt_low_gamma = TransactionCostAwareMVO(n_assets=n, gamma=0.1, tau=0.0, w_max=1.0)
    opt_high_gamma = TransactionCostAwareMVO(n_assets=n, gamma=50.0, tau=0.0, w_max=1.0)

    w_low = opt_low_gamma.solve(mu, sigma, w_prev)
    w_high = opt_high_gamma.solve(mu, sigma, w_prev)

    herfindahl_low = np.sum(w_low ** 2)
    herfindahl_high = np.sum(w_high ** 2)
    # with equal expected returns, higher risk aversion must diversify more
    # (lower Herfindahl concentration) away from the lowest-variance asset only
    assert herfindahl_high <= herfindahl_low + 1e-6
