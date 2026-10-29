"""Transaction-cost-aware mean-variance optimization (paper Section 5.6 /
7.1-7.2):

    max_w  mu^T w - gamma * w^T Sigma w - tau * ||w - w_prev||_1
    s.t.   1^T w = 1, w >= 0, ||w||_inf <= w_max

Library: cvxpy, applied to the convex problem of paper Section 5.6. The problem is
built once with cvxpy.Parameter objects (mu, Sigma, w_prev) following
cvxpy's Disciplined Parametrized Programming (DPP) pattern so that the
repeated daily re-solves in the backtest reuse one compiled problem instead
of rebuilding it from scratch every day.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np


class TransactionCostAwareMVO:
    def __init__(self, n_assets: int, gamma: float, tau: float, w_max: float):
        self.n = n_assets
        self.gamma = gamma
        self.tau = tau
        self.w_max = w_max

        self.mu_param = cp.Parameter(n_assets)
        self.sigma_param = cp.Parameter((n_assets, n_assets), PSD=True)
        self.w_prev_param = cp.Parameter(n_assets)

        w = cp.Variable(n_assets)
        risk_term = cp.quad_form(w, self.sigma_param)
        cost_term = cp.norm1(w - self.w_prev_param)
        objective = cp.Maximize(self.mu_param @ w - self.gamma * risk_term - self.tau * cost_term)
        constraints = [cp.sum(w) == 1, w >= 0, w <= self.w_max]
        self.problem = cp.Problem(objective, constraints)
        self.w = w

    def solve(self, mu: np.ndarray, sigma: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
        sigma = 0.5 * (sigma + sigma.T)
        eigvals = np.linalg.eigvalsh(sigma)
        if eigvals.min() < 0:
            sigma = sigma + (abs(eigvals.min()) + 1e-10) * np.eye(self.n)

        self.mu_param.value = mu
        self.sigma_param.value = sigma
        self.w_prev_param.value = w_prev

        try:
            self.problem.solve(solver=cp.ECOS, warm_start=True)
        except cp.error.SolverError:
            self.problem.solve(solver=cp.SCS, warm_start=True)

        if self.w.value is None:
            raise RuntimeError("MVO solver failed to find a feasible solution")

        w_sol = np.clip(np.asarray(self.w.value).flatten(), 0.0, self.w_max)
        total = w_sol.sum()
        if total <= 0:
            raise RuntimeError("MVO solution collapsed to zero weights")
        return w_sol / total
