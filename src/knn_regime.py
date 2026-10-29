"""Non-parametric K-nearest-neighbors regime model (paper Section 6).

Library: sklearn.neighbors.NearestNeighbors for the neighbor search and
sklearn.covariance.LedoitWolf for the shrinkage covariance estimate, exactly
applied to the covariance estimates used by the HMM and KNN regime models (paper Sections 3, 5.3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors


@dataclass
class KNNRegimeModel:
    """For each test day t, finds the K nearest neighbors of x_t in the
    historical feature matrix (excluding day t itself) and estimates the
    conditional mean/covariance of asset returns from the realized returns
    on the neighbor days (Section 6, steps 2-3)."""

    n_assets: int
    n_neighbors: int
    lookback_l: int

    def conditional_moments(
        self, feature_history: np.ndarray, return_history: np.ndarray, t: int
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        """``feature_history``/``return_history`` are aligned arrays with
        row t itself included; neighbors are searched only in rows [0, t)
        (strict causality: the day-t query vector never appears as its own
        neighbor, and no information at or after t is used)."""
        x_t = feature_history[t : t + 1]
        train_X = feature_history[:t]
        train_R = return_history[:t]

        n_train = train_X.shape[0]
        diag: dict = {"n_train": n_train}
        if n_train < self.lookback_l:
            return None, None, diag

        k = min(self.n_neighbors, n_train)
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(train_X)
        _, idx = nn.kneighbors(x_t)
        neighbor_idx = idx[0]

        neighbor_returns = train_R[neighbor_idx]  # (k, n_assets)
        mu_t = neighbor_returns.mean(axis=0)

        if k >= 2:
            lw = LedoitWolf().fit(neighbor_returns)
            sigma_t = lw.covariance_
            diag["ledoit_alpha"] = float(lw.shrinkage_)
        else:
            sigma_t = np.cov(neighbor_returns.T) if neighbor_returns.shape[0] > 1 else np.eye(self.n_assets)

        diag["k_neighbors"] = k
        return mu_t, sigma_t, diag
