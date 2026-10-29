"""Wasserstein HMM regime model (paper Section 5).

Library: hmmlearn.hmm.GaussianHMM is used for Gaussian HMM estimation and
filtering (paper Section 5.3). Everything specific to
this paper -- predictive model-order selection (5.2), template tracking via
the closed-form 2-Wasserstein distance (5.4), exponential template updates
(5.4) and mixture-moment aggregation (5.5/7.1) -- is not provided by
hmmlearn and is implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf

from src.wasserstein import gaussian_wasserstein2_sq


def _forward_log_predictive(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """One-step-ahead predictive log-density log p(x_t | x_1:t-1, K) for every
    t = 0..T-1 under a fitted (not refit) Gaussian HMM, per the PredLL
    criterion of Section 5.2. These are exactly the normalization constants
    of the log-space forward recursion: log P(x_1:t) - log P(x_1:t-1).
    Custom implementation (paper Eq. in Section 5.2); hmmlearn's public
    `score`/`score_samples` only expose the joint/log-likelihood of a whole
    sequence conditioned on its own startprob, which would incorrectly reset
    F_{t-1} at the first index of X instead of carrying over the belief
    accumulated on the history preceding X.
    """
    n_obs, n_components = X.shape[0], model.n_components
    log_b = np.column_stack(
        [
            multivariate_normal(mean=model.means_[k], cov=model.covars_[k], allow_singular=True).logpdf(X)
            for k in range(n_components)
        ]
    )
    log_startprob = np.log(np.clip(model.startprob_, 1e-300, None))
    log_transmat = np.log(np.clip(model.transmat_, 1e-300, None))

    log_pred = np.zeros(n_obs)
    log_alpha = log_startprob + log_b[0]
    log_pred[0] = logsumexp(log_alpha)
    log_alpha = log_alpha - log_pred[0]
    for t in range(1, n_obs):
        log_alpha_pred = logsumexp(log_alpha[:, None] + log_transmat, axis=0)
        log_alpha = log_alpha_pred + log_b[t]
        log_pred[t] = logsumexp(log_alpha)
        log_alpha = log_alpha - log_pred[t]
    return log_pred


def _log_b_row(model: GaussianHMM, x_row: np.ndarray) -> np.ndarray:
    """Log emission density of a single observation under every component
    of a fitted Gaussian HMM, log b_k(x) for k=1..K."""
    return np.array(
        [
            multivariate_normal(mean=model.means_[k], cov=model.covars_[k], allow_singular=True).logpdf(x_row)
            for k in range(model.n_components)
        ]
    )


def _forward_filter_full(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Final normalized log filtered belief log P(z_T = k | x_1:T) after a
    full forward pass over X (Section 5.3 filtered probabilities p_{t,k}),
    used once whenever the HMM is (re-)fit."""
    n_obs = X.shape[0]
    log_b = np.column_stack(
        [
            multivariate_normal(mean=model.means_[k], cov=model.covars_[k], allow_singular=True).logpdf(X)
            for k in range(model.n_components)
        ]
    )
    log_startprob = np.log(np.clip(model.startprob_, 1e-300, None))
    log_transmat = np.log(np.clip(model.transmat_, 1e-300, None))

    log_alpha = log_startprob + log_b[0]
    log_alpha -= logsumexp(log_alpha)
    for t in range(1, n_obs):
        log_alpha_pred = logsumexp(log_alpha[:, None] + log_transmat, axis=0)
        log_alpha = log_alpha_pred + log_b[t]
        log_alpha -= logsumexp(log_alpha)
    return log_alpha


def _forward_filter_step(model: GaussianHMM, log_alpha_prev: np.ndarray, x_row: np.ndarray) -> np.ndarray:
    """One incremental forward-filter update log P(z_t=k | x_1:t) given the
    previous normalized belief and a fixed (already estimated) transition
    matrix / emission model. This is the standard HMM filtering recursion
    (Section 5.3): it reproduces exactly what a full forward pass over the
    expanding history would give for the new day, without re-estimating the
    Gaussian HMM parameters, since those are held fixed between the periodic
    (Section 5.2, F_K-spaced) re-estimation dates."""
    log_transmat = np.log(np.clip(model.transmat_, 1e-300, None))
    log_alpha_pred = logsumexp(log_alpha_prev[:, None] + log_transmat, axis=0)
    log_alpha = log_alpha_pred + _log_b_row(model, x_row)
    log_alpha -= logsumexp(log_alpha)
    return log_alpha


def _hmm_free_params(n_components: int, n_features: int) -> int:
    """Number of free parameters of a full-covariance Gaussian HMM, used as
    Complexity(K) in Section 5.2: 'proportional to K or to the number of
    free parameters'."""
    k, d = n_components, n_features
    start_probs = k - 1
    transition = k * (k - 1)
    means = k * d
    covariances = k * d * (d + 1) // 2
    return start_probs + transition + means + covariances


def _fit_gaussian_hmm(
    X: np.ndarray, n_components: int, n_iter: int, tol: float, min_covar: float, seed: int
) -> GaussianHMM:
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=n_iter,
        tol=tol,
        min_covar=min_covar,
        random_state=seed,
        init_params="stmc",
        params="stmc",
    )
    model.fit(X)
    return model


class RegimeTemplates:
    """Persistent regime templates Theta_g = (mu_g, Sigma_g), g=1..G
    (Section 5.4), updated via exponential smoothing (eta)."""

    def __init__(self, n_templates: int, n_features: int, eta: float):
        self.G = n_templates
        self.d = n_features
        self.eta = eta
        self.mu = np.zeros((n_templates, n_features))
        self.sigma = np.stack([np.eye(n_features) for _ in range(n_templates)])
        self.initialized = False

    def initialize(self, comp_mus: np.ndarray, comp_sigmas: np.ndarray) -> None:
        """Seeds the G templates directly from the components of a
        calibration-window HMM fit with K=G states (implementation choice,
        documented in results/RESULTS.md: the paper leaves template
        initialization unspecified)."""
        k = comp_mus.shape[0]
        for g in range(self.G):
            src = g % k
            self.mu[g] = comp_mus[src]
            self.sigma[g] = comp_sigmas[src]
        self.initialized = True

    def nearest(self, mu_k: np.ndarray, sigma_k: np.ndarray) -> int:
        dists = [
            gaussian_wasserstein2_sq(mu_k, sigma_k, self.mu[g], self.sigma[g])
            for g in range(self.G)
        ]
        return int(np.argmin(dists))

    def update(self, assignments: dict[int, list[tuple[np.ndarray, np.ndarray, float]]]) -> None:
        """Exponential smoothing update: template_g <- (1-eta)*template_g +
        eta * (posterior-weighted average of components mapped to g)."""
        for g, comps in assignments.items():
            weights = np.array([w for _, _, w in comps])
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(comps)) / len(comps)
            avg_mu = sum(w * mu for (mu, _, _), w in zip(comps, weights))
            avg_sigma = sum(w * sigma for (_, sigma, _), w in zip(comps, weights))
            self.mu[g] = (1 - self.eta) * self.mu[g] + self.eta * avg_mu
            self.sigma[g] = (1 - self.eta) * self.sigma[g] + self.eta * avg_sigma


@dataclass
class WassersteinHMMRegimeModel:
    """Daily rolling Wasserstein-HMM regime model producing conditional
    first and second moments of asset returns for the MVO layer."""

    n_assets: int
    k_min: int
    k_max: int
    order_selection_freq: int
    validation_slice_len: int
    complexity_lambda: float
    n_iter: int
    tol: float
    min_covar: float
    min_train_obs: int
    template_count: int
    eta: float
    calibration_window: int
    random_state: int = 0

    templates: RegimeTemplates = field(init=False)
    current_K: int | None = field(init=False, default=None)
    last_selection_t: int = field(init=False, default=-10 ** 9)
    ledoit_alpha: float = field(init=False, default=0.0)
    _model: GaussianHMM | None = field(init=False, default=None)
    _log_alpha: np.ndarray | None = field(init=False, default=None)
    _filtered_upto_n: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        n_features = 3 * self.n_assets
        self.templates = RegimeTemplates(self.template_count, n_features, self.eta)

    def _ledoit_wolf_alpha(self, X: np.ndarray) -> float:
        """Global Ledoit-Wolf shrinkage intensity estimated on the available
        feature history (paper Section 2: 'Ledoit-Wolf shrinkage ...
        should be used for covariance stability'), applied below to each
        HMM component covariance for numerical/estimation stability."""
        lw = LedoitWolf().fit(X)
        return float(lw.shrinkage_)

    def _shrink(self, sigma: np.ndarray) -> np.ndarray:
        d = sigma.shape[0]
        target = (np.trace(sigma) / d) * np.eye(d)
        return (1 - self.ledoit_alpha) * sigma + self.ledoit_alpha * target

    def _select_order(self, history: np.ndarray) -> int:
        """Predictive model-order selection (Section 5.2): choose
        K_t = argmax_K { PredLL(K;t) - lambda_K * Complexity(K) }."""
        n = history.shape[0]
        v_len = min(self.validation_slice_len, n // 4)
        v_len = max(v_len, 10)
        train, val = history[:-v_len], history[-v_len:]

        best_k, best_score = self.k_min, -np.inf
        for k in range(self.k_min, self.k_max + 1):
            try:
                model = _fit_gaussian_hmm(
                    train, k, self.n_iter, self.tol, self.min_covar, self.random_state
                )
                # PredLL(K;t) = sum_{s in V_t} log p(x_s | F_{s-1}, K) (Eq. Section 5.2):
                # one-step-ahead predictive log-density on V_t under the model fitted on
                # train only, conditioning on the full preceding history train+val[:s].
                log_pred = _forward_log_predictive(model, np.vstack([train, val]))
                pred_ll = float(log_pred[train.shape[0]:].sum())
            except Exception:
                continue
            complexity = _hmm_free_params(k, history.shape[1])
            score = pred_ll - self.complexity_lambda * complexity
            if score > best_score:
                best_score, best_k = score, k
        return best_k

    def conditional_moments(
        self, feature_history: np.ndarray, t: int
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        """Returns (mu_t, Sigma_t) over the N asset-return dimensions for the
        decision made at the row labelled t, using feature_history[: t+1]
        (strictly F_{t-1}-measurable, see src/features.py)."""
        history = feature_history[: t + 1]
        n_obs, n_features = history.shape
        diag: dict = {}

        if n_obs < self.min_train_obs:
            return None, None, diag

        if not self.templates.initialized:
            calib = history[: self.calibration_window] if n_obs >= self.calibration_window else history
            calib_model = _fit_gaussian_hmm(
                calib, self.template_count, self.n_iter, self.tol, self.min_covar, self.random_state
            )
            self.templates.initialize(calib_model.means_, calib_model.covars_)
            self.ledoit_alpha = self._ledoit_wolf_alpha(calib)

        needs_refit = self.current_K is None or (t - self.last_selection_t) >= self.order_selection_freq
        if needs_refit:
            self.current_K = self._select_order(history)
            self.last_selection_t = t
            self.ledoit_alpha = self._ledoit_wolf_alpha(history)

        if needs_refit or self._model is None:
            # Full re-estimation (EM) of the K_t-state Gaussian HMM, done at the
            # same F_K cadence as model-order selection (Section 5.2). The
            # component parameters (mu_{t,k}, Sigma_{t,k}) of Section 5.3 are
            # then held fixed until the next re-estimation date; their
            # time-evolution is carried instead by the persistent, daily
            # exponentially-smoothed templates of Section 5.4. This keeps the
            # filtered probabilities p_{t,k} daily and strictly causal while
            # avoiding a redundant full EM re-estimation on every trading day.
            model = _fit_gaussian_hmm(
                history, self.current_K, self.n_iter, self.tol, self.min_covar, self.random_state
            )
            log_alpha = _forward_filter_full(model, history)
            self._model = model
            self._filtered_upto_n = n_obs
        else:
            model = self._model
            log_alpha = self._log_alpha
            # Sequential one-step forward updates for every new row appended
            # to the expanding history since the last processed day.
            for row in history[self._filtered_upto_n:n_obs]:
                log_alpha = _forward_filter_step(model, log_alpha, row)
            self._filtered_upto_n = n_obs

        self._log_alpha = log_alpha
        p_tk = np.exp(log_alpha)  # filtered P(z_t = k | F_{t-1}), see src/features.py docstring
        comp_mus = model.means_
        comp_sigmas = np.array([self._shrink(model.covars_[k]) for k in range(self.current_K)])

        assignments: dict[int, list[tuple[np.ndarray, np.ndarray, float]]] = {}
        comp_to_template = np.zeros(self.current_K, dtype=int)
        for k in range(self.current_K):
            g = self.templates.nearest(comp_mus[k], comp_sigmas[k])
            comp_to_template[k] = g
            assignments.setdefault(g, []).append((comp_mus[k], comp_sigmas[k], float(p_tk[k])))

        p_tg = np.zeros(self.template_count)
        for k in range(self.current_K):
            p_tg[comp_to_template[k]] += p_tk[k]

        self.templates.update(assignments)

        # Moment aggregation (Section 5.5): mu_t = sum_g p_tg mu_g, Sigma_t = sum_g p_tg Sigma_g
        mu_full = p_tg @ self.templates.mu
        sigma_full = np.tensordot(p_tg, self.templates.sigma, axes=(0, 0))

        n = self.n_assets
        mu_ret = mu_full[:n]
        sigma_ret = sigma_full[:n, :n]
        sigma_ret = 0.5 * (sigma_ret + sigma_ret.T)

        diag.update(
            {
                "K_t": self.current_K,
                "active_templates": int((p_tg > 1e-6).sum()),
                "p_tg": p_tg.copy(),
                "ledoit_alpha": self.ledoit_alpha,
            }
        )
        return mu_ret, sigma_ret, diag
