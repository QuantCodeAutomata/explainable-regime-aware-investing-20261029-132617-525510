"""Shared daily rolling backtest engine used by both the Wasserstein-HMM
(Section 7.1) and KNN (Section 7.2) experiments.

Custom implementation: the paper specifies the moment-estimation step (model
specific) and the MVO step (shared, Section 5.6), but not a generic backtest
loop; this module implements that loop once so both experiments call the
identical rebalancing, turnover and realized-return logic.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.optimizer import TransactionCostAwareMVO


MomentFn = Callable[[int], tuple[np.ndarray | None, np.ndarray | None, dict]]


def run_rolling_backtest(
    dates: pd.DatetimeIndex,
    log_ret: pd.DataFrame,
    moment_fn: MomentFn,
    test_start: str,
    gamma: float,
    tau: float,
    w_max: float,
) -> dict:
    """Runs the shared daily rebalancing loop over the out-of-sample period.

    Parameters
    ----------
    dates : the common DatetimeIndex of the aligned feature/return panel.
    log_ret : realized log returns (n_dates x n_assets), row t is the return
        earned on day t by the weights decided from information through t-1.
    moment_fn : callable(t) -> (mu_t, sigma_t, diag) returning the asset
        return moments for the decision made at integer position t, or
        (None, None, diag) if there is not yet enough history.
    test_start : first out-of-sample date (inclusive).

    Returns
    -------
    dict with 'weights' (DataFrame), 'returns' (Series, log), 'diagnostics'
    (DataFrame) all indexed by date, restricted to the out-of-sample period.
    """
    n_assets = log_ret.shape[1]
    tickers = list(log_ret.columns)
    optimizer = TransactionCostAwareMVO(n_assets, gamma, tau, w_max)

    test_start_ts = pd.Timestamp(test_start)
    start_idx = int(np.searchsorted(dates.values, test_start_ts.to_datetime64()))
    assert start_idx < len(dates), "test_start falls after the available sample"

    w_prev = np.full(n_assets, 1.0 / n_assets)
    weight_rows, return_rows, diag_rows, out_dates = [], [], [], []

    for t in range(start_idx, len(dates)):
        mu_t, sigma_t, diag = moment_fn(t)
        if mu_t is None:
            continue  # not enough history yet; day is excluded from the OOS series
        w_t = optimizer.solve(mu_t, sigma_t, w_prev)
        r_t = float(np.dot(w_t, log_ret.iloc[t].values))

        weight_rows.append(w_t)
        return_rows.append(r_t)
        diag_rows.append(diag)
        out_dates.append(dates[t])
        w_prev = w_t

    out_index = pd.DatetimeIndex(out_dates, name="date")
    weights_df = pd.DataFrame(weight_rows, index=out_index, columns=tickers)
    returns_series = pd.Series(return_rows, index=out_index, name="log_return")
    diagnostics_df = pd.DataFrame(diag_rows, index=out_index)

    return {"weights": weights_df, "returns": returns_series, "diagnostics": diagnostics_df}
