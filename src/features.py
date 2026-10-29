"""Causal feature construction shared by all experiments.

Paper Section 4 "Feature Representation": at each trading day t the market
state is x_t = [r_t; sigma_t; m_t], where r_t is the vector of daily log
returns, sigma_t a 60-day rolling volatility per asset, and m_t a 20-day
rolling mean return per asset, with "all features at time t ... computed
using information available up to t-1 (strict causality)".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns r_t = log(P_t / P_{t-1}) realized on day t."""
    log_ret = np.log(price_panel / price_panel.shift(1))
    return log_ret


def build_causal_features(
    price_panel: pd.DataFrame, vol_window: int, mom_window: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds the causal feature matrix x_t and the realized return panel.

    ``log_ret`` holds r_t, the return realized on day t (close t-1 -> close
    t). ``features`` holds the row labelled t built only from r_{t-1} and
    windows ending at t-1 (i.e. ``raw.shift(1)``), so the feature row x_t is
    F_{t-1}-measurable exactly as specified in Section 4: a portfolio weight
    decided from x_t can be safely applied to the return realized on day t
    without look-ahead.

    Returns
    -------
    features : DataFrame with columns ret_<ticker>, vol_<ticker>, mom_<ticker>
    log_ret : DataFrame of realized log returns (one column per ticker)
    """
    log_ret = compute_log_returns(price_panel)
    vol = log_ret.rolling(vol_window).std()
    mom = log_ret.rolling(mom_window).mean()

    raw = pd.concat(
        [log_ret.add_prefix("ret_"), vol.add_prefix("vol_"), mom.add_prefix("mom_")],
        axis=1,
    )
    features = raw.shift(1)
    return features, log_ret


def valid_feature_slice(
    features: pd.DataFrame, log_ret: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drops leading rows where either the feature row or the realized
    return is not yet defined (rolling-window burn-in and the first shifted
    row)."""
    valid_index = features.dropna().index.intersection(log_ret.dropna().index)
    return features.loc[valid_index], log_ret.loc[valid_index]
