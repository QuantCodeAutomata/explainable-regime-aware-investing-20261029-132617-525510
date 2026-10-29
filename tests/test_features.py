"""Unit tests for causal feature construction (paper Section 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import build_causal_features, compute_log_returns, valid_feature_slice


def _synthetic_prices(n: int = 300, n_assets: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    log_rets = rng.normal(0.0003, 0.01, size=(n, n_assets))
    prices = 100 * np.exp(np.cumsum(log_rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def test_log_returns_match_manual_formula():
    prices = _synthetic_prices(50, 1)
    log_ret = compute_log_returns(prices)
    expected = np.log(prices.iloc[2, 0] / prices.iloc[1, 0])
    assert log_ret.iloc[2, 0] == pytest.approx(expected)
    assert log_ret.iloc[0].isna().all()


def test_feature_row_is_strictly_lagged_relative_to_return_row():
    """x_t must only use information through t-1: feature row t must equal
    the pre-shift raw row t-1 (no look-ahead, paper Section 4)."""
    prices = _synthetic_prices(300, 2)
    features, log_ret = build_causal_features(prices, vol_window=10, mom_window=5)

    log_r = compute_log_returns(prices)
    vol = log_r.rolling(10).std()
    mom = log_r.rolling(5).mean()

    t = 200
    assert features["ret_A0"].iloc[t] == pytest.approx(log_r["A0"].iloc[t - 1])
    assert features["vol_A0"].iloc[t] == pytest.approx(vol["A0"].iloc[t - 1])
    assert features["mom_A0"].iloc[t] == pytest.approx(mom["A0"].iloc[t - 1])
    # the realized return at row t is NOT equal to the feature's return term
    assert features["ret_A0"].iloc[t] != pytest.approx(log_ret["A0"].iloc[t])


def test_rolling_windows_have_expected_lengths():
    prices = _synthetic_prices(100, 1)
    features, _ = build_causal_features(prices, vol_window=60, mom_window=20)
    # first vol_window+1 rows must be NaN (burn-in + one-day shift)
    assert features["vol_A0"].iloc[:60].isna().all()
    assert features["mom_A0"].iloc[:20].isna().all()
    assert not np.isnan(features["vol_A0"].iloc[61])


def test_valid_feature_slice_drops_all_nans():
    prices = _synthetic_prices(100, 2)
    features, log_ret = build_causal_features(prices, vol_window=60, mom_window=20)
    f, r = valid_feature_slice(features, log_ret)
    assert not f.isna().any().any()
    assert not r.isna().any().any()
    assert len(f) == len(r)
    assert len(f) < len(features)
