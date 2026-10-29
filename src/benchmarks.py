"""Benchmark portfolios used for comparison (paper Section 10): an
equal-weight portfolio and an SPX buy-and-hold portfolio."""
from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight_returns(log_ret: pd.DataFrame, test_start: str) -> pd.Series:
    """Daily-rebalanced equal-weight portfolio log return over the OOS
    window (Section 10: 'Equal-Weight Portfolio (20% each)')."""
    sliced = log_ret.loc[log_ret.index >= pd.Timestamp(test_start)]
    n = sliced.shape[1]
    simple = np.exp(sliced) - 1.0
    port_simple = simple.mean(axis=1)
    return np.log1p(port_simple).rename("log_return")


def buy_and_hold_returns(log_ret: pd.Series, test_start: str) -> pd.Series:
    """Buy-and-hold log return series for a single benchmark asset
    (Section 10: 'SPX Buy & Hold')."""
    return log_ret.loc[log_ret.index >= pd.Timestamp(test_start)].rename("log_return")
