"""Performance and diagnostic metrics (paper Sections 7.1/7.2 "Diagnostics"
and 8 "Out-of-Sample Performance").

Library: quantstats.stats provides Sharpe ratio, Sortino ratio and maximum
drawdown on simple (not log) returns, following its standard
mean/std-annualized convention; used here instead of re-implementing these
textbook formulas. Turnover and the effective number of positions are not
provided by quantstats and are implemented from the paper's description.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import quantstats.stats as qs_stats


def log_to_simple_returns(log_returns: pd.Series) -> pd.Series:
    return np.exp(log_returns) - 1.0


def sharpe_ratio(log_returns: pd.Series, rf: float, periods: int) -> float:
    simple = log_to_simple_returns(log_returns)
    return float(qs_stats.sharpe(simple, rf=rf, periods=periods, annualize=True))


def sortino_ratio(log_returns: pd.Series, rf: float, periods: int) -> float:
    simple = log_to_simple_returns(log_returns)
    return float(qs_stats.sortino(simple, rf=rf, periods=periods, annualize=True))


def max_drawdown(log_returns: pd.Series) -> float:
    simple = log_to_simple_returns(log_returns)
    return float(qs_stats.max_drawdown(simple))


def annualized_return(log_returns: pd.Series, periods: int) -> float:
    simple = log_to_simple_returns(log_returns)
    cum = (1 + simple).prod()
    n_years = len(simple) / periods
    return float(cum ** (1 / n_years) - 1) if n_years > 0 else float("nan")


def annualized_volatility(log_returns: pd.Series, periods: int) -> float:
    simple = log_to_simple_returns(log_returns)
    return float(simple.std(ddof=1) * np.sqrt(periods))


def daily_turnover(weights: pd.DataFrame) -> pd.Series:
    """Custom implementation: L1 turnover ||w_t - w_{t-1}||_1 per day
    (Section 7.1/7.2 'Diagnostics: Calculate daily turnover')."""
    return weights.diff().abs().sum(axis=1).fillna(0.0)


def effective_n_positions(weights: pd.DataFrame) -> pd.Series:
    """Custom implementation: inverse Herfindahl index N_eff = 1 / sum(w_i^2)
    (Section 8, table 'Effective # Positions (N_eff)')."""
    sq_sum = (weights ** 2).sum(axis=1)
    return 1.0 / sq_sum.replace(0.0, np.nan)


def cumulative_return_series(log_returns: pd.Series) -> pd.Series:
    simple = log_to_simple_returns(log_returns)
    return (1 + simple).cumprod() - 1.0


def summarize_performance(
    log_returns: pd.Series, weights: pd.DataFrame, rf: float, periods: int
) -> dict:
    turnover = daily_turnover(weights)
    neff = effective_n_positions(weights)
    return {
        "sharpe_ratio": sharpe_ratio(log_returns, rf, periods),
        "sortino_ratio": sortino_ratio(log_returns, rf, periods),
        "max_drawdown": max_drawdown(log_returns),
        "annualized_return": annualized_return(log_returns, periods),
        "annualized_volatility": annualized_volatility(log_returns, periods),
        "avg_daily_turnover": float(turnover.mean()),
        "avg_effective_n_positions": float(neff.mean()),
        "n_days": int(len(log_returns)),
    }
