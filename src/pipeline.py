"""Shared data-loading, benchmark and reporting pipeline used by every
experiment (paper Sections 3, 4, 10). Keeping this in one module guarantees
that the Wasserstein-HMM and KNN experiments see the identical universe,
feature construction, benchmark definitions and evaluation protocol, so that
only the regime/moment model differs between them.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data.data_loader import load_price_panel
from src import config
from src.benchmarks import buy_and_hold_returns, equal_weight_returns
from src.features import build_causal_features, valid_feature_slice
from src.metrics import cumulative_return_series, summarize_performance

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_aligned_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Loads the Section 3 cross-asset universe and builds the Section 4
    causal feature matrix, aligned to a common valid date index.

    Returns (features, log_ret, ticker_details).
    """
    panel, ticker_details = load_price_panel(config.TICKERS, config.SAMPLE_START, config.SAMPLE_END)
    features, log_ret = build_causal_features(panel, config.VOL_WINDOW, config.MOM_WINDOW)
    features, log_ret = valid_feature_slice(features, log_ret)
    logger.info(
        "Aligned panel: rows=%d, range=%s..%s, universe=%s, vol_window=%d, mom_window=%d",
        len(features), features.index.min(), features.index.max(), config.TICKERS,
        config.VOL_WINDOW, config.MOM_WINDOW,
    )
    return features, log_ret, ticker_details


def build_benchmarks(log_ret: pd.DataFrame, test_start: str) -> dict[str, pd.Series]:
    """Section 10 benchmarks: equal-weight (20% each) and SPX buy-and-hold."""
    return {
        "equal_weight": equal_weight_returns(log_ret, test_start),
        "spx_buy_and_hold": buy_and_hold_returns(log_ret[config.BENCHMARK_SPX_TICKER], test_start),
    }


def compute_all_metrics(
    strategy_name: str, strategy_returns: pd.Series, weights: pd.DataFrame, benchmarks: dict[str, pd.Series]
) -> dict:
    """Metrics for the strategy (Sections 7.1/7.2 diagnostics) and for every
    benchmark (Section 8/10), evaluated on their respective realized OOS
    return series."""
    out = {
        strategy_name: summarize_performance(
            strategy_returns, weights, config.RISK_FREE_RATE_ANNUAL, config.ANNUALIZATION_DAYS
        )
    }
    n_assets = weights.shape[1]
    for name, ret in benchmarks.items():
        if name == "equal_weight":
            bench_weights = pd.DataFrame(
                1.0 / n_assets, index=ret.index, columns=weights.columns
            )
        else:
            bench_weights = pd.DataFrame(0.0, index=ret.index, columns=weights.columns)
            bench_weights[config.BENCHMARK_SPX_TICKER] = 1.0
        out[name] = summarize_performance(
            ret, bench_weights, config.RISK_FREE_RATE_ANNUAL, config.ANNUALIZATION_DAYS
        )
    return out


def save_outputs(
    experiment_name: str,
    result: dict,
    benchmarks: dict[str, pd.Series],
    metrics: dict,
    extra: dict | None = None,
) -> None:
    """Writes weights/returns/diagnostics/metrics and the cumulative-return
    and weight-evolution plots directly under results/, as required for every
    experiment (Sections 8 and 10 of the paper)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result["weights"].to_csv(RESULTS_DIR / f"{experiment_name}_weights.csv")
    result["returns"].to_csv(RESULTS_DIR / f"{experiment_name}_returns.csv")
    result["diagnostics"].to_csv(RESULTS_DIR / f"{experiment_name}_diagnostics.csv")
    with open(RESULTS_DIR / f"{experiment_name}_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    if extra:
        for fname, payload in extra.items():
            with open(RESULTS_DIR / fname, "w") as fh:
                json.dump(payload, fh, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    cumulative_return_series(result["returns"]).mul(100).plot(ax=ax, label=experiment_name)
    for name, ret in benchmarks.items():
        cumulative_return_series(ret).mul(100).plot(ax=ax, label=name)
    ax.set_title(f"Cumulative return, out-of-sample ({experiment_name})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{experiment_name}_cumulative_return.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    result["weights"].plot.area(ax=ax, linewidth=0)
    ax.set_title(f"Portfolio weight evolution ({experiment_name})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.legend(loc="upper left", fontsize="small")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{experiment_name}_weights.png", dpi=150)
    plt.close(fig)
