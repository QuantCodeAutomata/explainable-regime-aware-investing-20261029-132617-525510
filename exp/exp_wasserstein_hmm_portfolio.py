"""Experiment: Parametric Regime Investing with Wasserstein HMM and
Transaction-Cost-Aware MVO (paper Sections 5, 7.1, 8.1, 9).

Run as: python -m exp.exp_wasserstein_hmm_portfolio
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from src import config
from src.backtest import run_rolling_backtest
from src.hmm_regime import WassersteinHMMRegimeModel
from src.pipeline import (
    RESULTS_DIR,
    build_benchmarks,
    compute_all_metrics,
    load_aligned_data,
    save_outputs,
)

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "exp_wasserstein_hmm_portfolio"


def run(order_selection_freq: int | None = None) -> dict:
    """Runs the full Wasserstein-HMM + transaction-cost-aware MVO pipeline
    (Section 7.1 steps 1-5) over the out-of-sample window and saves all
    artifacts in results/."""
    features, log_ret, ticker_details = load_aligned_data()
    n_assets = log_ret.shape[1]

    model = WassersteinHMMRegimeModel(
        n_assets=n_assets,
        k_min=config.HMM_K_MIN,
        k_max=config.HMM_K_MAX,
        order_selection_freq=order_selection_freq or config.HMM_ORDER_SELECTION_FREQ_DAYS,
        validation_slice_len=config.HMM_VALIDATION_SLICE_LEN,
        complexity_lambda=config.HMM_COMPLEXITY_LAMBDA,
        n_iter=config.HMM_N_ITER,
        tol=config.HMM_TOL,
        min_covar=config.HMM_MIN_COVAR,
        min_train_obs=config.HMM_MIN_TRAIN_OBS,
        template_count=config.TEMPLATE_COUNT,
        eta=config.TEMPLATE_SMOOTHING_ETA,
        calibration_window=config.TEMPLATE_CALIBRATION_WINDOW,
    )

    feature_values = features.values

    def moment_fn(t: int):
        return model.conditional_moments(feature_values, t)

    result = run_rolling_backtest(
        dates=features.index,
        log_ret=log_ret,
        moment_fn=moment_fn,
        test_start=config.TEST_START,
        gamma=config.MVO_GAMMA,
        tau=config.MVO_TAU,
        w_max=config.MVO_W_MAX,
    )

    benchmarks = build_benchmarks(log_ret, config.TEST_START)
    metrics = compute_all_metrics(EXPERIMENT_NAME, result["returns"], result["weights"], benchmarks)

    template_perf = template_conditional_performance(result, log_ret)

    save_outputs(
        EXPERIMENT_NAME,
        result,
        benchmarks,
        metrics,
        extra={f"{EXPERIMENT_NAME}_template_conditional_performance.json": template_perf},
    )
    logger.info("Wasserstein HMM experiment metrics: %s", json.dumps(metrics[EXPERIMENT_NAME], indent=2))
    return {"metrics": metrics, "result": result, "template_perf": template_perf}


def template_conditional_performance(result: dict, log_ret: pd.DataFrame) -> dict:
    """Research objective 4 (Section 9 "Economic Interpretation of Market
    Regimes"): asset and portfolio performance conditional on the dominant
    regime template, from the K_t selected and the active-template
    diagnostics logged during the backtest."""
    diag = result["diagnostics"]
    if "K_t" not in diag.columns:
        return {}
    oos_returns = log_ret.loc[diag.index]
    port_returns = result["returns"]

    dominant_template = diag["p_tg"].apply(
        lambda arr: int(np.argmax(arr)) if isinstance(arr, np.ndarray) else np.nan
    )
    perf: dict = {}
    for g in sorted(dominant_template.dropna().unique()):
        mask = dominant_template == g
        n_days = int(mask.sum())
        if n_days == 0:
            continue
        perf[str(int(g))] = {
            "n_days": n_days,
            "mean_asset_log_return": oos_returns.loc[mask].mean().to_dict(),
            "mean_portfolio_log_return": float(port_returns.loc[mask].mean()),
        }
    return perf


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true",
        help="Run with the paper-scale configuration (identical to the default here: "
        "the full cross-asset universe and sample period).",
    )
    parser.parse_args()
    run()
