"""Experiment: Non-Parametric Regime Investing using K-Nearest Neighbors
(paper Section 6, 7.2, 8.2).

Run as: python -m exp.exp_knn_comparison
"""
from __future__ import annotations

import argparse
import json
import logging

from src import config
from src.backtest import run_rolling_backtest
from src.knn_regime import KNNRegimeModel
from src.pipeline import build_benchmarks, compute_all_metrics, load_aligned_data, save_outputs

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "exp_knn_comparison"


def run(n_neighbors: int | None = None) -> dict:
    """Runs the non-parametric KNN local-moment estimator with the same
    transaction-cost-aware MVO optimizer used for the Wasserstein-HMM
    experiment (Section 7.2 steps 1-5) and saves all artifacts in results/."""
    features, log_ret, ticker_details = load_aligned_data()
    n_assets = log_ret.shape[1]

    model = KNNRegimeModel(
        n_assets=n_assets,
        n_neighbors=n_neighbors or config.KNN_N_NEIGHBORS,
        lookback_l=config.KNN_LOOKBACK_L,
    )

    feature_values = features.values
    return_values = log_ret.values

    def moment_fn(t: int):
        return model.conditional_moments(feature_values, return_values, t)

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

    save_outputs(EXPERIMENT_NAME, result, benchmarks, metrics)
    logger.info("KNN experiment metrics: %s", json.dumps(metrics[EXPERIMENT_NAME], indent=2))
    return {"metrics": metrics, "result": result}


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
