"""Shared pytest fixtures.

Registers the ``market_data`` marker (tests that call the QuantX Dataloader)
and provides a session-scoped, disk-cached fixture that loads a reduced
slice of the cross-asset universe so repeated test runs do not re-download
data.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: test calls the QuantX Dataloader and requires DATALOADER_API_KEY/MASSIVE_API_KEY"
    )


@pytest.fixture(scope="session")
def small_market_slice(request: pytest.FixtureRequest):
    """Loads two years of daily bars for three of the paper's five tickers
    (SPY, AGG, GLD) through ``data/data_loader.py`` and builds the causal
    feature matrix with a proportionally shorter window, caching the result
    across test sessions."""
    if not os.environ.get("DATALOADER_API_KEY"):
        pytest.skip("DATALOADER_API_KEY not set; skipping market-data test")

    cache_key = "massive/small_market_slice_v1"
    cached = request.config.cache.get(cache_key, None)
    if cached is not None:
        features = pd.DataFrame(cached["features"])
        features.index = pd.to_datetime(cached["index"])
        log_ret = pd.DataFrame(cached["log_ret"], index=features.index, columns=cached["tickers"])
        return features, log_ret

    from data.data_loader import load_price_panel
    from src.features import build_causal_features, valid_feature_slice

    tickers = ["SPY", "AGG", "GLD"]
    panel, _ = load_price_panel(tickers, "2015-01-01", "2016-12-31")
    features, log_ret = build_causal_features(panel, vol_window=15, mom_window=5)
    features, log_ret = valid_feature_slice(features, log_ret)

    request.config.cache.set(
        cache_key,
        {
            "features": features.reset_index(drop=True).to_dict(orient="list"),
            "index": [str(d) for d in features.index],
            "log_ret": log_ret.reset_index(drop=True).to_dict(orient="list"),
            "tickers": tickers,
        },
    )
    return features, log_ret


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
