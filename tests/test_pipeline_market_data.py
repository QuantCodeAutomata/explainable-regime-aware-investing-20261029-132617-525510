"""Market-data integration test for the shared data-loading path.

Exercises ``data/data_loader.py`` on a reduced slice of the paper's
cross-asset universe (Section 3) and checks the causal feature construction
(Section 4) on real bars, without comparing against the paper's reported
numbers (that is left to the full experiment runs in ``exp/``).
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.market_data
def test_small_market_slice_shapes_and_causality(small_market_slice):
    features, log_ret = small_market_slice

    assert len(features) == len(log_ret)
    assert features.shape[0] > 100
    assert log_ret.shape[1] == 3  # SPY, AGG, GLD
    assert features.index.is_monotonic_increasing
    assert not features.isna().any().any()
    assert not log_ret.isna().any().any()

    # Section 4: feature at t uses information up to t-1. Within the
    # (contiguous) trimmed slice, the ret_ block of row i must equal the
    # realized log return of row i-1, not of row i itself, i.e. the feature
    # is built from "raw.shift(1)" (src/features.py) and carries no same-day
    # information.
    n_assets = log_ret.shape[1]
    r_t = features.iloc[:, :n_assets].to_numpy()
    np.testing.assert_allclose(r_t[1:], log_ret.to_numpy()[:-1], atol=1e-10)
    assert not np.allclose(r_t[1:], log_ret.to_numpy()[1:], atol=1e-10)
