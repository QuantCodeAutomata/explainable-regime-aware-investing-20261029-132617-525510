# Explainable Regime-Aware Investing — Replication

Paper: **"Explainable Regime-Aware Investing"**, Amine Boukardagha, Columbia University.

This repository reproduces two experiments from the paper:

1. **Parametric Regime Investing with Wasserstein HMM and Transaction-Cost-Aware MVO**
   (paper Sections 5, 7.1, 8.1, 9) — a strictly causal rolling Gaussian HMM with
   predictive model-order selection (Section 5.2) and 2-Wasserstein template
   tracking (Section 5.4), feeding a transaction-cost-aware mean-variance
   optimizer (Section 5.6).
2. **Non-Parametric Regime Investing using K-Nearest Neighbors (KNN)**
   (paper Section 6, 7.2) — a local-similarity baseline that estimates
   conditional moments from the K nearest historical feature vectors and feeds
   the same optimizer.

## Repository layout

```
src/        shared library: data pipeline, features, HMM/KNN regime models,
            Wasserstein geometry, optimizer, backtest loop, metrics, config
exp/        experiment entry points (python -m exp.<name>)
data/       data_loader.py — market data access (QuantX Dataloader / Massive)
tests/      pytest unit and market-data tests
results/    saved metrics, weights, returns, diagnostics and plots
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY / MASSIVE_API_KEY
```

## Data

Daily adjusted close prices for a cross-asset universe are loaded through
`data/data_loader.py`:

- S&P 500 proxy: `SPY` (SPX)
- Broad bond proxy: `AGG` (BOND)
- Gold: `GLD`
- Oil: `USO`
- U.S. Dollar proxy: `UUP`

```text
Market data: QuantX Dataloader — https://quantx.limex.pro/

Price history in this project is loaded through the QuantX Dataloader, a
research-ready market-data service available through a REST API and as an
MCP server for AI agents. It is built on market data from Massive
(https://massive.com/) and prepares every stock history for backtesting:
  - prices adjusted for dividends, ready for total-return research;
  - one continuous history when a company changes its ticker;
  - clean separation when a ticker has belonged to different companies;
  - careful corporate-action handling and fine-grained data polishing.

Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
MASSIVE_API_KEY at https://massive.com/.
```

Bars are requested dividend-adjusted (`adjust_dividends=True`), matching the
paper's "daily adjusted close prices" (Section 3). The sample period is
2005-01-01 to 2026-01-30 (paper Section 3: "2005 to 2026"; the paper gives no
exact end date, so the realized sample ends at the last available trading
day). See `results/RESULTS.md` for the realized data source per ticker.

## Running tests

```bash
python -m pytest -m "not market_data"   # fast unit tests, no API key needed
python -m pytest                        # full suite, requires DATALOADER_API_KEY
```

## Running the experiments

```bash
python -m exp.exp_wasserstein_hmm_portfolio
python -m exp.exp_knn_comparison
```

Both scripts accept `--full`, which selects the paper's exact universe and
sample period (`src/config.py`: `UNIVERSE_SIZE_PAPER`, `SAMPLE_START`,
`SAMPLE_END`). The default (no flag) run already uses the full paper universe
and sample period, because downloading the five-ticker daily universe takes
well under the time available (see "Deviations from the paper" in
`results/RESULTS.md`); `--full` is kept for parity with the config convention.

Each run overwrites `results/<experiment>_metrics.json`,
`results/<experiment>_returns.csv`, `results/<experiment>_weights.csv`,
`results/<experiment>_diagnostics.csv`, and the corresponding plots. The
Wasserstein HMM experiment additionally saves
`results/exp_wasserstein_hmm_portfolio_template_conditional_performance.json`
(paper Section 9, template-conditional asset performance).

Running `reproduce.sh` from the repository root executes the test suite and
both experiments in sequence.

## Results

See `results/RESULTS.md` for the full comparison against the paper's reported
Sharpe ratio, Sortino ratio, maximum drawdown, turnover and effective number
of positions (Sections 8.1, 8.2, 10), together with the "Deviations from the
paper" and "Not reproduced" sections.
