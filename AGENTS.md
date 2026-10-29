# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this implements

A replication of two experiments from **"Explainable Regime-Aware Investing"**
(Amine Boukardagha, Columbia University):

- `exp_wasserstein_hmm_portfolio`: strictly causal rolling Gaussian HMM with
  predictive model-order selection (paper Section 5.2) and 2-Wasserstein
  template tracking (Section 5.4), feeding a transaction-cost-aware
  mean-variance optimizer (Section 5.6).
- `exp_knn_comparison`: a non-parametric K-nearest-neighbors baseline for the
  same optimizer (paper Section 6).

## Layout

- `src/` — flat library modules: `config.py` (all parameters, no classes),
  `features.py` (feature vector `x_t = [r_t; sigma_t; m_t]`), `hmm_regime.py`
  (Wasserstein-HMM regime model), `wasserstein.py` (closed-form 2-Wasserstein
  distance between Gaussians), `knn_regime.py` (KNN conditional-moment
  estimator), `optimizer.py` (transaction-cost-aware MVO via `cvxpy`),
  `backtest.py` (shared rolling-backtest loop), `metrics.py` (Sharpe, Sortino,
  max drawdown, turnover, effective number of positions), `benchmarks.py`,
  `pipeline.py` (data loading/alignment, benchmark construction, metrics
  aggregation, saving outputs).
- `exp/` — one script per experiment, calling only `src/` functions with their
  documented signatures.
- `data/data_loader.py` — the only place that calls the QuantX Dataloader /
  Massive market-data API. No other module loads market data.
- `tests/` — unit tests (synthetic fixtures) and `market_data`-marked tests
  that use a cached, reduced slice loaded through `data/data_loader.py`.
- `results/` — metrics, returns, weights, diagnostics and plots saved
  directly (no subdirectories), plus `RESULTS.md`.

## Conventions

- All experiment parameters live in `src/config.py` as module-level
  UPPER_CASE variables, each with a one-line comment stating its meaning,
  unit, and the paper section it comes from (or "not specified in the
  paper").
- Market data is loaded only through `data/data_loader.py`; no loader or
  generator lives in `src/` or `exp/`.
- API keys (`DATALOADER_API_KEY`, `MASSIVE_API_KEY`) are read from the
  environment via `python-dotenv` and kept out of code, logs and commits;
  only empty placeholders live in `.env.example`.
- Deviations from the paper are recorded only in `results/RESULTS.md`, never
  in code comments or config values beyond a one-line source note.

## Commands

```bash
pip install -r requirements.txt

python -m pytest -m "not market_data"   # fast unit tests
python -m pytest                        # full suite, requires DATALOADER_API_KEY in .env

python -m exp.exp_wasserstein_hmm_portfolio          # default run (full paper universe/period)
python -m exp.exp_wasserstein_hmm_portfolio --full   # explicit paper configuration
python -m exp.exp_knn_comparison
python -m exp.exp_knn_comparison --full
```

`reproduce.sh` runs the fast test suite and both experiments in sequence.

## After a rerun

Running an experiment overwrites its files in `results/`
(`<experiment>_metrics.json`, `_returns.csv`, `_weights.csv`,
`_diagnostics.csv`, plots, and — for the HMM experiment —
`_template_conditional_performance.json`). The numbers and verdicts in
`results/RESULTS.md` are not regenerated automatically: after a rerun, check
them against the new files in `results/` and update `results/RESULTS.md` by
hand.
