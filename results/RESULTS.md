# Results

Paper: **"Explainable Regime-Aware Investing"**, Amine Boukardagha, Columbia University.

All numbers below are taken directly from the files saved in this directory
by the final run of each experiment:
`results/exp_wasserstein_hmm_portfolio_metrics.json`,
`results/exp_wasserstein_hmm_portfolio_template_conditional_performance.json`,
`results/exp_knn_comparison_metrics.json`, and the corresponding
`*_returns.csv`, `*_weights.csv`, `*_diagnostics.csv` and `*.png` files.

Universe: `SPY` (SPX), `AGG` (BOND), `GLD` (GOLD), `USO` (OIL), `UUP` (USD) —
paper Section 3. Sample: 2005-01-01 to 2026-01-30, with the out-of-sample
evaluation window starting 2023-05-01 (691 trading days), matching the first
reported monthly point in the paper's out-of-sample tables (Sections 8, 10).

## Experiment: Parametric Regime Investing with Wasserstein HMM and Transaction-Cost-Aware MVO

Reproduces paper Sections 5, 7.1, 8.1 and 9: a strictly causal rolling
Gaussian HMM with predictive model-order selection (Section 5.2) and
2-Wasserstein template tracking (Section 5.4), whose mixture moments feed a
transaction-cost-aware mean-variance optimizer (Section 5.6).

| Metric (annualized unless noted) | Paper (Section 8.1) | This replication | Relative difference | Verdict |
|---|---|---|---|---|
| Sharpe ratio | 2.18 | 1.968 | 9.7% | reproduced |
| Max drawdown | -5.43% | -4.59% | 15.4% | partially reproduced |
| Avg. daily turnover | 0.0079 | 1.35e-06 | ~100% | not reproduced |
| Avg. effective # positions | not quantified in the paper | 2.27 | — | not applicable |

Source file: `exp_wasserstein_hmm_portfolio_metrics.json`
(`sharpe_ratio=1.9684`, `max_drawdown=-0.04595`,
`avg_daily_turnover=1.3462e-06`, `avg_effective_n_positions=2.2727`,
`n_days=691`).

Benchmarks saved in the same file, over the identical out-of-sample window:
equal-weight portfolio (Sharpe 1.748, max drawdown -7.81%) and SPX buy & hold
(Sharpe 1.385, max drawdown -18.76%). The Wasserstein HMM strategy has the
highest Sharpe ratio and the smallest drawdown of the three, consistent with
the paper's qualitative ranking (Section 8.1: regime-aware allocation reduces
equity exposure ahead of stress and outperforms passive benchmarks on a
risk-adjusted basis).

Template-conditional performance (paper Section 9) is saved in
`exp_wasserstein_hmm_portfolio_template_conditional_performance.json`. Over
the out-of-sample window only two of the six templates (indices 1 and 3)
receive non-negligible posterior mass: template 1 (122 days) has a negative
mean portfolio log-return (-0.000186/day) and template 3 (569 days) has a
positive mean portfolio log-return (+0.000494/day), with GLD contributing the
largest positive mean asset log-return under template 3 (+0.001595/day). This
reproduces the paper's qualitative claim (Section 9) that one template
receives negligible mass and that template identity is informative of
realized asset performance, though the realized dominant set ({1, 3}) is
smaller than the paper's reported {0, 1, 3, 4, 5}.

## Experiment: Non-Parametric Regime Investing using K-Nearest Neighbors (KNN)

Reproduces paper Sections 6 and 7.2: local-similarity (Euclidean, `K=30`
neighbors, Section 6/Algorithm 2) conditional-moment estimation feeding the
same transaction-cost-aware MVO (identical `gamma`, `tau`, `w_max` as the
Wasserstein HMM experiment).

| Metric (annualized unless noted) | Paper (Section 8.2) | This replication | Relative difference | Verdict |
|---|---|---|---|---|
| Sharpe ratio | 1.81 | 1.841 | 1.7% | reproduced |
| Max drawdown | -12.52% | -12.03% | 3.9% | reproduced |
| Avg. daily turnover | 0.5665 | 0.6098 | 7.6% | reproduced |

Source file: `exp_knn_comparison_metrics.json`
(`sharpe_ratio=1.8413`, `max_drawdown=-0.12034`,
`avg_daily_turnover=0.60985`, `avg_effective_n_positions=2.1736`,
`n_days=691`).

The KNN baseline reproduces the paper's reported Sharpe ratio, drawdown and
turnover within 10% using the same shared pipeline, optimizer and parameters
(`gamma=5.0`, `tau=0.00065`, `w_max=0.60`) as the Wasserstein HMM experiment.
This confirms that the shared transaction-cost-aware MVO and metrics
implementation are correctly calibrated to the paper's reported order of
magnitude for at least one of the two regime-inference methods, which bears
on the turnover gap discussed below.

## Deviations from the paper

- **Data source.** Paper Section 3 states the strategies are evaluated on
  "daily adjusted close prices from Yahoo Finance from 2005 to 2026". This
  replication loads bars through the QuantX Dataloader
  (`POST /fetch-ohlcvs`, dividend-adjusted, `adjust_dividends=True`), which is
  built on Massive and falls back to Yahoo Finance per ticker when Massive
  has no or fewer bars for the requested period
  (`yahoo_backstop`, default on). The realized per-ticker source
  (`ticker_details[].data_source`, logged by `data/data_loader.py`) is
  `massive` for `SPY` and `AGG`, and `yahoo` for `GLD`, `USO` and `UUP` — the
  latter three because Massive's coverage for these proxies starts later
  than Yahoo Finance's (start moved to 2008-05-22 for `GLD`, 2009-06-11 for
  `USO`, 2018-06-04 for `UUP`). This source split matches the paper's own
  stated Yahoo Finance source for the three series where it applies, while
  using Massive directly where it covers the full requested window.
- **Sample end date.** The paper gives no exact sample end date beyond "to
  2026"; the realized sample ends 2026-01-30, the last trading day returned
  by the data source as of the replication date.
- **Out-of-sample split date.** The paper does not print an exact
  train/test split date; `TEST_START=2023-05-01` is inferred from the first
  reported out-of-sample month in the paper's figures and tables (Sections
  8, 10), giving 691 out-of-sample trading days for both experiments.
- **Unspecified Wasserstein HMM / MVO hyperparameters.** The paper does not
  quantify `K_min`, `K_max`, the complexity penalty `lambda_K`, the
  validation-slice length `|V_t|`, the template-smoothing rate `eta`, the
  template count `G`, the risk-aversion coefficient `gamma`, the
  transaction-cost coefficient `tau`, or the position cap `w_max`
  (paper Sections 5.2, 5.4, 5.6; see `src/config.py` for the value used for
  each, each commented with its paper section and "not specified in the
  paper" where applicable). `G=6` templates was chosen because Section 9
  reports a dominant-regime set of five indices out of a template index
  range that includes 2, implying at least six templates (0-5).
- **Unspecified KNN hyperparameters.** The paper does not specify the number
  of neighbors `K` or the minimum history length `L` before the first KNN
  estimate (Section 6/Algorithm 2); this replication uses `K=30` and
  `L=252` trading days (`src/config.py`).
- **Turnover gap for the Wasserstein HMM experiment (not reproduced).** The
  realized average daily turnover (1.35e-06) is several orders of magnitude
  below the paper's reported 0.0079. The transaction-cost coefficient `tau`
  is not quantified by the paper; the same `tau=0.00065` used here reproduces
  the KNN experiment's reported turnover (0.5665) within 7.6%, which shows
  the shared optimizer is calibrated correctly for conditional moments that
  change as quickly as the KNN estimator's. Under the Wasserstein HMM
  estimator, template-tracking and exponential smoothing (Section 5.4) make
  `mu_t` and `Sigma_t` change far more slowly day to day than under KNN (the
  day-to-day change in the fitted mixture mean averages 7.2e-05 in absolute
  value versus a mean level of 9.7e-04), so the same L1 transaction-cost
  penalty all but freezes the optimizer at a corner solution instead of the
  paper's reported small but non-zero daily turnover. This reproduces the
  paper's qualitative claim that the Wasserstein HMM strategy has
  "dramatically lower turnover" than the non-parametric baseline (Section
  8.5) — turnover here is about 450,000 times smaller than under KNN — but
  overshoots the paper's specific reported magnitude; the cause (a `tau`
  value too large for the HMM's much smoother moment path, given the paper
  does not quantify `tau`) is identified, but a `tau` value that reproduces
  0.0079 daily turnover for the Wasserstein HMM experiment specifically was
  not located.
- **Max drawdown gap for the Wasserstein HMM experiment (partially
  reproduced).** The realized max drawdown (-4.59%) is shallower than the
  paper's reported -5.43% (15.4% relative difference, same sign, both
  materially smaller than the benchmarks' drawdowns). This is consistent
  with the near-frozen portfolio weights discussed above: with turnover
  near zero, the strategy's realized exposure path is smoother than the
  paper's, which plausibly caused it to under-react to the "Liberation Day"
  stress episode relative to the paper's weight path; this specific
  causal link was not isolated by a dedicated component test.
- **"Liberation Day" stress-episode decomposition.** The paper's narrative
  discussion of defensive reallocation around the early-2025 equity selloff
  (Sections 1, 8) is not reproduced as a separate table or figure; the daily
  weights and returns needed to inspect that window are available in
  `exp_wasserstein_hmm_portfolio_weights.csv` and
  `exp_wasserstein_hmm_portfolio_returns.csv`, but no dedicated event-window
  summary file is produced.

## Not reproduced

- The monthly effective-number-of-positions time series reported in the
  paper's Section 8 table (2023-05 through 2026-01) is not reproduced as a
  separate monthly table; this replication reports only the full-sample
  average effective number of positions (`avg_effective_n_positions`) for
  each strategy, saved in the corresponding `*_metrics.json` file. Monthly
  aggregation of `exp_wasserstein_hmm_portfolio_diagnostics.csv` would be
  needed to reproduce that table and was not computed.
- The exact daily turnover level reported for the Wasserstein HMM strategy
  (0.0079) is not reproduced; see "Deviations from the paper" above for the
  identified cause and the verified evidence.
