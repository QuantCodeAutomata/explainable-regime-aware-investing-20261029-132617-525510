"""Experiment parameters for the Wasserstein-HMM and KNN regime-investing replications.

Source: "Explainable Regime-Aware Investing", Amine Boukardagha, Columbia University.
All module-level variables are plain values (no classes/dicts) so that every
parameter can be traced to a section of the paper, or flagged as not specified.
"""

# ---------------------------------------------------------------------------
# Universe and sample period (paper Section 3 "Data and Market Universe")
# ---------------------------------------------------------------------------
# The paper evaluates a diversified cross-asset universe: S&P 500 proxy (SPX),
# broad bond proxy (BOND), gold (GOLD), oil (OIL) and a U.S. dollar proxy (USD).
TICKERS = ["SPY", "AGG", "GLD", "USO", "UUP"]  # [SPX, BOND, GOLD, OIL, USD] proxies, Section 3
ASSET_LABELS = {"SPY": "SPX", "AGG": "BOND", "GLD": "GOLD", "USO": "OIL", "UUP": "USD"}

SAMPLE_START = "2005-01-01"  # Paper Section 3: "daily adjusted close prices ... from 2005 to 2026"
SAMPLE_END = "2026-01-30"  # Last trading day available in the realized sample (paper gives no exact end date)

# Out-of-sample (test) split date. The paper evaluates all strategies over an
# "identical strictly out-of-sample (OOS) period" (Section 10) whose monthly
# tables (Figures 1, 7, 8, 11) begin at 2023-05 (cumulative PnL = 0) and end at
# 2026-01. The exact split date t0 is not printed in the text; it is inferred
# from the first reported OOS month.
TRAIN_END = "2023-04-30"  # last in-sample day (paper does not print an exact value)
TEST_START = "2023-05-01"  # first OOS day, inferred from Figure 1 / Table 7 (Section 8.1, 10.1)
TEST_END = SAMPLE_END

# ---------------------------------------------------------------------------
# Feature representation (paper Section 4 "Feature Representation")
# ---------------------------------------------------------------------------
VOL_WINDOW = 60  # sigma_t: 60-day rolling volatility of log returns, Section 4
MOM_WINDOW = 20  # m_t: 20-day rolling mean of log returns, Section 4

# ---------------------------------------------------------------------------
# Section 5: Wasserstein Hidden Markov Model (parametric regime detection)
# ---------------------------------------------------------------------------
HMM_K_MIN = 2  # K_min, lower bound of candidate HMM order grid, Section 5.2/7.1 (value not specified in the paper)
HMM_K_MAX = 6  # K_max, upper bound of candidate HMM order grid; set equal to the template count G, not specified in the paper
HMM_ORDER_SELECTION_FREQ_DAYS = 5  # F_K, weekly re-selection, Section 5.2 ("e.g., weekly")
HMM_VALIDATION_SLICE_LEN = 60  # |V_t|, length of the recent validation slice used in PredLL, Section 5.2 (not specified in the paper; set to one quarter)
HMM_COMPLEXITY_LAMBDA = 1.0  # lambda_K, complexity-penalty weight, Section 5.2 (not specified in the paper; scales a BIC-style penalty)
HMM_COVARIANCE_TYPE = "full"  # Needed so Sigma_t carries cross-asset correlation used by the MVO risk term, Section 5.3/5.6
HMM_N_ITER = 30  # EM iterations per daily refit, not specified in the paper; bounded for tractable daily re-estimation
HMM_TOL = 1e-3  # EM convergence tolerance, not specified in the paper
HMM_MIN_COVAR = 1e-4  # Covariance floor for numerical stability, not specified in the paper
HMM_MIN_TRAIN_OBS = 250  # Minimum history length (approx. one year) before the first daily HMM fit, not specified in the paper

# Section 5.4/5.5: Wasserstein template tracking
TEMPLATE_COUNT = 6  # G, number of persistent templates. Section 9 reports a realized dominant-regime set {0,1,3,4,5}, i.e. 6 template indices (0-5)
TEMPLATE_SMOOTHING_ETA = 0.10  # eta, exponential smoothing rate for template updates, Section 5.4 (not specified in the paper)
TEMPLATE_CALIBRATION_WINDOW = 250  # Length of the initial calibration window used to seed templates, Section 7.1 step 3 (not specified in the paper)

# ---------------------------------------------------------------------------
# Section 5.6 / 7.1-7.2: Transaction-cost-aware mean-variance optimization
# ---------------------------------------------------------------------------
MVO_GAMMA = 5.0  # gamma, risk-aversion coefficient, Section 5.6 (not quantified in the paper)
MVO_TAU = 0.00065  # tau, L1 transaction-cost coefficient, Section 5.6 (not quantified in the paper;
# calibrated so that the L1 no-trade band of the transaction-cost penalty yields the
# order of magnitude of daily turnover the paper reports for the Wasserstein HMM
# strategy, verified on synthetic time-varying mu with a fixed Sigma: tau=0.0010
# freezes weights at a corner solution (turnover ~1e-6), while tau=0.00065 yields
# turnover of the same order as the paper's reported value, see results/RESULTS.md)
MVO_W_MAX = 0.60  # w_max, per-asset weight cap, Section 5.6 (not quantified in the paper)

# ---------------------------------------------------------------------------
# Section 6 / 7.2: Non-parametric KNN regime inference
# ---------------------------------------------------------------------------
KNN_N_NEIGHBORS = 30  # K, number of nearest neighbors, Section 6/Algorithm 2 (not specified in the paper)
KNN_LOOKBACK_L = 252  # L, minimum history (approx. one trading year) before the first KNN estimate, Algorithm 2 (not specified in the paper)

# ---------------------------------------------------------------------------
# Diagnostics and performance evaluation (paper Section 7.1/7.2 "Diagnostics", Section 8/10)
# ---------------------------------------------------------------------------
ANNUALIZATION_DAYS = 252  # Trading days per year used to annualize Sharpe/Sortino, standard convention; not specified in the paper
RISK_FREE_RATE_ANNUAL = 0.0  # Risk-free rate used in Sharpe/Sortino, not specified in the paper

EQUAL_WEIGHT = 1.0 / len(TICKERS)  # Section 10: "Equal-Weight Portfolio (20% each)"
BENCHMARK_SPX_TICKER = "SPY"  # Section 10: "SPX Buy & Hold"

# ---------------------------------------------------------------------------
# Download-time / execution-scope controls
# ---------------------------------------------------------------------------
# The full paper universe (5 tickers, 2005-2026 daily bars) downloads quickly,
# so no narrowing of the universe or sample period is required; UNIVERSE_SIZE_RUN
# equals UNIVERSE_SIZE_PAPER.
UNIVERSE_SIZE_PAPER = 5
UNIVERSE_SIZE_RUN = 5
