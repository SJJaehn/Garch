"""
Rolling-window portfolio backtest (the whole pipeline in one script).

We compare portfolio construction methods (MVP, HRP, ERC) using several
covariance estimators (historical, GARCH constant-correlation, DCC), plus a
naive 1/N benchmark.

Returns are LOG returns for the modelling part (GARCH/DCC like additive,
well-behaved returns). For evaluation and plotting we convert back to SIMPLE
returns, because a portfolio return is a weighted sum of *simple* asset returns,
not of log returns.

The engine is data-in / files-out: ``run_backtest`` takes the config and the
already-loaded data, so nothing is read at import and there is no global state.
Each rolling window is independent and deterministic, so it is farmed out to a
process pool; workers receive the (read-only) returns once via an initializer.

Run order in this file: data loading -> models (GARCH / covariance / portfolio)
-> metrics & losses -> the rolling-window engine. This module is a library; the
batch run (which datasets / horizons) is driven from ``main.py``:

    python main.py            # edit the SETTINGS block there to choose runs

Worker processes re-import this module (and ``main``) under the 'spawn' start
method (macOS default), which is safe because importing either has no side effects.
"""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import riskfolio as rp
from arch import arch_model
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

import main as config


# =============================================================================
# Data loading: prices, log returns and the risk-free series
# =============================================================================
# All functions are side-effect free (nothing runs at import) and take their
# inputs explicitly, so the same loaders are reused by analyze.py.

def load_prices(dataset: str) -> pd.DataFrame:
    """Load a dataset's price frame (assets in columns, dates in the index)."""
    filepath, date_format = config.DATASETS[dataset]
    prices = pd.read_csv(filepath, index_col=0)
    prices.index = pd.to_datetime(prices.index, format=date_format, errors="coerce")
    prices = prices[prices.index.notna()]
    # keep dates where at least half the assets have a price
    prices = prices.loc[prices.notna().sum(axis=1) >= int(0.5 * prices.shape[1])]
    return prices


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns r_t = ln(P_t / P_{t-1}); drop assets that are flat >50% of the time."""
    log_returns = np.log(prices / prices.shift(1)).iloc[1:]
    zero_frac = (log_returns == 0).sum() / log_returns.notna().sum()
    return log_returns.loc[:, zero_frac < 0.5]


_GERMAN_MONTHS = {"Mär": "Mar", "Mai": "May", "Okt": "Oct", "Dez": "Dec"}


def read_risk_free_level(risk_free_file: str | None = None) -> pd.Series:
    """Read the Fed Funds total-return *index level*, indexed by date (sorted)."""
    path = risk_free_file or config.risk_free_file()
    rf = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    dates = rf["Exchange Date"]
    for de, en in _GERMAN_MONTHS.items():
        dates = dates.str.replace(de, en, regex=False)  # German -> English months
    level = pd.Series(rf["Close"].values, index=pd.to_datetime(dates, format="%d-%b-%Y"))
    return level.sort_index()


def align_risk_free(level: pd.Series, index) -> pd.Series:
    """
    Daily SIMPLE risk-free return aligned to ``index``.

    The metrics work on simple returns, so we align the index *level* to the
    target dates (forward-filling gaps) and take its percentage change, so the
    risk-free accrual spans exactly the same day spacing as the portfolio returns.
    """
    return level.reindex(index).ffill().pct_change()


def load_risk_free(price_index, risk_free_file: str | None = None) -> pd.Series:
    """Convenience: read the level and align it to ``price_index`` in one step."""
    return align_risk_free(read_risk_free_level(risk_free_file), price_index)


def load_dataset(dataset: str, risk_free_file: str | None = None):
    """Return (prices, log_returns, rf_daily) for a dataset, rf aligned to prices."""
    prices = load_prices(dataset)
    log_returns = to_log_returns(prices)
    rf = load_risk_free(prices.index, risk_free_file)
    return prices, log_returns, rf


# =============================================================================
# Univariate GARCH fitting
# =============================================================================
# A single fit_garch_univariate call fits a GARCH(p, q) to every asset and
# returns both the forecast variances and the standardized residuals, so the
# GARCH (constant-correlation) and DCC covariance estimators can share one fit.

def _ewma_fallback(series, lam=0.94):
    """
    RiskMetrics EWMA variance, used when the GARCH fit does not converge.
    Returns (forecast_variance, standardized_residuals).  EWMA assumes the
    variance is flat going forward, so the horizon forecast is just the last value.
    """
    ewma_var = series.pow(2).ewm(alpha=1 - lam).mean()
    cond_vol = np.sqrt(ewma_var).replace(0.0, np.nan)
    return float(ewma_var.iloc[-1]), series / cond_vol


def fit_garch_univariate(returns, prediction_window=1, p=1, q=1):
    """
    Fit a univariate GARCH(p, q) to every asset.

    Returns (variances, std_resid):
      - variances : Series of forecast variances (back on the raw return scale)
      - std_resid : DataFrame of standardized residuals (assets in columns)

    Non-convergence handling: retry the fit with more iterations, and if it still
    fails fall back to an EWMA variance so the asset is kept (rather than silently
    dropped, which would shrink the GARCH/DCC universe relative to Historical).
    """
    horizon = max(prediction_window, 1)
    variances = {}
    resid = {}
    for col in returns.columns:
        series = returns[col].dropna()
        if len(series) < 30:
            continue  # genuinely too little data to model this asset

        var = sr = None
        if len(series) >= 50:
            try:
                # arch is happier with percent returns; undo the scaling with /100**2
                model = arch_model(series * 100, vol="GARCH", p=p, q=q)
                res = model.fit(disp="off", show_warning=False)
                if res.convergence_flag != 0:  # retry with a bigger iteration budget
                    res = model.fit(disp="off", show_warning=False,
                                    options={"maxiter": 2000, "ftol": 1e-6})
                if res.convergence_flag == 0:
                    var = res.forecast(horizon=horizon).variance.iloc[-1].mean() / (100 ** 2)
                    sr = res.std_resid
            except Exception:
                var = None

        if var is None:  # GARCH failed / too little data -> EWMA fallback
            reason = "insufficient data" if len(series) < 50 else "GARCH did not converge"
            print(f"[fallback] GARCH->EWMA for {col} ({reason})", flush=True)
            var, sr = _ewma_fallback(series)

        variances[col] = var
        resid[col] = sr

    return pd.Series(variances, dtype=float), pd.DataFrame(resid)


# =============================================================================
# Covariance estimators
# =============================================================================
# Each takes (log) returns and/or GARCH outputs and returns a covariance
# DataFrame (assets in both axes).

def historical_covariance(returns):
    """Plain historical sample covariance."""
    return returns.cov()


def constant_correlation_covariance(returns, variances):
    """GARCH variances + static historical correlation -> covariance matrix."""
    cols = list(variances.index)
    if len(cols) == 0:
        return pd.DataFrame()

    std = np.sqrt(variances.values)
    corr = returns[cols].corr().fillna(0.0).values
    np.fill_diagonal(corr, 1.0)

    cov = corr * np.outer(std, std)
    cov += np.eye(len(cols)) * 1e-8  # tiny jitter for numerical stability
    return pd.DataFrame(cov, index=cols, columns=cols)


def dcc_covariance(variances, std_resid):
    """
    DCC-GARCH covariance (Engle 2002): combine the GARCH forecast variances with
    a Dynamic Conditional Correlation model estimated on the standardized
    residuals.

      Q_t = (1 - a - b) * Qbar + a * z_{t-1} z_{t-1}' + b * Q_{t-1}
      R_t = normalise(Q_t),   Sigma = D * R * D   (D = diag of forecast std)

    Speed: the log-likelihood uses a Cholesky factorisation for the log-det and
    quadratic form, and we optimise the two parameters (a, b) with L-BFGS-B, so
    only a handful of likelihood evaluations are needed.
    """
    cols = list(std_resid.columns)
    if len(cols) == 0:
        return pd.DataFrame()

    z = std_resid.dropna().values            # common dates only
    n_obs, n = z.shape
    std = np.sqrt(variances[cols].values)

    # too few assets / observations for a correlation model -> diagonal cov
    if n < 2 or n_obs < n + 2:
        print(f"[fallback] DCC->diagonal covariance (n={n}, n_obs={n_obs})", flush=True)
        cov = np.diag(std ** 2) + np.eye(n) * 1e-8
        return pd.DataFrame(cov, index=cols, columns=cols)

    q_bar = np.corrcoef(z, rowvar=False)     # unconditional correlation of the z's

    def neg_loglik(params):
        a, b = params
        if a <= 0 or b <= 0 or a + b >= 0.9999:
            return 1e12
        omega = (1 - a - b) * q_bar
        Q = q_bar.copy()
        ll = 0.0
        for t in range(n_obs):
            d = np.sqrt(np.diag(Q))
            R = Q / np.outer(d, d)
            try:
                c, low = cho_factor(R, lower=True, check_finite=False)
            except Exception:
                return 1e12
            logdet = 2.0 * np.log(np.diag(c)).sum()
            ll += logdet + z[t] @ cho_solve((c, low), z[t], check_finite=False)
            Q = omega + a * np.outer(z[t], z[t]) + b * Q
        return 0.5 * ll

    opt = minimize(neg_loglik, x0=[0.02, 0.95], method="L-BFGS-B",
                   bounds=[(1e-4, 0.3), (1e-4, 0.999)], options={"maxiter": 50})
    a, b = opt.x
    if not (a > 0 and b > 0 and a + b < 1):
        print(f"[fallback] DCC optimizer misbehaved (a={a:.3g}, b={b:.3g}); using a=0.02, b=0.95", flush=True)
        a, b = 0.02, 0.95  # typical values if the optimizer misbehaves

    # roll the recursion through the sample to get the next-step correlation
    omega = (1 - a - b) * q_bar
    Q = q_bar.copy()
    for t in range(n_obs):
        Q = omega + a * np.outer(z[t], z[t]) + b * Q
    d = np.sqrt(np.diag(Q))
    R = Q / np.outer(d, d)

    cov = R * np.outer(std, std)
    cov += np.eye(n) * 1e-8
    return pd.DataFrame(cov, index=cols, columns=cols)


# =============================================================================
# Portfolio weighting schemes
# =============================================================================
# MVP and ERC go through riskfolio-lib with our *externally estimated* covariance
# (GARCH / DCC / historical) injected directly. HRP stays self-implemented
# because no library accepts a user-supplied covariance for it.

def _make_portfolio(cov_matrix):
    """
    Build a riskfolio ``Portfolio`` with our covariance injected and a stable
    solver configuration (see config.SOLVERS / SOL_PARAMS for the cvxpy backend
    note). Returns (port, cols).
    """
    cols = list(cov_matrix.index)
    n = len(cols)

    cov = np.asarray(cov_matrix, dtype=float)
    cov = (cov + cov.T) / 2.0 + np.eye(n) * 1e-8   # symmetric + jitter -> PSD/stable
    cov = pd.DataFrame(cov, index=cols, columns=cols)

    # riskfolio needs a returns frame for the asset names/shape; under
    # model='Classic', hist=True the optimisation reads port.cov / port.mu only,
    # so the actual return values are unused.
    dummy = pd.DataFrame(np.zeros((2, n)), columns=cols)
    port = rp.Portfolio(returns=dummy)
    port.cov = cov
    port.mu = pd.DataFrame(np.zeros((1, n)), columns=cols)
    port.solvers = list(config.SOLVERS)
    port.sol_params = {k: dict(v) for k, v in config.SOL_PARAMS.items()}
    return port, cols


def _clean_weights(w, cols, index):
    """Turn a riskfolio weights frame into a normalised, long-only Series."""
    if w is None or getattr(w, "empty", True):
        warnings.warn("riskfolio optimisation returned no solution; using equal weights")
        return pd.Series(1.0 / len(cols), index=index)
    s = w["weights"].reindex(cols).fillna(0.0).clip(lower=0.0)
    total = s.sum()
    if total <= 0:
        return pd.Series(1.0 / len(cols), index=index)
    return (s / total).reindex(index)


def naive_weights(assets):
    """Equal-weight (1/N) portfolio over ``assets``."""
    assets = list(assets)
    n = len(assets)
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / n, index=assets)


def mvp_weights(cov_matrix):
    """Long-only Minimum Variance Portfolio (riskfolio MinRisk, rm='MV')."""
    if len(cov_matrix) == 0:
        return pd.Series(dtype=float)
    port, cols = _make_portfolio(cov_matrix)
    w = port.optimization(model="Classic", rm="MV", obj="MinRisk", hist=True)
    return _clean_weights(w, cols, cov_matrix.index)


def erc_weights(cov_matrix):
    """Equal Risk Contribution portfolio (riskfolio risk parity, rm='MV')."""
    if len(cov_matrix) == 0:
        return pd.Series(dtype=float)
    port, cols = _make_portfolio(cov_matrix)
    w = port.rp_optimization(model="Classic", rm="MV", b=None, hist=True)
    return _clean_weights(w, cols, cov_matrix.index)


def hrp_weights(cov_matrix):
    """Hierarchical Risk Parity (Lopez de Prado 2016)."""
    n = len(cov_matrix)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=cov_matrix.index)

    cov = cov_matrix.copy()
    cov = (cov + cov.T) / 2.0  # make sure it is symmetric

    # 1) correlation distance matrix
    std = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))

    # 2) hierarchical clustering -> quasi-diagonal ordering of the assets
    link = linkage(squareform(dist, checks=False), method="single")
    order = leaves_list(link)
    assets = list(cov.index[order])

    def cluster_variance(items):
        sub = cov.loc[items, items].values
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    # 3) recursive bisection: split the ordering in half, give more weight to the
    #    lower-variance half each time
    weights = pd.Series(1.0, index=assets)
    clusters = [assets]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        half = len(cluster) // 2
        left, right = cluster[:half], cluster[half:]
        var_left = cluster_variance(left)
        var_right = cluster_variance(right)
        alpha = 1.0 - var_left / (var_left + var_right)
        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
        clusters += [left, right]

    return (weights / weights.sum()).reindex(cov_matrix.index)


_WEIGHTERS = {
    "MVP": mvp_weights,
    "HRP": hrp_weights,
    "ERC": erc_weights,
}


def get_weights(model, cov_matrix):
    """Dispatch to the weighting scheme named by ``model`` (MVP / HRP / ERC)."""
    try:
        return _WEIGHTERS[model](cov_matrix)
    except KeyError:
        raise ValueError(f"unknown model: {model}")


# =============================================================================
# Performance metrics and covariance-forecast losses
# =============================================================================

def calculate_summary_metrics(daily_returns, rf_daily=0.0):
    """
    Annualised performance statistics from a series of simple daily returns.

    rf_daily is the SIMPLE daily risk-free return (a scalar, or an array aligned
    to daily_returns), subtracted to get excess returns for Sharpe / Sortino.
    """
    arr = np.asarray(daily_returns)
    rf = np.asarray(rf_daily)
    excess = arr - rf

    ann_ret = np.prod(1 + arr) ** (252 / len(arr)) - 1   # CAGR (geometric)
    ann_std = arr.std(ddof=1) * np.sqrt(252)             # sample std (raw, = portfolio vol)

    # arithmetic annualised excess return, consistent with the std used below
    ann_excess = excess.mean() * 252
    # std of the EXCESS returns for the Sharpe denominator (textbook definition)
    ann_excess_std = excess.std(ddof=1) * np.sqrt(252)
    # downside deviation of the excess returns
    semi_dev = np.sqrt(np.mean(np.minimum(excess, 0) ** 2)) * np.sqrt(252)

    cumulative = np.cumprod(1 + arr)
    drawdowns = cumulative / np.maximum.accumulate(cumulative) - 1
    max_dd = drawdowns.min()

    var_95 = np.percentile(arr, 5)
    cvar_95 = arr[arr <= var_95].mean()

    return {
        "Ann. Return":     ann_ret,
        "Ann. Std":        ann_std,
        "Ann. Sharpe":     ann_excess / ann_excess_std if ann_excess_std > 0 else np.nan,
        "Ann. Sharpe (rf=0)": (arr.mean() * 252) / ann_std if ann_std > 0 else np.nan,
        "Ann. Sortino":    ann_excess / semi_dev if semi_dev > 0 else np.nan,
        "Max Drawdown":    max_dd,
        "Calmar Ratio":    ann_ret / abs(max_dd) if max_dd < 0 else np.nan,
        "CVaR (95%)":      cvar_95,
        "Skewness":        pd.Series(arr).skew(),
        "Excess Kurtosis": pd.Series(arr).kurt(),
    }


def _qlike(cov, realized):
    """
    Multivariate QLIKE loss of a forecast covariance ``cov`` (H) against the
    realized returns over the test window (Patton 2011; Laurent et al. 2012):

        QLIKE = log|H| + tr(H^{-1} S),   S = (1/pw) * sum_t r_t r_t'

    S is the realized (uncentered) second-moment proxy of the test-window log
    returns. The loss is minimised, in expectation, when H equals the true
    covariance, so a lower QLIKE means a better covariance forecast. Returns NaN
    if H is not positive definite / singular.
    """
    H = np.asarray(cov, dtype=float)
    R = np.asarray(realized, dtype=float)          # (pw, n) test-window log returns
    if H.shape[0] == 0 or R.shape[0] == 0:
        return np.nan
    S = R.T @ R / R.shape[0]                        # (n, n) average second moment
    sign, logdet = np.linalg.slogdet(H)
    if sign <= 0:
        return np.nan
    try:
        return float(logdet + np.trace(np.linalg.solve(H, S)))
    except np.linalg.LinAlgError:
        return np.nan


def _cov_rmse(cov, realized):
    """
    Frobenius RMSE of a forecast covariance ``cov`` (H) against the realized
    second-moment proxy  S = (1/pw) * sum_t r_t r_t'  of the test-window returns:

        Cov RMSE = sqrt( mean_{i,j} (H_ij - S_ij)^2 ) = ||H - S||_F / N

    i.e. the typical per-element error between the forecast and realized covariance
    (the Frobenius / Euclidean loss; Patton & Sheppard 2009). Lower is better. As
    with QLIKE the realized proxy is noisy for pw=1, so the level is meaningful only
    in relative terms across estimators (E[S] is the true covariance, so the noise
    averages out across windows).
    """
    H = np.asarray(cov, dtype=float)
    R = np.asarray(realized, dtype=float)
    if H.shape[0] == 0 or R.shape[0] == 0:
        return np.nan
    S = R.T @ R / R.shape[0]
    return float(np.sqrt(np.mean((H - S) ** 2)))


def _risk_contribution_rmse(weights, realized):
    """
    RMSE of the realized risk contributions from the equal-risk target (1/N).

    Using the test-window realized second moment  S = (1/pw) * sum_t r_t r_t'  as
    the covariance proxy, asset i's relative risk contribution is
        RC_i = w_i (S w)_i / (w' S w),   with  sum_i RC_i = 1.
    An Equal-Risk-Contribution portfolio targets RC_i = 1/N, so a lower RMSE means
    the weights kept risk closer to balanced *out-of-sample* -> a better covariance
    forecast. Returns NaN if the realized portfolio variance is ~0.
    """
    w = np.asarray(weights, dtype=float)
    n = w.shape[0]
    R = np.asarray(realized, dtype=float)
    if n == 0 or R.shape[0] == 0:
        return np.nan
    S = R.T @ R / R.shape[0]
    port_var = float(w @ S @ w)
    if port_var <= 1e-300:
        return np.nan
    rc = w * (S @ w) / port_var
    return float(np.sqrt(np.mean((rc - 1.0 / n) ** 2)))


# =============================================================================
# Rolling-window engine
# =============================================================================
# Read-only per-worker state, populated once by _init_worker (avoids pickling the
# returns frame on every task and works under both 'spawn' and 'fork').
_WORKER: dict = {}


def _init_worker(log_returns, cfg):
    _WORKER["log_returns"] = log_returns
    _WORKER["config"] = cfg


def process_window(start):
    """Process-pool entry point: run one window using the worker's shared state."""
    return run_window(start, _WORKER["log_returns"], _WORKER["config"])


def run_window(start, log_returns, cfg):
    """
    Run one rolling window (pure: data + config in, results out).

    Returns (per_period, per_period_log, records, dates, weights_info,
    formation_date, qlike_info, covrmse_info).
    """
    tw, pw = cfg.train_window, cfg.prediction_window
    train = log_returns.iloc[start : start + tw]
    test  = log_returns.iloc[start + tw : start + tw + pw]

    # Universe is decided from TRAINING information only (no look-ahead into the
    # test window): assets with >=90% observations in the window and a valid
    # observation on the last training day (i.e. tradeable at formation time).
    train = train.loc[:, train.notna().mean() >= 0.9]
    train = train.loc[:, train.iloc[-1].notna()]
    if train.shape[1] == 0 or test.empty:
        return {}, [], [], {}, None, {}, {}

    # same universe in the test window; a missing test return means the asset did
    # not trade that day, so its return is 0 (we don't drop it after the fact).
    test = test[train.columns]
    test_simple = (np.exp(test) - 1).fillna(0.0)
    # log-return portfolio proxy: the forecast std is sqrt(w'Σw) with Σ the
    # covariance of LOG returns, so its realized counterpart is the std of the
    # weighted sum of log returns (same scale -> apples-to-apples calibration).
    test_log = test.fillna(0.0)

    # build the requested covariance matrices; fit GARCH only once and reuse it
    cov_by_method = {}
    if "Historical" in cfg.cov_methods:
        cov_by_method["Historical"] = historical_covariance(train)
    if "GARCH" in cfg.cov_methods or "DCC" in cfg.cov_methods:
        variances, std_resid = fit_garch_univariate(
            train, cfg.prediction_window, cfg.garch_p, cfg.garch_q)
        if "GARCH" in cfg.cov_methods:
            cov_by_method["GARCH"] = constant_correlation_covariance(train, variances)
        if "DCC" in cfg.cov_methods:
            cov_by_method["DCC"] = dcc_covariance(variances, std_resid)

    # per-step covariance-forecast quality: QLIKE and Frobenius RMSE of each
    # estimator's matrix against the realized test-window returns (lower is better).
    qlike_info = {}
    covrmse_info = {}
    for method, cov in cov_by_method.items():
        if cov is None or cov.shape[1] == 0:
            continue
        realized = test[list(cov.columns)].fillna(0.0).values
        qlike_info[method] = _qlike(cov.values, realized)
        covrmse_info[method] = _cov_rmse(cov.values, realized)

    # historical cov is also used for the naive portfolio's forecast std
    cov_hist_full = cov_by_method.get("Historical")
    if cov_hist_full is None:
        cov_hist_full = historical_covariance(train)

    # list of (model, cov_type, weights, cov) to evaluate
    naive_w = naive_weights(train.columns)
    jobs = [("Naive", "N/A", naive_w, cov_hist_full)]
    for method, cov in cov_by_method.items():
        if cov is None or cov.shape[1] == 0:
            continue
        for model in cfg.models:
            jobs.append((model, method, get_weights(model, cov), cov))

    per_period = {}                         # name -> simple-return series (performance)
    per_period_log = {}                     # name -> log-return series (calibration)
    records = []
    weights_info = {}                       # name -> (target weights, drifted end weights)
    formation_date = test.index[0]          # rebalance date for this window
    for model, cov_type, weights, cov in jobs:
        name = "Naive" if model == "Naive" else f"{model} {cov_type}"
        cols = list(weights.index)
        port_returns = test_simple[cols] @ weights
        per_period[name] = port_returns.values.tolist()
        per_period_log[name] = (test_log[cols] @ weights).values.tolist()

        # drifted end-of-window weights (for turnover): each asset's start weight
        # grows with its gross return over the holding window, then renormalise.
        gross = (1.0 + test_simple[cols]).prod(axis=0)
        end_val = weights * gross
        end_weights = end_val / end_val.sum()
        weights_info[name] = (weights, end_weights)

        w = weights.values
        forecasted_std = float(np.sqrt(w @ cov.loc[cols, cols].values @ w))
        # ERC quality: realized risk-contribution RMSE from the 1/N target
        # (out-of-sample; only the ERC objective targets equal risk contributions)
        rc_rmse = np.nan
        if model == "ERC":
            rc_rmse = _risk_contribution_rmse(w, test[cols].fillna(0.0).values)
        records.append({
            "Model": model,
            "Covariance Type": cov_type,
            "Window Index": start // cfg.prediction_window,
            "Mean Return": port_returns.mean(),
            "Forecasted Std": forecasted_std,
            "RC RMSE": rc_rmse,
        })
    return (per_period, per_period_log, records, list(test.index), weights_info,
            formation_date, qlike_info, covrmse_info)


def run_backtest(cfg, log_returns, rf, verbose=True):
    """
    Run the full rolling-window backtest for ``cfg`` over ``log_returns`` and
    write the results into ``cfg.output_dir``. ``rf`` is the daily simple
    risk-free return (aligned to the price dates). Returns the summary DataFrame.
    """
    starts = list(range(0, len(log_returns) - cfg.train_window - cfg.prediction_window + 1,
                        cfg.prediction_window))
    total = len(starts)

    results = {}          # name -> list of simple period returns (performance)
    results_log = {}      # name -> list of log period returns (vol calibration)
    records = []
    period_dates = []
    weights_hist = {}     # name -> list of (formation_date, target weights) in window order
    end_hist = {}         # name -> list of drifted end-of-window weights (same order)
    qlike_hist = []       # list of (formation_date, {cov_type: qlike}) in window order
    covrmse_hist = []     # list of (formation_date, {cov_type: cov rmse}) in window order
    with ProcessPoolExecutor(max_workers=cfg.max_workers,
                             initializer=_init_worker,
                             initargs=(log_returns, cfg)) as executor:
        for i, (per_period, per_period_log, recs, dates, winfo, fdate, qinfo, crinfo) in enumerate(
                executor.map(process_window, starts), start=1):
            records.extend(recs)
            period_dates.extend(dates)
            for name, rets in per_period.items():
                results.setdefault(name, []).extend(rets)
            for name, rets in per_period_log.items():
                results_log.setdefault(name, []).extend(rets)
            for name, (target_w, end_w) in winfo.items():
                weights_hist.setdefault(name, []).append((fdate, target_w))
                end_hist.setdefault(name, []).append(end_w)
            if qinfo:
                qlike_hist.append((fdate, qinfo))
            if crinfo:
                covrmse_hist.append((fdate, crinfo))
            if verbose:
                print(f"\r{i}/{total} windows completed", end="", flush=True)
    if verbose:
        print()

    if not records:
        print("No valid rolling windows (train window longer than the data?).")
        return None

    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    metrics = pd.DataFrame(records)
    metrics.to_csv(f"{out_dir}/backtest_metrics.csv", index=False)

    # per-period returns per model (rows = dates), so any subset can be replotted later
    if period_dates:
        returns_df = pd.DataFrame(results, index=pd.to_datetime(period_dates))
        returns_df.index.name = "Date"
        returns_df.to_csv(f"{out_dir}/returns.csv")

    avg_turnover = _average_turnover(weights_hist, end_hist)

    if cfg.log_weights:
        _write_weights(weights_hist, f"{out_dir}/weights.csv")

    # per-step QLIKE per covariance estimator (rows = rebalance dates) + the average
    avg_qlike = {}
    if qlike_hist:
        qlike_df = pd.DataFrame([{"Date": fdate, **qinfo} for fdate, qinfo in qlike_hist])
        qlike_df = qlike_df.set_index("Date").sort_index()
        qlike_df.to_csv(f"{out_dir}/qlike.csv")
        avg_qlike = qlike_df.mean().to_dict()

    # per-step Frobenius cov RMSE per covariance estimator + the average
    avg_covrmse = {}
    if covrmse_hist:
        covrmse_df = pd.DataFrame([{"Date": fdate, **crinfo} for fdate, crinfo in covrmse_hist])
        covrmse_df = covrmse_df.set_index("Date").sort_index()
        covrmse_df.to_csv(f"{out_dir}/cov_rmse.csv")
        avg_covrmse = covrmse_df.mean().to_dict()

    summary = _build_summary(results, results_log, metrics, period_dates, rf, avg_turnover,
                             avg_qlike, avg_covrmse)
    summary.to_csv(f"{out_dir}/summary.csv", index=False)
    if verbose:
        print("Annualized performance summary:")
        print(summary.to_string(index=False))

    if any(results.values()):
        _plot_portfolio_value(results, period_dates, f"{out_dir}/Simulation.png")

    return summary


# =============================================================================
# Result assembly helpers
# =============================================================================

def _average_turnover(weights_hist, end_hist):
    """
    Average turnover per strategy: at each rebalance, how much weight is traded to
    go from the previous window's drifted weights to the new target weights (sum of
    absolute weight changes = buys + sells, over the union of both universes).
    """
    avg_turnover = {}
    for name, hist in weights_hist.items():
        ends = end_hist[name]
        per_window = []
        for k in range(1, len(hist)):
            prev_end, target = ends[k - 1], hist[k][1]
            idx = prev_end.index.union(target.index)
            per_window.append(float(np.abs(target.reindex(idx).fillna(0.0)
                                            - prev_end.reindex(idx).fillna(0.0)).sum()))
        avg_turnover[name] = float(np.mean(per_window)) if per_window else np.nan
    return avg_turnover


def _write_weights(weights_hist, path):
    """Per-window target weights in long format (Date, Model, Covariance, Asset, Weight)."""
    weight_rows = []
    for name, hist in weights_hist.items():
        model, cov_type = ("Naive", "N/A") if name == "Naive" else name.rsplit(" ", 1)
        for fdate, target_w in hist:
            for asset, wt in target_w.items():
                weight_rows.append({"Date": fdate, "Model": model,
                                    "Covariance Type": cov_type,
                                    "Asset": asset, "Weight": wt})
    pd.DataFrame(weight_rows).to_csv(path, index=False)


def _build_summary(results, results_log, metrics, period_dates, rf, avg_turnover, avg_qlike, avg_covrmse):
    # average forecast (annualised) std per model/cov type
    avg_fcst = metrics.groupby(["Model", "Covariance Type"])["Forecasted Std"].mean() * np.sqrt(252)
    # average ERC risk-contribution RMSE per model/cov type (NaN for non-ERC)
    avg_rc = metrics.groupby(["Model", "Covariance Type"])["RC RMSE"].mean()
    # daily risk-free return aligned to the evaluated dates (same order as results)
    rf_aligned = rf.reindex(pd.to_datetime(period_dates)).fillna(0.0).values

    summary_rows = []
    for name, rets in results.items():
        rets = np.array(rets)
        if rets.size == 0:
            continue
        model, cov_type = ("Naive", "N/A") if name == "Naive" else name.rsplit(" ", 1)
        row = calculate_summary_metrics(rets, rf_aligned)
        fcst = avg_fcst.get((model, cov_type), np.nan)
        row["Model"] = model
        row["Covariance Type"] = cov_type
        row["Ann. Std (fcst)"] = fcst
        # Calibration ratio: realized vs forecast vol on the SAME (log) scale the
        # model forecasts. The forecast is sqrt(w'Σw) with Σ a LOG-return covariance,
        # so we compare it to the realized vol of the log-return portfolio proxy
        # (not the simple-return Ann. Std, which is the performance vol). For daily
        # data the two realized vols differ negligibly; this just makes it exact.
        log_rets = np.asarray(results_log.get(name, []))
        real_std_log = log_rets.std(ddof=1) * np.sqrt(252) if log_rets.size > 1 else np.nan
        row["Real / Fcst Std"] = real_std_log / fcst if (fcst and fcst > 0) else np.nan
        # QLIKE and cov RMSE are properties of the covariance matrix; Naive uses the historical one
        cov_key = "Historical" if cov_type == "N/A" else cov_type
        row["Avg QLIKE"] = avg_qlike.get(cov_key, np.nan)
        row["Avg Cov RMSE"] = avg_covrmse.get(cov_key, np.nan)
        # ERC risk-contribution RMSE (NaN for non-ERC models)
        row["ERC RC RMSE"] = avg_rc.get((model, cov_type), np.nan)
        row["Avg Turnover"] = avg_turnover.get(name, np.nan)
        summary_rows.append(row)

    col_order = ["Model", "Covariance Type", "Ann. Return", "Ann. Std", "Ann. Std (fcst)",
                 "Real / Fcst Std", "Avg QLIKE", "Avg Cov RMSE", "ERC RC RMSE", "Avg Turnover",
                 "Ann. Sharpe", "Ann. Sharpe (rf=0)", "Ann. Sortino", "Max Drawdown", "Calmar Ratio", "CVaR (95%)",
                 "Skewness", "Excess Kurtosis"]
    return pd.DataFrame(summary_rows).sort_values(["Model", "Covariance Type"])[col_order]


def _plot_portfolio_value(results, period_dates, path):
    import matplotlib
    matplotlib.use("Agg")  # headless-safe; the plot is built in the main process only
    import matplotlib.pyplot as plt

    x_axis = pd.to_datetime(period_dates)
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, rets in results.items():
        ax.plot(x_axis, 100 * np.cumprod(1 + np.array(rets)), label=name)
    ax.set_title("Portfoliowert (Start = 100)")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Portfoliowert")
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# Entry point lives in main.py (it holds the datasets/horizons to run).
# =============================================================================

if __name__ == "__main__":
    print("This is the engine module. Run `python main.py` to start a backtest "
          "(edit the SETTINGS block at the top of main.py to choose datasets/horizons).")
