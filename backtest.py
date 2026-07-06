"""
Rolling-window portfolio backtest.

Compares the portfolio construction methods (MVP, HRP, ERC and a naive 1/N
benchmark) under different covariance estimators (historical sample covariance,
GARCH with constant correlation, DCC).

The modelling part works on LOG returns (GARCH/DCC assume additive,
well-behaved returns). The performance evaluation works on SIMPLE returns,
because a portfolio return is the weighted sum of simple asset returns, not of
log returns.

The datasets and window sizes to run are chosen in main.py:

    python main.py
"""
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
# Data loading
# =============================================================================

"""
Loads the price data of a dataset (assets in the columns, dates in the index).
"""
def load_prices(dataset):
    filepath, date_format = config.DATASETS[dataset]
    prices = pd.read_csv(filepath, index_col=0)
    prices.index = pd.to_datetime(prices.index, format=date_format, errors="coerce")
    prices = prices[prices.index.notna()]  # dropping rows where the date could not be parsed
    # keeping only the dates where at least half of the assets have a price
    prices = prices.loc[prices.notna().sum(axis=1) >= int(0.5 * prices.shape[1])]
    return prices


"""
Calculates the log returns r_t = ln(P_t / P_{t-1}). Assets that show no price
movement on more than 50% of the days (stale series) are dropped.
"""
def to_log_returns(prices):
    log_returns = np.log(prices / prices.shift(1)).iloc[1:]
    zero_frac = (log_returns == 0).sum() / log_returns.notna().sum()
    return log_returns.loc[:, zero_frac < 0.5]


"""
Reads the Fed Funds total-return index (a daily index level, sorted by date).
The csv export uses German month names, so those are translated first.
"""
def read_risk_free_level():
    if not os.path.exists(config.RISK_FREE_FILE):
        raise FileNotFoundError(f"Risk-free file not found: {config.RISK_FREE_FILE}")
    rf = pd.read_csv(config.RISK_FREE_FILE, sep=";", decimal=",", encoding="utf-8-sig")
    dates = rf["Exchange Date"]
    for de, en in {"Mär": "Mar", "Mai": "May", "Okt": "Oct", "Dez": "Dec"}.items():
        dates = dates.str.replace(de, en, regex=False)  # German -> English month names
    level = pd.Series(rf["Close"].values, index=pd.to_datetime(dates, format="%d-%b-%Y"))
    return level.sort_index()


"""
Daily simple risk-free return aligned to the given dates. The index level is
forward-filled onto the target dates and then the percentage change is taken,
so the risk-free accrual spans exactly the same day spacing as the portfolio
returns.
"""
def load_risk_free(dates):
    level = read_risk_free_level()
    return level.reindex(dates).ffill().pct_change()


"""
Loads everything needed for one dataset: the prices, the log returns and the
risk-free return series (aligned to the price dates).
"""
def load_dataset(dataset):
    prices = load_prices(dataset)
    log_returns = to_log_returns(prices)
    rf = load_risk_free(prices.index)
    return prices, log_returns, rf


# =============================================================================
# Univariate GARCH fitting
# =============================================================================

"""
RiskMetrics EWMA variance, used as a fallback when the GARCH fit does not
converge. Returns the forecast variance and the standardized residuals. EWMA
assumes a flat variance going forward, so the forecast is just the last value.
"""
def ewma_fallback(series, lam=0.94):
    ewma_var = series.pow(2).ewm(alpha=1 - lam).mean()
    cond_vol = np.sqrt(ewma_var).replace(0.0, np.nan)
    return float(ewma_var.iloc[-1]), series / cond_vol


"""
Fits a univariate GARCH(p, q) to every asset and returns
  - variances : Series with the forecast variance per asset
  - std_resid : DataFrame with the standardized residuals per asset

Both the GARCH (constant correlation) and the DCC covariance need these
outputs, so the fit is done once and shared. When a fit does not converge it is
retried with more iterations, and if it still fails the EWMA fallback is used
so the asset is kept in the universe instead of being silently dropped.
"""
def fit_garch(returns, prediction_window=1, p=1, q=1):
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
                # arch works better with percent returns, the /100**2 undoes the scaling
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

        if var is None:  # GARCH failed -> EWMA fallback
            reason = "insufficient data" if len(series) < 50 else "GARCH did not converge"
            print(f"[fallback] GARCH->EWMA for {col} ({reason})", flush=True)
            var, sr = ewma_fallback(series)

        variances[col] = var
        resid[col] = sr

    return pd.Series(variances, dtype=float), pd.DataFrame(resid)


# =============================================================================
# Covariance estimators
# =============================================================================
# Each estimator returns a covariance DataFrame (assets on both axes).

"""
Plain historical sample covariance.
"""
def historical_covariance(returns):
    return returns.cov()


"""
CCC-GARCH covariance (Bollerslev 1990): the GARCH forecast variances are
combined with the constant sample correlation of the standardized residuals.
The standardized residuals are used instead of the raw returns because the
volatility clustering is already filtered out of them (high-vol days would
otherwise inflate the correlation); it also means that CCC and DCC work with
the same correlation input.
"""
def garch_covariance(std_resid, variances):
    cols = list(variances.index)
    if len(cols) == 0:
        return pd.DataFrame()

    std = np.sqrt(variances.values)
    corr = std_resid[cols].corr().fillna(0.0).values
    np.fill_diagonal(corr, 1.0)

    cov = corr * np.outer(std, std)
    cov += np.eye(len(cols)) * 1e-8  # tiny jitter for numerical stability
    return pd.DataFrame(cov, index=cols, columns=cols)


"""
DCC-GARCH covariance (Engle 2002): the GARCH forecast variances are combined
with a Dynamic Conditional Correlation model estimated on the standardized
residuals z_t:

    Q_t = (1 - a - b) * Q_bar + a * z_{t-1} z_{t-1}' + b * Q_{t-1}
    R_t = normalize(Q_t),   Sigma = D * R * D   (D = diag of the forecast std)

The two parameters (a, b) are estimated by maximum likelihood (the likelihood
uses a Cholesky factorization for speed and is optimized with L-BFGS-B).

For horizon > 1 the correlation forecast mean-reverts toward Q_bar
(Engle & Sheppard 2001),
    E[Q_{T+h}] = Q_bar + (a + b)^(h-1) * (Q_{T+1} - Q_bar),
and the normalized correlations are averaged over h = 1..horizon to match the
horizon-averaged GARCH variances.
"""
def dcc_covariance(variances, std_resid, horizon=1):
    cols = list(std_resid.columns)
    if len(cols) == 0:
        return pd.DataFrame()

    z = std_resid.dropna().values            # keep only the common dates
    n_obs, n = z.shape
    std = np.sqrt(variances[cols].values)

    # too few assets / observations for a correlation model -> diagonal covariance
    if n < 2 or n_obs < n + 2:
        print(f"[fallback] DCC->diagonal covariance (n={n}, n_obs={n_obs})", flush=True)
        cov = np.diag(std ** 2) + np.eye(n) * 1e-8
        return pd.DataFrame(cov, index=cols, columns=cols)

    q_bar = np.corrcoef(z, rowvar=False)     # unconditional correlation of the z's

    # negative log-likelihood of the DCC recursion for the parameters (a, b)
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
    if not (a > 0 and b > 0 and a + b < 1):  # typical values if the optimizer misbehaves
        print(f"[fallback] DCC optimizer misbehaved (a={a:.3g}, b={b:.3g}); using a=0.02, b=0.95", flush=True)
        a, b = 0.02, 0.95

    # roll the recursion through the sample to get Q_{T+1} ...
    omega = (1 - a - b) * q_bar
    Q = q_bar.copy()
    for t in range(n_obs):
        Q = omega + a * np.outer(z[t], z[t]) + b * Q

    # ... then mean-revert toward q_bar over the horizon and average the
    # normalized correlations (for horizon == 1 this is just Q_{T+1})
    horizon = max(horizon, 1)
    R = np.zeros_like(Q)
    for h in range(1, horizon + 1):
        Q_h = q_bar + (a + b) ** (h - 1) * (Q - q_bar)
        d = np.sqrt(np.diag(Q_h))
        R += Q_h / np.outer(d, d)
    R /= horizon

    cov = R * np.outer(std, std)
    cov += np.eye(n) * 1e-8
    return pd.DataFrame(cov, index=cols, columns=cols)


# =============================================================================
# Portfolio weighting schemes
# =============================================================================
# MVP and ERC are solved with riskfolio-lib, with our own covariance matrix
# (historical / GARCH / DCC) plugged in directly. HRP is implemented by hand
# because riskfolio does not accept a user-supplied covariance matrix for it.

"""
Builds a riskfolio Portfolio object with our covariance matrix plugged in.
riskfolio needs a returns DataFrame for the asset names/shape, but with
model='Classic' and hist=True the optimization only reads port.cov and port.mu,
so the actual return values are never used (they are just zeros).
"""
def make_portfolio(cov_matrix):
    cols = list(cov_matrix.index)
    n = len(cols)

    cov = np.asarray(cov_matrix, dtype=float)
    cov = (cov + cov.T) / 2.0 + np.eye(n) * 1e-8  # force symmetry + jitter for stability
    cov = pd.DataFrame(cov, index=cols, columns=cols)

    dummy = pd.DataFrame(np.zeros((2, n)), columns=cols)
    port = rp.Portfolio(returns=dummy)
    port.cov = cov
    port.mu = pd.DataFrame(np.zeros((1, n)), columns=cols)
    # stable solver configuration, see the note in main.py
    port.solvers = list(config.SOLVERS)
    port.sol_params = {k: dict(v) for k, v in config.SOL_PARAMS.items()}
    return port, cols


"""
Turns a riskfolio weights frame into a normalized, long-only weights Series.
If the optimization returned no solution, equal weights are used instead.
"""
def clean_weights(w, cols, index):
    if w is None or getattr(w, "empty", True):
        warnings.warn("riskfolio optimisation returned no solution; using equal weights")
        return pd.Series(1.0 / len(cols), index=index)
    s = w["weights"].reindex(cols).fillna(0.0).clip(lower=0.0)
    total = s.sum()
    if total <= 0:
        return pd.Series(1.0 / len(cols), index=index)
    return (s / total).reindex(index)


"""
Equal-weight (1/N) portfolio.
"""
def naive_weights(assets):
    assets = list(assets)
    n = len(assets)
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / n, index=assets)


"""
Long-only Minimum Variance Portfolio (riskfolio MinRisk with rm='MV').
"""
def mvp_weights(cov_matrix):
    if len(cov_matrix) == 0:
        return pd.Series(dtype=float)
    port, cols = make_portfolio(cov_matrix)
    w = port.optimization(model="Classic", rm="MV", obj="MinRisk", hist=True)
    return clean_weights(w, cols, cov_matrix.index)


"""
Equal Risk Contribution portfolio (riskfolio risk parity with rm='MV').
"""
def erc_weights(cov_matrix):
    if len(cov_matrix) == 0:
        return pd.Series(dtype=float)
    port, cols = make_portfolio(cov_matrix)
    w = port.rp_optimization(model="Classic", rm="MV", b=None, hist=True)
    return clean_weights(w, cols, cov_matrix.index)


"""
Hierarchical Risk Parity (Lopez de Prado 2016).
"""
def hrp_weights(cov_matrix):
    n = len(cov_matrix)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=cov_matrix.index)

    cov = cov_matrix.copy()
    cov = (cov + cov.T) / 2.0  # make sure the matrix is symmetric

    # 1) correlation distance matrix
    std = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))

    # 2) hierarchical clustering -> quasi-diagonal ordering of the assets
    link = linkage(squareform(dist, checks=False), method="single")
    order = leaves_list(link)
    assets = list(cov.index[order])

    # variance of a sub-cluster using inverse-variance weights
    def cluster_variance(items):
        sub = cov.loc[items, items].values
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    # 3) recursive bisection: split the ordering in half and give more weight
    #    to the half with the lower variance, repeat within each half
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


"""
Returns the portfolio weights for the given model name.
"""
def get_weights(model, cov_matrix):
    if model == "MVP":
        return mvp_weights(cov_matrix)
    elif model == "HRP":
        return hrp_weights(cov_matrix)
    elif model == "ERC":
        return erc_weights(cov_matrix)
    raise ValueError(f"unknown model: {model}")


# =============================================================================
# Performance metrics and covariance-forecast losses
# =============================================================================

"""
Annualized performance statistics for a series of simple daily returns.
rf_daily is the simple daily risk-free return (a scalar or an array aligned to
daily_returns) and is subtracted to get the excess returns for Sharpe/Sortino.
"""
def calculate_summary_metrics(daily_returns, rf_daily=0.0):
    arr = np.asarray(daily_returns)
    rf = np.asarray(rf_daily)
    excess = arr - rf

    ann_ret = np.prod(1 + arr) ** (252 / len(arr)) - 1   # CAGR (geometric)
    ann_std = arr.std(ddof=1) * np.sqrt(252)             # realized portfolio vol

    # arithmetic annualized excess return, consistent with the std used below
    ann_excess = excess.mean() * 252
    # std of the EXCESS returns for the Sharpe denominator (textbook definition)
    ann_excess_std = excess.std(ddof=1) * np.sqrt(252)
    # downside deviation of the excess returns (for Sortino)
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


"""
Multivariate QLIKE loss of a forecast covariance H against the realized returns
of the test window (Patton 2011; Laurent et al. 2012):

    QLIKE = log|H| + tr(H^{-1} S),   S = (1/pw) * sum_t r_t r_t'

S is the realized second-moment proxy of the test-window log returns. In
expectation the loss is minimized when H equals the true covariance, so a lower
QLIKE means a better covariance forecast. Returns NaN if H is not positive
definite.
"""
def qlike_loss(cov, realized):
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


"""
Frobenius RMSE between a forecast covariance H and the realized second-moment
proxy S of the test window (Patton & Sheppard 2009):

    Cov RMSE = sqrt( mean_ij (H_ij - S_ij)^2 )

This is the typical per-element error of the forecast, lower is better. The
proxy S is noisy for short test windows, but its expected value is the true
covariance, so the noise averages out across the windows and the level stays
comparable between the estimators.
"""
def cov_rmse(cov, realized):
    H = np.asarray(cov, dtype=float)
    R = np.asarray(realized, dtype=float)
    if H.shape[0] == 0 or R.shape[0] == 0:
        return np.nan
    S = R.T @ R / R.shape[0]
    return float(np.sqrt(np.mean((H - S) ** 2)))


"""
RMSE of the realized risk contributions from the equal-risk target (1/N).
With the realized second moment S as covariance proxy, asset i contributes

    RC_i = w_i (S w)_i / (w' S w),   sum_i RC_i = 1.

An Equal-Risk-Contribution portfolio targets RC_i = 1/N, so a lower RMSE means
the weights kept the risk balanced out-of-sample -> a better covariance
forecast. Returns NaN if the realized portfolio variance is ~0.
"""
def risk_contribution_rmse(weights, realized):
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

# Shared read-only data inside each worker process, filled once by init_worker
# (this avoids sending the full returns DataFrame with every single window)
worker_data = {}


def init_worker(log_returns, train_window, prediction_window):
    worker_data["log_returns"] = log_returns
    worker_data["train_window"] = train_window
    worker_data["prediction_window"] = prediction_window


# Entry point for the process pool: runs one window using the worker data
def process_window(start):
    return run_window(start, worker_data["log_returns"],
                      worker_data["train_window"], worker_data["prediction_window"])


"""
Runs one rolling window: fits the covariance estimators on the training window,
builds the portfolios and evaluates them on the test window.

Returns (per_period, per_period_log, records, dates, weights_info,
formation_date, qlike_info, covrmse_info).
"""
def run_window(start, log_returns, train_window, prediction_window):
    train = log_returns.iloc[start : start + train_window]
    test  = log_returns.iloc[start + train_window : start + train_window + prediction_window]

    # The investable universe is decided from TRAINING information only (no
    # look-ahead into the test window): an asset needs at least 90% observations
    # in the window and a valid observation on the last training day (i.e. it is
    # tradeable when the portfolio is formed).
    train = train.loc[:, train.notna().mean() >= 0.9]
    train = train.loc[:, train.iloc[-1].notna()]
    if train.shape[1] == 0 or test.empty:
        return {}, {}, [], [], {}, None, {}, {}

    # Same universe in the test window; a missing test return means the asset
    # did not trade that day, so its return is 0 (it is not dropped afterwards).
    test = test[train.columns]
    test_simple = (np.exp(test) - 1).fillna(0.0)  # simple returns for the performance
    # Log returns for the vol-calibration check: the forecast std sqrt(w'Σw)
    # uses a covariance of LOG returns, so its realized counterpart must be
    # measured on the same scale.
    test_log = test.fillna(0.0)

    # Building the requested covariance matrices (the GARCH fit is done only
    # once and shared between the GARCH and DCC estimators)
    cov_by_method = {}
    if "Historical" in config.COV_METHODS:
        cov_by_method["Historical"] = historical_covariance(train)
    if "GARCH" in config.COV_METHODS or "DCC" in config.COV_METHODS:
        variances, std_resid = fit_garch(train, prediction_window,
                                         config.GARCH_P, config.GARCH_Q)
        if "GARCH" in config.COV_METHODS:
            cov_by_method["GARCH"] = garch_covariance(std_resid, variances)
        if "DCC" in config.COV_METHODS:
            cov_by_method["DCC"] = dcc_covariance(variances, std_resid, prediction_window)

    # Forecast quality of every covariance matrix against the realized test
    # returns: QLIKE and Frobenius RMSE (lower is better for both)
    qlike_info = {}
    covrmse_info = {}
    for method, cov in cov_by_method.items():
        if cov is None or cov.shape[1] == 0:
            continue
        realized = test[list(cov.columns)].fillna(0.0).values
        qlike_info[method] = qlike_loss(cov.values, realized)
        covrmse_info[method] = cov_rmse(cov.values, realized)

    # the historical covariance is also needed for the naive portfolio's forecast std
    cov_hist_full = cov_by_method.get("Historical")
    if cov_hist_full is None:
        cov_hist_full = historical_covariance(train)

    # List of all (model, cov_type, weights, cov) combinations to evaluate
    naive_w = naive_weights(train.columns)
    jobs = [("Naive", "N/A", naive_w, cov_hist_full)]
    for method, cov in cov_by_method.items():
        if cov is None or cov.shape[1] == 0:
            continue
        for model in config.MODELS:
            jobs.append((model, method, get_weights(model, cov), cov))

    # Weights the models would have chosen with hindsight: the same weighting
    # scheme run on the realized covariance of the prediction window (the same
    # uncentered proxy S = R'R/pw that QLIKE uses). Cached per (model, universe)
    # because the result is identical for every forecast covariance. Note that
    # for prediction_window=1 S has rank 1, so these weights are very noisy and
    # the L1 distance is only meaningful relative to the other estimators.
    realized_weights_cache = {}
    def realized_weights(model, cols):
        key = (model, tuple(cols))
        if key not in realized_weights_cache:
            R = test[cols].fillna(0.0).values
            S = R.T @ R / R.shape[0] + np.eye(len(cols)) * 1e-8
            S = pd.DataFrame(S, index=cols, columns=cols)
            try:
                realized_weights_cache[key] = get_weights(model, S)
            except Exception:
                realized_weights_cache[key] = None
        return realized_weights_cache[key]

    per_period = {}       # name -> list of simple returns (performance)
    per_period_log = {}   # name -> list of log returns (vol calibration)
    records = []
    weights_info = {}     # name -> (target weights, drifted end-of-window weights)
    formation_date = test.index[0]  # the rebalancing date of this window
    for model, cov_type, weights, cov in jobs:
        name = "Naive" if model == "Naive" else f"{model} {cov_type}"
        cols = list(weights.index)

        # Buy-and-hold within the window: the weights are set once at formation
        # and then drift with the returns, so day t's portfolio return is
        # V_t / V_{t-1} - 1 along the portfolio value path (for
        # prediction_window = 1 this is exactly the weighted sum w'r).
        value = (1.0 + test_simple[cols]).cumprod(axis=0) @ weights
        prev_value = value.shift(1)
        prev_value.iloc[0] = float(weights.sum())  # = 1, the value at formation
        port_returns = value / prev_value - 1.0
        per_period[name] = port_returns.values.tolist()
        # the calibration series keeps FIXED weights on purpose, because the
        # forecast std is calculated with the formation weights
        per_period_log[name] = (test_log[cols] @ weights).values.tolist()

        # Drifted end-of-window weights (needed for the turnover): every start
        # weight grows with its asset's gross return, then renormalize
        gross = (1.0 + test_simple[cols]).prod(axis=0)
        end_val = weights * gross
        end_weights = end_val / end_val.sum()
        weights_info[name] = (weights, end_weights)

        w = weights.values
        forecasted_std = float(np.sqrt(w @ cov.loc[cols, cols].values @ w))
        # ERC quality: how far the realized risk contributions are from the
        # equal (1/N) target (only the ERC objective targets equal contributions)
        rc_rmse = np.nan
        if model == "ERC":
            rc_rmse = risk_contribution_rmse(w, test[cols].fillna(0.0).values)
        # L1 distance ||w_real - w_forecast||_1 between the traded weights and
        # the weights the same model would have chosen with the realized
        # covariance of the prediction window (NaN for Naive, which does not
        # estimate its weights from a covariance)
        l1_dist = np.nan
        if model != "Naive":
            real_w = realized_weights(model, cols)
            if real_w is not None and not real_w.isna().any():
                l1_dist = float(np.abs(weights - real_w).sum())
        records.append({
            "Model": model,
            "Covariance Type": cov_type,
            "Window Index": start // prediction_window,
            "Mean Return": port_returns.mean(),
            "Forecasted Std": forecasted_std,
            "RC RMSE": rc_rmse,
            "L1 Weight Dist": l1_dist,
        })
    return (per_period, per_period_log, records, list(test.index), weights_info,
            formation_date, qlike_info, covrmse_info)


"""
Runs the full rolling-window backtest for one dataset / window combination and
writes all result files into the output folder. rf is the daily simple
risk-free return aligned to the price dates. Returns the summary DataFrame.
"""
def run_backtest(dataset, log_returns, rf, train_window, prediction_window, verbose=True):
    starts = list(range(0, len(log_returns) - train_window - prediction_window + 1,
                        prediction_window))
    total = len(starts)

    results = {}       # name -> list of simple period returns (performance)
    results_log = {}   # name -> list of log period returns (vol calibration)
    records = []
    period_dates = []
    weights_hist = {}  # name -> list of (formation date, target weights) in window order
    end_hist = {}      # name -> list of drifted end-of-window weights (same order)
    qlike_hist = []    # list of (formation date, {cov_type: qlike}) in window order
    covrmse_hist = []  # list of (formation date, {cov_type: cov rmse}) in window order

    # every window is independent, so they are spread over worker processes
    with ProcessPoolExecutor(max_workers=config.MAX_WORKERS,
                             initializer=init_worker,
                             initargs=(log_returns, train_window, prediction_window)) as executor:
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

    out_dir = config.output_dir(dataset, train_window, prediction_window)
    os.makedirs(out_dir, exist_ok=True)
    metrics = pd.DataFrame(records)
    metrics.to_csv(f"{out_dir}/backtest_metrics.csv", index=False)

    # per-period returns per model (rows = dates), so any subset can be replotted later
    if period_dates:
        returns_df = pd.DataFrame(results, index=pd.to_datetime(period_dates))
        returns_df.index.name = "Date"
        returns_df.to_csv(f"{out_dir}/returns.csv")

    avg_turnover = average_turnover(weights_hist, end_hist)

    if config.LOG_WEIGHTS:
        write_weights(weights_hist, f"{out_dir}/weights.csv")

    # per-window QLIKE per covariance estimator (rows = rebalance dates) + the average
    avg_qlike = {}
    if qlike_hist:
        qlike_df = pd.DataFrame([{"Date": fdate, **qinfo} for fdate, qinfo in qlike_hist])
        qlike_df = qlike_df.set_index("Date").sort_index()
        qlike_df.to_csv(f"{out_dir}/qlike.csv")
        avg_qlike = qlike_df.mean().to_dict()

    # per-window Frobenius cov RMSE per covariance estimator + the average
    avg_covrmse = {}
    if covrmse_hist:
        covrmse_df = pd.DataFrame([{"Date": fdate, **crinfo} for fdate, crinfo in covrmse_hist])
        covrmse_df = covrmse_df.set_index("Date").sort_index()
        covrmse_df.to_csv(f"{out_dir}/cov_rmse.csv")
        avg_covrmse = covrmse_df.mean().to_dict()

    summary = build_summary(results, results_log, metrics, period_dates, rf, avg_turnover,
                            avg_qlike, avg_covrmse)
    summary.to_csv(f"{out_dir}/summary.csv", index=False)
    if verbose:
        print("Annualized performance summary:")
        print(summary.to_string(index=False))

    if any(results.values()):
        plot_portfolio_value(results, period_dates, f"{out_dir}/Simulation.png")

    return summary


# =============================================================================
# Result assembly helpers
# =============================================================================

"""
Average turnover per strategy: at every rebalance, how much weight is traded to
get from the previous window's drifted weights to the new target weights (sum
of the absolute weight changes = buys + sells, over the union of both universes).
"""
def average_turnover(weights_hist, end_hist):
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


"""
Writes the per-window target weights in long format
(Date, Model, Covariance Type, Asset, Weight).
"""
def write_weights(weights_hist, path):
    weight_rows = []
    for name, hist in weights_hist.items():
        model, cov_type = ("Naive", "N/A") if name == "Naive" else name.rsplit(" ", 1)
        for fdate, target_w in hist:
            for asset, wt in target_w.items():
                weight_rows.append({"Date": fdate, "Model": model,
                                    "Covariance Type": cov_type,
                                    "Asset": asset, "Weight": wt})
    pd.DataFrame(weight_rows).to_csv(path, index=False)


"""
Builds the summary table: one row per strategy with the annualized performance
metrics, the forecast-vs-realized vol calibration, the covariance losses and
the average turnover.
"""
def build_summary(results, results_log, metrics, period_dates, rf, avg_turnover,
                  avg_qlike, avg_covrmse):
    # average forecast (annualized) std per model / covariance type
    avg_fcst = metrics.groupby(["Model", "Covariance Type"])["Forecasted Std"].mean() * np.sqrt(252)
    # average ERC risk-contribution RMSE per model / covariance type (NaN for non-ERC)
    avg_rc = metrics.groupby(["Model", "Covariance Type"])["RC RMSE"].mean()
    # average L1 distance to the realized-covariance weights (NaN for Naive)
    avg_l1 = metrics.groupby(["Model", "Covariance Type"])["L1 Weight Dist"].mean()
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
        # Calibration ratio: realized vs forecast vol on the SAME (log) scale
        # that the model forecasts on. For daily data the log and simple vols
        # differ only negligibly, this just makes the comparison exact.
        log_rets = np.asarray(results_log.get(name, []))
        real_std_log = log_rets.std(ddof=1) * np.sqrt(252) if log_rets.size > 1 else np.nan
        row["Real / Fcst Std"] = real_std_log / fcst if (fcst and fcst > 0) else np.nan
        # QLIKE and cov RMSE belong to the covariance matrix; Naive uses the historical one
        cov_key = "Historical" if cov_type == "N/A" else cov_type
        row["Avg QLIKE"] = avg_qlike.get(cov_key, np.nan)
        row["Avg Cov RMSE"] = avg_covrmse.get(cov_key, np.nan)
        # ERC risk-contribution RMSE (NaN for the other models)
        row["ERC RC RMSE"] = avg_rc.get((model, cov_type), np.nan)
        # L1 distance to the hindsight weights of the same model (NaN for Naive)
        row["Avg L1 Dist"] = avg_l1.get((model, cov_type), np.nan)
        row["Avg Turnover"] = avg_turnover.get(name, np.nan)
        summary_rows.append(row)

    col_order = ["Model", "Covariance Type", "Ann. Return", "Ann. Std", "Ann. Std (fcst)",
                 "Real / Fcst Std", "Avg QLIKE", "Avg Cov RMSE", "ERC RC RMSE", "Avg L1 Dist",
                 "Avg Turnover",
                 "Ann. Sharpe", "Ann. Sharpe (rf=0)", "Ann. Sortino", "Max Drawdown", "Calmar Ratio", "CVaR (95%)",
                 "Skewness", "Excess Kurtosis"]
    return pd.DataFrame(summary_rows).sort_values(["Model", "Covariance Type"])[col_order]


"""
Plots the portfolio value (start = 100) of every strategy over time.
"""
def plot_portfolio_value(results, period_dates, path):
    import matplotlib
    matplotlib.use("Agg")  # rendering without a display
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


if __name__ == "__main__":
    print("This file only contains the backtest engine. Run `python main.py` to "
          "start a backtest (the settings are at the top of main.py).")
