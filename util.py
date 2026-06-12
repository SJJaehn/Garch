"""
Helper functions for the rolling portfolio backtest:
covariance estimators, portfolio weighting schemes and performance metrics.

All covariance estimators expect (log) returns as a DataFrame with assets in
columns and dates in rows, and return a covariance DataFrame.
"""
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


# =============================================================================
# Covariance estimators
# =============================================================================

def estimate_cov_matrix_historical(returns):
    """Plain historical sample covariance."""
    return returns.cov()


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

    The GARCH and DCC estimators share this so we only fit the (expensive)
    univariate models once per window.
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
            var, sr = _ewma_fallback(series)

        variances[col] = var
        resid[col] = sr

    return pd.Series(variances, dtype=float), pd.DataFrame(resid)


def cov_constant_correlation(returns, variances):
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


def cov_dcc(variances, std_resid):
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

def get_mvp_weights(cov_matrix):
    """Long-only Minimum Variance Portfolio (SLSQP with analytic gradient)."""
    n = len(cov_matrix)
    cov = np.array(cov_matrix) + np.eye(n) * 1e-8
    result = minimize(
        lambda w: w @ cov @ w,
        x0=np.ones(n) / n,
        jac=lambda w: 2 * cov @ w,          # analytic gradient -> faster solve
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)},
        options={"ftol": 1e-10},
    )
    return pd.Series(result.x, index=cov_matrix.index)


def get_erc_weights(cov_matrix):
    """Equal Risk Contribution portfolio, solved by fixed-point iteration."""
    n = len(cov_matrix)
    if n == 0:
        return pd.Series(dtype=float)

    cov = np.array(cov_matrix) + np.eye(n) * 1e-8
    weights = np.ones(n) / n
    for _ in range(1000):
        prev = weights.copy()
        risk_contributions = np.maximum(weights * (cov @ weights), 1e-12)
        weights *= (risk_contributions.sum() / n) / risk_contributions
        weights /= weights.sum()
        if np.abs(weights - prev).max() < 1e-8:
            break
    return pd.Series(weights, index=cov_matrix.index)


def get_hrp_weights(cov_matrix):
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


# =============================================================================
# Performance metrics
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
    ann_std = arr.std(ddof=1) * np.sqrt(252)             # sample std

    # arithmetic annualised excess return, consistent with the std used below
    ann_excess = excess.mean() * 252
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
        "Ann. Sharpe":     ann_excess / ann_std  if ann_std > 0 else np.nan,
        "Ann. Sortino":    ann_excess / semi_dev if semi_dev > 0 else np.nan,
        "Max Drawdown":    max_dd,
        "Calmar Ratio":    ann_ret / abs(max_dd) if max_dd < 0 else np.nan,
        "CVaR (95%)":      cvar_95,
        "Skewness":        pd.Series(arr).skew(),
        "Excess Kurtosis": pd.Series(arr).kurt(),
    }
