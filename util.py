import numpy as np
import pandas as pd
from arch import arch_model


def estimate_cov_matrix_garch(returns, prediction_window=21, p=1, q=1, window_label=""):
    n_assets = returns.shape[1]
    variances = np.full(n_assets, np.nan)

    for i in range(n_assets):
        series = returns.iloc[:, i].dropna()
        result = arch_model(series * 100, vol="GARCH", p=p, q=q).fit(disp="off", show_warning=False)
        if result.convergence_flag != 0:
            label = f"  [{window_label}]" if window_label else ""
            print(f"  GARCH no convergence: {returns.columns[i]}{label} — asset excluded from GARCH portfolios")
            continue
        if prediction_window == 0:
            variances[i] = float(result.conditional_volatility.iloc[-1]) ** 2 / (100**2)
        else:
            variances[i] = result.forecast(horizon=prediction_window).variance.iloc[-1].mean() / (100**2)

    valid = ~np.isnan(variances)
    cols = returns.columns[valid]
    variances = variances[valid]
    n_valid = len(cols)

    corr_matrix = returns[cols].corr().fillna(0).values
    np.fill_diagonal(corr_matrix, 1.0)
    cov_matrix = np.zeros((n_valid, n_valid))

    for i in range(n_valid):
        cov_matrix[i, i] = variances[i]
        for j in range(i + 1, n_valid):
            cov_ij = corr_matrix[i, j] * np.sqrt(variances[i] * variances[j])
            cov_matrix[i, j] = cov_ij
            cov_matrix[j, i] = cov_ij

    cov_matrix += np.eye(n_valid) * 1e-8
    return pd.DataFrame(cov_matrix, index=cols, columns=cols)


def estimate_cov_matrix_historical(returns):
    """
    Estimates a covariance matrix using historical returns.
    """
    return returns.cov()


def get_mvp_weights(cov_matrix):
    """
    Computes the weights of the long-only Minimum Variance Portfolio (MVP).
    """
    from scipy.optimize import minimize
    n = len(cov_matrix)
    cov = np.array(cov_matrix) + np.eye(n) * 1e-8
    result = minimize(
        lambda w: w @ cov @ w,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-12},
    )
    return pd.Series(result.x, index=cov_matrix.index)


def get_erc_weights(cov_matrix):
    """
    Computes the weights of the Equal Risk Contribution (ERC) Portfolio.
    """
    n = len(cov_matrix)
    if n == 0:
        return pd.Series(dtype=float)

    cov = np.array(cov_matrix)
    weights = np.ones(n) / n

    for _ in range(1000):
        prev = weights.copy()
        risk_contributions = weights * (cov @ weights)
        risk_contributions = np.maximum(risk_contributions, 1e-12)
        total_risk = risk_contributions.sum()
        weights *= (total_risk / n) / risk_contributions
        weights /= weights.sum()
        if np.abs(weights - prev).max() < 1e-8:
            break

    return pd.Series(weights, index=cov_matrix.index)


def get_hrp_weights(cov_matrix):
    """
    Computes the weights of the Hierarchical Risk Parity (HRP) Portfolio.
    """
    n = len(cov_matrix)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=cov_matrix.index)

    if isinstance(cov_matrix, pd.DataFrame):
        cov = cov_matrix.copy().astype(float)
    else:
        cov = pd.DataFrame(np.asarray(cov_matrix, dtype=float))

    # HRP assumes a symmetric covariance matrix. Small diagonal jitter keeps
    # the clustering step numerically stable when variances are tiny.
    cov = (cov + cov.T) / 2.0
    cov += np.eye(n) * 1e-12

    # Convert covariance into a correlation-distance matrix for clustering.
    std = np.sqrt(np.clip(np.diag(cov.values), 1e-12, None))
    corr = cov.values / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))

    try:
        from scipy.cluster.hierarchy import leaves_list, linkage

        condensed_dist = dist[np.triu_indices(n, k=1)]
        link = linkage(condensed_dist, method="single")
        sort_ix = leaves_list(link)
    except Exception:
        # If clustering is unavailable, fall back to inverse-variance weights.
        inv_var = 1.0 / np.clip(np.diag(cov.values), 1e-12, None)
        w = inv_var / inv_var.sum()
        return pd.Series(w, index=cov.index)

    sorted_assets = list(cov.index[sort_ix])
    weights = pd.Series(1.0, index=sorted_assets)

    def _cluster_variance(cluster_assets):
        # Estimate the variance of a sub-cluster using inverse-variance weights.
        sub_cov = cov.loc[cluster_assets, cluster_assets].values
        ivp = 1.0 / np.clip(np.diag(sub_cov), 1e-12, None)
        ivp = ivp / ivp.sum()
        return float(ivp @ sub_cov @ ivp)

    # Start with the full dendrogram ordering and split it into balanced halves.
    clusters = [sorted_assets]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue

        split = len(cluster) // 2
        c1, c2 = cluster[:split], cluster[split:]

        # Allocate more weight to the side with lower variance.
        v1 = _cluster_variance(c1)
        v2 = _cluster_variance(c2)
        alpha = v2 / (v1 + v2) if (v1 + v2) > 0 else 0.5

        weights[c1] *= alpha
        weights[c2] *= 1.0 - alpha

        clusters.append(c1)
        clusters.append(c2)

    weights = weights / weights.sum()
    return weights.reindex(cov.index)


def calculate_metrics(returns, weights):
    portfolio_returns = returns @ weights
    return {
        "per_period_returns": portfolio_returns,
        "mean_return": portfolio_returns.mean(),
        "std_dev": portfolio_returns.std(),
    }


def calculate_summary_metrics(daily_returns: np.ndarray) -> dict:
    arr = daily_returns
    ann_ret = np.prod(1 + arr) ** (252 / len(arr)) - 1  # CAGR
    ann_std = arr.std() * np.sqrt(252)

    # Sortino: downside semi-deviation as denominator (only penalises negative returns)
    semi_dev = np.sqrt(np.mean(np.minimum(arr, 0) ** 2)) * np.sqrt(252)

    # Max drawdown
    cum = np.cumprod(1 + arr)
    drawdowns = cum / np.maximum.accumulate(cum) - 1
    max_dd = drawdowns.min()

    # CVaR at 95% confidence (expected loss in the worst 5% of days)
    var_95 = np.percentile(arr, 5)
    cvar_95 = arr[arr <= var_95].mean()

    return {
        "Ann. Return":       ann_ret,
        "Ann. Std":          ann_std,
        "Ann. Sharpe":       ann_ret / ann_std          if ann_std  > 0 else np.nan,
        "Ann. Sortino":      ann_ret / semi_dev         if semi_dev > 0 else np.nan,
        "Max Drawdown":      max_dd,
        "Calmar Ratio":      ann_ret / abs(max_dd)      if max_dd   < 0 else np.nan,
        "CVaR (95%)":        cvar_95,
        "Skewness":          pd.Series(arr).skew(),
        "Excess Kurtosis":   pd.Series(arr).kurt(),
    }
