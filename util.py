import numpy as np
import pandas as pd
from arch import arch_model


def estimate_cov_matrix_garch(returns, prediction_window=21, p=1, q=1):
    """
    Estimates a covariance matrix using GARCH variances and return correlations.
    """

    n_assets = returns.shape[1]
    variances = np.zeros(n_assets)

    for i in range(n_assets):
        series = returns.iloc[:, i].dropna()
        scaled_series = (
            series * 100
        )  # Scale returns to allow for better convergence in GARCH estimation and to get rid of warning messages
        model = arch_model(scaled_series, vol="GARCH", p=p, q=q)
        result = model.fit(disp="off", show_warning=False)
        if result.convergence_flag == 0:
            forecast = result.forecast(horizon=prediction_window)
            variances[i] = forecast.variance.iloc[-1].mean() / (
                100**2
            )  # Mean forecasted variance over the horizon, undo scaling
        else:
            variances[i] = series.var()

    corr_matrix = returns.corr().values
    cov_matrix = np.zeros((n_assets, n_assets))

    for i in range(n_assets):
        cov_matrix[i, i] = variances[i]
        for j in range(i + 1, n_assets):
            cov_ij = corr_matrix[i, j] * np.sqrt(variances[i] * variances[j])
            cov_matrix[i, j] = cov_ij
            cov_matrix[j, i] = cov_ij

    # Small regularization to avoid numerical issues (e.g. when passing to an optimizer)
    cov_matrix += np.eye(n_assets) * 1e-8

    return pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)


def estimate_cov_matrix_historical(returns):
    """
    Estimates a covariance matrix using historical returns.
    """
    return returns.cov()


def get_mvp_weights(cov_matrix):
    """
    Computes the weights of the Minimum Variance Portfolio (MVP).
    """
    n = len(cov_matrix)
    ones = np.ones(n)
    cov = np.array(cov_matrix)
    cov = (
        cov + np.eye(n) * 1e-8
    )  # Regularize to avoid numerical issues with near-singular matrices

    weights = np.linalg.solve(cov, ones)
    weights = weights / np.sum(weights)
    return pd.Series(weights, index=cov_matrix.index)


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
        risk_contributions = weights * (cov @ weights)
        risk_contributions = np.maximum(
            risk_contributions, 1e-12
        )  # Avoid division by zero
        total_risk = risk_contributions.sum()
        target_contributions = total_risk / n
        weights *= target_contributions / risk_contributions
        weights /= weights.sum()

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
    """
    Calculates performance metrics for a given portfolio.
    """
    portfolio_returns = returns @ weights
    mean_return = portfolio_returns.mean()
    std_dev = portfolio_returns.std()
    sharpe_ratio = (
        mean_return / std_dev if std_dev > 0 else 0
    )  # Not annualized, assumes returns are at the same frequency
    return {
        "per_period_returns": portfolio_returns,
        "mean_return": mean_return,
        "std_dev": std_dev,
        "sharpe_ratio": sharpe_ratio,
    }
