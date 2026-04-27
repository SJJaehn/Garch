import pandas as pd
import numpy as np
from arch import arch_model


def estimate_cov_matrix_garch(returns, prediction_window=21, p=1, q=1):
    """
    Estimates a covariance matrix using GARCH variances and return correlations.
    """

    n_assets = returns.shape[1]
    variances = np.zeros(n_assets)

    for i in range(n_assets):
        series = returns.iloc[:, i].dropna()
        scaled_series = series * 100  # Scale returns to allow for better convergence in GARCH estimation and to get rid of warning messages
        model = arch_model(scaled_series, vol='Garch', p=p, q=q)
        result = model.fit(disp='off', show_warning=False)
        if result.convergence_flag == 0:
            forecast = result.forecast(horizon=prediction_window)
            variances[i] = forecast.variance.iloc[-1].mean() / (100 ** 2)  # Mean forecasted variance over the horizon, undo scaling
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
    cov = cov + np.eye(n) * 1e-8  # Regularize to avoid numerical issues with near-singular matrices

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
        risk_contributions = np.maximum(risk_contributions, 1e-12)  # Avoid division by zero
        total_risk = risk_contributions.sum()
        target_contributions = total_risk / n
        weights *= target_contributions / risk_contributions
        weights /= weights.sum()

    return pd.Series(weights, index=cov_matrix.index)


def get_naive_weights(cov_matrix):
    """
    Computes the weights of the Naive Portfolio (equal weights).
    """
    n = len(cov_matrix)
    weights = np.ones(n) / n
    return pd.Series(weights, index=cov_matrix.index)


def calculate_metrics(returns, weights):
    """
    Calculates performance metrics for a given portfolio.
    """
    portfolio_returns = returns @ weights
    mean_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()
    sharpe_ratio = mean_return / volatility if volatility > 0 else 0  # Not annualized, assumes returns are at the same frequency
    return {
        "per_period_returns": portfolio_returns,
        "mean_return": mean_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
    }