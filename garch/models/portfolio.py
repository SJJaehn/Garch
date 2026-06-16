"""
Portfolio weighting schemes.

  mvp_weights   long-only minimum variance      (riskfolio-lib)
  erc_weights   equal risk contribution         (riskfolio-lib risk parity)
  hrp_weights   hierarchical risk parity         (self-implemented)
  naive_weights 1/N

MVP and ERC go through riskfolio-lib with our *externally estimated* covariance
(GARCH / DCC / historical) injected directly — riskfolio normally estimates the
covariance from returns, but the whole point here is to compare estimators, so we
set ``port.cov`` ourselves. HRP stays self-implemented because no library accepts
a user-supplied covariance for it.
"""
import warnings

import numpy as np
import pandas as pd
import riskfolio as rp
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from garch import config


# =============================================================================
# riskfolio helpers
# =============================================================================

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


# =============================================================================
# Weighting schemes
# =============================================================================

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


# =============================================================================
# Dispatch
# =============================================================================

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
