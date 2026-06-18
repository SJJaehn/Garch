"""
Covariance estimators.

Each takes (log) returns and/or GARCH outputs and returns a covariance DataFrame
(assets in both axes). Three flavours:

  historical_covariance         plain sample covariance
  constant_correlation_covariance   GARCH variances + static historical correlation
  dcc_covariance                GARCH variances + Engle (2002) dynamic correlation
"""
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


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
