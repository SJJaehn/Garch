"""
Univariate GARCH fitting.

A single ``fit_garch_univariate`` call fits a GARCH(p, q) to every asset and
returns both the forecast variances and the standardized residuals, so the
GARCH (constant-correlation) and DCC covariance estimators can share one fit.
"""
import numpy as np
import pandas as pd
from arch import arch_model


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
            var, sr = _ewma_fallback(series)

        variances[col] = var
        resid[col] = sr

    return pd.Series(variances, dtype=float), pd.DataFrame(resid)
