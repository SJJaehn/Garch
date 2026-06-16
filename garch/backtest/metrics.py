"""Annualised performance statistics for a series of simple daily returns."""
import numpy as np
import pandas as pd


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
        "Ann. Sortino":    ann_excess / semi_dev if semi_dev > 0 else np.nan,
        "Max Drawdown":    max_dd,
        "Calmar Ratio":    ann_ret / abs(max_dd) if max_dd < 0 else np.nan,
        "CVaR (95%)":      cvar_95,
        "Skewness":        pd.Series(arr).skew(),
        "Excess Kurtosis": pd.Series(arr).kurt(),
    }
