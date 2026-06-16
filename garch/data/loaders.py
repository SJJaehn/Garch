"""
Data loading: prices, log returns and the risk-free series.

All functions are side-effect free (nothing runs at import) and take their inputs
explicitly, so the same loaders are reused by the backtest and the analysis.
"""
import numpy as np
import pandas as pd

from garch import config


# =============================================================================
# Prices and returns
# =============================================================================

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


# =============================================================================
# Risk-free series (Fed Funds total-return index)
# =============================================================================

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


# =============================================================================
# Convenience bundle
# =============================================================================

def load_dataset(dataset: str, risk_free_file: str | None = None):
    """Return (prices, log_returns, rf_daily) for a dataset, rf aligned to prices."""
    prices = load_prices(dataset)
    log_returns = to_log_returns(prices)
    rf = load_risk_free(prices.index, risk_free_file)
    return prices, log_returns, rf
