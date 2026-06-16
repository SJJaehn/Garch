"""
Per-combo time-series charts saved into each backtest folder:
  rolling_sharpe.png, cumulative_sharpe.png, portfolio_value.png

Rolling Sharpe_t    = mean(excess over last WINDOW days) / std(...) * sqrt(252)
Cumulative Sharpe_t = mean(excess since inception)        / std(...) * sqrt(252)
excess = portfolio return - risk-free return (Fed Funds, same as the backtest).
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from garch import config
from garch.analysis import (COLUMNS, COV_FILTER, DATE_END, DATE_START, MIN_PERIODS,
                            MODEL_FILTER, SMOOTH, WINDOW)
from garch.data import loaders


def select_columns(all_cols):
    if COLUMNS:
        return [c for c in COLUMNS if c in all_cols]
    sel = []
    for c in all_cols:
        if c == "Naive":
            if "Naive" in MODEL_FILTER:
                sel.append(c)
            continue
        model, cov = c.rsplit(" ", 1)
        if model in MODEL_FILTER and cov in COV_FILTER:
            sel.append(c)
    return sel


def plot_lines(df, title, ylabel, out, logy=False, zero_line=True):
    fig, ax = plt.subplots(figsize=(13, 6))
    for col in df.columns:
        ax.plot(df.index, df[col], label=col, linewidth=1.1)
    if logy:
        ax.set_yscale("log")
    elif zero_line:
        ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def per_combo_charts():
    root = config.RESULTS_DIR
    rf_level = loaders.read_risk_free_level()
    for path in sorted(glob.glob(f"{root}/*/*/returns.csv")):
        folder = os.path.dirname(path)
        combo = os.path.relpath(folder, root)

        ret = pd.read_csv(path, index_col=0, parse_dates=True)
        cols = select_columns(list(ret.columns))
        if not cols:
            continue
        ret = ret[cols].loc[DATE_START:DATE_END]
        if ret.empty:
            continue

        rf = loaders.align_risk_free(rf_level, ret.index).fillna(0.0)
        excess = ret.sub(rf, axis=0)

        roll = (excess.rolling(WINDOW).mean() / excess.rolling(WINDOW).std()) * np.sqrt(252)
        if SMOOTH > 1:
            roll = roll.rolling(SMOOTH).mean()
        roll = roll.dropna(how="all")

        cum = (excess.expanding(MIN_PERIODS).mean() / excess.expanding(MIN_PERIODS).std()) * np.sqrt(252)
        cum = cum.dropna(how="all")

        if not roll.empty:
            tag = f", {SMOOTH}d-smoothed" if SMOOTH > 1 else ""
            plot_lines(roll, f"Rolling {WINDOW}-day annualised Sharpe{tag} - {combo}",
                       "Sharpe ratio", f"{folder}/rolling_sharpe.png")
        if not cum.empty:
            plot_lines(cum, f"Cumulative annualised Sharpe - {combo}",
                       "Sharpe ratio", f"{folder}/cumulative_sharpe.png")
        plot_lines(100 * (1 + ret).cumprod(), f"Portfolio value (log scale) - {combo}",
                   "Portfolio value (start = 100)", f"{folder}/portfolio_value.png", logy=True)
        print(f"charts: {combo}")
