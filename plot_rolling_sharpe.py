"""
Plot rolling and cumulative (annualised) Sharpe ratios for every combo under
Ergebnisse/.

  rolling Sharpe_t    = mean(excess over last WINDOW days) / std(...) * sqrt(252)
  cumulative Sharpe_t = mean(excess since inception up to t) / std(...) * sqrt(252)

excess = portfolio return - risk-free return (Fed Funds, same as main.py).

Also plots the portfolio value (equity curve, log y-axis) on the same frame and
filters, so the Sharpe charts and the strategy growth are directly comparable.

Saves rolling_sharpe.png, cumulative_sharpe.png and portfolio_value.png into each
combo folder.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import main  # reuse the risk-free series

WINDOW       = 126    # rolling window length (trading days)
SMOOTH       = 1    # extra moving-average smoothing of the rolling Sharpe (1 = off)
MIN_PERIODS  = 252    # min observations before a cumulative Sharpe is shown

# Fixed evaluation frame: restrict every combo to the same dates so they are
# directly comparable (None = use each combo's full history).
DATE_START   = "2005-01-01"   # e.g. "2003-02-12"
DATE_END     = None   # e.g. "2026-06-09"

# Filters: pick which models and which covariance types to plot.
MODEL_FILTER = ["MVP"]#, "HRP", "ERC", "Naive"]        # subset of these
COV_FILTER   = ["Historical", "GARCH", "DCC"]        # subset of these
COLUMNS      = None  # explicit override; if set, ignores the filters above


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


def plot_sharpe(df, title, out):
    fig, ax = plt.subplots(figsize=(13, 6))
    for col in df.columns:
        ax.plot(df.index, df[col], label=col, linewidth=1.1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe ratio")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_value(ret, title, out):
    """Equity curve (portfolio value, start = 100) on a log y-axis."""
    value = 100 * (1 + ret).cumprod()
    fig, ax = plt.subplots(figsize=(13, 6))
    for col in value.columns:
        ax.plot(value.index, value[col], label=col, linewidth=1.1)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value (log scale, start = 100)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


for path in sorted(glob.glob("Ergebnisse/*/*/returns.csv")):
    folder = os.path.dirname(path)
    combo = os.path.relpath(folder, "Ergebnisse")

    ret = pd.read_csv(path, index_col=0, parse_dates=True)
    cols = select_columns(list(ret.columns))
    if not cols:
        print(f"skip {combo} (no columns match the filter)")
        continue
    ret = ret[cols]

    # restrict to the fixed evaluation frame, if set
    ret = ret.loc[DATE_START:DATE_END]
    if ret.empty:
        print(f"skip {combo} (no data in the fixed frame)")
        continue

    rf = main.rf_daily.reindex(ret.index).fillna(0.0)
    excess = ret.sub(rf, axis=0)

    # rolling Sharpe (optionally smoothed)
    roll = (excess.rolling(WINDOW).mean() / ret.rolling(WINDOW).std()) * np.sqrt(252)
    if SMOOTH > 1:
        roll = roll.rolling(SMOOTH).mean()
    roll = roll.dropna(how="all")

    # cumulative (expanding) Sharpe
    cum = (excess.expanding(MIN_PERIODS).mean() / ret.expanding(MIN_PERIODS).std()) * np.sqrt(252)
    cum = cum.dropna(how="all")

    if not roll.empty:
        smooth_tag = f", {SMOOTH}d-smoothed" if SMOOTH > 1 else ""
        plot_sharpe(roll, f"Rolling {WINDOW}-day annualised Sharpe{smooth_tag} — {combo}",
                    f"{folder}/rolling_sharpe.png")
    if not cum.empty:
        plot_sharpe(cum, f"Cumulative annualised Sharpe — {combo}",
                    f"{folder}/cumulative_sharpe.png")

    # portfolio value (equity curve) on the same frame / filters, log y-axis
    plot_value(ret, f"Portfolio value (log scale) — {combo}",
               f"{folder}/portfolio_value.png")
    print(f"done {combo}")
