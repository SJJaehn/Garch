"""
Analysis & summary for the backtest results in Ergebnisse/.

Part 1 - per-combo time-series charts saved into each combo folder:
           rolling_sharpe.png, cumulative_sharpe.png, portfolio_value.png
Part 2 - aggregate every summary.csv into one table and a few overview charts,
         saved into Ergebnisse/Zusammenfassung/:
           aggregate_results.xlsx
           sharpe_vs_prediction.png   (Sharpe vs prediction horizon, per model)
           sharpe_vs_training.png     (Sharpe vs training period, per model)
           sharpe_by_model_cov.png    (mean Sharpe by model x covariance, bars)

Rolling Sharpe_t    = mean(excess over last WINDOW days) / std(...) * sqrt(252)
Cumulative Sharpe_t = mean(excess since inception)        / std(...) * sqrt(252)
excess = portfolio return - risk-free return (Fed Funds, same as main.py).
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import main  # reuse the risk-free series

# --- configuration -----------------------------------------------------------

ROOT        = "Ergebnisse"
SUMMARY_DIR = f"{ROOT}/Zusammenfassung"

# per-combo time-series charts
WINDOW       = 126           # rolling window length (trading days)
SMOOTH       = 1             # extra moving-average smoothing of rolling Sharpe (1 = off)
MIN_PERIODS  = 252           # min observations before a cumulative Sharpe is shown
DATE_START   = "2010-01-01"  # fixed evaluation frame (None = full history)
DATE_END     = None
MODEL_FILTER = ["MVP", "HRP", "ERC", "Naive"]   # which models to draw
COV_FILTER   = ["Historical", "GARCH", "DCC"]   # which covariance types to draw
COLUMNS      = None          # explicit column override; bypasses the filters

# overview charts
MODELS     = ["MVP", "HRP", "ERC"]
COVS       = ["Historical", "GARCH", "DCC"]
COV_COLORS = {"Historical": "tab:blue", "GARCH": "tab:orange", "DCC": "tab:green"}
SHARPE_COL = "Ann. Sharpe (frame)"   # the charts use the framed Sharpe (DATE_START/END)


# =============================================================================
# Part 1 - per-combo time-series charts
# =============================================================================

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
    for path in sorted(glob.glob(f"{ROOT}/*/*/returns.csv")):
        folder = os.path.dirname(path)
        combo = os.path.relpath(folder, ROOT)

        ret = pd.read_csv(path, index_col=0, parse_dates=True)
        cols = select_columns(list(ret.columns))
        if not cols:
            continue
        ret = ret[cols].loc[DATE_START:DATE_END]
        if ret.empty:
            continue

        rf = main.rf_daily.reindex(ret.index).fillna(0.0)
        excess = ret.sub(rf, axis=0)

        roll = (excess.rolling(WINDOW).mean() / ret.rolling(WINDOW).std()) * np.sqrt(252)
        if SMOOTH > 1:
            roll = roll.rolling(SMOOTH).mean()
        roll = roll.dropna(how="all")

        cum = (excess.expanding(MIN_PERIODS).mean() / ret.expanding(MIN_PERIODS).std()) * np.sqrt(252)
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


# =============================================================================
# Part 2 - aggregate table + overview charts
# =============================================================================

def framed_sharpe(returns_path):
    """Annualised Sharpe per portfolio over the fixed frame, from returns.csv."""
    if not os.path.exists(returns_path):
        return {}
    ret = pd.read_csv(returns_path, index_col=0, parse_dates=True).loc[DATE_START:DATE_END]
    if ret.empty:
        return {}
    rf = main.rf_daily.reindex(ret.index).fillna(0.0)
    excess = ret.sub(rf, axis=0)
    sharpe = (excess.mean() * 252) / (ret.std(ddof=1) * np.sqrt(252))
    return sharpe.to_dict()


def build_table():
    """Collect every summary.csv into one tidy DataFrame."""
    frames = []
    for path in glob.glob(f"{ROOT}/*/*/summary.csv"):
        folder = os.path.dirname(path)
        name = os.path.basename(folder)
        if "resampled" in name or "spot" in name:
            continue  # stale legacy outputs
        dataset = os.path.basename(os.path.dirname(folder))
        parts = name.split("_")
        train, pred = int(parts[0]), int(parts[1])

        garch = "1,1"
        if len(parts) >= 3:
            m = re.fullmatch(r"g(\d+)-(\d+)", parts[2])
            if m:
                garch = f"{m.group(1)},{m.group(2)}"

        df = pd.read_csv(path)
        cov = df["Covariance Type"].fillna("").astype(str)
        option = (df["Model"].astype(str) + " " + cov).str.strip()
        df.insert(0, "Dataset", dataset)
        df.insert(1, "Training Period", train)
        df.insert(2, "Prediction Horizon", pred)
        df.insert(3, "Option", option)
        df.insert(4, "Model", df.pop("Model"))
        # pandas reads the "N/A" label (Naive) as NaN; restore it for clean grouping
        df.insert(5, "Covariance Type", df.pop("Covariance Type").fillna("N/A"))
        df.insert(6, "GARCH(p,q)", garch)
        # Sharpe recomputed over the fixed frame (full-sample metrics stay alongside)
        df["Ann. Sharpe (frame)"] = df["Option"].map(framed_sharpe(f"{folder}/returns.csv"))
        frames.append(df)

    if not frames:
        raise SystemExit(f"No summary.csv files found under {ROOT}/")
    table = pd.concat(frames, ignore_index=True)
    return table.sort_values(
        ["Dataset", "Training Period", "Prediction Horizon", "GARCH(p,q)", "Option"]
    ).reset_index(drop=True)


def plot_sharpe_vs(table, axis_col, out, title):
    """One subplot per model: mean Sharpe vs `axis_col`, one line per covariance type."""
    naive = table[table["Model"] == "Naive"].groupby(axis_col)[SHARPE_COL].mean()
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 5), sharey=True)
    for ax, mdl in zip(axes, MODELS):
        sub = table[table["Model"] == mdl]
        for cov in COVS:
            s = sub[sub["Covariance Type"] == cov].groupby(axis_col)[SHARPE_COL].mean()
            if not s.empty:
                ax.plot(s.index, s.values, marker="o", label=cov, color=COV_COLORS[cov])
        if not naive.empty:
            ax.plot(naive.index, naive.values, "k--", linewidth=1, label="Naive")
        ax.set_title(mdl)
        ax.set_xlabel(axis_col)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Mean annualised Sharpe")
    axes[-1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_sharpe_bar(table, out):
    """Grouped bars: mean Sharpe by model x covariance type, with Naive reference."""
    g = table.groupby(["Model", "Covariance Type"])[SHARPE_COL].mean()
    x = np.arange(len(MODELS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cov in enumerate(COVS):
        vals = [g.get((m, cov), np.nan) for m in MODELS]
        ax.bar(x + (i - 1) * width, vals, width, label=cov, color=COV_COLORS[cov])
    naive = g.get(("Naive", "N/A"), np.nan)
    if not np.isnan(naive):
        ax.axhline(naive, color="black", linestyle="--", linewidth=1, label=f"Naive ({naive:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Mean annualised Sharpe")
    ax.set_title(f"Mean Sharpe by model and covariance type  [{DATE_START or 'start'} .. {DATE_END or 'end'}]")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def summary_outputs():
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    table = build_table()
    table.to_excel(f"{SUMMARY_DIR}/aggregate_results.xlsx", index=False)
    print(f"table: {len(table)} rows -> {SUMMARY_DIR}/aggregate_results.xlsx")

    frame = f"  [{DATE_START or 'start'} .. {DATE_END or 'end'}]"

    # --- overall charts (averaged over the other axis) -----------------------
    plot_sharpe_vs(table, "Prediction Horizon", f"{SUMMARY_DIR}/sharpe_vs_prediction.png",
                   "Mean annualised Sharpe vs prediction horizon" + frame)
    plot_sharpe_vs(table, "Training Period", f"{SUMMARY_DIR}/sharpe_vs_training.png",
                   "Mean annualised Sharpe vs training period" + frame)
    plot_sharpe_bar(table, f"{SUMMARY_DIR}/sharpe_by_model_cov.png")

    # --- per prediction horizon: Sharpe vs training period -------------------
    d = f"{SUMMARY_DIR}/sharpe_vs_training_by_pred"
    os.makedirs(d, exist_ok=True)
    for pred in sorted(table["Prediction Horizon"].unique()):
        sub = table[table["Prediction Horizon"] == pred]
        if sub["Training Period"].nunique() < 2:
            continue  # nothing to plot "over training" with a single training period
        plot_sharpe_vs(sub, "Training Period", f"{d}/pred_{pred}.png",
                       f"Sharpe vs training period  |  prediction horizon = {pred}{frame}")

    # --- per training period: Sharpe vs prediction horizon -------------------
    d = f"{SUMMARY_DIR}/sharpe_vs_prediction_by_train"
    os.makedirs(d, exist_ok=True)
    for train in sorted(table["Training Period"].unique()):
        sub = table[table["Training Period"] == train]
        if sub["Prediction Horizon"].nunique() < 2:
            continue  # nothing to plot "over prediction" with a single horizon
        plot_sharpe_vs(sub, "Prediction Horizon", f"{d}/train_{train}.png",
                       f"Sharpe vs prediction horizon  |  training period = {train}{frame}")

    print(f"summary charts -> {SUMMARY_DIR}/  (frame: {DATE_START} .. {DATE_END})")


if __name__ == "__main__":
    per_combo_charts()
    summary_outputs()
