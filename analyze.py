"""
Analyse the backtest results in Ergebnisse/.

  Part 1 - per-combo time-series charts (rolling/cumulative Sharpe, portfolio value)
            saved into each backtest folder.
  Part 2 - aggregate every summary.csv into one Excel table + overview charts under
            Ergebnisse/Zusammenfassung/.

Rolling Sharpe_t    = mean(excess over last WINDOW days) / std(...) * sqrt(252)
Cumulative Sharpe_t = mean(excess since inception)        / std(...) * sqrt(252)
excess = portfolio return - risk-free return (Fed Funds, same as the backtest).

The shared knobs (evaluation frame, which models/covariances to draw, colours)
live in the SETTINGS block below. The risk-free loaders are reused from
backtest.py (importing it is side-effect free — its run lives under __main__).

    python analyze.py
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from backtest import align_risk_free, read_risk_free_level

# =============================================================================
# SETTINGS  -- shared by the per-combo charts and the overview charts
# =============================================================================

# --- per-combo time-series charts --------------------------------------------
WINDOW       = 126           # rolling window length (trading days)
SMOOTH       = 1             # extra moving-average smoothing of rolling Sharpe (1 = off)
MIN_PERIODS  = 252           # min observations before a cumulative Sharpe is shown
DATE_START   = "2010-01-01"  # fixed evaluation frame (None = full history)
DATE_END     = None
MODEL_FILTER = ["MVP", "HRP", "ERC", "Naive"]   # which models to draw
COV_FILTER   = ["Historical", "GARCH", "DCC"]   # which covariance types to draw
COLUMNS      = None          # explicit column override; bypasses the filters

# --- overview charts ---------------------------------------------------------
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
    ax.set_xlabel("Datum")
    ax.set_ylabel(ylabel)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def per_combo_charts():
    root = config.RESULTS_DIR
    rf_level = read_risk_free_level()
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

        rf = align_risk_free(rf_level, ret.index).fillna(0.0)
        excess = ret.sub(rf, axis=0)

        roll = (excess.rolling(WINDOW).mean() / excess.rolling(WINDOW).std()) * np.sqrt(252)
        if SMOOTH > 1:
            roll = roll.rolling(SMOOTH).mean()
        roll = roll.dropna(how="all")

        cum = (excess.expanding(MIN_PERIODS).mean() / excess.expanding(MIN_PERIODS).std()) * np.sqrt(252)
        cum = cum.dropna(how="all")

        if not roll.empty:
            tag = f", {SMOOTH}T-geglättet" if SMOOTH > 1 else ""
            plot_lines(roll, f"Rollierender {WINDOW}-Tage-Sharpe (annualisiert){tag} – {combo}",
                       "Sharpe-Ratio", f"{folder}/rolling_sharpe.png")
        if not cum.empty:
            plot_lines(cum, f"Kumulierter Sharpe (annualisiert) – {combo}",
                       "Sharpe-Ratio", f"{folder}/cumulative_sharpe.png")
        plot_lines(100 * (1 + ret).cumprod(), f"Portfoliowert (log-Skala) – {combo}",
                   "Portfoliowert (Start = 100)", f"{folder}/portfolio_value.png", logy=True)
        print(f"charts: {combo}")


# =============================================================================
# Part 2 - aggregate table + overview charts
# =============================================================================

def framed_sharpe(returns_path, rf_level):
    """Annualised Sharpe per portfolio over the fixed frame, from returns.csv."""
    if not os.path.exists(returns_path):
        return {}
    ret = pd.read_csv(returns_path, index_col=0, parse_dates=True).loc[DATE_START:DATE_END]
    if ret.empty:
        return {}
    rf = align_risk_free(rf_level, ret.index).fillna(0.0)
    excess = ret.sub(rf, axis=0)
    sharpe = (excess.mean() * 252) / (excess.std(ddof=1) * np.sqrt(252))
    return sharpe.to_dict()


def build_table():
    """Collect every summary.csv into one tidy DataFrame."""
    root = config.RESULTS_DIR
    rf_level = read_risk_free_level()
    frames = []
    for path in glob.glob(f"{root}/*/*/summary.csv"):
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
        df["Ann. Sharpe (frame)"] = df["Option"].map(framed_sharpe(f"{folder}/returns.csv", rf_level))
        frames.append(df)

    if not frames:
        raise SystemExit(f"No summary.csv files found under {root}/")
    table = pd.concat(frames, ignore_index=True)
    return table.sort_values(
        ["Dataset", "Training Period", "Prediction Horizon", "GARCH(p,q)", "Option"]
    ).reset_index(drop=True)


# German display labels for axis columns (the DataFrame columns keep their names)
_AXIS_DE = {"Prediction Horizon": "Prognosehorizont", "Training Period": "Trainingszeitraum"}


# metrics to chart, each into its own Zusammenfassung subfolder:
#   (column in the table, output subfolder, axis/title label)
METRICS = [
    (SHARPE_COL,     "Sharpe",           "Durchschnittlicher Sharpe (annualisiert)"),
    ("Ann. Std",     "Realisierte_Vola", "Realisierte Volatilität (annualisiert)"),
    ("Avg QLIKE",    "QLIKE",            "Durchschnittlicher QLIKE"),
    ("Avg Cov RMSE", "Kovarianz_RMSE",   "Durchschnittlicher Kovarianz-RMSE"),
    ("ERC RC RMSE",  "ERC_RC_RMSE",      "ERC Risikobeitrags-RMSE"),
]


def plot_metric_vs(table, axis_col, metric_col, ylabel, out, title):
    """One subplot per model: mean `metric_col` vs `axis_col`, one line per covariance type."""
    naive = table[table["Model"] == "Naive"].groupby(axis_col)[metric_col].mean()
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 5), sharey=True)
    for ax, mdl in zip(axes, MODELS):
        sub = table[table["Model"] == mdl]
        for cov in COVS:
            s = sub[sub["Covariance Type"] == cov].groupby(axis_col)[metric_col].mean()
            if not s.empty:
                ax.plot(s.index, s.values, marker="o", label=cov, color=COV_COLORS[cov])
        if not naive.empty:
            ax.plot(naive.index, naive.values, "k--", linewidth=1, label="Naive")
        ax.set_title(mdl)
        ax.set_xlabel(_AXIS_DE.get(axis_col, axis_col))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_metric_bar(table, metric_col, ylabel, out, title):
    """Grouped bars: mean `metric_col` by model x covariance type, with Naive reference."""
    g = table.groupby(["Model", "Covariance Type"])[metric_col].mean()
    x = np.arange(len(MODELS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cov in enumerate(COVS):
        vals = [g.get((m, cov), np.nan) for m in MODELS]
        ax.bar(x + (i - 1) * width, vals, width, label=cov, color=COV_COLORS[cov])
    naive = g.get(("Naive", "N/A"), np.nan)
    if not np.isnan(naive):
        ax.axhline(naive, color="black", linestyle="--", linewidth=1, label=f"Naive ({naive:.3g})")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def dataset_charts(table, out_dir, metric_col, label):
    """The full chart set for one dataset's slice and one metric."""
    os.makedirs(out_dir, exist_ok=True)
    frame = f"  [{DATE_START or 'Start'} .. {DATE_END or 'Ende'}]"

    # --- overall charts (averaged over the other axis) -----------------------
    plot_metric_vs(table, "Prediction Horizon", metric_col, label,
                   f"{out_dir}/vs_prediction.png", f"{label} vs. Prognosehorizont" + frame)
    plot_metric_vs(table, "Training Period", metric_col, label,
                   f"{out_dir}/vs_training.png", f"{label} vs. Trainingszeitraum" + frame)
    plot_metric_bar(table, metric_col, label, f"{out_dir}/by_model_cov.png",
                    f"{label} nach Modell und Kovarianztyp" + frame)

    # --- per prediction horizon: metric vs training period -------------------
    d = f"{out_dir}/vs_training_by_pred"
    os.makedirs(d, exist_ok=True)
    for pred in sorted(table["Prediction Horizon"].unique()):
        sub = table[table["Prediction Horizon"] == pred]
        if sub["Training Period"].nunique() < 2:
            continue  # nothing to plot "over training" with a single training period
        plot_metric_vs(sub, "Training Period", metric_col, label, f"{d}/pred_{pred}.png",
                       f"{label} vs. Trainingszeitraum  |  Prognosehorizont = {pred}{frame}")

    # --- per training period: metric vs prediction horizon -------------------
    d = f"{out_dir}/vs_prediction_by_train"
    os.makedirs(d, exist_ok=True)
    for train in sorted(table["Training Period"].unique()):
        sub = table[table["Training Period"] == train]
        if sub["Prediction Horizon"].nunique() < 2:
            continue  # nothing to plot "over prediction" with a single horizon
        plot_metric_vs(sub, "Prediction Horizon", metric_col, label, f"{d}/train_{train}.png",
                       f"{label} vs. Prognosehorizont  |  Trainingszeitraum = {train}{frame}")


def summary_outputs():
    summary_dir = config.SUMMARY_DIR
    os.makedirs(summary_dir, exist_ok=True)
    table = build_table()

    # one overall Excel covering every dataset
    table.to_excel(f"{summary_dir}/aggregate_results.xlsx", index=False)
    print(f"table: {len(table)} rows -> {summary_dir}/aggregate_results.xlsx")

    # one subfolder per metric; inside it the same chart set per dataset
    datasets = sorted(table["Dataset"].unique())
    for metric_col, folder, label in METRICS:
        if metric_col not in table.columns or table[metric_col].notna().sum() == 0:
            print(f"skip {folder}: no '{metric_col}' values in the results")
            continue
        done = []
        for dataset in datasets:
            sub = table[table["Dataset"] == dataset]
            if sub[metric_col].notna().sum() == 0:
                continue  # this dataset has no values for the metric yet -> skip
            dataset_charts(sub, f"{summary_dir}/{folder}/{dataset}", metric_col, label)
            done.append(dataset)
        print(f"summary charts ({folder}) -> {summary_dir}/{folder}/  {done}")

    print(f"frame: {DATE_START} .. {DATE_END}")


if __name__ == "__main__":
    per_combo_charts()
    summary_outputs()
