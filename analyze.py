"""
Analyse the backtest results in Ergebnisse/.

  Part 1 - per-combo time-series charts (rolling/cumulative Sharpe, portfolio value)
            plus bar charts (Sharpe and realized vola by model x covariance type,
            QLIKE by covariance type) saved into each backtest folder.
  Part 2 - aggregate every summary.csv into one Excel table + overview charts under
            Ergebnisse/Zusammenfassung/.

Rolling Sharpe_t    = mean(return over last WINDOW days) / std(...) * sqrt(252)
Cumulative Sharpe_t = mean(return since inception)        / std(...) * sqrt(252)
All Sharpe figures use a zero risk-free rate (excess return = portfolio return).

The shared knobs (evaluation frame, which models/covariances to draw, colours)
live in the SETTINGS block below.

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

import main as config

# =============================================================================
# SETTINGS  -- shared by the per-combo charts and the overview charts
# =============================================================================

# --- per-combo time-series charts --------------------------------------------
WINDOW       = 504           # rolling window length (trading days)
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


def option_series_bar(s, ylabel, out, title, show_naive=True):
    """Grouped bars for one run: an option-indexed Series ("MVP GARCH", ..., "Naive")
    reshaped into the table layout plot_metric_bar expects."""
    rows = []
    for opt, val in s.items():
        mdl, cov = ("Naive", "N/A") if opt == "Naive" else opt.rsplit(" ", 1)
        rows.append({"Model": mdl, "Covariance Type": cov, "Value": val})
    plot_metric_bar(pd.DataFrame(rows), "Value", ylabel, out, title, show_naive)


def plot_cov_bar(means, ylabel, out, title):
    """One bar per covariance type (QLIKE has no model dimension within a run)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(means.index, means.values, width=0.6,
                  color=[COV_COLORS.get(c, "tab:gray") for c in means.index])
    # differences are small vs. the absolute level -> labels carry the comparison
    ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def per_combo_charts():
    root = config.RESULTS_DIR
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

        # Sharpe with a zero risk-free rate: excess return = portfolio return
        roll = (ret.rolling(WINDOW).mean() / ret.rolling(WINDOW).std()) * np.sqrt(252)
        if SMOOTH > 1:
            roll = roll.rolling(SMOOTH).mean()
        roll = roll.dropna(how="all")

        cum = (ret.expanding(MIN_PERIODS).mean() / ret.expanding(MIN_PERIODS).std()) * np.sqrt(252)
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

        # per-run bar charts over the evaluation frame, styled like by_model_cov.png
        frame = f"  [{DATE_START or 'Start'} .. {DATE_END or 'Ende'}]"
        # Sharpe with a zero risk-free rate: excess return = portfolio return
        sharpe = (ret.mean() * 252) / (ret.std(ddof=1) * np.sqrt(252))
        option_series_bar(sharpe, "Sharpe-Ratio", f"{folder}/sharpe_bar.png",
                          f"Sharpe (annualisiert) – {combo}{frame}")
        vola = ret.std(ddof=1) * np.sqrt(252)
        option_series_bar(vola, "Realisierte Volatilität (annualisiert)",
                          f"{folder}/realized_vol_bar.png",
                          f"Realisierte Volatilität (annualisiert) – {combo}{frame}")

        # mean QLIKE per covariance type; no Naive benchmark (it has no own forecast)
        qpath = f"{folder}/qlike.csv"
        if os.path.exists(qpath):
            q = pd.read_csv(qpath, index_col=0, parse_dates=True).loc[DATE_START:DATE_END]
            q = q[[c for c in q.columns if c in COV_FILTER]].dropna(how="all")
            if not q.empty:
                plot_cov_bar(q.mean(), "Durchschnittlicher QLIKE", f"{folder}/qlike_bar.png",
                             f"Durchschnittlicher QLIKE – {combo}{frame}")
        print(f"charts: {combo}")


# =============================================================================
# Part 2 - aggregate table + overview charts
# =============================================================================

def framed_sharpe(returns_path):
    """Annualised Sharpe (rf=0) per portfolio over the fixed frame, from returns.csv."""
    if not os.path.exists(returns_path):
        return {}
    ret = pd.read_csv(returns_path, index_col=0, parse_dates=True).loc[DATE_START:DATE_END]
    if ret.empty:
        return {}
    # Sharpe with a zero risk-free rate: excess return = portfolio return
    sharpe = (ret.mean() * 252) / (ret.std(ddof=1) * np.sqrt(252))
    return sharpe.to_dict()


def build_table():
    """Collect every summary.csv into one tidy DataFrame."""
    root = config.RESULTS_DIR
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
        df["Ann. Sharpe (frame)"] = df["Option"].map(framed_sharpe(f"{folder}/returns.csv"))
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
#   (column in the table, output subfolder, axis/title label, draw Naive reference)
# no Naive reference for the covariance-forecast metrics: Naive uses the
# historical covariance, so its line would just duplicate "Historical"
METRICS = [
    (SHARPE_COL,     "Sharpe",           "Durchschnittlicher Sharpe (annualisiert)", True),
    ("Ann. Std",     "Realisierte_Vola", "Realisierte Volatilität (annualisiert)",   True),
    ("Avg QLIKE",    "QLIKE",            "Durchschnittlicher QLIKE",                 False),
    ("Avg Cov RMSE", "Kovarianz_RMSE",   "Durchschnittlicher Kovarianz-RMSE",        False),
    ("ERC RC RMSE",  "ERC_RC_RMSE",      "ERC Risikobeitrags-RMSE",                  False),
]


def plot_metric_vs(table, axis_col, metric_col, ylabel, out, title, show_naive=True):
    """One subplot per model: mean `metric_col` vs `axis_col`, one line per covariance type."""
    naive = table[table["Model"] == "Naive"].groupby(axis_col)[metric_col].mean()
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 5), sharey=True)
    for ax, mdl in zip(axes, MODELS):
        sub = table[table["Model"] == mdl]
        for cov in COVS:
            s = sub[sub["Covariance Type"] == cov].groupby(axis_col)[metric_col].mean()
            if not s.empty:
                ax.plot(s.index, s.values, marker="o", label=cov, color=COV_COLORS[cov])
        if show_naive and not naive.empty:
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


def plot_metric_bar(table, metric_col, ylabel, out, title, show_naive=True):
    """Grouped bars: mean `metric_col` by model x covariance type, with Naive reference."""
    g = table.groupby(["Model", "Covariance Type"])[metric_col].mean()
    x = np.arange(len(MODELS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cov in enumerate(COVS):
        vals = [g.get((m, cov), np.nan) for m in MODELS]
        ax.bar(x + (i - 1) * width, vals, width, label=cov, color=COV_COLORS[cov])
    naive = g.get(("Naive", "N/A"), np.nan) if show_naive else np.nan
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


def dataset_charts(table, out_dir, metric_col, label, show_naive):
    """The full chart set for one dataset's slice and one metric."""
    os.makedirs(out_dir, exist_ok=True)
    frame = f"  [{DATE_START or 'Start'} .. {DATE_END or 'Ende'}]"

    # --- overall charts (averaged over the other axis) -----------------------
    plot_metric_vs(table, "Prediction Horizon", metric_col, label,
                   f"{out_dir}/vs_prediction.png", f"{label} vs. Prognosehorizont" + frame,
                   show_naive)
    plot_metric_vs(table, "Training Period", metric_col, label,
                   f"{out_dir}/vs_training.png", f"{label} vs. Trainingszeitraum" + frame,
                   show_naive)
    plot_metric_bar(table, metric_col, label, f"{out_dir}/by_model_cov.png",
                    f"{label} nach Modell und Kovarianztyp" + frame, show_naive)

    # --- per prediction horizon: metric vs training period -------------------
    d = f"{out_dir}/vs_training_by_pred"
    os.makedirs(d, exist_ok=True)
    for pred in sorted(table["Prediction Horizon"].unique()):
        sub = table[table["Prediction Horizon"] == pred]
        if sub["Training Period"].nunique() < 2:
            continue  # nothing to plot "over training" with a single training period
        plot_metric_vs(sub, "Training Period", metric_col, label, f"{d}/pred_{pred}.png",
                       f"{label} vs. Trainingszeitraum  |  Prognosehorizont = {pred}{frame}",
                       show_naive)

    # --- per training period: metric vs prediction horizon -------------------
    d = f"{out_dir}/vs_prediction_by_train"
    os.makedirs(d, exist_ok=True)
    for train in sorted(table["Training Period"].unique()):
        sub = table[table["Training Period"] == train]
        if sub["Prediction Horizon"].nunique() < 2:
            continue  # nothing to plot "over prediction" with a single horizon
        plot_metric_vs(sub, "Prediction Horizon", metric_col, label, f"{d}/train_{train}.png",
                       f"{label} vs. Prognosehorizont  |  Trainingszeitraum = {train}{frame}",
                       show_naive)


def summary_outputs():
    summary_dir = config.SUMMARY_DIR
    os.makedirs(summary_dir, exist_ok=True)
    table = build_table()

    # one overall Excel covering every dataset
    table.to_excel(f"{summary_dir}/aggregate_results.xlsx", index=False)
    print(f"table: {len(table)} rows -> {summary_dir}/aggregate_results.xlsx")

    # one subfolder per metric; inside it the same chart set per dataset
    datasets = sorted(table["Dataset"].unique())
    for metric_col, folder, label, show_naive in METRICS:
        if metric_col not in table.columns or table[metric_col].notna().sum() == 0:
            print(f"skip {folder}: no '{metric_col}' values in the results")
            continue
        done = []
        for dataset in datasets:
            sub = table[table["Dataset"] == dataset]
            if sub[metric_col].notna().sum() == 0:
                continue  # this dataset has no values for the metric yet -> skip
            dataset_charts(sub, f"{summary_dir}/{folder}/{dataset}", metric_col, label, show_naive)
            done.append(dataset)
        print(f"summary charts ({folder}) -> {summary_dir}/{folder}/  {done}")

    print(f"frame: {DATE_START} .. {DATE_END}")


if __name__ == "__main__":
    per_combo_charts()
    summary_outputs()
