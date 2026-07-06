"""
Analyzes the backtest results in Ergebnisse/.

  Part 1 - per-run time-series charts (rolling/cumulative Sharpe, portfolio
           value) plus bar charts (Sharpe and realized vola per model x
           covariance type, QLIKE per covariance type), saved into each
           backtest folder.
  Part 2 - collects every summary.csv into one Excel table and draws overview
           charts under Ergebnisse/Zusammenfassung/.

Rolling Sharpe_t    = mean(returns of the last WINDOW days) / std(...) * sqrt(252)
Cumulative Sharpe_t = mean(returns since the start)         / std(...) * sqrt(252)

Every Sharpe chart is drawn twice: once with rf=0 (plain returns, the default
file name) and once with the returns in excess of the Fed Funds series
(*_rf.png files and the Sharpe_rf overview folder).

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
# Settings (shared by the per-run charts and the overview charts)
# =============================================================================

# --- per-run time-series charts ----------------------------------------------
WINDOW       = 504           # rolling window length (trading days)
SMOOTH       = 1             # extra moving-average smoothing of the rolling Sharpe (1 = off)
MIN_PERIODS  = 252           # minimum observations before a cumulative Sharpe is shown
DATE_START   = "01.01.2010"  # fixed evaluation frame (None = full history)
DATE_END     = None
MODEL_FILTER = ["MVP", "HRP", "ERC", "Naive"]   # which models to draw
COV_FILTER   = ["Historical", "GARCH", "DCC"]   # which covariance types to draw
COLUMNS      = None          # explicit column override; bypasses the filters

# --- overview charts ---------------------------------------------------------
MODELS     = ["MVP", "HRP", "ERC"]
COVS       = ["Historical", "GARCH", "DCC"]
COV_COLORS = {"Historical": "tab:blue", "GARCH": "tab:orange", "DCC": "tab:green"}
# the overview charts use the Sharpes recomputed over the DATE_START/END frame
SHARPE_COL    = "Ann. Sharpe (frame)"
SHARPE_RF_COL = "Ann. Sharpe rf (frame)"


# cache for the risk-free index level, so the file is only read once
rf_level = None

"""
Daily simple risk-free return aligned to the given dates. backtest is imported
here (and not at the top) because it pulls in riskfolio/arch, which is slow.
"""
def rf_daily(index):
    global rf_level
    if rf_level is None:
        import backtest
        rf_level = backtest.read_risk_free_level()
    return rf_level.reindex(index).ffill().pct_change()


"""
Annualized Sharpe ratio per column; in excess of the daily rf if given (else rf=0).
"""
def ann_sharpe(ret, rf=None):
    if rf is not None:
        ret = ret.sub(rf, axis=0)
    return (ret.mean() * 252) / (ret.std(ddof=1) * np.sqrt(252))


# =============================================================================
# Part 1 - per-run time-series charts
# =============================================================================

"""
Applies the MODEL_FILTER / COV_FILTER (or the explicit COLUMNS override) to the
column names of a returns.csv.
"""
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


"""
Simple line chart with one line per column, saved to a file.
"""
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


"""
Grouped bar chart for one run: takes a Series indexed by the option names
("MVP GARCH", ..., "Naive") and reshapes it into the table layout that
plot_metric_bar expects.
"""
def option_series_bar(s, ylabel, out, title):
    rows = []
    for opt, val in s.items():
        mdl, cov = ("Naive", "N/A") if opt == "Naive" else opt.rsplit(" ", 1)
        rows.append({"Model": mdl, "Covariance Type": cov, "Value": val})
    plot_metric_bar(pd.DataFrame(rows), "Value", ylabel, out, title)


"""
One bar per covariance type (for metrics that have no model dimension).
"""
def plot_cov_bar(means, ylabel, out, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(means.index, means.values, width=0.6,
                  color=[COV_COLORS.get(c, "tab:gray") for c in means.index])
    # the differences are small compared to the absolute level, so the value
    # labels carry the comparison
    ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


"""
Draws the chart set for every backtest folder that contains a returns.csv:
rolling and cumulative Sharpe (with and without rf), portfolio value,
Sharpe/vola bar charts and the average QLIKE per covariance type.
"""
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

        frame = f"  [{DATE_START or 'Start'} .. {DATE_END or 'Ende'}]"
        tag = f", {SMOOTH}T-geglättet" if SMOOTH > 1 else ""
        # every Sharpe chart twice: rf=0 (plain returns) and in excess of the rf series
        exc = ret.sub(rf_daily(ret.index), axis=0)
        for suffix, rf_tag, r in (("", "rf=0", ret), ("_rf", "mit rf", exc)):
            roll = (r.rolling(WINDOW).mean() / r.rolling(WINDOW).std()) * np.sqrt(252)
            if SMOOTH > 1:
                roll = roll.rolling(SMOOTH).mean()
            roll = roll.dropna(how="all")

            cum = (r.expanding(MIN_PERIODS).mean() / r.expanding(MIN_PERIODS).std()) * np.sqrt(252)
            cum = cum.dropna(how="all")

            if not roll.empty:
                plot_lines(roll, f"Rollierender {WINDOW}-Tage-Sharpe (annualisiert, {rf_tag}){tag} – {combo}",
                           "Sharpe-Ratio", f"{folder}/rolling_sharpe{suffix}.png")
            if not cum.empty:
                plot_lines(cum, f"Kumulierter Sharpe (annualisiert, {rf_tag}) – {combo}",
                           "Sharpe-Ratio", f"{folder}/cumulative_sharpe{suffix}.png")
            # bar chart of the Sharpes over the evaluation frame
            option_series_bar(ann_sharpe(r), "Sharpe-Ratio", f"{folder}/sharpe_bar{suffix}.png",
                              f"Sharpe (annualisiert, {rf_tag}) – {combo}{frame}")

        plot_lines(100 * (1 + ret).cumprod(), f"Portfoliowert (log-Skala) – {combo}",
                   "Portfoliowert (Start = 100)", f"{folder}/portfolio_value.png", logy=True)

        vola = ret.std(ddof=1) * np.sqrt(252)
        option_series_bar(vola, "Realisierte Volatilität (annualisiert)",
                          f"{folder}/realized_vol_bar.png",
                          f"Realisierte Volatilität (annualisiert) – {combo}{frame}")

        # average QLIKE per covariance type; no Naive benchmark here because it
        # has no own covariance forecast
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

"""
Annualized Sharpes per portfolio over the fixed evaluation frame, computed from
a returns.csv. Returns ({option: sharpe rf=0}, {option: sharpe in excess of rf}).
"""
def framed_sharpe(returns_path):
    if not os.path.exists(returns_path):
        return {}, {}
    ret = pd.read_csv(returns_path, index_col=0, parse_dates=True).loc[DATE_START:DATE_END]
    if ret.empty:
        return {}, {}
    return ann_sharpe(ret).to_dict(), ann_sharpe(ret, rf_daily(ret.index)).to_dict()


"""
Collects every summary.csv under the results folder into one tidy DataFrame,
with the dataset / training period / prediction horizon parsed from the folder
names and the framed Sharpes added.
"""
def build_table():
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

        # folder names like 1008_1_g2-1 carry a non-default GARCH order
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
        # Sharpes recomputed over the fixed frame (the full-sample metrics stay alongside)
        sharpe0, sharpe_rf = framed_sharpe(f"{folder}/returns.csv")
        df[SHARPE_COL] = df["Option"].map(sharpe0)
        df[SHARPE_RF_COL] = df["Option"].map(sharpe_rf)
        frames.append(df)

    if not frames:
        raise SystemExit(f"No summary.csv files found under {root}/")
    table = pd.concat(frames, ignore_index=True)
    return table.sort_values(
        ["Dataset", "Training Period", "Prediction Horizon", "GARCH(p,q)", "Option"]
    ).reset_index(drop=True)


# German axis labels (the DataFrame columns keep their English names)
AXIS_LABELS_DE = {"Prediction Horizon": "Prognosehorizont", "Training Period": "Trainingszeitraum"}


# Metrics to chart, each into its own Zusammenfassung subfolder:
#   (column in the table, output subfolder, axis/title label, cov_only)
# cov_only means the metric only depends on the covariance forecast (it is
# identical for every model and Naive just duplicates "Historical"), so those
# charts drop the model panels and the Naive reference entirely.
METRICS = [
    (SHARPE_COL,     "Sharpe",           "Durchschnittlicher Sharpe (annualisiert, rf=0)",   False),
    (SHARPE_RF_COL,  "Sharpe_rf",        "Durchschnittlicher Sharpe (annualisiert, mit rf)", False),
    ("Ann. Std",     "Realisierte_Vola", "Realisierte Volatilität (annualisiert)",           False),
    ("Avg QLIKE",    "QLIKE",            "Durchschnittlicher QLIKE",                         True),
    ("Avg Cov RMSE", "Kovarianz_RMSE",   "Durchschnittlicher Kovarianz-RMSE",                True),
    ("ERC RC RMSE",  "ERC_RC_RMSE",      "ERC Risikobeitrags-RMSE",                          True),
]


"""
Mean of metric_col plotted against axis_col, one line per covariance type.
One subplot per model plus a dashed Naive reference line, unless cov_only is
set (then there is no model dimension and no Naive).
"""
def plot_metric_vs(table, axis_col, metric_col, ylabel, out, title, cov_only=False):
    panels = [None] if cov_only else MODELS
    fig, axes = plt.subplots(1, len(panels), figsize=(8 if cov_only else 5 * len(panels), 5),
                             sharey=True, squeeze=False)
    for ax, mdl in zip(axes[0], panels):
        sub = table if mdl is None else table[table["Model"] == mdl]
        for cov in COVS:
            s = sub[sub["Covariance Type"] == cov].groupby(axis_col)[metric_col].mean()
            if not s.empty:
                ax.plot(s.index, s.values, marker="o", label=cov, color=COV_COLORS[cov])
        if mdl is not None:
            naive = table[table["Model"] == "Naive"].groupby(axis_col)[metric_col].mean()
            if not naive.empty:
                ax.plot(naive.index, naive.values, "k--", linewidth=1, label="Naive")
            ax.set_title(mdl)
        ax.set_xlabel(AXIS_LABELS_DE.get(axis_col, axis_col))
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel(ylabel)
    axes[0, -1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


"""
Grouped bar chart: mean of metric_col per model x covariance type, with a
dashed line for the Naive benchmark.
"""
def plot_metric_bar(table, metric_col, ylabel, out, title):
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


"""
Draws the full chart set for one dataset and one metric: the overall charts
plus the breakdowns per prediction horizon and per training period.
"""
def dataset_charts(table, out_dir, metric_col, label, cov_only):
    os.makedirs(out_dir, exist_ok=True)
    frame = f"  [{DATE_START or 'Start'} .. {DATE_END or 'Ende'}]"

    # --- overall charts (averaged over the other axis) ------------------------
    plot_metric_vs(table, "Prediction Horizon", metric_col, label,
                   f"{out_dir}/vs_prediction.png", f"{label} vs. Prognosehorizont" + frame,
                   cov_only)
    plot_metric_vs(table, "Training Period", metric_col, label,
                   f"{out_dir}/vs_training.png", f"{label} vs. Trainingszeitraum" + frame,
                   cov_only)
    if cov_only:
        plot_cov_bar(table.groupby("Covariance Type")[metric_col].mean().reindex(COVS).dropna(),
                     label, f"{out_dir}/by_cov.png", f"{label} nach Kovarianztyp" + frame)
    else:
        plot_metric_bar(table, metric_col, label, f"{out_dir}/by_model_cov.png",
                        f"{label} nach Modell und Kovarianztyp" + frame)

    # --- per prediction horizon: metric vs training period --------------------
    d = f"{out_dir}/vs_training_by_pred"
    os.makedirs(d, exist_ok=True)
    for pred in sorted(table["Prediction Horizon"].unique()):
        sub = table[table["Prediction Horizon"] == pred]
        if sub["Training Period"].nunique() < 2:
            continue  # nothing to plot "over training" with a single training period
        plot_metric_vs(sub, "Training Period", metric_col, label, f"{d}/pred_{pred}.png",
                       f"{label} vs. Trainingszeitraum  |  Prognosehorizont = {pred}{frame}",
                       cov_only)

    # --- per training period: metric vs prediction horizon --------------------
    d = f"{out_dir}/vs_prediction_by_train"
    os.makedirs(d, exist_ok=True)
    for train in sorted(table["Training Period"].unique()):
        sub = table[table["Training Period"] == train]
        if sub["Prediction Horizon"].nunique() < 2:
            continue  # nothing to plot "over prediction" with a single horizon
        plot_metric_vs(sub, "Prediction Horizon", metric_col, label, f"{d}/train_{train}.png",
                       f"{label} vs. Prognosehorizont  |  Trainingszeitraum = {train}{frame}",
                       cov_only)


"""
Writes the aggregate Excel table and draws the overview charts: one subfolder
per metric, and inside it the same chart set for every dataset.
"""
def summary_outputs():
    summary_dir = config.SUMMARY_DIR
    os.makedirs(summary_dir, exist_ok=True)
    table = build_table()

    # one overall Excel covering every dataset
    table.to_excel(f"{summary_dir}/aggregate_results.xlsx", index=False)
    print(f"table: {len(table)} rows -> {summary_dir}/aggregate_results.xlsx")

    datasets = sorted(table["Dataset"].unique())
    for metric_col, folder, label, cov_only in METRICS:
        if metric_col not in table.columns or table[metric_col].notna().sum() == 0:
            print(f"skip {folder}: no '{metric_col}' values in the results")
            continue
        done = []
        for dataset in datasets:
            sub = table[table["Dataset"] == dataset]
            if sub[metric_col].notna().sum() == 0:
                continue  # this dataset has no values for the metric yet
            dataset_charts(sub, f"{summary_dir}/{folder}/{dataset}", metric_col, label, cov_only)
            done.append(dataset)
        print(f"summary charts ({folder}) -> {summary_dir}/{folder}/  {done}")

    print(f"frame: {DATE_START} .. {DATE_END}")


if __name__ == "__main__":
    per_combo_charts()
    summary_outputs()
