"""
Aggregate every summary.csv into one table and a few overview charts, saved into
Ergebnisse/Zusammenfassung/:
  aggregate_results.xlsx
  sharpe_vs_prediction.png   (Sharpe vs prediction horizon, per model)
  sharpe_vs_training.png     (Sharpe vs training period, per model)
  sharpe_by_model_cov.png    (mean Sharpe by model x covariance, bars)
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from garch import config
from garch.analysis import (COVS, COV_COLORS, DATE_END, DATE_START, MODELS, SHARPE_COL)
from garch.data import loaders


def framed_sharpe(returns_path, rf_level):
    """Annualised Sharpe per portfolio over the fixed frame, from returns.csv."""
    if not os.path.exists(returns_path):
        return {}
    ret = pd.read_csv(returns_path, index_col=0, parse_dates=True).loc[DATE_START:DATE_END]
    if ret.empty:
        return {}
    rf = loaders.align_risk_free(rf_level, ret.index).fillna(0.0)
    excess = ret.sub(rf, axis=0)
    sharpe = (excess.mean() * 252) / (excess.std(ddof=1) * np.sqrt(252))
    return sharpe.to_dict()


def build_table():
    """Collect every summary.csv into one tidy DataFrame."""
    root = config.RESULTS_DIR
    rf_level = loaders.read_risk_free_level()
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


def dataset_charts(table, out_dir):
    """All overview charts for a single dataset's slice of the table."""
    os.makedirs(out_dir, exist_ok=True)
    frame = f"  [{DATE_START or 'start'} .. {DATE_END or 'end'}]"

    # --- overall charts (averaged over the other axis) -----------------------
    plot_sharpe_vs(table, "Prediction Horizon", f"{out_dir}/sharpe_vs_prediction.png",
                   "Mean annualised Sharpe vs prediction horizon" + frame)
    plot_sharpe_vs(table, "Training Period", f"{out_dir}/sharpe_vs_training.png",
                   "Mean annualised Sharpe vs training period" + frame)
    plot_sharpe_bar(table, f"{out_dir}/sharpe_by_model_cov.png")

    # --- per prediction horizon: Sharpe vs training period -------------------
    d = f"{out_dir}/sharpe_vs_training_by_pred"
    os.makedirs(d, exist_ok=True)
    for pred in sorted(table["Prediction Horizon"].unique()):
        sub = table[table["Prediction Horizon"] == pred]
        if sub["Training Period"].nunique() < 2:
            continue  # nothing to plot "over training" with a single training period
        plot_sharpe_vs(sub, "Training Period", f"{d}/pred_{pred}.png",
                       f"Sharpe vs training period  |  prediction horizon = {pred}{frame}")

    # --- per training period: Sharpe vs prediction horizon -------------------
    d = f"{out_dir}/sharpe_vs_prediction_by_train"
    os.makedirs(d, exist_ok=True)
    for train in sorted(table["Training Period"].unique()):
        sub = table[table["Training Period"] == train]
        if sub["Prediction Horizon"].nunique() < 2:
            continue  # nothing to plot "over prediction" with a single horizon
        plot_sharpe_vs(sub, "Prediction Horizon", f"{d}/train_{train}.png",
                       f"Sharpe vs prediction horizon  |  training period = {train}{frame}")


def summary_outputs():
    summary_dir = config.SUMMARY_DIR
    os.makedirs(summary_dir, exist_ok=True)
    table = build_table()

    # one overall Excel covering every dataset
    table.to_excel(f"{summary_dir}/aggregate_results.xlsx", index=False)
    print(f"table: {len(table)} rows -> {summary_dir}/aggregate_results.xlsx")

    # charts generated separately per dataset (TRBC, SP500, ...)
    for dataset in sorted(table["Dataset"].unique()):
        sub = table[table["Dataset"] == dataset]
        out_dir = f"{summary_dir}/{dataset}"
        dataset_charts(sub, out_dir)
        print(f"summary charts ({dataset}) -> {out_dir}/")

    print(f"frame: {DATE_START} .. {DATE_END}")
