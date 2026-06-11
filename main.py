import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import (
    calculate_metrics,
    calculate_summary_metrics,
    estimate_cov_matrix_garch,
    estimate_cov_matrix_historical,
    get_erc_weights,
    get_hrp_weights,
    get_mvp_weights,
)

# --- Configuration -----------------------------------------------------------

DATASETS: dict[str, tuple[str, str]] = {                  # (filepath, date_format)
    "TRBC":  ("TRBC_Business_Sectors_clean.csv", "%Y-%m-%d"),  # yyyy-mm-dd
    "SP500": ("S&P500_Adj.csv",                  "%d.%m.%y"),  # dd.mm.yy
}

MAX_WORKERS = 6
DATASET              = "TRBC"
TRAIN_WINDOW         = 3*252
PREDICTION_WINDOW    = 1
USE_RESAMPLING_GARCH = True

_GARCH_MODE = "resampled" if USE_RESAMPLING_GARCH else "spot"
_OUTPUT_DIR = f"Abbildungen/{DATASET}/{TRAIN_WINDOW}_{PREDICTION_WINDOW}_{_GARCH_MODE}"

_MODEL_TYPE: dict[str, tuple[str, str]] = {
    "HRP GARCH":     ("HRP",   "GARCH"),
    "HRP Historical":("HRP",   "Historical"),
    "MVP GARCH":     ("MVP",   "GARCH"),
    "MVP Historical":("MVP",   "Historical"),
    "ERC GARCH":     ("ERC",   "GARCH"),
    "ERC Historical":("ERC",   "Historical"),
    "Naive":         ("Naive", "N/A"),
}

# --- Load data ---------------------------------------------------------------

filepath, date_format = DATASETS[DATASET]
prices = pd.read_csv(filepath, index_col=0)
prices.index = pd.to_datetime(prices.index, format=date_format, errors="coerce")
prices = prices[prices.index.notna()]
prices = prices.loc[prices.notna().sum(axis=1) >= int(0.5 * prices.shape[1])]
returns = prices.pct_change(fill_method=None).iloc[1:]
zero_frac = (returns == 0).sum() / returns.notna().sum()
returns = returns.loc[:, zero_frac < 0.5]

# --- Rolling backtest --------------------------------------------------------


def process_window(start):
    """Process a single rolling window; returns (per_period_returns, records)."""
    train = returns.iloc[start : start + TRAIN_WINDOW]
    test  = returns.iloc[start + TRAIN_WINDOW : start + TRAIN_WINDOW + PREDICTION_WINDOW]

    # Per window, drop stocks with <90% observations; rows are kept as-is.
    train = train.loc[:, train.notna().mean(axis=0) >= 0.9]
    test  = test[train.columns].dropna(axis=1)
    train = train[test.columns]

    if train.empty or test.empty or train.shape[1] == 0:
        return {}, [], []

    garch_horizon = PREDICTION_WINDOW if USE_RESAMPLING_GARCH else 0
    label = f"{train.index[0].date()} – {train.index[-1].date()}"
    cov_garch  = estimate_cov_matrix_garch(train, prediction_window=garch_horizon, window_label=label)
    cov_hist   = estimate_cov_matrix_historical(train)
    test_garch = test[cov_garch.columns]
    test_hist  = test[cov_hist.columns]

    portfolios = {
        "HRP GARCH":     (get_hrp_weights(cov_garch), test_garch),
        "HRP Historical":(get_hrp_weights(cov_hist),  test_hist),
        "MVP GARCH":     (get_mvp_weights(cov_garch), test_garch),
        "MVP Historical":(get_mvp_weights(cov_hist),  test_hist),
        "ERC GARCH":     (get_erc_weights(cov_garch), test_garch),
        "ERC Historical":(get_erc_weights(cov_hist),  test_hist),
        "Naive":         (pd.Series(np.ones(len(train.columns)) / len(train.columns), index=train.columns), test_hist),
    }

    per_period: dict[str, list] = {}
    recs = []
    for name, (weights, test_slice) in portfolios.items():
        m = calculate_metrics(test_slice, weights)
        w = weights.values
        cov = (cov_garch if _MODEL_TYPE[name][1] == "GARCH" else cov_hist).values
        forecasted_std = float(np.sqrt(w @ cov @ w))
        per_period[name] = m["per_period_returns"].values.tolist()
        model, cov_type = _MODEL_TYPE[name]
        recs.append({
            "Model": model,
            "Covariance Type": cov_type,
            "#Rolling Windows": start // PREDICTION_WINDOW,
            "Mean Return": m["mean_return"],
            "Forecasted Std": forecasted_std,
        })
    return per_period, recs, list(test.index)


def main():
    results: dict[str, list] = {n: [] for n in _MODEL_TYPE}
    records = []
    period_dates: list = []

    starts = list(range(0, len(returns) - TRAIN_WINDOW - PREDICTION_WINDOW, PREDICTION_WINDOW))
    total = len(starts)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, (per_period, recs, dates) in enumerate(executor.map(process_window, starts), start=1):
            records.extend(recs)
            period_dates.extend(dates)
            for name, rets in per_period.items():
                results[name].extend(rets)
            print(f"\r{i}/{total} windows completed", end="", flush=True)
    print()

    metrics = pd.DataFrame(records)

    # --- Summary -------------------------------------------------------------

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    metrics.to_csv(f"{_OUTPUT_DIR}/backtest_metrics.csv", index=False)

    summary_rows = []
    for name, rets in results.items():
        if not rets:
            continue
        model, cov_type = _MODEL_TYPE[name]
        mask = (metrics["Model"] == model) & (metrics["Covariance Type"] == cov_type)
        avg_forecasted_ann = metrics.loc[mask, "Forecasted Std"].mean() * np.sqrt(252)
        row = calculate_summary_metrics(np.array(rets))
        ann_std = row["Ann. Std"]
        row["Model"] = model
        row["Covariance Type"] = cov_type
        row["Ann. Std (fcst)"] = avg_forecasted_ann
        row["Real / Fcst Std"] = ann_std / avg_forecasted_ann if avg_forecasted_ann > 0 else np.nan
        summary_rows.append(row)

    col_order = ["Model", "Covariance Type", "Ann. Return", "Ann. Std", "Ann. Std (fcst)",
                 "Real / Fcst Std", "Ann. Sharpe", "Ann. Sortino", "Max Drawdown",
                 "Calmar Ratio", "CVaR (95%)", "Skewness", "Excess Kurtosis"]
    summary = pd.DataFrame(summary_rows).sort_values(["Model", "Covariance Type"])[col_order]
    summary.to_csv(f"{_OUTPUT_DIR}/summary.csv", index=False)
    print("Annualized performance summary:")
    print(summary.to_string(index=False))

    # --- Plot ----------------------------------------------------------------

    if not any(results.values()):
        print("No valid rolling windows to plot.")
        return

    cumulative = {name: 100 * np.cumprod(1 + np.array(rets)) for name, rets in results.items()}
    x_axis = pd.to_datetime(period_dates)
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, values in cumulative.items():
        ax.plot(x_axis, values, label=name)
    ax.set_title("Portfolio Value Starting at 100")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.savefig(f"{_OUTPUT_DIR}/Simulation.png")


if __name__ == "__main__":
    main()
