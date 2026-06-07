import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import (
    calculate_metrics,
    calculate_summary_metrics,
    estimate_cov_matrix_garch,
    estimate_cov_matrix_historical,
    get_erc_weights,
    # get_hrp_weights,
    get_mvp_weights,
)

# --- Configuration -----------------------------------------------------------

DATASETS: dict[str, tuple[str, bool]] = {       # (filepath, dayfirst)
    "TRBC":  ("TRBC_Daily.csv",  False),          # yyyy-mm-dd
    "SP500": ("S&P500_Adj.csv",  True),            # dd.mm.yyyy
}

DATASET              = "SP500"
TRAIN_WINDOW         = 2*252  # 2 years of trading days
PREDICTION_WINDOW    = 21       # 1 month of trading days
USE_RESAMPLING_GARCH = False      # True → GARCH forecast over horizon; False → last conditional vol

# -----------------------------------------------------------------------------

_GARCH_MODE = "resampled" if USE_RESAMPLING_GARCH else "spot"
_OUTPUT_DIR = f"Abbildungen/{DATASET}/{TRAIN_WINDOW}_{PREDICTION_WINDOW}_{_GARCH_MODE}"

_MODEL_TYPE: dict[str, tuple[str, str]] = {
    # "HRP GARCH":     ("HRP",   "GARCH"),
    # "HRP Historical":("HRP",   "Historical"),
    "MVP GARCH":     ("MVP",   "GARCH"),
    "MVP Historical":("MVP",   "Historical"),
    "ERC GARCH":     ("ERC",   "GARCH"),
    "ERC Historical":("ERC",   "Historical"),
    "Naive":         ("Naive", "N/A"),
}


def load_returns(filepath, dayfirst=False, max_zero_frac=0.5, min_coverage=0.75):
    prices = pd.read_csv(filepath, index_col=0)
    prices.index = pd.to_datetime(prices.index, dayfirst=dayfirst, errors="coerce")
    prices = prices[prices.index.notna()]
    prices = prices.loc[prices.notna().sum(axis=1) >= int(min_coverage * prices.shape[1])]
    returns = prices.pct_change(fill_method=None)
    zero_frac = (returns == 0).sum() / returns.notna().sum()
    return returns.loc[:, zero_frac < max_zero_frac]


def run_rolling_backtest(returns, use_resampling_garch=False):
    results: dict[str, list] = {n: [] for n in _MODEL_TYPE}
    records = []

    for start in range(0, len(returns) - TRAIN_WINDOW - PREDICTION_WINDOW, PREDICTION_WINDOW):
        train = returns.iloc[start : start + TRAIN_WINDOW]
        test  = returns.iloc[start + TRAIN_WINDOW : start + TRAIN_WINDOW + PREDICTION_WINDOW]

        # Drop timesteps where 75%+ of series are missing, then drop any security with any NaN
        train = train.dropna(axis=0, thresh=int(0.75 * train.shape[1])).dropna(axis=1)
        test  = test.dropna(axis=0, thresh=int(0.75 * test.shape[1]))
        test  = test[train.columns].dropna(axis=1)
        train = train[test.columns]

        if train.empty or test.empty or train.shape[1] == 0:
            continue

        garch_horizon = PREDICTION_WINDOW if use_resampling_garch else 0
        label = f"{train.index[0].date()} – {train.index[-1].date()}"
        cov_garch = estimate_cov_matrix_garch(train, prediction_window=garch_horizon, window_label=label)
        cov_hist  = estimate_cov_matrix_historical(train)
        test_garch = test[cov_garch.columns]
        test_hist  = test[cov_hist.columns]
        portfolios = {
            # "HRP GARCH":     (get_hrp_weights(cov_garch), test_garch),
            # "HRP Historical":(get_hrp_weights(cov_hist),  test_hist),
            "MVP GARCH":     (get_mvp_weights(cov_garch), test_garch),
            "MVP Historical":(get_mvp_weights(cov_hist),  test_hist),
            "ERC GARCH":     (get_erc_weights(cov_garch), test_garch),
            "ERC Historical":(get_erc_weights(cov_hist),  test_hist),
            "Naive":         (pd.Series(np.ones(len(train.columns)) / len(train.columns), index=train.columns), test_hist),
        }

        for name, (weights, test_slice) in portfolios.items():
            m = calculate_metrics(test_slice, weights)
            w = weights.values
            cov = (cov_garch if _MODEL_TYPE[name][1] == "GARCH" else cov_hist).values
            forecasted_std = float(np.sqrt(w @ cov @ w))

            results[name].extend(m["per_period_returns"].values.tolist())
            model, cov_type = _MODEL_TYPE[name]
            records.append({
                "Model": model,
                "Covariance Type": cov_type,
                "#Rolling Windows": start // PREDICTION_WINDOW,
                "Mean Return": m["mean_return"],
                "Forecasted Std": forecasted_std,
            })

    return results, pd.DataFrame(records)


def plot_results(results, output_dir, start_value=100):
    cumulative = {
        name: start_value * np.cumprod(1 + np.array(rets))
        for name, rets in results.items()
    }
    n = len(next(iter(cumulative.values())))
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, values in cumulative.items():
        ax.plot(np.arange(1, n + 1), values, label=name)
    ax.set_title("Portfolio Value Starting at 100")
    ax.set_xlabel("Period Number")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f"{output_dir}/Simulation.png")


def main():
    filepath, dayfirst = DATASETS[DATASET]
    returns = load_returns(filepath, dayfirst=dayfirst)
    results, metrics = run_rolling_backtest(returns, use_resampling_garch=USE_RESAMPLING_GARCH)

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    metrics.to_csv(f"{_OUTPUT_DIR}/backtest_metrics.csv", index=False)

    # Annualized summary computed from the full return series (valid for any PREDICTION_WINDOW)
    summary_rows = []
    for name, rets in results.items():
        if not rets:
            continue
        model, cov_type = _MODEL_TYPE[name]
        mask = (metrics["Model"] == model) & (metrics["Covariance Type"] == cov_type)
        avg_forecasted_ann = metrics.loc[mask, "Forecasted Std"].mean() * np.sqrt(252)
        row = calculate_summary_metrics(np.array(rets))
        ann_std = row["Ann. Std"]
        row["Covariance Type"] = cov_type
        row["Model"] = model
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

    if not any(results.values()):
        print("No valid rolling windows to plot.")
        return

    plot_results(results, _OUTPUT_DIR)


if __name__ == "__main__":
    main()
