import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import (
    calculate_metrics,
    estimate_cov_matrix_garch,
    estimate_cov_matrix_historical,
    get_erc_weights,
    get_hrp_weights,
    get_mvp_weights,
)

# --- Configuration -----------------------------------------------------------

DATASETS: dict[str, str] = {
    "TRBC":  "TRBC_Daily.csv",
    "SP500": "SP500_Daily.csv",   # placeholder path
}

DATASET              = "TRBC"
TRAIN_WINDOW         = 2 * 252   # 2 years of trading days
PREDICTION_WINDOW    = 1
USE_RESAMPLING_GARCH = False      # True → GARCH forecast over horizon; False → last conditional vol

# -----------------------------------------------------------------------------

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

# Which in-sample covariance each portfolio was built from
_COV_SOURCE: dict[str, str] = {
    "HRP GARCH":     "garch",
    "HRP Historical":"hist",
    "MVP GARCH":     "garch",
    "MVP Historical":"hist",
    "ERC GARCH":     "garch",
    "ERC Historical":"hist",
    "Naive":         "hist",
}


def load_returns(filepath, max_zero_frac=0.5):
    prices = pd.read_csv(filepath, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices[prices.index.notna()]
    returns = prices.pct_change(fill_method=None)
    # Drop columns where most non-NaN returns are zero (stale / frozen price series)
    zero_frac = (returns == 0).sum() / returns.notna().sum()
    return returns.loc[:, zero_frac < max_zero_frac]


def run_rolling_backtest(returns, use_resampling_garch=False):
    results: dict[str, list] = {n: [] for n in _MODEL_TYPE}
    records = []

    for start in range(0, len(returns) - TRAIN_WINDOW - PREDICTION_WINDOW, PREDICTION_WINDOW):
        train = returns.iloc[start : start + TRAIN_WINDOW]
        test  = returns.iloc[start + TRAIN_WINDOW : start + TRAIN_WINDOW + PREDICTION_WINDOW]

        train = train.dropna(how="all")
        test  = test.dropna(how="all")
        train = train.dropna(axis=1, thresh=int(0.95 * len(train)))
        test  = test[train.columns].dropna(axis=1, thresh=int(0.95 * len(test)))
        train = train[test.columns].dropna()
        test  = test.dropna()

        if train.empty or test.empty or train.shape[1] == 0:
            continue

        garch_horizon = PREDICTION_WINDOW if use_resampling_garch else 0
        cov_garch = estimate_cov_matrix_garch(train, prediction_window=garch_horizon)
        cov_hist  = estimate_cov_matrix_historical(train)
        cov_map   = {"garch": cov_garch, "hist": cov_hist}

        portfolios = {
            "HRP GARCH":     get_hrp_weights(cov_garch),
            "HRP Historical":get_hrp_weights(cov_hist),
            "MVP GARCH":     get_mvp_weights(cov_garch),
            "MVP Historical":get_mvp_weights(cov_hist),
            "ERC GARCH":     get_erc_weights(cov_garch),
            "ERC Historical":get_erc_weights(cov_hist),
            "Naive":         pd.Series(np.ones(len(train.columns)) / len(train.columns), index=train.columns),
        }

        for name, weights in portfolios.items():
            m = calculate_metrics(test, weights)
            w = weights.values
            cov = cov_map[_COV_SOURCE[name]].values
            forecasted_std = float(np.sqrt(w @ cov @ w))

            results[name].extend(m["per_period_returns"].tolist())
            model, cov_type = _MODEL_TYPE[name]
            records.append({
                "Model": model,
                "Covariance Type": cov_type,
                "#Rolling Windows": start // PREDICTION_WINDOW,
                "Sharpe Ratio": m["sharpe_ratio"],
                "Mean Return": m["mean_return"],
                "Realized Std": m["std_dev"],
                "Forecasted Std": forecasted_std,
                # > 1: model underestimated risk; < 1: model overestimated risk
                "Realized / Forecasted Std": m["std_dev"] / forecasted_std if forecasted_std > 0 else np.nan,
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
    returns = load_returns(DATASETS[DATASET])
    results, metrics = run_rolling_backtest(returns, use_resampling_garch=USE_RESAMPLING_GARCH)

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    metrics.to_csv(f"{_OUTPUT_DIR}/backtest_metrics.csv", index=False)

    groupby_cols = ["Model", "Covariance Type"]
    print("Average Sharpe ratios by model and covariance type:")
    print(
        metrics.groupby(groupby_cols, dropna=False)["Sharpe Ratio"]
        .mean().reset_index().sort_values(groupby_cols).to_string(index=False)
    )

    vol_cols = ["Realized Std", "Forecasted Std", "Realized / Forecasted Std"]
    print("\nForecasted vs realized volatility (ratio > 1 = underestimated risk, < 1 = overestimated):")
    print(
        metrics.groupby(groupby_cols, dropna=False)[vol_cols]
        .mean().reset_index().sort_values(groupby_cols).to_string(index=False)
    )

    if not any(results.values()):
        print("No valid rolling windows to plot.")
        return

    plot_results(results, _OUTPUT_DIR)


if __name__ == "__main__":
    main()
