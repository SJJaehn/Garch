import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import (
    estimate_cov_matrix_garch,
    estimate_cov_matrix_historical,
    get_mvp_weights,
    get_erc_weights,
    calculate_metrics,
    get_hrp_weights
)

TRAIN_WINDOW = 252  # 1 year of trading days
PREDICTION_WINDOW = 21  # ~1 month of trading days


def load_returns(filepath):
    prices = pd.read_csv(filepath, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices[prices.index.notna()]
    return prices.pct_change()


def clean_window(train, test):
    train = train.dropna(axis=1)
    test = test[train.columns]
    train = train.dropna()
    test = test.dropna()
    return train, test


def compute_portfolio_returns(train_returns, test_returns):
    cov_garch = estimate_cov_matrix_garch(train_returns, prediction_window=PREDICTION_WINDOW)
    cov_hist = estimate_cov_matrix_historical(train_returns)

    portfolios = {
        "HRP GARCH":      get_hrp_weights(cov_garch),
        "HRP Historical": get_hrp_weights(cov_hist),
        "MVP GARCH":      get_mvp_weights(cov_garch),
        "MVP Historical": get_mvp_weights(cov_hist),
        "ERC GARCH":      get_erc_weights(cov_garch),
        "ERC Historical": get_erc_weights(cov_hist),
        "Naive":          np.ones(len(train_returns.columns)) / len(train_returns.columns)
    }

    return {
        name: calculate_metrics(test_returns, weights)
        for name, weights in portfolios.items()
    }


def run_rolling_backtest(returns):
    results = {"HRP GARCH": [], "HRP Historical": [], "MVP GARCH": [], "MVP Historical": [], "ERC GARCH": [], "ERC Historical": [], "Naive": []}
    metrics_df = pd.DataFrame(columns=["Model", "Covariance Type", "#Rolling Windows", "Sharpe Ratio", "Mean Return", "Std Dev"])

    for start in range(0, len(returns) - TRAIN_WINDOW - PREDICTION_WINDOW, PREDICTION_WINDOW):
        train = returns.iloc[start : start + TRAIN_WINDOW]
        test = returns.iloc[start + TRAIN_WINDOW : start + TRAIN_WINDOW + PREDICTION_WINDOW]

        train, test = clean_window(train, test)

        if train.empty or test.empty or train.shape[1] == 0:
            continue

        window_metrics = compute_portfolio_returns(train, test)
        for name, metrics in window_metrics.items():
            # Keep each period return from the prediction window instead of collapsing to one mean return.
            results[name].extend(metrics["per_period_returns"].tolist())
            metrics_df.loc[len(metrics_df)] = {
                "Model": name.split()[0],  # "MVP", "ERC", or "Naive"
                "Covariance Type": name.split()[1] if len(name.split()) > 1 else "N/A",  # "GARCH", "Historical", or "N/A" for Naive
                "#Rolling Windows": start // PREDICTION_WINDOW,  # Number of complete prediction windows processed for this model
                "Sharpe Ratio": metrics["sharpe_ratio"],
                "Mean Return": metrics["mean_return"],
                "Std Dev": metrics["std_dev"],
            }


    return results, metrics_df


def to_cumulative(returns_dict, start_value=100):
    return {
        name: start_value * np.cumprod(1 + np.array(rets))
        for name, rets in returns_dict.items()

    }


def plot_results(cumulative):
    n = len(next(iter(cumulative.values())))
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, values in cumulative.items():
        ax.plot(x, values, label=name)

    ax.set_title("Portfolio Value Starting at 100")
    ax.set_xlabel("Period Number")
    ax.set_ylabel("Portfolio Value")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()


def main():
    returns = load_returns("SP500_Daily.csv")
    # returns = returns.sample(n=min(100, returns.shape[1]), axis=1, random_state=42)
    results, metrics = run_rolling_backtest(returns)

    metrics.to_csv("backtest_metrics.csv", index=False)

    if not any(results.values()):
        print("No valid rolling windows to plot.")
        return

    cumulative = to_cumulative(results)
    plot_results(cumulative)


if __name__ == "__main__":
    main()