"""
Run a single rolling-window portfolio backtest.

Edit the BacktestConfig below (or its defaults in garch/config.py) to choose the
dataset, training/prediction windows, GARCH order, covariance estimators and
portfolio models. Results are written to config.output_dir.

    python main.py
"""
from garch.backtest.engine import run_backtest
from garch.config import BacktestConfig
from garch.data.loaders import load_dataset


def main():
    config = BacktestConfig(dataset="DCC_sim", train_window=1008, prediction_window=1, max_workers=6)
    _, log_returns, rf = load_dataset(config.dataset)
    run_backtest(config, log_returns, rf)


if __name__ == "__main__":
    main()
