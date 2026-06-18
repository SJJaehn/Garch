"""
Batch runner: run the backtest over several (dataset, train, pred) combinations.

Unlike the old version, this mutates no global state — each run is an independent
BacktestConfig and the data is loaded once per dataset.

    python run_all.py
"""
from garch.backtest.engine import run_backtest
from garch.config import BacktestConfig
from garch.data.loaders import load_dataset

# Full grid: every training window (252-day steps, ~1..10 years) x prediction horizon.
GRID_TRAIN = [252 * i for i in range(1, 11)]   # 252, 504, ..., 2520
GRID_PRED  = [1, 5, 10, 21]

# (train_window, prediction_window) combinations per dataset.
COMBOS = {
    #"DCC_sim": [(1008, 1)],  # just one combo for the synthetic DCC data
    #"GARCH_sim": [(1008, 1)],  # just one combo for the synthetic GARCH data
    #"MonteCarlo": [(1008, 1)],  # just one combo for the synthetic Monte Carlo data
    "TRBC": [(train, pred) for train in GRID_TRAIN for pred in GRID_PRED],
}


def main():
    for dataset, combos in COMBOS.items():
        _, log_returns, rf = load_dataset(dataset)
        for train, pred in combos:
            config = BacktestConfig(dataset=dataset, train_window=train, prediction_window=pred, max_workers=6)
            print(f"\n=== {dataset} {train}_{pred} ===", flush=True)
            run_backtest(config, log_returns, rf)


if __name__ == "__main__":
    main()
