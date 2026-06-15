"""Temporary batch runner: regenerates every Abbildungen/{dataset}/{train}_{pred}
combination with the current main.py logic.

Safe to delete. Uses the 'fork' start method so the worker processes inherit the
config/data we set on the `main` module (with 'spawn' they would re-import main.py
and fall back to its default constants).
"""
import multiprocessing as mp

import numpy as np
import pandas as pd

import main

# (TRAIN_WINDOW, PREDICTION_WINDOW) combinations per dataset.
# Day-ahead forecast (pred=1) with a 1008-day (~4y) train window on every
# artificial dataset.
COMBOS = {
    "MonteCarlo": [(1008, 1)],
    "GARCH_sim":  [(1008, 1)],
    "DCC_sim":    [(1008, 1)],
}


def load_dataset(dataset):
    """Reload main.prices/main.log_returns for `dataset` (main loads only once at import)."""
    main.DATASET = dataset
    filepath, date_format = main.DATASETS[dataset]
    prices = pd.read_csv(filepath, index_col=0)
    prices.index = pd.to_datetime(prices.index, format=date_format, errors="coerce")
    prices = prices[prices.index.notna()]
    prices = prices.loc[prices.notna().sum(axis=1) >= int(0.5 * prices.shape[1])]
    log_returns = np.log(prices / prices.shift(1)).iloc[1:]
    zero_frac = (log_returns == 0).sum() / log_returns.notna().sum()
    log_returns = log_returns.loc[:, zero_frac < 0.5]
    main.prices = prices
    main.log_returns = log_returns
    main.rf_daily = main.load_risk_free(prices.index)  # risk-free aligned to these dates


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    for dataset, combos in COMBOS.items():
        load_dataset(dataset)
        for train, pred in combos:
            main.TRAIN_WINDOW = train
            main.PREDICTION_WINDOW = pred
            main._OUTPUT_DIR = f"Ergebnisse/{dataset}/{train}_{pred}"
            print(f"\n=== {dataset} {train}_{pred} ===", flush=True)
            main.main()
