"""Temporary batch runner: regenerates every Abbildungen/{dataset}/{train}_{pred}
combination with the current main.py logic.

Safe to delete. Uses the 'fork' start method so the worker processes inherit the
config/data we set on the `main` module (with 'spawn' they would re-import main.py
and fall back to its default constants).
"""
import multiprocessing as mp

import pandas as pd

import main

# (TRAIN_WINDOW, PREDICTION_WINDOW) combinations per dataset.
COMBOS = {
    "TRBC": [
        (252, 1), (252, 5), (252, 10), (252, 15), (252, 21),
        (504, 1), (504, 10), (504, 21),
        (756, 1), (1008, 1), (756, 21), (1008, 21), (756, 10), (1008, 10),
    ],
    "SP500": [
        (252, 1), (252, 21),
        (504, 1), (504, 10), (504, 21),
    ],
}


def load_dataset(dataset):
    """Reload main.prices/main.returns for `dataset` (main loads only once at import)."""
    main.DATASET = dataset
    filepath, date_format = main.DATASETS[dataset]
    prices = pd.read_csv(filepath, index_col=0)
    prices.index = pd.to_datetime(prices.index, format=date_format, errors="coerce")
    prices = prices[prices.index.notna()]
    prices = prices.loc[prices.notna().sum(axis=1) >= int(0.5 * prices.shape[1])]
    returns = prices.pct_change(fill_method=None).iloc[1:]
    zero_frac = (returns == 0).sum() / returns.notna().sum()
    returns = returns.loc[:, zero_frac < 0.5]
    main.prices = prices
    main.returns = returns


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    for dataset, combos in COMBOS.items():
        load_dataset(dataset)
        for train, pred in combos:
            main.TRAIN_WINDOW = train
            main.PREDICTION_WINDOW = pred
            main._OUTPUT_DIR = f"Abbildungen/{dataset}/{train}_{pred}"
            print(f"\n=== {dataset} {train}_{pred} ===", flush=True)
            main.main()
