"""Temporary batch runner: reruns every existing Abbildungen/TRBC combination.

Safe to delete. Uses the 'fork' start method so the worker processes inherit the
config we set on the `main` module (with 'spawn' they would re-import main.py and
fall back to its default constants).
"""
import multiprocessing as mp

import main

# (TRAIN_WINDOW, PREDICTION_WINDOW) combinations found under Abbildungen/TRBC/
COMBOS = [
    (252, 1), (252, 5), (252, 10), (252, 15), (252, 21),
    (504, 1), (504, 10), (504, 21),
]

if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    for train, pred in COMBOS:
        for resampling in (True, False):
            main.TRAIN_WINDOW = train
            main.PREDICTION_WINDOW = pred
            main.USE_RESAMPLING_GARCH = resampling
            main._GARCH_MODE = "resampled" if resampling else "spot"
            main._OUTPUT_DIR = f"Abbildungen/{main.DATASET}/{train}_{pred}_{main._GARCH_MODE}"
            print(f"\n=== {train}_{pred}_{main._GARCH_MODE} ===", flush=True)
            main.main()
