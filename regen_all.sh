#!/bin/bash
# Regenerates every result used in the paper after the holiday-date fix in
# backtest.py (to_log_returns now drops dates where no asset moves at all).
# Run from the repo root: ./regen_all.sh
set -euo pipefail
cd "$(dirname "$0")"

# 1. Re-simulate the three synthetic datasets (Monte Carlo, GARCH, DCC).
python3 datagen.py

# 2. Re-run the full empirical + synthetic grid: 10 training windows x
#    4 prediction horizons, for TRBC and the 3 synthetic datasets.
python3 - <<'PYEOF'
import main as config
import backtest

config.RUN_DATASETS  = ["TRBC", "MonteCarlo", "GARCH_sim", "DCC_sim"]
config.TRAIN_WINDOWS = [252, 504, 756, 1008, 1260, 1512, 1764, 2016, 2268, 2520]
config.PRED_WINDOWS  = [1, 5, 10, 21]

for dataset in config.RUN_DATASETS:
    prices, log_returns, rf = backtest.load_dataset(dataset)
    for train_window in config.TRAIN_WINDOWS:
        for pred_window in config.PRED_WINDOWS:
            print(f"\n=== {dataset} {train_window}_{pred_window} ===", flush=True)
            backtest.run_backtest(dataset, log_returns, rf, train_window, pred_window)
PYEOF

# 3. Rebuild every LaTeX table in paper/Abbildungen/ from the fresh CSVs.
python3 paper/tabellen.py

echo "Done. Recompile paper.tex (pdflatex -> biber -> pdflatex x2) and re-check"
echo "the numbers flagged in paper/suggestions.md before trusting the new PDF."
