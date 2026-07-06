# GARCH/DCC Portfolio Backtest

A rolling-window backtest that compares **covariance estimators** against
**portfolio construction methods** on real and synthetic price data.

- **Covariance estimators:** historical sample covariance, GARCH(1,1) with
  constant correlation, and DCC-GARCH (Engle 2002, dynamic correlation).
- **Portfolio methods:** Minimum Variance (MVP), Equal Risk Contribution (ERC),
  Hierarchical Risk Parity (HRP), plus a naive 1/N benchmark.
- **Datasets:** empirical (`TRBC`, `SP500`, `Dow`) and synthetic (`MonteCarlo`,
  `GARCH_sim`, `DCC_sim`) generated with known second-moment dynamics.

## Layout

Four flat scripts at the repo root — no packages to navigate. The pipeline runs
top-to-bottom: generate data → backtest → analyze.

```
main.py         settings (which datasets / window sizes to run) + shared
                configuration (paths, dataset registry, solver settings);
                `python main.py` runs the whole grid
datagen.py      RUN: generate the synthetic datasets -> DATA/Artifical/
backtest.py     the engine: data loaders + GARCH/DCC + covariance estimators
                + portfolio weights + metrics + the rolling-window loop
analyze.py      RUN: per-run charts + aggregate Excel table + overview charts
```

## Setup

```bash
pip install -r requirements.txt
```

> **cvxpy note:** riskfolio-lib (used for MVP/ERC) solves through cvxpy. cvxpy's
> default C++ canonicalisation backend can abort with numpy 2.x, so every solve
> is routed through the SciPy backend (`main.py` → `SOL_PARAMS`). No action
> needed; this just works.

## Usage

Run everything from the repository root:

```bash
python datagen.py    # 1. (re)generate DATA/Artifical/{monte_carlo,garch,dcc}.csv
python main.py       # 2. backtest every dataset x training window x horizon
                     #    combination from the settings block in main.py
                     #    -> Ergebnisse/<dataset>/<train>_<pred>/
python analyze.py    # 3. charts per run + Ergebnisse/Zusammenfassung/aggregate_results.xlsx
```

A single backtest is just a one-entry grid in `main.py`, e.g.
`RUN_DATASETS = ["DCC_sim"]`, `TRAIN_WINDOWS = [1008]`, `PRED_WINDOWS = [1]`.

### Outputs

Each backtest writes to `Ergebnisse/<dataset>/<train>_<pred>/`:
`returns.csv`, `backtest_metrics.csv`, `summary.csv`, `qlike.csv`,
`cov_rmse.csv`, `weights.csv` (optional), and `Simulation.png`. `analyze.py`
adds per-run charts and an aggregate Excel table + overview charts under
`Ergebnisse/Zusammenfassung/`.

## Notes

- **Configuration** lives in the settings block at the top of `main.py`
  (datasets, window sizes, covariance estimators, models, GARCH order).
- **Determinism:** the backtest contains no randomness, so results are
  reproducible. The only RNG is in `datagen.py` (`SEED = 1`).
- **Risk-free rate** is read from `DATA/Empirical/FED_FUNDS.csv` — the Fed Funds
  total-return index *level* (columns `Exchange Date`;`Close`, German number
  format, starts at 100). The loaders take its `pct_change` to get the daily
  *simple* risk-free return used for excess returns (Sharpe / Sortino). To update
  it, replace the file keeping the same name and format (`main.RISK_FREE_FILE`).
- **ERC** is solved by riskfolio-lib's risk-parity optimiser (true equal risk
  contribution). The previous fixed-point implementation could fail to converge,
  so ERC results differ from older runs; everything else matches.
- The synthetic data folder is `DATA/Artifical/` (a historical typo kept on
  purpose so existing generated data is not orphaned).
