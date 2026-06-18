# GARCH/DCC Portfolio Backtest

A rolling-window backtest that compares **covariance estimators** against
**portfolio construction methods** on real and synthetic price data.

- **Covariance estimators:** historical sample covariance, GARCH(1,1) with
  constant correlation, and DCC-GARCH (Engle 2002, dynamic correlation).
- **Portfolio methods:** Minimum Variance (MVP), Equal Risk Contribution (ERC),
  Hierarchical Risk Parity (HRP), plus a naive 1/N benchmark.
- **Datasets:** empirical (`TRBC`, `SP500`) and synthetic (`MonteCarlo`,
  `GARCH_sim`, `DCC_sim`) generated with known second-moment dynamics.

## Layout

Four flat scripts at the repo root — no packages to navigate. The pipeline runs
top-to-bottom: generate data → backtest → analyze.

```
config.py       params only: paths, dataset registry, solver settings,
                BacktestConfig, and the COMBOS batch grid (no I/O at import)
datagen.py      RUN: generate the synthetic datasets -> DATA/Artifical/
backtest.py     RUN: data loaders + GARCH + covariance + portfolio + metrics
                + the rolling-window engine + the COMBOS grid loop
analyze.py      RUN: per-combo charts + aggregate Excel table + overview charts
```

## Setup

```bash
pip install -r requirements.txt
```

> **cvxpy note:** riskfolio-lib (used for MVP/ERC) solves through cvxpy. cvxpy's
> default C++ canonicalisation backend can abort with numpy 2.x, so every solve
> is routed through the SciPy backend (`config.py` → `SOL_PARAMS`). No action
> needed; this just works.

## Usage

Run everything from the repository root:

```bash
python datagen.py    # 1. (re)generate DATA/Artifical/{monte_carlo,garch,dcc}.csv
python backtest.py   # 2. run every (dataset, train, pred) in config.COMBOS
                     #    -> Ergebnisse/<dataset>/<train>_<pred>/
python analyze.py    # 3. charts per combo + Ergebnisse/Zusammenfassung/aggregate_results.xlsx
```

A single backtest is just a one-entry `COMBOS` in `config.py`
(e.g. `{"DCC_sim": [(1008, 1)]}`).

### Outputs

Each backtest writes to `Ergebnisse/<dataset>/<train>_<pred>/`:
`returns.csv`, `backtest_metrics.csv`, `summary.csv`, `weights.csv` (optional),
and `Simulation.png`. `analyze.py` adds per-combo charts and an aggregate Excel
table + overview charts under `Ergebnisse/Zusammenfassung/`.

## Notes

- **Configuration** is centralised in `config.py`. The backtest takes a single
  immutable `BacktestConfig`; nothing is loaded or mutated at import time.
- **Determinism:** the backtest contains no randomness, so results are
  reproducible. The only RNG is in `datagen.py` (`SEED = 42`).
- **Risk-free rate** is read from `DATA/Empirical/FED_FUNDS.csv` — the Fed Funds
  total-return index *level* (columns `Exchange Date`;`Close`, German number
  format, starts at 100). The loaders take its `pct_change` to get the daily
  *simple* risk-free return used for excess returns (Sharpe / Sortino). To update
  it, replace the file keeping the same name and format (`config.RISK_FREE_FILE`).
- **ERC** is solved by riskfolio-lib's risk-parity optimiser (true equal risk
  contribution). The previous fixed-point implementation could fail to converge,
  so ERC results differ from older runs; everything else matches.
- The synthetic data folder is `DATA/Artifical/` (a historical typo kept on
  purpose so existing generated data is not orphaned).
