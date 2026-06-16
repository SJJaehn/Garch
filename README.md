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

```
garch/                      importable package (all the logic)
  config.py                 central configuration (BacktestConfig, dataset registry, paths)
  data/loaders.py           load prices / log returns / risk-free series
  data/synthetic.py         generate the artificial datasets (GJR-GARCH + Student-t)
  data/cleaning.py          clean the raw TRBC Datastream export
  models/garch.py           univariate GARCH fitting (arch)
  models/covariance.py      historical / constant-correlation / DCC covariance
  models/portfolio.py       MVP & ERC (riskfolio-lib), HRP (custom), naive
  backtest/engine.py        the rolling-window backtest
  backtest/metrics.py       annualised performance statistics
  analysis/charts.py        per-combo time-series charts
  analysis/aggregate.py     aggregate table + overview charts

main.py                     run one backtest          (edit BacktestConfig)
run_all.py                  run a batch of backtests   (edit COMBOS)
artificial_data.py          (re)generate synthetic data
analyze.py                  charts + aggregate summary
clean_data.py               clean the raw TRBC export
```

## Setup

```bash
pip install -r requirements.txt
```

> **cvxpy note:** riskfolio-lib (used for MVP/ERC) solves through cvxpy. cvxpy's
> default C++ canonicalisation backend can abort with numpy 2.x, so every solve
> is routed through the SciPy backend (`garch/config.py` → `SOL_PARAMS`). No
> action needed; this just works.

## Usage

Run everything from the repository root:

```bash
python artificial_data.py   # 1. (re)generate DATA/Artifical/{monte_carlo,garch,dcc}.csv
python main.py              # 2. run one backtest -> Ergebnisse/<dataset>/<train>_<pred>/
python run_all.py           # 2b. or run a batch of (dataset, train, pred) combos
python analyze.py           # 3. charts per combo + Ergebnisse/Zusammenfassung/aggregate_results.xlsx
```

### Outputs

Each backtest writes to `Ergebnisse/<dataset>/<train>_<pred>/`:
`returns.csv`, `backtest_metrics.csv`, `summary.csv`, `weights.csv` (optional),
and `Simulation.png`. `analyze.py` adds per-combo charts and an aggregate Excel
table + overview charts under `Ergebnisse/Zusammenfassung/`.

## Notes

- **Configuration** is centralised in `garch/config.py`. The backtest takes a
  single immutable `BacktestConfig`; nothing is loaded or mutated at import time.
- **Determinism:** the backtest contains no randomness, so results are
  reproducible. The only RNG is in `data/synthetic.py` (`SEED = 42`).
- **ERC** is solved by riskfolio-lib's risk-parity optimiser (true equal risk
  contribution). The previous fixed-point implementation could fail to converge,
  so ERC results differ from older runs; everything else matches.
- The synthetic data folder is `DATA/Artifical/` (a historical typo kept on
  purpose so existing generated data is not orphaned).
