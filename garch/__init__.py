"""
GARCH/DCC rolling-window portfolio backtest.

A small package that compares covariance estimators (historical, GARCH with
constant correlation, DCC-GARCH) against portfolio construction methods
(minimum-variance, equal-risk-contribution, hierarchical risk parity, and a
naive 1/N benchmark) on real and synthetic price data.

Layout:
  config         central configuration (pure data, no I/O on import)
  data.loaders   load prices / log returns / risk-free series
  data.synthetic generate the artificial datasets
  data.cleaning  clean the raw TRBC Datastream export
  models.garch   univariate GARCH fitting (arch)
  models.covariance  historical / constant-correlation / DCC covariance
  models.portfolio   MVP & ERC (riskfolio-lib), HRP (custom), naive
  backtest.engine    the rolling-window backtest
  backtest.metrics   annualised performance statistics
  analysis.charts    per-combo time-series charts
  analysis.aggregate aggregate table + overview charts
"""
