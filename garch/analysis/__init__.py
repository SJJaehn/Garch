"""
Analysis layer: charts and aggregate tables built from the backtest outputs.

Shared analysis settings live here so both ``charts`` and ``aggregate`` read the
same knobs (the evaluation frame, which models/covariances to draw, colours...).
"""

# --- per-combo time-series charts --------------------------------------------
WINDOW       = 126           # rolling window length (trading days)
SMOOTH       = 1             # extra moving-average smoothing of rolling Sharpe (1 = off)
MIN_PERIODS  = 252           # min observations before a cumulative Sharpe is shown
DATE_START   = "2010-01-01"  # fixed evaluation frame (None = full history)
DATE_END     = None
MODEL_FILTER = ["MVP", "HRP", "ERC", "Naive"]   # which models to draw
COV_FILTER   = ["Historical", "GARCH", "DCC"]   # which covariance types to draw
COLUMNS      = None          # explicit column override; bypasses the filters

# --- overview charts ---------------------------------------------------------
MODELS     = ["MVP", "HRP", "ERC"]
COVS       = ["Historical", "GARCH", "DCC"]
COV_COLORS = {"Historical": "tab:blue", "GARCH": "tab:orange", "DCC": "tab:green"}
SHARPE_COL = "Ann. Sharpe (frame)"   # the charts use the framed Sharpe (DATE_START/END)
