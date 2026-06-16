"""
Analyse the backtest results in Ergebnisse/.

  Part 1 - per-combo time-series charts (rolling/cumulative Sharpe, portfolio value)
  Part 2 - aggregate every summary.csv into one Excel table + overview charts

    python analyze.py
"""
from garch.analysis.aggregate import summary_outputs
from garch.analysis.charts import per_combo_charts


if __name__ == "__main__":
    per_combo_charts()
    summary_outputs()
