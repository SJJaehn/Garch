"""
Rolling-window portfolio backtest.

We compare portfolio construction methods (MVP, HRP, ERC) using several
covariance estimators (historical, GARCH constant-correlation, DCC), plus a
naive 1/N benchmark.

Returns are LOG returns for the modelling part (GARCH/DCC like additive,
well-behaved returns). For evaluation and plotting we convert back to SIMPLE
returns, because a portfolio return is a weighted sum of *simple* asset returns,
not of log returns.

The engine is data-in / files-out: ``run_backtest`` takes the config and the
already-loaded data, so nothing is read at import and there is no global state.
Each rolling window is independent and deterministic, so it is farmed out to a
process pool; workers receive the (read-only) returns once via an initializer.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from garch.backtest.metrics import calculate_summary_metrics
from garch.models import covariance as cov_mod
from garch.models import garch as garch_mod
from garch.models import portfolio as port_mod

# Read-only per-worker state, populated once by _init_worker (avoids pickling the
# returns frame on every task and works under both 'spawn' and 'fork').
_WORKER: dict = {}


def _init_worker(log_returns, config):
    _WORKER["log_returns"] = log_returns
    _WORKER["config"] = config


def process_window(start):
    """Process-pool entry point: run one window using the worker's shared state."""
    return run_window(start, _WORKER["log_returns"], _WORKER["config"])


def _qlike(cov, realized):
    """
    Multivariate QLIKE loss of a forecast covariance ``cov`` (H) against the
    realized returns over the test window (Patton 2011; Laurent et al. 2012):

        QLIKE = log|H| + tr(H^{-1} S),   S = (1/pw) * sum_t r_t r_t'

    S is the realized (uncentered) second-moment proxy of the test-window log
    returns. The loss is minimised, in expectation, when H equals the true
    covariance, so a lower QLIKE means a better covariance forecast. Returns NaN
    if H is not positive definite / singular.
    """
    H = np.asarray(cov, dtype=float)
    R = np.asarray(realized, dtype=float)          # (pw, n) test-window log returns
    if H.shape[0] == 0 or R.shape[0] == 0:
        return np.nan
    S = R.T @ R / R.shape[0]                        # (n, n) average second moment
    sign, logdet = np.linalg.slogdet(H)
    if sign <= 0:
        return np.nan
    try:
        return float(logdet + np.trace(np.linalg.solve(H, S)))
    except np.linalg.LinAlgError:
        return np.nan


def _risk_contribution_rmse(weights, realized):
    """
    RMSE of the realized risk contributions from the equal-risk target (1/N).

    Using the test-window realized second moment  S = (1/pw) * sum_t r_t r_t'  as
    the covariance proxy, asset i's relative risk contribution is
        RC_i = w_i (S w)_i / (w' S w),   with  sum_i RC_i = 1.
    An Equal-Risk-Contribution portfolio targets RC_i = 1/N, so a lower RMSE means
    the weights kept risk closer to balanced *out-of-sample* -> a better covariance
    forecast. Returns NaN if the realized portfolio variance is ~0.
    """
    w = np.asarray(weights, dtype=float)
    n = w.shape[0]
    R = np.asarray(realized, dtype=float)
    if n == 0 or R.shape[0] == 0:
        return np.nan
    S = R.T @ R / R.shape[0]
    port_var = float(w @ S @ w)
    if port_var <= 1e-300:
        return np.nan
    rc = w * (S @ w) / port_var
    return float(np.sqrt(np.mean((rc - 1.0 / n) ** 2)))


def run_window(start, log_returns, config):
    """
    Run one rolling window (pure: data + config in, results out).

    Returns (per_period, records, dates, weights_info, formation_date, qlike_info).
    """
    tw, pw = config.train_window, config.prediction_window
    train = log_returns.iloc[start : start + tw]
    test  = log_returns.iloc[start + tw : start + tw + pw]

    # Universe is decided from TRAINING information only (no look-ahead into the
    # test window): assets with >=90% observations in the window and a valid
    # observation on the last training day (i.e. tradeable at formation time).
    train = train.loc[:, train.notna().mean() >= 0.9]
    train = train.loc[:, train.iloc[-1].notna()]
    if train.shape[1] == 0 or test.empty:
        return {}, [], [], {}, None, {}

    # same universe in the test window; a missing test return means the asset did
    # not trade that day, so its return is 0 (we don't drop it after the fact).
    test = test[train.columns]
    test_simple = (np.exp(test) - 1).fillna(0.0)

    # build the requested covariance matrices; fit GARCH only once and reuse it
    cov_by_method = {}
    if "Historical" in config.cov_methods:
        cov_by_method["Historical"] = cov_mod.historical_covariance(train)
    if "GARCH" in config.cov_methods or "DCC" in config.cov_methods:
        variances, std_resid = garch_mod.fit_garch_univariate(
            train, config.prediction_window, config.garch_p, config.garch_q)
        if "GARCH" in config.cov_methods:
            cov_by_method["GARCH"] = cov_mod.constant_correlation_covariance(train, variances)
        if "DCC" in config.cov_methods:
            cov_by_method["DCC"] = cov_mod.dcc_covariance(variances, std_resid)

    # per-step covariance-forecast quality: QLIKE of each estimator's matrix
    # against the realized test-window returns (lower is better).
    qlike_info = {}
    for method, cov in cov_by_method.items():
        if cov is None or cov.shape[1] == 0:
            continue
        realized = test[list(cov.columns)].fillna(0.0).values
        qlike_info[method] = _qlike(cov.values, realized)

    # historical cov is also used for the naive portfolio's forecast std
    cov_hist_full = cov_by_method.get("Historical")
    if cov_hist_full is None:
        cov_hist_full = cov_mod.historical_covariance(train)

    # list of (model, cov_type, weights, cov) to evaluate
    naive_w = port_mod.naive_weights(train.columns)
    jobs = [("Naive", "N/A", naive_w, cov_hist_full)]
    for method, cov in cov_by_method.items():
        if cov is None or cov.shape[1] == 0:
            continue
        for model in config.models:
            jobs.append((model, method, port_mod.get_weights(model, cov), cov))

    per_period = {}
    records = []
    weights_info = {}                       # name -> (target weights, drifted end weights)
    formation_date = test.index[0]          # rebalance date for this window
    for model, cov_type, weights, cov in jobs:
        name = "Naive" if model == "Naive" else f"{model} {cov_type}"
        cols = list(weights.index)
        port_returns = test_simple[cols] @ weights
        per_period[name] = port_returns.values.tolist()

        # drifted end-of-window weights (for turnover): each asset's start weight
        # grows with its gross return over the holding window, then renormalise.
        gross = (1.0 + test_simple[cols]).prod(axis=0)
        end_val = weights * gross
        end_weights = end_val / end_val.sum()
        weights_info[name] = (weights, end_weights)

        w = weights.values
        forecasted_std = float(np.sqrt(w @ cov.loc[cols, cols].values @ w))
        # ERC quality: realized risk-contribution RMSE from the 1/N target
        # (out-of-sample; only the ERC objective targets equal risk contributions)
        rc_rmse = np.nan
        if model == "ERC":
            rc_rmse = _risk_contribution_rmse(w, test[cols].fillna(0.0).values)
        records.append({
            "Model": model,
            "Covariance Type": cov_type,
            "Window Index": start // config.prediction_window,
            "Mean Return": port_returns.mean(),
            "Forecasted Std": forecasted_std,
            "RC RMSE": rc_rmse,
        })
    return per_period, records, list(test.index), weights_info, formation_date, qlike_info


def run_backtest(config, log_returns, rf, verbose=True):
    """
    Run the full rolling-window backtest for ``config`` over ``log_returns`` and
    write the results into ``config.output_dir``. ``rf`` is the daily simple
    risk-free return (aligned to the price dates). Returns the summary DataFrame.
    """
    starts = list(range(0, len(log_returns) - config.train_window - config.prediction_window + 1,
                        config.prediction_window))
    total = len(starts)

    results = {}          # name -> list of simple period returns
    records = []
    period_dates = []
    weights_hist = {}     # name -> list of (formation_date, target weights) in window order
    end_hist = {}         # name -> list of drifted end-of-window weights (same order)
    qlike_hist = []       # list of (formation_date, {cov_type: qlike}) in window order
    with ProcessPoolExecutor(max_workers=config.max_workers,
                             initializer=_init_worker,
                             initargs=(log_returns, config)) as executor:
        for i, (per_period, recs, dates, winfo, fdate, qinfo) in enumerate(
                executor.map(process_window, starts), start=1):
            records.extend(recs)
            period_dates.extend(dates)
            for name, rets in per_period.items():
                results.setdefault(name, []).extend(rets)
            for name, (target_w, end_w) in winfo.items():
                weights_hist.setdefault(name, []).append((fdate, target_w))
                end_hist.setdefault(name, []).append(end_w)
            if qinfo:
                qlike_hist.append((fdate, qinfo))
            if verbose:
                print(f"\r{i}/{total} windows completed", end="", flush=True)
    if verbose:
        print()

    if not records:
        print("No valid rolling windows (train window longer than the data?).")
        return None

    out_dir = config.output_dir
    os.makedirs(out_dir, exist_ok=True)
    metrics = pd.DataFrame(records)
    metrics.to_csv(f"{out_dir}/backtest_metrics.csv", index=False)

    # per-period returns per model (rows = dates), so any subset can be replotted later
    if period_dates:
        returns_df = pd.DataFrame(results, index=pd.to_datetime(period_dates))
        returns_df.index.name = "Date"
        returns_df.to_csv(f"{out_dir}/returns.csv")

    avg_turnover = _average_turnover(weights_hist, end_hist)

    if config.log_weights:
        _write_weights(weights_hist, f"{out_dir}/weights.csv")

    # per-step QLIKE per covariance estimator (rows = rebalance dates) + the average
    avg_qlike = {}
    if qlike_hist:
        qlike_df = pd.DataFrame([{"Date": fdate, **qinfo} for fdate, qinfo in qlike_hist])
        qlike_df = qlike_df.set_index("Date").sort_index()
        qlike_df.to_csv(f"{out_dir}/qlike.csv")
        avg_qlike = qlike_df.mean().to_dict()

    summary = _build_summary(results, metrics, period_dates, rf, avg_turnover, avg_qlike)
    summary.to_csv(f"{out_dir}/summary.csv", index=False)
    if verbose:
        print("Annualized performance summary:")
        print(summary.to_string(index=False))

    if any(results.values()):
        _plot_portfolio_value(results, period_dates, f"{out_dir}/Simulation.png")

    return summary


# =============================================================================
# Result assembly helpers
# =============================================================================

def _average_turnover(weights_hist, end_hist):
    """
    Average turnover per strategy: at each rebalance, how much weight is traded to
    go from the previous window's drifted weights to the new target weights (sum of
    absolute weight changes = buys + sells, over the union of both universes).
    """
    avg_turnover = {}
    for name, hist in weights_hist.items():
        ends = end_hist[name]
        per_window = []
        for k in range(1, len(hist)):
            prev_end, target = ends[k - 1], hist[k][1]
            idx = prev_end.index.union(target.index)
            per_window.append(float(np.abs(target.reindex(idx).fillna(0.0)
                                            - prev_end.reindex(idx).fillna(0.0)).sum()))
        avg_turnover[name] = float(np.mean(per_window)) if per_window else np.nan
    return avg_turnover


def _write_weights(weights_hist, path):
    """Per-window target weights in long format (Date, Model, Covariance, Asset, Weight)."""
    weight_rows = []
    for name, hist in weights_hist.items():
        model, cov_type = ("Naive", "N/A") if name == "Naive" else name.rsplit(" ", 1)
        for fdate, target_w in hist:
            for asset, wt in target_w.items():
                weight_rows.append({"Date": fdate, "Model": model,
                                    "Covariance Type": cov_type,
                                    "Asset": asset, "Weight": wt})
    pd.DataFrame(weight_rows).to_csv(path, index=False)


def _build_summary(results, metrics, period_dates, rf, avg_turnover, avg_qlike):
    # average forecast (annualised) std per model/cov type
    avg_fcst = metrics.groupby(["Model", "Covariance Type"])["Forecasted Std"].mean() * np.sqrt(252)
    # average ERC risk-contribution RMSE per model/cov type (NaN for non-ERC)
    avg_rc = metrics.groupby(["Model", "Covariance Type"])["RC RMSE"].mean()
    # daily risk-free return aligned to the evaluated dates (same order as results)
    rf_aligned = rf.reindex(pd.to_datetime(period_dates)).fillna(0.0).values

    summary_rows = []
    for name, rets in results.items():
        rets = np.array(rets)
        if rets.size == 0:
            continue
        model, cov_type = ("Naive", "N/A") if name == "Naive" else name.rsplit(" ", 1)
        row = calculate_summary_metrics(rets, rf_aligned)
        fcst = avg_fcst.get((model, cov_type), np.nan)
        row["Model"] = model
        row["Covariance Type"] = cov_type
        row["Ann. Std (fcst)"] = fcst
        row["Real / Fcst Std"] = row["Ann. Std"] / fcst if fcst and fcst > 0 else np.nan
        # QLIKE is a property of the covariance matrix; Naive uses the historical one
        row["Avg QLIKE"] = avg_qlike.get("Historical" if cov_type == "N/A" else cov_type, np.nan)
        # ERC risk-contribution RMSE (NaN for non-ERC models)
        row["ERC RC RMSE"] = avg_rc.get((model, cov_type), np.nan)
        row["Avg Turnover"] = avg_turnover.get(name, np.nan)
        summary_rows.append(row)

    col_order = ["Model", "Covariance Type", "Ann. Return", "Ann. Std", "Ann. Std (fcst)",
                 "Real / Fcst Std", "Avg QLIKE", "ERC RC RMSE", "Avg Turnover", "Ann. Sharpe",
                 "Ann. Sortino", "Max Drawdown", "Calmar Ratio", "CVaR (95%)", "Skewness",
                 "Excess Kurtosis"]
    return pd.DataFrame(summary_rows).sort_values(["Model", "Covariance Type"])[col_order]


def _plot_portfolio_value(results, period_dates, path):
    import matplotlib
    matplotlib.use("Agg")  # headless-safe; the plot is built in the main process only
    import matplotlib.pyplot as plt

    x_axis = pd.to_datetime(period_dates)
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, rets in results.items():
        ax.plot(x_axis, 100 * np.cumprod(1 + np.array(rets)), label=name)
    ax.set_title("Portfolio Value Starting at 100")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
