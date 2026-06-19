# =============================================================================
# validate_with_R.R  --  R re-implementation of backtest.py for cross-validation
# =============================================================================
#
# Purpose
# -------
# This script reproduces the *entire* Python rolling-window backtest pipeline
# (backtest.py + config.py) so its numbers can be checked against the Python
# results. The DATA FILTERING, PREPROCESSING, ROLLING-WINDOW LOGIC and the
# PERFORMANCE METRICS are mimicked exactly (same formulas, same conventions),
# but the parts that are worth validating with an independent implementation are
# delegated to established R packages instead of being hand-coded:
#
#     Python (backtest.py)                  R (this file)
#     ------------------------------------  -----------------------------------
#     arch.arch_model         (GARCH)       rugarch::ugarchfit/forecast
#     custom DCC recursion     (DCC)        rmgarch::dccfit/dccforecast
#     riskfolio MinRisk        (MVP)        quadprog::solve.QP
#     riskfolio rp_optimization(ERC)        riskParityPortfolio::riskParityPortfolio
#     self-coded HRP           (HRP)        de Prado HRP via hclust (single linkage)
#
# HRP NOTE: HierPortfolios::HRP_Portfolio was dropped because it clusters on the
# *Euclidean distance between rows* of the correlation-distance matrix D (de Prado's
# original "double distance"), which gives a different leaf order and thus different
# weights from Python/riskfolio. Python (and riskfolio's default) cluster on D
# directly; we replicate that here so the HRP weights match exactly.
#
# Everything else (sample/EWMA covariance, the 1/N benchmark, QLIKE, Cov-RMSE,
# ERC risk-contribution RMSE, all annualised summary statistics incl. pandas'
# exact skew/kurtosis definitions, turnover) is computed here with the *same*
# arithmetic as Python, so any discrepancy isolates the GARCH/DCC/optimiser step.
#
# NOTE: the risk-free series is dropped for now (rf = 0), so Sharpe/Sortino use
# raw returns as excess returns and "Ann. Sharpe" == "Ann. Sharpe (rf=0)".
#
# Output: for each (dataset, train, pred) combo it writes, NEXT TO the Python
# outputs in Ergebnisse/<dataset>/<train>_<pred>/, the files
#     summary_R.csv  returns_R.csv  backtest_metrics_R.csv
#     qlike_R.csv    cov_rmse_R.csv  weights_R.csv
# so you can diff e.g. summary.csv vs summary_R.csv directly.
#
#   Rscript validate_with_R.R
#
# !! COST WARNING: by default COMBOS mirrors config.py = SP500 with ~480 assets.
#    A 480-asset DCC-GARCH fit (rmgarch) per window over ~400 windows takes many
#    hours. To validate the *methodology* cheaply, point COMBOS at "Dow" (30) or
#    "TRBC" (28 sectors) -- see the COMBOS block below.
# =============================================================================

suppressWarnings(suppressMessages({
  library(rugarch)
  library(rmgarch)
  library(quadprog)
  library(riskParityPortfolio)
}))

Sys.setlocale("LC_TIME", "C")  # parse English month abbreviations regardless of OS locale

# =============================================================================
# Configuration  (mirrors config.py)
# =============================================================================

DATA_DIR      <- "DATA"
EMPIRICAL_DIR <- file.path(DATA_DIR, "Empirical")
ARTIFICIAL_DIR<- file.path(DATA_DIR, "Artifical")   # historical typo kept on purpose
RESULTS_DIR   <- "Ergebnisse"

# name -> list(path, date_format)
DATASETS <- list(
  TRBC       = list(file.path(EMPIRICAL_DIR, "TRBC_Business_Sectors_clean.csv"), "%Y-%m-%d"),
  SP500      = list(file.path(EMPIRICAL_DIR, "S&P500_Adj.csv"),                  "%d.%m.%y"),
  Dow        = list(file.path(EMPIRICAL_DIR, "Dow_Adj.csv"),                     "%Y-%m-%d"),
  MonteCarlo = list(file.path(ARTIFICIAL_DIR, "monte_carlo.csv"),                "%Y-%m-%d"),
  GARCH_sim  = list(file.path(ARTIFICIAL_DIR, "garch.csv"),                      "%Y-%m-%d"),
  DCC_sim    = list(file.path(ARTIFICIAL_DIR, "dcc.csv"),                        "%Y-%m-%d")
)

# Backtest defaults (== config.BacktestConfig defaults used by the active run)
DEFAULT_CFG <- list(
  garch_p     = 1L,
  garch_q     = 1L,
  cov_methods = c("Historical", "GARCH", "DCC"),
  models      = c("MVP", "HRP", "ERC"),
  log_weights = TRUE
)

# (train_window, prediction_window) per dataset -- mirrors the active config.COMBOS.
# Edit this to validate on a cheap dataset, e.g.  list(Dow = list(c(1004, 10))).
COMBOS <- list(
  SP500 = list(c(1004L, 10L), c(1004L, 21L))
)

# =============================================================================
# Data loading: prices, log returns and the risk-free series  (== backtest.py)
# =============================================================================

load_prices <- function(dataset) {
  spec <- DATASETS[[dataset]]
  filepath <- spec[[1]]; date_format <- spec[[2]]
  raw <- read.csv(filepath, check.names = FALSE, stringsAsFactors = FALSE,
                  fileEncoding = "UTF-8-BOM")
  dates <- as.Date(raw[[1]], format = date_format)        # errors="coerce" -> NA
  P <- as.matrix(data.frame(lapply(raw[-1], as.numeric), check.names = FALSE))
  keep_date <- !is.na(dates)
  dates <- dates[keep_date]; P <- P[keep_date, , drop = FALSE]
  # keep dates where at least half the assets have a price  (int(0.5 * n_assets))
  thr <- as.integer(0.5 * ncol(P))
  keep_row <- rowSums(!is.na(P)) >= thr
  dates <- dates[keep_row]; P <- P[keep_row, , drop = FALSE]
  ord <- order(dates)                                     # ensure chronological
  list(dates = dates[ord], prices = P[ord, , drop = FALSE])
}

to_log_returns <- function(px) {
  P <- px$prices
  lr <- log(P[-1, , drop = FALSE] / P[-nrow(P), , drop = FALSE])
  dates <- px$dates[-1]
  zeros   <- colSums(lr == 0, na.rm = TRUE)
  present <- colSums(!is.na(lr))
  zero_frac <- zeros / present
  keep <- !is.na(zero_frac) & (zero_frac < 0.5)           # drop assets flat >50%
  list(dates = dates, returns = lr[, keep, drop = FALSE])
}

load_dataset <- function(dataset) {
  px <- load_prices(dataset)
  lr <- to_log_returns(px)
  list(log_returns = lr$returns, dates = lr$dates)
}

# =============================================================================
# Univariate GARCH via rugarch  (replaces fit_garch_univariate)
# =============================================================================
# Returns the per-asset horizon-averaged forecast variance on the RAW return
# scale, with the same EWMA fallback as Python when a fit will not converge.

.ewma_var_last <- function(series, lam = 0.94) {
  # pandas series.pow(2).ewm(alpha=1-lam).mean() (adjust=True), last value
  x2 <- rev(series)^2
  w  <- lam^(seq_along(x2) - 1L)
  sum(w * x2) / sum(w)
}

# constant-mean sGARCH(p,q), normal innovations -- the arch_model default
.uspec <- function(p, q) {
  ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(p, q)),
             mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
             distribution.model = "norm")
}

fit_garch_variances <- function(returns, prediction_window = 1L, p = 1L, q = 1L) {
  horizon <- max(prediction_window, 1L)
  spec <- .uspec(p, q)
  cols <- colnames(returns)
  variances <- setNames(rep(NA_real_, length(cols)), cols)
  for (col in cols) {
    series <- as.numeric(na.omit(returns[, col]))
    if (length(series) < 30L) next                 # too little data -> drop asset
    var_hat <- NA_real_
    if (length(series) >= 50L) {
      var_hat <- tryCatch({
        # arch is happier with percent returns; undo the scaling with /100^2
        fit <- ugarchfit(spec, series * 100, solver = "hybrid",
                         fit.control = list(scale = 1))
        if (convergence(fit) != 0)                  # retry with a different solver
          fit <- ugarchfit(spec, series * 100, solver = "gosolnp",
                           fit.control = list(scale = 1))
        if (convergence(fit) != 0) stop("no convergence")
        s2 <- as.numeric(sigma(ugarchforecast(fit, n.ahead = horizon)))^2
        mean(s2) / (100^2)
      }, error = function(e) NA_real_)
    }
    if (is.na(var_hat)) {                           # GARCH failed -> EWMA fallback
      reason <- if (length(series) < 50L) "insufficient data" else "GARCH did not converge"
      cat(sprintf("[fallback] GARCH->EWMA for %s (%s)\n", col, reason))
      var_hat <- .ewma_var_last(series)
    }
    variances[col] <- var_hat
  }
  variances[!is.na(variances)]
}

# =============================================================================
# Covariance estimators  (== backtest.py, DCC via rmgarch)
# =============================================================================

historical_covariance <- function(returns) {
  cov(returns, use = "pairwise.complete.obs")          # pandas .cov() (pairwise, ddof=1)
}

constant_correlation_covariance <- function(returns, variances) {
  cols <- names(variances)
  if (length(cols) == 0) return(matrix(numeric(0), 0, 0))
  std  <- sqrt(variances)
  corr <- cor(returns[, cols, drop = FALSE], use = "pairwise.complete.obs")
  corr[is.na(corr)] <- 0; diag(corr) <- 1
  cov <- corr * outer(std, std)
  cov <- cov + diag(1e-8, length(cols))
  dimnames(cov) <- list(cols, cols)
  cov
}

# DCC-GARCH covariance from rmgarch. Mirrors Python's aggregation: horizon-
# averaged conditional variance combined with the 1-step-ahead DCC correlation.
dcc_covariance <- function(returns, variances, prediction_window = 1L, p = 1L, q = 1L) {
  cols <- colnames(returns)
  Z <- returns[stats::complete.cases(returns), , drop = FALSE]  # common dates (== std_resid.dropna())
  n_obs <- nrow(Z); n <- ncol(Z)
  diag_fallback <- function() {
    v <- variances[cols]; v[is.na(v)] <- 0
    cov <- diag(v, n) + diag(1e-8, n); dimnames(cov) <- list(cols, cols); cov
  }
  if (n < 2 || n_obs < n + 2) {
    cat(sprintf("[fallback] DCC->diagonal covariance (n=%d, n_obs=%d)\n", n, n_obs))
    return(diag_fallback())
  }
  horizon <- max(prediction_window, 1L)
  out <- tryCatch({
    spec  <- .uspec(p, q)
    mspec <- multispec(replicate(n, spec))
    dspec <- dccspec(uspec = mspec, dccOrder = c(1, 1), distribution = "mvnorm")
    fit   <- dccfit(dspec, data = Z * 100, fit.control = list(eval.se = FALSE))
    fc    <- dccforecast(fit, n.ahead = horizon)
    covH  <- rcov(fc)[[1]]                       # [n, n, horizon] (scaled)
    corH  <- rcor(fc)[[1]]
    vbar  <- if (horizon > 1)
      apply(covH, 3, function(m) diag(m)) |> rowMeans() else diag(covH[, , 1])
    R1    <- corH[, , 1]
    cov   <- (diag(sqrt(vbar)) %*% R1 %*% diag(sqrt(vbar))) / (100^2)
    cov   <- cov + diag(1e-8, n); dimnames(cov) <- list(cols, cols); cov
  }, error = function(e) {
    cat(sprintf("[fallback] DCC fit failed (%s) -> diagonal covariance\n", conditionMessage(e)))
    diag_fallback()
  })
  out
}

# =============================================================================
# Portfolio weighting schemes  (R packages; HRP/MVP/ERC)
# =============================================================================

.make_pd <- function(cov) {                       # symmetric + jitter, force PD
  n <- nrow(cov)
  M <- (cov + t(cov)) / 2 + diag(1e-8, n)
  ev <- min(eigen(M, symmetric = TRUE, only.values = TRUE)$values)
  if (ev <= 0) M <- M + diag(abs(ev) + 1e-8, n)
  dimnames(M) <- dimnames(cov); M
}

.clean_weights <- function(w, cols) {             # normalise, long-only (== _clean_weights)
  s <- setNames(rep(0, length(cols)), cols)
  s[names(w)] <- as.numeric(w)
  s[!is.finite(s)] <- 0; s[s < 0] <- 0
  tot <- sum(s)
  if (tot <= 0) return(setNames(rep(1 / length(cols), length(cols)), cols))
  s / tot
}

naive_weights <- function(assets)
  setNames(rep(1 / length(assets), length(assets)), assets)

mvp_weights <- function(cov_matrix) {             # long-only min variance (quadprog)
  cols <- colnames(cov_matrix); n <- length(cols)
  if (n == 0) return(numeric(0))
  if (n == 1) return(setNames(1, cols))
  D <- .make_pd(cov_matrix)
  Amat <- cbind(rep(1, n), diag(n))               # sum(w)=1 ; w >= 0
  bvec <- c(1, rep(0, n))
  w <- tryCatch(solve.QP(2 * D, rep(0, n), Amat, bvec, meq = 1)$solution,
                error = function(e) rep(1 / n, n))
  .clean_weights(setNames(w, cols), cols)
}

erc_weights <- function(cov_matrix) {             # equal risk contribution (riskParityPortfolio)
  cols <- colnames(cov_matrix); n <- length(cols)
  if (n == 0) return(numeric(0))
  if (n == 1) return(setNames(1, cols))
  D <- .make_pd(cov_matrix)
  w <- tryCatch(riskParityPortfolio(D)$w, error = function(e) rep(1 / n, n))
  .clean_weights(setNames(w, cols), cols)
}

hrp_weights <- function(cov_matrix) {             # hierarchical risk parity (de Prado, == Python/riskfolio)
  cols <- colnames(cov_matrix); n <- length(cols)
  if (n == 0) return(numeric(0))
  if (n == 1) return(setNames(1, cols))
  M <- (cov_matrix + t(cov_matrix)) / 2
  std  <- sqrt(diag(M)); corr <- M / outer(std, std)
  corr[corr > 1] <- 1; corr[corr < -1] <- -1
  D <- sqrt(0.5 * (1 - corr))                     # correlation distance; cluster on D DIRECTLY
  ord <- tryCatch(hclust(as.dist(D), method = "single")$order, error = function(e) seq_len(n))
  w <- rep(1, n); idx <- list(ord)                # recursive bisection (== backtest.py hrp_weights)
  while (length(idx) > 0) {
    new_idx <- list()
    for (i in idx) {
      m <- floor(length(i) / 2); a <- i[1:m]; b <- i[-(1:m)]
      Ca <- as.matrix(M[a, a]); Cb <- as.matrix(M[b, b])
      wa <- (1 / diag(Ca)) / sum(1 / diag(Ca)); wb <- (1 / diag(Cb)) / sum(1 / diag(Cb))
      va <- as.numeric(wa %*% Ca %*% wa); vb <- as.numeric(wb %*% Cb %*% wb)
      alpha <- 1 - va / (va + vb)
      w[a] <- w[a] * alpha; w[b] <- w[b] * (1 - alpha)
      if (length(a) > 1) new_idx <- c(new_idx, list(a))
      if (length(b) > 1) new_idx <- c(new_idx, list(b))
    }
    idx <- new_idx
  }
  .clean_weights(setNames(w, cols), cols)
}

get_weights <- function(model, cov_matrix) {
  switch(model, MVP = mvp_weights(cov_matrix), HRP = hrp_weights(cov_matrix),
         ERC = erc_weights(cov_matrix), stop(paste("unknown model:", model)))
}

# =============================================================================
# Performance metrics and covariance-forecast losses  (== backtest.py exactly)
# =============================================================================

# pandas Series.skew() -- adjusted Fisher-Pearson standardized moment (G1)
.pandas_skew <- function(x) {
  x <- x[!is.na(x)]; n <- length(x)
  if (n < 3) return(NA_real_)
  d <- x - mean(x); m2 <- sum(d^2); m3 <- sum(d^3)
  if (m2 == 0) return(0)
  (n * sqrt(n - 1) / (n - 2)) * (m3 / m2^1.5)
}

# pandas Series.kurt() -- sample EXCESS kurtosis (G2)
.pandas_kurt <- function(x) {
  x <- x[!is.na(x)]; n <- length(x)
  if (n < 4) return(NA_real_)
  d <- x - mean(x); m2 <- sum(d^2); m4 <- sum(d^4)
  if (m2 == 0) return(0)
  num <- n * (n + 1) * (n - 1) * m4
  den <- (n - 2) * (n - 3) * m2^2
  num / den - 3 * (n - 1)^2 / ((n - 2) * (n - 3))
}

calculate_summary_metrics <- function(daily_returns, rf_daily = 0) {
  arr <- as.numeric(daily_returns)
  rf  <- if (length(rf_daily) == 1) rep(rf_daily, length(arr)) else as.numeric(rf_daily)
  excess <- arr - rf
  N <- length(arr)

  ann_ret <- prod(1 + arr)^(252 / N) - 1                  # CAGR (geometric)
  ann_std <- sd(arr) * sqrt(252)                          # sample std (ddof=1)
  ann_excess     <- mean(excess) * 252
  ann_excess_std <- sd(excess) * sqrt(252)
  semi_dev <- sqrt(mean(pmin(excess, 0)^2)) * sqrt(252)

  cumulative <- cumprod(1 + arr)
  drawdowns  <- cumulative / cummax(cumulative) - 1
  max_dd <- min(drawdowns)

  var_95  <- as.numeric(quantile(arr, 0.05, type = 7))    # numpy percentile == type 7
  cvar_95 <- mean(arr[arr <= var_95])

  list(
    `Ann. Return`        = ann_ret,
    `Ann. Std`           = ann_std,
    `Ann. Sharpe`        = if (ann_excess_std > 0) ann_excess / ann_excess_std else NA_real_,
    `Ann. Sharpe (rf=0)` = if (ann_std > 0) (mean(arr) * 252) / ann_std else NA_real_,
    `Ann. Sortino`       = if (semi_dev > 0) ann_excess / semi_dev else NA_real_,
    `Max Drawdown`       = max_dd,
    `Calmar Ratio`       = if (max_dd < 0) ann_ret / abs(max_dd) else NA_real_,
    `CVaR (95%)`         = cvar_95,
    `Skewness`           = .pandas_skew(arr),
    `Excess Kurtosis`    = .pandas_kurt(arr)
  )
}

.qlike <- function(cov, realized) {                       # log|H| + tr(H^-1 S)
  H <- as.matrix(cov); R <- as.matrix(realized)
  if (nrow(H) == 0 || nrow(R) == 0) return(NA_real_)
  S <- crossprod(R) / nrow(R)
  dt <- determinant(H, logarithm = TRUE)
  if (dt$sign <= 0) return(NA_real_)
  tryCatch(as.numeric(dt$modulus) + sum(diag(solve(H, S))),
           error = function(e) NA_real_)
}

.cov_rmse <- function(cov, realized) {                    # ||H - S||_F / N
  H <- as.matrix(cov); R <- as.matrix(realized)
  if (nrow(H) == 0 || nrow(R) == 0) return(NA_real_)
  S <- crossprod(R) / nrow(R)
  sqrt(mean((H - S)^2))
}

.risk_contribution_rmse <- function(weights, realized) {  # RMSE of realized RC from 1/N
  w <- as.numeric(weights); n <- length(w)
  R <- as.matrix(realized)
  if (n == 0 || nrow(R) == 0) return(NA_real_)
  S <- crossprod(R) / nrow(R)
  port_var <- as.numeric(t(w) %*% S %*% w)
  if (port_var <= 1e-300) return(NA_real_)
  rc <- w * as.numeric(S %*% w) / port_var
  sqrt(mean((rc - 1 / n)^2))
}

# =============================================================================
# Rolling-window engine  (== backtest.py run_window / run_backtest)
# =============================================================================

run_window <- function(start, log_returns, dates, cfg) {
  tw <- cfg$train_window; pw <- cfg$prediction_window
  train_idx <- (start + 1L):(start + tw)
  test_idx  <- (start + tw + 1L):(start + tw + pw)
  train <- log_returns[train_idx, , drop = FALSE]
  test  <- log_returns[test_idx, , drop = FALSE]

  # universe from TRAINING info only: >=90% obs and a valid last training day
  keep <- (colMeans(!is.na(train)) >= 0.9) & (!is.na(train[nrow(train), ]))
  train <- train[, keep, drop = FALSE]
  if (ncol(train) == 0 || nrow(test) == 0) return(NULL)

  test <- test[, colnames(train), drop = FALSE]
  test_simple <- exp(test) - 1; test_simple[is.na(test_simple)] <- 0

  # covariance matrices (fit univariate GARCH once, reuse for GARCH & DCC)
  cov_by_method <- list()
  if ("Historical" %in% cfg$cov_methods)
    cov_by_method[["Historical"]] <- historical_covariance(train)
  if ("GARCH" %in% cfg$cov_methods || "DCC" %in% cfg$cov_methods) {
    variances <- fit_garch_variances(train, pw, cfg$garch_p, cfg$garch_q)
    if ("GARCH" %in% cfg$cov_methods)
      cov_by_method[["GARCH"]] <- constant_correlation_covariance(train, variances)
    if ("DCC" %in% cfg$cov_methods)
      cov_by_method[["DCC"]] <- dcc_covariance(train, variances, pw, cfg$garch_p, cfg$garch_q)
  }

  # per-step covariance-forecast quality (QLIKE & Frobenius RMSE vs realized)
  qlike_info <- list(); covrmse_info <- list()
  for (method in names(cov_by_method)) {
    cov <- cov_by_method[[method]]
    if (is.null(cov) || ncol(cov) == 0) next
    realized <- test[, colnames(cov), drop = FALSE]; realized[is.na(realized)] <- 0
    qlike_info[[method]]   <- .qlike(cov, realized)
    covrmse_info[[method]] <- .cov_rmse(cov, realized)
  }

  cov_hist_full <- cov_by_method[["Historical"]]
  if (is.null(cov_hist_full)) cov_hist_full <- historical_covariance(train)

  # jobs: Naive + every (model x method)
  naive_w <- naive_weights(colnames(train))
  jobs <- list(list(model = "Naive", cov_type = "N/A", weights = naive_w, cov = cov_hist_full))
  for (method in names(cov_by_method)) {
    cov <- cov_by_method[[method]]
    if (is.null(cov) || ncol(cov) == 0) next
    for (model in cfg$models)
      jobs[[length(jobs) + 1L]] <- list(model = model, cov_type = method,
                                        weights = get_weights(model, cov), cov = cov)
  }

  per_period <- list(); records <- list(); weights_info <- list()
  formation_date <- dates[test_idx[1]]
  for (job in jobs) {
    name <- if (job$model == "Naive") "Naive" else paste(job$model, job$cov_type)
    cols <- names(job$weights)
    w    <- as.numeric(job$weights)
    port_returns <- as.numeric(test_simple[, cols, drop = FALSE] %*% w)
    per_period[[name]] <- port_returns

    gross <- apply(1 + test_simple[, cols, drop = FALSE], 2, prod)  # drifted end weights
    end_val <- job$weights * gross
    weights_info[[name]] <- list(target = job$weights, end = end_val / sum(end_val))

    covm <- job$cov[cols, cols, drop = FALSE]
    forecasted_std <- sqrt(as.numeric(t(w) %*% covm %*% w))
    rc_rmse <- NA_real_
    if (job$model == "ERC") {
      realized <- test[, cols, drop = FALSE]; realized[is.na(realized)] <- 0
      rc_rmse <- .risk_contribution_rmse(w, realized)
    }
    records[[length(records) + 1L]] <- data.frame(
      Model = job$model, `Covariance Type` = job$cov_type,
      `Window Index` = start %/% pw, `Mean Return` = mean(port_returns),
      `Forecasted Std` = forecasted_std, `RC RMSE` = rc_rmse,
      check.names = FALSE, stringsAsFactors = FALSE)
  }

  list(per_period = per_period, records = do.call(rbind, records),
       dates = dates[test_idx], weights_info = weights_info,
       formation_date = formation_date, qlike_info = qlike_info,
       covrmse_info = covrmse_info)
}

run_backtest <- function(cfg, log_returns, dates, out_dir, verbose = TRUE) {
  n <- nrow(log_returns); tw <- cfg$train_window; pw <- cfg$prediction_window
  if (n - tw - pw < 0) { cat("No valid rolling windows.\n"); return(invisible(NULL)) }
  starts <- seq(0L, n - tw - pw, by = pw)
  total <- length(starts)

  results <- list(); records <- list(); period_dates <- as.Date(character(0))
  weights_hist <- list(); end_hist <- list()
  qlike_acc <- list(); covrmse_acc <- list()
  qlike_rows <- list(); covrmse_rows <- list()

  for (i in seq_along(starts)) {
    res <- run_window(starts[i], log_returns, dates, cfg)
    if (verbose) cat(sprintf("\r%d/%d windows completed", i, total))
    if (is.null(res)) next
    records[[length(records) + 1L]] <- res$records
    period_dates <- c(period_dates, res$dates)
    for (name in names(res$per_period))
      results[[name]] <- c(results[[name]], res$per_period[[name]])
    for (name in names(res$weights_info)) {
      weights_hist[[name]] <- c(weights_hist[[name]],
        list(list(date = res$formation_date, target = res$weights_info[[name]]$target)))
      end_hist[[name]] <- c(end_hist[[name]], list(res$weights_info[[name]]$end))
    }
    if (length(res$qlike_info) > 0) {
      for (m in names(res$qlike_info))   qlike_acc[[m]]   <- c(qlike_acc[[m]], res$qlike_info[[m]])
      for (m in names(res$covrmse_info)) covrmse_acc[[m]] <- c(covrmse_acc[[m]], res$covrmse_info[[m]])
      qlike_rows[[length(qlike_rows) + 1L]]     <- c(list(Date = res$formation_date), res$qlike_info)
      covrmse_rows[[length(covrmse_rows) + 1L]] <- c(list(Date = res$formation_date), res$covrmse_info)
    }
  }
  if (verbose) cat("\n")
  if (length(records) == 0) { cat("No valid rolling windows.\n"); return(invisible(NULL)) }

  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  metrics <- do.call(rbind, records)
  write.csv(metrics, file.path(out_dir, "backtest_metrics_R.csv"), row.names = FALSE)

  # per-period returns per model (rows = dates)
  ret_df <- data.frame(Date = format(period_dates, "%Y-%m-%d"),
                       as.data.frame(results, check.names = FALSE), check.names = FALSE)
  write.csv(ret_df, file.path(out_dir, "returns_R.csv"), row.names = FALSE)

  avg_turnover <- .average_turnover(weights_hist, end_hist)
  if (isTRUE(cfg$log_weights)) .write_weights(weights_hist, file.path(out_dir, "weights_R.csv"))

  avg_qlike <- sapply(qlike_acc, function(v) mean(v, na.rm = TRUE))
  avg_covrmse <- sapply(covrmse_acc, function(v) mean(v, na.rm = TRUE))
  if (length(qlike_rows) > 0) {
    write.csv(.stack_rows(qlike_rows), file.path(out_dir, "qlike_R.csv"), row.names = FALSE)
    write.csv(.stack_rows(covrmse_rows), file.path(out_dir, "cov_rmse_R.csv"), row.names = FALSE)
  }

  summary <- .build_summary(results, metrics, period_dates, avg_turnover,
                            avg_qlike, avg_covrmse)
  write.csv(summary, file.path(out_dir, "summary_R.csv"), row.names = FALSE)
  if (verbose) { cat("Annualized performance summary (R):\n"); print(summary, row.names = FALSE) }
  invisible(summary)
}

# =============================================================================
# Result assembly helpers  (== backtest.py)
# =============================================================================

.stack_rows <- function(rows) {                   # list of named lists -> data.frame
  cols <- unique(unlist(lapply(rows, names)))
  df <- do.call(rbind, lapply(rows, function(r) {
    r$Date <- format(r$Date, "%Y-%m-%d")
    as.data.frame(c(r, setNames(rep(list(NA), length(setdiff(cols, names(r)))),
                                setdiff(cols, names(r)))), check.names = FALSE)
  }))
  df <- df[, cols, drop = FALSE]
  df[order(df$Date), , drop = FALSE]
}

.average_turnover <- function(weights_hist, end_hist) {
  out <- setNames(rep(NA_real_, length(weights_hist)), names(weights_hist))
  for (name in names(weights_hist)) {
    hist <- weights_hist[[name]]; ends <- end_hist[[name]]
    if (length(hist) < 2) next
    per_window <- numeric(0)
    for (k in 2:length(hist)) {
      prev_end <- ends[[k - 1]]; target <- hist[[k]]$target
      idx <- union(names(prev_end), names(target))
      a <- setNames(rep(0, length(idx)), idx); b <- a
      a[names(target)]   <- target
      b[names(prev_end)] <- prev_end
      per_window <- c(per_window, sum(abs(a - b)))
    }
    out[name] <- mean(per_window)
  }
  out
}

.write_weights <- function(weights_hist, path) {
  rows <- list()
  for (name in names(weights_hist)) {
    if (name == "Naive") { model <- "Naive"; cov_type <- "N/A" }
    else { sp <- strsplit(name, " ")[[1]]; cov_type <- tail(sp, 1); model <- paste(head(sp, -1), collapse = " ") }
    for (entry in weights_hist[[name]]) {
      tw <- entry$target
      rows[[length(rows) + 1L]] <- data.frame(
        Date = format(entry$date, "%Y-%m-%d"), Model = model,
        `Covariance Type` = cov_type, Asset = names(tw), Weight = as.numeric(tw),
        check.names = FALSE, stringsAsFactors = FALSE)
    }
  }
  write.csv(do.call(rbind, rows), path, row.names = FALSE)
}

.split_name <- function(name) {
  if (name == "Naive") return(list(model = "Naive", cov_type = "N/A"))
  sp <- strsplit(name, " ")[[1]]
  list(model = paste(head(sp, -1), collapse = " "), cov_type = tail(sp, 1))
}

.build_summary <- function(results, metrics, period_dates, avg_turnover,
                           avg_qlike, avg_covrmse) {
  key <- paste(metrics$Model, metrics$`Covariance Type`)
  avg_fcst <- tapply(metrics$`Forecasted Std`, key, mean, na.rm = TRUE) * sqrt(252)
  avg_rc   <- tapply(metrics$`RC RMSE`, key, function(v)
                     if (all(is.na(v))) NA_real_ else mean(v, na.rm = TRUE))

  rows <- list()
  for (name in names(results)) {
    rets <- results[[name]]
    if (length(rets) == 0) next
    sp <- .split_name(name)
    m <- calculate_summary_metrics(rets, 0)   # risk-free dropped for now (rf = 0)
    k <- paste(sp$model, sp$cov_type)
    fcst <- if (k %in% names(avg_fcst)) avg_fcst[[k]] else NA_real_
    cov_key <- if (sp$cov_type == "N/A") "Historical" else sp$cov_type
    rows[[length(rows) + 1L]] <- data.frame(
      Model = sp$model, `Covariance Type` = sp$cov_type,
      `Ann. Return` = m$`Ann. Return`, `Ann. Std` = m$`Ann. Std`,
      `Ann. Std (fcst)` = fcst,
      `Real / Fcst Std` = if (!is.na(fcst) && fcst > 0) m$`Ann. Std` / fcst else NA_real_,
      `Avg QLIKE` = if (cov_key %in% names(avg_qlike)) avg_qlike[[cov_key]] else NA_real_,
      `Avg Cov RMSE` = if (cov_key %in% names(avg_covrmse)) avg_covrmse[[cov_key]] else NA_real_,
      `ERC RC RMSE` = if (k %in% names(avg_rc)) avg_rc[[k]] else NA_real_,
      `Avg Turnover` = if (name %in% names(avg_turnover)) avg_turnover[[name]] else NA_real_,
      `Ann. Sharpe` = m$`Ann. Sharpe`, `Ann. Sortino` = m$`Ann. Sortino`,
      `Max Drawdown` = m$`Max Drawdown`, `Calmar Ratio` = m$`Calmar Ratio`,
      `CVaR (95%)` = m$`CVaR (95%)`, `Skewness` = m$`Skewness`,
      `Excess Kurtosis` = m$`Excess Kurtosis`,
      check.names = FALSE, stringsAsFactors = FALSE)
  }
  df <- do.call(rbind, rows)
  df[order(df$Model, df$`Covariance Type`), , drop = FALSE]
}

# =============================================================================
# Batch runner -- every (dataset, train, pred) in COMBOS
# =============================================================================

run_all <- function(combos = COMBOS, out_root = RESULTS_DIR, cfg_overrides = list()) {
  for (dataset in names(combos)) {
    cat(sprintf("\nLoading dataset %s ...\n", dataset))
    d <- load_dataset(dataset)
    for (tp in combos[[dataset]]) {
      train <- as.integer(tp[1]); pred <- as.integer(tp[2])
      cfg <- modifyList(modifyList(DEFAULT_CFG, list(dataset = dataset,
               train_window = train, prediction_window = pred)), cfg_overrides)
      tag <- if (cfg$garch_p == 1 && cfg$garch_q == 1) "" else
        sprintf("_g%d-%d", cfg$garch_p, cfg$garch_q)
      out_dir <- file.path(out_root, dataset, sprintf("%d_%d%s", train, pred, tag))
      cat(sprintf("\n=== %s %d_%d (R) ===\n", dataset, train, pred))
      run_backtest(cfg, d$log_returns, d$dates, out_dir)
    }
  }
}

if (sys.nframe() == 0L) run_all()
