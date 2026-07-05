"""
Generate the synthetic price datasets — no real data is read in.

We build three datasets of the same shape (N_OBS x N_ASSETS), all sharing the
same unconditional risk profile (long-run vols + correlation) so the only thing
that differs is the second-moment *dynamics*:

  1) monte_carlo.csv : i.i.d. draws from a multivariate normal with a CONSTANT
                       mean and covariance.  -> the "null": a static historical
                       covariance is the right model, GARCH/DCC only add noise.
  2) garch.csv       : a GARCH(1,1) per asset (volatility clustering) tied
                       together by a STATIC correlation.  -> a GARCH/constant-
                       correlation estimator should beat the static historical
                       one (esp. for covariance-sensitive portfolios like MVP).
  3) dcc.csv         : GARCH(1,1) marginals PLUS a Dynamic Conditional
                       Correlation (Engle 2002) recursion, so the correlation
                       itself moves through time.  -> a DCC estimator should beat
                       both the static historical and the constant-correlation
                       GARCH one.

Tune the block below. The defaults are deliberately chosen with strong (but
still commonly-seen) clustering and dynamic-correlation parameters so the
"correct" model wins by a clear margin on its matching dataset.

Each path is converted to a price series and written to ./DATA/Artifical/ with a
Date column followed by one column per asset (same layout the loaders expect).

    python datagen.py
"""
import os

import numpy as np
import pandas as pd

import main as config

# =============================================================================
# CONFIG  -- everything that defines the data-generating process
# =============================================================================

N_OBS    = 6000          # number of price rows (dates)
N_ASSETS = 20            # number of assets (columns)
SEED     = 1
OUTPUT_DIR = config.ARTIFICIAL_DIR

START_DATE  = "2000-01-03"   # first date; a business-day calendar is built from here
START_PRICE = 100.0          # every asset starts here
TRADING_DAYS = 252

# --- mean / risk premium -----------------------------------------------------
# Return is proportional to risk: each asset's annual *arithmetic* expected
# return is  ANNUAL_RISK_FREE + RISK_PRICE * annual_vol  -- a constant-Sharpe
# market where RISK_PRICE is the market price of risk (the Sharpe ratio).
ANNUAL_RISK_FREE = 0.02      # intercept of the risk-return line (zero-vol return)
RISK_PRICE       = 0.50      # annual excess return earned per unit of annual vol

# --- unconditional (long-run) volatility, annualised ------------------------
# each asset gets its own long-run vol drawn uniformly from this range
ANNUAL_VOL_LOW  = 0.15
ANNUAL_VOL_HIGH = 0.35

# --- unconditional correlation (one-factor structure) -----------------------
AVG_CORRELATION = 0.35       # target average pairwise correlation (0..1)

# --- GARCH(1,1) volatility dynamics (shared by the garch & dcc datasets) -----
# omega is set per asset so the unconditional variance matches uncond_var.
# alpha+beta is the persistence; a larger alpha means stronger volatility
# clustering -> a bigger edge for GARCH over a static historical covariance.
GARCH_ALPHA = 0.08           # ARCH: reaction to last shock
GARCH_BETA  = 0.90           # GARCH: persistence of variance (alpha+beta = 0.98)

# --- DCC dynamic-correlation params (dcc dataset only) ----------------------
# bigger DCC_A -> correlation swings harder in response to shocks.  Values are
# typical equity DCC estimates.  DCC_A + DCC_B must be < 1.
DCC_A = 0.03                 # correlation reaction to shocks
DCC_B = 0.96                 # correlation persistence


# =============================================================================
# Parameter construction
# =============================================================================

def build_parameters(rng):
    """Turn the CONFIG block into concrete per-asset arrays / matrices."""
    annual_vol = rng.uniform(ANNUAL_VOL_LOW, ANNUAL_VOL_HIGH, N_ASSETS)
    uncond_var = (annual_vol / np.sqrt(TRADING_DAYS)) ** 2           # daily variance

    # return proportional to risk: the target annual *arithmetic* return grows
    # linearly with volatility.  Back out the log-drift mu so the simulated
    # *simple* return hits that mean exactly:  E[e^r] - 1 = target  =>
    #   mu = ln(1 + target_daily) - var/2     (undo the Jensen variance term)
    target_annual = ANNUAL_RISK_FREE + RISK_PRICE * annual_vol
    target_daily  = target_annual / TRADING_DAYS
    mu = np.log1p(target_daily) - 0.5 * uncond_var                  # daily log drift

    # one-factor correlation: corr_ij = l_i * l_j (i != j), diag = 1.  Equal
    # loadings l = sqrt(c) give an average correlation of exactly c; we add a
    # little heterogeneity around it so the matrix isn't perfectly equicorrelated.
    base = np.sqrt(AVG_CORRELATION)
    loadings = np.clip(base + rng.normal(0, 0.08, N_ASSETS), 0.1, 0.97)
    corr = np.outer(loadings, loadings)
    np.fill_diagonal(corr, 1.0)

    # GARCH(1,1): pin omega so the unconditional variance omega/(1-alpha-beta)
    # equals uncond_var.
    omega = uncond_var * (1.0 - GARCH_ALPHA - GARCH_BETA)

    return {
        "mu": mu,
        "uncond_var": uncond_var,
        "corr": corr,
        "omega": omega,
        "alpha": GARCH_ALPHA,
        "beta": GARCH_BETA,
    }


# =============================================================================
# Simulators  (all work in raw daily log-return units)
# =============================================================================

def simulate_monte_carlo(p, n, rng):
    """Constant mean and covariance: the null with no second-moment dynamics."""
    std = np.sqrt(p["uncond_var"])
    cov = p["corr"] * np.outer(std, std)
    return rng.multivariate_normal(p["mu"], cov, size=n)


def simulate_garch(p, n, rng):
    """GARCH(1,1) marginals with a STATIC correlation between the innovations."""
    mu, omega, alpha, beta = p["mu"], p["omega"], p["alpha"], p["beta"]
    chol = np.linalg.cholesky(p["corr"])
    # correlated standardized innovations z_t = L u_t,  u_t ~ N(0, I)
    z = rng.standard_normal((n, N_ASSETS)) @ chol.T

    out = np.empty((n, N_ASSETS))
    h = p["uncond_var"].copy()                  # start at the unconditional variance
    eps_prev = np.zeros(N_ASSETS)
    for t in range(n):
        h = omega + alpha * eps_prev ** 2 + beta * h
        eps = np.sqrt(h) * z[t]
        out[t] = mu + eps
        eps_prev = eps
    return out


def simulate_dcc(p, n, rng):
    """GARCH(1,1) marginals + a DCC recursion driving a time-varying correlation."""
    mu, omega, alpha, beta = p["mu"], p["omega"], p["alpha"], p["beta"]
    q_bar = p["corr"]                           # unconditional correlation target
    omega_q = (1.0 - DCC_A - DCC_B) * q_bar

    out = np.empty((n, N_ASSETS))
    Q = q_bar.copy()
    z_prev = np.zeros(N_ASSETS)
    h = p["uncond_var"].copy()
    eps_prev = np.zeros(N_ASSETS)
    for t in range(n):
        # 1) DCC correlation for this step (uses last step's standardized shock)
        Q = omega_q + DCC_A * np.outer(z_prev, z_prev) + DCC_B * Q
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)
        L = np.linalg.cholesky(R + np.eye(N_ASSETS) * 1e-10)

        # 2) draw correlated standardized residuals z_t ~ N(0, R_t)
        z_t = L @ rng.standard_normal(N_ASSETS)

        # 3) GARCH conditional variance per asset -> returns
        h = omega + alpha * eps_prev ** 2 + beta * h
        eps = np.sqrt(h) * z_t
        out[t] = mu + eps

        eps_prev = eps
        z_prev = z_t
    return out


# =============================================================================
# Output
# =============================================================================

def returns_to_prices(returns, dates, columns):
    """(T x N) log returns -> (T+1 x N) price frame, every asset starting flat."""
    paths = START_PRICE * np.exp(returns.cumsum(axis=0))
    full = np.vstack([np.full(N_ASSETS, START_PRICE), paths])
    out = pd.DataFrame(full, index=dates, columns=columns)
    out.index.name = "Date"
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    p = build_parameters(rng)
    columns = [f"A{i + 1:02d}" for i in range(N_ASSETS)]
    dates = pd.bdate_range(start=START_DATE, periods=N_OBS)   # N_OBS price rows
    n_ret = N_OBS - 1                                          # one fewer return
    persistence = p["alpha"] + p["beta"]
    print(f"Generating {N_OBS} dates x {N_ASSETS} assets "
          f"(GARCH persistence={persistence:.2f}, DCC a+b={DCC_A + DCC_B:.2f}).")

    for name, sim in [("monte_carlo", simulate_monte_carlo),
                      ("garch",       simulate_garch),
                      ("dcc",         simulate_dcc)]:
        print(f"  {name} ...")
        returns = sim(p, n_ret, rng)
        returns_to_prices(returns, dates, columns).to_csv(f"{OUTPUT_DIR}/{name}.csv")

    print(f"Done. Wrote 3 files to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
