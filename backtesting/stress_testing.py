"""
backtesting/stress_testing.py
------------------------------
Institutional-grade Monte Carlo VaR and Stress Testing Engine.

Capabilities
------------
1. Monte Carlo VaR  — 10,000 simulations driven by the historical
   covariance matrix (Cholesky decomposition). Reports VaR and CVaR
   at 95%, 97.5%, and 99% confidence.

2. Historical Stress Tests  — replay named market crises
   (COVID crash, 2008 GFC, 2020 India lockdown, etc.) through the
   current portfolio weights and covariance structure.

3. Factor Shock Scenarios  — user-defined shocks to individual
   factors (e.g. "what if momentum collapses −30%?").

4. Volatility Regimes  — compare VaR under normal, elevated, and
   crisis volatility assumptions.

All results can be persisted to PostgreSQL and recalled by the dashboard.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import warnings
import numpy as np
import pandas as pd
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from loguru import logger
from sqlalchemy import text

from config import TRADING_DAYS_YEAR, RISK_FREE_RATE
from database.connection import get_db_engine

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────

N_SIMULATIONS    = 10_000
CONFIDENCE_LEVELS = [0.90, 0.95, 0.975, 0.99]
HOLDING_PERIODS   = [1, 5, 10, 21]     # days

# Named historical stress scenarios (peak-to-trough approximate returns)
STRESS_SCENARIOS: Dict[str, Dict] = {
    "COVID Crash 2020": {
        "description": "Global markets fell sharply in Feb–Mar 2020. "
                       "NIFTY 50 dropped ~38% in 40 days.",
        "market_shock": -0.38,
        "vol_multiplier": 3.5,
        "duration_days": 40,
        "period": "Feb–Mar 2020",
    },
    "Global Financial Crisis 2008": {
        "description": "NIFTY 50 fell ~60% from Jan 2008 to Mar 2009. "
                       "Credit markets froze globally.",
        "market_shock": -0.60,
        "vol_multiplier": 4.0,
        "duration_days": 300,
        "period": "Jan 2008–Mar 2009",
    },
    "IL&FS Crisis 2018": {
        "description": "Indian NBFC crisis triggered by IL&FS default. "
                       "NIFTY fell ~15% over 3 months.",
        "market_shock": -0.15,
        "vol_multiplier": 2.0,
        "duration_days": 90,
        "period": "Sep–Dec 2018",
    },
    "India Lockdown 2020": {
        "description": "Initial India-specific shock from nationwide lockdown "
                       "announcement. Markets fell ~13% in one week.",
        "market_shock": -0.13,
        "vol_multiplier": 2.5,
        "duration_days": 7,
        "period": "Mar 2020 (1 week)",
    },
    "Demonetisation 2016": {
        "description": "RBI demonetisation shock. Markets fell ~8% in 2 days, "
                       "recovery took 3 months.",
        "market_shock": -0.08,
        "vol_multiplier": 1.8,
        "duration_days": 5,
        "period": "Nov 2016",
    },
    "Russia-Ukraine Shock 2022": {
        "description": "Geopolitical shock with oil price surge and FII outflows. "
                       "NIFTY fell ~15% over 6 weeks.",
        "market_shock": -0.15,
        "vol_multiplier": 2.0,
        "duration_days": 42,
        "period": "Feb–Mar 2022",
    },
    "Mild Correction": {
        "description": "Typical 10% market correction as seen several times per decade.",
        "market_shock": -0.10,
        "vol_multiplier": 1.5,
        "duration_days": 30,
        "period": "Generic",
    },
    "Severe Bear Market": {
        "description": "Extreme scenario: 50% drawdown over 12 months.",
        "market_shock": -0.50,
        "vol_multiplier": 4.5,
        "duration_days": 252,
        "period": "Hypothetical",
    },
    "Adani Group Crisis 2023": {
        "description": "Hindenburg Research report triggered sharp sell-off in Adani "
                       "group stocks. NIFTY fell ~5%, banking/infra stocks hit hard.",
        "market_shock": -0.08,
        "vol_multiplier": 2.2,
        "duration_days": 10,
        "period": "Jan–Feb 2023",
    },
    "INR Depreciation Shock": {
        "description": "Rapid INR/USD depreciation (15%+ in 6 months) leading to "
                       "FII outflows and imported inflation pressure.",
        "market_shock": -0.18,
        "vol_multiplier": 2.0,
        "duration_days": 120,
        "period": "Generic INR shock",
    },
    "RBI Emergency Rate Hike": {
        "description": "Hypothetical: RBI hikes repo rate by 100bps in emergency "
                       "session to defend currency or control inflation.",
        "market_shock": -0.12,
        "vol_multiplier": 1.8,
        "duration_days": 21,
        "period": "Hypothetical",
    },
    "Oil Price Surge (+50%)": {
        "description": "Oil spikes 50% due to Middle East conflict or OPEC cut. "
                       "India imports 85% of oil — inflationary, CAD widening.",
        "market_shock": -0.14,
        "vol_multiplier": 1.9,
        "duration_days": 60,
        "period": "Hypothetical",
    },
    "Indian Banking Crisis": {
        "description": "Hypothetical systemic NPA crisis. NIFTY Bank falls 40%, "
                       "RBI intervenes with emergency liquidity measures.",
        "market_shock": -0.35,
        "vol_multiplier": 3.5,
        "duration_days": 90,
        "period": "Hypothetical",
    },
    "Global Recession 2025": {
        "description": "US/EU enter deep recession, global demand collapses, "
                       "IT exports fall 30%, FII sell ₹2L cr in 6 months.",
        "market_shock": -0.28,
        "vol_multiplier": 3.0,
        "duration_days": 180,
        "period": "Hypothetical",
    },
    "Taper Tantrum India 2013": {
        "description": "Fed tapering announcement caused EM selloff. NIFTY fell "
                       "~12%, INR hit 68/USD. Classic EM contagion scenario.",
        "market_shock": -0.12,
        "vol_multiplier": 2.0,
        "duration_days": 45,
        "period": "May–Aug 2013",
    },
}


# ─── Covariance Matrix ────────────────────────────────────────────────────────

def compute_covariance_matrix(
    returns_matrix: pd.DataFrame,
    window: int = 252,
    method: str = "historical",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute the annualised covariance matrix and mean daily returns.

    Parameters
    ----------
    returns_matrix : wide DataFrame of daily returns (index=date, cols=tickers)
    window         : rolling window in trading days
    method         : 'historical' | 'ewma' (exponentially weighted)

    Returns
    -------
    cov_matrix     : annualised covariance (n × n DataFrame)
    mean_returns   : mean daily return per ticker
    """
    data = returns_matrix.tail(window).dropna(axis=1, how="all")
    data = data.fillna(0)

    if method == "ewma":
        # Exponentially weighted — more weight to recent observations
        # pandas ewm span ≈ half-life of 63 days
        cov_daily = data.ewm(span=63, min_periods=30).cov().iloc[-len(data.columns):]
        cov_daily = cov_daily.droplevel(0)    # remove date level
    else:
        cov_daily = data.cov()

    cov_annual  = cov_daily * TRADING_DAYS_YEAR
    mean_daily  = data.mean()
    return cov_annual, mean_daily


def cholesky_decompose(cov_matrix: pd.DataFrame) -> np.ndarray:
    """
    Cholesky decomposition for Monte Carlo simulation.
    Adds a small regularisation term to ensure positive-definiteness,
    which can fail with highly correlated assets or short histories.
    """
    cov = cov_matrix.values
    # Regularise: add small diagonal term (0.1% of mean diagonal)
    eps = 1e-6 * np.mean(np.diag(cov))
    cov = cov + eps * np.eye(len(cov))

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Fall back to eigenvalue clipping if still not PD
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(cov)

    return L


# ─── Monte Carlo Engine ───────────────────────────────────────────────────────

def run_monte_carlo(
    weights: Dict[str, float],
    returns_matrix: pd.DataFrame,
    n_simulations: int = N_SIMULATIONS,
    holding_period: int = 1,
    window: int = 252,
    vol_multiplier: float = 1.0,
    seed: Optional[int] = 42,
) -> Dict:
    """
    Run Monte Carlo simulation for portfolio VaR.

    Method
    ------
    1. Compute historical covariance matrix (annualised).
    2. Scale to the requested holding period.
    3. Cholesky-decompose to produce correlated random shocks.
    4. Simulate 10,000 portfolio P&L paths.
    5. Read off VaR and CVaR at each confidence level.

    Parameters
    ----------
    weights         : {ticker: weight} — will be re-normalised
    returns_matrix  : daily returns DataFrame
    n_simulations   : number of Monte Carlo paths
    holding_period  : days to project (1 = daily VaR, 21 = monthly)
    vol_multiplier  : scale volatility (>1 = stress, <1 = calm)
    seed            : random seed for reproducibility

    Returns
    -------
    dict with simulation results, VaR/CVaR table, and raw P&L array
    """
    if seed is not None:
        np.random.seed(seed)

    # Align tickers to those in both weights and returns
    tickers = [t for t in weights if t in returns_matrix.columns]
    if len(tickers) < 2:
        raise ValueError(f"Need at least 2 tickers with returns data. Got: {tickers}")

    w = np.array([weights[t] for t in tickers], dtype=float)
    w = w / w.sum()   # normalise

    # Covariance matrix
    cov_annual, mean_daily = compute_covariance_matrix(returns_matrix[tickers], window)
    cov_annual = cov_annual.loc[tickers, tickers]

    # Scale to holding period and apply vol multiplier
    # Annualised → daily → holding period
    cov_period  = (cov_annual / TRADING_DAYS_YEAR) * holding_period * (vol_multiplier ** 2)
    mean_period = mean_daily[tickers].values * holding_period

    # Cholesky decomposition
    L = cholesky_decompose(pd.DataFrame(cov_period, index=tickers, columns=tickers))

    # Simulate: Z is (n_sim × n_assets) matrix of standard normals
    Z          = np.random.standard_normal((n_simulations, len(tickers)))
    corr_shocks = Z @ L.T   # shape (n_sim, n_assets) — correlated returns

    # Apply drift (mean return)
    sim_returns = corr_shocks + mean_period   # (n_sim, n_assets)

    # Portfolio P&L = weighted sum
    portfolio_pnl = sim_returns @ w   # (n_sim,)

    # VaR and CVaR at each confidence level
    var_table = {}
    for cl in CONFIDENCE_LEVELS:
        threshold    = np.percentile(portfolio_pnl, (1 - cl) * 100)
        tail_losses  = portfolio_pnl[portfolio_pnl <= threshold]
        cvar         = float(tail_losses.mean()) if len(tail_losses) > 0 else threshold
        var_table[cl] = {
            "var":  float(threshold),
            "cvar": float(cvar),
        }

    # Distribution statistics
    pnl_mean    = float(portfolio_pnl.mean())
    pnl_std     = float(portfolio_pnl.std())
    pnl_skew    = float(pd.Series(portfolio_pnl).skew())
    pnl_kurt    = float(pd.Series(portfolio_pnl).kurtosis())
    prob_loss   = float((portfolio_pnl < 0).mean())
    expected_loss = float(portfolio_pnl[portfolio_pnl < 0].mean()) if (portfolio_pnl < 0).any() else 0.0

    return {
        "tickers":          tickers,
        "weights":          {t: float(w[i]) for i, t in enumerate(tickers)},
        "n_simulations":    n_simulations,
        "holding_period":   holding_period,
        "vol_multiplier":   vol_multiplier,
        "var_table":        var_table,
        "pnl_mean":         pnl_mean,
        "pnl_std":          pnl_std,
        "pnl_skew":         pnl_skew,
        "pnl_kurtosis":     pnl_kurt,
        "prob_loss":        prob_loss,
        "expected_loss":    expected_loss,
        "portfolio_pnl":    portfolio_pnl,   # raw array for plotting
        "run_date":         datetime.now().isoformat(),
    }


def run_multi_horizon_var(
    weights: Dict[str, float],
    returns_matrix: pd.DataFrame,
    n_simulations: int = N_SIMULATIONS,
) -> pd.DataFrame:
    """
    Run Monte Carlo VaR across all standard holding periods and confidence levels.
    Returns a DataFrame with rows = holding periods, columns = (confidence, metric).
    """
    rows = []
    for hp in HOLDING_PERIODS:
        result = run_monte_carlo(weights, returns_matrix,
                                 n_simulations=n_simulations,
                                 holding_period=hp)
        row = {"Holding Period": f"{hp}d"}
        for cl, vals in result["var_table"].items():
            pct = int(cl * 100)
            row[f"VaR {pct}%"]  = vals["var"]
            row[f"CVaR {pct}%"] = vals["cvar"]
        rows.append(row)

    return pd.DataFrame(rows).set_index("Holding Period")




# ─── Geometric Brownian Motion Simulation ────────────────────────────────────

def run_gbm_simulation(
    weights: Dict[str, float],
    returns_matrix: pd.DataFrame,
    n_simulations: int = N_SIMULATIONS,
    horizon_days: int = 252,
    seed: Optional[int] = 42,
) -> Dict:
    """
    Simulate portfolio paths using Geometric Brownian Motion (GBM).

    GBM assumes log-returns are normally distributed with constant drift (μ)
    and volatility (σ). Each path represents one possible future trajectory
    of the portfolio over `horizon_days` trading days.

    Formula per step:
        S_{t+1} = S_t × exp((μ - σ²/2)Δt + σ√Δt × Z)
    where Z ~ N(0,1) correlated across assets via Cholesky decomposition.

    Returns
    -------
    dict with:
      - paths        : (n_simulations × horizon_days) array of cumulative returns
      - final_values : distribution of terminal portfolio values
      - percentiles  : fan-chart percentiles (5, 25, 50, 75, 95)
      - var_terminal : VaR on terminal distribution at 95% and 99%
    """
    if seed is not None:
        np.random.seed(seed)

    tickers = [t for t in weights if t in returns_matrix.columns]
    w = np.array([weights[t] for t in tickers], dtype=float)
    w = w / w.sum()

    data = returns_matrix[tickers].tail(252).fillna(0)
    mu_daily  = data.mean().values          # drift per day
    cov_ann, _ = compute_covariance_matrix(data)
    cov_daily  = cov_ann.loc[tickers, tickers].values / TRADING_DAYS_YEAR

    L = cholesky_decompose(pd.DataFrame(cov_daily, index=tickers, columns=tickers))

    dt = 1.0  # one trading day per step
    n  = len(tickers)

    # Simulate all paths: shape (n_sim, horizon_days, n_assets)
    Z = np.random.standard_normal((n_simulations, horizon_days, n))
    corr_shocks = Z @ L.T  # correlated normal shocks

    # GBM log-return per step per asset
    vol_daily = np.sqrt(np.diag(cov_daily))
    drift = (mu_daily - 0.5 * vol_daily ** 2) * dt
    diffusion = corr_shocks * vol_daily * np.sqrt(dt)

    log_returns = drift + diffusion        # (n_sim, horizon, n_assets)
    cum_log     = np.cumsum(log_returns, axis=1)
    price_paths = np.exp(cum_log)          # relative price paths starting at 1.0

    # Portfolio value paths = weighted sum of asset paths
    port_paths  = price_paths @ w          # (n_sim, horizon_days)
    final_vals  = port_paths[:, -1]        # terminal values

    # Fan chart percentiles
    pcts = np.percentile(port_paths, [5, 10, 25, 50, 75, 90, 95], axis=0)

    # Terminal VaR
    terminal_losses = 1 - final_vals
    var_95  = float(np.percentile(terminal_losses, 95))
    var_99  = float(np.percentile(terminal_losses, 99))
    cvar_95 = float(terminal_losses[terminal_losses >= var_95].mean())
    cvar_99 = float(terminal_losses[terminal_losses >= var_99].mean())

    return {
        "paths":           port_paths,       # (n_sim, horizon_days) — use for fan chart
        "final_values":    final_vals,        # (n_sim,) — terminal distribution
        "percentiles":     pcts,              # (7, horizon_days) — fan chart bands
        "percentile_labels": ["5%","10%","25%","50%","75%","90%","95%"],
        "horizon_days":    horizon_days,
        "n_simulations":   n_simulations,
        "var_95_terminal": var_95,
        "var_99_terminal": var_99,
        "cvar_95_terminal": cvar_95,
        "cvar_99_terminal": cvar_99,
        "prob_loss":        float((final_vals < 1.0).mean()),
        "prob_gain_10pct":  float((final_vals > 1.10).mean()),
        "median_terminal":  float(np.median(final_vals)),
        "mean_terminal":    float(final_vals.mean()),
    }

# ─── Stress Testing ───────────────────────────────────────────────────────────

def run_stress_scenario(
    scenario_name: str,
    weights: Dict[str, float],
    returns_matrix: pd.DataFrame,
    portfolio_value: float = 100.0,
    n_simulations: int = N_SIMULATIONS,
) -> Dict:
    """
    Apply a named stress scenario and compute the expected portfolio loss.

    Method
    ------
    1. Retrieve the scenario definition (market shock + vol multiplier).
    2. Apply the market shock as a systematic return component using
       each ticker's historical beta to the benchmark.
    3. Run Monte Carlo with elevated volatility on top of the shock.
    4. Report P&L statistics.
    """
    scenario = STRESS_SCENARIOS.get(scenario_name)
    if not scenario:
        raise ValueError(f"Unknown scenario: {scenario_name}. "
                         f"Available: {list(STRESS_SCENARIOS.keys())}")

    market_shock    = scenario["market_shock"]
    vol_multiplier  = scenario["vol_multiplier"]
    duration_days   = scenario["duration_days"]

    # Run Monte Carlo under stressed volatility
    mc_result = run_monte_carlo(
        weights, returns_matrix,
        n_simulations=n_simulations,
        holding_period=duration_days,
        vol_multiplier=vol_multiplier,
        seed=42,
    )

    # Overlay the systematic market shock on top of Monte Carlo results
    # Each ticker's shock contribution ≈ beta × market_shock
    tickers = mc_result["tickers"]
    betas = _estimate_betas(tickers, returns_matrix)

    w_arr = np.array([mc_result["weights"][t] for t in tickers])
    b_arr = np.array([betas.get(t, 1.0)       for t in tickers])

    # Portfolio-level systematic shock
    systematic_shock = float(w_arr @ b_arr * market_shock)

    # Shift the Monte Carlo P&L distribution by the systematic shock
    stressed_pnl = mc_result["portfolio_pnl"] + systematic_shock

    # Recompute VaR/CVaR on stressed distribution
    stressed_var_table = {}
    for cl in CONFIDENCE_LEVELS:
        threshold = np.percentile(stressed_pnl, (1 - cl) * 100)
        tail      = stressed_pnl[stressed_pnl <= threshold]
        stressed_var_table[cl] = {
            "var":  float(threshold),
            "cvar": float(tail.mean()) if len(tail) > 0 else float(threshold),
        }

    # In ₹ terms (assuming portfolio_value is in ₹ or index units)
    p99_loss_pct  = stressed_var_table[0.99]["cvar"]
    p99_loss_abs  = abs(p99_loss_pct * portfolio_value)

    return {
        "scenario":          scenario_name,
        "description":       scenario["description"],
        "period":            scenario["period"],
        "market_shock":      market_shock,
        "vol_multiplier":    vol_multiplier,
        "duration_days":     duration_days,
        "systematic_shock":  systematic_shock,
        "var_table":         stressed_var_table,
        "stressed_pnl":      stressed_pnl,
        "p99_loss_pct":      p99_loss_pct,
        "p99_loss_abs":      p99_loss_abs,
        "portfolio_value":   portfolio_value,
        "run_date":          datetime.now().isoformat(),
    }


def run_all_stress_scenarios(
    weights: Dict[str, float],
    returns_matrix: pd.DataFrame,
    portfolio_value: float = 100.0,
    n_simulations: int = N_SIMULATIONS,
) -> pd.DataFrame:
    """Run every named scenario and return a summary DataFrame."""
    rows = []
    for name in STRESS_SCENARIOS:
        try:
            result = run_stress_scenario(
                name, weights, returns_matrix, portfolio_value, n_simulations
            )
            rows.append({
                "Scenario":         name,
                "Period":           result["period"],
                "Market Shock":     result["market_shock"],
                "Portfolio Shock":  result["systematic_shock"],
                "VaR 99%":          result["var_table"][0.99]["var"],
                "CVaR 99%":         result["var_table"][0.99]["cvar"],
                "VaR 95%":          result["var_table"][0.95]["var"],
                "CVaR 95%":         result["var_table"][0.95]["cvar"],
                "Duration (days)":  result["duration_days"],
            })
        except Exception as e:
            logger.warning(f"  Scenario '{name}' failed: {e}")

    return pd.DataFrame(rows)


# ─── Volatility Regime Analysis ───────────────────────────────────────────────

def run_regime_var(
    weights: Dict[str, float],
    returns_matrix: pd.DataFrame,
    n_simulations: int = N_SIMULATIONS,
) -> pd.DataFrame:
    """
    Compare VaR (1-day, 99%) across three volatility regimes:
      - Normal  (vol_multiplier = 1.0)
      - Elevated (vol_multiplier = 1.5)
      - Crisis  (vol_multiplier = 2.5)
    """
    regimes = {
        "Normal market":   1.0,
        "Elevated stress": 1.5,
        "Crisis":          2.5,
    }
    rows = []
    for regime, mult in regimes.items():
        result = run_monte_carlo(
            weights, returns_matrix,
            n_simulations=n_simulations,
            holding_period=1,
            vol_multiplier=mult,
            seed=42,
        )
        row = {"Regime": regime, "Vol Multiplier": f"{mult:.1f}×"}
        for cl, vals in result["var_table"].items():
            pct = int(cl * 100)
            row[f"VaR {pct}%"]  = vals["var"]
            row[f"CVaR {pct}%"] = vals["cvar"]
        rows.append(row)

    return pd.DataFrame(rows)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _estimate_betas(
    tickers: List[str],
    returns_matrix: pd.DataFrame,
    window: int = 252,
) -> Dict[str, float]:
    """Estimate each ticker's beta to the equal-weighted universe return."""
    data   = returns_matrix[tickers].tail(window).fillna(0)
    market = data.mean(axis=1)   # equal-weighted proxy
    betas  = {}
    for t in tickers:
        try:
            cov = np.cov(data[t].values, market.values)
            betas[t] = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 1.0
        except Exception:
            betas[t] = 1.0
    return betas


def portfolio_correlation_matrix(
    tickers: List[str],
    returns_matrix: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """Return pairwise correlation matrix for the portfolio."""
    data = returns_matrix[tickers].tail(window).fillna(0)
    return data.corr()


def component_var(
    weights: Dict[str, float],
    returns_matrix: pd.DataFrame,
    confidence: float = 0.99,
    holding_period: int = 1,
) -> pd.Series:
    """
    Compute each position's marginal contribution to portfolio VaR.
    Component VaR sums to total portfolio VaR.
    """
    tickers = [t for t in weights if t in returns_matrix.columns]
    w = np.array([weights[t] for t in tickers], dtype=float)
    w = w / w.sum()

    cov_annual, _ = compute_covariance_matrix(returns_matrix[tickers])
    cov_daily  = cov_annual.loc[tickers, tickers].values / TRADING_DAYS_YEAR
    cov_period = cov_daily * holding_period

    port_variance = float(w @ cov_period @ w)
    port_std      = np.sqrt(port_variance)

    # z-score for confidence level
    from scipy.stats import norm
    z = norm.ppf(confidence)

    # Marginal VaR = ∂VaR/∂w_i = z × (Σw)_i / σ_p
    sigma_w    = cov_period @ w
    marginal   = z * sigma_w / port_std
    component  = marginal * w   # element-wise
    pct_contrib = component / (z * port_std) * 100

    return pd.DataFrame({
        "Ticker":            tickers,
        "Weight":            w,
        "Component VaR":     component,
        "Marginal VaR":      marginal,
        "% of Total VaR":   pct_contrib,
    }).set_index("Ticker")


# ─── Persistence ─────────────────────────────────────────────────────────────

def save_var_result(result: Dict, engine=None) -> None:
    """Persist Monte Carlo VaR results to PostgreSQL stress_test_results table."""
    if engine is None:
        engine = get_db_engine()

    # Extract 1-day VaR summary for quick recall
    vt = result.get("var_table", {})
    record = {
        "run_date":          result.get("run_date", datetime.now().isoformat()),
        "tickers":           json.dumps(result.get("tickers", [])),
        "weights":           json.dumps({k: round(v, 6) for k, v in result.get("weights", {}).items()}),
        "n_simulations":     result.get("n_simulations", N_SIMULATIONS),
        "holding_period":    result.get("holding_period", 1),
        "vol_multiplier":    result.get("vol_multiplier", 1.0),
        "var_90":            vt.get(0.90, {}).get("var"),
        "cvar_90":           vt.get(0.90, {}).get("cvar"),
        "var_95":            vt.get(0.95, {}).get("var"),
        "cvar_95":           vt.get(0.95, {}).get("cvar"),
        "var_975":           vt.get(0.975, {}).get("var"),
        "cvar_975":          vt.get(0.975, {}).get("cvar"),
        "var_99":            vt.get(0.99, {}).get("var"),
        "cvar_99":           vt.get(0.99, {}).get("cvar"),
        "prob_loss":         result.get("prob_loss"),
        "expected_loss":     result.get("expected_loss"),
        "pnl_skew":          result.get("pnl_skew"),
        "pnl_kurtosis":      result.get("pnl_kurtosis"),
        "scenario":          result.get("scenario", "Monte Carlo"),
    }

    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO stress_test_results
                        (run_date, tickers, weights, n_simulations, holding_period,
                         vol_multiplier, var_90, cvar_90, var_95, cvar_95,
                         var_975, cvar_975, var_99, cvar_99,
                         prob_loss, expected_loss, pnl_skew, pnl_kurtosis, scenario)
                    VALUES
                        (:run_date, CAST(:tickers AS jsonb), CAST(:weights AS jsonb),
                         :n_simulations, :holding_period, :vol_multiplier,
                         :var_90, :cvar_90, :var_95, :cvar_95,
                         :var_975, :cvar_975, :var_99, :cvar_99,
                         :prob_loss, :expected_loss, :pnl_skew, :pnl_kurtosis, :scenario)
                """),
                record,
            )
        logger.info("✅ VaR result saved to stress_test_results")
    except Exception as e:
        logger.warning(f"Could not save VaR result: {e}")
