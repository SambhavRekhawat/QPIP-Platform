"""
analytics/factor_risk_decomposition.py
----------------------------------------
Two engines in one module:

1. Factor Risk Decomposition
   - OLS regression of portfolio returns on factor returns
   - Computes each factor's contribution to return AND risk
   - Uses factor scores already in the `factors` PostgreSQL table

2. Copula Dependency Modelling
   - Gaussian copula (linear dependence — baseline)
   - Student-t copula (fat tails — more realistic for equity crises)
   - Tail dependence coefficient between asset pairs
   - Joint downside probability matrix

All functions accept data directly — no PostgreSQL I/O except the
factor loader, which reads from the existing `factors` table.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger
from scipy import stats
from scipy.optimize import minimize
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from sqlalchemy import text

from config import TRADING_DAYS_YEAR
from database.connection import get_db_engine

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  PART 1 — FACTOR RISK DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

# ─── Factor Return Proxies ────────────────────────────────────────────────────

def build_factor_return_proxies(
    returns_matrix: pd.DataFrame,
    factor_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct factor return time series from cross-sectional factor scores.

    Method: Long-short factor mimicking portfolios.
    For each factor:
      - Sort tickers by factor score
      - Top 30% = long leg,  Bottom 30% = short leg
      - Factor return = mean(long leg returns) - mean(short leg returns)

    This is a standard academic approach (Fama-French style).
    """
    if factor_df.empty or returns_matrix.empty:
        return pd.DataFrame()

    score_cols = {
        "momentum_factor": "momentum_score",
        "quality_factor":  "quality_score",
        "value_factor":    "value_score",
        "lowvol_factor":   "volatility_score",
    }

    factor_returns: Dict[str, pd.Series] = {}

    for fname, score_col in score_cols.items():
        if score_col not in factor_df.columns:
            continue

        scores = factor_df.set_index("ticker")[score_col].dropna()
        available = [t for t in scores.index if t in returns_matrix.columns]
        if len(available) < 4:
            continue

        scores = scores[available].sort_values(ascending=False)
        n      = len(scores)
        top    = scores.iloc[:max(1, n // 3)].index.tolist()
        bot    = scores.iloc[-max(1, n // 3):].index.tolist()

        long_r  = returns_matrix[top].mean(axis=1)
        short_r = returns_matrix[bot].mean(axis=1)
        factor_returns[fname] = (long_r - short_r).rename(fname)

    if not factor_returns:
        return pd.DataFrame()

    return pd.DataFrame(factor_returns).dropna()


# ─── OLS Factor Regression ────────────────────────────────────────────────────

def run_factor_regression(
    portfolio_returns: pd.Series,
    factor_returns_df: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> Dict:
    """
    Regress portfolio returns on factor returns + market (benchmark) return.

    Returns
    -------
    dict with:
      - factor_betas       : coefficient for each factor
      - factor_t_stats     : t-statistic for each factor
      - r_squared          : model fit
      - alpha              : intercept (unexplained excess return)
      - factor_return_contribution  : factor beta × annualised factor return
      - residual_return    : alpha return
    """
    # Align all series on common dates
    bench = benchmark_returns.rename("market_factor")
    all_factors = pd.concat([factor_returns_df, bench], axis=1)
    combined    = pd.concat([portfolio_returns.rename("port"), all_factors], axis=1).dropna()

    if len(combined) < 30:
        return {"error": "Insufficient data for factor regression (need ≥30 observations)"}

    y = combined["port"]
    X = add_constant(combined.drop(columns=["port"]))

    model  = OLS(y, X).fit()
    coefs  = model.params
    tstats = model.tvalues
    pvals  = model.pvalues

    # Factor names (exclude const)
    factor_cols = [c for c in X.columns if c != "const"]

    factor_betas = {col: float(coefs[col]) for col in factor_cols}
    factor_tstat = {col: float(tstats[col]) for col in factor_cols}
    factor_pval  = {col: float(pvals[col])  for col in factor_cols}

    # Return contribution: β × mean factor return (annualised)
    factor_mean_ret = combined[factor_cols].mean() * TRADING_DAYS_YEAR
    return_contrib  = {col: float(factor_betas[col] * factor_mean_ret[col])
                       for col in factor_cols}

    # Risk contribution: β² × factor variance
    factor_var = combined[factor_cols].var() * TRADING_DAYS_YEAR
    risk_contrib = {col: float(factor_betas[col] ** 2 * factor_var[col])
                    for col in factor_cols}
    total_risk   = sum(risk_contrib.values()) + float(model.resid.var() * TRADING_DAYS_YEAR)
    risk_pct     = {col: float(v / total_risk * 100) if total_risk > 0 else 0
                    for col, v in risk_contrib.items()}

    return {
        "factor_betas":        factor_betas,
        "factor_t_stats":      factor_tstat,
        "factor_p_values":     factor_pval,
        "r_squared":           float(model.rsquared),
        "r_squared_adj":       float(model.rsquared_adj),
        "alpha_daily":         float(coefs.get("const", 0)),
        "alpha_annual":        float(coefs.get("const", 0)) * TRADING_DAYS_YEAR,
        "return_contribution": return_contrib,
        "risk_contribution":   risk_contrib,
        "risk_pct":            risk_pct,
        "residual_var":        float(model.resid.var() * TRADING_DAYS_YEAR),
        "n_obs":               int(len(combined)),
        "model_summary":       str(model.summary()),
    }


def factor_attribution_table(regression_result: Dict) -> pd.DataFrame:
    """Convert regression output into a clean attribution table for display."""
    if "error" in regression_result:
        return pd.DataFrame([{"Error": regression_result["error"]}])

    betas   = regression_result["factor_betas"]
    tstats  = regression_result["factor_t_stats"]
    ret_c   = regression_result["return_contribution"]
    risk_c  = regression_result["risk_pct"]

    rows = []
    for factor in betas:
        label = factor.replace("_factor", "").replace("_", " ").title()
        rows.append({
            "Factor":              label,
            "Beta (Exposure)":     round(betas[factor], 3),
            "t-stat":              round(tstats[factor], 2),
            "Return Contribution": f"{ret_c.get(factor, 0)*100:.2f}%",
            "% of Portfolio Risk": f"{risk_c.get(factor, 0):.1f}%",
            "Significant":         "✅" if abs(tstats[factor]) > 1.96 else "—",
        })

    # Add alpha row
    rows.append({
        "Factor":              "Alpha (Unexplained)",
        "Beta (Exposure)":     "—",
        "t-stat":              "—",
        "Return Contribution": f"{regression_result['alpha_annual']*100:.2f}%",
        "% of Portfolio Risk": f"{regression_result['residual_var']/max(sum(regression_result['risk_contribution'].values())+regression_result['residual_var'],1e-9)*100:.1f}%",
        "Significant":         "—",
    })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 — COPULA DEPENDENCY MODELLING
# ══════════════════════════════════════════════════════════════════════════════

# ─── Marginal CDFs ────────────────────────────────────────────────────────────

def empirical_cdf(returns: pd.Series) -> pd.Series:
    """Transform returns to uniform [0,1] via empirical CDF (rank-based)."""
    return returns.rank(pct=True)


def to_uniform(returns_matrix: pd.DataFrame) -> pd.DataFrame:
    """Transform each column to uniform [0,1] margins via empirical CDF."""
    return returns_matrix.apply(empirical_cdf, axis=0)


def to_normal(uniform_df: pd.DataFrame) -> pd.DataFrame:
    """Map uniform margins to standard normal via inverse normal CDF."""
    return uniform_df.apply(lambda col: stats.norm.ppf(col.clip(0.001, 0.999)))


# ─── Gaussian Copula ─────────────────────────────────────────────────────────

def fit_gaussian_copula(
    returns_matrix: pd.DataFrame,
) -> Dict:
    """
    Fit a Gaussian copula to the return data.
    The copula captures linear dependence structure (equivalent to
    assuming multivariate normality of the margins).

    Returns
    -------
    dict with correlation matrix (copula parameter) and diagnostics.
    """
    uniform   = to_uniform(returns_matrix.dropna())
    normal    = to_normal(uniform)
    corr_mat  = normal.corr()

    return {
        "type":        "Gaussian",
        "correlation": corr_mat,
        "n_obs":       len(returns_matrix.dropna()),
        "description": "Assumes elliptical dependence. Underestimates tail co-movement.",
    }


# ─── Student-t Copula ────────────────────────────────────────────────────────

def fit_t_copula(
    returns_matrix: pd.DataFrame,
    df_init: float = 5.0,
) -> Dict:
    """
    Fit a Student-t copula via MLE on the degrees-of-freedom parameter.

    The t-copula is the standard in credit risk and equity tail modelling
    because it allows for tail dependence — assets tend to co-crash more
    than a Gaussian copula would predict.

    Parameters
    ----------
    df_init : initial guess for degrees of freedom (lower = fatter tails)
    """
    data = returns_matrix.dropna()
    uniform = to_uniform(data)
    normal  = to_normal(uniform)
    corr_m  = normal.corr().values
    n       = len(data.columns)

    # ── MLE for degrees of freedom ────────────────────────────────────────────
    def neg_log_likelihood(params):
        nu = max(params[0], 2.1)   # enforce ν > 2 for finite variance
        try:
            # Multivariate t log-likelihood
            x   = normal.values
            det = np.linalg.det(corr_m)
            inv = np.linalg.inv(corr_m)
            if det <= 0:
                return 1e10
            q = np.einsum("ij,jk,ik->i", x, inv, x)   # Mahalanobis² per obs
            T = len(x)
            ll = (
                T * (stats.gammaln((nu + n) / 2)
                     - stats.gammaln(nu / 2)
                     - (n / 2) * np.log(nu * np.pi)
                     - 0.5 * np.log(det))
                - ((nu + n) / 2) * np.sum(np.log(1 + q / nu))
            )
            return -ll
        except Exception:
            return 1e10

    result = minimize(neg_log_likelihood, [df_init],
                      bounds=[(2.1, 50)], method="L-BFGS-B")
    nu_opt = float(result.x[0]) if result.success else df_init

    # ── Tail dependence coefficient ───────────────────────────────────────────
    # λ_L = 2 · t_{ν+1}( -√((ν+1)(1-ρ)/(1+ρ)) )   Embrechts et al.
    tdc_matrix = pd.DataFrame(index=data.columns, columns=data.columns, dtype=float)
    for i, ti in enumerate(data.columns):
        for j, tj in enumerate(data.columns):
            if i == j:
                tdc_matrix.loc[ti, tj] = 1.0
            else:
                rho = float(corr_m[i, j])
                rho = np.clip(rho, -0.999, 0.999)
                arg = -np.sqrt((nu_opt + 1) * (1 - rho) / (1 + rho))
                tdc = 2 * stats.t.cdf(arg, df=nu_opt + 1)
                tdc_matrix.loc[ti, tj] = float(tdc)

    return {
        "type":              "Student-t",
        "degrees_of_freedom": nu_opt,
        "correlation":       pd.DataFrame(corr_m, index=data.columns, columns=data.columns),
        "tail_dependence":   tdc_matrix,
        "n_obs":             len(data),
        "description":       f"t-copula with ν={nu_opt:.1f} df. "
                             f"Lower ν = fatter tails. ν<5 = significant tail co-movement.",
    }


# ─── Monte Carlo from Copula ─────────────────────────────────────────────────

def simulate_from_t_copula(
    returns_matrix: pd.DataFrame,
    weights: Dict[str, float],
    n_simulations: int = 10_000,
    nu: Optional[float] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate portfolio returns using the t-copula dependency structure.
    More realistic than pure Cholesky Monte Carlo for tail scenarios.

    Returns
    -------
    1-D array of simulated portfolio daily returns (length n_simulations)
    """
    np.random.seed(seed)

    data    = returns_matrix.dropna()
    tickers = [t for t in weights if t in data.columns]
    w       = np.array([weights[t] for t in tickers])
    w      /= w.sum()

    # Fit copula
    uniform  = to_uniform(data[tickers])
    normal_z = to_normal(uniform)
    corr_m   = normal_z.corr().values
    n        = len(tickers)

    if nu is None:
        cop_result = fit_t_copula(data[tickers])
        nu = cop_result["degrees_of_freedom"]

    # Cholesky of correlation matrix
    try:
        L = np.linalg.cholesky(corr_m + 1e-6 * np.eye(n))
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(np.eye(n))

    # Simulate from t-distribution
    Z   = np.random.standard_normal((n_simulations, n)) @ L.T
    chi = np.random.chisquare(nu, size=n_simulations)
    T_z = Z / np.sqrt(chi / nu)[:, None]   # t-distributed correlated shocks

    # Map back to uniform via t CDF
    U_sim = stats.t.cdf(T_z, df=nu)

    # Map uniform to empirical return distribution (inverse empirical CDF)
    sim_returns = np.zeros_like(U_sim)
    for i, ticker in enumerate(tickers):
        empirical_quantiles = np.percentile(data[ticker].values, U_sim[:, i] * 100)
        sim_returns[:, i]   = empirical_quantiles

    return sim_returns @ w


# ─── Tail Dependence Summary ─────────────────────────────────────────────────

def tail_dependence_summary(
    returns_matrix: pd.DataFrame,
    tickers: List[str],
) -> pd.DataFrame:
    """
    Compute pairwise tail dependence for a portfolio.
    Returns a matrix showing which pairs tend to crash together.
    """
    data = returns_matrix[[t for t in tickers if t in returns_matrix.columns]].dropna()
    if len(data) < 60:
        return pd.DataFrame()

    cop = fit_t_copula(data)
    tdc = cop["tail_dependence"]

    # Format nicely
    tdc_display = tdc.copy().astype(float).round(3)
    return tdc_display


# ─── Copula vs Gaussian comparison ───────────────────────────────────────────

def copula_vs_gaussian_var(
    returns_matrix: pd.DataFrame,
    weights: Dict[str, float],
    confidence: float = 0.99,
    n_simulations: int = 10_000,
) -> Dict:
    """
    Compare VaR estimates from:
      (a) Standard Cholesky (Gaussian) Monte Carlo
      (b) t-Copula simulation

    The difference quantifies the model risk from ignoring tail dependence.
    """
    from backtesting.stress_testing import run_monte_carlo

    tickers = [t for t in weights if t in returns_matrix.columns]
    data    = returns_matrix[tickers]

    # Gaussian VaR
    mc_gauss = run_monte_carlo(weights, data, n_simulations=n_simulations, seed=42)
    gauss_var = mc_gauss["var_table"][confidence]["var"]
    gauss_cvar = mc_gauss["var_table"][confidence]["cvar"]

    # t-copula VaR
    try:
        cop_pnl  = simulate_from_t_copula(data, weights, n_simulations, seed=42)
        cop_var  = float(np.percentile(cop_pnl, (1 - confidence) * 100))
        tail     = cop_pnl[cop_pnl <= cop_var]
        cop_cvar = float(tail.mean()) if len(tail) > 0 else cop_var
    except Exception as e:
        logger.warning(f"  t-copula simulation failed: {e}")
        cop_var  = gauss_var
        cop_cvar = gauss_cvar

    return {
        "confidence":     confidence,
        "gaussian_var":   gauss_var,
        "gaussian_cvar":  gauss_cvar,
        "t_copula_var":   cop_var,
        "t_copula_cvar":  cop_cvar,
        "var_difference": cop_var  - gauss_var,
        "cvar_difference": cop_cvar - gauss_cvar,
        "interpretation": (
            "t-Copula shows higher tail risk — assets tend to crash together more than "
            "the Gaussian model assumes."
            if cop_var < gauss_var else
            "Models are broadly consistent — tail dependence is not materially different."
        ),
    }
