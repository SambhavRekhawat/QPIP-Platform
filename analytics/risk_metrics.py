"""
analytics/risk_metrics.py
--------------------------
Computes all standard quantitative risk and performance metrics:
  Sharpe, Sortino, Calmar, Max Drawdown, Beta, Alpha,
  Information Ratio, VaR (historical), CVaR, rolling metrics.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from config import TRADING_DAYS_YEAR, RISK_FREE_RATE


# ─── Annualisation Helper ─────────────────────────────────────────────────────

def _ann(daily_std: float) -> float:
    return daily_std * np.sqrt(TRADING_DAYS_YEAR)


# ─── Core Metrics ─────────────────────────────────────────────────────────────

def annualised_return(returns: pd.Series) -> float:
    """Compound Annualised Growth Rate from daily returns."""
    n = len(returns.dropna())
    if n < 2:
        return np.nan
    cum = (1 + returns.dropna()).prod()
    return cum ** (TRADING_DAYS_YEAR / n) - 1


def annualised_volatility(returns: pd.Series) -> float:
    return _ann(returns.dropna().std())


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = RISK_FREE_RATE,
) -> float:
    ann_ret = annualised_return(returns)
    ann_vol = annualised_volatility(returns)
    if ann_vol == 0:
        return np.nan
    return (ann_ret - risk_free) / ann_vol


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = RISK_FREE_RATE,
    target: float = 0.0,
) -> float:
    """Downside deviation based Sortino ratio."""
    excess  = returns.dropna() - target / TRADING_DAYS_YEAR
    downside = excess[excess < 0]
    if len(downside) < 2:
        return np.nan
    downside_std = _ann(downside.std())
    ann_ret = annualised_return(returns)
    if downside_std == 0:
        return np.nan
    return (ann_ret - risk_free) / downside_std


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative float."""
    cum_returns = (1 + returns.fillna(0)).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown    = (cum_returns - rolling_max) / rolling_max
    return float(drawdown.min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown time series."""
    cum = (1 + returns.fillna(0)).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


def calmar_ratio(returns: pd.Series) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.nan
    return annualised_return(returns) / mdd


def beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 10:
        return np.nan
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    var_bench = cov[1, 1]
    if var_bench == 0:
        return np.nan
    return cov[0, 1] / var_bench


def alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free: float = RISK_FREE_RATE,
) -> float:
    b = beta(returns, benchmark_returns)
    ann_r = annualised_return(returns)
    ann_b = annualised_return(benchmark_returns)
    return ann_r - (risk_free + b * (ann_b - risk_free))


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    excess  = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_YEAR)


def var_historical(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Historical VaR at given confidence level (positive number)."""
    return abs(float(returns.dropna().quantile(1 - confidence)))


def cvar_historical(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Expected Shortfall (CVaR)."""
    threshold = returns.dropna().quantile(1 - confidence)
    tail = returns[returns <= threshold]
    if tail.empty:
        return np.nan
    return abs(float(tail.mean()))


def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    return returns_df.dropna().corr()


def hit_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Fraction of periods where portfolio beat benchmark."""
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    beats = (aligned.iloc[:, 0] > aligned.iloc[:, 1]).sum()
    return beats / len(aligned) if len(aligned) > 0 else np.nan


# ─── Rolling Metrics ─────────────────────────────────────────────────────────

def rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    risk_free: float = RISK_FREE_RATE,
) -> pd.Series:
    rf_daily = risk_free / TRADING_DAYS_YEAR
    roll_mean = returns.rolling(window).mean()
    roll_std  = returns.rolling(window).std()
    return (roll_mean - rf_daily) / roll_std * np.sqrt(TRADING_DAYS_YEAR)


def rolling_beta(
    returns: pd.Series,
    benchmark: pd.Series,
    window: int = 63,
) -> pd.Series:
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    r  = aligned.iloc[:, 0]
    b  = aligned.iloc[:, 1]
    cov = r.rolling(window).cov(b)
    var = b.rolling(window).var()
    return cov / var


def rolling_volatility(
    returns: pd.Series,
    window: int = 21,
) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS_YEAR)


# ─── Full Metrics Summary ─────────────────────────────────────────────────────

def full_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    name: str = "Portfolio",
) -> Dict:
    """
    Return a dict of all key risk/performance metrics for a return series.
    """
    r = returns.dropna()
    b = benchmark_returns.reindex(r.index).dropna()
    r = r.reindex(b.index)

    return {
        "name":               name,
        "total_return":       float((1 + r).prod() - 1),
        "ann_return":         float(annualised_return(r)),
        "ann_volatility":     float(annualised_volatility(r)),
        "sharpe_ratio":       float(sharpe_ratio(r)),
        "sortino_ratio":      float(sortino_ratio(r)),
        "calmar_ratio":       float(calmar_ratio(r)),
        "max_drawdown":       float(max_drawdown(r)),
        "beta":               float(beta(r, b)),
        "alpha":              float(alpha(r, b)),
        "information_ratio":  float(information_ratio(r, b)),
        "var_95":             float(var_historical(r, 0.95)),
        "cvar_95":            float(cvar_historical(r, 0.95)),
        "hit_ratio":          float(hit_ratio(r, b)),
        "skewness":           float(r.skew()),
        "kurtosis":           float(r.kurtosis()),
        "num_trading_days":   int(len(r)),
    }


def benchmark_comparison(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """
    Side-by-side metrics table for portfolio vs benchmark.
    """
    port_m  = full_metrics(portfolio_returns, benchmark_returns, "Portfolio")
    bench_m = full_metrics(benchmark_returns, benchmark_returns, "NIFTY 50")

    metrics_to_show = [
        "total_return", "ann_return", "ann_volatility", "sharpe_ratio",
        "sortino_ratio", "max_drawdown", "beta", "alpha", "information_ratio",
        "var_95", "hit_ratio",
    ]
    rows = []
    for m in metrics_to_show:
        rows.append({
            "Metric":    m.replace("_", " ").title(),
            "Portfolio": port_m.get(m, np.nan),
            "NIFTY 50":  bench_m.get(m, np.nan),
        })
    return pd.DataFrame(rows)
