"""
analytics/portfolio.py
-----------------------
Portfolio construction, weighting methods, return series generation,
and persistence to PostgreSQL.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import json
import numpy as np
import pandas as pd
from datetime import date, datetime
from loguru import logger
from sqlalchemy import text
from typing import Dict, List, Optional, Tuple

from config import (
    WEIGHTING_METHODS, BENCHMARK_TICKER, TRADING_DAYS_YEAR, RISK_FREE_RATE
)
from analytics.risk_metrics import drawdown_series, rolling_sharpe, rolling_volatility
from database.connection import get_db_engine


# ─── Weighting Methods ────────────────────────────────────────────────────────

def equal_weights(tickers: List[str]) -> Dict[str, float]:
    n = len(tickers)
    return {t: 1.0 / n for t in tickers}


def market_cap_weights(
    tickers: List[str],
    market_caps: Dict[str, float],
) -> Dict[str, float]:
    """Weights proportional to market capitalisation."""
    caps = {t: market_caps.get(t, 1.0) for t in tickers}
    total = sum(caps.values())
    if total == 0:
        return equal_weights(tickers)
    return {t: caps[t] / total for t in tickers}


def validate_custom_weights(
    weights: Dict[str, float],
    tickers: List[str],
    tolerance: float = 1e-3,
) -> Dict[str, float]:
    """Validate and normalise custom weights."""
    total = sum(weights.get(t, 0.0) for t in tickers)
    if abs(total - 1.0) > tolerance:
        logger.warning(f"Custom weights sum to {total:.4f}, normalising to 1.0")
        total = max(total, 1e-9)
        return {t: weights.get(t, 0.0) / total for t in tickers}
    return {t: weights.get(t, 0.0) for t in tickers}


def compute_weights(
    tickers: List[str],
    method: str,
    market_caps: Optional[Dict[str, float]] = None,
    custom_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Dispatch to the right weighting function."""
    if method == "Equal Weighted":
        return equal_weights(tickers)
    elif method == "Market Cap Weighted":
        if not market_caps:
            logger.warning("No market caps provided, falling back to equal weights.")
            return equal_weights(tickers)
        return market_cap_weights(tickers, market_caps)
    elif method == "Custom Weighted":
        if not custom_weights:
            logger.warning("No custom weights provided, falling back to equal weights.")
            return equal_weights(tickers)
        return validate_custom_weights(custom_weights, tickers)
    else:
        raise ValueError(f"Unknown weighting method: {method}")


# ─── Portfolio Return Series ──────────────────────────────────────────────────

def compute_portfolio_returns(
    returns_matrix: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """
    Given a wide returns DataFrame (rows=dates, cols=tickers) and a weights dict,
    compute the weighted portfolio return series.
    Gracefully skips any ticker not present in the returns matrix.
    """
    tickers_available = [t for t in weights if t in returns_matrix.columns]
    missing = [t for t in weights if t not in returns_matrix.columns]
    if missing:
        import warnings
        warnings.warn(f"Tickers not in returns matrix (skipped): {missing}")
    if not tickers_available:
        raise ValueError(
            f"No matching tickers found in returns matrix. "
            f"Requested: {list(weights.keys())}, "
            f"Available: {returns_matrix.columns.tolist()}"
        )

    w = pd.Series({t: weights[t] for t in tickers_available})
    w = w / w.sum()  # re-normalise after filtering

    port_returns = returns_matrix[tickers_available].dot(w)
    port_returns.name = "portfolio_return"
    return port_returns


def compute_portfolio_nav(
    portfolio_returns: pd.Series,
    initial_value: float = 100.0,
) -> pd.Series:
    """Cumulative NAV series starting at initial_value."""
    return initial_value * (1 + portfolio_returns.fillna(0)).cumprod()


# ─── Attribution ─────────────────────────────────────────────────────────────

def returns_attribution(
    returns_matrix: pd.DataFrame,
    weights: Dict[str, float],
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """
    Brinson-style return attribution.
    Returns a DataFrame showing each ticker's contribution.
    """
    tickers = [t for t in weights if t in returns_matrix.columns]
    period_returns = returns_matrix[tickers].mean()
    bench_avg = benchmark_returns.reindex(returns_matrix.index).mean()

    rows = []
    for t in tickers:
        w   = weights.get(t, 0)
        r   = float(period_returns.get(t, 0))
        contrib = w * r
        rows.append({
            "Ticker":          t,
            "Weight":          round(w * 100, 2),
            "Return (%)":      round(r * 100 * 252, 2),  # annualised approx
            "Contribution (%)": round(contrib * 100 * 252, 2),
            "vs Benchmark":    round((r - bench_avg) * 100 * 252, 2),
        })
    df = pd.DataFrame(rows).sort_values("Contribution (%)", ascending=False)
    return df


# ─── Portfolio Summary ────────────────────────────────────────────────────────

def portfolio_summary(
    tickers: List[str],
    weights: Dict[str, float],
    returns_matrix: pd.DataFrame,
    benchmark_returns: pd.Series,
    method: str,
) -> Dict:
    """
    Compute a comprehensive portfolio summary including NAV, risk metrics, drawdown.
    """
    from analytics.risk_metrics import full_metrics

    port_returns = compute_portfolio_returns(returns_matrix, weights)
    bench_aligned = benchmark_returns.reindex(port_returns.index)

    metrics = full_metrics(port_returns, bench_aligned, "Portfolio")

    nav    = compute_portfolio_nav(port_returns)
    dd     = drawdown_series(port_returns)
    r_sharpe = rolling_sharpe(port_returns)
    r_vol  = rolling_volatility(port_returns)

    return {
        "tickers":          tickers,
        "weights":          weights,
        "method":           method,
        "metrics":          metrics,
        "portfolio_returns": port_returns,
        "nav":              nav,
        "drawdown":         dd,
        "rolling_sharpe":   r_sharpe,
        "rolling_vol":      r_vol,
        "benchmark_returns": bench_aligned,
        "benchmark_nav":    compute_portfolio_nav(bench_aligned.fillna(0)),
    }


# ─── Portfolio ID ─────────────────────────────────────────────────────────────

def make_portfolio_id(tickers: List[str], method: str) -> str:
    key = "_".join(sorted(tickers)) + "_" + method
    return hashlib.md5(key.encode()).hexdigest()[:16]


# ─── Persistence ─────────────────────────────────────────────────────────────

def upsert_portfolio_returns(
    summary: Dict,
    engine=None,
) -> int:
    """Persist portfolio return time series to PostgreSQL."""
    if engine is None:
        engine = get_db_engine()

    portfolio_id = make_portfolio_id(summary["tickers"], summary["method"])
    port_ret     = summary["portfolio_returns"]
    bench_ret    = summary["benchmark_returns"]
    dd           = summary["drawdown"]
    r_sharpe     = summary["rolling_sharpe"]
    r_vol        = summary["rolling_vol"]

    combined = pd.DataFrame({
        "portfolio_return":  port_ret,
        "benchmark_return":  bench_ret,
        "drawdown":          dd,
        "rolling_sharpe":    r_sharpe,
        "rolling_vol":       r_vol,
    }).dropna(subset=["portfolio_return"])

    combined["excess_return"]         = combined["portfolio_return"] - combined["benchmark_return"]
    combined["cumulative_portfolio"]  = (1 + combined["portfolio_return"].fillna(0)).cumprod() - 1
    combined["cumulative_benchmark"]  = (1 + combined["benchmark_return"].fillna(0)).cumprod() - 1

    def _f(v):
        """Convert numpy float to plain Python float; NaN → None."""
        if v is None:
            return None
        try:
            f = float(v)
            return None if f != f else f   # NaN check
        except (TypeError, ValueError):
            return None

    records = []
    for dt, row in combined.iterrows():
        records.append({
            "portfolio_id":          portfolio_id,
            "date":                  dt.date() if hasattr(dt, "date") else dt,
            "portfolio_return":      _f(row.get("portfolio_return")),
            "benchmark_return":      _f(row.get("benchmark_return")),
            "excess_return":         _f(row.get("excess_return")),
            "cumulative_portfolio":  _f(row.get("cumulative_portfolio")),
            "cumulative_benchmark":  _f(row.get("cumulative_benchmark")),
            "rolling_sharpe":        _f(row.get("rolling_sharpe")),
            "rolling_vol":           _f(row.get("rolling_vol")),
            "drawdown":              _f(row.get("drawdown")),
            "weighting_method":      summary["method"],
            "tickers":               json.dumps(summary["tickers"]),
            "weights":               json.dumps(
                {k: round(float(v), 6) for k, v in summary["weights"].items()}
            ),
        })

    if not records:
        return 0

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO portfolio_returns
                    (portfolio_id, date, portfolio_return, benchmark_return,
                     excess_return, cumulative_portfolio, cumulative_benchmark,
                     rolling_sharpe, rolling_vol, drawdown,
                     weighting_method, tickers, weights)
                VALUES
                    (:portfolio_id, :date, :portfolio_return, :benchmark_return,
                     :excess_return, :cumulative_portfolio, :cumulative_benchmark,
                     :rolling_sharpe, :rolling_vol, :drawdown,
                     :weighting_method, CAST(:tickers AS jsonb), CAST(:weights AS jsonb))
                ON CONFLICT (portfolio_id, date) DO UPDATE SET
                    portfolio_return = EXCLUDED.portfolio_return,
                    benchmark_return = EXCLUDED.benchmark_return,
                    excess_return    = EXCLUDED.excess_return,
                    rolling_sharpe   = EXCLUDED.rolling_sharpe,
                    rolling_vol      = EXCLUDED.rolling_vol,
                    drawdown         = EXCLUDED.drawdown
            """),
            records,
        )

    logger.info(f"✅ Upserted {len(records)} portfolio return rows (id={portfolio_id})")
    return len(records)


def load_portfolio_returns(
    portfolio_id: str,
    engine=None,
) -> pd.DataFrame:
    if engine is None:
        engine = get_db_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT * FROM portfolio_returns WHERE portfolio_id = :pid ORDER BY date"),
            conn, params={"pid": portfolio_id},
        )
    return df