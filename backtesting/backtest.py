"""
backtesting/backtest.py
------------------------
Walk-forward backtesting engine with:
  - Periodic rebalancing (monthly default)
  - Transaction cost modelling
  - Full performance attribution
  - Persistence to PostgreSQL
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from datetime import date, timedelta
from loguru import logger
from sqlalchemy import text
from typing import Dict, List, Optional, Callable

from config import (
    BACKTEST_MONTHS, REBALANCE_FREQUENCY, TRANSACTION_COST_BPS,
    TRADING_DAYS_YEAR, BENCHMARK_TICKER, RISK_FREE_RATE
)
from analytics.risk_metrics import (
    annualised_return, annualised_volatility, sharpe_ratio,
    sortino_ratio, max_drawdown, beta, alpha, information_ratio,
    hit_ratio, var_historical
)
from database.connection import get_db_engine


# ─── Core Backtest Engine ─────────────────────────────────────────────────────

def run_backtest(
    tickers: List[str],
    returns_matrix: pd.DataFrame,
    benchmark_returns: pd.Series,
    weighting_method: str = "Equal Weighted",
    market_caps: Optional[Dict[str, float]] = None,
    custom_weights: Optional[Dict[str, float]] = None,
    rebalance_freq: str = REBALANCE_FREQUENCY,
    txn_cost_bps: float = TRANSACTION_COST_BPS,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    strategy_name: str = "Portfolio Strategy",
) -> Dict:
    """
    Walk-forward backtest over the returns matrix.

    Parameters
    ----------
    tickers          : list of NSE ticker strings
    returns_matrix   : wide DataFrame of daily returns (index=date, cols=tickers)
    benchmark_returns: NIFTY 50 daily returns aligned to same index
    weighting_method : "Equal Weighted" | "Market Cap Weighted" | "Custom Weighted"
    txn_cost_bps     : one-way transaction cost in basis points
    """
    from analytics.portfolio import compute_weights

    # ── Date filtering ──────────────────────────────────────────────────────
    rm = returns_matrix.copy()
    if start_date:
        rm = rm[rm.index >= pd.to_datetime(start_date)]
    if end_date:
        rm = rm[rm.index <= pd.to_datetime(end_date)]

    available_tickers = [t for t in tickers if t in rm.columns]
    rm = rm[available_tickers].dropna(how="all")

    bench = benchmark_returns.reindex(rm.index).fillna(0)

    if rm.empty or len(available_tickers) == 0:
        logger.warning("Backtest: no data available in selected date range.")
        return {}

    # ── Rebalance dates ──────────────────────────────────────────────────────
    # Group by rebalance_freq to get period-end dates
    rebalance_dates = rm.resample(rebalance_freq).last().index.tolist()

    # ── Walk-forward simulation ──────────────────────────────────────────────
    portfolio_values   = [100.0]
    benchmark_values   = [100.0]
    daily_port_returns: List[float] = []
    daily_bench_returns: List[float] = []
    rebalance_log = []
    txn_costs_total = 0.0

    current_weights = None
    rebalance_idx   = 0

    all_dates = rm.index.tolist()

    for i, dt in enumerate(all_dates):
        # ── Check if we should rebalance ───────────────────────────────────
        if (rebalance_idx < len(rebalance_dates) and
                dt >= rebalance_dates[rebalance_idx]):

            # Compute new weights using data available up to this point
            lookback_rm = rm.iloc[max(0, i - 252):i]

            new_weights = compute_weights(
                available_tickers,
                weighting_method,
                market_caps=market_caps,
                custom_weights=custom_weights,
            )

            # Transaction cost: sum of |weight change| * cost
            if current_weights is not None:
                turnover = sum(
                    abs(new_weights.get(t, 0) - current_weights.get(t, 0))
                    for t in available_tickers
                )
                cost = turnover * txn_cost_bps / 10_000
                txn_costs_total += cost
                # Apply cost as return drag
                portfolio_values[-1] *= (1 - cost)

            current_weights = new_weights
            rebalance_log.append({
                "date":    str(dt.date()),
                "weights": {k: round(v, 4) for k, v in current_weights.items()},
            })
            rebalance_idx += 1

        if current_weights is None:
            # Before first rebalance: use equal weights
            current_weights = {t: 1.0 / len(available_tickers)
                               for t in available_tickers}

        # ── Portfolio return for day i ─────────────────────────────────────
        row = rm.iloc[i]
        port_ret = sum(
            current_weights.get(t, 0) * (row.get(t, 0) or 0)
            for t in available_tickers
        )
        bench_ret = float(bench.iloc[i]) if i < len(bench) else 0.0

        daily_port_returns.append(port_ret)
        daily_bench_returns.append(bench_ret)

        new_pv = portfolio_values[-1] * (1 + port_ret)
        new_bv = benchmark_values[-1] * (1 + bench_ret)
        portfolio_values.append(new_pv)
        benchmark_values.append(new_bv)

    # Drop the initial 100 seed
    portfolio_values  = portfolio_values[1:]
    benchmark_values  = benchmark_values[1:]

    port_series  = pd.Series(daily_port_returns, index=all_dates, name="portfolio")
    bench_series = pd.Series(daily_bench_returns, index=all_dates, name="benchmark")

    # ── Monthly returns ──────────────────────────────────────────────────────
    monthly_port = port_series.resample("ME").apply(lambda r: (1 + r).prod() - 1)
    monthly_bench = bench_series.resample("ME").apply(lambda r: (1 + r).prod() - 1)
    monthly_excess = monthly_port - monthly_bench

    # ── Metrics ──────────────────────────────────────────────────────────────
    total_return   = (portfolio_values[-1] / 100.0) - 1
    bench_total    = (benchmark_values[-1] / 100.0) - 1
    cagr           = annualised_return(port_series)
    vol            = annualised_volatility(port_series)
    sharpe         = sharpe_ratio(port_series)
    sortino        = sortino_ratio(port_series)
    mdd            = max_drawdown(port_series)
    b_              = beta(port_series, bench_series)
    a_              = alpha(port_series, bench_series)
    ir             = information_ratio(port_series, bench_series)
    hr             = hit_ratio(port_series, bench_series)

    results = {
        "strategy_name":      strategy_name,
        "tickers":            available_tickers,
        "weighting_method":   weighting_method,
        "start_date":         str(all_dates[0].date()),
        "end_date":           str(all_dates[-1].date()),
        "num_rebalances":     len(rebalance_log),
        "transaction_costs":  round(txn_costs_total, 6),
        "total_return":       round(total_return, 4),
        "benchmark_return":   round(bench_total, 4),
        "excess_return":      round(total_return - bench_total, 4),
        "cagr":               round(cagr, 4),
        "volatility":         round(vol, 4),
        "sharpe_ratio":       round(sharpe, 4),
        "sortino_ratio":      round(sortino, 4),
        "max_drawdown":       round(mdd, 4),
        "beta":               round(b_, 4),
        "alpha":              round(a_, 4),
        "information_ratio":  round(ir, 4),
        "hit_ratio":          round(hr, 4),
        "portfolio_nav":      pd.Series(portfolio_values, index=all_dates),
        "benchmark_nav":      pd.Series(benchmark_values, index=all_dates),
        "daily_returns":      port_series,
        "benchmark_daily":    bench_series,
        "monthly_returns":    {str(k.date()): round(v, 6)
                               for k, v in monthly_port.items()},
        "monthly_excess":     {str(k.date()): round(v, 6)
                               for k, v in monthly_excess.items()},
        "rebalance_log":      rebalance_log,
    }

    logger.info(
        f"✅ Backtest complete | CAGR={cagr:.1%} | Sharpe={sharpe:.2f} | "
        f"MDD={mdd:.1%} | Alpha={a_:.1%}"
    )
    return results


# ─── Multi-Strategy Comparison ───────────────────────────────────────────────

def run_strategy_comparison(
    tickers: List[str],
    returns_matrix: pd.DataFrame,
    benchmark_returns: pd.Series,
    market_caps: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Run backtest for all three weighting methods and compare results.
    """
    strategies = ["Equal Weighted", "Market Cap Weighted"]
    if market_caps:
        strategies.append("Market Cap Weighted")

    rows = []
    for method in strategies:
        result = run_backtest(
            tickers, returns_matrix, benchmark_returns,
            weighting_method=method,
            market_caps=market_caps,
            strategy_name=method,
        )
        if result:
            rows.append({
                "Strategy":         result["strategy_name"],
                "CAGR":             f"{result['cagr']:.1%}",
                "Volatility":       f"{result['volatility']:.1%}",
                "Sharpe Ratio":     f"{result['sharpe_ratio']:.2f}",
                "Sortino Ratio":    f"{result['sortino_ratio']:.2f}",
                "Max Drawdown":     f"{result['max_drawdown']:.1%}",
                "Alpha":            f"{result['alpha']:.1%}",
                "Beta":             f"{result['beta']:.2f}",
                "Info Ratio":       f"{result['information_ratio']:.2f}",
                "Hit Ratio":        f"{result['hit_ratio']:.1%}",
                "Total Return":     f"{result['total_return']:.1%}",
                "vs Benchmark":     f"{result['excess_return']:+.1%}",
            })

    return pd.DataFrame(rows)


# ─── Persistence ─────────────────────────────────────────────────────────────

def save_backtest_result(result: Dict, engine=None) -> int:
    if engine is None:
        engine = get_db_engine()
    if not result:
        return 0

    record = {
        "strategy_name":    result.get("strategy_name"),
        "start_date":       result.get("start_date"),
        "end_date":         result.get("end_date"),
        "tickers":          json.dumps(result.get("tickers", [])),
        "weighting_method": result.get("weighting_method"),
        "cagr":             result.get("cagr"),
        "sharpe_ratio":     result.get("sharpe_ratio"),
        "sortino_ratio":    result.get("sortino_ratio"),
        "information_ratio": result.get("information_ratio"),
        "alpha":            result.get("alpha"),
        "beta":             result.get("beta"),
        "max_drawdown":     result.get("max_drawdown"),
        "hit_ratio":        result.get("hit_ratio"),
        "total_return":     result.get("total_return"),
        "benchmark_return": result.get("benchmark_return"),
        "excess_return":    result.get("excess_return"),
        "volatility":       result.get("volatility"),
        "num_rebalances":   result.get("num_rebalances"),
        "transaction_costs": result.get("transaction_costs"),
        "monthly_returns":  json.dumps(result.get("monthly_returns", {})),
        "config_params":    json.dumps({
            "rebalance_freq":  REBALANCE_FREQUENCY,
            "txn_cost_bps":    TRANSACTION_COST_BPS,
        }),
    }

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO backtest_results
                    (strategy_name, start_date, end_date, tickers,
                     weighting_method, cagr, sharpe_ratio, sortino_ratio,
                     information_ratio, alpha, beta, max_drawdown, hit_ratio,
                     total_return, benchmark_return, excess_return, volatility,
                     num_rebalances, transaction_costs, monthly_returns, config_params)
                VALUES
                    (:strategy_name, :start_date, :end_date, CAST(:tickers AS jsonb),
                     :weighting_method, :cagr, :sharpe_ratio, :sortino_ratio,
                     :information_ratio, :alpha, :beta, :max_drawdown, :hit_ratio,
                     :total_return, :benchmark_return, :excess_return, :volatility,
                     :num_rebalances, :transaction_costs, CAST(:monthly_returns AS jsonb),
                     CAST(:config_params AS jsonb))
            """),
            record,
        )

    logger.info("✅ Backtest result saved to DB")
    return 1