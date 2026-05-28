"""
analytics/liquidity_risk.py
----------------------------
Liquidity-adjusted risk analytics engine.

Computes:
  - Liquidity score per ticker using average daily volume and volatility
  - Liquidity-adjusted VaR (adds liquidation cost to market VaR)
  - Estimated liquidation horizon (days to exit position cleanly)
  - Concentration risk (HHI index + alerts)
  - Amihud illiquidity ratio

Uses volume data already stored in the `prices` PostgreSQL table.
Does NOT duplicate anything in risk_metrics.py or stress_testing.py.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger
from sqlalchemy import text

from config import TRADING_DAYS_YEAR, RISK_FREE_RATE
from database.connection import get_db_engine


# ─── Volume & Price Data Loader ───────────────────────────────────────────────

def load_volume_data(
    tickers: List[str],
    window: int = 63,
    engine=None,
) -> pd.DataFrame:
    """
    Load daily volume and close price from PostgreSQL `prices` table.
    Returns a wide DataFrame indexed by date.
    """
    if engine is None:
        engine = get_db_engine()

    placeholders = ", ".join([f"'{t}'" for t in tickers])
    query = f"""
        SELECT ticker, date, close, volume
        FROM   prices
        WHERE  ticker IN ({placeholders})
        ORDER  BY date ASC
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, parse_dates=["date"])

    if df.empty:
        return pd.DataFrame()

    # Pivot to wide format for easy slicing
    volume = df.pivot(index="date", columns="ticker", values="volume")
    closes = df.pivot(index="date", columns="ticker", values="close")

    return volume.tail(window), closes.tail(window)


# ─── 1. Average Daily Volume & ADV Ratio ──────────────────────────────────────

def average_daily_volume(
    volume_df: pd.DataFrame,
    window: int = 21,
) -> pd.Series:
    """20-day average daily volume per ticker (in shares)."""
    return volume_df.tail(window).mean()


def daily_traded_value(
    volume_df: pd.DataFrame,
    closes_df: pd.DataFrame,
) -> pd.DataFrame:
    """Daily traded value in ₹ (volume × close price)."""
    common_dates = volume_df.index.intersection(closes_df.index)
    return volume_df.loc[common_dates] * closes_df.loc[common_dates]


# ─── 2. Amihud Illiquidity Ratio ─────────────────────────────────────────────

def amihud_illiquidity(
    returns_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    closes_df: pd.DataFrame,
    window: int = 63,
) -> pd.Series:
    """
    Amihud (2002) illiquidity ratio:
        ILLIQ = mean(|r_t| / Volume_t × Close_t)

    Higher = more illiquid. Units: (% return) / (₹ volume)
    Normalised to 0–1 scale for display.
    """
    dtv     = daily_traded_value(volume_df.tail(window), closes_df.tail(window))
    ret_abs = returns_df.tail(window).abs()

    common_tickers = ret_abs.columns.intersection(dtv.columns)
    common_dates   = ret_abs.index.intersection(dtv.index)

    ratio = ret_abs.loc[common_dates, common_tickers] / dtv.loc[common_dates, common_tickers].replace(0, np.nan)
    illiq = ratio.mean() * 1e6   # scale for readability

    # Normalise 0-1
    if illiq.max() > illiq.min():
        illiq_norm = (illiq - illiq.min()) / (illiq.max() - illiq.min())
    else:
        illiq_norm = pd.Series(0.5, index=illiq.index)

    return illiq_norm.rename("amihud_illiquidity")


# ─── 3. Liquidation Horizon ───────────────────────────────────────────────────

def liquidation_horizon(
    weights: Dict[str, float],
    portfolio_value: float,
    volume_df: pd.DataFrame,
    closes_df: pd.DataFrame,
    adv_participation: float = 0.20,   # assume max 20% of ADV per day
) -> pd.DataFrame:
    """
    Estimate how many trading days it would take to exit each position
    assuming we can trade `adv_participation` × ADV per day without
    significantly moving the market.

    Parameters
    ----------
    adv_participation : fraction of average daily volume we can trade
                        without excessive market impact (standard: 10–20%)
    """
    adv = average_daily_volume(volume_df, window=21)
    last_close = closes_df.iloc[-1]

    rows = []
    for ticker, weight in weights.items():
        if ticker not in adv.index or ticker not in last_close.index:
            continue

        position_value  = abs(weight) * portfolio_value
        price           = float(last_close[ticker]) if last_close[ticker] > 0 else 1.0
        shares_held     = position_value / price
        daily_tradeable = float(adv[ticker]) * adv_participation
        daily_value     = daily_tradeable * price

        if daily_value > 0:
            horizon_days = max(1, int(np.ceil(position_value / daily_value)))
        else:
            horizon_days = 999   # effectively illiquid

        rows.append({
            "Ticker":               ticker,
            "Weight (%)":           round(weight * 100, 2),
            "Position Value (₹Cr)": round(position_value / 1e7, 2),
            "ADV (shares)":         int(adv.get(ticker, 0)),
            "Liquidation Days":     horizon_days,
            "Risk Level":           "🔴 High" if horizon_days > 5
                                    else ("🟡 Medium" if horizon_days > 2 else "🟢 Low"),
        })

    df = pd.DataFrame(rows).sort_values("Liquidation Days", ascending=False)
    return df


# ─── 4. Concentration Risk ────────────────────────────────────────────────────

def herfindahl_index(weights: Dict[str, float]) -> float:
    """
    Herfindahl-Hirschman Index (HHI) measures concentration.
    HHI = Σ w_i²  (range 1/n to 1.0)
    HHI = 1/n means perfectly diversified.
    HHI = 1.0 means all in one stock.
    """
    w = np.array(list(weights.values()))
    w = w / w.sum()
    return float(np.sum(w ** 2))


def effective_n(weights: Dict[str, float]) -> float:
    """
    Effective number of bets = 1 / HHI.
    A portfolio with 10 equal weights has effective_n = 10.
    A concentrated portfolio has effective_n closer to 1.
    """
    return 1.0 / herfindahl_index(weights)


def concentration_alerts(
    weights: Dict[str, float],
    threshold_single: float = 0.30,
    threshold_top3: float = 0.70,
) -> List[Dict]:
    """
    Generate concentration alerts:
      - Any single position > 30%
      - Top 3 positions > 70% combined
    """
    w = {t: abs(v) for t, v in weights.items()}
    total = sum(w.values())
    w_norm = {t: v / total for t, v in w.items()}

    alerts = []
    sorted_w = sorted(w_norm.items(), key=lambda x: -x[1])

    for ticker, weight in sorted_w:
        if weight > threshold_single:
            alerts.append({
                "level":   "⚠️ WARNING",
                "message": f"{ticker} is {weight:.1%} of portfolio — exceeds {threshold_single:.0%} single-stock limit",
                "ticker":  ticker,
            })

    top3 = sum(w for _, w in sorted_w[:3])
    if top3 > threshold_top3:
        alerts.append({
            "level":   "⚠️ WARNING",
            "message": f"Top 3 positions = {top3:.1%} — portfolio is highly concentrated",
            "ticker":  "Top 3",
        })

    hhi = herfindahl_index(weights)
    en  = effective_n(weights)
    if en < len(weights) * 0.5:
        alerts.append({
            "level":   "ℹ️ INFO",
            "message": f"Effective diversification = {en:.1f} stocks (HHI={hhi:.3f}). "
                       f"Full diversification would give {len(weights):.0f}.",
            "ticker":  "Portfolio",
        })

    return alerts


# ─── 5. Liquidity-Adjusted VaR ────────────────────────────────────────────────

def liquidity_adjusted_var(
    weights: Dict[str, float],
    returns_matrix: pd.DataFrame,
    volume_df: pd.DataFrame,
    closes_df: pd.DataFrame,
    portfolio_value: float = 100.0,
    confidence: float = 0.99,
    adv_participation: float = 0.20,
    bid_ask_spread_bps: float = 10.0,   # assumed bid-ask spread in bps
) -> Dict:
    """
    Liquidity-Adjusted VaR (LVaR) = Market VaR + Liquidity Cost.

    Liquidity cost has two components:
      1. Spread cost: bid-ask spread × position size
      2. Market impact: liquidation cost over the holding horizon

    Reference: Bangia et al. (1999), BIS Paper on Liquidity Risk.
    """
    from analytics.risk_metrics import var_historical, cvar_historical

    tickers    = [t for t in weights if t in returns_matrix.columns]
    w          = np.array([weights[t] for t in tickers])
    w         /= w.sum()

    # Market VaR from historical simulation
    port_returns = returns_matrix[tickers].tail(252).fillna(0) @ w
    mvar  = float(var_historical(port_returns, confidence))
    mcvar = float(cvar_historical(port_returns, confidence))

    # Liquidity horizon per ticker
    liq_df  = liquidation_horizon(weights, portfolio_value, volume_df, closes_df, adv_participation)
    max_liq = int(liq_df["Liquidation Days"].max()) if not liq_df.empty else 1

    # Scale VaR to liquidation horizon (square root of time rule)
    horizon_var  = mvar  * np.sqrt(max_liq)
    horizon_cvar = mcvar * np.sqrt(max_liq)

    # Spread cost: bid-ask spread × weighted position
    spread_cost = bid_ask_spread_bps / 10_000 / 2   # one-way half-spread
    total_spread_cost = sum(
        abs(weights.get(t, 0)) * spread_cost for t in tickers
    )

    # Liquidity-adjusted VaR
    lvar  = horizon_var  + total_spread_cost
    lcvar = horizon_cvar + total_spread_cost

    return {
        "market_var_1d":         mvar,
        "market_cvar_1d":        mcvar,
        "liquidity_horizon_days": max_liq,
        "horizon_var":           horizon_var,
        "horizon_cvar":          horizon_cvar,
        "spread_cost":           total_spread_cost,
        "liquidity_adjusted_var":  lvar,
        "liquidity_adjusted_cvar": lcvar,
        "lvar_premium":          lvar - mvar,
        "confidence":            confidence,
        "liquidation_detail":    liq_df,
    }


# ─── 6. Full Liquidity Report ─────────────────────────────────────────────────

def full_liquidity_report(
    weights: Dict[str, float],
    returns_matrix: pd.DataFrame,
    portfolio_value: float = 100.0,
    engine=None,
) -> Dict:
    """
    Master function: compute the complete liquidity risk profile.
    Loads volume data from PostgreSQL and returns all metrics.
    """
    if engine is None:
        engine = get_db_engine()

    tickers = list(weights.keys())

    try:
        volume_df, closes_df = load_volume_data(tickers, window=63, engine=engine)
    except Exception as e:
        logger.warning(f"  Could not load volume data from DB: {e}")
        # Return degraded report with just concentration metrics
        return {
            "error":         str(e),
            "hhi":           herfindahl_index(weights),
            "effective_n":   effective_n(weights),
            "alerts":        concentration_alerts(weights),
            "liq_detail":    pd.DataFrame(),
            "lvar":          None,
        }

    hhi    = herfindahl_index(weights)
    eff_n  = effective_n(weights)
    alerts = concentration_alerts(weights)
    liq_df = liquidation_horizon(weights, portfolio_value, volume_df, closes_df)

    # Amihud ratio
    try:
        rets_window = returns_matrix[tickers].tail(63)
        amihud      = amihud_illiquidity(rets_window, volume_df, closes_df)
    except Exception:
        amihud = pd.Series(dtype=float)

    # LVaR
    try:
        lvar_result = liquidity_adjusted_var(
            weights, returns_matrix, volume_df, closes_df, portfolio_value
        )
    except Exception as e:
        logger.warning(f"  LVaR computation failed: {e}")
        lvar_result = None

    return {
        "hhi":           hhi,
        "effective_n":   eff_n,
        "alerts":        alerts,
        "liq_detail":    liq_df,
        "amihud":        amihud,
        "lvar":          lvar_result,
        "portfolio_value": portfolio_value,
    }
