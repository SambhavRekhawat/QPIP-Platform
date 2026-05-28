"""
data_engine/market_data.py
--------------------------
Pulls historical OHLCV data from yfinance, computes daily/log returns,
rolling volatility, and persists everything to PostgreSQL.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from loguru import logger
from sqlalchemy import text
from typing import List, Optional, Dict

from config import (
    NSE_SUFFIX, BENCHMARK_TICKER, DATA_LOOKBACK_YEARS,
    TRADING_DAYS_YEAR, RISK_FREE_RATE
)
from database.connection import get_db_engine


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _to_yf_ticker(ticker: str) -> str:
    """Convert NSE ticker (e.g. RELIANCE) to yfinance format (RELIANCE.NS)."""
    if ticker.startswith("^"):
        return ticker
    if ticker.endswith(NSE_SUFFIX):
        return ticker
    return ticker + NSE_SUFFIX


def get_date_range(years: int = DATA_LOOKBACK_YEARS):
    end   = datetime.today()
    start = end - timedelta(days=365 * years + 30)   # small buffer
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# ─── Price Download ──────────────────────────────────────────────────────────

def fetch_ohlcv(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    years: int = DATA_LOOKBACK_YEARS,
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV for a list of NSE tickers.
    Returns a dict {ticker: DataFrame} with columns
    [open, high, low, close, adj_close, volume].
    """
    if start is None or end is None:
        start, end = get_date_range(years)

    yf_tickers = [_to_yf_ticker(t) for t in tickers]
    logger.info(f"Downloading OHLCV for {len(tickers)} tickers: {start} → {end}")

    raw = yf.download(
        yf_tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    result: Dict[str, pd.DataFrame] = {}
    for ticker, yft in zip(tickers, yf_tickers):
        try:
            if len(yf_tickers) == 1:
                df = raw.copy()
            else:
                df = raw.xs(yft, axis=1, level=1).copy()

            # Handle both flat and MultiIndex column names from yfinance
            cols = []
            for c in df.columns:
                if isinstance(c, tuple):
                    c = c[0]   # take the field name, e.g. ('Close', '^NSEI') → 'Close'
                cols.append(str(c).lower().replace(" ", "_"))
            df.columns = cols
            df.rename(columns={"adj_close": "adj_close"}, inplace=True)
            df.index = pd.to_datetime(df.index)
            df.dropna(subset=["close"], inplace=True)
            result[ticker] = df
            logger.debug(f"  {ticker}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"  {ticker}: failed to parse — {e}")

    return result


def fetch_benchmark(
    start: Optional[str] = None,
    end: Optional[str] = None,
    years: int = DATA_LOOKBACK_YEARS,
) -> pd.DataFrame:
    """Download NIFTY 50 benchmark data."""
    if start is None or end is None:
        start, end = get_date_range(years)
    data = fetch_ohlcv([BENCHMARK_TICKER], start=start, end=end)
    return data.get(BENCHMARK_TICKER, pd.DataFrame())


# ─── Returns Computation ─────────────────────────────────────────────────────

def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a price DataFrame, compute:
      - daily_return (simple pct change)
      - log_return
      - rolling_vol_21d
      - rolling_vol_63d
      - cumulative_return
    """
    df = price_df.copy()
    df["daily_return"]    = df["close"].pct_change()
    df["log_return"]      = np.log(df["close"] / df["close"].shift(1))
    df["rolling_vol_21d"] = df["daily_return"].rolling(21).std() * np.sqrt(TRADING_DAYS_YEAR)
    df["rolling_vol_63d"] = df["daily_return"].rolling(63).std() * np.sqrt(TRADING_DAYS_YEAR)
    df["cumulative_return"] = (1 + df["daily_return"].fillna(0)).cumprod() - 1
    return df


def compute_excess_returns(
    ticker_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """Align ticker and benchmark returns and compute excess returns."""
    combined = pd.DataFrame({
        "daily_return":    ticker_returns,
        "benchmark_return": benchmark_returns,
    }).dropna()
    combined["excess_return"] = combined["daily_return"] - combined["benchmark_return"]
    return combined


# ─── Persistence ─────────────────────────────────────────────────────────────

def _to_float(val) -> float:
    """Convert numpy float64 (or any numeric) to plain Python float for PostgreSQL."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (f != f) else f   # converts NaN → None
    except (TypeError, ValueError):
        return None


def upsert_prices(ticker: str, df: pd.DataFrame, engine=None) -> int:
    """
    Upsert OHLCV rows into the `prices` table.
    Returns number of rows written.
    """
    if engine is None:
        engine = get_db_engine()

    records = []
    for date, row in df.iterrows():
        records.append({
            "ticker":    ticker,
            "date":      date.date(),
            "open":      _to_float(row.get("open")),
            "high":      _to_float(row.get("high")),
            "low":       _to_float(row.get("low")),
            "close":     _to_float(row.get("close")),
            "adj_close": _to_float(row.get("adj_close")),
            "volume":    _to_float(row.get("volume")),
        })

    if not records:
        return 0

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO prices (ticker, date, open, high, low, close, adj_close, volume)
                VALUES (:ticker, :date, :open, :high, :low, :close, :adj_close, :volume)
                ON CONFLICT (ticker, date) DO UPDATE SET
                    open      = EXCLUDED.open,
                    high      = EXCLUDED.high,
                    low       = EXCLUDED.low,
                    close     = EXCLUDED.close,
                    adj_close = EXCLUDED.adj_close,
                    volume    = EXCLUDED.volume,
                    updated_at = NOW()
            """),
            records,
        )

    logger.info(f"  ✅ Upserted {len(records)} price rows for {ticker}")
    return len(records)


def upsert_returns(ticker: str, df: pd.DataFrame, engine=None) -> int:
    """Upsert computed returns into the `returns` table."""
    if engine is None:
        engine = get_db_engine()

    records = []
    for date, row in df.iterrows():
        records.append({
            "ticker":            ticker,
            "date":              date.date(),
            "daily_return":      _to_float(row.get("daily_return")),
            "log_return":        _to_float(row.get("log_return")),
            "rolling_vol_21d":   _to_float(row.get("rolling_vol_21d")),
            "rolling_vol_63d":   _to_float(row.get("rolling_vol_63d")),
            "cumulative_return": _to_float(row.get("cumulative_return")),
            "benchmark_return":  _to_float(row.get("benchmark_return")),
            "excess_return":     _to_float(row.get("excess_return")),
        })

    if not records:
        return 0

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO returns
                    (ticker, date, daily_return, log_return,
                     rolling_vol_21d, rolling_vol_63d, cumulative_return,
                     benchmark_return, excess_return)
                VALUES
                    (:ticker, :date, :daily_return, :log_return,
                     :rolling_vol_21d, :rolling_vol_63d, :cumulative_return,
                     :benchmark_return, :excess_return)
                ON CONFLICT (ticker, date) DO UPDATE SET
                    daily_return     = EXCLUDED.daily_return,
                    log_return       = EXCLUDED.log_return,
                    rolling_vol_21d  = EXCLUDED.rolling_vol_21d,
                    rolling_vol_63d  = EXCLUDED.rolling_vol_63d,
                    cumulative_return = EXCLUDED.cumulative_return,
                    benchmark_return = EXCLUDED.benchmark_return,
                    excess_return    = EXCLUDED.excess_return
            """),
            records,
        )
    return len(records)


# ─── ETL Pipeline ────────────────────────────────────────────────────────────

def run_market_data_pipeline(
    tickers: List[str],
    years: int = DATA_LOOKBACK_YEARS,
    engine=None,
) -> Dict[str, pd.DataFrame]:
    """
    Full ETL: download → compute → store.
    Returns a dict of {ticker: enriched_df} for downstream use.
    """
    if engine is None:
        engine = get_db_engine()

    start, end = get_date_range(years)

    # Fetch benchmark once
    logger.info("Fetching benchmark (NIFTY 50)…")
    bench_raw    = fetch_benchmark(start=start, end=end)
    bench_with_r = compute_returns(bench_raw) if not bench_raw.empty else pd.DataFrame()
    bench_returns = bench_with_r["daily_return"] if not bench_with_r.empty else pd.Series(dtype=float)
    upsert_prices(BENCHMARK_TICKER, bench_raw, engine)

    # Fetch tickers
    all_data = fetch_ohlcv(tickers, start=start, end=end)
    enriched: Dict[str, pd.DataFrame] = {}

    for ticker, price_df in all_data.items():
        if price_df.empty:
            logger.warning(f"No data for {ticker}, skipping.")
            continue

        # Prices → DB
        upsert_prices(ticker, price_df, engine)

        # Returns computation
        ret_df = compute_returns(price_df)

        # Merge benchmark returns
        if not bench_returns.empty:
            exc = compute_excess_returns(ret_df["daily_return"], bench_returns)
            ret_df["benchmark_return"] = exc["benchmark_return"].reindex(ret_df.index)
            ret_df["excess_return"]    = exc["excess_return"].reindex(ret_df.index)

        # Returns → DB
        upsert_returns(ticker, ret_df, engine)
        enriched[ticker] = ret_df

    logger.info(f"✅ Market data pipeline complete for {len(enriched)} tickers.")
    return enriched


# ─── Data Readers ────────────────────────────────────────────────────────────

def load_prices_from_db(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    engine=None,
) -> Dict[str, pd.DataFrame]:
    """Read price data from PostgreSQL and return as dict of DataFrames."""
    if engine is None:
        engine = get_db_engine()

    result = {}
    for ticker in tickers:
        query = "SELECT * FROM prices WHERE ticker = :ticker"
        params: dict = {"ticker": ticker}
        if start:
            query += " AND date >= :start"
            params["start"] = start
        if end:
            query += " AND date <= :end"
            params["end"] = end
        query += " ORDER BY date ASC"

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params, index_col="date", parse_dates=["date"])
        df.index = pd.to_datetime(df.index)
        result[ticker] = df

    return result


def load_close_matrix(
    tickers: List[str],
    start: Optional[str] = None,
    engine=None,
) -> pd.DataFrame:
    """Return a wide DataFrame of adjusted closing prices (columns = tickers)."""
    all_prices = load_prices_from_db(tickers, start=start, engine=engine)
    frames = {}
    for t, df in all_prices.items():
        if df.empty:
            continue
        col = df["close"]
        # Guard: squeeze single-column DataFrames to Series
        if isinstance(col, pd.DataFrame):
            col = col.squeeze(axis=1)
        frames[t] = col.rename(t)
    result = pd.DataFrame(frames).sort_index()
    # Drop duplicate columns — can happen when yfinance returns aliased tickers
    result = result.loc[:, ~result.columns.duplicated(keep="first")]
    # Ensure every column is plain float64
    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors="coerce").astype("float64")
    return result


def load_returns_matrix(
    tickers: List[str],
    start: Optional[str] = None,
    engine=None,
) -> pd.DataFrame:
    """Return a wide DataFrame of daily returns (columns = tickers)."""
    closes = load_close_matrix(tickers, start=start, engine=engine)
    return closes.pct_change().dropna()