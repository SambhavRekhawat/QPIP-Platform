"""
export_data.py
--------------
Exports all computed results from PostgreSQL to lightweight files
in the data/ folder so Streamlit Community Cloud can serve the
dashboard without needing a database connection.

Run this after every pipeline update:
    python export_data.py

Then push the data/ folder to GitHub:
    git add data/
    git commit -m "Data refresh -- YYYY-MM-DD"
    git push

Streamlit Cloud redeploys automatically within ~30 seconds.
"""

import os
import sys
import json
import shutil
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─── Output directory ────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _save_parquet(df: pd.DataFrame, filename: str) -> int:
    """Save a DataFrame as Parquet. Returns row count."""
    if df is None or df.empty:
        logger.warning(f"  ⚠️  {filename} — empty, skipping")
        return 0
    path = DATA_DIR / filename
    df.to_parquet(path, index=True, compression="snappy")
    size_kb = path.stat().st_size // 1024
    logger.info(f"  ✅ {filename} — {len(df):,} rows, {size_kb} KB")
    return len(df)


def _save_json(data: dict, filename: str) -> None:
    """Save a dict as JSON."""
    path = DATA_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    size_kb = path.stat().st_size // 1024
    logger.info(f"  ✅ {filename} — {size_kb} KB")


def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all columns are JSON/Parquet-safe types.
    Converts numpy types, pd.NA, and object columns.
    """
    for col in df.columns:
        try:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).replace("nan", "").replace("None", "")
            elif pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        except Exception:
            pass
    return df


# ─── Database readers ─────────────────────────────────────────────────────────

def _read_table(engine, table: str, order_by: str = None,
                limit: int = None) -> pd.DataFrame:
    """Read a full table from PostgreSQL into a DataFrame."""
    from sqlalchemy import text
    query = f"SELECT * FROM {table}"
    if order_by:
        query += f" ORDER BY {order_by}"
    if limit:
        query += f" LIMIT {limit}"
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df
    except Exception as e:
        logger.warning(f"  Could not read {table}: {e}")
        return pd.DataFrame()


# ─── Export functions ────────────────────────────────────────────────────────

def export_prices(engine) -> None:
    """Export prices table — last 5 years of OHLCV for all tickers."""
    logger.info("Exporting prices…")
    df = _read_table(engine, "prices",
                     order_by="ticker, date DESC")
    if df.empty:
        return
    # Pivot close prices to wide format for fast dashboard reads
    close_wide = df.pivot_table(
        index="date", columns="ticker", values="close"
    ).sort_index()
    close_wide.index = pd.to_datetime(close_wide.index)
    _save_parquet(close_wide, "close_prices.parquet")

    # Also save volumes
    vol_wide = df.pivot_table(
        index="date", columns="ticker", values="volume"
    ).sort_index()
    vol_wide.index = pd.to_datetime(vol_wide.index)
    _save_parquet(vol_wide, "volumes.parquet")


def export_returns(engine) -> None:
    """Export returns table — daily returns wide format."""
    logger.info("Exporting returns…")
    df = _read_table(engine, "returns", order_by="ticker, date DESC")
    if df.empty:
        return
    # Wide format: index=date, columns=tickers
    ret_wide = df.pivot_table(
        index="date", columns="ticker", values="daily_return"
    ).sort_index()
    ret_wide.index = pd.to_datetime(ret_wide.index)
    _save_parquet(ret_wide, "returns.parquet")

    # Benchmark returns (from any ticker's benchmark_return column)
    bench = df[["date", "benchmark_return"]].drop_duplicates("date").set_index("date")
    bench.index = pd.to_datetime(bench.index)
    bench = bench.sort_index()
    _save_parquet(bench, "benchmark_returns.parquet")


def export_fundamentals(engine) -> None:
    """Export latest fundamentals for all tickers."""
    logger.info("Exporting fundamentals…")
    df = _read_table(engine, "latest_fundamentals")
    if df.empty:
        # Fall back to full table and deduplicate
        df = _read_table(engine, "fundamentals", order_by="ticker, as_of_date DESC")
        if not df.empty:
            df = df.drop_duplicates(subset="ticker", keep="first")
    _save_parquet(_safe_df(df), "fundamentals.parquet")


def export_factors(engine) -> None:
    """Export latest factor scores for all tickers."""
    logger.info("Exporting factors…")
    df = _read_table(engine, "latest_factors")
    if df.empty:
        df = _read_table(engine, "factors", order_by="ticker, date DESC")
        if not df.empty:
            df = df.drop_duplicates(subset="ticker", keep="first")
    _save_parquet(_safe_df(df), "factors.parquet")


def export_signals(engine) -> None:
    """Export latest ML signals for all tickers."""
    logger.info("Exporting signals…")
    df = _read_table(engine, "latest_signals")
    if df.empty:
        df = _read_table(engine, "signals", order_by="ticker, signal_date DESC")
        if not df.empty:
            df = df.drop_duplicates(subset="ticker", keep="first")

    if df.empty:
        return

    # Parse JSONB columns stored as strings
    for col in ["shap_values", "top_positive_drivers", "top_negative_drivers"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else (x or "{}")
            )
    _save_parquet(_safe_df(df), "signals.parquet")


def export_portfolio_returns(engine) -> None:
    """Export portfolio return time series (latest portfolio run)."""
    logger.info("Exporting portfolio returns…")
    df = _read_table(engine, "portfolio_returns",
                     order_by="portfolio_id, date DESC")
    if df.empty:
        return

    # Keep latest portfolio_id (most recent run)
    if "portfolio_id" in df.columns and "created_at" in df.columns:
        latest_id = (
            df.groupby("portfolio_id")["created_at"].max()
            .idxmax()
        )
        df = df[df["portfolio_id"] == latest_id]

    # Parse jsonb columns
    for col in ["tickers", "weights"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (dict, list)) else (x or "[]")
            )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    _save_parquet(_safe_df(df), "portfolio_returns.parquet")


def export_backtest_results(engine) -> None:
    """Export the most recent backtest result."""
    logger.info("Exporting backtest results…")
    df = _read_table(engine, "backtest_results",
                     order_by="created_at DESC",
                     limit=10)
    if df.empty:
        return
    for col in ["tickers", "monthly_returns", "config_params"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (dict, list)) else (x or "{}")
            )
    _save_parquet(_safe_df(df), "backtest_results.parquet")


def export_events(engine) -> None:
    """Export recent events / news sentiment."""
    logger.info("Exporting events…")
    df = _read_table(engine, "events",
                     order_by="event_date DESC",
                     limit=500)
    if df.empty:
        return
    _save_parquet(_safe_df(df), "events.parquet")


def write_manifest(tickers: list, export_start: datetime) -> None:
    """Write a manifest.json with metadata about the export."""
    elapsed = (datetime.now() - export_start).seconds
    manifest = {
        "exported_at":    datetime.now().isoformat(),
        "export_date":    date.today().isoformat(),
        "tickers":        tickers,
        "n_tickers":      len(tickers),
        "elapsed_sec":    elapsed,
        "files": [str(f.name) for f in DATA_DIR.glob("*.parquet")]
        + [str(f.name) for f in DATA_DIR.glob("*.json")
           if f.name != "manifest.json"],
    }
    _save_json(manifest, "manifest.json")
    logger.info(f"  Manifest written — {len(manifest['files'])} files")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_export(verbose: bool = True) -> None:
    """
    Full export pipeline. Reads every relevant table from PostgreSQL
    and saves to data/ folder in Parquet + JSON format.
    """
    from database.connection import get_db_engine, test_connection

    logger.info("=" * 56)
    logger.info("QPIP Data Export — starting")
    logger.info(f"Output directory: {DATA_DIR}")
    logger.info("=" * 56)

    # Check database is reachable
    if not test_connection():
        logger.error("❌ Cannot reach PostgreSQL. Run main.py first to populate the database.")
        sys.exit(1)

    engine = get_db_engine()
    start  = datetime.now()

    # Run all exports
    export_prices(engine)
    export_returns(engine)
    export_fundamentals(engine)
    export_factors(engine)
    export_signals(engine)
    export_portfolio_returns(engine)
    export_backtest_results(engine)
    export_events(engine)

    # Read tickers from the prices table
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            tickers = pd.read_sql(
                text("SELECT DISTINCT ticker FROM prices WHERE ticker != '^NSEI' ORDER BY ticker"),
                conn
            )["ticker"].tolist()
    except Exception:
        tickers = []

    write_manifest(tickers, start)

    # Final summary
    files      = list(DATA_DIR.glob("*.parquet")) + list(DATA_DIR.glob("*.json"))
    total_size = sum(f.stat().st_size for f in files) // 1024
    elapsed    = (datetime.now() - start).seconds

    logger.info("=" * 56)
    logger.info(f"✅ Export complete in {elapsed}s")
    logger.info(f"   {len(files)} files, {total_size:,} KB total")
    logger.info(f"   {len(tickers)} tickers exported")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  git add data/")
    logger.info('  git commit -m "Data refresh -- ' + date.today().isoformat() + '"')
    logger.info("  git push")
    logger.info("  → Streamlit Cloud redeploys in ~30 seconds")
    logger.info("=" * 56)


if __name__ == "__main__":
    run_export()
