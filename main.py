"""
main.py
-------
Master orchestrator for the Quantitative Portfolio Intelligence Platform.

Usage:
    # Run full pipeline for a list of tickers
    python main.py --tickers RELIANCE TCS HDFCBANK INFY ICICIBANK --method "Equal Weighted"

    # Setup database only
    python main.py --setup-db

    # Test database connection
    python main.py --test-db
"""

import argparse
import sys
from loguru import logger


def setup_database():
    """Create all PostgreSQL tables."""
    logger.info("🗄️  Setting up database…")
    from database.connection import init_database, test_connection
    if not test_connection():
        logger.error("Cannot reach PostgreSQL. Check your .env DB settings.")
        sys.exit(1)
    init_database()
    logger.info("✅ Database ready.")


def run_pipeline(
    tickers: list,
    method: str = "Equal Weighted",
    custom_weights: dict = None,
    years: int = 2,
    skip_scrape: bool = False,
    skip_ml: bool = False,
):
    """Run the full data + analytics + ML pipeline."""
    from utils.helpers import validate_tickers
    tickers = validate_tickers(tickers)

    if len(tickers) < 5:
        logger.error(f"Need at least 5 tickers. Got: {tickers}")
        sys.exit(1)

    logger.info(f"🚀 Starting pipeline | Tickers: {tickers} | Method: {method}")

    # ── 1. Database ───────────────────────────────────────────────────────────
    from database.connection import get_db_engine, init_database, test_connection
    if not test_connection():
        logger.error("Database unreachable. Please check DB_* settings in .env")
        sys.exit(1)
    engine = get_db_engine()
    init_database(engine)

    # ── 2. Market Data ────────────────────────────────────────────────────────
    logger.info("📥 Fetching market data…")
    from data_engine.market_data import (
        run_market_data_pipeline, load_close_matrix, load_returns_matrix,
        fetch_benchmark, compute_returns,
    )
    from config import BENCHMARK_TICKER

    run_market_data_pipeline(tickers, years=years, engine=engine)

    close_m   = load_close_matrix(tickers, engine=engine)
    returns_m = load_returns_matrix(tickers, engine=engine)

    bench_raw = fetch_benchmark(years=years)
    bench_r   = compute_returns(bench_raw)["daily_return"] if not bench_raw.empty else None

    if bench_r is None:
        logger.error("Failed to load benchmark data.")
        sys.exit(1)

    logger.info(f"  Price matrix: {close_m.shape} | Returns matrix: {returns_m.shape}")

    # ── 3. Fundamentals ───────────────────────────────────────────────────────
    fundamentals_df = None
    if not skip_scrape:
        logger.info("📊 Fetching fundamentals…")
        from data_engine.fundamentals import run_fundamentals_pipeline
        fundamentals_df = run_fundamentals_pipeline(tickers, engine=engine)
        logger.info(f"  Fundamentals loaded: {len(fundamentals_df)} rows")
    else:
        import pandas as pd
        fundamentals_df = pd.DataFrame()

    # ── 4. Portfolio Construction ─────────────────────────────────────────────
    logger.info("🏗️  Building portfolio…")
    from analytics.portfolio import compute_weights, portfolio_summary, upsert_portfolio_returns

    market_caps = {}
    if fundamentals_df is not None and not fundamentals_df.empty and "market_cap" in fundamentals_df.columns:
        market_caps = dict(zip(fundamentals_df["ticker"], fundamentals_df["market_cap"].fillna(1e10)))

    weights = compute_weights(tickers, method, market_caps=market_caps, custom_weights=custom_weights)
    logger.info(f"  Weights: { {k: f'{v:.1%}' for k, v in weights.items()} }")

    summary = portfolio_summary(tickers, weights, returns_m, bench_r, method)
    upsert_portfolio_returns(summary, engine=engine)

    metrics = summary["metrics"]
    logger.info(
        f"  Portfolio | Return={metrics['ann_return']:.1%} | "
        f"Sharpe={metrics['sharpe_ratio']:.2f} | "
        f"MaxDD={metrics['max_drawdown']:.1%}"
    )

    # ── 5. Factor Scores ──────────────────────────────────────────────────────
    logger.info("⚡ Computing factor scores…")
    from factors.composite import run_factor_pipeline
    factor_df = run_factor_pipeline(
        tickers, close_m, returns_m, fundamentals_df, bench_r, engine=engine
    )
    logger.info(f"  Factor scores computed for {len(factor_df)} tickers")

    # ── 6. Event Intelligence ─────────────────────────────────────────────────
    logger.info("📰 Fetching events & sentiment…")
    try:
        from data_engine.events import run_events_pipeline
        sentiment_df = run_events_pipeline(tickers, days_back=7, engine=engine)
        logger.info(f"  Sentiment summary: {len(sentiment_df)} tickers")
    except Exception as e:
        logger.warning(f"  Events pipeline skipped: {e}")
        import pandas as pd
        sentiment_df = pd.DataFrame()

    # ── 7. ML Signals ─────────────────────────────────────────────────────────
    signals_df = None
    if not skip_ml:
        logger.info("🤖 Running ML signal engine…")
        from ml_engine.models import run_ml_pipeline
        signals_df, shap_df = run_ml_pipeline(
            tickers, close_m, returns_m, factor_df, fundamentals_df,
            bench_r, sentiment_df, engine=engine, retrain=True,
        )
        if signals_df is not None and not signals_df.empty:
            logger.info("  ML Signals:")
            for _, row in signals_df.iterrows():
                logger.info(
                    f"    {row['ticker']:12s} → {row['signal']:4s} "
                    f"(BUY:{row['buy_probability']:.0%} | "
                    f"SELL:{row['sell_probability']:.0%})"
                )

    # ── 8. Backtest ───────────────────────────────────────────────────────────
    logger.info("📉 Running backtest…")
    from backtesting.backtest import run_backtest, save_backtest_result
    bt_result = run_backtest(
        tickers, returns_m, bench_r,
        weighting_method=method,
        market_caps=market_caps,
        strategy_name=f"{method} Portfolio",
    )
    save_backtest_result(bt_result, engine=engine)

    logger.info("=" * 60)
    logger.info("✅ Pipeline complete!")
    logger.info(f"  Run `streamlit run dashboard/app.py` to view your dashboard")
    logger.info("=" * 60)

    return {
        "summary":      summary,
        "factor_df":    factor_df,
        "signals_df":   signals_df,
        "backtest":     bt_result,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantitative Portfolio Intelligence Platform",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="NSE ticker symbols (e.g. RELIANCE TCS HDFCBANK INFY ICICIBANK)",
    )
    parser.add_argument(
        "--method",
        choices=["Equal Weighted", "Market Cap Weighted", "Custom Weighted"],
        default="Equal Weighted",
        help="Portfolio weighting method",
    )
    parser.add_argument(
        "--years", type=int, default=2,
        help="Years of historical data to use (default: 2)",
    )
    parser.add_argument(
        "--setup-db", action="store_true",
        help="Create database tables and exit",
    )
    parser.add_argument(
        "--test-db", action="store_true",
        help="Test database connection and exit",
    )
    parser.add_argument(
        "--skip-scrape", action="store_true",
        help="Skip Screener.in scraping (use cached fundamentals)",
    )
    parser.add_argument(
        "--skip-ml", action="store_true",
        help="Skip ML signal generation",
    )

    args = parser.parse_args()

    if args.test_db:
        from database.connection import test_connection
        ok = test_connection()
        sys.exit(0 if ok else 1)

    if args.setup_db:
        setup_database()
        sys.exit(0)

    # Default tickers if none provided
    from config import DEFAULT_TICKERS
    tickers = args.tickers or DEFAULT_TICKERS[:7]

    run_pipeline(
        tickers=tickers,
        method=args.method,
        years=args.years,
        skip_scrape=args.skip_scrape,
        skip_ml=args.skip_ml,
    )
