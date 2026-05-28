"""
data_engine/fundamentals.py
----------------------------
Scrapes fundamental financial data from Screener.in and
stores it in the `fundamentals` PostgreSQL table.
Includes fallback to yfinance info when Screener is unavailable.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime
from bs4 import BeautifulSoup
from loguru import logger
from sqlalchemy import text
from typing import Dict, List, Optional

from config import (
    SCREENER_BASE_URL, SCREENER_TIMEOUT_SEC, NSE_SUFFIX
)
from database.connection import get_db_engine


# ─── Screener.in Scraper ─────────────────────────────────────────────────────

SCREENER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_FIELD_MAP = {
    "Market Cap":       "market_cap",
    "P/E":              "pe_ratio",
    "Price to book":    "pb_ratio",
    "ROE":              "roe",
    "ROCE":             "roce",
    "Debt to equity":   "debt_to_equity",
    "Promoter holding": "promoter_holding",
    "EPS":              "eps",
    "Dividend Yield":   "dividend_yield",
    "Current ratio":    "current_ratio",
    "Asset Turnover":   "asset_turnover",
}


def _clean_number(text: str) -> Optional[float]:
    """Strip % / Cr / commas from a Screener string and return float."""
    if not text:
        return None
    text = text.strip().replace(",", "").replace("%", "")
    # Crore → just keep the number (assume already Crore scale)
    text = text.replace("Cr.", "").replace("Cr", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def scrape_screener(ticker: str) -> Dict:
    """
    Scrape key fundamentals from Screener.in for a single NSE ticker.
    Returns a flat dict of fundamental fields.
    """
    url = SCREENER_BASE_URL.format(ticker=ticker.upper())
    result = {"ticker": ticker, "as_of_date": date.today(), "source": "screener.in"}

    try:
        resp = requests.get(url, headers=SCREENER_HEADERS,
                            timeout=SCREENER_TIMEOUT_SEC)
        if resp.status_code != 200:
            logger.warning(f"  Screener HTTP {resp.status_code} for {ticker}")
            return result

        soup = BeautifulSoup(resp.text, "lxml")

        # ── Key ratios section ──────────────────────────────────────────────
        for li in soup.select("#top-ratios li"):
            name_tag = li.find("span", class_="name")
            val_tag  = li.find("span", class_="value")
            if not name_tag or not val_tag:
                continue
            name = name_tag.get_text(strip=True)
            val  = val_tag.get_text(strip=True)
            col  = _FIELD_MAP.get(name)
            if col:
                result[col] = _clean_number(val)

        # ── Profit & Loss for growth metrics ───────────────────────────────
        pl_table = soup.find("section", id="profit-loss")
        if pl_table:
            rows = pl_table.select("table tbody tr")
            for row in rows:
                label = row.find("td")
                if not label:
                    continue
                label_txt = label.get_text(strip=True).lower()
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue
                # Last two columns = most recent periods
                try:
                    vals = [_clean_number(c.get_text(strip=True)) for c in cells[-3:]]
                    vals = [v for v in vals if v is not None]
                    if len(vals) >= 2 and vals[-2] and vals[-2] != 0:
                        growth = (vals[-1] - vals[-2]) / abs(vals[-2]) * 100
                    else:
                        growth = None
                except Exception:
                    growth = None

                if "sales" in label_txt or "revenue" in label_txt:
                    result["revenue_growth"] = growth
                elif "profit" in label_txt:
                    result["profit_growth"] = growth

        # ── Operating Margin ────────────────────────────────────────────────
        for row in (pl_table.select("table tbody tr") if pl_table else []):
            label = (row.find("td") or {}).get_text(strip=True).lower()
            if "opm" in label or "operating" in label:
                cells = row.find_all("td")
                if len(cells) >= 2:
                    result["operating_margin"] = _clean_number(
                        cells[-1].get_text(strip=True)
                    )
                break

        logger.info(f"  ✅ Screener scraped: {ticker}")

    except requests.exceptions.Timeout:
        logger.warning(f"  Screener timeout for {ticker}")
    except Exception as e:
        logger.error(f"  Screener error for {ticker}: {e}")

    return result


def _yfinance_fallback(ticker: str) -> Dict:
    """Fetch fundamentals from yfinance as a fallback."""
    yft = ticker + NSE_SUFFIX if not ticker.endswith(NSE_SUFFIX) else ticker
    info = yf.Ticker(yft).info

    def _get(key, scale=1.0):
        v = info.get(key)
        return float(v) * scale if v is not None else None

    return {
        "ticker":           ticker,
        "as_of_date":       date.today(),
        "source":           "yfinance",
        "pe_ratio":         _get("trailingPE"),
        "pb_ratio":         _get("priceToBook"),
        "roe":              _get("returnOnEquity", 100),
        "roce":             None,
        "debt_to_equity":   _get("debtToEquity"),
        "revenue_growth":   _get("revenueGrowth", 100),
        "profit_growth":    _get("earningsGrowth", 100),
        "operating_margin": _get("operatingMargins", 100),
        "promoter_holding": None,
        "market_cap":       _get("marketCap", 1 / 1e7),  # → Crore
        "eps":              _get("trailingEps"),
        "dividend_yield":   _get("dividendYield", 100),
        "current_ratio":    _get("currentRatio"),
        "asset_turnover":   _get("assetTurnover"),
    }


def fetch_fundamentals(ticker: str, prefer_screener: bool = True) -> Dict:
    """
    Fetch fundamentals for a ticker.
    Primary: Screener.in scraping.
    Fallback: yfinance info.
    """
    result = {}
    if prefer_screener:
        result = scrape_screener(ticker)
        # Check if we got meaningful data
        meaningful = [k for k in ["pe_ratio", "pb_ratio", "roe", "roce"]
                      if result.get(k) is not None]
        if len(meaningful) < 2:
            logger.info(f"  Screener sparse for {ticker}, using yfinance fallback")
            yf_data = _yfinance_fallback(ticker)
            # Fill missing keys
            for k, v in yf_data.items():
                if result.get(k) is None and v is not None:
                    result[k] = v
    else:
        result = _yfinance_fallback(ticker)

    return result


def fetch_fundamentals_batch(
    tickers: List[str],
    delay_sec: float = 2.0,
    prefer_screener: bool = True,
) -> pd.DataFrame:
    """Scrape fundamentals for multiple tickers with polite delay."""
    records = []
    for ticker in tickers:
        rec = fetch_fundamentals(ticker, prefer_screener=prefer_screener)
        records.append(rec)
        if prefer_screener:
            time.sleep(delay_sec)

    df = pd.DataFrame(records)
    return df


# ─── Persistence ─────────────────────────────────────────────────────────────

def upsert_fundamentals(df: pd.DataFrame, engine=None) -> int:
    """Upsert fundamentals DataFrame into PostgreSQL."""
    if engine is None:
        engine = get_db_engine()

    cols = [
        "ticker", "as_of_date", "pe_ratio", "pb_ratio", "roe", "roce",
        "debt_to_equity", "revenue_growth", "profit_growth",
        "operating_margin", "promoter_holding", "market_cap",
        "eps", "dividend_yield", "current_ratio", "asset_turnover", "source",
    ]

    records = []
    for _, row in df.iterrows():
        rec = {}
        for c in cols:
            val = row.get(c)
            if pd.isna(val) if isinstance(val, float) else val is None:
                rec[c] = None
            else:
                rec[c] = val
        records.append(rec)

    if not records:
        return 0

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO fundamentals
                    (ticker, as_of_date, pe_ratio, pb_ratio, roe, roce,
                     debt_to_equity, revenue_growth, profit_growth,
                     operating_margin, promoter_holding, market_cap,
                     eps, dividend_yield, current_ratio, asset_turnover, source)
                VALUES
                    (:ticker, :as_of_date, :pe_ratio, :pb_ratio, :roe, :roce,
                     :debt_to_equity, :revenue_growth, :profit_growth,
                     :operating_margin, :promoter_holding, :market_cap,
                     :eps, :dividend_yield, :current_ratio, :asset_turnover, :source)
                ON CONFLICT (ticker, as_of_date) DO UPDATE SET
                    pe_ratio         = EXCLUDED.pe_ratio,
                    pb_ratio         = EXCLUDED.pb_ratio,
                    roe              = EXCLUDED.roe,
                    roce             = EXCLUDED.roce,
                    debt_to_equity   = EXCLUDED.debt_to_equity,
                    revenue_growth   = EXCLUDED.revenue_growth,
                    profit_growth    = EXCLUDED.profit_growth,
                    operating_margin = EXCLUDED.operating_margin,
                    promoter_holding = EXCLUDED.promoter_holding,
                    market_cap       = EXCLUDED.market_cap,
                    updated_at       = NOW()
            """),
            records,
        )

    logger.info(f"✅ Upserted {len(records)} fundamental records")
    return len(records)


def load_fundamentals(tickers: List[str], engine=None) -> pd.DataFrame:
    """Load latest fundamentals for a list of tickers from PostgreSQL."""
    if engine is None:
        engine = get_db_engine()

    placeholders = ", ".join([f"'{t}'" for t in tickers])
    query = f"""
        SELECT DISTINCT ON (ticker) *
        FROM   fundamentals
        WHERE  ticker IN ({placeholders})
        ORDER  BY ticker, as_of_date DESC
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    return df


# ─── ETL Pipeline ────────────────────────────────────────────────────────────

def run_fundamentals_pipeline(
    tickers: List[str],
    engine=None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Full fundamentals ETL: check staleness → scrape → store → return.
    Only re-scrapes if data is more than 6 days old (or force_refresh=True).
    """
    if engine is None:
        engine = get_db_engine()

    if not force_refresh:
        existing = load_fundamentals(tickers, engine)
        if not existing.empty:
            existing["as_of_date"] = pd.to_datetime(existing["as_of_date"])
            fresh_mask = (
                (datetime.now() - existing["as_of_date"]).dt.days <= 6
            )
            fresh_tickers = set(existing.loc[fresh_mask, "ticker"].tolist())
            stale_tickers = [t for t in tickers if t not in fresh_tickers]
            if not stale_tickers:
                logger.info("✅ All fundamentals are fresh. Skipping scrape.")
                return existing
            tickers = stale_tickers
            logger.info(f"Stale tickers to refresh: {stale_tickers}")

    df = fetch_fundamentals_batch(tickers, prefer_screener=True)
    upsert_fundamentals(df, engine)
    return load_fundamentals(tickers, engine)
