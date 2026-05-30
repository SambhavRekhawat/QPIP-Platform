"""
data_engine/events.py
---------------------
Event Intelligence Engine:
  - Pulls market news via NewsAPI and fallback RSS feeds
  - Performs VADER + TextBlob sentiment analysis
  - Classifies events (earnings, macro, corporate action, etc.)
  - Persists to PostgreSQL `events` table
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from loguru import logger
from sqlalchemy import text
from typing import List, Dict, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from config import NEWS_API_KEY
from database.connection import get_db_engine


# ─── Sentiment Analyser ──────────────────────────────────────────────────────

_vader = SentimentIntensityAnalyzer()


def analyse_sentiment(text_: str) -> Dict:
    """Run VADER + TextBlob sentiment and return a merged score."""
    if not text_ or not text_.strip():
        return {"score": 0.0, "label": "Neutral", "vader": 0.0, "textblob": 0.0}

    vader_scores  = _vader.polarity_scores(text_)
    vader_compound = vader_scores["compound"]        # -1 to +1

    tb = TextBlob(text_)
    tb_polarity   = tb.sentiment.polarity            # -1 to +1

    # Weighted blend: 60% VADER, 40% TextBlob
    combined = 0.6 * vader_compound + 0.4 * tb_polarity

    label = "Positive" if combined > 0.1 else ("Negative" if combined < -0.1 else "Neutral")

    return {
        "score":    round(combined, 4),
        "label":    label,
        "vader":    round(vader_compound, 4),
        "textblob": round(tb_polarity, 4),
    }


# ─── Event Classification ────────────────────────────────────────────────────

_EVENT_PATTERNS = {
    "earnings":      r"(earnings|profit|revenue|quarterly|results|ebitda|pat|eps)",
    "block_deal":    r"(block deal|bulk deal|off.market|stake sale)",
    "insider":       r"(promoter|insider|director|bought|sold|acquisition)",
    "macro":         r"(rbi|repo rate|inflation|cpi|gdp|budget|fiscal|fed|rate cut|rate hike)",
    "corp_action":   r"(dividend|bonus|split|rights issue|buyback|merger|acquisition|demerger)",
    "analyst":       r"(target|upgrade|downgrade|buy rating|sell rating|coverage|initiating)",
    "regulatory":    r"(sebi|nse|bse|penalty|notice|compliance|investigation)",
}


def classify_event(headline: str) -> str:
    """Classify event type from headline text."""
    headline_lower = headline.lower()
    for event_type, pattern in _EVENT_PATTERNS.items():
        if re.search(pattern, headline_lower):
            return event_type
    return "general"


# ─── News Fetching ────────────────────────────────────────────────────────────

def _newsapi_fetch(query: str, days_back: int = 7) -> List[Dict]:
    """Fetch articles from NewsAPI."""
    if not NEWS_API_KEY:
        return []

    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":          query,
        "from":       from_date,
        "sortBy":     "publishedAt",
        "language":   "en",
        "pageSize":   20,
        "apiKey":     NEWS_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if resp.status_code == 200 and data.get("status") == "ok":
            arts = data.get("articles", [])
            logger.info(f"  NewsAPI: {len(arts)} articles for query '{query[:40]}'")
            return arts
        # Surface the actual NewsAPI error so the cause is obvious
        logger.warning(f"  NewsAPI error [{data.get('code')}]: {data.get('message')}")
    except Exception as e:
        logger.warning(f"  NewsAPI request failed: {e}")
    return []


def _rss_fetch(feed_url: str, timeout: int = 5) -> List[Dict]:
    """RSS feed parser with strict timeout. Returns [] if slow or blocked."""
    try:
        import feedparser
        # Pre-fetch with requests so we can enforce a timeout
        resp = requests.get(
            feed_url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code != 200:
            return []
        feed = feedparser.parse(resp.text)
        articles = []
        for entry in feed.entries[:20]:
            articles.append({
                "title":       entry.get("title", ""),
                "description": entry.get("summary", ""),
                "url":         entry.get("link", ""),
                "publishedAt": entry.get("published", ""),
                "source":      {"name": feed.feed.get("title", "RSS")},
            })
        return articles
    except Exception:
        return []


# NSE RSS feeds (public)
NSE_RSS_FEEDS = [
    "https://www.nseindia.com/rss/corporate-annoucement.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/MCtopnews.xml",
]


def fetch_ticker_news(ticker: str, days_back: int = 7) -> List[Dict]:
    """Fetch news for a specific NSE ticker."""
    # Skip only if NO key is configured at all
    if not NEWS_API_KEY or NEWS_API_KEY in ("", "your_newsapi_key_here"):
        logger.debug(f"  Skipping news for {ticker} — no NewsAPI key configured")
        return []
    articles = _newsapi_fetch(
        query=f"{ticker} NSE India stock",
        days_back=days_back,
    )
    return articles


def fetch_macro_news(days_back: int = 3) -> List[Dict]:
    """Fetch India macro / market news."""
    if not NEWS_API_KEY or NEWS_API_KEY in ("", "your_newsapi_key_here"):
        logger.debug("  Skipping macro news — no NewsAPI key configured")
        return []
    articles = _newsapi_fetch(
        query="India market RBI SEBI Nifty Sensex",
        days_back=days_back,
    )
    return articles[:20]


# ─── Event Processing ────────────────────────────────────────────────────────

def process_articles(
    articles: List[Dict],
    ticker: Optional[str] = None,
) -> List[Dict]:
    """
    Given raw news API articles, enrich with sentiment and classification.
    Returns a list of processed event dicts ready for DB insertion.
    """
    events = []
    for art in articles:
        headline  = art.get("title", "") or ""
        body      = art.get("description", "") or ""
        full_text = f"{headline}. {body}".strip()

        sentiment = analyse_sentiment(full_text)
        event_type = classify_event(full_text)

        # Parse date
        raw_date = art.get("publishedAt", "")
        try:
            event_date = pd.to_datetime(raw_date).date()
        except Exception:
            event_date = date.today()

        events.append({
            "ticker":          ticker,
            "event_date":      event_date,
            "headline":        headline[:500],
            "source":          (art.get("source") or {}).get("name", ""),
            "url":             art.get("url", "")[:500],
            "event_type":      event_type,
            "sentiment_score": sentiment["score"],
            "sentiment_label": sentiment["label"],
            "impact_score":    abs(sentiment["score"]),
            "processed":       True,
            "raw_content":     full_text[:1000],
        })

    return events


# ─── Persistence ─────────────────────────────────────────────────────────────

def upsert_events(events: List[Dict], engine=None) -> int:
    """Insert events into PostgreSQL (no conflict key – always insert)."""
    if engine is None:
        engine = get_db_engine()
    if not events:
        return 0

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO events
                    (ticker, event_date, headline, source, url, event_type,
                     sentiment_score, sentiment_label, impact_score,
                     processed, raw_content)
                VALUES
                    (:ticker, :event_date, :headline, :source, :url, :event_type,
                     :sentiment_score, :sentiment_label, :impact_score,
                     :processed, :raw_content)
            """),
            events,
        )
    logger.info(f"✅ Inserted {len(events)} events into DB")
    return len(events)


def load_recent_events(
    tickers: List[str],
    days_back: int = 30,
    engine=None,
) -> pd.DataFrame:
    """Load recent events for a list of tickers from PostgreSQL."""
    if engine is None:
        engine = get_db_engine()

    cutoff = (date.today() - timedelta(days=days_back)).isoformat()
    placeholders = ", ".join([f"'{t}'" for t in tickers])
    query = f"""
        SELECT * FROM events
        WHERE  (ticker IN ({placeholders}) OR ticker IS NULL)
        AND    event_date >= '{cutoff}'
        ORDER  BY event_date DESC
        LIMIT  200
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    return df


def compute_ticker_sentiment_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate event sentiment per ticker.
    Returns a DataFrame with columns: ticker, avg_sentiment, event_count, dominant_label.
    """
    if events_df.empty or "ticker" not in events_df.columns:
        return pd.DataFrame()

    grp = events_df.dropna(subset=["ticker"]).groupby("ticker")
    summary = grp["sentiment_score"].agg(
        avg_sentiment="mean",
        event_count="count",
        max_impact="max",
    ).reset_index()

    # Dominant label
    def _dominant_label(sub):
        pos = (sub["sentiment_label"] == "Positive").sum()
        neg = (sub["sentiment_label"] == "Negative").sum()
        if pos > neg:
            return "Positive"
        elif neg > pos:
            return "Negative"
        return "Neutral"

    label_df = (
        events_df.dropna(subset=["ticker"])
        .groupby("ticker")
        .apply(_dominant_label)
        .reset_index(name="dominant_sentiment")
    )
    summary = summary.merge(label_df, on="ticker", how="left")
    return summary


# ─── ETL Pipeline ────────────────────────────────────────────────────────────

def run_events_pipeline(
    tickers: List[str],
    days_back: int = 7,
    engine=None,
) -> pd.DataFrame:
    """
    Full event ETL: fetch → process → store → return aggregated sentiment.
    """
    if engine is None:
        engine = get_db_engine()

    all_events: List[Dict] = []

    # Per-ticker news
    for ticker in tickers:
        raw = fetch_ticker_news(ticker, days_back=days_back)
        processed = process_articles(raw, ticker=ticker)
        all_events.extend(processed)

    # Macro news (untagged to ticker)
    macro_raw = fetch_macro_news(days_back=days_back)
    all_events.extend(process_articles(macro_raw, ticker=None))

    if all_events:
        upsert_events(all_events, engine)

    # Return aggregated summary
    events_df = load_recent_events(tickers, days_back=days_back, engine=engine)
    return compute_ticker_sentiment_summary(events_df)