"""
factors/composite.py
---------------------
Multi-factor investing engine.
Computes Momentum, Quality, Value, Low Volatility,
and a Composite Factor score for each ticker.
Persists factor scores to PostgreSQL.

Scale-safe: works correctly with 5 or 500 tickers.
Every numeric column is sanitised via clean_numeric_series()
before any arithmetic, fillna, or type-cast operation.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import date
from loguru import logger
from sqlalchemy import text
from typing import Dict, List, Optional

from config import FACTOR_LOOKBACK, FACTOR_WEIGHTS, TRADING_DAYS_YEAR
from database.connection import get_db_engine


# ─── Central numeric sanitiser ────────────────────────────────────────────────

def clean_numeric_series(series) -> pd.Series:
    """
    Guarantee the input becomes a plain float64 Series with no NaN, Inf, or
    object values.  Accepts Series, DataFrame, ndarray, or scalar.

    With 300+ tickers, yfinance occasionally returns a column as a
    single-column DataFrame instead of a Series.  pd.to_numeric() rejects
    DataFrames outright — so we squeeze to 1-D first.

    Handles:
      - Single-column DataFrame  → squeezed to Series
      - pandas nullable Int64 / Float64
      - numpy object arrays containing strings or None
      - np.inf / -np.inf from division
      - NaN from missing fundamentals

    Always returns a float64 Series filled with 0.0 for any bad value.
    """
    # ── Step 0: coerce DataFrame → Series ─────────────────────────────────────
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 1:
            series = series.iloc[:, 0]   # single column → Series
        else:
            # Multi-column: take the mean across columns as a best-effort
            series = series.mean(axis=1)

    # ── Step 1: numpy arrays / scalars → Series ───────────────────────────────
    if not isinstance(series, pd.Series):
        try:
            series = pd.Series(series)
        except Exception:
            return pd.Series(dtype="float64")

    return (
        pd.to_numeric(series, errors="coerce")   # strings / objects → NaN
        .astype("float64")                        # nullable Int64 → float64
        .replace([np.inf, -np.inf], np.nan)       # Inf → NaN
        .fillna(0.0)                              # NaN → 0.0
    )


def clean_numeric_df(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Apply clean_numeric_series to every column (or a named subset) of a DataFrame.
    Returns a copy — never mutates the original.
    """
    df = df.copy()
    target = cols if cols is not None else df.columns.tolist()
    for col in target:
        if col in df.columns:
            df[col] = clean_numeric_series(df[col])
    return df


# ─── Normalisation ────────────────────────────────────────────────────────────

def cross_sectional_zscore(series: pd.Series, clip: float = 3.0) -> pd.Series:
    """Z-score normalise across tickers; clip outliers at ±clip σ."""
    # Sanitise before any arithmetic
    s = clean_numeric_series(series)
    mean = s.mean()
    std  = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    z = (s - mean) / std
    return z.clip(-clip, clip)


def percentile_rank(series: pd.Series) -> pd.Series:
    """Rank from 0 to 100 across tickers."""
    return clean_numeric_series(series).rank(pct=True) * 100


# ─── Momentum Factor ──────────────────────────────────────────────────────────

def _scalar(x) -> float:
    """
    Safely extract a single float from anything iloc might return.

    With 300+ tickers yfinance occasionally stores a column as a
    single-column DataFrame instead of a Series, so iloc returns
    a one-row Series instead of a scalar.  This helper handles that.
    """
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.iloc[0] if len(x) > 0 else np.nan
    try:
        f = float(x)
        return np.nan if (f != f or np.isinf(f)) else f
    except (TypeError, ValueError):
        return np.nan


def compute_momentum_scores(close_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 1M / 6M / 12M price momentum for every ticker in close_matrix.
    Returns a DataFrame with one row per ticker and a composite momentum_score.
    Scale-safe: uses _scalar() to handle cases where a yfinance column is
    returned as a DataFrame instead of a Series (common with 200+ tickers).
    """
    results = {}

    for ticker in close_matrix.columns:
        raw = close_matrix[ticker]

        # Squeeze: if yfinance returned a single-column DataFrame instead of
        # a Series, convert it. This is the root cause of the float(Series) error.
        if isinstance(raw, pd.DataFrame):
            raw = raw.squeeze(axis=1)

        prices = raw.dropna()

        if len(prices) < FACTOR_LOOKBACK["momentum_12m"]:
            results[ticker] = {
                "momentum_1m": np.nan,
                "momentum_6m": np.nan,
                "momentum_12m": np.nan,
            }
            continue

        # Skip the last 21 days — standard practice to avoid short-term reversal
        skip    = min(21, len(prices) - 1)
        end_idx = -skip

        p_now  = _scalar(prices.iloc[end_idx])
        p_1m   = _scalar(prices.iloc[max(end_idx - 21,  -len(prices))])
        p_6m   = _scalar(prices.iloc[max(end_idx - FACTOR_LOOKBACK["momentum_6m"],  -len(prices))])
        p_12m  = _scalar(prices.iloc[max(end_idx - FACTOR_LOOKBACK["momentum_12m"], -len(prices))])

        results[ticker] = {
            "momentum_1m":  (p_now / p_1m  - 1) if (p_1m  and p_1m  > 0) else np.nan,
            "momentum_6m":  (p_now / p_6m  - 1) if (p_6m  and p_6m  > 0) else np.nan,
            "momentum_12m": (p_now / p_12m - 1) if (p_12m and p_12m > 0) else np.nan,
        }

    df = pd.DataFrame(results).T
    df = clean_numeric_df(df)   # ← sanitise raw momentum values

    # Cross-sectional z-score each lookback window
    for col in ["momentum_1m", "momentum_6m", "momentum_12m"]:
        df[col + "_z"] = cross_sectional_zscore(df[col])

    df["momentum_score"] = clean_numeric_series(
        0.30 * df["momentum_1m_z"]
        + 0.35 * df["momentum_6m_z"]
        + 0.35 * df["momentum_12m_z"]
    )
    return df


# ─── Quality Factor ───────────────────────────────────────────────────────────

def compute_quality_scores(fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Quality = high ROE + high ROCE + high margins + low debt + promoter confidence.
    Each metric is sanitised, median-imputed, then cross-sectionally z-scored.
    """
    df = fundamentals_df.set_index("ticker").copy()

    quality_cols = {
        "roe":              1.0,   # higher = better
        "roce":             1.0,
        "operating_margin": 1.0,
        "promoter_holding": 0.5,
        "debt_to_equity":  -1.0,   # lower = better (flip sign)
        "revenue_growth":   0.5,
        "profit_growth":    0.5,
    }

    z_scores = {}
    for col, direction in quality_cols.items():
        if col not in df.columns:
            continue
        raw = clean_numeric_series(df[col])          # ← sanitise first
        median = raw[raw != 0].median()
        raw    = raw.replace(0, median if not np.isnan(median) else 0.0)
        z_scores[col] = cross_sectional_zscore(raw) * direction

    if not z_scores:
        df["quality_score"] = 0.0
        return df[["quality_score"]].reset_index()

    z_df = pd.DataFrame(z_scores)
    df["quality_score"] = clean_numeric_series(z_df.mean(axis=1))
    return df[["quality_score"]].reset_index()


# ─── Value Factor ─────────────────────────────────────────────────────────────

def compute_value_scores(fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Value = low PE + low PB + high dividend yield.
    Extreme PE values are winsorised before z-scoring.
    """
    df = fundamentals_df.set_index("ticker").copy()

    value_cols = {
        "pe_ratio":       -1.0,   # lower = more value
        "pb_ratio":       -1.0,
        "dividend_yield":  1.0,   # higher = more value
    }

    z_scores = {}
    for col, direction in value_cols.items():
        if col not in df.columns:
            continue
        raw = clean_numeric_series(df[col])          # ← sanitise first
        # Winsorise extreme values (common with PE ratios at 500+ tickers)
        lo, hi = raw.quantile(0.05), raw.quantile(0.95)
        raw    = raw.clip(lo, hi)
        z_scores[col] = cross_sectional_zscore(raw) * direction

    if not z_scores:
        df["value_score"] = 0.0
        return df[["value_score"]].reset_index()

    z_df = pd.DataFrame(z_scores)
    df["value_score"] = clean_numeric_series(z_df.mean(axis=1))
    return df[["value_score"]].reset_index()


# ─── Low Volatility Factor ────────────────────────────────────────────────────

def compute_low_vol_scores(
    returns_matrix: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """Lower realised volatility → higher score."""
    tail = returns_matrix.tail(window)
    # std() on a DataFrame returns a Series — safe. But guard anyway.
    std_series = tail.std()
    if isinstance(std_series, pd.DataFrame):
        std_series = std_series.squeeze(axis=1)
    vols = clean_numeric_series(std_series * np.sqrt(TRADING_DAYS_YEAR))
    vol_df = pd.DataFrame({"raw_vol": vols})
    vol_df["low_vol_score"] = cross_sectional_zscore(-vol_df["raw_vol"])
    vol_df.index.name = "ticker"
    return vol_df[["low_vol_score"]].reset_index()


# ─── Beta Computation ─────────────────────────────────────────────────────────

def compute_betas(
    returns_matrix: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Rolling OLS beta vs benchmark for each ticker."""
    aligned_bench = benchmark_returns.reindex(returns_matrix.index).fillna(0)
    tail_ret   = returns_matrix.tail(window)
    tail_bench = aligned_bench.tail(window)

    betas = {}
    for ticker in tail_ret.columns:
        col = tail_ret[ticker]
        # Squeeze single-column DataFrames → Series (yfinance 300+ ticker issue)
        if isinstance(col, pd.DataFrame):
            col = col.squeeze(axis=1)
        r = clean_numeric_series(col.dropna())
        b = clean_numeric_series(tail_bench.reindex(r.index))
        if len(r) < 30:
            betas[ticker] = np.nan
            continue
        try:
            cov = np.cov(r.values, b.values)
            betas[ticker] = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else np.nan
        except Exception:
            betas[ticker] = np.nan

    return pd.Series(betas, name="beta", dtype="float64")


# ─── Composite Score ──────────────────────────────────────────────────────────

def compute_composite_scores(
    tickers: List[str],
    close_matrix: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    benchmark_returns: pd.Series,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Master function: compute all factor scores and composite ranking.

    Scale-safe design:
      - Filters to only tickers with actual price data before any computation.
      - Every sub-score DataFrame is sanitised before merging.
      - After merge, ALL columns are forced to clean float64 via clean_numeric_df().
      - Rankings stored as float64 (not Int64) to survive NaN values safely.

    Returns one row per available ticker.
    """
    if weights is None:
        weights = FACTOR_WEIGHTS

    # ── 1. Filter to tickers that have real data ───────────────────────────────
    available = [
        t for t in tickers
        if t in close_matrix.columns and t in returns_matrix.columns
    ]
    skipped = set(tickers) - set(available)
    if skipped:
        logger.warning(f"  Factor engine skipping {len(skipped)} tickers with no data: {skipped}")
    if not available:
        logger.error("No tickers available for factor computation.")
        return pd.DataFrame()

    # ── 2. Compute each factor sub-score ──────────────────────────────────────
    mom_df = compute_momentum_scores(close_matrix[available])
    mom_df = mom_df[["momentum_1m", "momentum_6m", "momentum_12m", "momentum_score"]]

    if not fundamentals_df.empty:
        fund_avail = fundamentals_df[fundamentals_df["ticker"].isin(available)]
        qual_df    = compute_quality_scores(fund_avail).set_index("ticker")
        val_df     = compute_value_scores(fund_avail).set_index("ticker")
    else:
        qual_df = pd.DataFrame({"quality_score": 0.0}, index=available)
        qual_df.index.name = "ticker"
        val_df  = pd.DataFrame({"value_score":   0.0}, index=available)
        val_df.index.name = "ticker"

    lv_df = compute_low_vol_scores(returns_matrix[available]).set_index("ticker")
    betas  = compute_betas(returns_matrix[available], benchmark_returns)

    # ── 3. Deduplicate indices before merging ─────────────────────────────────
    # With 300+ tickers, yfinance can produce duplicate ticker labels which
    # propagate into the factor DataFrames, causing pd.concat to fail.
    def _dedup(df):
        """Keep only the first occurrence of each index label."""
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep="first")]
        return df

    mom_df  = _dedup(mom_df)
    qual_df = _dedup(qual_df)
    val_df  = _dedup(val_df)
    lv_df   = _dedup(lv_df)

    # Also deduplicate the available list itself
    seen = set()
    available = [t for t in available if not (t in seen or seen.add(t))]

    # ── 4. Merge ──────────────────────────────────────────────────────────────
    result = pd.concat([mom_df, qual_df, val_df, lv_df], axis=1)
    result = result.reindex(available)
    result["beta"] = betas.reindex(result.index)

    # ── 4. Centralised sanitisation of ALL columns ────────────────────────────
    #    This is the key fix: every column becomes clean float64 / 0.0.
    #    No Int64, no object, no NaN, no Inf anywhere from this point on.
    SCORE_COLS = [
        "momentum_1m", "momentum_6m", "momentum_12m",
        "momentum_score", "quality_score", "value_score",
        "low_vol_score", "beta",
    ]
    result = clean_numeric_df(result, cols=SCORE_COLS)

    # ── 5. Composite score ────────────────────────────────────────────────────
    result["composite_score"] = clean_numeric_series(
        weights.get("momentum", 0.25) * result["momentum_score"]
        + weights.get("quality",  0.30) * result["quality_score"]
        + weights.get("value",    0.25) * result["value_score"]
        + weights.get("low_vol",  0.20) * result["low_vol_score"]
    )

    # ── 6. Rankings (float64 — never Int64) ───────────────────────────────────
    for score_col, rank_col in [
        ("momentum_score",  "momentum_rank"),
        ("quality_score",   "quality_rank"),
        ("value_score",     "value_rank"),
        ("low_vol_score",   "low_vol_rank"),
        ("composite_score", "composite_rank"),
    ]:
        if score_col in result.columns:
            result[rank_col] = (
                result[score_col]
                .rank(ascending=False, na_option="bottom", method="min")
                .astype("float64")
            )

    result.index.name = "ticker"
    result["date"] = date.today()
    return result.reset_index()


# ─── Persistence ─────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    """Convert any value to plain Python float for PostgreSQL. NaN → None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (f != f or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> Optional[int]:
    """
    Convert a rank value (float64 like 1.0, 2.0 …) to plain Python int.
    PostgreSQL INTEGER columns reject numpy/pandas float64 — this converts
    1.0 → 1, 2.0 → 2 etc. and returns None for NaN or Inf.
    """
    if val is None:
        return None
    try:
        f = float(val)
        if f != f or np.isinf(f):   # NaN or Inf
            return None
        return int(round(f))
    except (TypeError, ValueError):
        return None


def upsert_factors(df: pd.DataFrame, engine=None) -> int:
    if engine is None:
        engine = get_db_engine()

    # ── Debug: log any tickers that still have NaN/Inf in key columns ─────────
    key_cols = ["momentum_score", "quality_score", "value_score",
                "composite_score", "momentum_rank", "composite_rank"]
    debug_cols = [c for c in key_cols if c in df.columns]
    if debug_cols:
        bad_mask = df[debug_cols].isnull().any(axis=1) |                    df[debug_cols].apply(lambda col: col.map(
                       lambda v: isinstance(v, float) and np.isinf(v)
                   )).any(axis=1)
        bad_tickers = df.loc[bad_mask, "ticker"].tolist() if "ticker" in df.columns else []
        if bad_tickers:
            logger.warning(f"  ⚠️  Tickers with NaN/Inf values (will be stored as NULL): "
                           f"{bad_tickers}")
            for t in bad_tickers:
                row = df[df["ticker"] == t].iloc[0]
                issues = {col: row[col] for col in debug_cols
                          if pd.isna(row[col]) or (isinstance(row[col], float) and np.isinf(row[col]))}
                logger.debug(f"     {t}: {issues}")
        else:
            logger.debug(f"  ✅ All {len(df)} tickers have clean numeric values")

    records = []
    for _, row in df.iterrows():
        records.append({
            "ticker":           row.get("ticker"),
            "date":             row.get("date", date.today()),
            "momentum_6m":      _safe_float(row.get("momentum_6m")),
            "momentum_12m":     _safe_float(row.get("momentum_12m")),
            "momentum_1m":      _safe_float(row.get("momentum_1m")),
            "volatility_score": _safe_float(row.get("low_vol_score")),
            "quality_score":    _safe_float(row.get("quality_score")),
            "value_score":      _safe_float(row.get("value_score")),
            "composite_score":  _safe_float(row.get("composite_score")),
            "momentum_rank":    _safe_int(row.get("momentum_rank")),
            "quality_rank":     _safe_int(row.get("quality_rank")),
            "value_rank":       _safe_int(row.get("value_rank")),
            "low_vol_rank":     _safe_int(row.get("low_vol_rank")),
            "composite_rank":   _safe_int(row.get("composite_rank")),
            "beta":             _safe_float(row.get("beta")),
        })

    if not records:
        return 0

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO factors
                    (ticker, date, momentum_6m, momentum_12m, momentum_1m,
                     volatility_score, quality_score, value_score, composite_score,
                     momentum_rank, quality_rank, value_rank, low_vol_rank,
                     composite_rank, beta)
                VALUES
                    (:ticker, :date, :momentum_6m, :momentum_12m, :momentum_1m,
                     :volatility_score, :quality_score, :value_score, :composite_score,
                     :momentum_rank, :quality_rank, :value_rank, :low_vol_rank,
                     :composite_rank, :beta)
                ON CONFLICT (ticker, date) DO UPDATE SET
                    momentum_6m      = EXCLUDED.momentum_6m,
                    momentum_12m     = EXCLUDED.momentum_12m,
                    momentum_1m      = EXCLUDED.momentum_1m,
                    volatility_score = EXCLUDED.volatility_score,
                    quality_score    = EXCLUDED.quality_score,
                    value_score      = EXCLUDED.value_score,
                    composite_score  = EXCLUDED.composite_score,
                    composite_rank   = EXCLUDED.composite_rank,
                    beta             = EXCLUDED.beta
            """),
            records,
        )

    logger.info(f"✅ Upserted {len(records)} factor rows")
    return len(records)


def load_factors(tickers: List[str], engine=None) -> pd.DataFrame:
    if engine is None:
        engine = get_db_engine()
    placeholders = ", ".join([f"'{t}'" for t in tickers])
    query = f"""
        SELECT DISTINCT ON (ticker) *
        FROM   factors
        WHERE  ticker IN ({placeholders})
        ORDER  BY ticker, date DESC
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_factor_pipeline(
    tickers: List[str],
    close_matrix: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    benchmark_returns: pd.Series,
    engine=None,
) -> pd.DataFrame:
    """Full factor ETL pipeline. Safe for any universe size."""
    if engine is None:
        engine = get_db_engine()

    logger.info(f"Computing multi-factor scores for {len(tickers)} tickers…")
    factor_df = compute_composite_scores(
        tickers, close_matrix, returns_matrix,
        fundamentals_df, benchmark_returns,
    )
    upsert_factors(factor_df, engine)
    logger.info("✅ Factor pipeline complete.")
    return factor_df