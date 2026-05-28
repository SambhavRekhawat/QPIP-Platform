"""
utils/helpers.py
----------------
Shared utility functions used across the platform.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import date, datetime


def format_pct(value: Optional[float], decimals: int = 2) -> str:
    """Format a float as a percentage string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_num(value: Optional[float], decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.{decimals}f}"


def format_crore(value: Optional[float]) -> str:
    """Format a number in Indian Crore notation."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if value >= 1_00_000:
        return f"₹{value/1_00_000:.1f}L Cr"
    if value >= 1_000:
        return f"₹{value/1_000:.1f}K Cr"
    return f"₹{value:.0f} Cr"


def safe_divide(a: float, b: float, default: float = np.nan) -> float:
    try:
        if b == 0 or np.isnan(b):
            return default
        return a / b
    except Exception:
        return default


def flatten_dict(d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
    """Recursively flatten a nested dict."""
    items: List = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def winsorise(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorise a series at given quantile boundaries."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def reindex_to_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill a DataFrame to include all trading days."""
    if df.empty:
        return df
    full_idx = pd.bdate_range(start=df.index.min(), end=df.index.max())
    return df.reindex(full_idx).ffill()


def validate_tickers(tickers: List[str]) -> List[str]:
    """Clean and upper-case ticker list, remove duplicates."""
    seen = set()
    clean = []
    for t in tickers:
        t = t.strip().upper().replace(".NS", "").replace(".BO", "")
        if t and t not in seen:
            seen.add(t)
            clean.append(t)
    return clean


def chunk_list(lst: List, size: int) -> List[List]:
    """Split a list into chunks of given size."""
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def today_str() -> str:
    return date.today().isoformat()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
