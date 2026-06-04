"""
config.py
---------
Central configuration for the Quantitative Portfolio Intelligence Platform.
All environment variables, database settings, API keys, and platform constants live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Project Paths ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
EXPORTS_DIR = BASE_DIR / "exports"
MODELS_DIR = BASE_DIR / "ml_engine" / "saved_models"

for _dir in [LOGS_DIR, EXPORTS_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── PostgreSQL Database ──────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME", "quant_platform"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password"),
}

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

# ─── API Keys ─────────────────────────────────────────────────────────────────
NEWS_API_KEY     = os.getenv("NEWS_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

# ─── Market Data Settings ─────────────────────────────────────────────────────
BENCHMARK_TICKER   = "^NSEI"           # NIFTY 50
BENCHMARK_LABEL    = "NIFTY 50"
NSE_SUFFIX         = ".NS"             # yfinance NSE suffix
DATA_LOOKBACK_YEARS = 5
RISK_FREE_RATE      = 0.067            # India 10Y G-Sec approx
TRADING_DAYS_YEAR   = 252

# ─── Fundamental Scraping ─────────────────────────────────────────────────────
SCREENER_BASE_URL       = "https://www.screener.in/company/{ticker}/consolidated/"
SCREENER_REFRESH_DAY    = 4            # Friday (weekday index, 0=Mon)
SCREENER_TIMEOUT_SEC    = 15

# ─── Weighting Methods ────────────────────────────────────────────────────────
WEIGHTING_METHODS = ["Equal Weighted", "Market Cap Weighted", "Custom Weighted"]

# ─── Factor Settings ─────────────────────────────────────────────────────────
FACTOR_LOOKBACK = {
    "momentum_6m":  126,
    "momentum_12m": 252,
    "volatility":    63,
}

FACTOR_WEIGHTS = {
    "momentum": 0.25,
    "quality":  0.30,
    "value":    0.25,
    "low_vol":  0.20,
}

# ─── ML Engine ────────────────────────────────────────────────────────────────
ML_SIGNAL_THRESHOLD = {
    "buy":  0.60,
    "sell": 0.40,
}
ML_TEST_SIZE         = 0.20
ML_RANDOM_STATE      = 42
ML_CV_FOLDS          = 5

# ─── Backtesting ──────────────────────────────────────────────────────────────
BACKTEST_MONTHS         = 12
REBALANCE_FREQUENCY     = "ME"          # Month End
TRANSACTION_COST_BPS    = 20            # 20 bps one-way

# ─── Dashboard ────────────────────────────────────────────────────────────────
DASHBOARD_TITLE     = "Quantitative Portfolio Intelligence Platform"
DASHBOARD_SUBTITLE  = "Institutional Equity Research & Risk Analytics"
CHART_THEME         = "plotly_dark"
BRAND_COLOR         = "#00D4FF"
ACCENT_COLOR        = "#FF6B35"
SUCCESS_COLOR       = "#00C853"
DANGER_COLOR        = "#FF1744"
NEUTRAL_COLOR       = "#90A4AE"

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL  = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE   = str(LOGS_DIR / "platform.log")
LOG_ROTATION = "10 MB"

# ─── Default Universe (NSE Large-Cap) ────────────────────────────────────────
DEFAULT_TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
]