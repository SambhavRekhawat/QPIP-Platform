"""
database/models.py
------------------
SQLAlchemy ORM models for all platform tables.
"""

from sqlalchemy import (
    Column, Integer, Float, String, Date, DateTime, Text,
    Boolean, ForeignKey, UniqueConstraint, Index, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.connection import Base


# ─── Prices ──────────────────────────────────────────────────────────────────
class Price(Base):
    __tablename__ = "prices"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    ticker     = Column(String(20), nullable=False, index=True)
    date       = Column(Date, nullable=False, index=True)
    open       = Column(Float)
    high       = Column(Float)
    low        = Column(Float)
    close      = Column(Float, nullable=False)
    adj_close  = Column(Float)
    volume     = Column(Float)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_price_ticker_date"),
        Index("ix_prices_ticker_date", "ticker", "date"),
    )

    def __repr__(self):
        return f"<Price {self.ticker} {self.date} close={self.close}>"


# ─── Returns ─────────────────────────────────────────────────────────────────
class Return(Base):
    __tablename__ = "returns"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    ticker           = Column(String(20), nullable=False, index=True)
    date             = Column(Date, nullable=False, index=True)
    daily_return     = Column(Float)
    log_return       = Column(Float)
    rolling_vol_21d  = Column(Float)
    rolling_vol_63d  = Column(Float)
    cumulative_return = Column(Float)
    benchmark_return = Column(Float)
    excess_return    = Column(Float)
    created_at       = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_return_ticker_date"),
    )


# ─── Fundamentals ────────────────────────────────────────────────────────────
class Fundamental(Base):
    __tablename__ = "fundamentals"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    ticker           = Column(String(20), nullable=False, index=True)
    as_of_date       = Column(Date, nullable=False)
    pe_ratio         = Column(Float)
    pb_ratio         = Column(Float)
    roe              = Column(Float)
    roce             = Column(Float)
    debt_to_equity   = Column(Float)
    revenue_growth   = Column(Float)
    profit_growth    = Column(Float)
    operating_margin = Column(Float)
    promoter_holding = Column(Float)
    market_cap       = Column(Float)
    eps              = Column(Float)
    dividend_yield   = Column(Float)
    current_ratio    = Column(Float)
    asset_turnover   = Column(Float)
    source           = Column(String(50), default="screener.in")
    created_at       = Column(DateTime, server_default=func.now())
    updated_at       = Column(DateTime, onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("ticker", "as_of_date", name="uq_fund_ticker_date"),
    )


# ─── Factors ─────────────────────────────────────────────────────────────────
class Factor(Base):
    __tablename__ = "factors"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    ticker               = Column(String(20), nullable=False, index=True)
    date                 = Column(Date, nullable=False, index=True)
    momentum_6m          = Column(Float)
    momentum_12m         = Column(Float)
    momentum_1m          = Column(Float)
    volatility_score     = Column(Float)
    quality_score        = Column(Float)
    value_score          = Column(Float)
    composite_score      = Column(Float)
    momentum_rank        = Column(Integer)
    quality_rank         = Column(Integer)
    value_rank           = Column(Integer)
    low_vol_rank         = Column(Integer)
    composite_rank       = Column(Integer)
    beta                 = Column(Float)
    created_at           = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_factor_ticker_date"),
    )


# ─── ML Signals ──────────────────────────────────────────────────────────────
class Signal(Base):
    __tablename__ = "signals"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    ticker               = Column(String(20), nullable=False, index=True)
    signal_date          = Column(Date, nullable=False, index=True)
    signal               = Column(String(10))   # BUY / HOLD / SELL
    model_name           = Column(String(50))
    buy_probability      = Column(Float)
    hold_probability     = Column(Float)
    sell_probability     = Column(Float)
    confidence           = Column(Float)
    shap_values          = Column(JSON)
    top_positive_drivers = Column(JSON)
    top_negative_drivers = Column(JSON)
    created_at           = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("ticker", "signal_date", "model_name",
                         name="uq_signal_ticker_date_model"),
    )


# ─── Portfolio Returns ────────────────────────────────────────────────────────
class PortfolioReturn(Base):
    __tablename__ = "portfolio_returns"

    id                    = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id          = Column(String(100), nullable=False, index=True)
    date                  = Column(Date, nullable=False)
    portfolio_return      = Column(Float)
    benchmark_return      = Column(Float)
    excess_return         = Column(Float)
    cumulative_portfolio  = Column(Float)
    cumulative_benchmark  = Column(Float)
    rolling_sharpe        = Column(Float)
    rolling_vol           = Column(Float)
    drawdown              = Column(Float)
    weighting_method      = Column(String(50))
    tickers               = Column(JSON)
    weights               = Column(JSON)
    created_at            = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("portfolio_id", "date", name="uq_portfolio_date"),
    )


# ─── Events / News ───────────────────────────────────────────────────────────
class Event(Base):
    __tablename__ = "events"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    ticker           = Column(String(20), index=True)
    event_date       = Column(Date, nullable=False, index=True)
    headline         = Column(Text)
    source           = Column(String(100))
    url              = Column(Text)
    event_type       = Column(String(50))   # earnings / macro / corp_action / block_deal
    sentiment_score  = Column(Float)        # -1 to +1
    sentiment_label  = Column(String(20))   # Positive / Neutral / Negative
    impact_score     = Column(Float)
    processed        = Column(Boolean, default=False)
    raw_content      = Column(Text)
    created_at       = Column(DateTime, server_default=func.now())


# ─── Backtest Results ─────────────────────────────────────────────────────────
class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name    = Column(String(100), nullable=False)
    run_date         = Column(DateTime, server_default=func.now())
    start_date       = Column(Date)
    end_date         = Column(Date)
    tickers          = Column(JSON)
    weighting_method = Column(String(50))
    cagr             = Column(Float)
    sharpe_ratio     = Column(Float)
    sortino_ratio    = Column(Float)
    information_ratio = Column(Float)
    alpha            = Column(Float)
    beta             = Column(Float)
    max_drawdown     = Column(Float)
    hit_ratio        = Column(Float)
    total_return     = Column(Float)
    benchmark_return = Column(Float)
    excess_return    = Column(Float)
    volatility       = Column(Float)
    num_rebalances   = Column(Integer)
    transaction_costs = Column(Float)
    monthly_returns  = Column(JSON)
    config_params    = Column(JSON)
    created_at       = Column(DateTime, server_default=func.now())
