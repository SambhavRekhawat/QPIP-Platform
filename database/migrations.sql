-- ============================================================
-- migrations.sql
-- Quantitative Portfolio Intelligence Platform  (Project 5 + 6)
-- PostgreSQL Schema — deduplicated canonical version
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─── Core market data ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS prices (
    id          SERIAL PRIMARY KEY,
    ticker      VARCHAR(20)  NOT NULL,
    date        DATE         NOT NULL,
    open        FLOAT,
    high        FLOAT,
    low         FLOAT,
    close       FLOAT        NOT NULL,
    adj_close   FLOAT,
    volume      FLOAT,
    created_at  TIMESTAMP    DEFAULT NOW(),
    updated_at  TIMESTAMP,
    CONSTRAINT uq_price_ticker_date UNIQUE (ticker, date)
);
CREATE INDEX IF NOT EXISTS ix_prices_ticker_date  ON prices (ticker, date DESC);
CREATE INDEX IF NOT EXISTS ix_prices_date         ON prices (date DESC);
-- Partial index speeds large-universe queries (200-500 tickers)
CREATE INDEX IF NOT EXISTS ix_prices_ticker_close ON prices (ticker, date DESC) INCLUDE (close, volume);

CREATE TABLE IF NOT EXISTS returns (
    id                SERIAL PRIMARY KEY,
    ticker            VARCHAR(20)  NOT NULL,
    date              DATE         NOT NULL,
    daily_return      FLOAT,
    log_return        FLOAT,
    rolling_vol_21d   FLOAT,
    rolling_vol_63d   FLOAT,
    cumulative_return FLOAT,
    benchmark_return  FLOAT,
    excess_return     FLOAT,
    created_at        TIMESTAMP    DEFAULT NOW(),
    CONSTRAINT uq_return_ticker_date UNIQUE (ticker, date)
);
CREATE INDEX IF NOT EXISTS ix_returns_ticker_date ON returns (ticker, date DESC);

CREATE TABLE IF NOT EXISTS fundamentals (
    id               SERIAL PRIMARY KEY,
    ticker           VARCHAR(20)  NOT NULL,
    as_of_date       DATE         NOT NULL,
    pe_ratio         FLOAT,
    pb_ratio         FLOAT,
    roe              FLOAT,
    roce             FLOAT,
    debt_to_equity   FLOAT,
    revenue_growth   FLOAT,
    profit_growth    FLOAT,
    operating_margin FLOAT,
    promoter_holding FLOAT,
    market_cap       FLOAT,
    eps              FLOAT,
    dividend_yield   FLOAT,
    current_ratio    FLOAT,
    asset_turnover   FLOAT,
    source           VARCHAR(50)  DEFAULT 'screener.in',
    created_at       TIMESTAMP    DEFAULT NOW(),
    updated_at       TIMESTAMP,
    CONSTRAINT uq_fund_ticker_date UNIQUE (ticker, as_of_date)
);

CREATE TABLE IF NOT EXISTS factors (
    id               SERIAL PRIMARY KEY,
    ticker           VARCHAR(20)  NOT NULL,
    date             DATE         NOT NULL,
    momentum_6m      FLOAT,
    momentum_12m     FLOAT,
    momentum_1m      FLOAT,
    volatility_score FLOAT,
    quality_score    FLOAT,
    value_score      FLOAT,
    composite_score  FLOAT,
    momentum_rank    INTEGER,
    quality_rank     INTEGER,
    value_rank       INTEGER,
    low_vol_rank     INTEGER,
    composite_rank   INTEGER,
    beta             FLOAT,
    created_at       TIMESTAMP    DEFAULT NOW(),
    CONSTRAINT uq_factor_ticker_date UNIQUE (ticker, date)
);
CREATE INDEX IF NOT EXISTS ix_factors_ticker_date ON factors (ticker, date DESC);

CREATE TABLE IF NOT EXISTS signals (
    id                    SERIAL PRIMARY KEY,
    ticker                VARCHAR(20)  NOT NULL,
    signal_date           DATE         NOT NULL,
    signal                VARCHAR(10),
    model_name            VARCHAR(50),
    buy_probability       FLOAT,
    hold_probability      FLOAT,
    sell_probability      FLOAT,
    confidence            FLOAT,
    shap_values           JSONB,
    top_positive_drivers  JSONB,
    top_negative_drivers  JSONB,
    created_at            TIMESTAMP    DEFAULT NOW(),
    CONSTRAINT uq_signal_ticker_date_model UNIQUE (ticker, signal_date, model_name)
);
CREATE INDEX IF NOT EXISTS ix_signals_ticker_date ON signals (ticker, signal_date DESC);

CREATE TABLE IF NOT EXISTS portfolio_returns (
    id                   SERIAL PRIMARY KEY,
    portfolio_id         VARCHAR(100) NOT NULL,
    date                 DATE         NOT NULL,
    portfolio_return     FLOAT,
    benchmark_return     FLOAT,
    excess_return        FLOAT,
    cumulative_portfolio FLOAT,
    cumulative_benchmark FLOAT,
    rolling_sharpe       FLOAT,
    rolling_vol          FLOAT,
    drawdown             FLOAT,
    weighting_method     VARCHAR(50),
    tickers              JSONB,
    weights              JSONB,
    created_at           TIMESTAMP    DEFAULT NOW(),
    CONSTRAINT uq_portfolio_date UNIQUE (portfolio_id, date)
);

CREATE TABLE IF NOT EXISTS events (
    id              SERIAL PRIMARY KEY,
    ticker          VARCHAR(20),
    event_date      DATE         NOT NULL,
    headline        TEXT,
    source          VARCHAR(100),
    url             TEXT,
    event_type      VARCHAR(50),
    sentiment_score FLOAT,
    sentiment_label VARCHAR(20),
    impact_score    FLOAT,
    processed       BOOLEAN      DEFAULT FALSE,
    raw_content     TEXT,
    created_at      TIMESTAMP    DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_events_ticker_date ON events (ticker, event_date DESC);

CREATE TABLE IF NOT EXISTS backtest_results (
    id                SERIAL PRIMARY KEY,
    strategy_name     VARCHAR(100) NOT NULL,
    run_date          TIMESTAMP    DEFAULT NOW(),
    start_date        DATE,
    end_date          DATE,
    tickers           JSONB,
    weighting_method  VARCHAR(50),
    cagr              FLOAT,
    sharpe_ratio      FLOAT,
    sortino_ratio     FLOAT,
    information_ratio FLOAT,
    alpha             FLOAT,
    beta              FLOAT,
    max_drawdown      FLOAT,
    hit_ratio         FLOAT,
    total_return      FLOAT,
    benchmark_return  FLOAT,
    excess_return     FLOAT,
    volatility        FLOAT,
    num_rebalances    INTEGER,
    transaction_costs FLOAT,
    monthly_returns   JSONB,
    config_params     JSONB,
    created_at        TIMESTAMP    DEFAULT NOW()
);

-- ─── Risk & Simulation outputs ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS stress_test_results (
    id              SERIAL PRIMARY KEY,
    run_date        TIMESTAMP    DEFAULT NOW(),
    tickers         JSONB,
    weights         JSONB,
    n_simulations   INTEGER      DEFAULT 10000,
    holding_period  INTEGER      DEFAULT 1,
    vol_multiplier  FLOAT        DEFAULT 1.0,
    var_90          FLOAT,
    cvar_90         FLOAT,
    var_95          FLOAT,
    cvar_95         FLOAT,
    var_975         FLOAT,
    cvar_975        FLOAT,
    var_99          FLOAT,
    cvar_99         FLOAT,
    prob_loss       FLOAT,
    expected_loss   FLOAT,
    pnl_skew        FLOAT,
    pnl_kurtosis    FLOAT,
    scenario        VARCHAR(100) DEFAULT 'Monte Carlo',
    created_at      TIMESTAMP    DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_stress_run_date ON stress_test_results (run_date DESC);

CREATE TABLE IF NOT EXISTS var_results (
    id             SERIAL PRIMARY KEY,
    run_date       TIMESTAMP    DEFAULT NOW(),
    portfolio_id   VARCHAR(100),
    method         VARCHAR(30),   -- historical | parametric | monte_carlo
    confidence     FLOAT,
    holding_period INTEGER,
    var_value      FLOAT,
    cvar_value     FLOAT,
    tickers        JSONB,
    weights        JSONB,
    created_at     TIMESTAMP    DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS monte_carlo_results (
    id              SERIAL PRIMARY KEY,
    run_date        TIMESTAMP    DEFAULT NOW(),
    portfolio_id    VARCHAR(100),
    n_simulations   INTEGER,
    horizon_days    INTEGER,
    method          VARCHAR(30),   -- cholesky | gbm | copula
    mean_terminal   FLOAT,
    std_terminal    FLOAT,
    var_95          FLOAT,
    cvar_95         FLOAT,
    var_99          FLOAT,
    cvar_99         FLOAT,
    prob_loss       FLOAT,
    tickers         JSONB,
    weights         JSONB,
    config          JSONB,
    created_at      TIMESTAMP    DEFAULT NOW()
);

-- ─── Optimization outputs ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS optimization_results (
    id               SERIAL PRIMARY KEY,
    run_date         TIMESTAMP    DEFAULT NOW(),
    portfolio_id     VARCHAR(100),
    method           VARCHAR(50),  -- max_sharpe | min_vol | risk_parity | black_litterman | cvar
    tickers          JSONB,
    input_weights    JSONB,
    optimal_weights  JSONB,
    expected_return  FLOAT,
    expected_vol     FLOAT,
    sharpe_ratio     FLOAT,
    cvar_95          FLOAT,
    risk_contrib     JSONB,
    notes            TEXT,
    created_at       TIMESTAMP    DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_optim_portfolio ON optimization_results (portfolio_id, run_date DESC);

-- ─── Liquidity outputs ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS liquidity_risk_results (
    id               SERIAL PRIMARY KEY,
    run_date         TIMESTAMP    DEFAULT NOW(),
    portfolio_id     VARCHAR(100),
    tickers          JSONB,
    weights          JSONB,
    adv_scores       JSONB,
    liquidation_days JSONB,
    amihud_scores    JSONB,
    la_var_95        FLOAT,
    la_var_99        FLOAT,
    hhi_index        FLOAT,
    effective_n      FLOAT,
    alerts           JSONB,
    created_at       TIMESTAMP    DEFAULT NOW()
);

-- ─── Factor risk outputs ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS factor_risk_results (
    id              SERIAL PRIMARY KEY,
    run_date        TIMESTAMP    DEFAULT NOW(),
    portfolio_id    VARCHAR(100),
    r_squared       FLOAT,
    factor_betas    JSONB,
    factor_pvalues  JSONB,
    risk_contrib    JSONB,
    idiosyncratic   FLOAT,
    created_at      TIMESTAMP    DEFAULT NOW()
);

-- ─── Useful views ─────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW latest_signals AS
    SELECT DISTINCT ON (ticker)
           ticker, signal_date, signal, model_name,
           buy_probability, confidence,
           top_positive_drivers, top_negative_drivers
    FROM   signals
    ORDER  BY ticker, signal_date DESC;

CREATE OR REPLACE VIEW latest_fundamentals AS
    SELECT DISTINCT ON (ticker)
           ticker, as_of_date, pe_ratio, pb_ratio, roe, roce,
           debt_to_equity, revenue_growth, profit_growth,
           operating_margin, promoter_holding, market_cap
    FROM   fundamentals
    ORDER  BY ticker, as_of_date DESC;

CREATE OR REPLACE VIEW latest_factors AS
    SELECT DISTINCT ON (ticker)
           ticker, date, momentum_6m, momentum_12m,
           quality_score, value_score, composite_score,
           composite_rank, beta
    FROM   factors
    ORDER  BY ticker, date DESC;
