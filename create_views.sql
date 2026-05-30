-- Run once to create the three "latest" views the Research Copilot needs.
-- Safe to run multiple times (CREATE OR REPLACE).

CREATE OR REPLACE VIEW latest_signals AS
    SELECT DISTINCT ON (ticker)
           ticker, signal_date, signal, model_name,
           buy_probability, hold_probability, sell_probability, confidence,
           shap_values, top_positive_drivers, top_negative_drivers
    FROM   signals
    ORDER  BY ticker, signal_date DESC;

CREATE OR REPLACE VIEW latest_fundamentals AS
    SELECT DISTINCT ON (ticker)
           ticker, as_of_date, pe_ratio, pb_ratio, roe, roce,
           debt_to_equity, revenue_growth, profit_growth,
           operating_margin, promoter_holding, market_cap, dividend_yield
    FROM   fundamentals
    ORDER  BY ticker, as_of_date DESC;

CREATE OR REPLACE VIEW latest_factors AS
    SELECT DISTINCT ON (ticker)
           ticker, date, momentum_1m, momentum_6m, momentum_12m,
           quality_score, value_score, volatility_score, composite_score,
           momentum_rank, quality_rank, value_rank, composite_rank, beta
    FROM   factors
    ORDER  BY ticker, date DESC;
