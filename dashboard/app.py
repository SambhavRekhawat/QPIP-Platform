"""
dashboard/app.py
-----------------
Institutional-grade Streamlit dashboard for the
Quantitative Portfolio Intelligence Platform.

Includes:
  - Full GLOSSARY with plain-English explanations for every metric
  - metric_with_help()  — st.metric + hover tooltip in one call
  - section_intro()     — collapsible "What does this tab show?" explainer
  - glossary tab        — searchable reference for all terms

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import streamlit as st

warnings.filterwarnings("ignore")

from config import (
    DASHBOARD_TITLE, DASHBOARD_SUBTITLE, BRAND_COLOR, ACCENT_COLOR,
    SUCCESS_COLOR, DANGER_COLOR, NEUTRAL_COLOR, DEFAULT_TICKERS,
    WEIGHTING_METHODS, BENCHMARK_LABEL,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Theme ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root { --brand:#00D4FF; --accent:#FF6B35; --success:#00C853;
            --danger:#FF1744; --border:#1E2A3A; }
    .stApp { background-color: #060B14; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0A1628 0%,#060B14 100%);
        border-right: 1px solid var(--border);
    }
    [data-testid="metric-container"] {
        background:#0E1A2E; border:1px solid var(--border);
        border-radius:8px; padding:12px;
    }
    h1,h2,h3 { color:#E0E8F0 !important; }
    .stMarkdown h4 { color:var(--brand); }
    .stTabs [data-baseweb="tab-list"] { background:#0A1628; border-radius:8px; gap:4px; }
    .stTabs [data-baseweb="tab"] { background:transparent; color:#90A4AE;
        border-radius:6px; font-weight:500; }
    .stTabs [aria-selected="true"] { background:#00D4FF22; color:#00D4FF !important;
        border-bottom:2px solid #00D4FF; }
    .signal-buy  { background:rgba(0,200,83,0.13); color:#00C853;
        border:1px solid rgba(0,200,83,0.4); border-radius:4px; padding:2px 8px; font-weight:700; }
    .signal-sell { background:rgba(255,23,68,0.13); color:#FF1744;
        border:1px solid rgba(255,23,68,0.4); border-radius:4px; padding:2px 8px; font-weight:700; }
    .signal-hold { background:rgba(255,107,53,0.13); color:#FF6B35;
        border:1px solid rgba(255,107,53,0.4); border-radius:4px; padding:2px 8px; font-weight:700; }
    .stDataFrame { border:1px solid var(--border); border-radius:8px; }
    hr { border-color:var(--border); }
    /* Glossary cards */
    .g-card { background:#0E1A2E; border:1px solid #1E2A3A; border-radius:10px;
               padding:14px 16px; margin-bottom:10px; }
    .g-term { font-size:15px; font-weight:600; color:#00D4FF; margin-bottom:4px; }
    .g-formula { font-family:monospace; font-size:12px; color:#607080;
                 background:#060B14; padding:4px 8px; border-radius:4px;
                 margin:6px 0; display:inline-block; }
    .g-good { color:#00C853; font-size:12px; }
    .g-bad  { color:#FF1744; font-size:12px; }
    .g-tag  { display:inline-block; font-size:11px; padding:2px 8px;
               border-radius:99px; margin-right:4px; font-weight:500; }
    .g-tag-risk  { background:rgba(255,23,68,0.15);   color:#FF1744; }
    .g-tag-perf  { background:rgba(0,200,83,0.15);    color:#00C853; }
    .g-tag-factor{ background:rgba(0,212,255,0.15);   color:#00D4FF; }
    .g-tag-ml    { background:rgba(255,107,53,0.15);  color:#FF6B35; }
</style>
""", unsafe_allow_html=True)


# ─── Plotly Theme ─────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#060B14", plot_bgcolor="#060B14",
    font=dict(family="Inter, sans-serif", color="#90A4AE", size=12),
    xaxis=dict(gridcolor="#0E1A2E", zeroline=False),
    yaxis=dict(gridcolor="#0E1A2E", zeroline=False),
    legend=dict(bgcolor="#0A1628", bordercolor="#1E2A3A", borderwidth=1),
    margin=dict(l=40, r=20, t=40, b=40),
)


# ══════════════════════════════════════════════════════════════════════════════
# GLOSSARY — single source of truth for every metric explanation
# ══════════════════════════════════════════════════════════════════════════════

GLOSSARY: Dict[str, Dict] = {

    # ── Performance metrics ───────────────────────────────────────────────────
    "Total Return": {
        "short": "How much ₹100 invested grew (or shrank) over the full period.",
        "long": (
            "The simple percentage gain from the first to the last date in the dataset. "
            "A Total Return of +35% means ₹100 became ₹135. It does NOT account for "
            "how long that took — use CAGR for a time-adjusted view."
        ),
        "formula": "(Final Value / Initial Value) − 1",
        "good": "Higher is better. Compare to NIFTY 50's return over the same period.",
        "benchmark": "NIFTY 50 has historically returned ~12–14% per year.",
        "tag": "perf",
    },
    "CAGR": {
        "short": "Compound Annual Growth Rate — the steady annual return that would produce the same result.",
        "long": (
            "CAGR answers: 'If my portfolio grew at the same rate every year, what would "
            "that rate be?' It removes the noise of volatile years and gives you a single "
            "comparable number. A CAGR of 15% means your money doubled roughly every 5 years."
        ),
        "formula": "(Final Value / Initial Value)^(252 / trading_days) − 1",
        "good": "For Indian equities, >12% CAGR beats the long-run NIFTY 50 average.",
        "benchmark": "NIFTY 50 CAGR: ~12–14% over 10 years.",
        "tag": "perf",
    },
    "Ann. Return": {
        "short": "Annualised return — same as CAGR. The portfolio's yearly growth rate.",
        "long": (
            "Scales your actual returns up or down to a per-year figure so you can compare "
            "portfolios measured over different time periods. If you earned 6% over 6 months, "
            "the annualised return is approximately 12%."
        ),
        "formula": "(1 + total_return)^(252 / trading_days) − 1",
        "good": ">12% beats the long-run NIFTY 50 average.",
        "benchmark": "NIFTY 50: ~12–14% annualised.",
        "tag": "perf",
    },
    "Sharpe Ratio": {
        "short": "Return earned per unit of total risk. Higher = more efficient portfolio.",
        "long": (
            "The Sharpe Ratio asks: 'How much extra return did I earn for each unit of "
            "volatility I took on?' A ratio of 1.0 means you earned 1% of excess return "
            "for every 1% of annualised volatility. It penalises both upside and downside "
            "swings equally. Risk-free rate used: India 10Y G-Sec (~6.7%)."
        ),
        "formula": "(Ann. Return − Risk-Free Rate) / Ann. Volatility",
        "good": ">1.0 is good. >1.5 is excellent. Most Indian MFs sit between 0.5–0.9.",
        "benchmark": "Typical active Indian equity fund: 0.5–0.9.",
        "tag": "perf",
    },
    "Sortino Ratio": {
        "short": "Like Sharpe, but only penalises downside volatility — not upside swings.",
        "long": (
            "The Sortino Ratio is a refinement of Sharpe. It recognises that investors "
            "don't mind large gains — only large losses. It divides excess return by "
            "'downside deviation' (standard deviation of negative returns only). "
            "A Sortino of 1.5 with a Sharpe of 1.0 suggests most of your volatility "
            "is on the upside — a very good sign."
        ),
        "formula": "(Ann. Return − Risk-Free Rate) / Downside Deviation",
        "good": ">1.5 is excellent. Always compare to the same portfolio's Sharpe.",
        "benchmark": "Should be higher than Sharpe Ratio for a good portfolio.",
        "tag": "perf",
    },
    "Information Ratio": {
        "short": "How consistently the portfolio beats its benchmark per unit of active risk.",
        "long": (
            "The Information Ratio (IR) measures the quality of active management. "
            "It divides the portfolio's average daily excess return over NIFTY 50 by the "
            "standard deviation of those excess returns (called 'Tracking Error'). "
            "A high IR means you beat the benchmark consistently — not just occasionally."
        ),
        "formula": "Mean(Portfolio Return − Benchmark Return) / Std(Excess Returns) × √252",
        "good": ">0.5 is good. >1.0 is exceptional. Many fund managers never reach 0.5.",
        "benchmark": "Top-quartile Indian active funds: IR ~0.4–0.7.",
        "tag": "perf",
    },
    "Hit Ratio": {
        "short": "Percentage of days the portfolio outperformed NIFTY 50.",
        "long": (
            "Simple but powerful: on what fraction of trading days did your portfolio "
            "return more than the NIFTY 50? A hit ratio of 55% means you beat the "
            "benchmark on more than half of all days. Combined with the magnitude of "
            "wins vs losses, this tells you the quality of active returns."
        ),
        "formula": "Days(Portfolio Return > Benchmark Return) / Total Days",
        "good": ">52% is meaningful. >55% is strong over a full year.",
        "benchmark": "Random chance = 50%. Good active management: 52–58%.",
        "tag": "perf",
    },

    # ── Risk metrics ──────────────────────────────────────────────────────────
    "Max Drawdown": {
        "short": "The largest peak-to-trough loss. Worst-case scenario over the period.",
        "long": (
            "Max Drawdown answers: 'If I bought at the worst possible time and sold at "
            "the bottom, how much would I have lost?' For example, −25% means at some "
            "point your portfolio fell 25% from its previous peak before recovering. "
            "This is the single most important metric for risk-averse investors."
        ),
        "formula": "min((Portfolio Value − Rolling Peak) / Rolling Peak)",
        "good": "Closer to 0% is better. <−20% is concerning for a diversified portfolio.",
        "benchmark": "NIFTY 50 fell ~60% in 2008 and ~38% in 2020.",
        "tag": "risk",
    },
    "Ann. Volatility": {
        "short": "Annualised standard deviation of daily returns — how much the portfolio swings.",
        "long": (
            "Volatility measures how much your portfolio's daily returns scatter around "
            "their average. A volatility of 20% means in a typical year your portfolio "
            "could swing ±20% from its average return. Lower is better for the same level "
            "of return — which is what the Sharpe Ratio captures."
        ),
        "formula": "Std(Daily Returns) × √252",
        "good": "<18% is low for Indian equities. >25% is high.",
        "benchmark": "NIFTY 50 historical volatility: ~16–20%.",
        "tag": "risk",
    },
    "Beta": {
        "short": "How much the portfolio moves relative to NIFTY 50. 1.0 = moves with the market.",
        "long": (
            "Beta measures market sensitivity. A Beta of 1.2 means when NIFTY 50 rises "
            "1%, your portfolio tends to rise 1.2% — and falls 1.2% when NIFTY falls 1%. "
            "A Beta of 0.8 is more defensive. Beta above 1.3 is considered high-risk. "
            "Beta measures systematic risk — the part you cannot diversify away."
        ),
        "formula": "Cov(Portfolio Returns, Benchmark Returns) / Var(Benchmark Returns)",
        "good": "0.8–1.1 for balanced exposure. >1.3 is aggressive.",
        "benchmark": "By definition, NIFTY 50 Beta = 1.0.",
        "tag": "risk",
    },
    "Alpha": {
        "short": "Excess return after adjusting for market risk (Beta). True value added.",
        "long": (
            "Alpha is the return your portfolio earned ABOVE what you would expect given "
            "how much market risk (Beta) you took. If the market rose 10%, your Beta is 1.2, "
            "and your portfolio rose 14%, your Alpha = 14% − (6.7% + 1.2 × (10% − 6.7%)) = ~3.3%. "
            "Positive Alpha means skilled stock selection. Negative Alpha means you underperformed "
            "for the risk taken."
        ),
        "formula": "Portfolio Return − (Risk-Free Rate + Beta × (Market Return − Risk-Free Rate))",
        "good": ">0% means you beat the market on a risk-adjusted basis.",
        "benchmark": "Most active funds in India deliver Alpha of −2% to +3% annually.",
        "tag": "risk",
    },
    "VaR (95%)": {
        "short": "Value at Risk — the worst daily loss you'd expect 95% of the time.",
        "long": (
            "Historical VaR at 95% confidence answers: 'On a bad day, how much could I lose?' "
            "A VaR of 2.1% means that on 95 out of 100 trading days, you lost no more than 2.1%. "
            "On the remaining 5 days (tail events), you could lose more. "
            "VaR does NOT tell you how bad those 5 worst days are — that's CVaR."
        ),
        "formula": "5th percentile of historical daily return distribution",
        "good": "<2% daily VaR is reasonable for a diversified Indian equity portfolio.",
        "benchmark": "NIFTY 50 daily VaR (95%): ~1.5–2.0%.",
        "tag": "risk",
    },
    "CVaR (95%)": {
        "short": "Conditional VaR — average loss on the worst 5% of days. Tail risk measure.",
        "long": (
            "CVaR (also called Expected Shortfall) goes beyond VaR by asking: "
            "'On those bad days when I exceed the VaR threshold, what do I lose on average?' "
            "It captures the severity of tail losses, not just their threshold. "
            "CVaR is always worse (higher) than VaR and is preferred by risk managers "
            "for stress-testing."
        ),
        "formula": "Mean(Daily Returns | Daily Return < VaR threshold)",
        "good": "Should be <3–4% for a diversified portfolio.",
        "benchmark": "NIFTY 50 CVaR (95%): ~2.5–3.5% on bad days.",
        "tag": "risk",
    },
    "Calmar Ratio": {
        "short": "CAGR divided by Max Drawdown. Reward-per-unit of worst-case loss.",
        "long": (
            "The Calmar Ratio rewards strategies that grow steadily without catastrophic "
            "drawdowns. A Calmar of 0.5 means for every 1% of max drawdown, you earned "
            "0.5% of annual return. Strategies with high Calmar ratios are particularly "
            "valued by institutional investors who cannot afford large losses."
        ),
        "formula": "CAGR / |Max Drawdown|",
        "good": ">0.5 is acceptable. >1.0 is excellent.",
        "benchmark": "NIFTY 50 Calmar: ~0.3–0.5 over most rolling 3-year periods.",
        "tag": "risk",
    },

    # ── Factor scores ─────────────────────────────────────────────────────────
    "Momentum Score": {
        "short": "Stocks that have risen strongly in the past 6–12 months tend to keep rising.",
        "long": (
            "The Momentum factor is based on a well-documented market anomaly: stocks that "
            "have outperformed over the past 6–12 months (skipping the last 1 month to avoid "
            "short-term reversal) tend to continue outperforming over the next 3–6 months. "
            "We combine 1-month, 6-month, and 12-month returns, then cross-sectionally "
            "z-score them so the score is relative to all stocks in the universe. "
            "Score > 0 means above-average momentum."
        ),
        "formula": "0.30×z(1M_ret) + 0.35×z(6M_ret) + 0.35×z(12M_ret)",
        "good": "Score > +1.0 is strong momentum. < −1.0 is weak (possible mean-reversion candidate).",
        "benchmark": "Scores are z-scored — 0.0 is median, ±1.0 is one standard deviation.",
        "tag": "factor",
    },
    "Quality Score": {
        "short": "Companies with high ROE, low debt, and growing profits score higher.",
        "long": (
            "Quality investing targets companies with durable competitive advantages: "
            "high return on equity (ROE), high return on capital employed (ROCE), "
            "healthy operating margins, low debt, strong promoter confidence (high promoter holding), "
            "and consistent revenue and profit growth. Each metric is cross-sectionally z-scored "
            "and weighted. A high quality score means the company is financially stronger "
            "than most peers in the universe."
        ),
        "formula": "Weighted z-score of ROE, ROCE, OPM, promoter%, D/E, revenue growth, profit growth",
        "good": "Score > +1.0 signals a high-quality business.",
        "benchmark": "Scores are z-scored relative to your universe.",
        "tag": "factor",
    },
    "Value Score": {
        "short": "Stocks trading cheaply relative to earnings and book value score higher.",
        "long": (
            "Value investing looks for stocks the market has underpriced relative to their "
            "fundamentals. We use PE ratio (lower = cheaper), PB ratio (lower = cheaper), "
            "and dividend yield (higher = more value). Each metric is winsorised at the 5th/95th "
            "percentile to remove extreme outliers (e.g. loss-making companies with infinite PE), "
            "then cross-sectionally z-scored. A high value score means the stock looks cheap "
            "relative to its peers."
        ),
        "formula": "Weighted z-score of (−PE), (−PB), (+Dividend Yield)",
        "good": "Score > +1.0 suggests the stock is undervalued vs peers.",
        "benchmark": "Scores are z-scored — use relative comparisons within your universe.",
        "tag": "factor",
    },
    "Low Vol Score": {
        "short": "Stocks with lower price volatility historically deliver better risk-adjusted returns.",
        "long": (
            "The Low Volatility anomaly is one of the most robust in finance: contrary to "
            "standard theory, lower-volatility stocks tend to outperform higher-volatility "
            "stocks on a risk-adjusted basis. We compute 63-day realised volatility for each "
            "stock, then invert it (lower vol → higher score) and z-score cross-sectionally. "
            "This factor pairs well with momentum and quality."
        ),
        "formula": "cross_sectional_zscore(−realised_volatility_63d × √252)",
        "good": "Score > +1.0 means the stock is less volatile than most peers.",
        "benchmark": "Scores are z-scored — 0.0 is the median volatility in the universe.",
        "tag": "factor",
    },
    "Composite Score": {
        "short": "Blended score combining all four factors. The overall rank of each stock.",
        "long": (
            "The Composite Score is a weighted average of the four factor scores: "
            "Momentum (25%), Quality (30%), Value (25%), and Low Volatility (20%). "
            "These weights reflect a tilt toward quality and value — a common institutional "
            "approach. Stocks ranking high on multiple factors simultaneously are most "
            "attractive. The composite rank shows each stock's position in your universe "
            "from 1 (best) to N (worst)."
        ),
        "formula": "0.25×Momentum + 0.30×Quality + 0.25×Value + 0.20×Low_Vol",
        "good": "Composite rank 1 = best stock in universe on combined factors.",
        "benchmark": "Composite score > +0.5 is above average on all factors combined.",
        "tag": "factor",
    },

    # ── ML / Signal metrics ───────────────────────────────────────────────────
    "BUY Signal": {
        "short": "ML ensemble predicts the stock will outperform NIFTY 50 over the next ~21 days.",
        "long": (
            "A BUY signal is generated when the ensemble of XGBoost, LightGBM, and Random Forest "
            "models assigns a probability ≥ 60% to the BUY class. The models are trained on "
            "factor scores, price momentum, fundamental metrics, and sentiment data. "
            "Labels were generated using rank-based forward excess returns: stocks in the top "
            "40% of forward performance vs NIFTY 50 were labelled BUY."
        ),
        "formula": "BUY if P(outperform) ≥ 0.60 from ensemble model",
        "good": "Confidence > 70% makes a BUY signal more reliable.",
        "benchmark": "With random chance, BUY accuracy would be ~40% (class proportion).",
        "tag": "ml",
    },
    "SELL Signal": {
        "short": "ML ensemble predicts the stock will underperform NIFTY 50 over the next ~21 days.",
        "long": (
            "A SELL signal indicates the model assigns a high probability to the SELL class "
            "(bottom ~35% of forward performance). This does not mean the stock will fall — "
            "only that it is likely to underperform the NIFTY 50 benchmark. "
            "In a rising market, a SELL-rated stock might still go up — just less than NIFTY."
        ),
        "formula": "SELL if P(underperform) ≥ 0.60 from ensemble model",
        "good": "A SELL with confidence >70% suggests multiple factors are pointing negative.",
        "benchmark": "This is a relative signal — always vs NIFTY 50, not absolute direction.",
        "tag": "ml",
    },
    "HOLD Signal": {
        "short": "ML ensemble sees no strong directional edge — expected to be in line with NIFTY 50.",
        "long": (
            "HOLD means the model's probabilities are roughly balanced between BUY and SELL, "
            "or no single signal crosses the 60% threshold. It is not a recommendation to "
            "exit — it means the model has low conviction and the stock is expected to roughly "
            "track the benchmark. HOLD is appropriate when factor scores are mixed."
        ),
        "formula": "Neither BUY nor SELL threshold reached",
        "good": "A portfolio of mostly HOLD signals has low active risk vs benchmark.",
        "benchmark": "No edge either way vs NIFTY 50.",
        "tag": "ml",
    },
    "Model Confidence": {
        "short": "The maximum class probability from the ensemble — how sure the model is.",
        "long": (
            "Confidence is simply the highest probability across the three classes (BUY, HOLD, SELL) "
            "from the ensemble model. A confidence of 72% on BUY means the model assigns 72% "
            "probability to that stock outperforming. Confidence below 55% on any signal should "
            "be treated with caution — the model is essentially uncertain."
        ),
        "formula": "max(P(BUY), P(HOLD), P(SELL)) from ensemble",
        "good": ">65% is meaningful. >75% is high conviction.",
        "benchmark": "Random 3-class model would have confidence ~33%.",
        "tag": "ml",
    },
    "SHAP Value": {
        "short": "How much each feature pushed the model's prediction toward BUY or SELL.",
        "long": (
            "SHAP (SHapley Additive exPlanations) assigns each input feature a contribution "
            "score for a specific prediction. A positive SHAP value for 'momentum_score' means "
            "that feature pushed the model toward BUY. A negative SHAP on 'pe_ratio' means "
            "the high PE ratio pushed the model toward SELL. SHAP values always sum to the "
            "model's output (log-odds of BUY). This makes the model's reasoning fully transparent."
        ),
        "formula": "Shapley values from cooperative game theory applied to ML features",
        "good": "Larger absolute values = more important driver of that specific prediction.",
        "benchmark": "SHAP = 0 means the feature had no effect on this prediction.",
        "tag": "ml",
    },

    # ── Portfolio construction ─────────────────────────────────────────────────
    "Equal Weighted": {
        "short": "Each stock gets the same portfolio weight — simple and diversified.",
        "long": (
            "Equal Weighting assigns 1/N weight to each stock (e.g. 20% each for 5 stocks). "
            "It is the simplest approach and has a natural rebalancing benefit — it "
            "systematically buys stocks that have fallen and trims those that have risen. "
            "Research shows equal-weighted portfolios often outperform cap-weighted over long periods, "
            "but they require more rebalancing and may have higher transaction costs."
        ),
        "formula": "w_i = 1 / N for all stocks",
        "good": "Best for beginners or when you have no strong conviction on individual stocks.",
        "benchmark": "Simple to implement; no optimisation required.",
        "tag": "factor",
    },
    "Market Cap Weighted": {
        "short": "Larger companies get bigger weights — mirrors how NIFTY 50 is constructed.",
        "long": (
            "Market Cap Weighting assigns weight proportional to each company's total market "
            "capitalisation. RELIANCE (₹18L Cr) would get a much larger weight than a ₹1L Cr "
            "mid-cap. This naturally concentrates the portfolio in the largest, most liquid companies "
            "and minimises turnover. However, it can lead to over-concentration and 'buying high' "
            "as large-caps become more expensive."
        ),
        "formula": "w_i = Market_Cap_i / Sum(Market_Caps)",
        "good": "Suitable when you want index-like behaviour with lower tracking error.",
        "benchmark": "How NIFTY 50 and most ETFs are constructed.",
        "tag": "factor",
    },
    "Correlation": {
        "short": "How closely two stocks move together. +1 = identical, −1 = opposite, 0 = unrelated.",
        "long": (
            "Correlation measures the statistical relationship between two return series. "
            "A high positive correlation (>0.7) between two stocks means they tend to rise "
            "and fall together — they offer little diversification benefit. "
            "A low or negative correlation means they move independently, which reduces "
            "portfolio volatility. Good portfolios deliberately include stocks with low "
            "pairwise correlations."
        ),
        "formula": "Cov(R_A, R_B) / (Std(R_A) × Std(R_B))",
        "good": "Average pairwise correlation <0.5 suggests good diversification.",
        "benchmark": "Stocks in the same sector often have correlation >0.7.",
        "tag": "risk",
    },

    # ── Optimization methods ──────────────────────────────────────────────────
    "Efficient Frontier": {
        "short": "The curve of portfolios offering the highest return for every level of risk.",
        "long": (
            "Every point on the Efficient Frontier is a portfolio that cannot be improved — "
            "you cannot get more return without taking more risk. Points below the frontier "
            "are inefficient: you could get the same return with less risk by moving up to the curve. "
            "The frontier is generated by solving thousands of optimisation problems across a grid "
            "of target returns. Your current portfolio almost certainly sits below the frontier."
        ),
        "formula": "min w^T Σ w  subject to: w^T μ = target_return, Σw=1, w≥0",
        "good": "Move your portfolio toward the frontier to improve risk-adjusted returns.",
        "benchmark": "The Maximum Sharpe portfolio sits on the upper portion of the frontier.",
        "tag": "factor",
    },
    "Sharpe Optimisation": {
        "short": "Find weights that maximise the Sharpe Ratio — return per unit of risk.",
        "long": (
            "Mean-variance optimisation (Markowitz 1952) finds the portfolio weights that "
            "maximise the Sharpe Ratio subject to constraints (no shorts, max single weight). "
            "It uses the historical covariance matrix and mean returns as inputs. "
            "The result is mathematically optimal for the estimation window — but estimation "
            "error means the real-world best may differ. Use alongside Risk Parity for robustness."
        ),
        "formula": "max (w^T μ − r_f) / √(w^T Σ w)  subject to Σw=1, 0≤w≤max_weight",
        "good": "Optimised Sharpe should exceed your current portfolio's Sharpe by at least 0.1.",
        "benchmark": "Typical improvement over equal-weight: +0.1 to +0.4 Sharpe.",
        "tag": "factor",
    },
    "Risk Parity": {
        "short": "Allocate so every stock contributes equally to total portfolio risk.",
        "long": (
            "Risk Parity (Bridgewater-style) ignores return forecasts entirely. "
            "It solves for weights where each asset's marginal risk contribution equals 1/N of "
            "total portfolio risk. This means low-volatility assets get higher weights. "
            "It is more robust than Sharpe optimisation because it doesn't depend on uncertain "
            "return estimates — only the covariance matrix. The Spinu (2013) convex formulation "
            "is used: minimise 0.5 w^T Σ w − (1/N)·Σ log(w_i)."
        ),
        "formula": "w_i·(Σw)_i = σ_portfolio / N  for all i",
        "good": "Leads to more stable portfolios than Sharpe optimisation over long periods.",
        "benchmark": "All Risk Parity allocations have equal % contribution to total VaR.",
        "tag": "factor",
    },
    "Black-Litterman": {
        "short": "Blend market-implied returns with your own views to produce optimal weights.",
        "long": (
            "Black-Litterman (1990, Goldman Sachs) solves a key problem with Sharpe optimisation: "
            "tiny changes in return estimates cause wild changes in weights. "
            "BL starts from market equilibrium implied returns (CAPM), then blends them with "
            "investor views using Bayesian updating. If you have no views, you get the market portfolio. "
            "This platform uses ML signal probabilities as the view inputs — BUY = +2% view, SELL = −2%."
        ),
        "formula": "μ_BL = [(τΣ)⁻¹ + P^T Ω⁻¹ P]⁻¹ · [(τΣ)⁻¹ Π + P^T Ω⁻¹ Q]",
        "good": "More stable weight changes than pure Sharpe optimisation.",
        "benchmark": "Without views, converges to market cap weighted portfolio.",
        "tag": "factor",
    },
    "CVaR Optimisation": {
        "short": "Find weights that minimise the average loss in the worst 5% of scenarios.",
        "long": (
            "CVaR (Expected Shortfall) optimisation minimises tail risk directly, using the "
            "Rockafellar-Uryasev (2000) linear programming formulation. Unlike Sharpe optimisation "
            "which assumes returns are normally distributed, CVaR optimisation uses the actual "
            "historical return distribution including fat tails and skewness. "
            "It is particularly valuable for asymmetric or skewed portfolios."
        ),
        "formula": "min ζ + 1/((1−α)T) · Σ max(−r_t·w − ζ, 0)",
        "good": "Use when your portfolio has concentrated positions or asymmetric risk.",
        "benchmark": "CVaR-optimised portfolios typically have lower Max Drawdown than Sharpe-optimised.",
        "tag": "risk",
    },
    "Component VaR": {
        "short": "Each position's % contribution to total portfolio VaR. Sums to 100%.",
        "long": (
            "Component VaR decomposes portfolio risk to individual positions. "
            "A stock can have a large weight but small Component VaR if it is uncorrelated "
            "with the rest of the portfolio — and vice versa. "
            "The marginal VaR of position i = z × (Σw)_i / σ_portfolio. "
            "Component VaR = marginal VaR × weight. "
            "This is the key metric for identifying which positions to reduce to lower total risk."
        ),
        "formula": "CVaR_i = w_i · ∂VaR/∂w_i = z · w_i · (Σw)_i / σ_portfolio",
        "good": "Any position with >30% Component VaR in a 10-stock portfolio is a concentration risk.",
        "benchmark": "Equal Component VaR = each position contributes 1/N of total risk (Risk Parity).",
        "tag": "risk",
    },
    "Amihud Illiquidity": {
        "short": "Measures how much price moves per unit of trading volume. Higher = more illiquid.",
        "long": (
            "The Amihud (2002) illiquidity ratio is |daily return| / (daily trading volume in ₹). "
            "A high ratio means a small amount of trading causes a large price move — the stock "
            "is illiquid. Liquid stocks (RELIANCE, TCS) have very low Amihud ratios. "
            "Small-cap stocks can have ratios 100× higher. High illiquidity means you cannot "
            "exit your position quickly without moving the price against you."
        ),
        "formula": "Illiquidity = (1/T) · Σ |R_t| / (₹Volume_t)",
        "good": "Lower is more liquid. Flag stocks with Amihud ratio >0.01 for large positions.",
        "benchmark": "Nifty 50 stocks typically have ratios near 0.0001–0.001.",
        "tag": "risk",
    },
    "Liquidation Horizon": {
        "short": "Estimated number of days to exit your full position without excessive market impact.",
        "long": (
            "Liquidation horizon = position size / (participation rate × average daily volume). "
            "A participation rate of 20% means you trade at most 20% of the day's volume "
            "to avoid moving the price against yourself. "
            "If you hold ₹50L of a stock that trades ₹1Cr/day, your liquidation horizon "
            "at 20% participation = ₹50L / (20% × ₹1Cr) = 2.5 days. "
            "This matters enormously in a crisis when you need to exit quickly."
        ),
        "formula": "Horizon = position_value / (participation_rate × ADV_₹)",
        "good": "<5 days for any position is considered liquid.",
        "benchmark": "Nifty 50 stocks: <1 day for typical institutional position sizes.",
        "tag": "risk",
    },
    "GBM Simulation": {
        "short": "Geometric Brownian Motion — simulates future portfolio price paths under random shocks.",
        "long": (
            "GBM is the mathematical model underlying the Black-Scholes option pricing model. "
            "It simulates a portfolio's future value as: dS = S·(μ·dt + σ·√dt·Z) "
            "where μ is the drift (expected return), σ is volatility, and Z is a random shock. "
            "Running 10,000 paths creates a cone of possible futures. "
            "The probability cone (10th–90th percentile band) shows the range of plausible outcomes. "
            "GBM assumes log-normally distributed returns — it underestimates fat tails."
        ),
        "formula": "S(t) = S(0) · exp((μ − σ²/2)·t + σ·√t·Z)",
        "good": "The wider the cone, the more uncertain (high volatility) the portfolio.",
        "benchmark": "A 1-year GBM cone for NIFTY 50 spans roughly ±30% from current level.",
        "tag": "risk",
    },
    "Concentration Risk (HHI)": {
        "short": "Herfindahl-Hirschman Index — measures portfolio concentration. Higher = more concentrated.",
        "long": (
            "The HHI is the sum of squared portfolio weights. A perfectly equal-weighted portfolio "
            "of 10 stocks has HHI = 10 × (0.1)² = 0.10. A portfolio where one stock has 80% weight "
            "has HHI ≈ 0.64. Regulators use HHI to measure market concentration; we use it "
            "to measure portfolio concentration risk. "
            "Effective N = 1/HHI gives the equivalent number of equal-weight positions."
        ),
        "formula": "HHI = Σ w_i²  |  Effective N = 1/HHI",
        "good": "HHI < 0.20 (Effective N > 5) is considered diversified.",
        "benchmark": "Equal-weight 10-stock portfolio: HHI = 0.10, Effective N = 10.",
        "tag": "risk",
    },
    "Tracking Error": {
        "short": "How much the portfolio's returns deviate from the benchmark.",
        "long": (
            "Tracking Error is the standard deviation of the portfolio's excess returns "
            "over NIFTY 50. A TE of 5% means the portfolio's outperformance or "
            "underperformance vs NIFTY 50 fluctuates with a standard deviation of 5% per year. "
            "Index funds have TE near 0%. Active funds typically have TE of 3–8%."
        ),
        "formula": "Std(Portfolio Daily Return − Benchmark Daily Return) × √252",
        "good": "Higher TE = more active. Not inherently good or bad — depends on IR.",
        "benchmark": "Passive index funds: <0.5%. Active equity funds: 3–8%.",
        "tag": "risk",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# TOOLTIP HELPER SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def get_help_text(metric_name: str) -> str:
    """
    Return the full tooltip string for a given metric.
    Falls back to a graceful message if the metric isn't in the glossary.
    """
    g = GLOSSARY.get(metric_name)
    if not g:
        return metric_name
    lines = [g["short"], ""]
    lines.append(g["long"])
    if g.get("formula"):
        lines.append(f"\nFormula: {g['formula']}")
    if g.get("good"):
        lines.append(f"\n✅ {g['good']}")
    if g.get("benchmark"):
        lines.append(f"📊 Benchmark: {g['benchmark']}")
    return "\n".join(lines)


def metric_with_help(
    col,
    label: str,
    value: str,
    delta: Optional[str] = None,
    glossary_key: Optional[str] = None,
):
    """
    Drop-in replacement for st.metric() that adds a help tooltip.

    Usage:
        metric_with_help(c1, "Sharpe Ratio", "1.24", glossary_key="Sharpe Ratio")
    """
    key = glossary_key or label
    help_txt = get_help_text(key)
    with col:
        st.metric(label=label, value=value, delta=delta, help=help_txt)


def section_intro(title: str, summary: str, detail: str):
    """
    Renders a collapsible 'What does this section show?' expander
    at the top of each dashboard tab.
    """
    with st.expander(f"ℹ️ What does this section show?", expanded=False):
        st.markdown(f"**{title}**")
        st.markdown(summary)
        st.markdown(detail)


def glossary_card(term: str, entry: Dict):
    """Render a single glossary card in the Glossary tab."""
    tag_class = f"g-tag-{entry.get('tag', 'perf')}"
    tag_label = {"perf": "Performance", "risk": "Risk", "factor": "Factor", "ml": "ML/AI"}.get(
        entry.get("tag", "perf"), "Other"
    )
    st.markdown(f"""
    <div class="g-card">
        <div class="g-term">{term}
            <span class="g-tag {tag_class}">{tag_label}</span>
        </div>
        <div style="font-size:14px;color:#90A4AE;margin-bottom:8px;">{entry['short']}</div>
        <div style="font-size:13px;color:#607080;line-height:1.7;margin-bottom:8px;">{entry['long']}</div>
        {"<div class='g-formula'>"+entry['formula']+"</div>" if entry.get('formula') else ""}
        {"<div class='g-good' style='margin-top:6px'>✅ "+entry['good']+"</div>" if entry.get('good') else ""}
        {"<div style='font-size:12px;color:#607080;margin-top:3px'>📊 Benchmark: "+entry['benchmark']+"</div>" if entry.get('benchmark') else ""}
    </div>
    """, unsafe_allow_html=True)


# ─── Simple format helpers ────────────────────────────────────────────────────

def fmt_pct(v, digits=2):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    return f"{v*100:.{digits}f}%"

def fmt_val(v, digits=2):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    return f"{v:.{digits}f}"

def fmt_x(v, digits=2):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    return f"{v:.{digits}f}x"

def signal_badge(signal: str) -> str:
    return f'<span class="signal-{signal.lower()}">{signal}</span>'


# ─── Data Loading (cached — NO st.* calls inside) ─────────────────────────────

@st.cache_data(ttl=3600)
def load_market_data(tickers, years=3):
    from data_engine.market_data import (
        run_market_data_pipeline, load_close_matrix,
        load_returns_matrix, fetch_benchmark, compute_returns,
    )
    tickers = list(tickers)
    try:
        close_m   = load_close_matrix(tickers)
        returns_m = load_returns_matrix(tickers)
        bench_raw = fetch_benchmark()
        bench_r   = compute_returns(bench_raw)["daily_return"] if not bench_raw.empty else pd.Series(dtype=float)
        missing   = [t for t in tickers if t not in close_m.columns]
        if not close_m.empty and not missing:
            return close_m, returns_m, bench_r
        tickers_to_fetch = missing if missing else tickers
    except Exception:
        tickers_to_fetch = tickers

    enriched  = run_market_data_pipeline(tickers_to_fetch, years=years)
    try:
        close_m   = load_close_matrix(tickers)
        returns_m = load_returns_matrix(tickers)
    except Exception:
        close_m   = pd.DataFrame({t: df["close"]        for t, df in enriched.items()})
        returns_m = pd.DataFrame({t: df["daily_return"]  for t, df in enriched.items()})
    bench_raw = fetch_benchmark()
    bench_r   = compute_returns(bench_raw)["daily_return"] if not bench_raw.empty else pd.Series(dtype=float)
    return close_m, returns_m, bench_r


@st.cache_data(ttl=86400)
def load_fundamentals_data(tickers):
    from data_engine.fundamentals import run_fundamentals_pipeline
    return run_fundamentals_pipeline(tickers)


@st.cache_data(ttl=1800)
def run_full_analysis(tickers, weighting_method, custom_weights_json=None):
    from analytics.portfolio import compute_weights, portfolio_summary
    from factors.composite   import run_factor_pipeline
    from ml_engine.models    import run_ml_pipeline
    from data_engine.events  import run_events_pipeline

    close_m, returns_m, bench_r = load_market_data(tickers)
    fundamentals_df = load_fundamentals_data(tickers)

    market_caps = {}
    if not fundamentals_df.empty and "market_cap" in fundamentals_df.columns:
        market_caps = dict(zip(
            fundamentals_df["ticker"],
            fundamentals_df["market_cap"].fillna(1e10),
        ))

    custom_w = custom_weights_json or {}
    weights  = compute_weights(tickers, weighting_method, market_caps, custom_w)
    summary  = portfolio_summary(tickers, weights, returns_m, bench_r, weighting_method)
    factor_df = run_factor_pipeline(tickers, close_m, returns_m, fundamentals_df, bench_r)

    sentiment_df = pd.DataFrame()
    no_news_key  = False
    try:
        news_key = os.getenv("NEWS_API_KEY", "") if (os := __import__("os")) else ""
        if news_key and news_key != "your_newsapi_key_here":
            sentiment_df = run_events_pipeline(tickers, days_back=7)
        else:
            no_news_key = True
    except Exception:
        pass

    signals_df, shap_df = run_ml_pipeline(
        tickers, close_m, returns_m, factor_df, fundamentals_df, bench_r, sentiment_df
    )

    return {
        "close_m":      close_m,   "returns_m":  returns_m,
        "bench_r":      bench_r,   "fundamentals": fundamentals_df,
        "weights":      weights,   "summary":    summary,
        "factor_df":    factor_df, "sentiment_df": sentiment_df,
        "no_news_key":  no_news_key,
        "signals_df":   signals_df, "shap_df":   shap_df,
    }


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown("""
    <div style="text-align:center;padding:16px 0 8px 0;">
        <div style="font-size:24px;">📊</div>
        <div style="color:#00D4FF;font-size:14px;font-weight:700;letter-spacing:1px;">QPIP</div>
        <div style="color:#607080;font-size:10px;letter-spacing:2px;">INSTITUTIONAL ANALYTICS</div>
    </div><hr>
    """, unsafe_allow_html=True)

    st.sidebar.header("Portfolio Configuration")
    default_str = ", ".join(DEFAULT_TICKERS[:5])
    tickers_raw = st.sidebar.text_area(
        "NSE Tickers (min 5, comma-separated)", value=default_str, height=100,
        help="Enter NSE ticker symbols without the .NS suffix. E.g: RELIANCE, TCS, HDFCBANK",
    )
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if len(tickers) < 5:
        st.sidebar.error("⚠️ Please enter at least 5 tickers.")

    st.sidebar.divider()
    method = st.sidebar.selectbox(
        "Portfolio Weighting", WEIGHTING_METHODS, index=0,
        help=get_help_text("Equal Weighted"),
    )
    custom_weights = {}
    if method == "Custom Weighted" and tickers:
        st.sidebar.markdown("**Custom Weights (%)**")
        for t in tickers:
            w = st.sidebar.number_input(
                t, min_value=0.0, max_value=100.0,
                value=round(100 / len(tickers), 1), step=0.5, key=f"cw_{t}",
            )
            custom_weights[t] = w / 100.0

    st.sidebar.divider()
    data_years   = st.sidebar.slider("Data Lookback (years)", 1, 5, 3,
                                     help="More years = better ML training but slower first run.")
    run_backtest = st.sidebar.checkbox("Run Backtest", value=True,
                                       help="Walk-forward 12-month backtest with transaction costs.")
    st.sidebar.divider()
    run_btn = st.sidebar.button("🚀 Run Analysis", type="primary", width='stretch')
    return tickers, method, custom_weights, data_years, run_backtest, run_btn


# ─── Tab: Overview ────────────────────────────────────────────────────────────

def render_overview(data: Dict):
    section_intro(
        "Portfolio Overview",
        "Key performance and risk metrics for your portfolio vs NIFTY 50.",
        "The NAV chart shows how ₹100 invested at the start would have grown. "
        "Drawdown shows peak-to-trough losses over time. Hover any metric card for a full explanation.",
    )

    metrics = data["summary"]["metrics"]
    st.subheader("📈 Portfolio Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metric_with_help(c1, "Total Return",  fmt_pct(metrics.get("total_return")),  glossary_key="Total Return")
    metric_with_help(c2, "Ann. Return",   fmt_pct(metrics.get("ann_return")),    glossary_key="Ann. Return")
    metric_with_help(c3, "Sharpe Ratio",  fmt_val(metrics.get("sharpe_ratio")),  glossary_key="Sharpe Ratio")
    metric_with_help(c4, "Max Drawdown",  fmt_pct(metrics.get("max_drawdown")),  glossary_key="Max Drawdown")
    metric_with_help(c5, "Beta",          fmt_val(metrics.get("beta")),           glossary_key="Beta")
    metric_with_help(c6, "Alpha (Ann.)",  fmt_pct(metrics.get("alpha")),          glossary_key="Alpha")

    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        nav       = data["summary"]["nav"]
        bench_nav = data["summary"]["benchmark_nav"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nav.index, y=nav.values, name="Portfolio",
                                  line=dict(color=BRAND_COLOR, width=2.5),
                                  fill="tozeroy", fillcolor="rgba(0,212,255,0.08)"))
        fig.add_trace(go.Scatter(x=bench_nav.index, y=bench_nav.values,
                                  name=BENCHMARK_LABEL,
                                  line=dict(color=NEUTRAL_COLOR, width=1.5, dash="dot")))
        fig.update_layout(**PLOT_LAYOUT, title="Portfolio NAV vs NIFTY 50 (Base=100)", height=320)
        st.plotly_chart(fig, width='stretch')
    with col2:
        dd = data["summary"]["drawdown"]
        fig2 = go.Figure(go.Scatter(x=dd.index, y=dd.values * 100, name="Drawdown",
                                     fill="tozeroy", line=dict(color=DANGER_COLOR, width=1.5),
                                     fillcolor="rgba(255,23,68,0.13)"))
        fig2.update_layout(**PLOT_LAYOUT, title="Drawdown (%)", height=320, yaxis_ticksuffix="%")
        st.plotly_chart(fig2, width='stretch')


# ─── Tab: Allocation ──────────────────────────────────────────────────────────

def render_allocation(data: Dict):
    section_intro(
        "Portfolio Allocation",
        "How your capital is distributed across stocks.",
        "The attribution table shows each stock's contribution to total portfolio returns. "
        "'vs Benchmark' shows how much each position contributed relative to NIFTY 50.",
    )
    weights = data["weights"]
    tickers = list(weights.keys())
    w_vals  = [weights[t] * 100 for t in tickers]

    st.subheader("🍩 Portfolio Allocation")
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Pie(labels=tickers, values=w_vals, hole=0.5,
                                marker=dict(colors=px.colors.qualitative.Bold),
                                textinfo="label+percent",
                                hovertemplate="<b>%{label}</b><br>Weight: %{value:.1f}%<extra></extra>"))
        fig.update_layout(**PLOT_LAYOUT, title="Allocation", height=380, showlegend=False)
        st.plotly_chart(fig, width='stretch')
    with col2:
        fig2 = go.Figure(go.Bar(x=w_vals, y=tickers, orientation="h",
                                 marker_color=BRAND_COLOR,
                                 text=[f"{v:.1f}%" for v in w_vals], textposition="outside"))
        fig2.update_layout(**PLOT_LAYOUT, title="Weights", height=380, xaxis_ticksuffix="%")
        st.plotly_chart(fig2, width='stretch')

    if not data["returns_m"].empty:
        from analytics.portfolio import returns_attribution
        attr_df = returns_attribution(data["returns_m"], weights, data["bench_r"])
        st.dataframe(attr_df, width='stretch')


# ─── Tab: Correlation ─────────────────────────────────────────────────────────

def render_correlation(data: Dict):
    section_intro(
        "Correlation Matrix",
        "How closely each pair of stocks moves together.",
        "Dark red = highly correlated (move together — less diversification). "
        "Dark blue = negatively correlated (move opposite — maximum diversification). "
        "A well-diversified portfolio has most off-diagonal values below 0.5.",
    )
    from analytics.risk_metrics import correlation_matrix
    st.subheader("🔗 Correlation Matrix")
    returns_m = data["returns_m"].dropna()
    corr = correlation_matrix(returns_m)

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu_r", zmid=0, text=corr.round(2).values, texttemplate="%{text}",
        hovertemplate="<b>%{x} / %{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(title=dict(text="ρ", font=dict(color="#90A4AE")), tickfont=dict(color="#90A4AE")),
    ))
    fig.update_layout(**PLOT_LAYOUT, title="Pairwise Return Correlation", height=480)
    st.plotly_chart(fig, width='stretch')

    tickers = returns_m.columns.tolist()
    if len(tickers) >= 2:
        t1, t2 = tickers[0], tickers[1]
        roll_corr = returns_m[t1].rolling(63).corr(returns_m[t2])
        fig2 = go.Figure(go.Scatter(x=roll_corr.index, y=roll_corr.values,
                                     line=dict(color=ACCENT_COLOR)))
        fig2.add_hline(y=0, line_dash="dash", line_color=NEUTRAL_COLOR)
        fig2.update_layout(**PLOT_LAYOUT,
                           title=f"Rolling 63-Day Correlation: {t1} vs {t2}", height=260)
        st.plotly_chart(fig2, width='stretch')


# ─── Tab: Factors ─────────────────────────────────────────────────────────────

def render_factors(data: Dict):
    section_intro(
        "Factor Exposures",
        "Each stock's score on four systematic investment factors.",
        "Scores are cross-sectional z-scores: 0.0 = average stock, +1.0 = one std dev above average. "
        "Green = high score (good), Red = low score. Hover cells for ticker/factor details. "
        "The composite rank (1 = best) combines all four factors.",
    )
    factor_df = data["factor_df"]
    if factor_df.empty:
        st.info("Factor data not available.")
        return

    st.subheader("⚡ Factor Exposures")

    # Factor explanations inline
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Momentum", "📈", help=get_help_text("Momentum Score"))
        st.caption("Past price outperformance")
    with col2:
        st.metric("Quality", "🏛", help=get_help_text("Quality Score"))
        st.caption("ROE, ROCE, low debt")
    with col3:
        st.metric("Value", "💰", help=get_help_text("Value Score"))
        st.caption("Low PE, PB ratios")
    with col4:
        st.metric("Low Vol", "🛡", help=get_help_text("Low Vol Score"))
        st.caption("Lower price volatility")

    st.divider()

    score_cols = ["momentum_score", "quality_score", "value_score",
                  "volatility_score", "composite_score"]
    available  = [c for c in score_cols if c in factor_df.columns]
    if not available:
        st.info("Factor scores not yet computed.")
        return

    disp_df = factor_df[["ticker"] + available].set_index("ticker")
    score_label_map = {
        "momentum_score":   "Momentum",
        "quality_score":    "Quality",
        "value_score":      "Value",
        "volatility_score": "Low Vol",
        "composite_score":  "Composite",
    }
    disp_df.columns = [
        score_label_map.get(col, col.replace("_score","").replace("_"," ").title())
        for col in disp_df.columns
    ]

    st.markdown(
        "Scores are **cross-sectional z-scores** — each stock is measured relative to the full universe. "
        "**+1.0** = one standard deviation above average. **−1.0** = one below average. "
        "Green = strong. Red = weak. Hover the **?** beside any metric card for the full formula.",
        help=get_help_text("Composite Score"),
    )

    fig = go.Figure(go.Heatmap(
        z=disp_df.values, x=disp_df.columns.tolist(), y=disp_df.index.tolist(),
        colorscale="RdYlGn", zmid=0, text=disp_df.round(2).values, texttemplate="%{text}",
        hovertemplate="<b>%{y} | %{x}</b><br>Score: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Z-Score"),
    ))
    fig.update_layout(**PLOT_LAYOUT, title="Factor Score Heatmap (Z-Score)", height=max(300, len(disp_df) * 28))
    st.plotly_chart(fig, width='stretch')

    # Composite score bar chart — easier to read than radar for many tickers
    if "composite_score" in factor_df.columns:
        comp_sorted = factor_df[["ticker", "composite_score"]].sort_values(
            "composite_score", ascending=True
        ).tail(20)
        colors = [SUCCESS_COLOR if v >= 0 else DANGER_COLOR for v in comp_sorted["composite_score"]]
        fig_bar = go.Figure(go.Bar(
            x=comp_sorted["composite_score"], y=comp_sorted["ticker"],
            orientation="h", marker_color=colors,
            text=comp_sorted["composite_score"].round(2), textposition="outside",
        ))
        fig_bar.update_layout(**PLOT_LAYOUT,
                              title="Composite Factor Score (top 20, higher = better)",
                              height=max(300, len(comp_sorted) * 26), xaxis_zeroline=True)
        st.plotly_chart(fig_bar, width='stretch')

    # Radar charts for individual tickers (first 4)
    if len(factor_df) > 0:
        st.markdown("#### Per-Ticker Factor Radar")
        st.caption("Each axis is a factor score (z-score). Larger area = stronger multi-factor stock.")
        cols = st.columns(min(len(factor_df), 4))
        for i, (_, row) in enumerate(factor_df.head(4).iterrows()):
            with cols[i]:
                vals = [float(row.get("momentum_score", 0) or 0),
                        float(row.get("quality_score",  0) or 0),
                        float(row.get("value_score",    0) or 0),
                        float(row.get("volatility_score",0)or 0)]
                cats = ["Momentum", "Quality", "Value", "Low Vol"]
                fig_r = go.Figure(go.Scatterpolar(
                    r=vals + [vals[0]], theta=cats + [cats[0]],
                    fill="toself", fillcolor="rgba(0,212,255,0.19)",
                    line=dict(color=BRAND_COLOR), name=row.get("ticker",""),
                ))
                fig_r.update_layout(
                    polar=dict(bgcolor="#0A1628",
                               radialaxis=dict(gridcolor="#1E2A3A", range=[-3, 3]),
                               angularaxis=dict(gridcolor="#1E2A3A")),
                    paper_bgcolor="#060B14", font=dict(color="#90A4AE"),
                    title=dict(text=row.get("ticker",""), x=0.5, font=dict(color=BRAND_COLOR)),
                    showlegend=False, height=240, margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_r, width='stretch')


# ─── Tab: Signals ─────────────────────────────────────────────────────────────

def render_signals(data: Dict):
    section_intro(
        "ML Buy / Sell / Hold Signals",
        "AI-generated signals from an ensemble of XGBoost, LightGBM, and Random Forest models.",
        "Signals are relative to NIFTY 50 — a SELL signal means the stock is expected to "
        "underperform the index, not necessarily fall in absolute terms. "
        "Confidence above 65% makes a signal more actionable. "
        "Hover the '?' next to each signal type for a full explanation.",
    )
    signals_df = data.get("signals_df")
    st.subheader("🎯 ML Buy / Sell / Hold Signals")

    # Legend with help text
    leg1, leg2, leg3, leg4 = st.columns(4)
    leg1.metric("BUY",  "Outperform", help=get_help_text("BUY Signal"))
    leg2.metric("SELL", "Underperform", help=get_help_text("SELL Signal"))
    leg3.metric("HOLD", "In-line",    help=get_help_text("HOLD Signal"))
    leg4.metric("Confidence", "Model certainty", help=get_help_text("Model Confidence"))
    st.divider()

    if signals_df is None or signals_df.empty:
        st.info("Signals not yet generated.")
        return

    for _, row in signals_df.iterrows():
        sig   = row.get("signal", "HOLD")
        color = SUCCESS_COLOR if sig == "BUY" else (DANGER_COLOR if sig == "SELL" else ACCENT_COLOR)
        c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1, 1])
        c1.markdown(f"**{row['ticker']}**")
        c2.markdown(signal_badge(sig), unsafe_allow_html=True)
        c3.markdown(f"<span style='color:{SUCCESS_COLOR}'>▲ {row.get('buy_probability',0):.0%}</span>",
                    unsafe_allow_html=True)
        c4.markdown(f"<span style='color:{NEUTRAL_COLOR}'>◆ {row.get('hold_probability',0):.0%}</span>",
                    unsafe_allow_html=True)
        c5.markdown(f"<span style='color:{DANGER_COLOR}'>▼ {row.get('sell_probability',0):.0%}</span>",
                    unsafe_allow_html=True)
        st.divider()


# ─── Tab: SHAP ────────────────────────────────────────────────────────────────

def render_shap(data: Dict):
    section_intro(
        "SHAP Explainability",
        "Why did the model give this signal? SHAP values show the contribution of each feature.",
        "Green bars pushed the model toward BUY. Red bars pushed toward SELL. "
        "The length of each bar shows how much that feature influenced this specific prediction. "
        "This makes the AI's reasoning fully transparent — no black box.",
    )
    signals_df = data.get("signals_df")
    shap_df    = data.get("shap_df")
    st.subheader("🧠 SHAP Explainability")

    st.info(
        "**What is SHAP?** " + GLOSSARY["SHAP Value"]["long"],
        icon="ℹ️",
    )

    if signals_df is None or signals_df.empty:
        st.info("Run analysis to generate SHAP explanations.")
        return

    selected = st.selectbox("Select Ticker", signals_df["ticker"].tolist())
    if not selected:
        return

    row       = signals_df[signals_df["ticker"] == selected].iloc[0]
    sig       = row.get("signal", "HOLD")
    sig_color = SUCCESS_COLOR if sig == "BUY" else (DANGER_COLOR if sig == "SELL" else ACCENT_COLOR)
    conf      = row.get("confidence", 0)
    buy_p     = row.get("buy_probability",  0)
    hold_p    = row.get("hold_probability", 0)
    sell_p    = row.get("sell_probability", 0)

    # Signal header card
    st.markdown(f"""
    <div style="background:#0E1A2E;border:1px solid #1E2A3A;border-radius:8px;padding:16px;margin-bottom:16px;">
        <div style="font-size:20px;font-weight:700;color:{sig_color};">{selected} — {sig}</div>
        <div style="color:#90A4AE;font-size:13px;margin-top:4px;">
            Model Confidence: {conf:.1%} &nbsp;|&nbsp;
            ▲ BUY: {buy_p:.1%} &nbsp;|&nbsp;
            ◆ HOLD: {hold_p:.1%} &nbsp;|&nbsp;
            ▼ SELL: {sell_p:.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Plain-English narrative
    top_pos = row.get("top_positive_drivers") or {}
    top_neg = row.get("top_negative_drivers") or {}
    if isinstance(top_pos, str):
        try: top_pos = json.loads(top_pos)
        except: top_pos = {}
    if isinstance(top_neg, str):
        try: top_neg = json.loads(top_neg)
        except: top_neg = {}

    if top_pos or top_neg:
        with st.expander("📖 Plain-English Explanation", expanded=True):
            if top_pos:
                pos_str = ", ".join(
                    f"**{k.replace('_', ' ').title()}** (+{v:.3f})"
                    for k, v in list(top_pos.items())[:3]
                )
                st.markdown(f"✅ **Why {sig}:** The strongest signals pushing toward this call were {pos_str}.")
            if top_neg:
                neg_str = ", ".join(
                    f"**{k.replace('_', ' ').title()}** ({v:.3f})"
                    for k, v in list(top_neg.items())[:3]
                )
                st.markdown(f"⚠️ **Risks to watch:** Factors working against this signal: {neg_str}.")
            st.caption(
                "SHAP values are shown for the BUY class. Positive = pushes toward BUY, "
                "Negative = pushes toward SELL. Values are in log-odds units."
            )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ✅ Positive Drivers (push toward BUY)")
        if top_pos:
            fig = go.Figure(go.Bar(
                x=list(top_pos.values()),
                y=[k.replace("_", " ").title() for k in top_pos.keys()],
                orientation="h", marker_color=SUCCESS_COLOR,
                text=[f"{v:+.3f}" for v in top_pos.values()], textposition="outside",
            ))
            fig.update_layout(**PLOT_LAYOUT, height=300, title="SHAP: Positive Contributions")
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No positive drivers available.")

    with col2:
        st.markdown("#### ⚠️ Risk Factors (push toward SELL)")
        if top_neg:
            fig2 = go.Figure(go.Bar(
                x=list(top_neg.values()),
                y=[k.replace("_", " ").title() for k in top_neg.keys()],
                orientation="h", marker_color=DANGER_COLOR,
                text=[f"{v:+.3f}" for v in top_neg.values()], textposition="outside",
            ))
            fig2.update_layout(**PLOT_LAYOUT, height=300, title="SHAP: Negative Contributions")
            st.plotly_chart(fig2, width='stretch')
        else:
            st.info("No risk factors available.")

    if shap_df is not None and not shap_df.empty:
        st.markdown("#### 📊 Global Feature Importance (Mean |SHAP| across all tickers)")
        st.caption("Larger bar = this feature drives more predictions overall. "
                   "This shows what the model has learnt to rely on most.")
        mean_abs = shap_df.abs().mean().sort_values(ascending=True).tail(15)
        fig3 = go.Figure(go.Bar(
            x=mean_abs.values,
            y=[c.replace("_", " ").title() for c in mean_abs.index],
            orientation="h",
            marker=dict(color=mean_abs.values, colorscale="Viridis", showscale=True),
        ))
        fig3.update_layout(**PLOT_LAYOUT, title="Mean |SHAP| Feature Importance", height=420)
        st.plotly_chart(fig3, width='stretch')


# ─── Tab: Events ──────────────────────────────────────────────────────────────

def render_events(data: Dict):
    section_intro(
        "Event Intelligence",
        "News sentiment analysis for each ticker over the last 7 days.",
        "Sentiment is computed using VADER + TextBlob NLP models on news headlines. "
        "Score ranges from −1.0 (very negative) to +1.0 (very positive). "
        "Enable a NewsAPI key in your .env file to activate this tab.",
    )
    sentiment_df = data.get("sentiment_df")
    st.subheader("📰 Event Intelligence")

    if sentiment_df is None or sentiment_df.empty:
        st.info("No recent events loaded. Add a free NewsAPI key to your .env file to enable this tab.")
        return

    for _, row in sentiment_df.iterrows():
        score = row.get("avg_sentiment", 0)
        label = row.get("dominant_sentiment", "Neutral")
        color = SUCCESS_COLOR if score > 0.1 else (DANGER_COLOR if score < -0.1 else NEUTRAL_COLOR)
        st.markdown(f"""
        <div style="background:#0E1A2E;border-left:3px solid {color};
                    border-radius:4px;padding:10px 14px;margin:6px 0;">
            <span style="font-weight:700;color:#E0E8F0;">{row['ticker']}</span> &nbsp;|&nbsp;
            <span style="color:{color};">{label}</span> &nbsp;|&nbsp;
            <span style="color:#607080;">Score: {score:.3f}</span> &nbsp;|&nbsp;
            <span style="color:#607080;">{int(row.get('event_count',0))} events</span>
        </div>
        """, unsafe_allow_html=True)


# ─── Tab: Backtest ────────────────────────────────────────────────────────────

def render_backtest(data: Dict, tickers: List, method: str):
    section_intro(
        "Walk-Forward Backtest",
        "Simulates running this portfolio strategy over the past 12 months with monthly rebalancing.",
        "Walk-forward testing rebalances at each month-end using only information available at that time — "
        "no look-ahead bias. Transaction costs of 20bps (0.20%) one-way are deducted at each rebalance. "
        "Compare the NAV curve to NIFTY 50 to see when the strategy added or lost value.",
    )
    from backtesting.backtest import run_backtest, save_backtest_result

    st.subheader("📉 Backtesting Engine (12-Month Walk-Forward)")
    with st.spinner("Running backtest…"):
        result = run_backtest(
            tickers, data["returns_m"], data["bench_r"],
            weighting_method=method, strategy_name=f"{method} Portfolio",
        )
    if not result:
        st.error("Backtest failed — insufficient data.")
        return

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metric_with_help(c1, "CAGR",        fmt_pct(result["cagr"]),          glossary_key="CAGR")
    metric_with_help(c2, "Sharpe",      fmt_val(result["sharpe_ratio"]),  glossary_key="Sharpe Ratio")
    metric_with_help(c3, "Sortino",     fmt_val(result["sortino_ratio"]), glossary_key="Sortino Ratio")
    metric_with_help(c4, "Max Drawdown",fmt_pct(result["max_drawdown"]),  glossary_key="Max Drawdown")
    metric_with_help(c5, "Alpha",       fmt_pct(result["alpha"]),          glossary_key="Alpha")
    metric_with_help(c6, "Hit Ratio",   fmt_pct(result["hit_ratio"]),     glossary_key="Hit Ratio")

    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        nav = result["portfolio_nav"]; b_nav = result["benchmark_nav"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nav.index, y=nav.values, name="Portfolio",
                                  line=dict(color=BRAND_COLOR, width=2.5)))
        fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav.values, name=BENCHMARK_LABEL,
                                  line=dict(color=NEUTRAL_COLOR, width=1.5, dash="dot")))
        fig.update_layout(**PLOT_LAYOUT, title="Backtest NAV (Base=100)", height=320)
        st.plotly_chart(fig, width='stretch')
    with col2:
        monthly = result.get("monthly_returns", {})
        if monthly:
            m_df = pd.Series(monthly).reset_index()
            m_df.columns = ["date", "return"]
            m_df["date"] = pd.to_datetime(m_df["date"])
            fig2 = go.Figure(go.Bar(
                x=m_df["date"], y=m_df["return"] * 100,
                marker_color=[SUCCESS_COLOR if v >= 0 else DANGER_COLOR for v in m_df["return"]],
                text=[f"{v*100:+.1f}%" for v in m_df["return"]], textposition="outside",
            ))
            fig2.update_layout(**PLOT_LAYOUT, title="Monthly Returns (%)", height=320, yaxis_ticksuffix="%")
            st.plotly_chart(fig2, width='stretch')

    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Total Return | {fmt_pct(result['total_return'])} |
    | Benchmark Return | {fmt_pct(result['benchmark_return'])} |
    | Excess Return | {fmt_pct(result['excess_return'])} |
    | Information Ratio | {fmt_val(result['information_ratio'])} |
    | Beta | {fmt_val(result['beta'])} |
    | Rebalances | {result['num_rebalances']} |
    | Transaction Costs | {result['transaction_costs']:.4f} |
    """)

    if st.button("💾 Save Backtest Result"):
        save_backtest_result(result)
        st.success("Backtest result saved to database.")


# ─── Tab: Risk Metrics ────────────────────────────────────────────────────────

def render_risk_metrics(data: Dict):
    section_intro(
        "Risk Metrics",
        "Comprehensive risk and performance attribution vs NIFTY 50.",
        "Rolling Sharpe shows whether risk-adjusted performance is improving or degrading over time. "
        "The return distribution shows how fat-tailed your portfolio is vs the benchmark — "
        "fatter left tails = higher tail risk. Hover any metric card for a full plain-English explanation.",
    )
    from analytics.risk_metrics import benchmark_comparison, rolling_sharpe, rolling_volatility

    st.subheader("⚠️ Risk Metrics")
    port_r  = data["summary"]["portfolio_returns"]
    bench_r = data["bench_r"].reindex(port_r.index)
    metrics = data["summary"]["metrics"]

    # ── Metric cards with tooltips ─────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    metric_with_help(c1, "Sharpe Ratio",   fmt_val(metrics.get("sharpe_ratio")),  glossary_key="Sharpe Ratio")
    metric_with_help(c2, "Sortino Ratio",  fmt_val(metrics.get("sortino_ratio")), glossary_key="Sortino Ratio")
    metric_with_help(c3, "Calmar Ratio",   fmt_val(metrics.get("calmar_ratio")),  glossary_key="Calmar Ratio")
    metric_with_help(c4, "Info. Ratio",    fmt_val(metrics.get("information_ratio")), glossary_key="Information Ratio")
    c5, c6, c7, c8 = st.columns(4)
    metric_with_help(c5, "VaR (95%)",      fmt_pct(metrics.get("var_95")),         glossary_key="VaR (95%)")
    metric_with_help(c6, "CVaR (95%)",     fmt_pct(metrics.get("cvar_95")),        glossary_key="CVaR (95%)")
    metric_with_help(c7, "Ann. Volatility",fmt_pct(metrics.get("ann_volatility")), glossary_key="Ann. Volatility")
    metric_with_help(c8, "Hit Ratio",      fmt_pct(metrics.get("hit_ratio")),      glossary_key="Hit Ratio")

    # ── Comparison table with per-column help ─────────────────────────────────
    st.divider()
    comp_df = benchmark_comparison(port_r, bench_r)
    st.markdown("**Portfolio vs NIFTY 50 — Full Comparison**")

    # Map each row label to a glossary key for the column config help text
    col_help = {
        "Portfolio": "Your portfolio's value for this metric.",
        "NIFTY 50":  "NIFTY 50 benchmark value for the same metric and period.",
    }
    metric_help_map = {
        "Total Return":       get_help_text("Total Return"),
        "Ann Return":         get_help_text("Ann. Return"),
        "Ann Volatility":     get_help_text("Ann. Volatility"),
        "Sharpe Ratio":       get_help_text("Sharpe Ratio"),
        "Sortino Ratio":      get_help_text("Sortino Ratio"),
        "Max Drawdown":       get_help_text("Max Drawdown"),
        "Beta":               get_help_text("Beta"),
        "Alpha":              get_help_text("Alpha"),
        "Information Ratio":  get_help_text("Information Ratio"),
        "Var 95":             get_help_text("VaR (95%)"),
        "Hit Ratio":          get_help_text("Hit Ratio"),
    }
    st.dataframe(
        comp_df,
        width='stretch',
        hide_index=True,
        column_config={
            "Metric":    st.column_config.TextColumn("Metric",    width="medium"),
            "Portfolio": st.column_config.NumberColumn("Portfolio", format="%.4f",
                         help="Your portfolio's value. Hover the Metric name for explanation."),
            "NIFTY 50":  st.column_config.NumberColumn("NIFTY 50",  format="%.4f",
                         help="NIFTY 50 benchmark value for the same period."),
        }
    )
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        rs = rolling_sharpe(port_r, window=63)
        fig = go.Figure(go.Scatter(x=rs.index, y=rs.values, line=dict(color=BRAND_COLOR),
                                    fill="tozeroy", fillcolor="rgba(0,212,255,0.08)"))
        fig.add_hline(y=0, line_dash="dash", line_color=NEUTRAL_COLOR)
        fig.update_layout(**PLOT_LAYOUT, title="Rolling 63-Day Sharpe Ratio", height=300)
        st.plotly_chart(fig, width='stretch')
    with col2:
        rv = rolling_volatility(port_r, window=21) * 100
        fig2 = go.Figure(go.Scatter(x=rv.index, y=rv.values, line=dict(color=ACCENT_COLOR)))
        fig2.update_layout(**PLOT_LAYOUT,
                           title="Rolling 21-Day Annualised Volatility (%)",
                           yaxis_ticksuffix="%", height=300)
        st.plotly_chart(fig2, width='stretch')

    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=port_r.dropna() * 100, name="Portfolio",
                                 nbinsx=80, marker_color=BRAND_COLOR, opacity=0.7))
    fig3.add_trace(go.Histogram(x=bench_r.dropna() * 100, name=BENCHMARK_LABEL,
                                 nbinsx=80, marker_color=NEUTRAL_COLOR, opacity=0.5))
    fig3.update_layout(**PLOT_LAYOUT, title="Return Distribution",
                       barmode="overlay", height=300, xaxis_ticksuffix="%")
    st.plotly_chart(fig3, width='stretch')



# ─── Tab: Monte Carlo VaR & Stress Testing ───────────────────────────────────

def render_stress_testing(data: Dict):
    section_intro(
        "Monte Carlo VaR & Stress Testing",
        "10,000 simulations driven by your portfolio's historical covariance matrix.",
        "VaR (Value at Risk) answers: 'How much could I lose on a bad day?' "
        "CVaR (Expected Shortfall) answers: 'On those bad days, how bad does it get on average?' "
        "Stress tests replay historical crises through your current portfolio weights.",
    )

    from backtesting.stress_testing import (
        run_monte_carlo, run_multi_horizon_var, run_all_stress_scenarios,
        run_regime_var, component_var, save_var_result,
        STRESS_SCENARIOS, CONFIDENCE_LEVELS,
    )

    returns_m = data["returns_m"]
    weights   = data["weights"]
    tickers   = list(weights.keys())

    # ── Controls ──────────────────────────────────────────────────────────────
    st.subheader("🎲 Monte Carlo Value at Risk")
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        n_sims = st.select_slider(
            "Simulations",
            options=[1_000, 5_000, 10_000, 25_000, 50_000],
            value=10_000,
            help="More simulations = more accurate but slower. 10,000 is the standard.",
        )
    with ctrl2:
        holding_period = st.selectbox(
            "Holding Period",
            [1, 5, 10, 21],
            format_func=lambda x: f"{x} day{'s' if x > 1 else ''}",
            help="How many trading days ahead to project the loss.",
        )
    with ctrl3:
        confidence_display = st.multiselect(
            "Confidence Levels",
            ["90%", "95%", "97.5%", "99%"],
            default=["95%", "97.5%", "99%"],
            help="95% means: on 95% of simulated days you lose less than VaR.",
        )

    run_mc = st.button("▶  Run Monte Carlo", type="primary",
                       help="Runs 10,000 portfolio simulations using your historical covariance matrix.")

    if not run_mc and "mc_result" not in st.session_state:
        st.info("Configure options above and click **Run Monte Carlo** to generate VaR estimates.")
        # Still show stress tests below
    else:
        if run_mc:
            with st.spinner(f"Running {n_sims:,} Monte Carlo simulations…"):
                try:
                    mc = run_monte_carlo(
                        weights, returns_m,
                        n_simulations=n_sims,
                        holding_period=holding_period,
                        seed=42,
                    )
                    st.session_state["mc_result"] = mc
                except Exception as e:
                    st.error(f"Monte Carlo failed: {e}")
                    import traceback; st.code(traceback.format_exc())

        if "mc_result" in st.session_state:
            mc = st.session_state["mc_result"]
            vt = mc["var_table"]

            # ── VaR Summary cards ──────────────────────────────────────────
            cl_map = {"90%": 0.90, "95%": 0.95, "97.5%": 0.975, "99%": 0.99}
            selected_cls = [cl_map[c] for c in confidence_display if c in cl_map]
            if not selected_cls:
                selected_cls = [0.95, 0.975, 0.99]

            col_cards = st.columns(len(selected_cls) * 2)
            for i, cl in enumerate(selected_cls):
                pct_label = f"{int(cl*100)}%" if cl != 0.975 else "97.5%"
                var_val   = vt[cl]["var"]
                cvar_val  = vt[cl]["cvar"]
                col_cards[i*2].metric(
                    f"VaR {pct_label}",
                    fmt_pct(abs(var_val)),
                    delta=f"{pct_label} of simulations beat this",
                    help=f"On {int(cl*100)}% of days you lose less than {fmt_pct(abs(var_val))}. "
                         f"This is your {pct_label} Value at Risk over {holding_period} trading day(s).",
                )
                col_cards[i*2+1].metric(
                    f"CVaR {pct_label}",
                    fmt_pct(abs(cvar_val)),
                    delta="Expected loss on bad days",
                    help=f"On the worst {100-int(cl*100)}% of days, you lose {fmt_pct(abs(cvar_val))} on average. "
                         f"CVaR (Expected Shortfall) is more conservative than VaR.",
                )

            st.divider()

            # ── P&L Distribution chart ─────────────────────────────────────
            col1, col2 = st.columns([2, 1])
            with col1:
                pnl = mc["portfolio_pnl"] * 100   # convert to %
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=pnl, nbinsx=100, name="Simulated P&L",
                    marker_color=BRAND_COLOR, opacity=0.75,
                ))
                # VaR lines
                colors_for_cl = {0.90: "#FF6B35", 0.95: "#FF6B35",
                                  0.975: "#FF1744", 0.99: "#B71C1C"}
                for cl in selected_cls:
                    var_pct = vt[cl]["var"] * 100
                    lbl = f"{int(cl*100)}%" if cl != 0.975 else "97.5%"
                    fig.add_vline(
                        x=var_pct,
                        line_color=colors_for_cl.get(cl, DANGER_COLOR),
                        line_dash="dash", line_width=2,
                        annotation_text=f"VaR {lbl}",
                        annotation_position="top right",
                        annotation_font_color=colors_for_cl.get(cl, DANGER_COLOR),
                    )
                fig.update_layout(
                    **PLOT_LAYOUT,
                    title=f"Monte Carlo P&L Distribution ({n_sims:,} simulations, {holding_period}d holding)",
                    xaxis_title="Portfolio Return (%)",
                    yaxis_title="Frequency",
                    height=360,
                    showlegend=False,
                )
                st.plotly_chart(fig, width="stretch")

            with col2:
                st.markdown("**Distribution Statistics**")
                stats = {
                    "Simulations":    f"{mc['n_simulations']:,}",
                    "Mean P&L":       fmt_pct(mc["pnl_mean"]),
                    "Std Deviation":  fmt_pct(mc["pnl_std"]),
                    "Skewness":       fmt_val(mc["pnl_skew"]),
                    "Kurtosis":       fmt_val(mc["pnl_kurtosis"]),
                    "P(Loss > 0)":    fmt_pct(mc["prob_loss"]),
                    "Avg Loss|Loss":  fmt_pct(abs(mc["expected_loss"])),
                }
                for k, v in stats.items():
                    sc1, sc2 = st.columns([1.4, 1])
                    sc1.markdown(f"<span style='color:#607080;font-size:12px'>{k}</span>",
                                 unsafe_allow_html=True)
                    sc2.markdown(f"<span style='color:#E0E8F0;font-size:12px;font-weight:500'>{v}</span>",
                                 unsafe_allow_html=True)

                st.markdown("")
                if st.button("💾 Save VaR result", key="save_var"):
                    save_var_result(mc)
                    st.success("Saved to database.")

            # ── Multi-horizon VaR table ────────────────────────────────────
            st.divider()
            st.markdown("#### VaR Across Holding Periods")
            st.caption("Each cell shows the loss threshold at that confidence level for that holding period.")
            with st.spinner("Computing multi-horizon VaR…"):
                try:
                    mh_df = run_multi_horizon_var(weights, returns_m, n_simulations=min(n_sims, 5_000))
                    # Format as percentages
                    fmt_cols = {}
                    for col in mh_df.columns:
                        fmt_cols[col] = st.column_config.NumberColumn(col, format="%.2f%%")
                    display_df = (mh_df * 100).round(3)
                    st.dataframe(display_df, width="stretch",
                                 column_config=fmt_cols)
                except Exception as e:
                    st.warning(f"Multi-horizon VaR failed: {e}")

            # ── Component VaR ──────────────────────────────────────────────
            st.divider()
            st.markdown(
                "#### Component VaR — Which positions contribute most to portfolio risk?",
                help="Component VaR shows each stock's contribution to total portfolio VaR. "
                     "They sum to 100%. A position with high weight AND high volatility "
                     "AND high correlation to other positions will dominate portfolio risk.",
            )
            try:
                comp_df = component_var(weights, returns_m, confidence=0.99, holding_period=1)
                comp_df = comp_df.reset_index()
                comp_df["Weight"] = (comp_df["Weight"] * 100).round(2)
                comp_df["% of Total VaR"] = comp_df["% of Total VaR"].round(2)
                comp_df["Component VaR"] = (comp_df["Component VaR"] * 100).round(4)
                comp_df["Marginal VaR"]  = (comp_df["Marginal VaR"] * 100).round(4)

                # Bar chart
                fig_comp = go.Figure(go.Bar(
                    x=comp_df["Ticker"],
                    y=comp_df["% of Total VaR"],
                    marker_color=[DANGER_COLOR if v > 0 else SUCCESS_COLOR
                                  for v in comp_df["% of Total VaR"]],
                    text=[f"{v:.1f}%" for v in comp_df["% of Total VaR"]],
                    textposition="outside",
                ))
                fig_comp.update_layout(
                    **PLOT_LAYOUT,
                    title="Each Stock's % Contribution to Portfolio VaR (99%, 1-day)",
                    yaxis_title="% of Portfolio VaR",
                    height=300,
                )
                st.plotly_chart(fig_comp, width="stretch")
                st.dataframe(comp_df, width="stretch", hide_index=True)
            except Exception as e:
                st.warning(f"Component VaR failed: {e}")

    # ── Volatility Regime Comparison ──────────────────────────────────────────
    st.divider()
    st.subheader("📊 Volatility Regime Analysis")
    st.caption(
        "How would portfolio VaR change under different volatility environments? "
        "'Crisis' assumes 2.5× historical volatility — comparable to COVID or 2008 conditions."
    )
    with st.spinner("Computing regime VaR…"):
        try:
            reg_df = run_regime_var(weights, returns_m, n_simulations=5_000)
            st.dataframe(reg_df, width="stretch", hide_index=True,
                         column_config={
                             col: st.column_config.NumberColumn(col, format="%.3f%%")
                             for col in reg_df.columns if "VaR" in col or "CVaR" in col
                         })

            # Chart
            fig_reg = go.Figure()
            regime_colors = [SUCCESS_COLOR, ACCENT_COLOR, DANGER_COLOR]
            for i, row in reg_df.iterrows():
                for cl_label, cl_key in [("95%","VaR 95%"), ("99%","VaR 99%")]:
                    pass   # chart data built below

            # Grouped bar chart
            cl_cols = [c for c in reg_df.columns if c.startswith("VaR")]
            for j, col in enumerate(cl_cols):
                fig_reg.add_trace(go.Bar(
                    name=col,
                    x=reg_df["Regime"],
                    y=(reg_df[col].abs() * 100).round(3),
                    marker_color=[regime_colors[i % len(regime_colors)]
                                  for i in range(len(reg_df))],
                    opacity=0.5 + j * 0.15,
                ))
            fig_reg.update_layout(
                **PLOT_LAYOUT,
                title="1-Day VaR by Volatility Regime (%)",
                barmode="group",
                yaxis_title="VaR (%)",
                height=300,
            )
            st.plotly_chart(fig_reg, width="stretch")
        except Exception as e:
            st.warning(f"Regime analysis failed: {e}")

    # ── Stress Scenarios ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("⚡ Historical Stress Scenarios")
    st.markdown(
        "Each scenario applies a systematic market shock (based on the named event) "
        "plus elevated volatility to your current portfolio weights. "
        "Results show how much your portfolio would have lost under each crisis.",
        help="Stress testing is not forecasting — it is asking 'what if history repeats?' "
             "Use these to understand your tail exposure and concentration risk.",
    )

    # Let user pick scenarios
    all_scenario_names = list(STRESS_SCENARIOS.keys())
    selected_scenarios = st.multiselect(
        "Select scenarios to run",
        all_scenario_names,
        default=all_scenario_names[:5],
        help="Select which historical events to stress-test your portfolio against.",
    )

    if st.button("▶  Run Stress Tests", type="secondary"):
        with st.spinner("Running stress scenarios…"):
            try:
                stress_rows = []
                prog = st.progress(0, text="Running scenarios…")
                from backtesting.stress_testing import run_stress_scenario
                for i, name in enumerate(selected_scenarios):
                    if name not in STRESS_SCENARIOS:
                        continue
                    res = run_stress_scenario(name, weights, returns_m,
                                              n_simulations=5_000)
                    stress_rows.append({
                        "Scenario":        res["scenario"],
                        "Period":          res["period"],
                        "Mkt Shock":       fmt_pct(res["market_shock"]),
                        "Portfolio Loss":  fmt_pct(abs(res["systematic_shock"])),
                        "VaR 95%":         fmt_pct(abs(res["var_table"][0.95]["var"])),
                        "CVaR 95%":        fmt_pct(abs(res["var_table"][0.95]["cvar"])),
                        "VaR 99%":         fmt_pct(abs(res["var_table"][0.99]["var"])),
                        "CVaR 99%":        fmt_pct(abs(res["var_table"][0.99]["cvar"])),
                        "Duration":        f"{res['duration_days']}d",
                        "_shock_raw":      res["systematic_shock"],
                    })
                    prog.progress((i + 1) / len(selected_scenarios),
                                  text=f"Completed: {name}")

                st.session_state["stress_results"] = stress_rows
                prog.empty()
            except Exception as e:
                st.error(f"Stress test failed: {e}")
                import traceback; st.code(traceback.format_exc())

    if "stress_results" in st.session_state:
        rows = st.session_state["stress_results"]
        display_cols = [k for k in rows[0].keys() if not k.startswith("_")]
        disp_df = pd.DataFrame(rows)[display_cols]

        # Colour-code by severity
        def _severity_color(row):
            shock = next((r["_shock_raw"] for r in rows if r["Scenario"] == row["Scenario"]), 0)
            if shock < -0.30: return ["background-color:#2D0A0A"] * len(row)
            if shock < -0.15: return ["background-color:#2D1A0A"] * len(row)
            return [""] * len(row)

        st.dataframe(
            disp_df.style.apply(_severity_color, axis=1),
            width="stretch",
            hide_index=True,
        )

        # Tornado chart — portfolio loss by scenario
        sorted_rows = sorted(rows, key=lambda r: r["_shock_raw"])
        names  = [r["Scenario"] for r in sorted_rows]
        losses = [abs(r["_shock_raw"]) * 100 for r in sorted_rows]
        colors = [DANGER_COLOR if l > 20 else ACCENT_COLOR if l > 10 else SUCCESS_COLOR
                  for l in losses]
        fig_torn = go.Figure(go.Bar(
            x=losses, y=names, orientation="h",
            marker_color=colors,
            text=[f"{l:.1f}%" for l in losses],
            textposition="outside",
        ))
        fig_torn.update_layout(
            **PLOT_LAYOUT,
            title="Stress Test — Estimated Portfolio Loss by Scenario",
            xaxis_title="Estimated Portfolio Loss (%)",
            height=max(300, len(names) * 36),
        )
        st.plotly_chart(fig_torn, width="stretch")




# ─── Optimizer helper (compare all methods) ───────────────────────────────────

def _compare_all_methods(returns_m, weights, tickers, window):
    """Run all five optimization methods and return a list of result dicts."""
    from analytics.portfolio_optimizer import (
        optimize_max_sharpe, optimize_min_volatility, optimize_risk_parity,
        optimize_black_litterman, optimize_cvar, build_price_matrix,
    )
    prices = build_price_matrix(tickers, returns_m.tail(window))
    methods = {
        "Max Sharpe":       lambda: optimize_max_sharpe(tickers, prices),
        "Min Variance":     lambda: optimize_min_volatility(tickers, prices),
        "Risk Parity":      lambda: optimize_risk_parity(tickers, prices),
        "Black-Litterman":  lambda: optimize_black_litterman(tickers, prices),
        "CVaR Min":         lambda: optimize_cvar(tickers, returns_m.tail(window)),
    }
    rows = []
    for name, fn in methods.items():
        try:
            r = fn()
            rows.append({
                "Method":          name,
                "Ann. Return":     fmt_pct(r.get("ann_return")),
                "Ann. Volatility": fmt_pct(r.get("ann_vol")),
                "Sharpe Ratio":    fmt_val(r.get("sharpe")),
                "Status":          "✅" if r.get("success") else "⚠️",
            })
        except Exception as e:
            rows.append({"Method":name,"Ann. Return":"—","Ann. Volatility":"—",
                         "Sharpe Ratio":"—","Status":f"❌ {str(e)[:40]}"})
    return rows

# ─── Tab: Portfolio Optimization & Efficient Frontier ────────────────────────

def render_optimization(data: Dict):
    section_intro(
        "Portfolio Optimization",
        "Find the mathematically optimal portfolio weights using five institutional methods.",
        "Mean-Variance maximises Sharpe ratio. Risk Parity equalises each stock's risk "
        "contribution. Black-Litterman blends market equilibrium with your views. "
        "Minimum Variance minimises total volatility. CVaR Optimisation minimises tail loss.",
    )
    from analytics.portfolio_optimizer import (
        optimize_max_sharpe, optimize_min_volatility, optimize_risk_parity,
        optimize_black_litterman, optimize_cvar,
        compare_portfolios, compute_efficient_frontier,
        save_optimization_result, build_price_matrix,
    )

    returns_m = data["returns_m"]
    weights   = data["weights"]
    tickers   = list(weights.keys())

    # ── Controls ──────────────────────────────────────────────────────────────
    st.subheader("⚙️ Optimization Engine")
    col1, col2, col3 = st.columns(3)
    with col1:
        max_wt = st.slider("Max weight per stock (%)", 10, 60, 40, 5,
                            help="Upper bound on any single position. Prevents over-concentration.") / 100
    with col2:
        window = st.selectbox("Estimation window", [63, 126, 252, 504],
                               format_func=lambda x: f"{x} days ({x//21}m)",
                               index=2,
                               help="How much history to use for return/risk estimation.")
    with col3:
        method = st.selectbox("Optimization method",
                               ["Max Sharpe", "Min Variance", "Risk Parity",
                                "Black-Litterman", "CVaR Minimisation", "Compare All"],
                               help="Select which optimization objective to run.")

    run_opt = st.button("⚙️  Optimize Portfolio", type="primary")

    if not run_opt and "opt_result" not in st.session_state:
        st.info("Configure above and click **Optimize Portfolio** to compute optimal weights.")
    else:
        if run_opt:
            with st.spinner("Optimizing portfolio…"):
                try:
                    fn_map = {
                        "Max Sharpe":        lambda: optimize_max_sharpe(tickers, build_price_matrix(tickers, returns_m.tail(window)), max_wt),
                        "Min Variance":      lambda: optimize_min_volatility(tickers, build_price_matrix(tickers, returns_m.tail(window))),
                        "Risk Parity":       lambda: optimize_risk_parity(tickers, build_price_matrix(tickers, returns_m.tail(window))),
                        "Black-Litterman":   lambda: optimize_black_litterman(tickers, build_price_matrix(tickers, returns_m.tail(window))),
                        "CVaR Minimisation": lambda: optimize_cvar(tickers, returns_m.tail(window)),
                        "Compare All":       lambda: None,
                    }
                    if method == "Compare All":
                        opt_result = _compare_all_methods(returns_m, weights, tickers, window)
                        st.session_state["opt_compare"] = opt_result
                        st.session_state["opt_result"]  = None
                    else:
                        result = fn_map[method]()
                        st.session_state["opt_result"]  = result
                        st.session_state["opt_compare"] = None
                        if result.get("success"):
                            save_optimization_result(result, weights)
                except Exception as e:
                    st.error(f"Optimization failed: {e}")
                    import traceback; st.code(traceback.format_exc())

        # ── Show comparison table ────────────────────────────────────────────
        if st.session_state.get("opt_compare") is not None:
            comp_df = st.session_state["opt_compare"]
            st.markdown("#### All Methods Comparison")
            st.dataframe(comp_df.style.highlight_max(
                subset=["Sharpe Ratio"], color="#1D3A1D"
            ).highlight_min(
                subset=["Ann. Volatility"], color="#1D2A3A"
            ), width="stretch", hide_index=True)

        # ── Show single method result ────────────────────────────────────────
        elif st.session_state.get("opt_result") is not None:
            result = st.session_state["opt_result"]
            opt_w  = result.get("weights", {})

            # Metric cards
            m1, m2, m3 = st.columns(3)
            m1.metric("Optimized Return",    fmt_pct(result.get("ann_return")),
                       delta=fmt_pct(result.get("ann_return",0) - data["summary"]["metrics"].get("ann_return",0)),
                       help="Expected annualised return under optimised weights.")
            m2.metric("Optimized Volatility", fmt_pct(result.get("ann_vol")),
                       delta=fmt_pct(result.get("ann_vol",0) - data["summary"]["metrics"].get("ann_volatility",0)),
                       help="Expected annualised volatility under optimised weights.")
            m3.metric("Optimized Sharpe",     fmt_val(result.get("sharpe")),
                       delta=fmt_val(result.get("sharpe",0) - data["summary"]["metrics"].get("sharpe_ratio",0)),
                       help="Sharpe ratio under optimised weights.")

            st.divider()

            # Weight comparison chart
            current_w  = {t: weights.get(t, 0) for t in tickers}
            opt_w_full = {t: opt_w.get(t, 0) for t in tickers}

            fig = go.Figure()
            fig.add_trace(go.Bar(name="Current",   x=tickers,
                                  y=[current_w[t]*100 for t in tickers],
                                  marker_color=NEUTRAL_COLOR, opacity=0.7))
            fig.add_trace(go.Bar(name="Optimized", x=tickers,
                                  y=[opt_w_full.get(t,0)*100 for t in tickers],
                                  marker_color=BRAND_COLOR))
            fig.update_layout(**PLOT_LAYOUT, title="Current vs Optimized Weights (%)",
                              barmode="group", yaxis_ticksuffix="%", height=340)
            st.plotly_chart(fig, width="stretch")

            # Weight table
            wt_df = pd.DataFrame({
                "Ticker":           tickers,
                "Current Weight":   [f"{current_w[t]*100:.1f}%" for t in tickers],
                "Optimized Weight": [f"{opt_w_full.get(t,0)*100:.1f}%" for t in tickers],
                "Change":           [f"{(opt_w_full.get(t,0)-current_w[t])*100:+.1f}%" for t in tickers],
            })
            st.dataframe(wt_df, width="stretch", hide_index=True)

    # ── Efficient Frontier ────────────────────────────────────────────────────
    st.divider()
    st.subheader("📈 Efficient Frontier")
    st.markdown(
        "The efficient frontier shows all portfolios that offer the **maximum return "
        "for a given level of risk**. Any portfolio below the curve is suboptimal — "
        "you could get more return for the same risk, or less risk for the same return.",
        help=get_help_text("Efficient Frontier") if "Efficient Frontier" in GLOSSARY else None,
    )

    run_ef = st.button("📈 Generate Efficient Frontier", help="Takes ~15 seconds. Runs 3,000 random portfolios.")
    if run_ef or "ef_result" in st.session_state:
        if run_ef:
            with st.spinner("Generating efficient frontier (3,000 random portfolios + curve)…"):
                try:
                    ef = compute_efficient_frontier(tickers, build_price_matrix(tickers, returns_m.tail(window)), n_portfolios=3_000, n_frontier_points=40)
                    st.session_state["ef_result"] = ef
                except Exception as e:
                    st.error(f"Efficient frontier failed: {e}")
                    ef = None
        else:
            ef = st.session_state.get("ef_result")

        if ef:
            rand_df   = ef["random"]
            frontier  = ef["frontier"]
            min_var   = ef["min_var"]
            max_sharpe = ef["max_sharpe"]

            # Current portfolio point
            from analytics.portfolio_optimizer import expected_returns_historical, portfolio_performance
            from backtesting.stress_testing import compute_covariance_matrix
            cur_tickers = [t for t in tickers if t in returns_m.columns]
            mu   = expected_returns_historical(returns_m[cur_tickers], window).values
            cov_ann, _ = compute_covariance_matrix(returns_m[cur_tickers], window)
            w_cur = np.array([weights.get(t, 1/len(cur_tickers)) for t in cur_tickers])
            w_cur /= w_cur.sum()
            cur_r, cur_v, cur_s = portfolio_performance(w_cur, mu, cov_ann.values)

            fig = go.Figure()

            # Random portfolios (coloured by Sharpe)
            fig.add_trace(go.Scatter(
                x=rand_df["volatility"] * 100,
                y=rand_df["return"] * 100,
                mode="markers",
                marker=dict(
                    color=rand_df["sharpe"],
                    colorscale="Viridis",
                    size=3, opacity=0.4,
                    colorbar=dict(title=dict(text="Sharpe", font=dict(color="#90A4AE")),
                                   tickfont=dict(color="#90A4AE")),
                ),
                name="Random portfolios",
                hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
            ))

            # Frontier curve
            if frontier:
                f_vols = [p["vol"]*100 for p in frontier]
                f_rets = [p["ret"]*100 for p in frontier]
                fig.add_trace(go.Scatter(
                    x=f_vols, y=f_rets,
                    mode="lines", line=dict(color=BRAND_COLOR, width=2.5),
                    name="Efficient Frontier",
                ))

            # Key portfolios
            for label, res, color, sym in [
                ("Min Variance",  min_var,    SUCCESS_COLOR, "diamond"),
                ("Max Sharpe",    max_sharpe, ACCENT_COLOR,  "star"),
                ("Current",       {"ann_vol": cur_v, "ann_return": cur_r}, NEUTRAL_COLOR, "circle"),
            ]:
                if res:
                    fig.add_trace(go.Scatter(
                        x=[res.get("ann_vol",0)*100],
                        y=[res.get("ann_return",0)*100],
                        mode="markers+text",
                        marker=dict(color=color, size=14, symbol=sym,
                                    line=dict(color="white", width=1.5)),
                        text=[label], textposition="top center",
                        textfont=dict(color=color, size=11),
                        name=label,
                    ))

            fig.update_layout(
                **PLOT_LAYOUT,
                title="Efficient Frontier — Risk vs Return Tradeoff",
                xaxis_title="Annualised Volatility (%)",
                yaxis_title="Annualised Return (%)",
                height=480,
            )
            st.plotly_chart(fig, width="stretch")


# ─── Tab: Liquidity Risk ──────────────────────────────────────────────────────

def render_liquidity(data: Dict):
    section_intro(
        "Liquidity Risk",
        "How quickly can you exit this portfolio without moving the market?",
        "Liquidity risk is often ignored until it's too late. A portfolio of small-cap stocks "
        "may look attractive on paper but take weeks to exit. The Amihud ratio measures "
        "how much prices move per unit of trading volume — higher = more illiquid.",
    )
    from analytics.liquidity_risk import full_liquidity_report

    weights = data["weights"]
    returns_m = data["returns_m"]

    st.subheader("💧 Liquidity Risk Analysis")

    pv_input = st.number_input(
        "Portfolio Value (₹ Crore)",
        min_value=1.0, max_value=100_000.0, value=100.0, step=10.0,
        help="Used to estimate position sizes in ₹ terms for liquidation horizon.",
    )

    with st.spinner("Computing liquidity metrics from PostgreSQL volume data…"):
        try:
            report = full_liquidity_report(
                weights, returns_m,
                portfolio_value=pv_input * 1e7,  # Crore → ₹
            )
        except Exception as e:
            st.error(f"Liquidity report failed: {e}")
            return

    # ── Alerts ────────────────────────────────────────────────────────────────
    alerts = report.get("alerts", [])
    if alerts:
        for alert in alerts:
            level = alert.get("level", "ℹ️")
            if "WARNING" in level:
                st.warning(alert["message"])
            else:
                st.info(alert["message"])
    else:
        st.success("✅ No concentration alerts — portfolio appears well-diversified.")

    # ── Concentration metrics ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HHI Index",         fmt_val(report.get("hhi"), 4),
               help="Herfindahl-Hirschman Index. 1/n = perfect diversification. 1.0 = all in one stock.")
    c2.metric("Effective Positions", fmt_val(report.get("effective_n"), 1),
               help="Effective number of independent bets = 1/HHI. Higher is more diversified.")
    lvar = report.get("lvar") or {}
    c3.metric("Market VaR (99%, 1d)", fmt_pct(lvar.get("market_var_1d")),
               help="Standard historical VaR without liquidity adjustment.")
    c4.metric("Liquidity-Adj VaR",   fmt_pct(lvar.get("liquidity_adjusted_var")),
               help="VaR adjusted for the cost and time to liquidate positions. Always ≥ Market VaR.")

    st.divider()

    # ── Liquidation horizon table ──────────────────────────────────────────────
    liq_df = report.get("liq_detail", pd.DataFrame())
    if not liq_df.empty:
        st.markdown("#### Estimated Liquidation Horizon by Position")
        st.caption("Assumes trading at most 20% of average daily volume per day.")
        st.dataframe(liq_df, width="stretch", hide_index=True)

        # Horizon bar chart
        fig = go.Figure(go.Bar(
            x=liq_df["Ticker"],
            y=liq_df["Liquidation Days"],
            marker_color=[DANGER_COLOR if d > 5 else (ACCENT_COLOR if d > 2 else SUCCESS_COLOR)
                          for d in liq_df["Liquidation Days"]],
            text=liq_df["Liquidation Days"].astype(str) + "d",
            textposition="outside",
        ))
        fig.update_layout(**PLOT_LAYOUT, title="Days to Liquidate Each Position (20% ADV rule)",
                          yaxis_title="Trading Days", height=300)
        st.plotly_chart(fig, width="stretch")

    # ── Amihud heatmap ────────────────────────────────────────────────────────
    amihud = report.get("amihud", pd.Series(dtype=float))
    if not amihud.empty:
        st.divider()
        st.markdown("#### Amihud Illiquidity Ratio",
                    help="Measures price impact per ₹ of volume traded. "
                         "Higher = stock price moves more per unit of trading = more illiquid.")
        fig2 = go.Figure(go.Bar(
            x=amihud.index, y=amihud.values,
            marker=dict(
                color=amihud.values,
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(title=dict(text="Illiquidity", font=dict(color="#90A4AE")),
                               tickfont=dict(color="#90A4AE")),
            ),
        ))
        fig2.update_layout(**PLOT_LAYOUT, title="Amihud Illiquidity Score (normalised 0–1)",
                            yaxis_title="Illiquidity Score", height=300)
        st.plotly_chart(fig2, width="stretch")

    # ── LVaR premium ──────────────────────────────────────────────────────────
    if lvar:
        st.divider()
        st.markdown("#### Liquidity-Adjusted VaR Decomposition")
        lvar_rows = [
            {"Component": "Market VaR (99%, 1d)",     "Value": fmt_pct(lvar.get("market_var_1d"))},
            {"Component": "Horizon VaR (scaled to max liquidation period)",
                                                        "Value": fmt_pct(lvar.get("horizon_var"))},
            {"Component": "Bid-Ask Spread Cost",       "Value": fmt_pct(lvar.get("spread_cost"))},
            {"Component": "Liquidity-Adjusted VaR",   "Value": fmt_pct(lvar.get("liquidity_adjusted_var"))},
            {"Component": "LVaR Premium (extra cost)", "Value": fmt_pct(lvar.get("lvar_premium"))},
            {"Component": "Max Liquidation Horizon",   "Value": f"{lvar.get('liquidity_horizon_days',0)} days"},
        ]
        st.dataframe(pd.DataFrame(lvar_rows), width="stretch", hide_index=True)


# ─── Tab: Factor Risk Attribution ────────────────────────────────────────────

def render_factor_risk(data: Dict):
    section_intro(
        "Factor Risk Attribution",
        "Decompose portfolio returns and risk into systematic factor exposures.",
        "OLS regression tells us: how much of portfolio return comes from momentum? "
        "From quality? From the market itself? The t-statistic shows which exposures "
        "are statistically significant. The copula section shows how assets are "
        "correlated in the tails — which pairs tend to fall together in a crisis.",
    )
    from analytics.factor_risk_decomposition import (
        build_factor_return_proxies, run_factor_regression,
        factor_attribution_table, tail_dependence_summary,
        copula_vs_gaussian_var, fit_t_copula,
    )

    returns_m = data["returns_m"]
    factor_df = data["factor_df"]
    weights   = data["weights"]
    tickers   = list(weights.keys())
    port_r    = data["summary"]["portfolio_returns"]
    bench_r   = data["bench_r"].reindex(port_r.index)

    st.subheader("📊 Factor Return Attribution")

    # ── Factor regression ──────────────────────────────────────────────────────
    with st.spinner("Running OLS factor regression…"):
        try:
            factor_proxies = build_factor_return_proxies(returns_m, factor_df)
            reg_result     = run_factor_regression(port_r, factor_proxies, bench_r)
        except Exception as e:
            st.error(f"Factor regression failed: {e}")
            return

    if "error" in reg_result:
        st.warning(reg_result["error"])
    else:
        # R² and alpha cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² (Model Fit)",   fmt_val(reg_result["r_squared"], 3),
                   help="How much portfolio variance is explained by the 5 factors. 1.0 = perfect fit.")
        m2.metric("Adj. R²",          fmt_val(reg_result["r_squared_adj"], 3))
        m3.metric("Annual Alpha",     fmt_pct(reg_result["alpha_annual"]),
                   help="Return unexplained by factors — the manager's 'skill' component. Positive = good.")
        m4.metric("Observations",     str(reg_result["n_obs"]))

        st.divider()

        # Attribution table
        st.markdown("#### Factor Attribution Table")
        attr_df = factor_attribution_table(reg_result)
        st.dataframe(attr_df, width="stretch", hide_index=True)

        # Risk contribution pie
        risk_pct = reg_result.get("risk_pct", {})
        residual_pct = 100 - sum(risk_pct.values())
        labels = [f.replace("_factor","").replace("_"," ").title() for f in risk_pct] + ["Idiosyncratic"]
        values = list(risk_pct.values()) + [max(0, residual_pct)]

        col1, col2 = st.columns(2)
        with col1:
            fig_pie = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.45,
                marker=dict(colors=px.colors.qualitative.Bold),
                textinfo="label+percent",
            ))
            fig_pie.update_layout(**PLOT_LAYOUT, title="Risk Attribution by Factor", height=340, showlegend=False)
            st.plotly_chart(fig_pie, width="stretch")

        with col2:
            # Return contribution bar chart
            ret_contrib = reg_result.get("return_contribution", {})
            fc_labels = [f.replace("_factor","").replace("_"," ").title() for f in ret_contrib]
            fc_vals   = [v * 100 for v in ret_contrib.values()]
            fig_bar = go.Figure(go.Bar(
                x=fc_labels, y=fc_vals,
                marker_color=[SUCCESS_COLOR if v > 0 else DANGER_COLOR for v in fc_vals],
                text=[f"{v:+.2f}%" for v in fc_vals], textposition="outside",
            ))
            fig_bar.update_layout(**PLOT_LAYOUT, title="Return Contribution by Factor (%)",
                                   yaxis_ticksuffix="%", height=340)
            st.plotly_chart(fig_bar, width="stretch")

    # ── Copula Dependency ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔗 Tail Dependency Modelling (t-Copula)")
    st.markdown(
        "The **Gaussian** model assumes assets are independent in the tails. "
        "The **Student-t copula** captures the reality that stocks tend to "
        "crash *together* more than normal theory predicts. "
        "Higher tail dependence = higher joint downside risk.",
        help="Tail dependence coefficient λ: probability that asset A crashes given asset B crashes. "
             "0 = independent. 1 = always crash together.",
    )

    run_cop = st.button("🔗 Run Copula Analysis", help="Fits t-copula and computes pairwise tail dependence. ~10 seconds.")
    if run_cop or "copula_result" in st.session_state:
        if run_cop:
            with st.spinner("Fitting Student-t copula…"):
                try:
                    avail = [t for t in tickers if t in returns_m.columns]
                    tdc   = tail_dependence_summary(returns_m, avail)
                    cmp   = copula_vs_gaussian_var(returns_m, weights, confidence=0.99)
                    st.session_state["copula_result"] = {"tdc": tdc, "cmp": cmp}
                except Exception as e:
                    st.error(f"Copula analysis failed: {e}")
                    st.session_state["copula_result"] = None

        res = st.session_state.get("copula_result")
        if res:
            tdc = res["tdc"]
            cmp = res["cmp"]

            if not tdc.empty:
                fig_tdc = go.Figure(go.Heatmap(
                    z=tdc.values.astype(float),
                    x=tdc.columns.tolist(),
                    y=tdc.index.tolist(),
                    colorscale="Reds",
                    text=tdc.round(3).values.astype(str),
                    texttemplate="%{text}",
                    colorbar=dict(title=dict(text="λ (tail dep.)",
                                             font=dict(color="#90A4AE")),
                                   tickfont=dict(color="#90A4AE")),
                    zmin=0, zmax=1,
                ))
                fig_tdc.update_layout(**PLOT_LAYOUT,
                                       title="Pairwise Tail Dependence (t-Copula) — Higher = Crash Together More",
                                       height=420)
                st.plotly_chart(fig_tdc, width="stretch")

            # Copula vs Gaussian VaR comparison
            st.markdown("#### Copula vs Gaussian VaR Comparison")
            cop_rows = [
                {"Model": "Gaussian (Cholesky)",  "VaR 99%": fmt_pct(abs(cmp["gaussian_var"])),
                 "CVaR 99%": fmt_pct(abs(cmp["gaussian_cvar"]))},
                {"Model": "Student-t Copula",     "VaR 99%": fmt_pct(abs(cmp["t_copula_var"])),
                 "CVaR 99%": fmt_pct(abs(cmp["t_copula_cvar"]))},
                {"Model": "Difference (model risk)", "VaR 99%": fmt_pct(abs(cmp["var_difference"])),
                 "CVaR 99%": fmt_pct(abs(cmp["cvar_difference"]))},
            ]
            st.dataframe(pd.DataFrame(cop_rows), width="stretch", hide_index=True)
            st.caption(f"💡 {cmp['interpretation']}")


# ─── Tab: GBM Simulation (Probability Cone) ──────────────────────────────────

def render_gbm_simulation(data: Dict):
    section_intro(
        "GBM Path Simulation",
        "Project 10,000 possible portfolio trajectories over the next 12 months.",
        "Geometric Brownian Motion generates a fan of possible futures. "
        "The cone shows where 90% of simulated paths land. "
        "The wider the cone, the more uncertain the outcome. "
        "This is how investment banks visualise forward risk to clients.",
    )
    from backtesting.stress_testing import run_gbm_simulation

    returns_m = data["returns_m"]
    weights   = data["weights"]

    st.subheader("🌊 GBM Portfolio Path Simulation")
    c1, c2 = st.columns(2)
    with c1:
        horizon = st.selectbox("Forecast horizon", [21, 63, 126, 252],
                                format_func=lambda x: {21:"1 Month",63:"3 Months",
                                                        126:"6 Months",252:"1 Year"}[x],
                                index=3)
    with c2:
        n_paths = st.select_slider("Number of paths", [1_000, 5_000, 10_000, 25_000], value=10_000)

    run_gbm = st.button("🌊 Run GBM Simulation", type="primary")

    if run_gbm or "gbm_result" in st.session_state:
        if run_gbm:
            with st.spinner(f"Simulating {n_paths:,} portfolio paths over {horizon} days…"):
                try:
                    gbm = run_gbm_simulation(weights, returns_m, n_paths, horizon, seed=42)
                    st.session_state["gbm_result"] = gbm
                except Exception as e:
                    st.error(f"GBM simulation failed: {e}")
                    gbm = None
        else:
            gbm = st.session_state.get("gbm_result")

        if gbm:
            # Metric cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Median Outcome",   f"{gbm['median_terminal']*100:.1f}%",
                       help="50th percentile terminal portfolio value (base = 100%).")
            m2.metric("P(Loss)",          fmt_pct(gbm["prob_loss"]),
                       help="Probability the portfolio loses value over the horizon.")
            m3.metric("P(+10% Gain)",     fmt_pct(gbm["prob_gain_10pct"]),
                       help="Probability of gaining more than 10% over the horizon.")
            m4.metric("VaR 99% (terminal)", fmt_pct(gbm["var_99_terminal"]),
                       help="Worst-case loss (99th percentile) at the end of the horizon.")

            st.divider()

            # Fan chart
            pcts   = gbm["percentiles"]    # shape (7, horizon_days)
            labels = gbm["percentile_labels"]
            days   = list(range(1, gbm["horizon_days"] + 1))
            # Prepend day 0 (starting at 1.0)
            days_full = [0] + days

            fig = go.Figure()

            # Shaded bands: 5–95, 10–90, 25–75
            band_pairs = [(0, 6, "rgba(0,212,255,0.07)","5–95%"),
                          (1, 5, "rgba(0,212,255,0.12)","10–90%"),
                          (2, 4, "rgba(0,212,255,0.18)","25–75%")]

            for lo_idx, hi_idx, fill, name in band_pairs:
                lo_vals = [1.0] + (pcts[lo_idx] * 100).tolist()
                hi_vals = [1.0] + (pcts[hi_idx] * 100).tolist()

                fig.add_trace(go.Scatter(
                    x=days_full + days_full[::-1],
                    y=hi_vals + lo_vals[::-1],
                    fill="toself", fillcolor=fill,
                    line=dict(color="rgba(0,0,0,0)"),
                    name=name, showlegend=True,
                    hoverinfo="skip",
                ))

            # Median line
            median_line = [100.0] + (pcts[3] * 100).tolist()
            fig.add_trace(go.Scatter(
                x=days_full, y=median_line,
                line=dict(color=BRAND_COLOR, width=2.5),
                name="Median (50th pct)",
            ))

            # Baseline
            fig.add_hline(y=100, line_dash="dot", line_color=NEUTRAL_COLOR,
                           annotation_text="Starting value", annotation_position="right")

            fig.update_layout(
                **{k: v for k, v in PLOT_LAYOUT.items() if k != "legend"},
                title=f"GBM Probability Cone — {n_paths:,} Simulated Portfolio Paths",
                xaxis_title="Trading Days",
                yaxis_title="Portfolio Value (Base = 100)",
                height=450,
                showlegend=False,   # hover tooltip shows values; legend clutters 10k-path chart
            )
            st.plotly_chart(fig, width="stretch")

            # Terminal distribution histogram
            fig2 = go.Figure(go.Histogram(
                x=(gbm["final_values"] - 1) * 100,
                nbinsx=100,
                marker_color=BRAND_COLOR, opacity=0.75,
                name="Terminal P&L",
            ))
            fig2.add_vline(x=0, line_dash="dash", line_color=NEUTRAL_COLOR,
                            annotation_text="Break-even")
            fig2.add_vline(x=-gbm["var_99_terminal"]*100,
                            line_color=DANGER_COLOR, line_dash="dot",
                            annotation_text=f"VaR 99%: {gbm['var_99_terminal']*100:.1f}%",
                            annotation_font_color=DANGER_COLOR)
            fig2.update_layout(**PLOT_LAYOUT,
                                title=f"Terminal P&L Distribution at {horizon}-Day Horizon",
                                xaxis_title="Portfolio Return (%)",
                                height=300)
            st.plotly_chart(fig2, width="stretch")



# ─── Consolidated Tab: Quantitative Risk ──────────────────────────────────────

def render_quantitative_risk(data: Dict):
    """
    Combines: Risk Metrics, Monte Carlo VaR, and GBM Simulation Paths
    into a single tab with clearly labelled sections.
    All existing render quality is preserved — just reorganised under headers.
    """
    st.markdown("""
    <div style="background:#0E1A2E;border:1px solid #1E2A3A;border-radius:8px;
                padding:12px 16px;margin-bottom:16px;">
        <span style="color:#00D4FF;font-size:13px;font-weight:600">
        📉 Quantitative Risk
        </span>
        <span style="color:#607080;font-size:12px;margin-left:8px">
        Risk Metrics · Monte Carlo VaR · Stress Testing · GBM Simulation
        </span>
    </div>
    """, unsafe_allow_html=True)

    inner = st.tabs([
        "📊 Risk Metrics",
        "🎲 Monte Carlo VaR",
        "⚡ Stress Tests",
        "🌊 GBM Paths",
    ])
    with inner[0]: render_risk_metrics(data)
    with inner[1]: _render_mc_var_section(data)
    with inner[2]: _render_stress_section(data)
    with inner[3]: render_gbm_simulation(data)


def _render_mc_var_section(data: Dict):
    """Monte Carlo VaR section extracted from render_stress_testing."""
    from backtesting.stress_testing import (
        run_monte_carlo, run_multi_horizon_var, run_regime_var,
        component_var, save_var_result, CONFIDENCE_LEVELS,
    )
    returns_m = data["returns_m"]
    weights   = data["weights"]

    section_intro(
        "Monte Carlo VaR",
        "10,000 simulations driven by your portfolio's historical covariance matrix.",
        "VaR answers: 'How much could I lose on a bad day?' CVaR answers: 'On those bad days, "
        "how bad does it get on average?' Confidence levels: 95% = 1-in-20 day event, "
        "99% = 1-in-100 day event.",
    )
    st.subheader("🎲 Monte Carlo Value at Risk")
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        n_sims = st.select_slider("Simulations", [1_000, 5_000, 10_000, 25_000],
                                   value=10_000, key="mc_nsims",
                                   help="More = more accurate but slower. 10,000 is standard.")
    with ctrl2:
        hp = st.selectbox("Holding Period", [1, 5, 10, 21], key="mc_hp",
                           format_func=lambda x: f"{x} day{'s' if x>1 else ''}",
                           help="Days ahead to project. Scales volatility by √holding_period.")
    with ctrl3:
        cl_choices = st.multiselect("Confidence Levels", ["90%","95%","97.5%","99%"],
                                     default=["95%","97.5%","99%"], key="mc_cl")

    if st.button("▶ Run Monte Carlo", type="primary", key="run_mc2"):
        with st.spinner("Running simulations…"):
            try:
                mc = run_monte_carlo(weights, returns_m, n_simulations=n_sims, holding_period=hp)
                st.session_state["mc2_result"] = mc
            except Exception as e:
                st.error(f"Monte Carlo failed: {e}")

    mc = st.session_state.get("mc2_result")
    if mc:
        vt = mc["var_table"]
        cl_map = {"90%":0.90,"95%":0.95,"97.5%":0.975,"99%":0.99}
        sel_cls = [cl_map[c] for c in cl_choices if c in cl_map] or [0.95,0.975,0.99]
        cols = st.columns(len(sel_cls)*2)
        for i,cl in enumerate(sel_cls):
            lbl = f"{int(cl*100)}%" if cl!=0.975 else "97.5%"
            cols[i*2].metric(f"VaR {lbl}", fmt_pct(abs(vt[cl]["var"])),
                              help=get_help_text("VaR (95%)"))
            cols[i*2+1].metric(f"CVaR {lbl}", fmt_pct(abs(vt[cl]["cvar"])),
                                help=get_help_text("CVaR (95%)"))
        st.divider()
        pnl = mc["portfolio_pnl"]*100
        fig = go.Figure(go.Histogram(x=pnl, nbinsx=100, marker_color=BRAND_COLOR, opacity=0.75))
        for cl in sel_cls:
            lbl = f"{int(cl*100)}%" if cl!=0.975 else "97.5%"
            fig.add_vline(x=vt[cl]["var"]*100, line_dash="dash", line_color=DANGER_COLOR,
                          annotation_text=f"VaR {lbl}", annotation_font_color=DANGER_COLOR)
        fig.update_layout(**PLOT_LAYOUT, title=f"P&L Distribution ({n_sims:,} sims)",
                          xaxis_title="Portfolio Return (%)", height=340)
        st.plotly_chart(fig, width="stretch")

        st.divider()
        st.markdown("#### Multi-Horizon VaR")
        with st.spinner("Computing…"):
            try:
                mh = run_multi_horizon_var(weights, returns_m, n_simulations=min(n_sims,5_000))
                st.dataframe((mh*100).round(3), width="stretch")
            except Exception as e:
                st.warning(f"Multi-horizon VaR: {e}")

        st.divider()
        st.markdown("#### Component VaR  — which positions drive risk?",
                    help=get_help_text("Component VaR"))
        try:
            cv = component_var(weights, returns_m).reset_index()
            cv["Weight"] = (cv["Weight"]*100).round(2)
            cv["% of Total VaR"] = cv["% of Total VaR"].round(2)
            fig2 = go.Figure(go.Bar(x=cv["Ticker"], y=cv["% of Total VaR"],
                                     marker_color=[DANGER_COLOR if v>0 else SUCCESS_COLOR
                                                   for v in cv["% of Total VaR"]],
                                     text=[f"{v:.1f}%" for v in cv["% of Total VaR"]],
                                     textposition="outside"))
            fig2.update_layout(**PLOT_LAYOUT, title="% Contribution to Portfolio VaR (99%, 1d)",
                               height=280)
            st.plotly_chart(fig2, width="stretch")
        except Exception as e:
            st.warning(f"Component VaR: {e}")

        st.divider()
        st.markdown("#### Volatility Regime Comparison")
        with st.spinner("Computing regimes…"):
            try:
                reg = run_regime_var(weights, returns_m, n_simulations=3_000)
                st.dataframe(reg, width="stretch", hide_index=True)
            except Exception as e:
                st.warning(f"Regime analysis: {e}")


def _render_stress_section(data: Dict):
    """Stress testing scenarios section."""
    from backtesting.stress_testing import STRESS_SCENARIOS, run_stress_scenario
    returns_m = data["returns_m"]
    weights   = data["weights"]

    section_intro(
        "Historical & Hypothetical Stress Tests",
        "Replay named market crises through your current portfolio weights.",
        "Each scenario applies a systematic shock (based on the real event) plus elevated "
        "volatility. Results show estimated portfolio loss — not a forecast, but a "
        "'what if history repeats?' measure of tail exposure.",
    )
    st.subheader("⚡ Stress Scenarios")
    all_names = list(STRESS_SCENARIOS.keys())
    selected  = st.multiselect("Select scenarios", all_names, default=all_names[:6], key="stress_sel2")

    if st.button("▶ Run Stress Tests", type="secondary", key="run_stress2"):
        with st.spinner("Running scenarios…"):
            rows = []
            prog = st.progress(0)
            for i, name in enumerate(selected):
                try:
                    res = run_stress_scenario(name, weights, returns_m, n_simulations=5_000)
                    rows.append({
                        "Scenario":        res["scenario"],
                        "Period":          res["period"],
                        "Mkt Shock":       fmt_pct(res["market_shock"]),
                        "Portfolio Loss":  fmt_pct(abs(res["systematic_shock"])),
                        "VaR 95%":         fmt_pct(abs(res["var_table"][0.95]["var"])),
                        "CVaR 99%":        fmt_pct(abs(res["var_table"][0.99]["cvar"])),
                        "_raw":            res["systematic_shock"],
                    })
                    prog.progress((i+1)/len(selected))
                except Exception as e:
                    rows.append({"Scenario":name,"Period":"—","Mkt Shock":"—",
                                 "Portfolio Loss":f"❌ {e}","VaR 95%":"—","CVaR 99%":"—","_raw":0})
            st.session_state["stress2_results"] = rows
            prog.empty()

    rows = st.session_state.get("stress2_results")
    if rows:
        disp = [{k:v for k,v in r.items() if k!="_raw"} for r in rows]
        st.dataframe(pd.DataFrame(disp), width="stretch", hide_index=True)
        sorted_r = sorted(rows, key=lambda x: x["_raw"])
        fig = go.Figure(go.Bar(
            x=[abs(r["_raw"])*100 for r in sorted_r],
            y=[r["Scenario"] for r in sorted_r],
            orientation="h",
            marker_color=[DANGER_COLOR if abs(r["_raw"])>0.2 else ACCENT_COLOR for r in sorted_r],
            text=[f"{abs(r['_raw'])*100:.1f}%" for r in sorted_r],
            textposition="outside",
        ))
        fig.update_layout(**PLOT_LAYOUT, title="Estimated Portfolio Loss by Scenario",
                          xaxis_title="Loss (%)", height=max(280, len(sorted_r)*36))
        st.plotly_chart(fig, width="stretch")


# ─── Consolidated Tab: Risk Attribution ───────────────────────────────────────

def render_risk_attribution(data: Dict):
    """
    Combines: Liquidity Risk and Factor Risk Decomposition
    into a single tab with clearly labelled inner sections.
    """
    st.markdown("""
    <div style="background:#0E1A2E;border:1px solid #1E2A3A;border-radius:8px;
                padding:12px 16px;margin-bottom:16px;">
        <span style="color:#00D4FF;font-size:13px;font-weight:600">
        🔬 Risk Attribution
        </span>
        <span style="color:#607080;font-size:12px;margin-left:8px">
        Liquidity Risk · Factor Risk Decomposition · Copula Dependency
        </span>
    </div>
    """, unsafe_allow_html=True)

    inner = st.tabs([
        "💧 Liquidity",
        "📊 Factor Risk",
    ])
    with inner[0]: render_liquidity(data)
    with inner[1]: render_factor_risk(data)


# ─── Tab: Glossary ────────────────────────────────────────────────────────────

def render_glossary():
    st.subheader("📚 Metric Glossary")
    st.markdown(
        "Searchable reference for every metric in this platform. "
        "Click any metric card in the dashboard for the same explanation inline."
    )
    st.divider()

    search = st.text_input("🔍 Search glossary…", placeholder="e.g. Sharpe, CAGR, momentum")

    tag_filter = st.radio(
        "Filter by category",
        ["All", "Performance", "Risk", "Factor", "ML/AI"],
        horizontal=True,
    )
    tag_map = {"All": None, "Performance": "perf", "Risk": "risk",
               "Factor": "factor", "ML/AI": "ml"}
    selected_tag = tag_map[tag_filter]

    shown = 0
    for term, entry in GLOSSARY.items():
        # Tag filter
        if selected_tag and entry.get("tag") != selected_tag:
            continue
        # Search filter
        if search:
            s = search.lower()
            if s not in term.lower() and s not in entry["short"].lower() and s not in entry["long"].lower():
                continue
        glossary_card(term, entry)
        shown += 1

    if shown == 0:
        st.info(f"No metrics match '{search}'. Try a shorter search term.")


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    st.markdown(f"""
    <div style="border-bottom:1px solid #1E2A3A;padding-bottom:16px;margin-bottom:8px;">
        <h1 style="margin:0;font-size:28px;color:#E0E8F0;">📊 {DASHBOARD_TITLE}</h1>
        <p style="margin:4px 0 0 0;color:#00D4FF;font-size:13px;letter-spacing:1px;">
            {DASHBOARD_SUBTITLE.upper()}
        </p>
    </div>
    """, unsafe_allow_html=True)

    tickers, method, custom_weights, data_years, run_bt, run_btn = render_sidebar()

    if not run_btn and "analysis_data" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:80px;color:#607080;">
            <div style="font-size:48px;">📊</div>
            <h3 style="color:#90A4AE;">Configure your portfolio in the sidebar and click Run Analysis</h3>
            <p>Enter at least 5 NSE tickers, select a weighting method, and start your analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    if run_btn:
        if len(tickers) < 5:
            st.error("Please enter at least 5 tickers.")
            return
        with st.spinner("Running full analysis pipeline…"):
            try:
                data = run_full_analysis(
                    tickers, method,
                    custom_weights_json=custom_weights if method == "Custom Weighted" else None,
                )
                st.session_state["analysis_data"]    = data
                st.session_state["analysis_tickers"] = tickers
                st.session_state["analysis_method"]  = method
                if data.get("no_news_key"):
                    st.toast("ℹ️ No NewsAPI key — Events tab will be empty.", icon="ℹ️")
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    if "analysis_data" not in st.session_state:
        return

    data    = st.session_state["analysis_data"]
    tickers = st.session_state["analysis_tickers"]
    method  = st.session_state["analysis_method"]

    tabs = st.tabs([
        "📈 Overview",    "🍩 Allocation",    "🔗 Correlation",
        "⚡ Factors",     "🎯 Signals",       "🧠 SHAP",
        "📰 Events",      "📉 Backtest",
        "📉 Quant Risk",  "🔬 Risk Attribution",
        "⚙️ Optimisation",
        "📚 Glossary",
    ])

    with tabs[0]:  render_overview(data)
    with tabs[1]:  render_allocation(data)
    with tabs[2]:  render_correlation(data)
    with tabs[3]:  render_factors(data)
    with tabs[4]:  render_signals(data)
    with tabs[5]:  render_shap(data)
    with tabs[6]:  render_events(data)
    with tabs[7]:
        if run_bt:
            render_backtest(data, tickers, method)
        else:
            st.info("Enable 'Run Backtest' in the sidebar to see backtest results.")
    with tabs[8]:  render_quantitative_risk(data)
    with tabs[9]:  render_risk_attribution(data)
    with tabs[10]: render_optimization(data)
    with tabs[11]: render_glossary()


if __name__ == "__main__":
    main()