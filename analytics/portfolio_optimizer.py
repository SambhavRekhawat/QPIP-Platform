"""
analytics/portfolio_optimizer.py
----------------------------------
Institutional Portfolio Optimization Engine.

Implements:
1.  Mean-Variance Optimization     (max Sharpe / min volatility)
2.  Efficient Frontier generation
3.  Risk Parity Optimization       (equal risk contribution)
4.  Black-Litterman Optimization   (market equilibrium + views)
5.  CVaR (Expected Shortfall) Optimization
6.  Minimum Variance Portfolio

Uses PyPortfolioOpt as the primary solver with scipy fallback.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from loguru import logger
from sqlalchemy import text

from config import TRADING_DAYS_YEAR, RISK_FREE_RATE
from database.connection import get_db_engine
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

try:
    from pypfopt import EfficientFrontier, expected_returns, risk_models, BlackLittermanModel
    from pypfopt.efficient_frontier import EfficientCVaR
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    logger.warning("PyPortfolioOpt not installed. pip install pyportfolioopt  — using scipy fallback.")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def build_price_matrix(tickers, returns_matrix):
    data = returns_matrix[[t for t in tickers if t in returns_matrix.columns]].fillna(0)
    return (1 + data).cumprod() * 100


def compute_expected_returns(prices, frequency=TRADING_DAYS_YEAR):
    if PYPFOPT_AVAILABLE:
        try:
            return expected_returns.mean_historical_return(prices, frequency=frequency)
        except Exception:
            pass
    return prices.pct_change().dropna().mean() * frequency


def compute_cov_matrix(prices, frequency=TRADING_DAYS_YEAR):
    if PYPFOPT_AVAILABLE:
        try:
            return risk_models.CovarianceShrinkage(prices, frequency=frequency).ledoit_wolf()
        except Exception:
            pass
    return prices.pct_change().dropna().cov() * frequency


def _scipy_max_sharpe(mu, S):
    n = len(mu)
    def neg_sharpe(w):
        ret = float(w @ mu.values)
        vol = float(np.sqrt(w @ S.values @ w))
        return -(ret - RISK_FREE_RATE) / vol if vol > 1e-9 else 1e9
    res = minimize(neg_sharpe, np.ones(n)/n, method="SLSQP",
                   bounds=[(0,1)]*n,
                   constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
                   options={"maxiter":2000,"ftol":1e-10})
    w = res.x / res.x.sum()
    return {t: float(w[i]) for i, t in enumerate(mu.index)}


def _scipy_min_vol(S, tickers):
    n = len(tickers)
    def vol(w): return float(np.sqrt(w @ S.values @ w))
    res = minimize(vol, np.ones(n)/n, method="SLSQP",
                   bounds=[(0,1)]*n,
                   constraints=[{"type":"eq","fun":lambda w: w.sum()-1}])
    w = res.x / res.x.sum()
    return {t: float(w[i]) for i, t in enumerate(tickers)}


def _perf(weights_dict, mu, S):
    w = np.array([weights_dict.get(t, 0) for t in mu.index])
    ret = float(w @ mu.values)
    vol = float(np.sqrt(w @ S.values @ w)) if len(w)==len(S) else 0
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 1e-9 else 0
    return ret, vol, sharpe


# ─── Optimization Methods ─────────────────────────────────────────────────────

def optimize_max_sharpe(tickers, prices, max_single_weight=0.40):
    mu = compute_expected_returns(prices)
    S  = compute_cov_matrix(prices)

    if PYPFOPT_AVAILABLE:
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(0,1))
            ef.add_constraint(lambda w: w <= max_single_weight)
            ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
            weights = ef.clean_weights()
            ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE, verbose=False)
            return {"method":"max_sharpe","weights":{k:v for k,v in weights.items() if v>1e-4},
                    "expected_return":float(ret),"expected_vol":float(vol),"sharpe_ratio":float(sharpe),"mu":mu,"S":S}
        except Exception as e:
            logger.warning(f"EF max_sharpe failed ({e}), using scipy")

    weights = _scipy_max_sharpe(mu, S)
    ret, vol, sharpe = _perf(weights, mu, S)
    return {"method":"max_sharpe","weights":weights,"expected_return":ret,"expected_vol":vol,"sharpe_ratio":sharpe,"mu":mu,"S":S}


def optimize_min_volatility(tickers, prices):
    mu = compute_expected_returns(prices)
    S  = compute_cov_matrix(prices)

    if PYPFOPT_AVAILABLE:
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(0,1))
            ef.min_volatility()
            weights = ef.clean_weights()
            ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE, verbose=False)
            return {"method":"min_volatility","weights":{k:v for k,v in weights.items() if v>1e-4},
                    "expected_return":float(ret),"expected_vol":float(vol),"sharpe_ratio":float(sharpe),"mu":mu,"S":S}
        except Exception as e:
            logger.warning(f"EF min_vol failed ({e}), using scipy")

    weights = _scipy_min_vol(S, mu.index.tolist())
    ret, vol, sharpe = _perf(weights, mu, S)
    return {"method":"min_volatility","weights":weights,"expected_return":ret,"expected_vol":vol,"sharpe_ratio":sharpe,"mu":mu,"S":S}


def optimize_risk_parity(tickers, prices):
    mu = compute_expected_returns(prices)
    S  = compute_cov_matrix(prices)
    Sv = S.values
    n  = len(tickers)
    target = np.ones(n) / n

    def obj(w):
        pv = float(w @ Sv @ w)
        if pv <= 0: return 1e9
        rc = w * (Sv @ w) / pv
        return float(np.sum((rc - target)**2))

    res = minimize(obj, np.ones(n)/n, method="SLSQP",
                   bounds=[(0.001,1)]*n,
                   constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
                   options={"maxiter":5000,"ftol":1e-12})
    w = res.x / res.x.sum()
    weights = {t: float(w[i]) for i, t in enumerate(tickers)}

    pv = float(w @ Sv @ w)
    mrc = Sv @ w
    rc_arr = w * mrc / pv if pv > 0 else w
    ret, vol, sharpe = _perf(weights, mu, S)

    return {"method":"risk_parity","weights":weights,"expected_return":ret,"expected_vol":vol,
            "sharpe_ratio":sharpe,"risk_contribution":{t:float(rc_arr[i]) for i,t in enumerate(tickers)},
            "mu":mu,"S":S}


def optimize_black_litterman(tickers, prices, views=None, view_confidences=None, market_caps=None):
    if not PYPFOPT_AVAILABLE:
        logger.warning("PyPortfolioOpt needed for Black-Litterman. Falling back to max Sharpe.")
        return optimize_max_sharpe(tickers, prices)

    mu = compute_expected_returns(prices)
    S  = compute_cov_matrix(prices)

    if market_caps:
        total = sum(market_caps.get(t, 1e6) for t in tickers)
        mcap_w = {t: market_caps.get(t,1e6)/total for t in tickers}
    else:
        mcap_w = {t: 1.0/len(tickers) for t in tickers}

    try:
        if views:
            bl_views = {t: v for t,v in views.items() if t in tickers}
            confs    = view_confidences or [0.5]*len(bl_views)
            bl = BlackLittermanModel(S, pi="market", market_caps=mcap_w,
                                     absolute_views=bl_views, omega="idzorek",
                                     view_confidences=confs)
        else:
            bl = BlackLittermanModel(S, pi="market", market_caps=mcap_w)

        bl_returns = bl.bl_returns()
        bl_cov     = bl.bl_cov()
        ef = EfficientFrontier(bl_returns, bl_cov)
        ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        weights = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE, verbose=False)
    except Exception as e:
        logger.warning(f"Black-Litterman failed ({e}), falling back to max Sharpe")
        return optimize_max_sharpe(tickers, prices)

    return {"method":"black_litterman","weights":{k:v for k,v in weights.items() if v>1e-4},
            "expected_return":float(ret),"expected_vol":float(vol),"sharpe_ratio":float(sharpe),
            "views_used":views or {},"mu":mu,"S":S}


def optimize_cvar(tickers, returns_matrix):
    prices = build_price_matrix(tickers, returns_matrix)
    mu     = compute_expected_returns(prices)

    if not PYPFOPT_AVAILABLE:
        return optimize_max_sharpe(tickers, prices)

    hist_ret = returns_matrix[[t for t in tickers if t in returns_matrix.columns]].fillna(0)
    try:
        ef_cvar = EfficientCVaR(mu, hist_ret)
        ef_cvar.min_cvar()
        weights = ef_cvar.clean_weights()
        S = compute_cov_matrix(prices)
        ret, vol, sharpe = _perf(weights, mu, S)
        return {"method":"cvar_optimization","weights":{k:v for k,v in weights.items() if v>1e-4},
                "expected_return":ret,"expected_vol":vol,"sharpe_ratio":sharpe,"mu":mu,"S":S}
    except Exception as e:
        logger.warning(f"CVaR optimisation failed ({e})")
        return optimize_max_sharpe(tickers, prices)


# ─── Efficient Frontier ───────────────────────────────────────────────────────

def compute_efficient_frontier(tickers, prices, n_portfolios=5000, n_frontier_points=40):
    mu = compute_expected_returns(prices)
    S  = compute_cov_matrix(prices)
    n  = len(mu)
    np.random.seed(42)

    records = []
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n))
        ret = float(w @ mu.values)
        vol = float(np.sqrt(w @ S.values @ w))
        sharpe = (ret - RISK_FREE_RATE)/vol if vol>1e-9 else 0
        records.append({"return":ret,"volatility":vol,"sharpe":sharpe})
    random_df = pd.DataFrame(records)

    frontier_points = []
    if PYPFOPT_AVAILABLE:
        target_rets = np.linspace(mu.min()*1.05, mu.max()*0.90, n_frontier_points)
        for tr in target_rets:
            try:
                ef = EfficientFrontier(mu, S, weight_bounds=(0,1))
                ef.efficient_return(target_return=float(tr))
                fr, fv, fs = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)
                frontier_points.append({"return":float(fr),"volatility":float(fv),"sharpe":float(fs)})
            except Exception:
                pass

    max_sharpe_r = optimize_max_sharpe(tickers, prices)
    min_vol_r    = optimize_min_volatility(tickers, prices)

    return {"random_portfolios":random_df,
            "frontier_points":pd.DataFrame(frontier_points) if frontier_points else pd.DataFrame(),
            "max_sharpe":max_sharpe_r, "min_vol":min_vol_r, "mu":mu, "S":S}


def compare_portfolios(current_weights, optimised_result, mu, S):
    opt_weights  = optimised_result["weights"]
    all_tickers  = sorted(set(current_weights) | set(opt_weights))
    rows = []
    for t in all_tickers:
        cw = current_weights.get(t,0)
        ow = opt_weights.get(t,0)
        rows.append({"Ticker":t,"Current Weight":round(cw*100,2),"Optimal Weight":round(ow*100,2),
                     "Change (pp)":round((ow-cw)*100,2),
                     "Action":"↑ Increase" if ow>cw+0.02 else "↓ Reduce" if ow<cw-0.02 else "≈ Hold"})
    df = pd.DataFrame(rows)
    cw_arr = np.array([current_weights.get(t,0) for t in mu.index])
    c_ret  = float(cw_arr @ mu.values) if len(cw_arr)==len(mu) else 0
    c_vol  = float(np.sqrt(cw_arr @ S.values @ cw_arr)) if len(cw_arr)==len(S) else 0
    summary = {"current_return":c_ret,"current_vol":c_vol,
               "current_sharpe":(c_ret-RISK_FREE_RATE)/c_vol if c_vol>1e-9 else 0,
               "optimal_return":optimised_result["expected_return"],
               "optimal_vol":optimised_result["expected_vol"],
               "optimal_sharpe":optimised_result["sharpe_ratio"],
               "sharpe_improvement":optimised_result["sharpe_ratio"]-((c_ret-RISK_FREE_RATE)/c_vol if c_vol>1e-9 else 0)}
    return df, summary


def risk_contribution_table(weights, S):
    tickers = [t for t in weights if t in S.index]
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()
    Sv = S.loc[tickers,tickers].values
    pv = float(w @ Sv @ w)
    if pv <= 0: return pd.DataFrame()
    mrc = Sv @ w
    rc  = w * mrc
    pct = rc / pv * 100
    return pd.DataFrame({"Ticker":tickers,"Weight (%)": (w*100).round(2),
                          "Marginal Risk":mrc.round(6),"Risk Contribution":rc.round(6),
                          "% of Port. Risk":pct.round(2)}).sort_values("% of Port. Risk",ascending=False)


def save_optimization_result(result, current_weights, engine=None):
    if engine is None: engine = get_db_engine()
    record = {"method":result.get("method"),
               "tickers":json.dumps(list(result.get("weights",{}).keys())),
               "weights":json.dumps({k:round(v,6) for k,v in result.get("weights",{}).items()}),
               "expected_return":result.get("expected_return"),
               "expected_vol":result.get("expected_vol"),
               "sharpe_ratio":result.get("sharpe_ratio"),
               "current_weights":json.dumps({k:round(v,6) for k,v in current_weights.items()}),
               "improvement":0.0,"constraints":json.dumps({})}
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO optimization_results
                    (method,tickers,weights,expected_return,expected_vol,
                     sharpe_ratio,current_weights,improvement,constraints)
                VALUES
                    (:method,CAST(:tickers AS jsonb),CAST(:weights AS jsonb),
                     :expected_return,:expected_vol,:sharpe_ratio,
                     CAST(:current_weights AS jsonb),:improvement,
                     CAST(:constraints AS jsonb))"""), record)
        logger.info(f"✅ Saved optimization ({result.get('method')})")
    except Exception as e:
        logger.warning(f"Save optimization failed: {e}")


# ─── Convenience aliases used by the dashboard ───────────────────────────────
def expected_returns_historical(returns_matrix, window=252):
    """Return annualised mean returns from a returns DataFrame (alias for dashboard)."""
    data = returns_matrix.tail(window).fillna(0)
    return data.mean() * TRADING_DAYS_YEAR


def portfolio_performance(weights_arr, mu_arr, cov_arr):
    """
    Return (annualised_return, annualised_vol, sharpe) for a numpy weights vector.
    weights_arr : 1-D numpy array summing to 1
    mu_arr      : 1-D numpy array of annualised expected returns
    cov_arr     : 2-D numpy annualised covariance matrix
    """
    p_ret = float(np.dot(weights_arr, mu_arr))
    p_vol = float(np.sqrt(weights_arr @ cov_arr @ weights_arr))
    p_sharpe = (p_ret - RISK_FREE_RATE) / p_vol if p_vol > 0 else 0.0
    return p_ret, p_vol, p_sharpe
