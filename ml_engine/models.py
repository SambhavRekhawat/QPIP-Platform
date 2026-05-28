"""
ml_engine/models.py
--------------------
Machine Learning Signal Generation Engine.
  - Feature engineering from factors, returns, fundamentals, events
  - XGBoost / LightGBM / Random Forest classifiers
  - BUY / HOLD / SELL signal generation
  - SHAP explainability
  - Persistence of signals to PostgreSQL
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
import warnings
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path
from loguru import logger
from sqlalchemy import text
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
import xgboost as xgb
import lightgbm as lgb
import shap

from factors.composite import clean_numeric_series   # central sanitiser
from config import (
    ML_SIGNAL_THRESHOLD, ML_TEST_SIZE, ML_RANDOM_STATE,
    ML_CV_FOLDS, MODELS_DIR, TRADING_DAYS_YEAR
)
from database.connection import get_db_engine

warnings.filterwarnings("ignore")


# ─── Feature Engineering ─────────────────────────────────────────────────────

def build_feature_matrix(
    tickers: List[str],
    close_matrix: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    factor_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a cross-sectional feature matrix for ML training / inference.
    One row per ticker.
    """
    # Only process tickers that have data in the returns matrix
    available = [t for t in tickers if t in returns_matrix.columns]
    missing   = set(tickers) - set(available)
    if missing:
        logger.warning(f"  build_feature_matrix: skipping {missing} — not in returns matrix")
    tickers = available

    features = {}

    for ticker in tickers:
        feat = {}  # ticker is the dict key → becomes index → reset_index() adds it as column

        # ── Price / Return features ──────────────────────────────────────
        if ticker in returns_matrix.columns:
            ret = returns_matrix[ticker].dropna()
            feat["ann_return"]    = float((1 + ret).prod() ** (TRADING_DAYS_YEAR / max(len(ret), 1)) - 1)
            feat["volatility_63d"] = float(ret.tail(63).std() * np.sqrt(TRADING_DAYS_YEAR))
            feat["volatility_21d"] = float(ret.tail(21).std() * np.sqrt(TRADING_DAYS_YEAR))
            feat["skewness"]       = float(ret.tail(126).skew())
            feat["kurtosis"]       = float(ret.tail(126).kurtosis())
            # Recent momentum
            feat["ret_1m"]  = float(ret.tail(21).sum())
            feat["ret_3m"]  = float(ret.tail(63).sum())
            feat["ret_6m"]  = float(ret.tail(126).sum())
            feat["ret_12m"] = float(ret.tail(252).sum())

        # ── Factor scores ────────────────────────────────────────────────
        frow = factor_df[factor_df["ticker"] == ticker]
        if not frow.empty:
            frow = frow.iloc[0]
            feat["momentum_score"]   = float(frow.get("momentum_score",   np.nan) or np.nan)
            feat["quality_score"]    = float(frow.get("quality_score",    np.nan) or np.nan)
            feat["value_score"]      = float(frow.get("value_score",      np.nan) or np.nan)
            feat["low_vol_score"]    = float(frow.get("volatility_score", np.nan) or np.nan)
            feat["composite_score"]  = float(frow.get("composite_score",  np.nan) or np.nan)
            feat["beta"]             = float(frow.get("beta",             np.nan) or np.nan)
            feat["momentum_6m_raw"]  = float(frow.get("momentum_6m",     np.nan) or np.nan)
            feat["momentum_12m_raw"] = float(frow.get("momentum_12m",    np.nan) or np.nan)

        # ── Fundamental features ─────────────────────────────────────────
        fund = fundamentals_df[fundamentals_df["ticker"] == ticker] if not fundamentals_df.empty else pd.DataFrame()
        if not fund.empty:
            fund = fund.iloc[0]
            feat["pe_ratio"]         = float(fund.get("pe_ratio",         np.nan) or np.nan)
            feat["pb_ratio"]         = float(fund.get("pb_ratio",         np.nan) or np.nan)
            feat["roe"]              = float(fund.get("roe",              np.nan) or np.nan)
            feat["roce"]             = float(fund.get("roce",             np.nan) or np.nan)
            feat["debt_to_equity"]   = float(fund.get("debt_to_equity",   np.nan) or np.nan)
            feat["revenue_growth"]   = float(fund.get("revenue_growth",   np.nan) or np.nan)
            feat["profit_growth"]    = float(fund.get("profit_growth",    np.nan) or np.nan)
            feat["operating_margin"] = float(fund.get("operating_margin", np.nan) or np.nan)
            feat["promoter_holding"] = float(fund.get("promoter_holding", np.nan) or np.nan)

        # ── Event Sentiment features ─────────────────────────────────────
        if sentiment_df is not None and not sentiment_df.empty:
            sent_row = sentiment_df[sentiment_df["ticker"] == ticker]
            if not sent_row.empty:
                feat["sentiment_score"] = float(sent_row.iloc[0].get("avg_sentiment", 0))
                feat["event_count"]     = float(sent_row.iloc[0].get("event_count", 0))
            else:
                feat["sentiment_score"] = 0.0
                feat["event_count"]     = 0.0

        features[ticker] = feat

    df = pd.DataFrame.from_dict(features, orient="index")
    df.index.name = "ticker"
    return df.reset_index()


def prepare_training_labels(
    tickers: List[str],
    returns_matrix: pd.DataFrame,
    benchmark_returns: pd.Series,
    forward_period: int = 21,
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.02,
) -> pd.Series:
    """
    Create BUY/HOLD/SELL labels based on each ticker's forward excess return
    relative to the cross-sectional median on that date.

    This generates one label per ticker based on their most recent forward
    return rank — ensuring all 3 classes are always represented as long as
    we have at least 3 tickers, regardless of market direction.
    """
    bench_fwd = benchmark_returns.rolling(forward_period).sum().shift(-forward_period)

    # Build per-ticker forward excess returns
    fwd_excess = {}
    for ticker in tickers:
        if ticker not in returns_matrix.columns:
            continue
        ticker_fwd = returns_matrix[ticker].rolling(forward_period).sum().shift(-forward_period)
        excess = (ticker_fwd - bench_fwd).dropna()
        if not excess.empty:
            fwd_excess[ticker] = excess

    if not fwd_excess:
        return pd.Series(dtype=int)

    # Use the last available forward period for each ticker
    last_excess = pd.Series({t: s.iloc[-1] for t, s in fwd_excess.items()})

    # Rank-based labelling: top third → BUY, bottom third → SELL, middle → HOLD
    # This guarantees all 3 classes appear as long as len(tickers) >= 3
    n = len(last_excess)
    ranked = last_excess.rank(pct=True)   # 0..1

    labels = {}
    for ticker, pct_rank in ranked.items():
        raw = float(last_excess[ticker])
        if pct_rank >= 0.60:
            labels[ticker] = 2   # BUY  — top 40%
        elif pct_rank <= 0.35:
            labels[ticker] = 0   # SELL — bottom 35%
        else:
            labels[ticker] = 1   # HOLD — middle

    # Fallback: if still missing a class (e.g. only 5 tickers),
    # re-assign using absolute thresholds
    present = set(labels.values())
    if len(present) < 3:
        for ticker in labels:
            raw = float(last_excess[ticker])
            if raw > buy_threshold:
                labels[ticker] = 2
            elif raw < sell_threshold:
                labels[ticker] = 0
            else:
                labels[ticker] = 1

    return pd.Series(labels, name="label")


# ─── Model Training ──────────────────────────────────────────────────────────

FEATURE_COLS = [
    "ann_return", "volatility_63d", "volatility_21d", "skewness", "kurtosis",
    "ret_1m", "ret_3m", "ret_6m", "ret_12m",
    "momentum_score", "quality_score", "value_score", "low_vol_score",
    "composite_score", "beta", "momentum_6m_raw", "momentum_12m_raw",
    "pe_ratio", "pb_ratio", "roe", "roce", "debt_to_equity",
    "revenue_growth", "profit_growth", "operating_margin", "promoter_holding",
    "sentiment_score", "event_count",
]


def _get_feature_array(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select feature columns and return a clean float64 matrix.
    Uses clean_numeric_series() on every column so 500-ticker universes
    with mixed dtypes (Int64, object, nullable float) never cause cast errors.
    """
    available = [c for c in FEATURE_COLS if c in features_df.columns]
    X = features_df[available].copy()
    for col in X.columns:
        val = X[col]
        # Squeeze single-column DataFrames (yfinance 300+ ticker issue)
        if isinstance(val, pd.DataFrame):
            val = val.squeeze(axis=1)
        X[col] = clean_numeric_series(val)
    return X, available


def train_models(
    features_df: pd.DataFrame,
    labels: pd.Series,
) -> Dict:
    """
    Train XGBoost, LightGBM, and Random Forest classifiers.
    Returns a dict of {model_name: fitted_model}.
    """
    # Merge features and labels
    df = features_df.set_index("ticker").copy()
    df["label"] = labels.reindex(df.index)
    df.dropna(subset=["label"], inplace=True)

    if len(df) < 5:
        logger.warning("Insufficient data to train ML models.")
        return {}

    X, feat_cols = _get_feature_array(df.reset_index())
    y = df["label"].astype(int).values

    models = {
        "xgboost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=ML_RANDOM_STATE,
            verbosity=0,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            num_leaves=31,
            random_state=ML_RANDOM_STATE,
            verbosity=-1,
            force_row_wise=True,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=2,
            random_state=ML_RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    trained = {}
    for name, model in models.items():
        try:
            model.fit(X, y)
            trained[name] = {"model": model, "feature_cols": feat_cols}
            logger.info(f"  ✅ Trained {name}")
        except Exception as e:
            logger.error(f"  ❌ {name} training failed: {e}")

    # Save models
    for name, m in trained.items():
        path = Path(MODELS_DIR) / f"{name}.pkl"
        joblib.dump(m, str(path))
        logger.debug(f"  Saved {name} → {path}")

    return trained


def load_saved_models() -> Dict:
    """Load previously trained models from disk."""
    models = {}
    for name in ["xgboost", "lightgbm", "random_forest"]:
        path = Path(MODELS_DIR) / f"{name}.pkl"
        if path.exists():
            models[name] = joblib.load(str(path))
            logger.debug(f"  Loaded {name} from {path}")
    return models


# ─── Prediction & Signals ────────────────────────────────────────────────────

SIGNAL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


def _pad_to_3_classes(probs: np.ndarray, model) -> np.ndarray:
    """
    Guarantee the probability matrix has exactly 3 columns: [SELL, HOLD, BUY].

    When a model is trained on data that only contained 2 of the 3 classes
    (e.g. only HOLD and BUY because no stock had a SELL label), predict_proba
    returns only 2 columns.  We use the model's classes_ attribute to map each
    column to the right class index and fill missing classes with 0.
    """
    n = probs.shape[0]
    full = np.zeros((n, 3))

    classes = list(getattr(model, "classes_", range(probs.shape[1])))
    for col_idx, cls in enumerate(classes):
        try:
            target = int(cls)
            if 0 <= target <= 2 and col_idx < probs.shape[1]:
                full[:, target] = probs[:, col_idx]
        except (ValueError, TypeError):
            pass

    # Re-normalise rows to sum to 1 (guard against rounding)
    row_sums = full.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return full / row_sums


def generate_signals(
    features_df: pd.DataFrame,
    models: Dict,
    ensemble: bool = True,
) -> pd.DataFrame:
    """
    Run all trained models, optionally ensemble via voting,
    and return a signal DataFrame.
    """
    if not models:
        logger.warning("No trained models available for signal generation.")
        return pd.DataFrame()

    X, feat_cols = _get_feature_array(features_df)
    tickers = features_df["ticker"].tolist()

    all_probs: Dict[str, np.ndarray] = {}

    for name, m in models.items():
        model = m["model"]
        cols  = m["feature_cols"]
        x = X[cols].copy()
        for c in cols:
            if c not in x.columns:
                x[c] = 0
        try:
            probs = model.predict_proba(x[cols])
            all_probs[name] = _pad_to_3_classes(probs, model)
        except Exception as e:
            logger.warning(f"  Prediction failed for {name}: {e}")

    if not all_probs:
        return pd.DataFrame()

    # Ensemble: average probabilities across models
    if ensemble and len(all_probs) > 1:
        avg_probs = np.mean(list(all_probs.values()), axis=0)
        model_name = "ensemble"
    else:
        first_key = list(all_probs.keys())[0]
        avg_probs = all_probs[first_key]
        model_name = first_key

    predictions = np.argmax(avg_probs, axis=1)
    confidence  = np.max(avg_probs, axis=1)

    signals_list = []
    for i, ticker in enumerate(tickers):
        sell_p, hold_p, buy_p = (
            float(avg_probs[i, 0]),
            float(avg_probs[i, 1]),
            float(avg_probs[i, 2]) if avg_probs.shape[1] > 2 else 0.0,
        )
        signal_label = SIGNAL_MAP[int(predictions[i])]

        # Override with threshold logic
        if buy_p  >= ML_SIGNAL_THRESHOLD["buy"]:
            signal_label = "BUY"
        elif sell_p >= (1 - ML_SIGNAL_THRESHOLD["sell"]):
            signal_label = "SELL"

        signals_list.append({
            "ticker":            ticker,
            "signal_date":       date.today(),
            "signal":            signal_label,
            "model_name":        model_name,
            "buy_probability":   buy_p,
            "hold_probability":  hold_p,
            "sell_probability":  sell_p,
            "confidence":        float(confidence[i]),
        })

    return pd.DataFrame(signals_list)


# ─── SHAP Explainability ─────────────────────────────────────────────────────

def compute_shap_values(
    features_df: pd.DataFrame,
    models: Dict,
    model_name: str = "xgboost",
) -> Optional[pd.DataFrame]:
    """
    Compute SHAP values for the given model and feature set.
    Returns a DataFrame of SHAP values (rows=tickers, cols=features).
    """
    if model_name not in models:
        logger.warning(f"Model {model_name} not found for SHAP.")
        return None

    m = models[model_name]
    model = m["model"]
    cols  = m["feature_cols"]

    X, _ = _get_feature_array(features_df)
    X = X[cols]

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)

        # shap_vals can come back in 3 different shapes depending on
        # shap version and model type:
        #   A) list of arrays  → one array per class, each shape (n, features)
        #   B) 3-D array       → shape (n_samples, n_features, n_classes)
        #   C) 2-D array       → shape (n_samples, n_features)  [binary/regression]

        if isinstance(shap_vals, list):
            # Case A: pick BUY class (index 2) or last available
            idx = 2 if len(shap_vals) > 2 else len(shap_vals) - 1
            shap_array = shap_vals[idx]

        elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
            # Case B: 3-D array — slice out BUY class (last axis index 2)
            cls_idx = 2 if shap_vals.shape[2] > 2 else shap_vals.shape[2] - 1
            shap_array = shap_vals[:, :, cls_idx]

        else:
            # Case C: already 2-D
            shap_array = shap_vals

        shap_df = pd.DataFrame(shap_array, columns=cols,
                               index=features_df["ticker"].values)
        shap_df.index.name = "ticker"
        return shap_df

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None


def build_signal_explanations(
    signals_df: pd.DataFrame,
    shap_df: Optional[pd.DataFrame],
    n_drivers: int = 5,
) -> pd.DataFrame:
    """
    Annotate signals with top positive and negative SHAP drivers.
    """
    if shap_df is None or shap_df.empty:
        signals_df["top_positive_drivers"] = None
        signals_df["top_negative_drivers"] = None
        signals_df["shap_values"] = None
        return signals_df

    positive_drivers = []
    negative_drivers = []
    shap_jsons       = []

    for _, row in signals_df.iterrows():
        ticker = row["ticker"]
        if ticker not in shap_df.index:
            positive_drivers.append(None)
            negative_drivers.append(None)
            shap_jsons.append(None)
            continue

        shap_row = shap_df.loc[ticker].sort_values(ascending=False)

        pos = shap_row[shap_row > 0].head(n_drivers)
        neg = shap_row[shap_row < 0].tail(n_drivers)

        positive_drivers.append({k: round(v, 4) for k, v in pos.items()})
        negative_drivers.append({k: round(v, 4) for k, v in neg.items()})
        shap_jsons.append({k: round(v, 4) for k, v in shap_row.items()})

    signals_df["top_positive_drivers"] = positive_drivers
    signals_df["top_negative_drivers"] = negative_drivers
    signals_df["shap_values"]          = shap_jsons
    return signals_df


# ─── Persistence ─────────────────────────────────────────────────────────────

def upsert_signals(signals_df: pd.DataFrame, engine=None) -> int:
    if engine is None:
        engine = get_db_engine()
    if signals_df.empty:
        return 0

    def _f(v):
        if v is None: return None
        try:
            f = float(v); return None if f != f else f
        except (TypeError, ValueError): return None

    records = []
    for _, row in signals_df.iterrows():
        records.append({
            "ticker":               row.get("ticker"),
            "signal_date":          row.get("signal_date", date.today()),
            "signal":               row.get("signal"),
            "model_name":           row.get("model_name"),
            "buy_probability":      _f(row.get("buy_probability")),
            "hold_probability":     _f(row.get("hold_probability")),
            "sell_probability":     _f(row.get("sell_probability")),
            "confidence":           _f(row.get("confidence")),
            "shap_values":          json.dumps(row.get("shap_values") or {}),
            "top_positive_drivers": json.dumps(row.get("top_positive_drivers") or {}),
            "top_negative_drivers": json.dumps(row.get("top_negative_drivers") or {}),
        })

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO signals
                    (ticker, signal_date, signal, model_name,
                     buy_probability, hold_probability, sell_probability, confidence,
                     shap_values, top_positive_drivers, top_negative_drivers)
                VALUES
                    (:ticker, :signal_date, :signal, :model_name,
                     :buy_probability, :hold_probability, :sell_probability, :confidence,
                     CAST(:shap_values AS jsonb), CAST(:top_positive_drivers AS jsonb),
                     CAST(:top_negative_drivers AS jsonb))
                ON CONFLICT (ticker, signal_date, model_name) DO UPDATE SET
                    signal            = EXCLUDED.signal,
                    buy_probability   = EXCLUDED.buy_probability,
                    hold_probability  = EXCLUDED.hold_probability,
                    sell_probability  = EXCLUDED.sell_probability,
                    confidence        = EXCLUDED.confidence,
                    shap_values       = EXCLUDED.shap_values,
                    top_positive_drivers = EXCLUDED.top_positive_drivers,
                    top_negative_drivers = EXCLUDED.top_negative_drivers
            """),
            records,
        )

    logger.info(f"✅ Upserted {len(records)} signal rows")
    return len(records)


def load_signals(tickers: List[str], engine=None) -> pd.DataFrame:
    if engine is None:
        engine = get_db_engine()
    placeholders = ", ".join([f"'{t}'" for t in tickers])
    query = f"""
        SELECT DISTINCT ON (ticker)
               ticker, signal_date, signal, model_name,
               buy_probability, hold_probability, sell_probability,
               confidence, top_positive_drivers, top_negative_drivers
        FROM   signals
        WHERE  ticker IN ({placeholders})
        ORDER  BY ticker, signal_date DESC
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


# ─── Full ML Pipeline ────────────────────────────────────────────────────────

def run_ml_pipeline(
    tickers: List[str],
    close_matrix: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    factor_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    benchmark_returns: pd.Series,
    sentiment_df: Optional[pd.DataFrame] = None,
    engine=None,
    retrain: bool = True,
) -> pd.DataFrame:
    """
    Full ML pipeline:
      1. Feature engineering
      2. Train or load models
      3. Generate signals
      4. SHAP explainability
      5. Persist to DB
      6. Return signals DataFrame
    """
    if engine is None:
        engine = get_db_engine()

    logger.info("Building ML feature matrix…")
    features_df = build_feature_matrix(
        tickers, close_matrix, returns_matrix,
        factor_df, fundamentals_df, sentiment_df,
    )

    if retrain:
        logger.info("Training ML models…")
        labels = prepare_training_labels(tickers, returns_matrix, benchmark_returns)
        models = train_models(features_df, labels)
    else:
        logger.info("Loading saved ML models…")
        models = load_saved_models()
        if not models:
            logger.info("No saved models found, training fresh…")
            labels = prepare_training_labels(tickers, returns_matrix, benchmark_returns)
            models = train_models(features_df, labels)

    logger.info("Generating signals…")
    signals_df = generate_signals(features_df, models)

    logger.info("Computing SHAP values…")
    shap_df = compute_shap_values(features_df, models, model_name="xgboost")
    signals_df = build_signal_explanations(signals_df, shap_df)

    if not signals_df.empty:
        upsert_signals(signals_df, engine)

    logger.info("✅ ML pipeline complete.")
    return signals_df, shap_df