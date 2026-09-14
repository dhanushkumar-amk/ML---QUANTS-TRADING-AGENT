# ============================================================
# Final Production Feature Set & Model Packaging (Phase 34)
# ============================================================
"""
Assembly, registration, and artifact persistence of the final production feature set.

=============================================================================
THEORETICAL FOUNDATION: PRODUCTION FEATURE HYGIENE & SELECTION AUDITING
-----------------------------------------------------------------------------
In production quantitative trading, feature selection is NOT about accumulating the
maximum number of sophisticated variables. Every additional feature carries:
1. Estimation Risk: Extra degrees of freedom inflate out-of-sample prediction variance.
2. Multicollinearity: Correlated features destabilize tree split selection and linear weights.
3. Operational Fragility: Features derived from complex NLP pipelines introduce latency,
   external API dependency, and missing-data edge cases.

Based on the rigorous empirical findings of Phase 33:
- Excluded Features: Raw directional sentiment (`news_mean_sentiment`,
  `earnings_overall_sentiment`, `earnings_prepared_sentiment`, `earnings_qa_sentiment`,
  `earnings_qa_vs_prepared_delta`) failed the Diebold-Mariano test (p > 0.40) and their
  bootstrap Sharpe confidence intervals spanned zero. In large-cap equities (AAPL, MSFT, SPY),
  public news is priced in milliseconds, leaving delayed daily sentiment as noise.
- Retained Features: Core quantitative momentum, volatility, volume, and oscillator signals
  plus **News Attention Shocks** (`news_headline_volume`, `news_sentiment_volatility`).
  News volume and disagreement proxy attention jumps and volatility shocks without
  attempting to forecast direction from stale headlines.
=============================================================================
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.feature_registry import FeatureRegistry, feature_registry
from src.models.financial_metrics import compute_financial_metrics
from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.walk_forward import WalkForwardSplitter
from src.nlp.sentiment_scorer import FinBERTSentimentScorer, compute_daily_sentiment_series
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# 1. Canonical Feature Definitions
# ============================================================

PRODUCTION_QUANT_FEATURES: list[str] = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "natr_14",
    "volume_ratio_20",
]

PRODUCTION_SENTIMENT_FEATURES: list[str] = [
    "news_headline_volume",
    "news_sentiment_volatility",
]

FINAL_PRODUCTION_FEATURES: list[str] = PRODUCTION_QUANT_FEATURES + PRODUCTION_SENTIMENT_FEATURES

EXCLUDED_FEATURES_RATIONALE: dict[str, str] = {
    "news_mean_sentiment": (
        "Phase 33 Walk-Forward DM test p=0.58; zero incremental Sharpe in mega-caps "
        "where headline news is priced in milliseconds by HFT participants."
    ),
    "earnings_overall_sentiment": (
        "Quarterly publication is too sparse to produce continuous alpha; bootstrap "
        "Sharpe 95% CI spanned zero across all tested walk-forward folds."
    ),
    "earnings_prepared_sentiment": (
        "Scripted and sanitized by IR and corporate legal counsel; exhibits zero "
        "correlation with post-earnings announcement drift."
    ),
    "earnings_qa_sentiment": (
        "High variance across single quarters; redundant with realized volatility "
        "jumps around earnings dates."
    ),
    "earnings_qa_vs_prepared_delta": (
        "Sparse single-stock signal that fails to generalize to index ETFs (SPY) "
        "and degrades cross-asset model stability."
    ),
    "earnings_linguistic_uncertainty": (
        "Highly collinear with realized volatility (vol_20d) and ATR (VIF > 8.0); "
        "adds redundant variance without incremental information."
    ),
}


# ============================================================
# 2. Registry Status Synchronization
# ============================================================


def update_registry_with_production_status(
    registry: FeatureRegistry | None = None,
) -> pd.DataFrame:
    """Synchronize the global feature registry with production and excluded statuses.

    Parameters
    ----------
    registry : FeatureRegistry, optional
        Target registry. Defaults to global `feature_registry`.

    Returns
    -------
    audit_df : pd.DataFrame
        Table documenting all features, their status, and exclusion rationales.
    """
    reg = registry or feature_registry

    # Register production quantitative features if missing
    quant_specs = [
        ("ret_1d", "momentum", "1-day simple return", 1),
        ("ret_5d", "momentum", "5-day cumulative return", 5),
        ("ret_20d", "momentum", "20-day cumulative return", 20),
        ("vol_20d", "volatility", "20-day realized volatility", 20),
        ("rsi_14", "momentum", "14-day Relative Strength Index", 14),
        ("macd_line", "momentum", "MACD 12/26 EMA line", 26),
        ("macd_signal", "momentum", "MACD 9 EMA signal line", 9),
        ("natr_14", "volatility", "14-day Normalized ATR (% of close)", 14),
        ("volume_ratio_20", "volume", "Ratio of volume to 20-day MA volume", 20),
    ]
    for name, cat, desc, lookback in quant_specs:
        if not reg.contains(name):
            reg.register(
                name=name,
                category=cat,
                description=desc,
                lookback_horizon=lookback,
                status="production",
            )
        else:
            reg.mark_production(name)

    # Register production sentiment features
    sent_specs = [
        (
            "news_headline_volume",
            "nlp_sentiment",
            "Daily headline publication volume (log1p attention shock)",
            1,
        ),
        (
            "news_sentiment_volatility",
            "nlp_sentiment",
            "Daily headline sentiment standard deviation (disagreement)",
            1,
        ),
    ]
    for name, cat, desc, lookback in sent_specs:
        if not reg.contains(name):
            reg.register(
                name=name,
                category=cat,
                description=desc,
                lookback_horizon=lookback,
                status="production",
            )
        else:
            reg.mark_production(name)

    # Register excluded features with explicit reasons
    for name, reason in EXCLUDED_FEATURES_RATIONALE.items():
        if not reg.contains(name):
            reg.register(
                name=name,
                category="nlp_sentiment",
                description="Excluded experimental sentiment feature",
                lookback_horizon=1,
                status="excluded",
                exclusion_reason=reason,
            )
        else:
            reg.mark_excluded(name, reason)

    logger.info(
        "Registry updated: %d production features, %d excluded features.",
        len(reg.list_production_features()),
        len(reg.list_excluded_features()),
    )
    return reg.to_dataframe()


# ============================================================
# 3. Production Feature Dataset Builder
# ============================================================


def build_production_feature_dataset(
    ohlcv_df: pd.DataFrame,
    news_df: pd.DataFrame | None = None,
    ticker: str = "AAPL",
    scorer: FinBERTSentimentScorer | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Construct strictly vetted production feature matrix and 1-day direction target.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        OHLCV market data dataframe.
    news_df : pd.DataFrame, optional
        Historical headline news records.
    ticker : str, default 'AAPL'
    scorer : FinBERTSentimentScorer, optional

    Returns
    -------
    features_df : pd.DataFrame
        Matrix containing exactly `FINAL_PRODUCTION_FEATURES` (11 columns).
    target : pd.Series
        Binary forward direction target (1 if next day return > 0 else 0).
    prices : pd.Series
        Closing prices aligned with features index.
    """
    df = ohlcv_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.set_index("date")
        else:
            df.index = pd.to_datetime(df.index, utc=True)

    df = df.sort_index()

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    feats = pd.DataFrame(index=df.index)

    # 1. Quantitative Core Features
    feats["ret_1d"] = close.pct_change(1)
    feats["ret_5d"] = close.pct_change(5)
    feats["ret_20d"] = close.pct_change(20)
    feats["vol_20d"] = feats["ret_1d"].rolling(window=20).std()

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    feats["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD Line & Signal
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_l = ema12 - ema26
    feats["macd_line"] = macd_l
    feats["macd_signal"] = macd_l.ewm(span=9, adjust=False).mean()

    # NATR 14
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    feats["natr_14"] = (tr.rolling(14).mean() / close) * 100.0

    # Volume Expansion Ratio
    vol_ma20 = volume.rolling(20).mean()
    feats["volume_ratio_20"] = volume / (vol_ma20 + 1e-9)

    # 2. Vetted Sentiment Features (Volume & Volatility Shocks Only)
    if news_df is not None and not news_df.empty:
        if scorer is None:
            scorer = FinBERTSentimentScorer()
        daily_news = compute_daily_sentiment_series(ticker=ticker, df_news=news_df, scorer=scorer)
        if not daily_news.empty:
            daily_news["date"] = pd.to_datetime(daily_news["date"], utc=True)
            daily_news = daily_news.set_index("date").reindex(df.index)
            feats["news_headline_volume"] = np.log1p(daily_news["headline_volume"].fillna(0.0))
            feats["news_sentiment_volatility"] = daily_news["sentiment_volatility"].fillna(0.0)
        else:
            feats["news_headline_volume"] = 0.0
            feats["news_sentiment_volatility"] = 0.0
    else:
        feats["news_headline_volume"] = 0.0
        feats["news_sentiment_volatility"] = 0.0

    # Strict target definition: 1 if close_{t+1} > close_t else 0
    fwd_return = (close.shift(-1) - close) / close
    target = (fwd_return > 0).astype(int)

    # Filter warmup periods
    valid_mask = ~feats.isna().any(axis=1) & ~fwd_return.isna()
    clean_feats = feats.loc[valid_mask, FINAL_PRODUCTION_FEATURES].copy()
    clean_target = target.loc[valid_mask].copy()
    clean_prices = close.loc[valid_mask].copy()

    return clean_feats, clean_target, clean_prices


# ============================================================
# 4. Production Model Retraining & Versioned Artifact Packaging
# ============================================================


def train_and_save_production_model(
    ticker: str,
    ohlcv_df: pd.DataFrame,
    news_df: pd.DataFrame | None = None,
    version: str = "v1.0",
    artifact_dir: str | Path = "models/artifacts",
    n_splits: int = 5,
    embargo_bars: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train, validate, and persist the official production trading model.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    ohlcv_df : pd.DataFrame
        Historical OHLCV data.
    news_df : pd.DataFrame, optional
        Headline news data.
    version : str, default 'v1.0'
        Model version string.
    artifact_dir : str or Path, default 'models/artifacts'
        Artifacts directory.
    n_splits : int, default 5
    embargo_bars : int, default 5
    random_state : int, default 42

    Returns
    -------
    manifest : dict[str, Any]
        Production deployment metadata and out-of-fold validation metrics.
    """
    t = ticker.upper()
    logger.info("Building production dataset for %s...", t)
    X, y, prices = build_production_feature_dataset(ohlcv_df=ohlcv_df, news_df=news_df, ticker=t)

    n_samples = len(X)
    min_train = max(100, int(n_samples * 0.4))
    splitter = WalkForwardSplitter(
        n_splits=n_splits,
        embargo_bars=embargo_bars,
        min_train_size=min_train,
        window_type="expanding",
    )

    # 1. Walk-Forward Cross-Validation
    oof_preds = np.zeros(n_samples, dtype=float)
    oof_probs = np.zeros(n_samples, dtype=float)
    tested_mask = np.zeros(n_samples, dtype=bool)

    for _fold_idx, (tr_idx, te_idx) in enumerate(splitter.split(X, y), start=1):
        fold_model = GradientBoostingModel(
            backend="xgboost",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            random_state=random_state,
        )
        fold_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = fold_model.predict(X.iloc[te_idx])
        prob = fold_model.predict_proba(X.iloc[te_idx])
        if hasattr(prob, "ndim") and prob.ndim == 2:
            prob = prob[:, 1]

        oof_preds[te_idx] = pred
        oof_probs[te_idx] = prob
        tested_mask[te_idx] = True

    eval_idx = np.where(tested_mask)[0]
    y_test = y.iloc[eval_idx].values
    preds_test = oof_preds[eval_idx]
    probs_test = oof_probs[eval_idx]

    accuracy = float(np.mean(preds_test == y_test))
    brier_score = float(np.mean((probs_test - y_test) ** 2))

    # Strategy Returns
    p_eval = prices.iloc[eval_idx].values
    asset_rets = np.diff(p_eval) / p_eval[:-1]
    signals = (probs_test[:-1] >= 0.5).astype(float) * 2.0 - 1.0
    strategy_rets = signals * asset_rets
    fin_metrics = compute_financial_metrics(pd.Series(strategy_rets))

    # 2. Final Retraining on Full Dataset
    prod_model = GradientBoostingModel(
        backend="xgboost",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        random_state=random_state,
    )
    prod_model.fit(X, y)

    # 3. Artifact Persistence
    dest_dir = Path(artifact_dir) / f"production_model_{version}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    model_file = dest_dir / f"{t.lower()}_xgboost_{version}.json"
    meta_file = dest_dir / f"{t.lower()}_xgboost_{version}.meta.json"

    # Save booster model binary
    prod_model.model_.save_model(str(model_file))

    manifest: dict[str, Any] = {
        "model_name": f"{t}_production_xgboost",
        "version": version,
        "ticker": t,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_type": "GradientBoostingModel (XGBoost)",
        "features": FINAL_PRODUCTION_FEATURES,
        "feature_count": len(FINAL_PRODUCTION_FEATURES),
        "production_quant_features": PRODUCTION_QUANT_FEATURES,
        "production_sentiment_features": PRODUCTION_SENTIMENT_FEATURES,
        "excluded_features": EXCLUDED_FEATURES_RATIONALE,
        "validation_protocol": f"{n_splits}-fold walk-forward with {embargo_bars}-bar embargo",
        "validation_metrics": {
            "evaluated_bars": len(eval_idx),
            "accuracy": round(accuracy, 4),
            "brier_score": round(brier_score, 4),
            "sharpe_ratio": round(fin_metrics.get("sharpe_ratio", 0.0), 4),
            "cumulative_return": round(fin_metrics.get("cumulative_return", 0.0), 4),
            "max_drawdown": round(fin_metrics.get("max_drawdown", 0.0), 4),
            "win_rate": round(fin_metrics.get("win_rate", 0.0), 4),
        },
        "model_file": str(model_file.name),
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.03,
            "random_state": random_state,
        },
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Saved production model %s to %s", version, dest_dir)
    return manifest
