# ============================================================
# Unit Tests: Final Production Feature Set (Phase 34)
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.feature_registry import feature_registry
from src.features.final_feature_set import (
    EXCLUDED_FEATURES_RATIONALE,
    FINAL_PRODUCTION_FEATURES,
    PRODUCTION_QUANT_FEATURES,
    PRODUCTION_SENTIMENT_FEATURES,
    build_production_feature_dataset,
    train_and_save_production_model,
    update_registry_with_production_status,
)


def _generate_synthetic_ohlcv(n_bars: int = 250, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="B", tz="UTC")
    returns = rng.normal(0.0005, 0.015, n_bars)
    prices = 150.0 * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {
            "open": prices * (1 + rng.normal(0, 0.002, n_bars)),
            "high": prices * (1 + np.abs(rng.normal(0, 0.005, n_bars))),
            "low": prices * (1 - np.abs(rng.normal(0, 0.005, n_bars))),
            "close": prices,
            "volume": rng.uniform(1e6, 5e6, n_bars),
        },
        index=dates,
    )


def test_feature_registry_production_vs_excluded():
    """Verify feature registry correctly segregates production vs excluded features with reasons."""
    update_registry_with_production_status()

    prod_feats = feature_registry.list_production_features()
    excluded_map = feature_registry.list_excluded_features()

    # All final production features must be registered as production
    for f in FINAL_PRODUCTION_FEATURES:
        assert f in prod_feats
        meta = feature_registry.get(f)
        assert meta.status == "production"

    # All excluded features must have explicit, non-empty reasons documented
    for f, reason in EXCLUDED_FEATURES_RATIONALE.items():
        assert f in excluded_map
        assert len(excluded_map[f]) > 20
        meta = feature_registry.get(f)
        assert meta.status == "excluded"
        assert meta.exclusion_reason == reason


def test_final_production_feature_counts():
    """Verify production feature count conforms to the strict 11-feature specification."""
    assert len(PRODUCTION_QUANT_FEATURES) == 9
    assert len(PRODUCTION_SENTIMENT_FEATURES) == 2
    assert len(FINAL_PRODUCTION_FEATURES) == 11
    assert "news_headline_volume" in FINAL_PRODUCTION_FEATURES
    assert "news_sentiment_volatility" in FINAL_PRODUCTION_FEATURES
    assert "news_mean_sentiment" not in FINAL_PRODUCTION_FEATURES


def test_build_production_feature_dataset():
    """Verify build_production_feature_dataset constructs clean, non-null 11-feature matrix."""
    df_ohlcv = _generate_synthetic_ohlcv(200)
    feats, target, prices = build_production_feature_dataset(df_ohlcv, ticker="AAPL")

    assert list(feats.columns) == FINAL_PRODUCTION_FEATURES
    assert not feats.isna().any().any()
    assert len(feats) == len(target)
    assert len(feats) == len(prices)
    assert target.isin([0, 1]).all()
    assert len(feats) > 150  # Valid observations after warmup drop


def test_train_and_save_production_model(tmp_path: Path):
    """Verify production model training and versioned artifact persistence."""
    df_ohlcv = _generate_synthetic_ohlcv(200)

    manifest = train_and_save_production_model(
        ticker="AAPL",
        ohlcv_df=df_ohlcv,
        version="v1.0_test",
        artifact_dir=tmp_path,
        n_splits=3,
        embargo_bars=2,
        random_state=42,
    )

    assert manifest["version"] == "v1.0_test"
    assert manifest["ticker"] == "AAPL"
    assert manifest["feature_count"] == 11
    assert "accuracy" in manifest["validation_metrics"]
    assert "sharpe_ratio" in manifest["validation_metrics"]

    # Verify physical file existence
    dest_dir = tmp_path / "production_model_v1.0_test"
    model_file = dest_dir / manifest["model_file"]
    meta_file = dest_dir / "aapl_xgboost_v1.0_test.meta.json"

    assert model_file.exists()
    assert meta_file.exists()

    with open(meta_file, encoding="utf-8") as f:
        meta_loaded = json.load(f)
    assert meta_loaded["model_name"] == "AAPL_production_xgboost"
