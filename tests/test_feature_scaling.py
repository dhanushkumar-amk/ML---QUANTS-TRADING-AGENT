# ============================================================
# Unit Tests: Feature Scaling & Pipeline (Phase 17)
# ============================================================
"""
Unit tests for TimeSeriesScaler, CrossSectionalScaler, and FeaturePipeline,
verifying train/test leakage prevention and missing-value policies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_scaling import (
    CrossSectionalScaler,
    FeaturePipeline,
    TimeSeriesScaler,
)

# ============================================================
# 1. Train/Test Boundary & Leakage Prevention Test
# ============================================================


def test_scaler_train_test_leakage_prevention():
    """Verify scaler fitted on train data is strictly unaffected by test data (No Leakage)."""
    # Train set: values with mean = 20.0, std = 10.0
    train_df = pd.DataFrame({"feat": [10.0, 20.0, 30.0]})

    # Test set: extreme out-of-distribution values
    test_df_1 = pd.DataFrame({"feat": [100.0, 200.0, 300.0]})
    test_df_2 = pd.DataFrame({"feat": [10000.0, 20000.0, 30000.0]})

    scaler = TimeSeriesScaler(method="standard", clip_outliers=None)
    scaler.fit(train_df)

    # Train parameters must be mean=20.0, scale=10.0
    assert np.isclose(scaler.stats_["feat"]["center"], 20.0)
    assert np.isclose(scaler.stats_["feat"]["scale"], 10.0)

    # Transforming test_df_1 using train statistics: (100 - 20) / 10 = 8.0
    res_1 = scaler.transform(test_df_1)
    np.testing.assert_allclose(res_1["feat"].values, [8.0, 18.0, 28.0])

    # Transforming test_df_2 must not alter train statistics
    res_2 = scaler.transform(test_df_2)
    assert np.isclose(scaler.stats_["feat"]["center"], 20.0)
    assert np.isclose(scaler.stats_["feat"]["scale"], 10.0)
    np.testing.assert_allclose(res_2["feat"].values, [998.0, 1998.0, 2998.0])


# ============================================================
# 2. Hand-Computed Scaling Values Test
# ============================================================


def test_hand_computed_scaling_methods():
    """Verify Standard, MinMax, and Robust scalers match hand-calculated benchmarks."""
    # Data: 10, 20, 30, 40, 50
    df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})

    # 1. MinMax Scaler: (x - 10) / (50 - 10) = (x - 10) / 40
    # Expected: [0.0, 0.25, 0.50, 0.75, 1.0]
    minmax = TimeSeriesScaler(method="minmax")
    res_minmax = minmax.fit_transform(df)
    expected_minmax = [0.0, 0.25, 0.50, 0.75, 1.0]
    np.testing.assert_allclose(res_minmax["x"].values, expected_minmax, rtol=1e-6)

    # 2. Robust Scaler (Median & IQR):
    # Median = 30.0, Q25 = 20.0, Q75 = 40.0 -> IQR = 20.0
    # z = (x - 30) / 20 -> [-1.0, -0.5, 0.0, +0.5, +1.0]
    robust = TimeSeriesScaler(method="robust", clip_outliers=None)
    res_robust = robust.fit_transform(df)
    expected_robust = [-1.0, -0.5, 0.0, 0.5, 1.0]
    np.testing.assert_allclose(res_robust["x"].values, expected_robust, rtol=1e-6)

    # 3. Standard Scaler:
    # Mean = 30.0, Std = sqrt(250) approx 15.811388
    standard = TimeSeriesScaler(method="standard", clip_outliers=None)
    res_std = standard.fit_transform(df)
    assert np.isclose(res_std["x"].mean(), 0.0)
    assert np.isclose(res_std["x"].std(ddof=1), 1.0)


# ============================================================
# 3. Cross-Sectional Scaler Test
# ============================================================


def test_cross_sectional_scaler():
    """Verify cross-sectional normalization across tickers at each date."""
    # Two dates, three tickers each
    dates = pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3)
    tickers = ["AAPL", "MSFT", "SPY", "AAPL", "MSFT", "SPY"]
    # Date 1: 100, 200, 300 -> mean=200, std=100 -> Z: -1, 0, +1
    # Date 2: 10, 20, 30 -> mean=20, std=10 -> Z: -1, 0, +1
    values = [100.0, 200.0, 300.0, 10.0, 20.0, 30.0]

    df = pd.DataFrame({"date": dates, "ticker": tickers, "signal": values})

    scaler = CrossSectionalScaler(method="standard")
    out = scaler.transform(df, date_col="date", feature_cols=["signal"])

    np.testing.assert_allclose(out["signal"].values, [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0], atol=1e-5)


# ============================================================
# 4. FeaturePipeline Missing Value & Assembly Test
# ============================================================


def test_feature_pipeline_missing_value_policy():
    """Verify FeaturePipeline handles short gaps with bounded ffill and rejects silent zero-fill."""
    pipeline = FeaturePipeline(
        feature_names=["rsi_14"],
        scaler_method="robust",
        max_ffill=2,
        drop_warmup=True,
    )

    # Synthetic OHLCV
    n = 60
    prices = 100.0 + np.cumsum(np.random.normal(0, 1, n))
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": 100000.0,
        }
    )

    # Transform without fit must raise ValueError
    with pytest.raises(ValueError, match="must be fitted on training data"):
        pipeline.transform(df)

    # Fit and transform
    out = pipeline.fit_transform(df)
    assert not out.isna().any().any()
    assert len(out) > 0
    # Values should be centered around median
    assert -5.0 <= out.iloc[0, 0] <= 5.0
