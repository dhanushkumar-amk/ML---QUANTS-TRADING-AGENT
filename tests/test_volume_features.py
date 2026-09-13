# ============================================================
# Unit Tests: Volume Features (Phase 15)
# ============================================================
"""
Unit tests for On-Balance Volume (OBV), VWAP, ADL, CMF, Volume ROC,
Volume Z-Score, and Amihud Illiquidity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.volume_features import (
    VolumeFeatureExtractor,
    compute_adl,
    compute_amihud_illiquidity,
    compute_cmf,
    compute_obv,
    compute_volume_roc,
    compute_volume_zscore,
    compute_vwap,
)

# ============================================================
# 1. On-Balance Volume (OBV)
# ============================================================


def test_compute_obv_deterministic():
    """Verify OBV on deterministic up/down/flat price sequences."""
    # Prices: 10 -> 12 (up) -> 11 (down) -> 11 (flat) -> 14 (up)
    # Volume: 100, 200, 150, 80, 300
    df = pd.DataFrame(
        {
            "close": [10.0, 12.0, 11.0, 11.0, 14.0],
            "volume": [100.0, 200.0, 150.0, 80.0, 300.0],
        }
    )

    obv = compute_obv(df)

    # Expected:
    # bar 0: 100
    # bar 1: 100 + 200 = 300
    # bar 2: 300 - 150 = 150
    # bar 3: 150 (flat price, zero delta)
    # bar 4: 150 + 300 = 450
    expected = [100.0, 300.0, 150.0, 150.0, 450.0]
    np.testing.assert_allclose(obv.values, expected, rtol=1e-6)


# ============================================================
# 2. Volume-Weighted Average Price (VWAP)
# ============================================================


def test_compute_vwap_rolling():
    """Verify rolling VWAP matches hand-calculated sum(PV) / sum(V)."""
    # bar 0: H=12, L=10, C=11 -> Typ=11, V=100 -> PV=1100
    # bar 1: H=14, L=12, C=13 -> Typ=13, V=200 -> PV=2600
    # bar 2: H=16, L=14, C=15 -> Typ=15, V=300 -> PV=4500
    df = pd.DataFrame(
        {
            "high": [12.0, 14.0, 16.0],
            "low": [10.0, 12.0, 14.0],
            "close": [11.0, 13.0, 15.0],
            "volume": [100.0, 200.0, 300.0],
        }
    )

    # Window 2
    # bar 0: NaN (warmup)
    # bar 1: (1100 + 2600) / (100 + 200) = 3700 / 300 = 12.333333
    # bar 2: (2600 + 4500) / (200 + 300) = 7100 / 500 = 14.2
    vwap = compute_vwap(df, window=2, price_type="typical")

    assert np.isnan(vwap.iloc[0])
    assert np.isclose(vwap.iloc[1], 3700.0 / 300.0)
    assert np.isclose(vwap.iloc[2], 7100.0 / 500.0)


# ============================================================
# 3. Accumulation/Distribution Line (ADL) & CMF
# ============================================================


def test_compute_adl_and_cmf():
    """Verify ADL and CMF on extreme candlestick formations."""
    # Bar 0: Close at High -> CLV = +1, V=100 -> MFV = +100
    # Bar 1: Close at Low -> CLV = -1, V=200 -> MFV = -200
    # Bar 2: Close at Midpoint -> CLV = 0, V=300 -> MFV = 0
    df = pd.DataFrame(
        {
            "high": [10.0, 20.0, 30.0],
            "low": [6.0, 10.0, 20.0],
            "close": [10.0, 10.0, 25.0],
            "volume": [100.0, 200.0, 300.0],
        }
    )

    adl = compute_adl(df)
    # Expected ADL:
    # bar 0: +100
    # bar 1: 100 - 200 = -100
    # bar 2: -100 + 0 = -100
    expected_adl = [100.0, -100.0, -100.0]
    np.testing.assert_allclose(adl.values, expected_adl, rtol=1e-6)

    # Rolling CMF over window 2:
    # bar 0: NaN
    # bar 1: (+100 - 200) / (100 + 200) = -100 / 300 = -0.333333
    # bar 2: (-200 + 0) / (200 + 300) = -200 / 500 = -0.4
    cmf = compute_cmf(df, window=2)
    assert np.isnan(cmf.iloc[0])
    assert np.isclose(cmf.iloc[1], -1.0 / 3.0)
    assert np.isclose(cmf.iloc[2], -0.4)


# ============================================================
# 4. Volume Rate of Change & Volume Z-Score
# ============================================================


def test_compute_volume_roc_and_zscore():
    """Verify Volume ROC and Volume Z-Score calculation."""
    volumes = [100.0, 150.0, 200.0, 250.0]
    df = pd.DataFrame({"volume": volumes})

    # Window 2 ROC:
    # bar 2: (200 - 100) / 100 = +1.0
    # bar 3: (250 - 150) / 150 = +0.666667
    vroc = compute_volume_roc(df, window=2)
    assert np.isnan(vroc.iloc[0])
    assert np.isnan(vroc.iloc[1])
    assert np.isclose(vroc.iloc[2], 1.0)
    assert np.isclose(vroc.iloc[3], 100.0 / 150.0)

    # Volume Z-score
    zscore = compute_volume_zscore(df, window=3)
    # Window of [100, 150, 200] -> mean=150, std=50
    # bar 2: (200 - 150) / 50 = +1.0
    assert np.isclose(zscore.iloc[2], 1.0)


# ============================================================
# 5. Amihud Illiquidity Ratio
# ============================================================


def test_compute_amihud_illiquidity():
    """Verify Amihud illiquidity proxy correctly reflects price impact per dollar volume."""
    df = pd.DataFrame(
        {
            "close": [100.0, 102.0, 100.0],  # ret: NaN, +0.02, -0.0196
            "volume": [1000.0, 5000.0, 10000.0],
        }
    )

    daily, rolling = compute_amihud_illiquidity(df, window=2, scale=1e6)

    # bar 1: |ret| = 0.02, dollar_vol = 102 * 5000 = 510,000
    # daily = (0.02 / 510000) * 1e6 approx 0.0392157
    expected_bar1 = (0.02 / 510000.0) * 1e6
    assert np.isclose(daily.iloc[1], expected_bar1, rtol=1e-5)

    # Rolling window 2: bar 0 is NaN, bar 1 is NaN, bar 2 is mean(bar1, bar2)
    assert np.isnan(rolling.iloc[0])
    assert np.isnan(rolling.iloc[1])
    assert not np.isnan(rolling.iloc[2])
    assert rolling.iloc[2] > 0


# ============================================================
# 6. Extractor & Error Handling
# ============================================================


def test_volume_feature_extractor():
    """Verify VolumeFeatureExtractor generates all canonical columns."""
    n = 40
    rng = np.random.RandomState(42)
    prices = 100.0 + np.cumsum(rng.normal(0, 1, n))
    highs = prices + np.abs(rng.normal(1, 0.2, n))
    lows = prices - np.abs(rng.normal(1, 0.2, n))
    vols = rng.uniform(50000, 200000, n)

    df = pd.DataFrame({"open": prices, "high": highs, "low": lows, "close": prices, "volume": vols})

    extractor = VolumeFeatureExtractor(
        vwap_window=10, cmf_window=10, roc_window=5, zscore_window=10, amihud_window=10
    )
    res = extractor.transform(df, append=True)

    expected_cols = [
        "obv",
        "vwap_10",
        "adl",
        "cmf_10",
        "volume_roc_5",
        "volume_zscore_10",
        "amihud_illiquidity_10",
    ]
    for col in expected_cols:
        assert col in res.columns
        assert len(res[col].dropna()) > 0


def test_missing_columns_raise():
    """Missing required columns must raise ValueError."""
    bad_df = pd.DataFrame({"price": [10, 11, 12]})
    with pytest.raises(ValueError, match="DataFrame must contain"):
        compute_obv(bad_df)
