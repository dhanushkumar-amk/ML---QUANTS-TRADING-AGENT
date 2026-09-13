# ============================================================
# Unit Tests: Microstructure Proxies (Phase 15)
# ============================================================
"""
Unit tests for Corwin-Schultz spread, Roll spread, VPIN proxy,
Garman-Klass, and Parkinson volatility estimators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.microstructure_proxies import (
    MicrostructureProxyFeatureExtractor,
    compute_corwin_schultz_spread,
    compute_garman_klass_volatility,
    compute_parkinson_volatility,
    compute_roll_spread,
    compute_vpin_proxy,
)

# ============================================================
# 1. Corwin-Schultz (2012) High-Low Spread
# ============================================================


def test_corwin_schultz_spread_analytical():
    """Verify Corwin-Schultz spread calculation against manual analytical formula."""
    # Day 0: H=102, L=98
    # Day 1: H=104, L=99
    df = pd.DataFrame(
        {
            "high": [102.0, 104.0],
            "low": [98.0, 99.0],
        }
    )

    log_hl_0 = np.log(102.0 / 98.0)
    log_hl_1 = np.log(104.0 / 99.0)
    beta_manual = log_hl_0**2 + log_hl_1**2

    high_2d = 104.0
    low_2d = 98.0
    gamma_manual = np.log(high_2d / low_2d) ** 2

    denom = 3.0 - 2.0 * np.sqrt(2.0)
    alpha_manual = ((np.sqrt(2.0) - 1.0) * np.sqrt(beta_manual)) / denom - np.sqrt(
        gamma_manual / denom
    )
    spread_manual = 2.0 * (np.exp(alpha_manual) - 1.0) / (1.0 + np.exp(alpha_manual))
    expected_spread = max(0.0, spread_manual)

    daily, _ = compute_corwin_schultz_spread(df, window=2, clamp_negative=True)
    assert np.isclose(daily.iloc[1], expected_spread, rtol=1e-5)
    assert daily.iloc[1] >= 0.0


# ============================================================
# 2. Roll (1984) Serial Covariance Spread
# ============================================================


def test_roll_spread_bid_ask_bounce():
    """Verify Roll spread on pure alternating bid-ask bounce series."""
    # Synthetic alternating sequence bouncing between 99 and 101 (spread = 2.0)
    n = 60
    prices = [101.0 if i % 2 == 0 else 99.0 for i in range(n)]
    df = pd.DataFrame({"close": prices})

    s_dollar, s_pct = compute_roll_spread(df, window=20)

    # Valid values after warmup
    valid_dollar = s_dollar.dropna()
    assert len(valid_dollar) > 0
    # In pure alternating +/-2 series:
    # dp = [+2, -2, +2, -2...], dp_lag = [-2, +2, -2, +2...]
    # dp * dp_lag = -4.0, mean(dp) ~ 0 -> Cov ~ -4.0
    # s = 2 * sqrt(4.0) = 4.0
    assert np.isclose(valid_dollar.iloc[-1], 4.0, atol=0.5)
    # Fractional spread should be ~4.0 / 100 = 0.04 (4%)
    assert np.isclose(s_pct.iloc[-1], 0.04, atol=0.01)


# ============================================================
# 3. VPIN Proxy (Bulk Volume Classification)
# ============================================================


def test_vpin_proxy_extreme_cases():
    """Verify VPIN bounds: 1.0 for 100% one-sided flow, 0.0 for balanced flow."""
    n = 30

    # 1. Pure one-sided buying: Close == High on all bars
    one_sided_df = pd.DataFrame(
        {
            "high": [105.0] * n,
            "low": [95.0] * n,
            "close": [105.0] * n,  # Close at High -> buy_fraction = 1.0
            "volume": [1000.0] * n,
        }
    )
    vpin_one_sided = compute_vpin_proxy(one_sided_df, window=10)
    assert np.isclose(vpin_one_sided.iloc[-1], 1.0)

    # 2. Perfectly balanced two-way flow: Close == (High + Low) / 2
    balanced_df = pd.DataFrame(
        {
            "high": [105.0] * n,
            "low": [95.0] * n,
            "close": [100.0] * n,  # Close at Midpoint -> buy_fraction = 0.5
            "volume": [1000.0] * n,
        }
    )
    vpin_balanced = compute_vpin_proxy(balanced_df, window=10)
    assert np.isclose(vpin_balanced.iloc[-1], 0.0)


# ============================================================
# 4. Garman-Klass & Parkinson Volatility Estimators
# ============================================================


def test_garman_klass_and_parkinson_analytical():
    """Verify Garman-Klass and Parkinson match analytical candle formula."""
    # Single candle repeated: O=100, H=105, L=95, C=100
    n = 25
    df = pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [105.0] * n,
            "low": [95.0] * n,
            "close": [100.0] * n,
        }
    )

    log_hl = np.log(105.0 / 95.0)
    log_co = np.log(100.0 / 100.0)  # 0.0

    # Manual Parkinson variance: log(H/L)^2 / (4 * ln(2))
    var_p_manual = (log_hl**2) / (4.0 * np.log(2.0))
    vol_p_manual_ann = np.sqrt(var_p_manual) * np.sqrt(252.0)

    # Manual Garman-Klass variance: 0.5 * log(H/L)^2 - (2*ln(2) - 1) * log(C/O)^2
    var_gk_manual = 0.5 * (log_hl**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)
    vol_gk_manual_ann = np.sqrt(var_gk_manual) * np.sqrt(252.0)

    vol_p = compute_parkinson_volatility(df, window=10, annualized=True)
    vol_gk = compute_garman_klass_volatility(df, window=10, annualized=True)

    assert np.isclose(vol_p.iloc[-1], vol_p_manual_ann, rtol=1e-5)
    assert np.isclose(vol_gk.iloc[-1], vol_gk_manual_ann, rtol=1e-5)


# ============================================================
# 5. Extractor & Validation
# ============================================================


def test_microstructure_proxy_feature_extractor():
    """Verify MicrostructureProxyFeatureExtractor produces expected columns."""
    n = 50
    rng = np.random.RandomState(42)
    prices = 100.0 + np.cumsum(rng.normal(0, 1, n))
    opens = prices + rng.normal(0, 0.2, n)
    highs = np.maximum(prices, opens) + np.abs(rng.normal(1, 0.2, n))
    lows = np.minimum(prices, opens) - np.abs(rng.normal(1, 0.2, n))
    vols = rng.uniform(50000, 200000, n)

    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": prices, "volume": vols})

    extractor = MicrostructureProxyFeatureExtractor(spread_window=10, vpin_window=10, vol_window=10)
    out = extractor.transform(df, append=True)

    expected_cols = [
        "corwin_schultz_spread_10",
        "roll_spread_10",
        "vpin_proxy_10",
        "garman_klass_vol_10",
        "parkinson_vol_10",
    ]
    for col in expected_cols:
        assert col in out.columns
        assert len(out[col].dropna()) > 0


def test_invalid_prices_raise():
    """Negative or zero prices must raise ValueError."""
    bad_df = pd.DataFrame({"high": [-10.0, 20.0], "low": [5.0, 10.0]})
    with pytest.raises(ValueError, match="strictly positive"):
        compute_corwin_schultz_spread(bad_df)
