# ============================================================
# Unit Tests — Mean-Reversion Features (Phase 13)
# ============================================================
"""
Tests for mean-reversion features, Ornstein-Uhlenbeck half-life estimation,
Bollinger Bands, price Z-scores, RSI reversion framing, ATR-normalized MA distance,
and Stochastic Oscillators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.mean_reversion_features import (
    MeanReversionFeatureExtractor,
    compute_atr,
    compute_bollinger_bands,
    compute_ma_distance,
    compute_price_zscore,
    compute_rolling_half_life,
    compute_rsi_reversion,
    compute_stochastic_oscillator,
    compute_true_range,
    estimate_half_life,
)

# ============================================================
# 1. Price Z-Score Tests
# ============================================================


def test_price_zscore_hand_calculated():
    """Verify rolling Z-score formula on hand-computed sample."""
    # Prices: 10, 12, 14
    # Window = 3
    # Mean = (10 + 12 + 14) / 3 = 12.0
    # Variance (ddof=1) = ((10-12)^2 + (12-12)^2 + (14-12)^2) / 2 = 8 / 2 = 4.0
    # Std = 2.0
    # Z-score at idx 2: (14 - 12) / 2.0 = 1.0
    # Next price: 6
    # Prices in window [12, 14, 6]:
    # Mean = 32 / 3 = 10.666667
    # Std = sqrt(((12 - 10.667)^2 + (14 - 10.667)^2 + (6 - 10.667)^2) / 2) = sqrt(34.6667 / 2) = sqrt(17.3333) = 4.16333
    # Z-score at idx 3: (6 - 10.6667) / 4.16333 = -1.1209
    prices = [10.0, 12.0, 14.0, 6.0]
    df = pd.DataFrame({"close": prices})

    res = compute_price_zscore(df, windows=[3])
    assert "zscore_3d" in res.columns

    assert np.isnan(res["zscore_3d"].iloc[0])
    assert np.isnan(res["zscore_3d"].iloc[1])
    assert np.isclose(res["zscore_3d"].iloc[2], 1.0)

    expected_z3 = (6.0 - (32.0 / 3.0)) / np.std([12.0, 14.0, 6.0], ddof=1)
    assert np.isclose(res["zscore_3d"].iloc[3], expected_z3)


def test_price_zscore_zero_std_handling():
    """Flat prices must yield z-score of 0.0 rather than NaN/Inf."""
    df = pd.DataFrame({"close": [100.0, 100.0, 100.0, 100.0]})
    res = compute_price_zscore(df, windows=[3])
    assert np.isclose(res["zscore_3d"].iloc[2], 0.0)
    assert np.isclose(res["zscore_3d"].iloc[3], 0.0)


def test_price_zscore_invalid_window():
    df = pd.DataFrame({"close": [10.0, 20.0]})
    with pytest.raises(ValueError, match="Z-score lookback window must be >= 2"):
        compute_price_zscore(df, windows=[1])


# ============================================================
# 2. Bollinger Bands (%B and Bandwidth) Tests
# ============================================================


def test_bollinger_bands_hand_calculated():
    """Verify Bollinger upper, lower, %B, and bandwidth formulas."""
    # Prices: 10, 12, 14. Window=3, num_std=2.0
    # Mean = 12.0, Std = 2.0
    # Upper = 12 + 2*2 = 16.0
    # Lower = 12 - 2*2 = 8.0
    # %B = (14 - 8) / (16 - 8) = 6 / 8 = 0.75
    # Bandwidth = (16 - 8) / 12 = 8 / 12 = 0.666667
    prices = [10.0, 12.0, 14.0, 18.0, 6.0]
    df = pd.DataFrame({"close": prices})

    res = compute_bollinger_bands(df, window=3, num_std=2.0)

    middle_col = "bb_middle_3"
    upper_col = "bb_upper_3_2"
    lower_col = "bb_lower_3_2"
    pct_b_col = "bb_pct_b_3_2"
    bw_col = "bb_bandwidth_3_2"

    assert np.isclose(res[middle_col].iloc[2], 12.0)
    assert np.isclose(res[upper_col].iloc[2], 16.0)
    assert np.isclose(res[lower_col].iloc[2], 8.0)
    assert np.isclose(res[pct_b_col].iloc[2], 0.75)
    assert np.isclose(res[bw_col].iloc[2], 8.0 / 12.0)

    # With window=3 and num_std=1.0:
    # Upper = 12 + 1*2 = 14.0, Lower = 12 - 1*2 = 10.0
    res_k1 = compute_bollinger_bands(df, window=3, num_std=1.0)
    pct_b_k1 = "bb_pct_b_3_1"
    # At idx 2: price is 14.0 => %B = (14 - 10) / (14 - 10) = 1.0
    assert np.isclose(res_k1[pct_b_k1].iloc[2], 1.0)
    # At idx 3: price is 18.0 => price is strictly above upper band => %B > 1.0
    assert res_k1[pct_b_k1].iloc[3] > 1.0
    # At idx 4: price is 6.0 => price is strictly below lower band => %B < 0.0
    assert res_k1[pct_b_k1].iloc[4] < 0.0


def test_bollinger_bands_invalid_params():
    df = pd.DataFrame({"close": [10.0, 20.0]})
    with pytest.raises(ValueError, match="Bollinger window must be >= 2"):
        compute_bollinger_bands(df, window=1)

    with pytest.raises(ValueError, match="num_std must be positive"):
        compute_bollinger_bands(df, window=20, num_std=0.0)


# ============================================================
# 3. RSI Reversion Framing Tests
# ============================================================


def test_rsi_reversion_signals():
    """Verify discrete reversion signals (+1 oversold bounce, -1 overbought pullback)."""
    # 25 bars strongly down => RSI < 30 => signal = +1.0
    # 25 bars strongly up => RSI > 70 => signal = -1.0
    down_prices = list(range(130, 100, -1))
    df_down = pd.DataFrame({"close": down_prices})
    res_down = compute_rsi_reversion(df_down, window=14, oversold=30.0, overbought=70.0)

    sig_col = "rsi_reversion_signal_14"
    stretch_col = "rsi_stretch_14"

    # Deep oversold condition triggers +1.0 (mean-reversion bounce opportunity)
    assert res_down[sig_col].iloc[-1] == 1.0
    assert res_down[stretch_col].iloc[-1] < -0.40  # (RSI - 50) / 50 < -0.4

    # Deep overbought triggers -1.0 (mean-reversion pullback opportunity)
    up_prices = list(range(100, 135))
    df_up = pd.DataFrame({"close": up_prices})
    res_up = compute_rsi_reversion(df_up, window=14, oversold=30.0, overbought=70.0)
    assert res_up[sig_col].iloc[-1] == -1.0
    assert res_up[stretch_col].iloc[-1] > 0.40


# ============================================================
# 4. ATR and Normalized MA Distance Tests
# ============================================================


def test_compute_true_range_and_atr():
    """Verify True Range and ATR calculations against known sequence."""
    # Day 0: H=12, L=10, C=11 -> TR = 2
    # Day 1: H=15, L=11, C=14 -> TR = max(15-11=4, |15-11|=4, |11-11|=0) = 4
    # Day 2: H=13, L=9, C=10  -> TR = max(13-9=4, |13-14|=1, |9-14|=5) = 5
    df = pd.DataFrame(
        {
            "high": [12.0, 15.0, 13.0],
            "low": [10.0, 11.0, 9.0],
            "close": [11.0, 14.0, 10.0],
        }
    )
    tr = compute_true_range(df)
    assert np.isclose(tr.iloc[0], 2.0)
    assert np.isclose(tr.iloc[1], 4.0)
    assert np.isclose(tr.iloc[2], 5.0)

    # ATR with window=3: initial avg = (2 + 4 + 5) / 3 = 11 / 3 = 3.666667
    atr = compute_atr(df, window=3)
    assert np.isnan(atr.iloc[0])
    assert np.isnan(atr.iloc[1])
    assert np.isclose(atr.iloc[2], 11.0 / 3.0)


def test_compute_ma_distance():
    """Verify ATR-normalized and std-normalized distance from moving average."""
    n = 30
    prices = np.linspace(100, 130, n)
    df = pd.DataFrame(
        {
            "close": prices,
            "high": prices + 1.0,
            "low": prices - 1.0,
        }
    )

    res_atr = compute_ma_distance(df, windows=[10], normalize_by="atr", atr_window=10)
    assert "ma_dist_atr_10" in res_atr.columns
    assert res_atr["ma_dist_atr_10"].dropna().iloc[-1] > 0.0

    res_std = compute_ma_distance(df, windows=[10], normalize_by="std")
    assert "ma_dist_std_10" in res_std.columns
    assert res_std["ma_dist_std_10"].dropna().iloc[-1] > 0.0


# ============================================================
# 5. Ornstein-Uhlenbeck Half-Life Estimation Tests
# ============================================================


def test_half_life_synthetic_ornstein_uhlenbeck():
    """Test half-life estimator on synthetic OU process with KNOWN parameters.

    Model: dx_t = theta * (mu - x_t) * dt + sigma * dW_t
    With:
      theta = 0.05
      mu = 0.0
      sigma = 0.02
      dt = 1.0
    Theoretical Half-Life:
      t_{1/2} = ln(2) / theta = ln(2) / 0.05 = 13.8629 trading days.

    We generate a synthetic realization of length N = 25,000 and verify that
    our estimator recovers t_{1/2} within a tight tolerance (e.g. 13.86 +/- 1.0).
    """
    np.random.seed(42)
    n_steps = 25000
    theta_true = 0.05
    mu_true = 0.0
    sigma_true = 0.02
    dt = 1.0

    expected_half_life = np.log(2.0) / theta_true  # approx 13.8629

    x = np.zeros(n_steps)
    for t in range(1, n_steps):
        dx = theta_true * (mu_true - x[t - 1]) * dt + sigma_true * np.sqrt(dt) * np.random.randn()
        x[t] = x[t - 1] + dx

    estimated_hl = estimate_half_life(x)

    # Must be within 1.0 period of theoretical value (approx +/- 7%)
    assert (
        abs(estimated_hl - expected_half_life) < 1.0
    ), f"Expected half-life {expected_half_life:.2f}, got {estimated_hl:.2f}"


def test_half_life_random_walk_returns_inf():
    """An explosive series or high-persistence random walk must return infinite or very high half-life."""
    # 1. Deterministic explosive process (x_t = 1.05 * x_{t-1}) => beta = +0.05 > 0 => inf
    explosive = 10.0 * (1.05 ** np.arange(100))
    hl_exp = estimate_half_life(explosive)
    assert np.isinf(hl_exp)

    # 2. Random walk with positive drift yields very large half-life (> 200 bars)
    np.random.seed(123)
    random_walk = 100.0 + np.cumsum(np.random.normal(0.001, 0.01, 1000))
    hl_rw = estimate_half_life(random_walk)
    assert np.isinf(hl_rw) or hl_rw > 200.0


def test_rolling_half_life_on_dataframe():
    """Verify rolling half-life calculation on DataFrame with price spread."""
    np.random.seed(99)
    n = 200
    # Mean-reverting synthetic price oscillation around 100
    t = np.arange(n)
    oscillation = 100.0 + 5.0 * np.sin(t / 5.0) + np.random.normal(0, 0.5, n)
    df = pd.DataFrame({"close": oscillation})

    res = compute_rolling_half_life(df, window=60, ma_window=10)
    col = "half_life_60d"
    assert col in res.columns
    assert np.isnan(res[col].iloc[10])
    valid_hl = res[col].dropna()
    assert len(valid_hl) > 0
    # Estimated half-life for this oscillation should be positive and finite
    assert (valid_hl > 0).all()
    assert (valid_hl <= 60.0).all()


# ============================================================
# 6. Stochastic Oscillator Tests
# ============================================================


def test_stochastic_oscillator_hand_calculated():
    """Verify Fast %K and %D formulas against manual calculations."""
    # Prices over 3 periods:
    # t=0: H=12, L=8, C=10
    # t=1: H=15, L=9, C=14
    # t=2: H=14, L=7, C=8
    # Window = 3
    # At t=2:
    # Lowest Low = min(8, 9, 7) = 7.0
    # Highest High = max(12, 15, 14) = 15.0
    # Fast %K = (8 - 7) / (15 - 7) * 100 = 1 / 8 * 100 = 12.5%
    df = pd.DataFrame(
        {
            "high": [12.0, 15.0, 14.0],
            "low": [8.0, 9.0, 7.0],
            "close": [10.0, 14.0, 8.0],
        }
    )

    res = compute_stochastic_oscillator(df, k_window=3, d_window=2, slow_k_window=2)
    assert "stoch_k_3" in res.columns
    assert np.isnan(res["stoch_k_3"].iloc[1])
    assert np.isclose(res["stoch_k_3"].iloc[2], 12.5)


# ============================================================
# 7. Unified MeanReversionFeatureExtractor Tests
# ============================================================


def test_mean_reversion_feature_extractor():
    """Verify unified extractor runs and appends all expected columns."""
    n = 200
    prices = 100.0 + np.cumsum(np.random.RandomState(42).randn(n))
    df = pd.DataFrame(
        {
            "close": prices,
            "high": prices + 1.0,
            "low": prices - 1.0,
        }
    )

    extractor = MeanReversionFeatureExtractor(half_life_window=60)
    out = extractor.transform(df, append=True)

    expected_cols = [
        "close",
        "zscore_10d",
        "zscore_20d",
        "zscore_50d",
        "bb_middle_20",
        "bb_upper_20_2",
        "bb_lower_20_2",
        "bb_pct_b_20_2",
        "bb_bandwidth_20_2",
        "rsi_reversion_signal_14",
        "rsi_stretch_14",
        "ma_dist_atr_20",
        "ma_dist_atr_50",
        "stoch_k_14",
        "stoch_d_14_3",
        "stoch_slow_k",
        "stoch_slow_d",
        "stoch_reversion_signal",
        "half_life_60d",
    ]
    for col in expected_cols:
        assert col in out.columns, f"Expected column '{col}' missing from extractor output."

    # Test append=False returns only feature columns
    feats_only = extractor(df, append=False)
    assert "close" not in feats_only.columns
    assert "zscore_20d" in feats_only.columns
