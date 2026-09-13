# ============================================================
# Unit Tests — Momentum Features & Base Framework (Phase 12)
# ============================================================
"""
Tests for momentum features, FeatureBase interface, and FeatureRegistry.
Includes hand-calculated synthetic benchmarks, formula verifications,
and edge cases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.delisted_tickers import DelistedRegistry
from src.data_pipeline.universe_builder import UniverseBuilder
from src.features.feature_registry import FeatureRegistry, register_feature
from src.features.momentum_features import (
    MomentumFeatureExtractor,
    compute_cross_sectional_momentum,
    compute_jegadeesh_titman_momentum,
    compute_ma_crossover,
    compute_macd,
    compute_price_momentum,
    compute_rate_of_change,
    compute_rsi,
)

# ============================================================
# 1. Simple Price Momentum & 12-1 Month Tests
# ============================================================


def test_compute_price_momentum_hand_calculated():
    """Verify simple price momentum against hand-calculated values."""
    # Prices: 100, 105, 110, 120, 114
    prices = [100.0, 105.0, 110.0, 120.0, 114.0]
    df = pd.DataFrame({"close": prices})

    res = compute_price_momentum(df, windows=[1, 2, 4])

    assert "mom_1d" in res.columns
    assert "mom_2d" in res.columns
    assert "mom_4d" in res.columns

    # 1-day momentum
    # Index 1: (105 - 100) / 100 = 0.05
    # Index 3: (120 - 110) / 110 = 0.090909...
    assert np.isnan(res["mom_1d"].iloc[0])
    assert np.isclose(res["mom_1d"].iloc[1], 0.05)
    assert np.isclose(res["mom_1d"].iloc[3], 10.0 / 110.0)

    # 2-day momentum
    # Index 2: (110 - 100) / 100 = 0.10
    # Index 4: (114 - 110) / 110 = 0.0363636...
    assert np.isnan(res["mom_2d"].iloc[1])
    assert np.isclose(res["mom_2d"].iloc[2], 0.10)
    assert np.isclose(res["mom_2d"].iloc[4], (114.0 - 110.0) / 110.0)

    # 4-day momentum
    # Index 4: (114 - 100) / 100 = 0.14
    assert np.isnan(res["mom_4d"].iloc[3])
    assert np.isclose(res["mom_4d"].iloc[4], 0.14)


def test_compute_price_momentum_invalid_window():
    """Window < 1 must raise ValueError."""
    df = pd.DataFrame({"close": [10.0, 20.0]})
    with pytest.raises(ValueError, match="Lookback window must be >= 1"):
        compute_price_momentum(df, windows=[0])


def test_jegadeesh_titman_momentum_hand_calculated():
    """Verify 12-1 month momentum correctly skips the recent month."""
    # Synthetic series of length 10
    # total_window = 5, skip_window = 2
    # At t=5: p_lag_skip = p[5-2]=p[3], p_lag_total = p[5-5]=p[0]
    # return = (p[3] - p[0]) / p[0]
    prices = [100.0, 102.0, 105.0, 110.0, 115.0, 120.0, 130.0]
    df = pd.DataFrame({"close": prices})

    res = compute_jegadeesh_titman_momentum(df, total_window=5, skip_window=2)
    col = "mom_skip2_5d"
    assert col in res.columns

    # For t < 5, should be NaN
    for i in range(5):
        assert np.isnan(res[col].iloc[i])

    # At t=5 (price=120): skip=2 is index 3 (110), total=5 is index 0 (100)
    # expected: (110 - 100) / 100 = 0.10
    assert np.isclose(res[col].iloc[5], (110.0 - 100.0) / 100.0)

    # At t=6 (price=130): skip=2 is index 4 (115), total=5 is index 1 (102)
    # expected: (115 - 102) / 102
    assert np.isclose(res[col].iloc[6], (115.0 - 102.0) / 102.0)


def test_jegadeesh_titman_invalid_windows():
    df = pd.DataFrame({"close": [100.0, 110.0]})
    with pytest.raises(
        ValueError, match="total_window .* must be strictly greater than skip_window"
    ):
        compute_jegadeesh_titman_momentum(df, total_window=10, skip_window=10)


# ============================================================
# 2. Rate of Change (ROC) Tests
# ============================================================


def test_rate_of_change_hand_calculated():
    """Verify ROC percentage formula."""
    prices = [100.0, 102.0, 105.0, 110.0]
    df = pd.DataFrame({"close": prices})

    res = compute_rate_of_change(df, windows=[1, 3])
    assert "roc_1" in res.columns
    assert "roc_3" in res.columns

    # roc_1 at idx 1: (102 - 100) / 100 * 100 = 2.0%
    assert np.isclose(res["roc_1"].iloc[1], 2.0)

    # roc_3 at idx 3: (110 - 100) / 100 * 100 = 10.0%
    assert np.isclose(res["roc_3"].iloc[3], 10.0)


def test_rate_of_change_invalid_window():
    df = pd.DataFrame({"close": [100.0]})
    with pytest.raises(ValueError, match="ROC window must be >= 1"):
        compute_rate_of_change(df, windows=[0])


# ============================================================
# 3. Moving Average Crossover Tests
# ============================================================


def test_ma_crossover_hand_calculated():
    """Verify continuous spread, binary state, and golden/death cross events."""
    # Fast window = 2, slow window = 4
    # Prices:
    # 0: 10
    # 1: 10 -> fast_ma = 10.0, slow_ma = NaN
    # 2: 10 -> fast_ma = 10.0, slow_ma = NaN
    # 3: 10 -> fast_ma = 10.0, slow_ma = 10.0 -> spread = 0.0, bullish = 0.0
    # 4: 20 -> fast_ma = (10+20)/2 = 15.0, slow_ma = (10+10+10+20)/4 = 12.5 -> fast > slow (Golden Cross!)
    # 5: 5  -> fast_ma = (20+5)/2 = 12.5, slow_ma = (10+10+20+5)/4 = 11.25 -> fast > slow (still bullish)
    # 6: 2  -> fast_ma = (5+2)/2 = 3.5, slow_ma = (10+20+5+2)/4 = 9.25 -> fast < slow (Death Cross!)
    prices = [10.0, 10.0, 10.0, 10.0, 20.0, 5.0, 2.0]
    df = pd.DataFrame({"close": prices})

    res = compute_ma_crossover(df, fast_window=2, slow_window=4, ma_type="sma")

    spread_col = "sma_2_4_spread"
    bullish_col = "sma_2_4_bullish"
    cross_col = "sma_2_4_cross"

    assert np.isnan(res[spread_col].iloc[2])
    assert np.isclose(res[spread_col].iloc[3], 0.0)
    assert np.isclose(res[bullish_col].iloc[3], 0.0)

    # At idx 4: fast=15, slow=12.5 => spread = (15 - 12.5)/12.5 = 0.2
    assert np.isclose(res[spread_col].iloc[4], 0.20)
    assert res[bullish_col].iloc[4] == 1.0
    assert res[cross_col].iloc[4] == 1.0  # Golden Cross event!

    # At idx 5: fast=12.5, slow=11.25 => still bullish, no new cross
    assert res[bullish_col].iloc[5] == 1.0
    assert res[cross_col].iloc[5] == 0.0

    # At idx 6: fast=3.5, slow=9.25 => bearish, Death Cross event!
    assert res[bullish_col].iloc[6] == 0.0
    assert res[cross_col].iloc[6] == -1.0  # Death Cross event!


def test_ma_crossover_invalid_args():
    df = pd.DataFrame({"close": [10.0, 20.0]})
    with pytest.raises(ValueError, match="fast_window .* must be strictly less than slow_window"):
        compute_ma_crossover(df, fast_window=50, slow_window=50)

    with pytest.raises(ValueError, match="Unsupported ma_type"):
        compute_ma_crossover(df, fast_window=5, slow_window=10, ma_type="wma")


# ============================================================
# 4. Relative Strength Index (RSI) Tests
# ============================================================


def test_rsi_scratch_hand_calculated():
    """Verify RSI from scratch on a textbook price sequence.

    14-period series:
    If price strictly increases every period, gain > 0 and loss = 0 => RSI = 100.0.
    If price strictly decreases every period, gain = 0 and loss > 0 => RSI = 0.0.
    If price is constant, gains = losses = 0 => RSI = 50.0.
    """
    # 1. Strictly increasing series of 20 bars
    up_prices = list(range(100, 125))
    df_up = pd.DataFrame({"close": up_prices})
    res_up = compute_rsi(df_up, window=14)
    # After warmup (index >= 14), RSI should be 100.0
    assert np.isclose(res_up["rsi_14"].iloc[14], 100.0)
    assert np.isclose(res_up["rsi_14"].iloc[-1], 100.0)

    # 2. Strictly decreasing series of 20 bars
    down_prices = list(range(125, 100, -1))
    df_down = pd.DataFrame({"close": down_prices})
    res_down = compute_rsi(df_down, window=14)
    assert np.isclose(res_down["rsi_14"].iloc[14], 0.0)
    assert np.isclose(res_down["rsi_14"].iloc[-1], 0.0)

    # 3. Flat series
    flat_prices = [100.0] * 20
    df_flat = pd.DataFrame({"close": flat_prices})
    res_flat = compute_rsi(df_flat, window=14)
    assert np.isclose(res_flat["rsi_14"].iloc[14], 50.0)


def test_rsi_textbook_wilder_values():
    """Test RSI against standard manual calculation with alternating gains and losses.

    Window = 3:
    Prices:
      t=0: 100
      t=1: 102 (+2 gain, 0 loss)
      t=2: 101 (0 gain, 1 loss)
      t=3: 104 (+3 gain, 0 loss)
    Initial avg_gain = (2 + 0 + 3) / 3 = 5/3 = 1.666667
    Initial avg_loss = (0 + 1 + 0) / 3 = 1/3 = 0.333333
    RS = (5/3) / (1/3) = 5.0
    RSI = 100 - (100 / 6) = 100 * (5/6) = 83.333333%

    t=4: 102 (0 gain, 2 loss)
    Wilder update:
      avg_gain = (1.666667 * 2 + 0) / 3 = 3.333333 / 3 = 1.111111
      avg_loss = (0.333333 * 2 + 2) / 3 = 2.666667 / 3 = 0.888889
      RS = 1.111111 / 0.888889 = 1.25
      RSI = 100 - (100 / 2.25) = 55.555556%
    """
    prices = [100.0, 102.0, 101.0, 104.0, 102.0]
    df = pd.DataFrame({"close": prices})
    res = compute_rsi(df, window=3)

    assert np.isnan(res["rsi_3"].iloc[0])
    assert np.isnan(res["rsi_3"].iloc[1])
    assert np.isnan(res["rsi_3"].iloc[2])

    expected_rsi_3 = 100.0 * (5.0 / 6.0)
    assert np.isclose(res["rsi_3"].iloc[3], expected_rsi_3)

    expected_rsi_4 = 100.0 - (100.0 / 2.25)
    assert np.isclose(res["rsi_3"].iloc[4], expected_rsi_4)


def test_rsi_invalid_window():
    df = pd.DataFrame({"close": [10.0, 20.0]})
    with pytest.raises(ValueError, match="RSI window must be >= 1"):
        compute_rsi(df, window=0)


# ============================================================
# 5. MACD Tests
# ============================================================


def test_macd_hand_calculated():
    """Verify MACD line, signal line, and histogram components."""
    # Fast=3, Slow=6, Signal=2
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.randn(30))
    df = pd.DataFrame({"close": prices})

    res = compute_macd(df, fast_period=3, slow_period=6, signal_period=2)

    tag = "3_6_2"
    macd_line = res[f"macd_line_{tag}"]
    signal_line = res[f"macd_signal_{tag}"]
    hist = res[f"macd_hist_{tag}"]
    norm = res[f"macd_norm_{tag}"]
    bullish = res[f"macd_bullish_{tag}"]
    cross = res[f"macd_cross_{tag}"]

    # Check that hist is exactly macd_line - signal_line
    valid = hist.notna()
    np.testing.assert_allclose(hist[valid], (macd_line - signal_line)[valid])

    # Check normalized line: macd_line / close
    np.testing.assert_allclose(norm[valid], (macd_line / df["close"])[valid])

    # Check binary state: 1 when line > signal, 0 otherwise
    for i in np.where(valid)[0]:
        if macd_line.iloc[i] > signal_line.iloc[i]:
            assert bullish.iloc[i] == 1.0
        else:
            assert bullish.iloc[i] == 0.0

    # Cross flags must be in {-1.0, 0.0, 1.0}
    assert set(cross.dropna().unique()).issubset({-1.0, 0.0, 1.0})


def test_macd_invalid_periods():
    df = pd.DataFrame({"close": [10.0, 20.0]})
    with pytest.raises(ValueError, match="fast_period .* must be strictly less than slow_period"):
        compute_macd(df, fast_period=26, slow_period=12)


# ============================================================
# 6. Rank-Based Cross-Sectional Momentum Tests
# ============================================================


def test_cross_sectional_momentum_ranking():
    """Verify relative ranking across a synthetic universe of 4 assets."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    # Asset A: slow growth (+10%)
    # Asset B: huge winner (+50%)
    # Asset C: loser (-20%)
    # Asset D: moderate growth (+25%)
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 110.0],
            "B": [100.0, 105.0, 110.0, 120.0, 130.0, 135.0, 140.0, 145.0, 148.0, 150.0],
            "C": [100.0, 98.0, 95.0, 92.0, 90.0, 88.0, 85.0, 83.0, 82.0, 80.0],
            "D": [100.0, 102.0, 105.0, 108.0, 112.0, 115.0, 118.0, 120.0, 122.0, 125.0],
        },
        index=dates,
    )

    # Window=5, skip=1
    ranks = compute_cross_sectional_momentum(prices, window=5, skip_window=1)

    # Shape matches input
    assert ranks.shape == prices.shape

    # At the final date:
    # Asset returns over [t-5, t-1]:
    # Returns order: C (worst) < A < D < B (best)
    # Percentile ranks: C=0.25, A=0.50, D=0.75, B=1.00
    final_ranks = ranks.iloc[-1]
    assert final_ranks["C"] < final_ranks["A"] < final_ranks["D"] < final_ranks["B"]
    assert np.isclose(final_ranks["B"], 1.0)
    assert np.isclose(final_ranks["C"], 0.25)


def test_cross_sectional_momentum_with_universe_builder(tmp_path: Path):
    """Verify that UniverseBuilder filters inactive tickers at date t."""
    # Write a mini universe CSV
    const_file = tmp_path / "sp500_constituents.csv"
    changes_file = tmp_path / "sp500_historical_changes.csv"
    delisted_file = tmp_path / "delisted_tickers.csv"

    pd.DataFrame({"ticker": ["AAPL", "MSFT"], "name": ["Apple", "Microsoft"]}).to_csv(
        const_file, index=False
    )
    # Ticker OLD was removed on 2024-01-05
    pd.DataFrame(
        {
            "date": ["2024-01-05"],
            "ticker_added": ["AAPL"],
            "name_added": ["Apple"],
            "ticker_removed": ["OLD"],
            "name_removed": ["Old Corp"],
            "reason": ["Rebalanced"],
        }
    ).to_csv(changes_file, index=False)
    pd.DataFrame(columns=["ticker", "delisting_date", "reason", "notes"]).to_csv(
        delisted_file, index=False
    )

    ub = UniverseBuilder(
        constituents_path=const_file,
        changes_path=changes_file,
        delisted_registry=DelistedRegistry(data_path=delisted_file),
    )

    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    prices = pd.DataFrame(
        {
            "AAPL": np.linspace(100, 110, 8),
            "MSFT": np.linspace(200, 220, 8),
            "OLD": np.linspace(50, 60, 8),
        },
        index=dates,
    )

    ranks = compute_cross_sectional_momentum(
        prices,
        window=4,
        skip_window=1,
        universe_builder=ub,
    )

    # After 2024-01-05, OLD is no longer in universe => ranks for OLD must be NaN
    assert np.isnan(ranks.loc["2024-01-08", "OLD"])


# ============================================================
# 7. MomentumFeatureExtractor & FeatureBase Pattern Tests
# ============================================================


def test_momentum_feature_extractor():
    """Verify unified extractor runs and appends all expected columns."""
    n = 300
    prices = 100.0 + np.cumsum(np.random.RandomState(42).randn(n))
    df = pd.DataFrame({"close": prices})

    extractor = MomentumFeatureExtractor()
    out = extractor.transform(df, append=True)

    # Check input preservation
    assert "close" in out.columns
    assert len(out) == n

    # Check expected momentum features exist
    expected_cols = [
        "mom_5d",
        "mom_10d",
        "mom_20d",
        "mom_60d",
        "mom_120d",
        "mom_252d",
        "mom_12_1m",
        "roc_10",
        "roc_20",
        "sma_50_200_spread",
        "sma_50_200_bullish",
        "sma_50_200_cross",
        "rsi_14",
        "macd_line_12_26_9",
        "macd_signal_12_26_9",
        "macd_hist_12_26_9",
        "macd_norm_12_26_9",
        "macd_cross_12_26_9",
    ]
    for c in expected_cols:
        assert c in out.columns, f"Expected column {c} missing from extractor output."

    # Test append=False returns only feature columns
    feats_only = extractor(df, append=False)
    assert "close" not in feats_only.columns
    assert "mom_20d" in feats_only.columns


def test_feature_base_validation_errors():
    """FeatureBase must validate missing columns and empty DataFrames."""
    extractor = MomentumFeatureExtractor(price_col="close")

    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        extractor.transform(pd.DataFrame())

    with pytest.raises(ValueError, match="missing required column"):
        extractor.transform(pd.DataFrame({"open": [1, 2, 3]}))


# ============================================================
# 8. Feature Registry Tests
# ============================================================


def test_feature_registry_registration_and_computation():
    """Test registering, retrieving, and computing features via FeatureRegistry."""
    reg = FeatureRegistry()

    @register_feature(
        name="test_mom_3d",
        category="momentum",
        description="3-day momentum for test",
        lookback_horizon=3,
        registry=reg,
    )
    def dummy_mom(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"test_mom_3d": df["close"].pct_change(3)}, index=df.index)

    assert reg.contains("test_mom_3d")
    meta = reg.get("test_mom_3d")
    assert meta.lookback_horizon == 3
    assert meta.category == "momentum"

    df_test = pd.DataFrame({"close": [10, 11, 12, 13, 14, 15]})
    computed = reg.compute_features(df_test, feature_names=["test_mom_3d"])
    assert "test_mom_3d" in computed.columns
    assert np.isclose(computed["test_mom_3d"].iloc[3], (13.0 - 10.0) / 10.0)

    # DataFrame overview
    tbl = reg.to_dataframe()
    assert len(tbl) == 1
    assert tbl.iloc[0]["name"] == "test_mom_3d"
