# ============================================================
# Unit Tests: Kelly Criterion Position Sizing (Phase 35)
# ============================================================

import numpy as np
import pandas as pd

from src.portfolio.kelly_sizing import (
    adjust_multi_asset_kelly,
    derive_kelly_parameters,
    fractional_kelly,
    generate_kelly_allocation_series,
    kelly_criterion,
    position_size,
)


def test_kelly_criterion_hand_computed_examples():
    """Verify Kelly formula against exact hand-computed solutions."""
    # Case 1: Standard edge
    # p = 0.60, b = 1.0 -> f* = 0.60 - 0.40/1.0 = 0.20 (20%)
    assert abs(kelly_criterion(0.60, 1.0) - 0.20) < 1e-6

    # Case 2: Asymmetric payoff
    # p = 0.50, b = 2.0 -> f* = 0.50 - 0.50/2.0 = 0.25 (25%)
    assert abs(kelly_criterion(0.50, 2.0) - 0.25) < 1e-6

    # Case 3: High win rate, low payoff
    # p = 0.80, b = 0.5 -> f* = 0.80 - 0.20/0.5 = 0.40 (40%)
    assert abs(kelly_criterion(0.80, 0.5) - 0.40) < 1e-6


def test_kelly_criterion_negative_or_zero_edge():
    """Verify Kelly returns 0.0 when expected edge is non-positive."""
    # Case 1: Unfavorable coin toss (p = 0.40, b = 1.0) -> edge < 0
    assert kelly_criterion(0.40, 1.0) == 0.0

    # Case 2: Zero edge (p = 0.50, b = 1.0)
    assert kelly_criterion(0.50, 1.0) == 0.0

    # Case 3: Invalid boundaries
    assert kelly_criterion(0.0, 2.0) == 0.0
    assert kelly_criterion(1.0, 2.0) == 0.0
    assert kelly_criterion(0.6, 0.0) == 0.0
    assert kelly_criterion(0.6, -1.0) == 0.0


def test_fractional_kelly_scaling():
    """Verify fractional Kelly correctly scales the theoretical optimal fraction."""
    # Base: p = 0.60, b = 1.0 -> f* = 0.20
    # Half-Kelly (0.5x) -> 0.10
    assert abs(fractional_kelly(0.60, 1.0, fraction=0.5) - 0.10) < 1e-6

    # Quarter-Kelly (0.25x) -> 0.05
    assert abs(fractional_kelly(0.60, 1.0, fraction=0.25) - 0.05) < 1e-6

    # Zero or negative fraction
    assert fractional_kelly(0.60, 1.0, fraction=0.0) == 0.0
    assert fractional_kelly(0.60, 1.0, fraction=-0.5) == 0.0


def test_derive_kelly_parameters():
    """Verify parameter estimation from historical return series."""
    # 6 wins (+2% each), 4 losses (-1% each)
    # Win rate = 6/10 = 0.60, Payoff ratio = 0.02 / 0.01 = 2.0
    rets = np.array([0.02, 0.02, -0.01, 0.02, -0.01, 0.02, 0.02, -0.01, 0.02, -0.01])
    p, b = derive_kelly_parameters(rets)

    assert p == 0.60
    assert b == 2.0


def test_position_size_caps_and_confidence():
    """Verify position sizing respects hard safety ceiling and minimum confidence."""
    # Even if Kelly suggests a huge size (e.g. 50%), max_position_pct (25%) must cap it
    size = position_size(
        model_confidence=0.85,
        historical_win_rate=0.75,
        win_loss_ratio=2.0,
        kelly_fraction=1.0,
        max_position_pct=0.25,
    )
    assert size == 0.25

    # Confidence below min_confidence (0.50) -> position 0
    size_low_conf = position_size(
        model_confidence=0.48,
        historical_win_rate=0.60,
        win_loss_ratio=1.5,
        min_confidence=0.50,
    )
    assert size_low_conf == 0.0


def test_adjust_multi_asset_kelly():
    """Verify multi-asset correlation adjustment dampens leverage for correlated positions."""
    # Two assets with high correlation rho = 0.80
    corr_df = pd.DataFrame(
        [[1.0, 0.80], [0.80, 1.0]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )
    # Naive sizes: 20% AAPL, 20% MSFT (total gross = 40%)
    raw_positions = {"AAPL": 0.20, "MSFT": 0.20}
    adjusted = adjust_multi_asset_kelly(
        positions=raw_positions,
        corr_matrix=corr_df,
        max_portfolio_leverage=1.0,
    )

    assert "AAPL" in adjusted and "MSFT" in adjusted
    assert adjusted["AAPL"] > 0.0
    # The correlation adjustment should dampen each correlated position
    assert adjusted["AAPL"] <= raw_positions["AAPL"]
    assert sum(adjusted.values()) <= 1.0


def test_generate_kelly_allocation_series():
    """Verify allocation series generator produces complete, consistent time series."""
    rng = np.random.RandomState(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    probs = pd.Series(rng.uniform(0.45, 0.70, n), index=dates)
    rets = pd.Series(rng.normal(0.001, 0.015, n), index=dates)

    df_alloc = generate_kelly_allocation_series(
        probabilities=probs,
        asset_returns=rets,
        kelly_fraction=0.5,
        max_position_pct=0.25,
        rolling_window=30,
    )

    assert len(df_alloc) == n
    assert "full_kelly" in df_alloc.columns
    assert "fractional_kelly" in df_alloc.columns
    assert "capped_position_size" in df_alloc.columns
    assert (df_alloc["capped_position_size"] <= 0.25).all()
    assert (df_alloc["capped_position_size"] >= 0.0).all()
