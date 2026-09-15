# ============================================================
# Unit Tests — Cointegration & Pairs Trading (Phase 38)
# ============================================================
"""
Unit tests for pairs trading, cointegration tests (Engle-Granger and Johansen),
rolling hedge ratio calculations, spread z-scores, and trading signal generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.pairs_trading import (
    backtest_pairs_strategy,
    compute_rolling_hedge_ratio,
    compute_spread,
    compute_spread_zscore,
    compute_static_hedge_ratio,
    engle_granger_test,
    generate_pairs_signals,
    johansen_test,
    pairs_results_to_dataframe,
    screen_pairs,
)


@pytest.fixture
def synthetic_pairs_data() -> pd.DataFrame:
    """Generate synthetic price series with mathematically KNOWN cointegration properties.

    - X: Random walk I(1)
    - Y: Cointegrated with X (Y = 1.8 * X + stationary AR(1) noise) -> I(0) linear combination
    - Z: Independent random walk I(1) -> NOT cointegrated with X or Y
    """
    np.random.seed(42)
    n = 500

    # Underlying random walk X
    innov_x = np.random.normal(0, 1, size=n)
    x = 100.0 + np.cumsum(innov_x)

    # Stationary AR(1) noise for Y: epsilon_t = 0.6 * epsilon_{t-1} + noise
    noise_y = np.zeros(n)
    innov_noise = np.random.normal(0, 0.5, size=n)
    for t in range(1, n):
        noise_y[t] = 0.6 * noise_y[t - 1] + innov_noise[t]

    true_beta = 1.8
    true_alpha = 10.0
    y = true_alpha + true_beta * x + noise_y

    # Independent random walk Z
    innov_z = np.random.normal(0, 1, size=n)
    z = 80.0 + np.cumsum(innov_z)

    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({"X": x, "Y": y, "Z": z}, index=dates)


def test_engle_granger_cointegration_detection(synthetic_pairs_data: pd.DataFrame):
    """Test Engle-Granger detects cointegration for true cointegrated pair and rejects independent series."""
    df = synthetic_pairs_data

    # Test Y and X: should be strongly cointegrated (p < 0.01)
    stat, p_val, crits, hedge_ratio, intercept = engle_granger_test(df["Y"], df["X"])
    assert p_val < 0.01, f"Expected p < 0.01 for cointegrated pair, got {p_val}"
    assert stat < crits["5%"], "ADF stat should be more negative than 5% critical value"
    assert np.isclose(hedge_ratio, 1.8, atol=0.15), f"Estimated beta {hedge_ratio} should be close to 1.8"
    assert np.isclose(intercept, 10.0, atol=3.0), f"Estimated alpha {intercept} should be close to 10.0"

    # Test Z and X: independent random walks, should NOT be cointegrated (p > 0.05)
    stat_z, p_val_z, crits_z, _, _ = engle_granger_test(df["Z"], df["X"])
    assert p_val_z > 0.05, f"Expected p > 0.05 for non-cointegrated pair, got {p_val_z}"


def test_johansen_cointegration_test(synthetic_pairs_data: pd.DataFrame):
    """Test Johansen test correctly flags cointegrated pair vs non-cointegrated pair."""
    df = synthetic_pairs_data

    # Pair (Y, X)
    res_coint = johansen_test(df[["Y", "X"]])
    assert res_coint["is_cointegrated_trace"], "Johansen trace test should identify cointegration"
    assert res_coint["trace_stat"] > res_coint["trace_crit_95"]

    # Pair (Z, X)
    res_non_coint = johansen_test(df[["Z", "X"]])
    assert not res_non_coint["is_cointegrated"], "Independent pair should not pass both Johansen tests"


def test_screen_pairs_universe(synthetic_pairs_data: pd.DataFrame):
    """Test screen_pairs ranks cointegrated pairs above non-cointegrated pairs."""
    df = synthetic_pairs_data
    results = screen_pairs(df, method="engle_granger", p_value_threshold=0.05)

    assert len(results) == 3  # (X, Y), (X, Z), (Y, Z)
    top_pair = results[0]
    # The cointegrated pair should be at the top
    assert {top_pair.asset_x, top_pair.asset_y} == {"X", "Y"}
    assert top_pair.is_cointegrated is True
    assert top_pair.p_value < 0.05
    assert np.isfinite(top_pair.half_life)
    assert top_pair.half_life > 0.0

    df_res = pairs_results_to_dataframe(results)
    assert len(df_res) == 3
    assert "pair" in df_res.columns
    assert "hedge_ratio" in df_res.columns


def test_static_and_rolling_hedge_ratio(synthetic_pairs_data: pd.DataFrame):
    """Test static and rolling hedge ratio estimation."""
    df = synthetic_pairs_data

    # Static
    beta_static, alpha_static = compute_static_hedge_ratio(df["Y"], df["X"])
    assert np.isclose(beta_static, 1.8, atol=0.15)
    assert np.isclose(alpha_static, 10.0, atol=3.0)

    # Rolling
    rolling_hr = compute_rolling_hedge_ratio(df["Y"], df["X"], window=60)
    assert "hedge_ratio" in rolling_hr.columns
    assert "intercept" in rolling_hr.columns
    assert len(rolling_hr) == len(df)
    assert not rolling_hr.isna().any().any()
    # Mean of rolling beta should be near 1.8
    assert np.isclose(rolling_hr["hedge_ratio"].mean(), 1.8, atol=0.25)


def test_spread_and_zscore_computation(synthetic_pairs_data: pd.DataFrame):
    """Test spread calculation and rolling z-score properties."""
    df = synthetic_pairs_data
    rolling_hr = compute_rolling_hedge_ratio(df["Y"], df["X"], window=60)

    spread = compute_spread(
        series_y=df["Y"],
        series_x=df["X"],
        hedge_ratio=rolling_hr["hedge_ratio"],
        intercept=rolling_hr["intercept"],
    )
    assert len(spread) == len(df)

    zscore = compute_spread_zscore(spread, window=30)
    assert len(zscore) == len(df)
    assert not zscore.isna().any()

    # Z-scores should have zero mean and standard deviation approx 1.0
    valid_z = zscore.iloc[40:]
    assert abs(valid_z.mean()) < 0.2
    assert abs(valid_z.std() - 1.0) < 0.3


def test_pairs_signal_generation_state_machine():
    """Test state-machine signal transitions: entry, exit, and stop-loss."""
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    # Crafted z-scores:
    # 0: 0.0 (flat)
    # 1: -2.5 (trigger long spread +1)
    # 2: -1.5 (maintain long spread +1)
    # 3: -0.1 (exit mean-reversion 0)
    # 4: +2.2 (trigger short spread -1)
    # 5: +1.2 (maintain short spread -1)
    # 6: +3.8 (stop-loss breach, exit 0)
    # 7: +1.0 (remain flat 0)
    # 8: -3.6 (over threshold directly into stop, flat or stopped)
    # 9: 0.0 (flat)
    z_series = pd.Series([0.0, -2.5, -1.5, -0.1, 2.2, 1.2, 3.8, 1.0, 0.0, 0.0], index=idx)

    signals = generate_pairs_signals(
        z_series, entry_threshold=2.0, exit_threshold=0.2, stop_loss_threshold=3.5
    )

    expected = [0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    np.testing.assert_array_equal(signals.values, expected)


def test_backtest_pairs_strategy(synthetic_pairs_data: pd.DataFrame):
    """Test pairs backtesting execution, equity curve, trade log, and metrics."""
    df = synthetic_pairs_data
    rolling_hr = compute_rolling_hedge_ratio(df["Y"], df["X"], window=60)
    spread = compute_spread(
        df["Y"], df["X"], rolling_hr["hedge_ratio"], rolling_hr["intercept"]
    )
    zscore = compute_spread_zscore(spread, window=30)
    signals = generate_pairs_signals(zscore, entry_threshold=1.5, exit_threshold=0.0)

    result = backtest_pairs_strategy(
        price_y=df["Y"],
        price_x=df["X"],
        signals=signals,
        hedge_ratio=rolling_hr["hedge_ratio"],
        initial_capital=100000.0,
    )

    assert len(result.portfolio_equity) == len(df)
    assert result.portfolio_equity.iloc[0] == 100000.0
    assert result.max_drawdown <= 0.0
    assert 0.0 <= result.win_rate <= 1.0
    assert result.total_trades >= 0
    assert "total_return_pct" in result.summary_dict()
