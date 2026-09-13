# ============================================================
# Anti-Leakage & Look-Ahead Verification Suite (Phase 12)
# ============================================================
"""
Explicit automated tests verifying that NO feature extraction function
exhibits forward-looking bias or data leakage.

Methodology:
------------
1. Generate a continuous baseline market price series of length N (e.g. N = 120).
2. Choose an arbitrary historical cutoff point T < N (e.g. T = 70).
3. Compute feature values on the full unperturbed series: F_baseline[0..T].
4. Perform severe perturbations on future data points (t = T+1 .. N):
   - Future spike corruption (multiplying future prices by 1,000,000x)
   - Future inversion / negation
   - Future truncation (deleting all data after T)
5. Recompute feature values on the perturbed series: F_perturbed[0..T].
6. Assert that for EVERY feature column and EVERY row t in [0..T]:
       |F_baseline[t] - F_perturbed[t]| == 0.0  (within float64 machine epsilon)

If any feature leaks future data, this test will fail decisively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


@pytest.fixture
def synthetic_price_series() -> pd.DataFrame:
    """Generate reproducible geometric random walk prices."""
    np.random.seed(12345)
    n = 150
    returns = np.random.normal(0.0005, 0.015, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "close": prices,
            "open": prices * 0.995,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "volume": 1_000_000,
        },
        index=dates,
    )


def test_no_lookahead_price_momentum(synthetic_price_series: pd.DataFrame):
    """Verify compute_price_momentum is invariant to future price perturbations."""
    df = synthetic_price_series.copy()
    cutoff_idx = 80

    base_features = compute_price_momentum(df, windows=[5, 10, 20, 60])

    # Perturb all future prices drastically
    df_corrupted = df.copy()
    df_corrupted.iloc[cutoff_idx + 1 :, df.columns.get_loc("close")] *= 1000.0

    perturbed_features = compute_price_momentum(df_corrupted, windows=[5, 10, 20, 60])

    # Up to and including cutoff_idx, values must be strictly identical
    for col in base_features.columns:
        s_base = base_features[col].iloc[: cutoff_idx + 1]
        s_pert = perturbed_features[col].iloc[: cutoff_idx + 1]
        np.testing.assert_allclose(
            s_base.values,
            s_pert.values,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
            err_msg=f"Lookahead detected in compute_price_momentum for column {col}!",
        )


def test_no_lookahead_jegadeesh_titman(synthetic_price_series: pd.DataFrame):
    """Verify 12-1 month momentum is strictly past-only."""
    df = synthetic_price_series.copy()
    cutoff_idx = 75

    base = compute_jegadeesh_titman_momentum(df, total_window=50, skip_window=10)

    df_corrupted = df.copy()
    df_corrupted.iloc[cutoff_idx + 1 :, df.columns.get_loc("close")] = 999999.0

    perturbed = compute_jegadeesh_titman_momentum(df_corrupted, total_window=50, skip_window=10)

    for col in base.columns:
        s_base = base[col].iloc[: cutoff_idx + 1]
        s_pert = perturbed[col].iloc[: cutoff_idx + 1]
        np.testing.assert_allclose(
            s_base.values,
            s_pert.values,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
            err_msg=f"Lookahead detected in compute_jegadeesh_titman_momentum for column {col}!",
        )


def test_no_lookahead_rate_of_change(synthetic_price_series: pd.DataFrame):
    """Verify Rate of Change (ROC) has zero future leakage."""
    df = synthetic_price_series.copy()
    cutoff_idx = 70

    base = compute_rate_of_change(df, windows=[10, 20])

    df_corrupted = df.copy()
    df_corrupted.iloc[cutoff_idx + 1 :, df.columns.get_loc("close")] = 0.01

    perturbed = compute_rate_of_change(df_corrupted, windows=[10, 20])

    for col in base.columns:
        s_base = base[col].iloc[: cutoff_idx + 1]
        s_pert = perturbed[col].iloc[: cutoff_idx + 1]
        np.testing.assert_allclose(
            s_base.values,
            s_pert.values,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
            err_msg=f"Lookahead detected in compute_rate_of_change for column {col}!",
        )


def test_no_lookahead_ma_crossover(synthetic_price_series: pd.DataFrame):
    """Verify moving average spreads and crossovers do not peek into future."""
    df = synthetic_price_series.copy()
    cutoff_idx = 85

    base = compute_ma_crossover(df, fast_window=10, slow_window=30, ma_type="sma")

    df_corrupted = df.copy()
    df_corrupted.iloc[cutoff_idx + 1 :, df.columns.get_loc("close")] = 5000.0

    perturbed = compute_ma_crossover(df_corrupted, fast_window=10, slow_window=30, ma_type="sma")

    for col in base.columns:
        s_base = base[col].iloc[: cutoff_idx + 1]
        s_pert = perturbed[col].iloc[: cutoff_idx + 1]
        np.testing.assert_allclose(
            s_base.values,
            s_pert.values,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
            err_msg=f"Lookahead detected in compute_ma_crossover for column {col}!",
        )


def test_no_lookahead_rsi(synthetic_price_series: pd.DataFrame):
    """Verify scratch RSI calculation does not peek into future."""
    df = synthetic_price_series.copy()
    cutoff_idx = 60

    base = compute_rsi(df, window=14)

    # Corrupt future prices with extreme random values
    df_corrupted = df.copy()
    np.random.seed(999)
    df_corrupted.iloc[cutoff_idx + 1 :, df.columns.get_loc("close")] = np.random.uniform(
        1.0, 1000.0, len(df) - cutoff_idx - 1
    )

    perturbed = compute_rsi(df_corrupted, window=14)

    s_base = base["rsi_14"].iloc[: cutoff_idx + 1]
    s_pert = perturbed["rsi_14"].iloc[: cutoff_idx + 1]
    np.testing.assert_allclose(
        s_base.values,
        s_pert.values,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
        err_msg="Lookahead detected in scratch compute_rsi!",
    )


def test_no_lookahead_macd(synthetic_price_series: pd.DataFrame):
    """Verify MACD components and crossovers have zero lookahead."""
    df = synthetic_price_series.copy()
    cutoff_idx = 75

    base = compute_macd(df, fast_period=12, slow_period=26, signal_period=9)

    df_corrupted = df.copy()
    df_corrupted.iloc[cutoff_idx + 1 :, df.columns.get_loc("close")] *= 50.0

    perturbed = compute_macd(df_corrupted, fast_period=12, slow_period=26, signal_period=9)

    for col in base.columns:
        s_base = base[col].iloc[: cutoff_idx + 1]
        s_pert = perturbed[col].iloc[: cutoff_idx + 1]
        np.testing.assert_allclose(
            s_base.values,
            s_pert.values,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
            err_msg=f"Lookahead detected in compute_macd for column {col}!",
        )


def test_no_lookahead_truncation_invariance(synthetic_price_series: pd.DataFrame):
    """Truncating future rows entirely must produce identical values for past rows.

    This ensures that vectorization or caching does not depend on total series length.
    """
    df_full = synthetic_price_series.copy()
    cutoff_idx = 90
    df_truncated = df_full.iloc[: cutoff_idx + 1].copy()

    extractor = MomentumFeatureExtractor(
        momentum_windows=(5, 10, 20),
        roc_windows=(10,),
        fast_ma=10,
        slow_ma=30,
        rsi_window=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
    )

    feat_full = extractor.compute(df_full)
    feat_trunc = extractor.compute(df_truncated)

    for col in feat_full.columns:
        s_full = feat_full[col].iloc[: cutoff_idx + 1]
        s_trunc = feat_trunc[col]
        np.testing.assert_allclose(
            s_full.values,
            s_trunc.values,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
            err_msg=f"Truncation invariance failed for column {col}!",
        )


def test_no_lookahead_cross_sectional_momentum():
    """Verify that future changes in asset A's returns do not affect cross-sectional ranks at t <= T."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    np.random.seed(42)
    prices = pd.DataFrame(
        {
            "AAPL": 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 60))),
            "MSFT": 200.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 60))),
            "SPY": 400.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 60))),
        },
        index=dates,
    )

    cutoff_idx = 40
    base_ranks = compute_cross_sectional_momentum(prices, window=20, skip_window=5)

    # Corrupt AAPL prices after cutoff
    prices_corrupted = prices.copy()
    prices_corrupted.iloc[cutoff_idx + 1 :, 0] *= 100.0

    perturbed_ranks = compute_cross_sectional_momentum(prices_corrupted, window=20, skip_window=5)

    for col in base_ranks.columns:
        s_base = base_ranks[col].iloc[: cutoff_idx + 1]
        s_pert = perturbed_ranks[col].iloc[: cutoff_idx + 1]
        np.testing.assert_allclose(
            s_base.values,
            s_pert.values,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
            err_msg=f"Lookahead detected in cross-sectional momentum for {col}!",
        )
