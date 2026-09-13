# ============================================================
# Unit Tests: EDA Utilities (Phase 8)
# ============================================================
"""
Tests for exploratory data analysis functions in src/features/eda_utils.py.
Verifies statistical computations (skewness, excess kurtosis, volatility)
against synthetic distributions and ensures headless plotting runs cleanly.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# Safeguard for environments where Application Control restricts C-extensions
if "matplotlib._c_internal_utils" not in sys.modules:
    try:
        import matplotlib._c_internal_utils  # noqa: F401
    except ImportError:
        sys.modules["matplotlib._c_internal_utils"] = types.ModuleType(
            "matplotlib._c_internal_utils"
        )

import matplotlib

# Force headless backend before any other matplotlib operations
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.features.eda_utils import (
    compute_daily_returns,
    plot_correlation_matrix,
    plot_price_series,
    plot_returns_distribution,
    plot_rolling_volatility,
    plot_seasonality,
    summary_stats_table,
)


@pytest.fixture(autouse=True)
def close_figures() -> None:
    """Ensure all matplotlib figures are closed after each test."""
    yield
    plt.close("all")


@pytest.fixture
def synthetic_stock_df() -> pd.DataFrame:
    """Fixture generating 250 business days of synthetic geometric Brownian motion."""
    np.random.seed(42)
    dates = pd.bdate_range(start="2023-01-02", periods=250, freq="B")

    # Generate daily returns ~ N(0.0005, 0.015)
    daily_rets = np.random.normal(loc=0.0005, scale=0.015, size=250)
    prices = 100.0 * np.exp(np.cumsum(daily_rets))

    # Construct realistic OHLCV
    highs = prices * (1.0 + np.abs(np.random.normal(0, 0.005, size=250)))
    lows = prices * (1.0 - np.abs(np.random.normal(0, 0.005, size=250)))
    opens = (highs + lows) / 2.0
    volumes = np.random.randint(1_000_000, 5_000_000, size=250)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        },
        index=dates,
    )
    return df


# ============================================================
# Statistical Computation Tests
# ============================================================


def test_compute_daily_returns(synthetic_stock_df: pd.DataFrame) -> None:
    rets = compute_daily_returns(synthetic_stock_df)
    assert len(rets) == len(synthetic_stock_df) - 1
    assert not rets.isna().any()
    assert isinstance(rets, pd.Series)


def test_summary_stats_table_columns_and_shape(synthetic_stock_df: pd.DataFrame) -> None:
    dfs = {"SYNTH_A": synthetic_stock_df, "SYNTH_B": synthetic_stock_df * 1.5}
    stats_df = summary_stats_table(dfs=dfs)

    assert len(stats_df) == 2
    assert "SYNTH_A" in stats_df.index
    assert "SYNTH_B" in stats_df.index

    expected_cols = [
        "count",
        "mean_daily_pct",
        "ann_return_pct",
        "daily_vol_pct",
        "ann_vol_pct",
        "skewness",
        "excess_kurtosis",
        "min_return_pct",
        "max_return_pct",
        "sharpe_ratio",
    ]
    for col in expected_cols:
        assert col in stats_df.columns


def test_statistical_moments_accuracy() -> None:
    """Test that skewness and excess kurtosis are accurately computed against exact formulas."""
    np.random.seed(123)
    n = 2000

    # 1. Normal distribution (skew ~ 0, excess kurtosis ~ 0)
    norm_vals = np.random.normal(loc=0.0, scale=0.01, size=n)
    norm_prices = 100.0 * np.exp(np.cumsum(norm_vals))
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df_norm = pd.DataFrame({"close": norm_prices}, index=dates)

    # 2. Student-t distribution with df=4 (heavy-tailed, fat tails -> high excess kurtosis)
    t_vals = np.random.standard_t(df=4, size=n) * 0.01
    t_prices = 100.0 * np.exp(np.cumsum(t_vals))
    df_t = pd.DataFrame({"close": t_prices}, index=dates)

    dfs = {"NORMAL": df_norm, "FAT_TAILS": df_t}
    stats_df = summary_stats_table(dfs=dfs)

    # Analytical reference calculation on raw percentage returns
    norm_rets = compute_daily_returns(df_norm)
    t_rets = compute_daily_returns(df_t)

    # Fisher-Pearson unbiased skewness formula
    def calc_skew(s: pd.Series) -> float:
        m = s.mean()
        std = s.std(ddof=1)
        k = len(s)
        return (k / ((k - 1) * (k - 2))) * np.sum(((s - m) / std) ** 3)

    # Fisher excess kurtosis formula
    def calc_excess_kurt(s: pd.Series) -> float:
        m = s.mean()
        std = s.std(ddof=1)
        k = len(s)
        term1 = (k * (k + 1)) / ((k - 1) * (k - 2) * (k - 3))
        term2 = np.sum(((s - m) / std) ** 4)
        term3 = (3.0 * (k - 1) ** 2) / ((k - 2) * (k - 3))
        return term1 * term2 - term3

    expected_norm_skew = calc_skew(norm_rets)
    expected_norm_kurt = calc_excess_kurt(norm_rets)
    expected_t_kurt = calc_excess_kurt(t_rets)

    # Numerical verification
    assert np.isclose(stats_df.loc["NORMAL", "skewness"], expected_norm_skew, atol=1e-5)
    assert np.isclose(stats_df.loc["NORMAL", "excess_kurtosis"], expected_norm_kurt, atol=1e-5)
    assert np.isclose(stats_df.loc["FAT_TAILS", "excess_kurtosis"], expected_t_kurt, atol=1e-5)

    # Fat-tail check: Student-t excess kurtosis must be significantly greater than Normal
    assert stats_df.loc["FAT_TAILS", "excess_kurtosis"] > 1.5


# ============================================================
# Plotting Tests (Headless Verification)
# ============================================================


def test_plot_price_series(synthetic_stock_df: pd.DataFrame, tmp_path: Path) -> None:
    out_file = tmp_path / "price_series.png"
    fig, (ax1, ax2) = plot_price_series(
        ticker="SYNTH",
        df=synthetic_stock_df,
        save_path=out_file,
    )
    assert fig is not None
    assert ax1 is not None
    assert ax2 is not None
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_returns_distribution(synthetic_stock_df: pd.DataFrame, tmp_path: Path) -> None:
    out_file = tmp_path / "returns_dist.png"
    fig, ax = plot_returns_distribution(
        ticker="SYNTH",
        df=synthetic_stock_df,
        save_path=out_file,
    )
    assert fig is not None
    assert ax is not None
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_correlation_matrix(synthetic_stock_df: pd.DataFrame, tmp_path: Path) -> None:
    out_file = tmp_path / "corr_matrix.png"
    dfs = {
        "ASSET_1": synthetic_stock_df,
        "ASSET_2": synthetic_stock_df.copy() * 0.9,
    }
    fig, ax = plot_correlation_matrix(
        tickers=["ASSET_1", "ASSET_2"],
        dfs=dfs,
        save_path=out_file,
    )
    assert fig is not None
    assert ax is not None
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_rolling_volatility(synthetic_stock_df: pd.DataFrame, tmp_path: Path) -> None:
    out_file = tmp_path / "rolling_vol.png"
    fig, ax = plot_rolling_volatility(
        ticker="SYNTH",
        window=20,
        df=synthetic_stock_df,
        save_path=out_file,
    )
    assert fig is not None
    assert ax is not None
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_seasonality(synthetic_stock_df: pd.DataFrame, tmp_path: Path) -> None:
    out_file = tmp_path / "seasonality.png"
    fig, (ax1, ax2) = plot_seasonality(
        ticker="SYNTH",
        df=synthetic_stock_df,
        save_path=out_file,
    )
    assert fig is not None
    assert ax1 is not None
    assert ax2 is not None
    assert out_file.exists()
    assert out_file.stat().st_size > 0
