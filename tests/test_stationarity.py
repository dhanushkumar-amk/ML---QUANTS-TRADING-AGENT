# ============================================================
# Unit Tests for Stationarity Testing Module (Phase 9)
# ============================================================
"""
Tests for Augmented Dickey-Fuller (ADF), KPSS, and combined stationarity reporting.
Covers:
  1. White noise process (stationary).
  2. Random walk process (non-stationary / unit root).
  3. Linear trend process (non-stationary under constant regression).
  4. First-order differencing of random walk (achieves stationarity).
  5. Multi-series batch testing and reporting format.
  6. Edge cases: handling NaNs/Infs, short series error, order < 1 differencing.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.stationarity import (
    ADFResult,
    KPSSResult,
    StationarityReport,
    adf_test,
    difference_series,
    kpss_test,
    stationarity_report,
    test_multiple_series,
)


@pytest.fixture
def rng():
    """Deterministic random generator for reproducible test series."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def white_noise(rng) -> pd.Series:
    """Stationary white noise process X_t ~ N(0, 1)."""
    data = rng.standard_normal(size=500)
    return pd.Series(data, name="white_noise")


@pytest.fixture
def random_walk(rng) -> pd.Series:
    """Non-stationary unit-root random walk Y_t = sum(eps_t)."""
    innovations = rng.standard_normal(size=500)
    return pd.Series(np.cumsum(innovations), name="random_walk")


@pytest.fixture
def trend_with_noise(rng) -> pd.Series:
    """Deterministic trend series Z_t = 0.05*t + eps_t."""
    t = np.arange(500)
    noise = rng.standard_normal(size=500)
    return pd.Series(0.05 * t + noise, name="trend_series")


# ============================================================
# 1. White Noise Stationarity Tests
# ============================================================


def test_white_noise_adf(white_noise):
    """White noise must reject ADF null (unit root) at 1% and 5% levels."""
    result = adf_test(white_noise)
    assert isinstance(result, ADFResult)
    assert result.p_value < 0.05
    assert result.is_stationary_5pct
    assert result.is_stationary(0.05)
    assert result.n_observations > 0
    assert "1%" in result.critical_values
    assert "5%" in result.critical_values


def test_white_noise_kpss(white_noise):
    """White noise must fail to reject KPSS null (stationary) at 5% level."""
    result = kpss_test(white_noise)
    assert isinstance(result, KPSSResult)
    assert result.p_value >= 0.05
    assert result.is_stationary_5pct
    assert result.is_stationary(0.05)
    assert "5%" in result.critical_values


def test_white_noise_combined_report(white_noise):
    """StationarityReport on white noise must arrive at consensus 'stationary'."""
    report = stationarity_report(white_noise, name="white_noise", alpha=0.05)
    assert isinstance(report, StationarityReport)
    assert report.conclusion == "stationary"
    assert report.is_stationary is True
    assert "Both tests agree" in report.explanation

    d = report.to_dict()
    assert d["conclusion"] == "stationary"
    assert d["adf_stationary"] is True
    assert d["kpss_stationary"] is True


# ============================================================
# 2. Random Walk Non-Stationarity Tests
# ============================================================


def test_random_walk_adf(random_walk):
    """Random walk must fail to reject ADF null (unit root)."""
    result = adf_test(random_walk)
    assert result.p_value >= 0.05
    assert not result.is_stationary(0.05)


def test_random_walk_kpss(random_walk):
    """Random walk must reject KPSS null (stationary)."""
    result = kpss_test(random_walk)
    assert result.p_value < 0.05
    assert not result.is_stationary(0.05)


def test_random_walk_combined_report(random_walk):
    """StationarityReport on random walk must arrive at consensus 'non-stationary'."""
    report = stationarity_report(random_walk, name="random_walk", alpha=0.05)
    assert report.conclusion == "non-stationary"
    assert report.is_stationary is False
    assert "Both tests agree: ADF fails to reject" in report.explanation


# ============================================================
# 3. Deterministic Trend Process Tests
# ============================================================


def test_trend_with_noise_report(trend_with_noise):
    """Linear trend with noise tested against constant 'c' should be non-stationary."""
    report = stationarity_report(trend_with_noise, regression="c")
    assert report.conclusion in ("non-stationary", "inconclusive")


# ============================================================
# 4. Differencing Verification
# ============================================================


def test_differencing_random_walk(random_walk):
    """First differencing of random walk produces stationary series."""
    diff_series = difference_series(random_walk, order=1)
    assert len(diff_series) == len(random_walk) - 1
    assert diff_series.name == f"{random_walk.name}_diff1"

    report = stationarity_report(diff_series, name="diff_random_walk", alpha=0.05)
    assert report.conclusion == "stationary"
    assert report.is_stationary is True


def test_difference_series_invalid_order(random_walk):
    """Order < 1 must raise ValueError."""
    with pytest.raises(ValueError, match="Differencing order must be >= 1"):
        difference_series(random_walk, order=0)


def test_higher_order_differencing(random_walk):
    """Second-order differencing drops 2 rows."""
    diff2 = difference_series(random_walk, order=2)
    assert len(diff2) == len(random_walk) - 2
    assert diff2.name == f"{random_walk.name}_diff2"


# ============================================================
# 5. Batch Testing Utility
# ============================================================


def test_test_multiple_series(white_noise, random_walk):
    """Batch test creates DataFrame indexed by series with all required columns."""
    series_dict = {
        "wn": white_noise,
        "rw": random_walk,
    }
    df_results = test_multiple_series(series_dict, alpha=0.05)

    assert isinstance(df_results, pd.DataFrame)
    assert set(df_results.index) == {"wn", "rw"}
    expected_cols = [
        "adf_stat",
        "adf_pvalue",
        "adf_stationary",
        "kpss_stat",
        "kpss_pvalue",
        "kpss_stationary",
        "conclusion",
    ]
    for col in expected_cols:
        assert col in df_results.columns

    assert df_results.loc["wn", "conclusion"] == "stationary"
    assert df_results.loc["rw", "conclusion"] == "non-stationary"


def test_test_multiple_series_error_handling(white_noise):
    """If one series is invalid, batch test should record error without aborting."""
    series_dict = {
        "valid": white_noise,
        "too_short": pd.Series([1.0, 2.0, 3.0]),
    }
    df_results = test_multiple_series(series_dict, alpha=0.05)
    assert "valid" in df_results.index
    assert "too_short" in df_results.index
    assert df_results.loc["valid", "conclusion"] == "stationary"
    assert "ERROR" in df_results.loc["too_short", "conclusion"]


# ============================================================
# 6. Cleaning, Edge Cases, and Properties
# ============================================================


def test_clean_series_nans_and_infs(rng):
    """Series with NaNs and infinities are cleaned before testing."""
    raw = rng.standard_normal(size=100)
    raw[5] = np.nan
    raw[10] = np.inf
    raw[15] = -np.inf
    s = pd.Series(raw)

    report = stationarity_report(s, alpha=0.05)
    assert report.conclusion in ("stationary", "non-stationary", "inconclusive")


def test_short_series_raises():
    """Series with fewer than 10 observations raises ValueError."""
    short = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="too short"):
        adf_test(short)
    with pytest.raises(ValueError, match="too short"):
        kpss_test(short)


def test_result_properties(white_noise):
    """Verify ADFResult and KPSSResult properties under different critical values."""
    adf = adf_test(white_noise)
    assert isinstance(adf.is_stationary_1pct, (bool, np.bool_))
    assert isinstance(adf.is_stationary_10pct, (bool, np.bool_))

    kpss_res = kpss_test(white_noise)
    assert isinstance(kpss_res.is_stationary_1pct, (bool, np.bool_))
    assert isinstance(kpss_res.is_stationary_10pct, (bool, np.bool_))
