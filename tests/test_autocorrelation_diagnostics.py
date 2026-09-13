# ============================================================
# Unit Tests for Autocorrelation Diagnostics (Phase 11)
# ============================================================
"""
Tests for first-moment autocorrelation, Lo-MacKinlay Variance Ratio tests,
run length & streak analysis, and multi-horizon classification.

Covers:
  1. Synthetic pure Random Walk (no significant structure).
  2. Synthetic Mean-Reverting process (VR < 1, negative ACF).
  3. Synthetic Trending / Momentum process (VR > 1, positive ACF).
  4. ACF and PACF computation.
  5. Run length and Wald-Wolfowitz runs test.
  6. Multi-horizon analysis and batch summary DataFrame.
  7. Edge cases: NaNs/Infs, short series, zero variance.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.autocorrelation_diagnostics import (
    AutocorrelationReport,
    LjungBoxRawResult,
    RunLengthResult,
    VarianceRatioResult,
    autocorrelation_report,
    compute_acf_pacf,
    ljung_box_raw,
    multi_asset_autocorrelation_summary,
    multi_horizon_autocorrelation_analysis,
    run_length_analysis,
    variance_ratio_test,
)

# ============================================================
# Simulation Fixtures
# ============================================================


@pytest.fixture
def rng():
    """Deterministic random number generator."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def random_walk_prices() -> pd.Series:
    """Pure geometric random walk with i.i.d. Gaussian increments."""
    rng = np.random.default_rng(seed=123)
    n = 1000
    innovations = rng.normal(loc=0.0, scale=0.01, size=n)
    log_p = np.cumsum(innovations)
    prices = 100.0 * np.exp(log_p)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates, name="RW_price")


@pytest.fixture
def mean_reverting_series(rng) -> pd.Series:
    """Synthetic mean-reverting process (negative AR(1) returns): r_t = -0.4 r_{t-1} + eps_t."""
    n = 1000
    r = np.zeros(n)
    eps = rng.normal(scale=0.01, size=n)
    for t in range(1, n):
        r[t] = -0.4 * r[t - 1] + eps[t]

    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(r, index=dates, name="MR_returns")


@pytest.fixture
def momentum_series(rng) -> pd.Series:
    """Synthetic momentum / trending process (positive AR(1) returns): r_t = +0.4 r_{t-1} + eps_t."""
    n = 1000
    r = np.zeros(n)
    eps = rng.normal(scale=0.01, size=n)
    for t in range(1, n):
        r[t] = 0.4 * r[t - 1] + eps[t]

    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(r, index=dates, name="MOM_returns")


# ============================================================
# 1. ACF & PACF Tests
# ============================================================


def test_compute_acf_pacf(random_walk_prices):
    """Verify compute_acf_pacf returns correct array shapes and bounds."""
    ret = random_walk_prices.pct_change().dropna()
    acf_vals, acf_conf, pacf_vals, pacf_conf = compute_acf_pacf(ret, nlags=10, alpha=0.05)

    assert len(acf_vals) == 11
    assert acf_conf.shape == (11, 2)
    assert len(pacf_vals) == 11
    assert pacf_conf.shape == (11, 2)
    assert np.isclose(acf_vals[0], 1.0)
    assert np.isclose(pacf_vals[0], 1.0)


# ============================================================
# 2. Pure Random Walk Tests
# ============================================================


def test_random_walk_variance_ratio(random_walk_prices):
    """Random walk must have Variance Ratios near 1.0 and fail to reject null."""
    vr_results = variance_ratio_test(random_walk_prices, k_lags=(2, 5, 10))
    assert 2 in vr_results
    assert 5 in vr_results
    assert 10 in vr_results

    for _k, res in vr_results.items():
        assert isinstance(res, VarianceRatioResult)
        assert 0.80 <= res.vr <= 1.25
        assert abs(res.z_hetero) < 1.96
        assert res.pval_hetero >= 0.05
        assert res.interpretation == "random-walk"


def test_random_walk_autocorrelation_report(random_walk_prices):
    """Combined report on random walk should conclude 'no significant structure'."""
    ret = random_walk_prices.pct_change().dropna()
    report = autocorrelation_report(ret, prices=random_walk_prices, name="RW")
    assert isinstance(report, AutocorrelationReport)
    assert report.classification == "no significant structure (random walk)"
    assert "indistinguishable from a martingale difference" in report.explanation


# ============================================================
# 3. Mean-Reverting Process Tests
# ============================================================


def test_mean_reverting_diagnostics(mean_reverting_series):
    """Mean-reverting series must show VR < 1.0 and negative autocorrelation."""
    # 1. ACF
    acf_vals, _, _, _ = compute_acf_pacf(mean_reverting_series, nlags=5)
    assert acf_vals[1] < -0.20

    # 2. Ljung-Box test
    lb_res = ljung_box_raw(mean_reverting_series, lags=[5, 10])
    assert isinstance(lb_res, LjungBoxRawResult)
    assert lb_res.p_values[5] < 0.05

    # 3. Variance Ratio
    vr_res = variance_ratio_test(mean_reverting_series, k_lags=[2, 5], is_returns=True)
    assert vr_res[2].vr < 1.0
    assert vr_res[5].vr < 1.0
    assert vr_res[5].z_hetero < -1.96
    assert vr_res[5].interpretation == "mean-reversion"

    # 4. Report
    report = autocorrelation_report(mean_reverting_series, name="MR", horizon_days=1)
    assert report.classification == "mean-reversion-dominant"
    assert "mean-reversion detected" in report.explanation


# ============================================================
# 4. Trending / Momentum Process Tests
# ============================================================


def test_momentum_diagnostics(momentum_series):
    """Trending series must show VR > 1.0 and positive autocorrelation."""
    # 1. ACF
    acf_vals, _, _, _ = compute_acf_pacf(momentum_series, nlags=5)
    assert acf_vals[1] > 0.20

    # 2. Variance Ratio
    vr_res = variance_ratio_test(momentum_series, k_lags=[2, 5], is_returns=True)
    assert vr_res[2].vr > 1.0
    assert vr_res[5].vr > 1.0
    assert vr_res[5].z_hetero > 1.96
    assert vr_res[5].interpretation == "momentum"

    # 3. Report
    report = autocorrelation_report(momentum_series, name="MOM", horizon_days=1)
    assert report.classification == "momentum-dominant"
    assert "momentum detected" in report.explanation


# ============================================================
# 5. Run Length & Streak Analysis Tests
# ============================================================


def test_run_length_analysis(random_walk_prices, momentum_series):
    """Verify run length metrics and streak calculations."""
    ret_rw = random_walk_prices.pct_change().dropna()
    rl_rw = run_length_analysis(ret_rw)
    assert isinstance(rl_rw, RunLengthResult)
    assert rl_rw.total_runs > 0
    assert rl_rw.avg_positive_streak > 1.0
    assert rl_rw.avg_negative_streak > 1.0
    assert abs(rl_rw.z_stat) < 2.5

    # Momentum series should have fewer runs (clustering of positive/negative signs)
    rl_mom = run_length_analysis(momentum_series)
    assert rl_mom.z_stat < -1.96
    assert "momentum tendency" in rl_mom.streak_bias


# ============================================================
# 6. Multi-Horizon and Multi-Asset Batch Summary
# ============================================================


def test_multi_horizon_analysis(random_walk_prices):
    """Verify multi-horizon analysis runs across 1d, 5d, 20d horizons."""
    horizon_results = multi_horizon_autocorrelation_analysis(
        random_walk_prices,
        name="RW",
        horizons=(1, 5, 20),
    )
    assert set(horizon_results.keys()) == {1, 5, 20}
    for h, rep in horizon_results.items():
        assert isinstance(rep, AutocorrelationReport)
        assert rep.horizon_days == h


def test_multi_asset_summary(random_walk_prices):
    """Verify batch evaluation summarizes multiple assets into DataFrame."""
    prices_dict = {
        "Asset1": random_walk_prices,
        "Asset2": random_walk_prices * 1.5,
    }
    df_summary = multi_asset_autocorrelation_summary(prices_dict, horizons=(1, 5))
    assert isinstance(df_summary, pd.DataFrame)
    assert ("Asset1", 1) in df_summary.index
    assert ("Asset2", 5) in df_summary.index
    assert "vr_5" in df_summary.columns
    assert "classification" in df_summary.columns


def test_multi_asset_summary_error_handling(random_walk_prices):
    """Verify invalid short series records error without crashing."""
    prices_dict = {
        "Valid": random_walk_prices,
        "TooShort": pd.Series([100.0, 101.0]),
    }
    df_summary = multi_asset_autocorrelation_summary(prices_dict, horizons=(1, 5))
    assert ("TooShort", 1) in df_summary.index
    assert "ERROR" in df_summary.loc[("TooShort", 1), "classification"]


# ============================================================
# 7. Edge Cases & Error Handling
# ============================================================


def test_too_short_series_raises():
    """Series with fewer than 20 observations raises ValueError."""
    s = pd.Series([0.01] * 10)
    with pytest.raises(ValueError, match="too short"):
        compute_acf_pacf(s)
    with pytest.raises(ValueError, match="too short"):
        ljung_box_raw(s)
    with pytest.raises(ValueError, match="too short"):
        variance_ratio_test(s)


def test_clean_series_nans_and_infs(rng):
    """NaNs and Infs are filtered cleanly before analysis."""
    raw = rng.normal(0, 0.01, size=200)
    raw[10] = np.nan
    raw[25] = np.inf
    raw[50] = -np.inf
    s = pd.Series(raw)
    rep = autocorrelation_report(s, name="dirty")
    assert rep.classification in (
        "momentum-dominant",
        "mean-reversion-dominant",
        "no significant structure (random walk)",
    )


def test_zero_variance_raises():
    """Flat constant series with zero variance raises ValueError."""
    flat = pd.Series([100.0] * 50)
    with pytest.raises(ValueError, match="variance is effectively zero"):
        variance_ratio_test(flat)
