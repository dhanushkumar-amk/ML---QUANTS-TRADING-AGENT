# ============================================================
# Unit Tests for Volatility Clustering Analysis (Phase 10)
# ============================================================
"""
Tests for volatility proxies, ACF, Ljung-Box test, Engle ARCH-LM test,
and comprehensive volatility clustering consensus reporting.

Covers:
  1. Synthetic GARCH(1,1) process with strong known volatility clustering.
  2. Pure Gaussian white noise process with no volatility clustering.
  3. Discriminative power (GARCH vs. White Noise).
  4. Proxy computation (squared and absolute returns).
  5. ACF calculation and confidence bounds.
  6. Batch evaluation and error handling.
  7. Edge cases: short series, NaNs, Infs.
  8. Visualizations (plot_acf_squared_returns, plot_volatility_regimes).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.features.eda_utils import plot_acf_squared_returns, plot_volatility_regimes
from src.features.volatility_analysis import (
    ARCHLMResult,
    LjungBoxResult,
    VolatilityClusteringReport,
    arch_lm_test,
    batch_volatility_clustering,
    compute_acf_series,
    compute_volatility_proxies,
    ljung_box_test,
    volatility_clustering_report,
)

# ============================================================
# Simulation Fixtures
# ============================================================


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def white_noise_returns(rng) -> pd.Series:
    """Pure Gaussian white noise returns: X_t ~ N(0, 0.01^2).

    No volatility clustering, constant unconditional variance.
    """
    data = rng.normal(loc=0.0, scale=0.01, size=1000)
    dates = pd.date_range("2020-01-01", periods=1000, freq="B")
    return pd.Series(data, index=dates, name="white_noise")


@pytest.fixture
def garch11_returns(rng) -> pd.Series:
    """Simulated GARCH(1,1) process with heavy volatility clustering.

    Model:
      r_t = sigma_t * z_t,  z_t ~ N(0, 1)
      sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
      Parameters: omega=1e-5, alpha=0.12, beta=0.85 (alpha+beta = 0.97, high persistence).
    """
    n_total = 1500
    burn_in = 500
    omega = 1e-5
    alpha = 0.12
    beta = 0.85

    r = np.zeros(n_total)
    sigma2 = np.zeros(n_total)

    # Initial unconditional variance
    sigma2[0] = omega / (1.0 - alpha - beta)
    z = rng.standard_normal(n_total)
    r[0] = np.sqrt(sigma2[0]) * z[0]

    for t in range(1, n_total):
        sigma2[t] = omega + alpha * (r[t - 1] ** 2) + beta * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * z[t]

    clean_r = r[burn_in:]
    dates = pd.date_range("2020-01-01", periods=len(clean_r), freq="B")
    return pd.Series(clean_r, index=dates, name="garch_sim")


# ============================================================
# 1. Volatility Proxies Tests
# ============================================================


def test_compute_volatility_proxies(white_noise_returns):
    """Verify squared returns and absolute returns columns and mathematical accuracy."""
    df_proxies = compute_volatility_proxies(white_noise_returns)
    assert isinstance(df_proxies, pd.DataFrame)
    assert list(df_proxies.columns) == ["raw_returns", "squared_returns", "abs_returns"]
    assert len(df_proxies) == len(white_noise_returns)

    np.testing.assert_allclose(df_proxies["squared_returns"], white_noise_returns**2)
    np.testing.assert_allclose(df_proxies["abs_returns"], np.abs(white_noise_returns))


# ============================================================
# 2. Autocorrelation Function (ACF) Tests
# ============================================================


def test_compute_acf_series(white_noise_returns):
    """Verify compute_acf_series outputs array and 95% confidence intervals."""
    acf_vals, confint = compute_acf_series(white_noise_returns, nlags=15, alpha=0.05)
    assert len(acf_vals) == 16  # lag 0 through 15
    assert confint.shape == (16, 2)
    assert np.isclose(acf_vals[0], 1.0)


# ============================================================
# 3. GARCH(1,1) Volatility Clustering Tests
# ============================================================


def test_garch_arch_lm_test(garch11_returns):
    """GARCH(1,1) must decisively reject ARCH-LM null hypothesis of homoskedasticity."""
    result = arch_lm_test(garch11_returns, lags=10)
    assert isinstance(result, ARCHLMResult)
    assert result.lags == 10
    assert result.lm_stat > 15.0
    assert result.lm_pvalue < 0.05
    assert result.has_arch_effects(alpha=0.05) is True


def test_garch_ljung_box_test(garch11_returns):
    """Ljung-Box on squared GARCH returns must reject independence."""
    sq_returns = garch11_returns**2
    result = ljung_box_test(sq_returns, lags=[10, 20])
    assert isinstance(result, LjungBoxResult)
    assert 10 in result.p_values
    assert 20 in result.p_values
    assert result.p_values[10] < 0.05
    assert result.p_values[20] < 0.05
    assert result.is_autocorrelated(lag=10, alpha=0.05) is True
    assert result.is_autocorrelated(alpha=0.05) is True


def test_garch_clustering_report(garch11_returns):
    """Combined report on GARCH process must conclude 'confirmed'."""
    report = volatility_clustering_report(garch11_returns, name="garch", lags=10, alpha=0.05)
    assert isinstance(report, VolatilityClusteringReport)
    assert report.is_clustering_confirmed is True
    assert report.conclusion == "confirmed"
    assert "Strong statistical evidence" in report.explanation

    d = report.to_dict()
    assert d["is_clustering_confirmed"] is True
    assert d["conclusion"] == "confirmed"
    assert d["arch_lm_pvalue"] < 0.05
    assert d["ljung_box_pvalue"] < 0.05


# ============================================================
# 4. Pure White Noise (No Clustering) Tests
# ============================================================


def test_white_noise_arch_lm_test(white_noise_returns):
    """Pure white noise must fail to reject ARCH-LM null hypothesis."""
    result = arch_lm_test(white_noise_returns, lags=10)
    assert result.lm_pvalue >= 0.05
    assert result.has_arch_effects(alpha=0.05) is False


def test_white_noise_ljung_box_test(white_noise_returns):
    """Ljung-Box on squared white noise must fail to reject independence."""
    sq_returns = white_noise_returns**2
    result = ljung_box_test(sq_returns, lags=[10])
    assert result.p_values[10] >= 0.05
    assert result.is_autocorrelated(lag=10, alpha=0.05) is False


def test_white_noise_clustering_report(white_noise_returns):
    """Combined report on white noise must conclude 'absent'."""
    report = volatility_clustering_report(
        white_noise_returns, name="white_noise", lags=10, alpha=0.05
    )
    assert report.is_clustering_confirmed is False
    assert report.conclusion == "absent"
    assert "No statistical evidence" in report.explanation


# ============================================================
# 5. Discrimination Test (GARCH vs. White Noise)
# ============================================================


def test_discrimination_garch_vs_white_noise(garch11_returns, white_noise_returns):
    """ARCH-LM stat and squared ACF for GARCH must substantially exceed White Noise."""
    garch_lm = arch_lm_test(garch11_returns, lags=10)
    wn_lm = arch_lm_test(white_noise_returns, lags=10)
    assert garch_lm.lm_stat > wn_lm.lm_stat * 2.0

    garch_acf, _ = compute_acf_series(garch11_returns**2, nlags=5)
    wn_acf, _ = compute_acf_series(white_noise_returns**2, nlags=5)
    # Lag 1 autocorrelation of squared returns must be much higher for GARCH
    assert garch_acf[1] > wn_acf[1] + 0.05


# ============================================================
# 6. Batch Evaluation Utility
# ============================================================


def test_batch_volatility_clustering(garch11_returns, white_noise_returns):
    """Verify batch evaluation summarizes multiple series into a clean DataFrame."""
    data = {
        "GARCH": garch11_returns,
        "WN": white_noise_returns,
    }
    summary_df = batch_volatility_clustering(data, lags=10, alpha=0.05)
    assert isinstance(summary_df, pd.DataFrame)
    assert set(summary_df.index) == {"GARCH", "WN"}
    assert summary_df.loc["GARCH", "clustering_confirmed"]
    assert not summary_df.loc["WN", "clustering_confirmed"]


def test_batch_volatility_clustering_error_handling(white_noise_returns):
    """Verify invalid series records an error row without crashing batch evaluation."""
    data = {
        "valid": white_noise_returns,
        "too_short": pd.Series([0.01, 0.02]),
    }
    summary_df = batch_volatility_clustering(data, lags=10, alpha=0.05)
    assert "valid" in summary_df.index
    assert "too_short" in summary_df.index
    assert "ERROR" in summary_df.loc["too_short", "conclusion"]


# ============================================================
# 7. Edge Cases & Cleaning
# ============================================================


def test_series_with_nans_and_infs(rng):
    """Series with NaNs and infinities is cleaned prior to testing."""
    vals = rng.normal(0, 0.01, size=200)
    vals[10] = np.nan
    vals[20] = np.inf
    vals[30] = -np.inf
    s = pd.Series(vals)
    report = volatility_clustering_report(s, lags=5)
    assert report.conclusion in ("confirmed", "absent", "inconclusive")


def test_too_short_series_raises():
    """Series with fewer than 15 observations raises ValueError."""
    s = pd.Series([0.01] * 5)
    with pytest.raises(ValueError, match="too short"):
        arch_lm_test(s)
    with pytest.raises(ValueError, match="too short"):
        ljung_box_test(s)


# ============================================================
# 8. Visualization Functions Tests
# ============================================================


def test_plot_acf_squared_returns(garch11_returns, tmp_path):
    """Verify plot_acf_squared_returns creates dual-axis figure and saves image."""
    price_series = 100.0 * np.exp(garch11_returns.cumsum())
    df = pd.DataFrame({"close": price_series}, index=garch11_returns.index)

    out_file = tmp_path / "test_acf.png"
    fig, axes = plot_acf_squared_returns("GARCH_SIM", nlags=10, df=df, save_path=out_file)
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 2
    assert out_file.exists()
    plt.close(fig)


def test_plot_volatility_regimes(garch11_returns, tmp_path):
    """Verify plot_volatility_regimes creates figure with shaded regimes."""
    price_series = 100.0 * np.exp(garch11_returns.cumsum())
    df = pd.DataFrame({"close": price_series}, index=garch11_returns.index)

    out_file = tmp_path / "test_regimes.png"
    fig, ax = plot_volatility_regimes("GARCH_SIM", window=15, df=df, save_path=out_file)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert out_file.exists()
    plt.close(fig)
