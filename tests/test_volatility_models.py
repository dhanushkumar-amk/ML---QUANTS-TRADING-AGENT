# ============================================================
# Unit Tests — GARCH Volatility Models (Phase 14)
# ============================================================
"""
Tests for GARCH(1,1), GJR-GARCH, EGARCH, parameter recovery on synthetic DGPs,
residual diagnostics, forecasting, and feature extraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.volatility_models import (
    GARCHVolatilityFeatureExtractor,
    fit_egarch,
    fit_garch,
    fit_gjr_garch,
    forecast_volatility,
    select_best_volatility_model,
)


def simulate_garch_11(
    n: int = 3000,
    omega: float = 0.05,
    alpha: float = 0.10,
    beta: float = 0.82,
    seed: int = 42,
) -> np.ndarray:
    """Simulate a synthetic GARCH(1,1) return series with known parameters."""
    rng = np.random.RandomState(seed)
    z = rng.randn(n + 500)
    eps = np.zeros(n + 500)
    sigma2 = np.zeros(n + 500)

    # Unconditional variance seed
    sigma2[0] = omega / (1.0 - alpha - beta)
    eps[0] = np.sqrt(sigma2[0]) * z[0]

    for t in range(1, n + 500):
        sigma2[t] = omega + alpha * (eps[t - 1] ** 2) + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * z[t]

    # Drop burn-in periods
    return eps[500:]


# ============================================================
# 1. Synthetic Parameter Recovery Test
# ============================================================


def test_synthetic_garch_parameter_recovery():
    """Verify GARCH(1,1) maximum likelihood estimator recovers known DGP parameters."""
    true_omega = 0.05
    true_alpha = 0.10
    true_beta = 0.82

    sim_returns = simulate_garch_11(
        n=3500,
        omega=true_omega,
        alpha=true_alpha,
        beta=true_beta,
        seed=123,
    )

    # Fit GARCH(1,1)
    res = fit_garch(sim_returns, p=1, q=1, rescale=False)

    est_alpha = res.params.get("alpha[1]", 0.0)
    est_beta = res.params.get("beta[1]", 0.0)
    persistence = res.persistence

    # Assert recovery within reasonable econometric sampling bounds (+/- 0.06 for alpha, +/- 0.08 for beta)
    assert (
        abs(est_alpha - true_alpha) < 0.06
    ), f"Alpha estimate {est_alpha:.4f} diverged from true {true_alpha}"
    assert (
        abs(est_beta - true_beta) < 0.08
    ), f"Beta estimate {est_beta:.4f} diverged from true {true_beta}"

    # Stationarity condition
    assert res.is_stationary is True
    assert persistence < 1.0
    assert abs(persistence - (true_alpha + true_beta)) < 0.05


# ============================================================
# 2. Stationarity & Residual Diagnostics Tests
# ============================================================


def test_garch_residual_diagnostics():
    """Verify that standardized residuals have no remaining ARCH effects (clustering absorbed)."""
    sim_returns = simulate_garch_11(n=2000, seed=999)
    res = fit_garch(sim_returns, p=1, q=1)

    assert len(res.standardized_residuals) == len(sim_returns)
    assert len(res.conditional_volatility) == len(sim_returns)
    assert (res.conditional_volatility > 0).all()

    # Re-running ARCH-LM on standardized residuals should yield p > 0.01 (no remaining clustering)
    assert (
        res.arch_lm_pvalue > 0.01
    ), f"Expected no remaining ARCH effects, got p-value {res.arch_lm_pvalue:.4f}"
    assert res.remaining_arch_effects is False


# ============================================================
# 3. Asymmetric Volatility Models (GJR-GARCH & EGARCH)
# ============================================================


def test_gjr_garch_fit():
    """Verify GJR-GARCH fits and estimates asymmetry parameter gamma."""
    sim_returns = simulate_garch_11(n=1000, seed=77)
    res = fit_gjr_garch(sim_returns, p=1, o=1, q=1)

    assert res.model_name == "GJR-GARCH(1,1,1)"
    assert "gamma[1]" in res.params
    assert "alpha[1]" in res.params
    assert "beta[1]" in res.params
    assert res.is_stationary is True
    assert (res.conditional_volatility > 0).all()


def test_egarch_fit():
    """Verify EGARCH fits and computes log-variance parameters."""
    sim_returns = simulate_garch_11(n=1000, seed=88)
    res = fit_egarch(sim_returns, p=1, o=1, q=1)

    assert res.model_name == "EGARCH(1,1,1)"
    assert "gamma[1]" in res.params
    assert "beta[1]" in res.params
    assert np.isfinite(res.aic)
    assert np.isfinite(res.bic)
    assert (res.conditional_volatility > 0).all()


# ============================================================
# 4. Model Selection & Forecasting Tests
# ============================================================


def test_select_best_volatility_model():
    """Verify model comparison table and best model selection via AIC."""
    sim_returns = simulate_garch_11(n=1000, seed=42)
    best_res, comp_df = select_best_volatility_model(sim_returns, criterion="aic")

    assert len(comp_df) == 3
    assert set(comp_df["model"]) == {"GARCH(1,1)", "GJR-GARCH(1,1,1)", "EGARCH(1,1,1)"}
    # Best model AIC must equal minimum AIC in comparison table
    assert np.isclose(best_res.aic, comp_df["aic"].min())
    assert best_res.model_name == comp_df.iloc[0]["model"]


def test_forecast_volatility():
    """Verify N-step ahead volatility forecasting returns valid term structure."""
    sim_returns = simulate_garch_11(n=1000, seed=55)
    res = fit_garch(sim_returns)

    forecast_5 = forecast_volatility(res, horizon=5)
    assert len(forecast_5) == 5
    assert (forecast_5 > 0).all()
    assert list(forecast_5.index) == ["h_1", "h_2", "h_3", "h_4", "h_5"]


# ============================================================
# 5. Extractor & Error Handling Tests
# ============================================================


def test_garch_volatility_feature_extractor():
    """Verify GARCHVolatilityFeatureExtractor generates expected columns."""
    n = 150
    rng = np.random.RandomState(42)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    df = pd.DataFrame({"close": prices})

    extractor = GARCHVolatilityFeatureExtractor(window=70, refit_frequency=25)
    out = extractor.transform(df, append=True)

    assert "garch_vol" in out.columns
    assert "garch_vol_annualized" in out.columns
    assert len(out) == n

    # Values after warmup should be positive
    valid_vol = out["garch_vol"].dropna()
    assert len(valid_vol) > 0
    assert (valid_vol > 0).all()


def test_short_series_raises():
    """Series with fewer than 30 returns must raise ValueError."""
    short_df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    with pytest.raises(ValueError, match="too short for GARCH modeling"):
        fit_garch(short_df)
