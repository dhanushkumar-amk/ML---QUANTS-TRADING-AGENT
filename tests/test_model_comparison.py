# tests/test_model_comparison.py
"""Unit tests for model comparison framework and statistical significance testing."""

import numpy as np
import pandas as pd
import pytest

from src.models.model_comparison import (
    ModelComparisonHarness,
    bootstrap_sharpe_difference,
    diebold_mariano_test,
    profile_model_latency,
)


class DummyModel:
    """Mock model with configurable prediction behavior."""

    def __init__(self, proba_pos: float = 0.6) -> None:
        self.proba_pos = proba_pos

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DummyModel":
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.full((len(X), 2), [1.0 - self.proba_pos, self.proba_pos])
        return p

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def test_diebold_mariano_identical_forecasts():
    """Identical forecast errors should yield DM stat = 0.0 and p-value = 1.0."""
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0] * 5)
    p1 = np.full(50, 0.6)
    p2 = np.full(50, 0.6)

    stat, p_val = diebold_mariano_test(y_true, p1, p2, loss_type="brier")
    assert np.isclose(stat, 0.0)
    assert np.isclose(p_val, 1.0)


def test_diebold_mariano_known_significant_difference():
    """Model 1 with nearly perfect predictions vs Model 2 with random bad predictions."""
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.5, size=200)
    # Model 1 has low error with variance across samples
    p1 = np.clip(np.where(y_true == 1, 0.90, 0.10) + np.random.normal(0, 0.05, 200), 0.01, 0.99)
    # Model 2 has high error with variance across samples
    p2 = np.clip(np.where(y_true == 1, 0.15, 0.85) + np.random.normal(0, 0.05, 200), 0.01, 0.99)

    stat, p_val = diebold_mariano_test(y_true, p1, p2, loss_type="brier")
    # Model 1 has much lower loss than Model 2, so d_t = L(e1) - L(e2) < 0
    assert stat < -5.0
    assert p_val < 0.001


def test_diebold_mariano_logloss():
    """Diebold-Mariano test supports logloss loss type."""
    y_true = np.array([1, 0, 1, 0] * 10)
    p1 = np.full(40, 0.7)
    p2 = np.full(40, 0.5)

    stat, p_val = diebold_mariano_test(y_true, p1, p2, loss_type="logloss")
    assert isinstance(stat, float)
    assert 0.0 <= p_val <= 1.0


def test_diebold_mariano_invalid_inputs():
    """Test input validation for Diebold-Mariano."""
    y_true = np.array([1, 0, 1] * 5)
    p1 = np.array([0.8, 0.2] * 7)
    p2 = np.array([0.5, 0.5] * 7 + [0.5])

    with pytest.raises(ValueError, match="Length mismatch"):
        diebold_mariano_test(y_true, p1, p2)

    with pytest.raises(ValueError, match="Insufficient samples"):
        diebold_mariano_test(np.array([1, 0]), np.array([0.5, 0.5]), np.array([0.5, 0.5]))

    with pytest.raises(ValueError, match="Unknown loss_type"):
        diebold_mariano_test(
            np.ones(15), np.full(15, 0.5), np.full(15, 0.5), loss_type="invalid"  # type: ignore
        )


def test_bootstrap_sharpe_difference_identical():
    """Identical return series should have zero Sharpe difference and p-value = 1.0."""
    rets = np.random.normal(0.001, 0.01, size=100)
    diff, (ci_lower, ci_upper), p_val = bootstrap_sharpe_difference(
        rets, rets, n_bootstrap=100, random_state=42
    )

    assert np.isclose(diff, 0.0, atol=1e-6)
    assert np.isclose(ci_lower, 0.0, atol=1e-6)
    assert np.isclose(ci_upper, 0.0, atol=1e-6)
    assert np.isclose(p_val, 1.0)


def test_bootstrap_sharpe_difference_known_difference():
    """Consistently positive returns should significantly beat zero returns."""
    np.random.seed(42)
    rets_a = np.random.normal(0.005, 0.005, size=250)  # strongly positive
    rets_b = np.random.normal(-0.005, 0.005, size=250)  # strongly negative

    diff, (ci_lower, ci_upper), p_val = bootstrap_sharpe_difference(
        rets_a, rets_b, n_bootstrap=200, random_state=42
    )

    assert diff > 0
    assert ci_lower > 0  # 95% CI is strictly positive
    assert p_val < 0.05


def test_profile_model_latency():
    """Latency profiling should return positive training and inference times."""
    model = DummyModel()
    X = np.random.randn(50, 4)
    y = np.random.binomial(1, 0.5, size=50)

    train_ms, infer_us = profile_model_latency(model, X, y, n_inference_runs=5)
    assert train_ms >= 0.0
    assert infer_us > 0.0


def test_model_comparison_harness():
    """Test full tournament harness with mock models."""
    harness = ModelComparisonHarness(baseline_model_name="XGBoost_Baseline")

    N = 100
    y_true = np.random.binomial(1, 0.5, size=N)
    asset_returns = np.random.normal(0.0005, 0.01, size=N)

    # Register baseline
    p_base = np.random.uniform(0.45, 0.65, size=N)
    harness.register_model_results(
        model_name="XGBoost_Baseline",
        y_true=y_true,
        y_pred_proba=p_base,
        asset_returns=asset_returns,
        train_time_ms=120.0,
        inference_latency_us=25.0,
    )

    # Register superior model
    p_superior = np.where(y_true == 1, 0.85, 0.15)
    harness.register_model_results(
        model_name="Superior_Ensemble",
        y_true=y_true,
        y_pred_proba=p_superior,
        asset_returns=asset_returns,
        train_time_ms=250.0,
        inference_latency_us=60.0,
    )

    table = harness.compile_tournament_table()
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 2
    assert "Model" in table.columns
    assert "Accuracy" in table.columns
    assert "Sharpe" in table.columns
    assert "DM Stat" in table.columns
    assert "Sharpe CI (95%)" in table.columns

    # Baseline row checks
    baseline_row = table[table["Model"] == "XGBoost_Baseline"].iloc[0]
    assert baseline_row["DM Sig"] == "Baseline"
    assert baseline_row["Delta Sharpe"] == 0.0

    # Superior row checks
    superior_row = table[table["Model"] == "Superior_Ensemble"].iloc[0]
    assert superior_row["Accuracy"] > baseline_row["Accuracy"]
