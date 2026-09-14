# ============================================================
# Unit Tests — Gradient Boosting Models (Phase 21)
# ============================================================
"""
Tests for GradientBoostingModel (XGBoost/LightGBM), automated class balancing,
early stopping, prediction calibration, and model artifact persistence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.gradient_boosting_model import GradientBoostingModel


@pytest.fixture
def synthetic_gbm_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic dataset with 200 samples and nonlinear target dependency."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(0, 1, n)

    # Non-linear interaction: target positive if x1*x2 > 0 or x3 > 0.5
    latent = (x1 * x2) + 0.8 * x3 + np.random.normal(0, 0.2, n)
    y = pd.Series((latent > 0).astype(float), index=dates, name="target")

    df_feats = pd.DataFrame({"feat_1": x1, "feat_2": x2, "feat_3": x3}, index=dates)
    return df_feats, y


@pytest.fixture
def synthetic_gbm_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic continuous return dataset."""
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    y = pd.Series(0.05 * x1 - 0.03 * x2 + np.random.normal(0, 0.01, n), index=dates, name="returns")

    df_feats = pd.DataFrame({"f1": x1, "f2": x2}, index=dates)
    return df_feats, y


# ============================================================
# 1. Classification & Regression Training Tests
# ============================================================


def test_xgboost_classification_fit_and_predict(synthetic_gbm_classification_data):
    """Verify XGBoost classification trains, predicts discrete labels and valid probabilities."""
    X, y = synthetic_gbm_classification_data
    split_idx = 140
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]

    model = GradientBoostingModel(
        backend="xgboost",
        task_type="classification",
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
    )
    model.fit(X_train, y_train)

    assert model.is_fitted_
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})
    assert proba.shape == (len(X_test), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    # Feature importances check
    imp = model.feature_importances
    assert isinstance(imp, pd.Series)
    assert len(imp) == 3
    assert set(imp.index) == set(X.columns)


def test_xgboost_regression_fit_and_predict(synthetic_gbm_regression_data):
    """Verify XGBoost regression outputs continuous return estimates."""
    X, y = synthetic_gbm_regression_data
    split_idx = 100
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]

    model = GradientBoostingModel(
        backend="xgboost",
        task_type="regression",
        n_estimators=40,
        max_depth=2,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert preds.dtype == np.float32 or preds.dtype == np.float64

    # Regression should not support predict_proba
    with pytest.raises(ValueError, match="predict_proba is only supported"):
        model.predict_proba(X_test)


# ============================================================
# 2. Automated Class Balancing Tests
# ============================================================


def test_automated_class_balancing():
    """Verify scale_pos_weight is calculated from empirical label distribution."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
    # Severe imbalance: 85 zeros and 15 ones
    y = pd.Series([0.0] * 85 + [1.0] * 15)

    model = GradientBoostingModel(
        backend="xgboost",
        auto_balance_classes=True,
        early_stopping_rounds=0,
    )
    model.fit(X, y)

    # Expected scale_pos_weight = 85 / 15 = 5.667
    expected_w = 85.0 / 15.0
    assert np.isclose(model.scale_pos_weight_, expected_w, rtol=1e-3)


# ============================================================
# 3. Model Artifact Persistence (Save / Load) Tests
# ============================================================


def test_model_artifact_save_and_load(synthetic_gbm_classification_data, tmp_path):
    """Verify model save and load round-trip perfectly reproduces predictions."""
    X, y = synthetic_gbm_classification_data
    split_idx = 140
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]

    orig_model = GradientBoostingModel(
        backend="xgboost",
        task_type="classification",
        n_estimators=30,
        max_depth=3,
        random_state=42,
    )
    orig_model.fit(X_train, y_train)

    orig_preds = orig_model.predict(X_test)
    orig_proba = orig_model.predict_proba(X_test)

    # Save artifact to temporary directory
    saved_file = orig_model.save(
        artifact_dir=tmp_path,
        model_name="test_xgb_model",
        metrics={"test_acc": 0.85},
    )
    assert saved_file.exists()
    meta_file = tmp_path / "test_xgb_model.meta.json"
    assert meta_file.exists()

    # Load back using classmethod
    loaded_model = GradientBoostingModel.load(saved_file)

    assert loaded_model.is_fitted_
    assert loaded_model.feature_names_ == orig_model.feature_names_
    assert loaded_model.backend == orig_model.backend

    loaded_preds = loaded_model.predict(X_test)
    loaded_proba = loaded_model.predict_proba(X_test)

    # Assert bit-for-bit prediction agreement
    np.testing.assert_array_equal(orig_preds, loaded_preds)
    np.testing.assert_allclose(orig_proba, loaded_proba, rtol=1e-6, atol=1e-6)


def test_unfitted_model_raises():
    """Verify calling predict or save on an unfitted model raises error."""
    model = GradientBoostingModel()
    with pytest.raises(RuntimeError, match="must be fitted"):
        model.predict(pd.DataFrame({"x": [1]}))

    with pytest.raises(RuntimeError, match="Cannot save"):
        model.save()
