# ============================================================
# Unit Tests — Baseline Models & Benchmarks (Phase 19)
# ============================================================
"""
Tests for baseline models (Naive persistence, Logistic Regression, Decision Tree),
temporal train/test splitting, and evaluation metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baseline_model import (
    BaselineClassifier,
    NaivePersistenceModel,
    evaluate_classification,
    run_baseline_comparison,
    temporal_train_test_split,
)


@pytest.fixture
def synthetic_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Linearly separable synthetic dataset with DatetimeIndex."""
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    # Binary label with strong dependence on x1
    prob = 1.0 / (1.0 + np.exp(-(2.5 * x1 - 1.0 * x2)))
    y = pd.Series((prob > 0.5).astype(float), index=dates, name="target")

    df_feats = pd.DataFrame({"f1": x1, "f2": x2}, index=dates)
    return df_feats, y


# ============================================================
# Temporal Train/Test Split Tests
# ============================================================


def test_temporal_train_test_split(synthetic_classification_data):
    """Verify temporal train/test split maintains strict chronological order without shuffling."""
    X, y = synthetic_classification_data
    test_ratio = 0.25

    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_ratio=test_ratio)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert np.isclose(len(X_test) / len(X), test_ratio, atol=0.02)

    # Chronological integrity: all training timestamps strictly precede all test timestamps
    assert X_train.index.max() < X_test.index.min()
    assert y_train.index.max() < y_test.index.min()
    # No index overlap
    assert len(X_train.index.intersection(X_test.index)) == 0


def test_temporal_train_test_split_invalid():
    """Verify validation on invalid split parameters."""
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    df = pd.DataFrame({"x": range(60)}, index=dates)
    s = pd.Series(range(60), index=dates)

    with pytest.raises(ValueError, match="test_ratio must be strictly between 0 and 1"):
        temporal_train_test_split(df, s, test_ratio=1.5)

    with pytest.raises(ValueError, match="test_ratio must be strictly between 0 and 1"):
        temporal_train_test_split(df, s, test_ratio=0.0)


# ============================================================
# Naive Persistence Baseline Tests
# ============================================================


def test_naive_persistence_logic():
    """Verify naive persistence model predicts previous day's direction exactly."""
    dates_train = pd.date_range("2023-01-01", periods=3, freq="B")
    dates_test = pd.date_range("2023-01-06", periods=4, freq="B")

    # Train ends on 1.0
    y_train = pd.Series([0.0, 1.0, 1.0], index=dates_train)
    # Test ground truth
    y_test = pd.Series([0.0, 1.0, 1.0, 0.0], index=dates_test)

    X_train = pd.DataFrame({"f": [1, 2, 3]}, index=dates_train)
    X_test = pd.DataFrame({"f": [4, 5, 6, 7]}, index=dates_test)

    naive = NaivePersistenceModel()
    naive.fit(X_train, y_train)

    preds = naive.predict(X_test, y_actual=y_test)
    # Expected predictions:
    # Bar 0: last train target = 1.0
    # Bar 1: previous test target = y_test[0] = 0.0
    # Bar 2: previous test target = y_test[1] = 1.0
    # Bar 3: previous test target = y_test[2] = 1.0
    expected_preds = np.array([1.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(preds, expected_preds)

    # Probabilities must match predicted classes
    proba = naive.predict_proba(X_test, y_actual=y_test)
    assert proba.shape == (4, 2)
    for i, p in enumerate(expected_preds):
        assert proba[i, int(p)] == 1.0


def test_naive_unfitted_raises():
    """Verify unfitted model raises RuntimeError."""
    naive = NaivePersistenceModel()
    with pytest.raises(RuntimeError, match="must be fitted"):
        naive.predict(pd.DataFrame({"x": [1, 2]}))


# ============================================================
# Baseline Classifiers (Logistic & Decision Tree) Tests
# ============================================================


def test_baseline_logistic_regression(synthetic_classification_data):
    """Verify Logistic Regression trains and produces valid predictions and probabilities."""
    X, y = synthetic_classification_data
    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_ratio=0.3)

    model = BaselineClassifier(model_type="logistic_regression", C=1.0)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})
    assert proba.shape == (len(X_test), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    # Direction signal is strong in synthetic data, so test accuracy should be high (> 0.70)
    metrics = evaluate_classification(y_test, preds, proba)
    assert metrics["accuracy"] > 0.70
    assert metrics["roc_auc"] > 0.75


def test_baseline_decision_tree_interpretability(synthetic_classification_data):
    """Verify Decision Tree fits and provides human-readable decision rules."""
    X, y = synthetic_classification_data
    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_ratio=0.3)

    dt = BaselineClassifier(model_type="decision_tree", max_depth=3)
    dt.fit(X_train, y_train)

    preds = dt.predict(X_test)
    assert len(preds) == len(X_test)

    rules = dt.get_rules()
    assert isinstance(rules, str)
    assert "f1" in rules or "f2" in rules
    assert "<=" in rules or ">" in rules


def test_baseline_unfitted_raises():
    """Verify unfitted classifier raises error."""
    clf = BaselineClassifier(model_type="logistic_regression")
    with pytest.raises(RuntimeError, match="must be fitted"):
        clf.predict(pd.DataFrame({"f1": [1]}))


# ============================================================
# Evaluation & Comparison Engine Tests
# ============================================================


def test_evaluate_classification_metrics():
    """Verify evaluate_classification computes accurate statistical metrics."""
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 0, 1, 1])
    # Probabilities giving realistic calibration
    y_prob = np.zeros((8, 2))
    y_prob[:, 1] = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.1, 0.7, 0.6])
    y_prob[:, 0] = 1.0 - y_prob[:, 1]

    metrics = evaluate_classification(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "brier_score" in metrics

    # Accuracy check: 6 correct out of 8 = 0.75
    assert np.isclose(metrics["accuracy"], 6.0 / 8.0)
    assert metrics["roc_auc"] > 0.80


def test_run_baseline_comparison(synthetic_classification_data):
    """Verify run_baseline_comparison compiles all 3 baselines with excess metrics."""
    X, y = synthetic_classification_data
    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_ratio=0.3)

    comp_df, fitted_models = run_baseline_comparison(X_train, X_test, y_train, y_test)

    assert isinstance(comp_df, pd.DataFrame)
    assert set(comp_df.index) == {"Naive Persistence", "Logistic Regression", "Decision Tree"}
    assert "excess_vs_naive" in comp_df.columns
    assert "accuracy" in comp_df.columns
    assert "roc_auc" in comp_df.columns

    # Check excess calculation
    naive_acc = comp_df.loc["Naive Persistence", "accuracy"]
    assert np.isclose(comp_df.loc["Naive Persistence", "excess_vs_naive"], 0.0)
    assert np.isclose(
        comp_df.loc["Logistic Regression", "excess_vs_naive"],
        comp_df.loc["Logistic Regression", "accuracy"] - naive_acc,
    )

    assert "naive" in fitted_models
    assert "logistic_regression" in fitted_models
    assert "decision_tree" in fitted_models
