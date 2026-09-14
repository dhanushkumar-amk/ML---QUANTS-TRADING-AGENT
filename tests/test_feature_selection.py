# ============================================================
# Unit Tests — Feature Selection & Target Engineering (Phase 18)
# ============================================================
"""
Tests for feature selection, univariate screening, multicollinearity (VIF),
tree importance ranking, and prediction target construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_selection import (
    compute_multicollinearity,
    compute_tree_importance,
    compute_univariate_metrics,
    feature_selection_report,
    make_target,
)


@pytest.fixture
def synthetic_price_df() -> pd.DataFrame:
    """Deterministic synthetic price series for target testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    # Monotonically increasing prices with periodic dips
    prices = 100.0 + np.cumsum(np.random.normal(0.5, 1.0, size=100))
    return pd.DataFrame({"close": prices, "volume": 1000.0}, index=dates)


@pytest.fixture
def synthetic_feature_matrix_and_target() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic feature dataset with known causal, collinear, and noise features."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    # Feature 1: strong signal
    x_signal = np.random.normal(0, 1, n)
    # Feature 2: redundant collinear clone of x_signal
    x_collinear = x_signal + np.random.normal(0, 0.05, n)
    # Feature 3: moderate signal
    x_moderate = np.random.normal(0, 1, n)
    # Feature 4: pure noise
    x_noise = np.random.normal(0, 1, n)

    # Target: direction determined primarily by x_signal and partially by x_moderate
    latent = 2.0 * x_signal + 0.8 * x_moderate + np.random.normal(0, 0.3, n)
    target = pd.Series((latent > 0).astype(float), index=dates, name="target_dir_1d")

    df_feats = pd.DataFrame(
        {
            "signal_feat": x_signal,
            "collinear_feat": x_collinear,
            "moderate_feat": x_moderate,
            "noise_feat": x_noise,
        },
        index=dates,
    )
    return df_feats, target


# ============================================================
# Target Construction Tests
# ============================================================


def test_make_target_classification(synthetic_price_df: pd.DataFrame):
    """Verify make_target produces binary classification labels with strictly unobserved tail."""
    target = make_target(synthetic_price_df, horizon=1, task_type="classification")

    assert isinstance(target, pd.Series)
    assert target.name == "target_dir_1d"
    assert len(target) == len(synthetic_price_df)

    # The final horizon=1 bar must be NaN
    assert pd.isna(target.iloc[-1])
    # All preceding bars must be non-NaN and binary {0.0, 1.0}
    valid_targets = target.iloc[:-1]
    assert not valid_targets.isna().any()
    assert set(valid_targets.unique()).issubset({0.0, 1.0})

    # Mathematical identity check:
    # row 0 return is (P_1 - P_0) / P_0
    p0 = synthetic_price_df["close"].iloc[0]
    p1 = synthetic_price_df["close"].iloc[1]
    expected_dir = 1.0 if (p1 - p0) > 0 else 0.0
    assert target.iloc[0] == expected_dir


def test_make_target_regression(synthetic_price_df: pd.DataFrame):
    """Verify make_target produces forward percentage returns for regression."""
    horizon = 3
    target = make_target(synthetic_price_df, horizon=horizon, task_type="regression")

    assert target.name == f"target_ret_{horizon}d"
    # Final `horizon` bars must be NaN
    assert target.iloc[-horizon:].isna().all()
    assert not target.iloc[:-horizon].isna().any()

    # Exact return check on first observation
    p0 = synthetic_price_df["close"].iloc[0]
    ph = synthetic_price_df["close"].iloc[horizon]
    expected_ret = (ph - p0) / p0
    assert np.isclose(target.iloc[0], expected_ret, rtol=1e-7)


def test_make_target_invalid_inputs(synthetic_price_df: pd.DataFrame):
    """Test error handling on bad parameters."""
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        make_target(synthetic_price_df, horizon=0)

    with pytest.raises(ValueError, match="cannot be empty"):
        make_target(pd.DataFrame())

    with pytest.raises(ValueError, match="not found"):
        make_target(synthetic_price_df, price_col="non_existent")

    with pytest.raises(ValueError, match="Invalid task_type"):
        make_target(synthetic_price_df, task_type="unsupported")


# ============================================================
# Univariate Filtering Tests
# ============================================================


def test_compute_univariate_metrics(synthetic_feature_matrix_and_target):
    """Verify correlation and mutual information rank signal features above pure noise."""
    X, y = synthetic_feature_matrix_and_target
    uni_df = compute_univariate_metrics(X, y, task_type="classification")

    assert isinstance(uni_df, pd.DataFrame)
    assert "spearman_corr" in uni_df.columns
    assert "mutual_info" in uni_df.columns

    # Causal signal feature must have significantly higher MI and abs correlation than pure noise
    assert uni_df.loc["signal_feat", "mutual_info"] > uni_df.loc["noise_feat", "mutual_info"]
    assert (
        uni_df.loc["signal_feat", "abs_spearman_corr"]
        > uni_df.loc["noise_feat", "abs_spearman_corr"]
    )
    assert uni_df.loc["signal_feat", "abs_spearman_corr"] > 0.4
    assert uni_df.loc["noise_feat", "abs_spearman_corr"] < 0.25


# ============================================================
# Multicollinearity & VIF Tests
# ============================================================


def test_compute_multicollinearity(synthetic_feature_matrix_and_target):
    """Verify VIF and collinear pair detection correctly flags redundant duplicates."""
    X, _ = synthetic_feature_matrix_and_target
    vif_df, collinear_pairs = compute_multicollinearity(X, corr_threshold=0.85)

    assert isinstance(vif_df, pd.DataFrame)
    assert "vif" in vif_df.columns

    # Pairwise collinear detection: (signal_feat, collinear_feat) must be identified
    assert len(collinear_pairs) >= 1
    top_pair = collinear_pairs[0]
    pair_features = {top_pair[0], top_pair[1]}
    assert pair_features == {"signal_feat", "collinear_feat"}
    assert top_pair[2] > 0.90

    # VIF for collinear features must be elevated (> 10.0)
    assert vif_df.loc["signal_feat", "vif"] > 10.0
    assert vif_df.loc["collinear_feat", "vif"] > 10.0
    # Pure noise orthogonal feature must have low VIF (~1.0)
    assert vif_df.loc["noise_feat", "vif"] < 2.0


# ============================================================
# Tree Feature Importance Tests
# ============================================================


def test_compute_tree_importance(synthetic_feature_matrix_and_target):
    """Verify tree MDI importance accurately attributes top weight to true signal."""
    X, y = synthetic_feature_matrix_and_target
    imp_s = compute_tree_importance(X, y, task_type="classification")

    assert isinstance(imp_s, pd.Series)
    assert np.isclose(imp_s.sum(), 1.0, atol=1e-5)

    # Signal or collinear clone should capture the vast majority of importance
    top_feature = imp_s.idxmax()
    assert top_feature in {"signal_feat", "collinear_feat"}
    assert imp_s["noise_feat"] < imp_s[top_feature]


# ============================================================
# Full Feature Selection Report & Shortlist Tests
# ============================================================


def test_feature_selection_report(synthetic_feature_matrix_and_target):
    """Verify feature selection report outputs ranked metrics and prunes collinear duplicates."""
    X, y = synthetic_feature_matrix_and_target
    report_df, shortlist = feature_selection_report(
        X,
        y,
        task_type="classification",
        top_k=3,
        corr_threshold=0.85,
    )

    assert isinstance(report_df, pd.DataFrame)
    assert isinstance(shortlist, list)

    # Shortlist length must respect top_k
    assert len(shortlist) <= 3
    # Either signal_feat or collinear_feat should be selected, but NOT BOTH (collinearity pruning)
    assert ("signal_feat" in shortlist) ^ ("collinear_feat" in shortlist) or (
        "signal_feat" in shortlist and "collinear_feat" not in shortlist
    )
    # The pruned collinear duplicate should have an explicit pruning reason
    pruned_dup = "collinear_feat" if "signal_feat" in shortlist else "signal_feat"
    assert "Redundant with" in report_df.loc[pruned_dup, "prune_reason"]
