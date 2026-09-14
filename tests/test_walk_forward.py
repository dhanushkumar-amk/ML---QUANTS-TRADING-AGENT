# ============================================================
# Unit Tests — Walk-Forward Validation Framework (Phase 20)
# ============================================================
"""
Tests for WalkForwardSplitter, purge/embargo gap logic, expanding vs. rolling modes,
scikit-learn compatibility, and split summary/visualization.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.models.walk_forward import (
    WalkForwardSplitter,
    evaluate_walk_forward,
    plot_walk_forward_splits,
)


@pytest.fixture
def synthetic_time_series_data() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic time-series dataset of 600 trading days with DatetimeIndex."""
    np.random.seed(42)
    n = 600
    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    y = pd.Series((x1 + 0.5 * x2 > 0).astype(float), index=dates, name="target")

    df_feats = pd.DataFrame({"f1": x1, "f2": x2}, index=dates)
    return df_feats, y


# ============================================================
# 1. Boundary & Embargo Logic Tests
# ============================================================


def test_walk_forward_expanding_mode(synthetic_time_series_data):
    """Verify expanding window mode anchors train start at 0 and grows train partition."""
    X, y = synthetic_time_series_data
    splitter = WalkForwardSplitter(
        n_splits=4,
        window_type="expanding",
        embargo_bars=10,
        min_train_size=200,
    )

    splits = list(splitter.split(X, y))
    assert len(splits) == 4

    prev_train_len = 0
    for _fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        # 1. Expanding window: train starts at index 0 for all folds
        assert train_idx[0] == 0

        # 2. Train size grows strictly monotonically
        assert len(train_idx) > prev_train_len
        prev_train_len = len(train_idx)

        # 3. No overlap between train and test
        assert len(set(train_idx).intersection(set(test_idx))) == 0

        # 4. Strict chronological ordering
        assert train_idx[-1] < test_idx[0]

        # 5. Embargo gap respected: test_start - train_end == embargo_bars
        assert test_idx[0] - train_idx[-1] == 10 + 1


def test_walk_forward_rolling_mode(synthetic_time_series_data):
    """Verify rolling window mode maintains fixed training window and shifts train start forward."""
    X, y = synthetic_time_series_data
    fixed_train_size = 200
    splitter = WalkForwardSplitter(
        n_splits=4,
        window_type="rolling",
        train_size=fixed_train_size,
        embargo_bars=5,
        min_train_size=150,
    )

    splits = list(splitter.split(X, y))
    assert len(splits) == 4

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        # 1. Fixed train length
        assert len(train_idx) == fixed_train_size

        # 2. In rolling mode, after the first fold, train_start must advance forward
        if fold_idx > 1:
            prev_train_idx = splits[fold_idx - 2][0]
            assert train_idx[0] > prev_train_idx[0]

        # 3. No overlap
        assert len(set(train_idx).intersection(set(test_idx))) == 0

        # 4. Embargo gap respected
        assert test_idx[0] - train_idx[-1] == 5 + 1


def test_walk_forward_no_embargo(synthetic_time_series_data):
    """Verify when embargo=0, test begins immediately on the next bar."""
    X, y = synthetic_time_series_data
    splitter = WalkForwardSplitter(
        n_splits=3,
        window_type="expanding",
        embargo_bars=0,
        min_train_size=200,
    )

    splits = list(splitter.split(X, y))
    for train_idx, test_idx in splits:
        assert test_idx[0] == train_idx[-1] + 1


# ============================================================
# 2. Scikit-Learn Compatibility Tests
# ============================================================


def test_sklearn_cross_val_score_compatibility(synthetic_time_series_data):
    """Verify WalkForwardSplitter can be passed directly to sklearn.model_selection tools."""
    X, y = synthetic_time_series_data
    splitter = WalkForwardSplitter(n_splits=3, min_train_size=200, embargo_bars=5)

    model = LogisticRegression(C=1.0)
    scores = cross_val_score(model, X.values, y.values.astype(int), cv=splitter)

    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


# ============================================================
# 3. Split Summary & Visualization Tests
# ============================================================


def test_split_summary_table(synthetic_time_series_data):
    """Verify split_summary produces a clean metadata DataFrame with timestamps."""
    X, _ = synthetic_time_series_data
    splitter = WalkForwardSplitter(n_splits=4, min_train_size=200, embargo_bars=5)

    summary_df = splitter.split_summary(X)
    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 4
    assert "train_start_date" in summary_df.columns
    assert "test_end_date" in summary_df.columns
    assert "embargo_bars" in summary_df.columns


def test_plot_walk_forward_splits(synthetic_time_series_data):
    """Verify plotting function executes without error and returns Figure."""
    X, _ = synthetic_time_series_data
    splitter = WalkForwardSplitter(n_splits=3, min_train_size=200, embargo_bars=10)

    fig = plot_walk_forward_splits(splitter, X, title="Test Splits")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ============================================================
# 4. Evaluation Runner & Error Handling Tests
# ============================================================


def test_evaluate_walk_forward_runner(synthetic_time_series_data):
    """Verify evaluate_walk_forward computes fold metrics and collects OOF predictions."""
    X, y = synthetic_time_series_data
    splitter = WalkForwardSplitter(n_splits=3, min_train_size=200, embargo_bars=5)

    def eval_fn(y_true, y_pred, y_prob):
        acc = float(np.mean(y_true == y_pred))
        return {"accuracy": acc}

    def model_factory():
        return LogisticRegression(C=1.0)

    fold_df, oof_df = evaluate_walk_forward(
        model_factory, splitter, X, y, eval_fn, model_name="LogReg"
    )

    assert isinstance(fold_df, pd.DataFrame)
    assert len(fold_df) == 3
    assert "accuracy" in fold_df.columns

    assert isinstance(oof_df, pd.DataFrame)
    assert "y_true" in oof_df.columns
    assert "y_pred" in oof_df.columns
    assert len(oof_df) > 0


def test_walk_forward_invalid_inputs():
    """Verify validation on impossible split configurations."""
    with pytest.raises(ValueError, match="n_splits must be >= 1"):
        WalkForwardSplitter(n_splits=0)

    with pytest.raises(ValueError, match="embargo_bars must be >= 0"):
        WalkForwardSplitter(embargo_bars=-1)

    with pytest.raises(ValueError, match="window_type must be"):
        WalkForwardSplitter(window_type="invalid")

    with pytest.raises(ValueError, match="insufficient"):
        # Sample too small for min_train_size
        s = WalkForwardSplitter(min_train_size=500)
        list(s.split(np.zeros(100)))
