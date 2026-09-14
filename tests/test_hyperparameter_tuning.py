# ============================================================
# Unit Tests for Hyperparameter Tuning (Phase 22)
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.hyperparameter_tuning import (
    HyperparameterTuner,
    split_tuning_and_final_test,
)
from src.models.walk_forward import WalkForwardSplitter


@pytest.fixture
def synthetic_tuning_dataset():
    """Generate a clean synthetic dataset for testing hyperparameter tuning."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    f1 = np.random.normal(0, 1, n)
    f2 = np.random.normal(0, 1, n)
    f3 = 0.5 * f1 + np.random.normal(0, 0.5, n)

    logits = 0.8 * f1 - 0.5 * f2
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs > 0.5).astype(int)

    X = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3}, index=dates)
    y_series = pd.Series(y, index=dates, name="target")
    return X, y_series


class TestNestedSplitter:
    """Test the nested train/tune/test timeline partitioning logic."""

    def test_split_tuning_and_final_test_proportions(self, synthetic_tuning_dataset):
        """Confirm tuning and test periods are strictly chronological without overlap."""
        X, y = synthetic_tuning_dataset
        X_tune, X_test, y_tune, y_test = split_tuning_and_final_test(X, y, final_test_ratio=0.25)

        assert len(X_tune) + len(X_test) == len(X)
        assert len(y_tune) + len(y_test) == len(y)
        assert len(X_test) == 50
        assert len(X_tune) == 150

        # Verify strict temporal ordering: tuning end < test start
        assert X_tune.index[-1] < X_test.index[0]
        assert y_tune.index[-1] < y_test.index[0]

        # Verify no index overlap
        assert len(X_tune.index.intersection(X_test.index)) == 0

    def test_split_invalid_ratio(self, synthetic_tuning_dataset):
        """Confirm invalid final_test_ratio raises ValueError."""
        X, y = synthetic_tuning_dataset
        with pytest.raises(ValueError, match="final_test_ratio must be between"):
            split_tuning_and_final_test(X, y, final_test_ratio=0.80)


class TestHyperparameterTunerExecution:
    """Test Optuna walk-forward tuning loop end-to-end with low trial budget."""

    def test_tuner_fit_and_build_model(self, synthetic_tuning_dataset, tmp_path):
        """Test that Optuna study runs end-to-end and returns best model."""
        X, y = synthetic_tuning_dataset
        X_tune, X_test, y_tune, y_test = split_tuning_and_final_test(X, y, final_test_ratio=0.30)

        splitter = WalkForwardSplitter(
            n_splits=2,
            min_train_size=60,
            test_size=30,
            embargo_bars=3,
        )

        tuner = HyperparameterTuner(
            study_name="test_ci_study",
            backend="xgboost",
            task_type="classification",
            metric="roc_auc",
            n_trials=3,  # Small trial budget for fast CI test
            storage_dir=tmp_path / "tuning",
            random_state=42,
        )

        tuner.fit(X_tune, y_tune, splitter=splitter, n_jobs=1)

        assert tuner.study is not None
        assert len(tuner.study.trials) == 3
        assert tuner.best_value is not None
        assert "max_depth" in tuner.best_params
        assert "learning_rate" in tuner.best_params
        assert tuner.wall_clock_time > 0.0

        # Build best model and evaluate on unseen final test set
        best_model = tuner.build_best_model()
        assert isinstance(best_model, GradientBoostingModel)

        best_model.fit(X_tune, y_tune)
        preds = best_model.predict(X_test)
        probs = best_model.predict_proba(X_test)

        assert len(preds) == len(X_test)
        assert probs.shape == (len(X_test), 2)

    def test_tuner_plots(self, synthetic_tuning_dataset, tmp_path):
        """Test generation of optimization history and parameter importance plots."""
        X, y = synthetic_tuning_dataset
        X_tune, _, y_tune, _ = split_tuning_and_final_test(X, y, final_test_ratio=0.30)
        splitter = WalkForwardSplitter(n_splits=2, min_train_size=60, test_size=30)

        tuner = HyperparameterTuner(
            study_name="test_plot_study",
            n_trials=3,
            storage_dir=tmp_path / "tuning",
            random_state=42,
        )
        tuner.fit(X_tune, y_tune, splitter=splitter, n_jobs=1)

        # Plot optimization history
        hist_path = tuner.plot_optimization_history(output_dir=tmp_path / "reports")
        assert hist_path.exists()
