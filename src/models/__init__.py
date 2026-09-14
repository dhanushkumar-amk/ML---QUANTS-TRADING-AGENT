# src.models — model training, evaluation, registry
"""Model training, evaluation, hyperparameter tuning, and model registry."""

from src.models.baseline_model import (
    BaselineClassifier,
    NaivePersistenceModel,
    evaluate_classification,
    run_baseline_comparison,
    temporal_train_test_split,
)
from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.walk_forward import (
    WalkForwardSplitter,
    evaluate_walk_forward,
    plot_walk_forward_splits,
)

__all__ = [
    "temporal_train_test_split",
    "NaivePersistenceModel",
    "BaselineClassifier",
    "evaluate_classification",
    "run_baseline_comparison",
    "WalkForwardSplitter",
    "plot_walk_forward_splits",
    "evaluate_walk_forward",
    "GradientBoostingModel",
]
