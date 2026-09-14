# src.models — model training, evaluation, registry
"""Model training, evaluation, hyperparameter tuning, and model registry."""

from src.models.baseline_model import (
    BaselineClassifier,
    NaivePersistenceModel,
    evaluate_classification,
    run_baseline_comparison,
    temporal_train_test_split,
)
from src.models.financial_metrics import (
    calculate_calmar_ratio,
    calculate_drawdown_series,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_win_rate_and_profit_factor,
    compute_financial_metrics,
    convert_predictions_to_strategy_returns,
    financial_evaluation_report,
)
from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.hyperparameter_tuning import (
    HyperparameterTuner,
    WalkForwardObjective,
    split_tuning_and_final_test,
)
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
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_drawdown_series",
    "calculate_max_drawdown",
    "calculate_calmar_ratio",
    "calculate_win_rate_and_profit_factor",
    "convert_predictions_to_strategy_returns",
    "compute_financial_metrics",
    "financial_evaluation_report",
    "split_tuning_and_final_test",
    "WalkForwardObjective",
    "HyperparameterTuner",
]
