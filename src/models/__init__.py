# src.models — model training, evaluation, registry
"""Model training, evaluation, hyperparameter tuning, model registry, and DL sequence models."""

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
from src.models.lstm_model import LSTMModel, NumPyLSTMNetwork
from src.models.model_interpretability import (
    TreeSHAPExplainer,
    cross_reference_shap_vs_univariate,
    plot_shap_dependence,
    plot_shap_summary,
    plot_shap_waterfall,
)
from src.models.sequence_data_prep import (
    SequenceDataLoader,
    SequenceDataset,
    SequenceNormalizer,
    create_sliding_sequences,
    plot_sequence_window_sanity_check,
    walk_forward_sequence_split,
)
from src.models.transformer_model import (
    NumPyTransformerNetwork,
    TransformerModel,
    plot_attention_weights,
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
    "TreeSHAPExplainer",
    "plot_shap_summary",
    "plot_shap_waterfall",
    "plot_shap_dependence",
    "cross_reference_shap_vs_univariate",
    "create_sliding_sequences",
    "SequenceNormalizer",
    "SequenceDataset",
    "SequenceDataLoader",
    "walk_forward_sequence_split",
    "plot_sequence_window_sanity_check",
    "LSTMModel",
    "NumPyLSTMNetwork",
    "TransformerModel",
    "NumPyTransformerNetwork",
    "plot_attention_weights",
]
