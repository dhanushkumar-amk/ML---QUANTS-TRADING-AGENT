# ============================================================
# Baseline Models Benchmark Module (Phase 19)
# ============================================================
"""
Baseline model benchmarks for direction classification.

Establishes the minimal performance floor that all future complex architectures
(e.g., XGBoost in Phase 21, LSTMs/Transformers in Phase 26) must definitively beat
to justify increased parameter count and computational complexity.

Models implemented:
-------------------
1. Naive Persistence Baseline:
   Predicts tomorrow's market direction repeats today's direction:
   \\hat{Y}_t = Y_{t-1}.
   Serves as an indispensable quantitative reality check; many published
   academic and commercial trading models fail to outperform this trivial heuristic.

2. Logistic Regression:
   Linear baseline with L2 regularization for directional classification:
   P(Y_t = 1 | X_t) = \\sigma(W^T X_t + b).

3. Interpretable Decision Tree:
   Single shallow decision tree (depth <= 3) exposing explicit, human-readable
   decision rules and split thresholds across quantitative features.

NOTE ON FINANCIAL VS. ML METRICS:
---------------------------------
Evaluation in this module uses standard statistical ML metrics (Accuracy, Balanced
Accuracy, Precision, Recall, F1, ROC-AUC, Brier Score). Financial strategy metrics
(Annualized Sharpe Ratio, Maximum Drawdown, Profit Factor, Sortino) are deliberately
deferred to Phase 23 (Backtesting Engine). In quantitative trading, a 52% accuracy model
can be profitable with asymmetric payoffs, while a 60% accuracy model can blow up if
losses concentrate in tail events. Do not conflate ML accuracy with financial viability.

NOTE ON TEMPORAL VALIDATION:
----------------------------
The simple train/test split implemented here is a temporary simplification for Phase 19
benchmarking. Rigorous walk-forward validation (anchored/expanding and rolling window CV)
will be introduced in Phase 20.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier, export_text

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Temporal Train/Test Split (Temporary Simplified Split)
# ============================================================


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split time-series data chronologically without shuffling.

    TEMPORARY NOTICE:
    This chronological train/test split is a simplified partition for Phase 19 baseline
    benchmarking. Full walk-forward validation (expanding and rolling cross-validation)
    will be established in Phase 20.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix indexed by datetime.
    y : pd.Series
        Target series indexed by datetime.
    test_ratio : float, default 0.2
        Fraction of most recent observations reserved for testing.

    Returns
    -------
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    y_train : pd.Series
    y_test : pd.Series
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be strictly between 0 and 1, got {test_ratio}.")

    common_idx = X.dropna().index.intersection(y.dropna().index)
    if len(common_idx) < 50:
        raise ValueError(f"Insufficient aligned observations ({len(common_idx)}) for split.")

    X_aligned = X.loc[common_idx].sort_index()
    y_aligned = y.loc[common_idx].sort_index()

    n = len(X_aligned)
    split_idx = int(n * (1.0 - test_ratio))

    X_train = X_aligned.iloc[:split_idx].copy()
    X_test = X_aligned.iloc[split_idx:].copy()
    y_train = y_aligned.iloc[:split_idx].copy()
    y_test = y_aligned.iloc[split_idx:].copy()

    logger.info(
        "Temporal split completed: Train=%d (%s to %s), Test=%d (%s to %s)",
        len(X_train),
        X_train.index[0],
        X_train.index[-1],
        len(X_test),
        X_test.index[0],
        X_test.index[-1],
    )
    return X_train, X_test, y_train, y_test


# ============================================================
# 2. Naive Persistence Baseline Model
# ============================================================


class NaivePersistenceModel:
    """Predicts next period's direction is identical to the most recently observed direction.

    Heuristic: \\hat{Y}_t = Y_{t-1}.
    Essential honesty baseline: If ML models cannot beat naive persistence, they possess
    no predictive signal above autocorrelation inertia.
    """

    def __init__(self) -> None:
        self.last_train_target_: float | None = None
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NaivePersistenceModel:
        """Fit by storing the final observed training label."""
        if len(y) == 0:
            raise ValueError("Target series cannot be empty.")
        self.last_train_target_ = float(y.iloc[-1])
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, y_actual: pd.Series | None = None) -> np.ndarray:
        """Predict persistence over the test period.

        If `y_actual` is provided (the actual test labels), \\hat{Y}_t is the previous day's
        actual label Y_{t-1}. For t=0 of test, the last label from training is used.
        If `y_actual` is not provided, assumes persistence from training end.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict.")

        n = len(X)
        if n == 0:
            return np.array([], dtype=float)

        if y_actual is not None and len(y_actual) == n:
            # First prediction is last training value; subsequent predictions are y_actual shifted by 1
            preds = np.empty(n, dtype=float)
            preds[0] = self.last_train_target_
            preds[1:] = y_actual.values[:-1]
            return preds
        else:
            # Constant persistence
            return np.full(n, self.last_train_target_, dtype=float)

    def predict_proba(self, X: pd.DataFrame, y_actual: pd.Series | None = None) -> np.ndarray:
        """Return pseudo-probabilities for persistence (binary 0 or 1)."""
        preds = self.predict(X, y_actual=y_actual)
        proba = np.zeros((len(preds), 2), dtype=float)
        for i, p in enumerate(preds):
            if p == 1.0:
                proba[i, 1] = 1.0
            else:
                proba[i, 0] = 1.0
        return proba


# ============================================================
# 3. Simple Machine Learning Baselines
# ============================================================


class BaselineClassifier:
    """Unified wrapper for Logistic Regression and Decision Tree baselines.

    Parameters
    ----------
    model_type : {'logistic_regression', 'decision_tree'}
    max_depth : int, default 3
        Tree depth (only used for 'decision_tree').
    C : float, default 1.0
        Inverse regularization strength (only used for 'logistic_regression').
    random_state : int, default 42
    """

    def __init__(
        self,
        model_type: Literal["logistic_regression", "decision_tree"] = "logistic_regression",
        max_depth: int = 3,
        C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.max_depth = max_depth
        self.C = C
        self.random_state = random_state
        self.feature_names_: list[str] = []
        self.model_: Any = None
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaselineClassifier:
        """Fit baseline model strictly on training data."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if len(X) != len(y):
            raise ValueError(f"X ({len(X)}) and y ({len(y)}) lengths must match.")

        self.feature_names_ = list(X.columns)

        if self.model_type == "logistic_regression":
            self.model_ = LogisticRegression(
                C=self.C,
                solver="lbfgs",
                max_iter=1000,
                random_state=self.random_state,
            )
        elif self.model_type == "decision_tree":
            self.model_ = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=20,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown model_type: '{self.model_type}'.")

        self.model_.fit(X.values, y.values.astype(int))
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels (0 or 1)."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict.")
        return self.model_.predict(X[self.feature_names_].values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities [P(0), P(1)]."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict_proba.")
        return self.model_.predict_proba(X[self.feature_names_].values)

    def get_rules(self) -> str:
        """Export interpretable decision rules (only for decision_tree)."""
        if not self.is_fitted_ or self.model_type != "decision_tree":
            return "Decision rules are only available for fitted decision tree models."
        return export_text(self.model_, feature_names=self.feature_names_)


# ============================================================
# 4. Model Evaluation & Comparison Engine
# ============================================================


def evaluate_classification(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Calculate comprehensive machine learning metrics for directional classification.

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        True binary class labels (0 or 1).
    y_pred : np.ndarray
        Predicted binary class labels (0 or 1).
    y_prob : np.ndarray, optional
        Predicted probabilities array of shape (N, 2).

    Returns
    -------
    dict[str, float]
        Dictionary of diagnostic metrics:
        'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'brier_score'.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_t, y_p))
    bal_acc = float(balanced_accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    auc_score = np.nan
    brier = np.nan
    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                p1 = y_prob[:, 1]
            else:
                p1 = y_prob.ravel()
            if len(np.unique(y_t)) > 1:
                auc_score = float(roc_auc_score(y_t, p1))
            brier = float(brier_score_loss(y_t, p1))
        except Exception as e:
            logger.debug("AUC/Brier computation skipped: %s", e)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc_score,
        "brier_score": brier,
    }


def run_baseline_comparison(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    tree_max_depth: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train and evaluate Naive Persistence, Logistic Regression, and Decision Tree baselines.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training targets.
    y_test : pd.Series
        Test targets.
    tree_max_depth : int, default 3
    random_state : int, default 42

    Returns
    -------
    comparison_df : pd.DataFrame
        Table summarizing ML metrics and excess accuracy over naive persistence.
    fitted_models : dict[str, Any]
        Fitted model instances ('naive', 'logistic_regression', 'decision_tree').
    """
    logger.info("Executing baseline comparison suite on test sample of %d bars...", len(y_test))

    # 1. Naive Persistence Baseline
    naive = NaivePersistenceModel()
    naive.fit(X_train, y_train)
    pred_naive = naive.predict(X_test, y_actual=y_test)
    prob_naive = naive.predict_proba(X_test, y_actual=y_test)
    metrics_naive = evaluate_classification(y_test, pred_naive, prob_naive)

    # 2. Logistic Regression Baseline
    logit = BaselineClassifier(
        model_type="logistic_regression",
        C=1.0,
        random_state=random_state,
    )
    logit.fit(X_train, y_train)
    pred_logit = logit.predict(X_test)
    prob_logit = logit.predict_proba(X_test)
    metrics_logit = evaluate_classification(y_test, pred_logit, prob_logit)

    # 3. Decision Tree Baseline
    dt = BaselineClassifier(
        model_type="decision_tree",
        max_depth=tree_max_depth,
        random_state=random_state,
    )
    dt.fit(X_train, y_train)
    pred_dt = dt.predict(X_test)
    prob_dt = dt.predict_proba(X_test)
    metrics_dt = evaluate_classification(y_test, pred_dt, prob_dt)

    # Compile comparison DataFrame
    models_dict = {
        "Naive Persistence": metrics_naive,
        "Logistic Regression": metrics_logit,
        "Decision Tree": metrics_dt,
    }

    comparison_df = pd.DataFrame(models_dict).T
    naive_acc = metrics_naive["accuracy"]
    comparison_df["excess_vs_naive"] = comparison_df["accuracy"] - naive_acc

    fitted_models = {
        "naive": naive,
        "logistic_regression": logit,
        "decision_tree": dt,
    }

    return comparison_df, fitted_models
