# ============================================================
# Hyperparameter Tuning with Optuna & Nested Validation (Phase 22)
# ============================================================
"""
Production Bayesian hyperparameter optimization engine using Optuna.

=============================================================================
CORE DESIGN PRINCIPLE: NESTED WALK-FORWARD VALIDATION (PREVENTING LEAKAGE)
-----------------------------------------------------------------------------
A critical, ubiquitous failure mode in quantitative machine learning is tuning
hyperparameters on the same cross-validation splits used to report final performance,
or on a single static train/val split.

1. Why standard K-Fold / Single Split tuning fails:
   - A single split tunes hyperparameters to whatever idiosyncrasy or noise regime
     dominated that particular test window.
   - Shuffled K-Fold leaks future information across boundaries (violating causality).

2. Why NESTED Validation is required:
   To report a statistically honest, leak-free estimate of model performance,
   hyperparameter search MUST be segregated into a nested structure:

   [ --------------------- TOTAL TIMELINE --------------------- ]
   [ === TUNING PERIOD (e.g. 70%) === ] [ = FINAL TEST PERIOD (30%) = ]
     |                                    |
     +--> Evaluated via Walk-Forward      +--> Completely HELD OUT.
          Cross-Validation folds inside        NEVER seen by Optuna.
          the Optuna objective loop.           Final model trained on Tuning
                                               Period is evaluated here ONCE.

   If Optuna were allowed to evaluate on the final test window, hyperparameter
   selection would overfit the final test period (hyperparameter snooping bias),
   rendering out-of-sample metrics deceptive.

3. Search Budget vs. Wall-Clock Trade-Off:
   In walk-forward cross-validation, each Optuna trial trains and evaluates N folds
   (e.g., 5 folds). A 50-trial study fits 250 gradient boosted models.
   Therefore, pruning/early stopping, constrained search ranges, and parallel execution
   are essential.
=============================================================================
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.walk_forward import WalkForwardSplitter
from src.utils.logger import get_logger

# Suppress excessively verbose optuna logs by default
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = get_logger(__name__)


# ============================================================
# 1. Nested Train/Tune/Test Splitter
# ============================================================


def split_tuning_and_final_test(
    X: pd.DataFrame,
    y: pd.Series,
    final_test_ratio: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Partition chronological time-series data into a Tuning Period and a Final Test Period.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix indexed by datetime.
    y : pd.Series
        Target series indexed by datetime.
    final_test_ratio : float, default 0.25
        Fraction of most recent observations reserved strictly as the unseen final test set.

    Returns
    -------
    X_tune : pd.DataFrame
        Features for the tuning period (used in walk-forward Optuna objective).
    X_test : pd.DataFrame
        Features for the final unseen evaluation period.
    y_tune : pd.Series
        Target for the tuning period.
    y_test : pd.Series
        Target for the final test period.
    """
    if not 0.05 <= final_test_ratio <= 0.50:
        raise ValueError(f"final_test_ratio must be between 0.05 and 0.50, got {final_test_ratio}.")

    common_idx = X.dropna().index.intersection(y.dropna().index)
    if len(common_idx) < 100:
        raise ValueError(f"Insufficient aligned samples ({len(common_idx)}) for nested split.")

    X_clean = X.loc[common_idx].sort_index()
    y_clean = y.loc[common_idx].sort_index()

    n = len(X_clean)
    split_idx = int(n * (1.0 - final_test_ratio))

    X_tune = X_clean.iloc[:split_idx].copy()
    X_test = X_clean.iloc[split_idx:].copy()
    y_tune = y_clean.iloc[:split_idx].copy()
    y_test = y_clean.iloc[split_idx:].copy()

    logger.info(
        "Nested timeline split completed: Tuning Set=%d bars (%s to %s), "
        "Final Unseen Test Set=%d bars (%s to %s)",
        len(X_tune),
        X_tune.index[0] if isinstance(X_tune.index, pd.DatetimeIndex) else 0,
        X_tune.index[-1] if isinstance(X_tune.index, pd.DatetimeIndex) else len(X_tune) - 1,
        len(X_test),
        X_test.index[0] if isinstance(X_test.index, pd.DatetimeIndex) else len(X_tune),
        X_test.index[-1] if isinstance(X_test.index, pd.DatetimeIndex) else n - 1,
    )

    return X_tune, X_test, y_tune, y_test


# ============================================================
# 2. Optuna Objective Builder
# ============================================================


class WalkForwardObjective:
    """Callable Optuna objective function evaluating a model via Walk-Forward Cross-Validation.

    Parameters
    ----------
    X_tune : pd.DataFrame
        Feature matrix for tuning period.
    y_tune : pd.Series
        Target series for tuning period.
    splitter : WalkForwardSplitter
        Walk-forward splitter operating on X_tune.
    backend : {'xgboost', 'lightgbm'}, default 'xgboost'
    task_type : {'classification', 'regression'}, default 'classification'
    metric : {'roc_auc', 'accuracy', 'logloss'}, default 'roc_auc'
    random_state : int, default 42
    """

    def __init__(
        self,
        X_tune: pd.DataFrame,
        y_tune: pd.Series,
        splitter: WalkForwardSplitter,
        backend: Literal["xgboost", "lightgbm"] = "xgboost",
        task_type: Literal["classification", "regression"] = "classification",
        metric: Literal["roc_auc", "accuracy", "logloss"] = "roc_auc",
        random_state: int = 42,
    ) -> None:
        self.X_tune = X_tune
        self.y_tune = y_tune
        self.splitter = splitter
        self.backend = backend
        self.task_type = task_type
        self.metric = metric
        self.random_state = random_state

    def __call__(self, trial: optuna.Trial) -> float:
        """Sample hyperparameters and compute mean out-of-fold validation score."""
        # 1. Sample hyperparameters tailored to low-SNR financial time series
        # In financial trading, shallow depth (1-4) and heavy regularizations prevent memorizing noise
        max_depth = trial.suggest_int("max_depth", 1, 4)
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.15, log=True)
        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=25)
        subsample = trial.suggest_float("subsample", 0.5, 0.95, step=0.05)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.3, 0.95, step=0.05)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        early_stopping_rounds = trial.suggest_categorical("early_stopping_rounds", [10, 15, 20])

        fold_scores: list[float] = []

        # 2. Iterate across walk-forward folds within the tuning period
        for train_idx, val_idx in self.splitter.split(self.X_tune, self.y_tune):
            X_tr, y_tr = self.X_tune.iloc[train_idx], self.y_tune.iloc[train_idx]
            X_va, y_va = self.X_tune.iloc[val_idx], self.y_tune.iloc[val_idx]

            model = GradientBoostingModel(
                backend=self.backend,
                task_type=self.task_type,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                early_stopping_rounds=early_stopping_rounds,
                auto_balance_classes=True,
                random_state=self.random_state,
            )

            model.fit(X_tr, y_tr)

            if self.task_type == "classification":
                probs = model.predict_proba(X_va)
                p1 = probs[:, 1]
                preds = model.predict(X_va)

                if self.metric == "roc_auc":
                    if len(np.unique(y_va)) > 1:
                        score = float(roc_auc_score(y_va, p1))
                    else:
                        score = 0.5
                elif self.metric == "logloss":
                    # For Optuna maximize: return negative logloss
                    score = -float(log_loss(y_va, p1, eps=1e-7))
                else:  # accuracy
                    score = float(np.mean(preds == y_va))
            else:
                preds = model.predict(X_va)
                # Negative RMSE for maximization
                score = -float(np.sqrt(np.mean((preds - y_va) ** 2)))

            fold_scores.append(score)

        return float(np.mean(fold_scores))


# ============================================================
# 3. Production Hyperparameter Tuner
# ============================================================


class HyperparameterTuner:
    """Production Optuna tuner utilizing walk-forward cross-validation and nested evaluation.

    Parameters
    ----------
    study_name : str
        Unique identifier for the study (used for storage and reports).
    backend : {'xgboost', 'lightgbm'}, default 'xgboost'
    task_type : {'classification', 'regression'}, default 'classification'
    metric : {'roc_auc', 'accuracy', 'logloss'}, default 'roc_auc'
    n_trials : int, default 50
        Optimization budget.
    timeout : float, optional
        Maximum seconds allowed for tuning.
    storage_dir : Path or str, default 'models/tuning_studies'
    random_state : int, default 42
    """

    def __init__(
        self,
        study_name: str = "gbm_quant_tuning",
        backend: Literal["xgboost", "lightgbm"] = "xgboost",
        task_type: Literal["classification", "regression"] = "classification",
        metric: Literal["roc_auc", "accuracy", "logloss"] = "roc_auc",
        n_trials: int = 50,
        timeout: float | None = None,
        storage_dir: Path | str = "models/tuning_studies",
        random_state: int = 42,
    ) -> None:
        self.study_name = study_name
        self.backend = backend
        self.task_type = task_type
        self.metric = metric
        self.n_trials = n_trials
        self.timeout = timeout
        self.storage_dir = Path(storage_dir)
        self.random_state = random_state

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.storage_path = self.storage_dir / f"{self.study_name}.db"
        self.storage_url = f"sqlite:///{self.storage_path.resolve()}"

        self.study: optuna.Study | None = None
        self.best_params: dict[str, Any] = {}
        self.best_value: float | None = None
        self.wall_clock_time: float = 0.0

    def fit(
        self,
        X_tune: pd.DataFrame,
        y_tune: pd.Series,
        splitter: WalkForwardSplitter,
        n_jobs: int = 1,
    ) -> HyperparameterTuner:
        """Execute the Optuna hyperparameter optimization study over walk-forward folds.

        Parameters
        ----------
        X_tune : pd.DataFrame
            Features across tuning timeline.
        y_tune : pd.Series
            Target across tuning timeline.
        splitter : WalkForwardSplitter
            Walk-forward splitter operating on X_tune.
        n_jobs : int, default 1
            Number of parallel workers. Default 1 to avoid process conflicts on Windows.

        Returns
        -------
        HyperparameterTuner
            Fitted instance with populated study and best_params.
        """
        logger.info(
            "Initializing Optuna study '%s' [backend=%s, metric=%s, budget=%d trials]...",
            self.study_name,
            self.backend,
            self.metric,
            self.n_trials,
        )

        direction = "maximize"
        sampler = optuna.samplers.TPESampler(seed=self.random_state)

        # Create or load study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            direction=direction,
            sampler=sampler,
            load_if_exists=True,
        )

        objective = WalkForwardObjective(
            X_tune=X_tune,
            y_tune=y_tune,
            splitter=splitter,
            backend=self.backend,
            task_type=self.task_type,
            metric=self.metric,
            random_state=self.random_state,
        )

        t_start = time.perf_counter()

        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=n_jobs,
            show_progress_bar=False,
        )

        self.wall_clock_time = time.perf_counter() - t_start
        self.best_params = self.study.best_params
        self.best_value = float(self.study.best_value)

        logger.info(
            "Optuna study '%s' completed in %.2fs (%.2fs/trial). Best %s=%.4f. Best params: %s",
            self.study_name,
            self.wall_clock_time,
            self.wall_clock_time / max(len(self.study.trials), 1),
            self.metric,
            self.best_value,
            self.best_params,
        )

        return self

    def build_best_model(self) -> GradientBoostingModel:
        """Construct a GradientBoostingModel instance initialized with the optimal hyperparameters."""
        if not self.best_params:
            raise ValueError("Tuner has not been fitted yet. Call fit() first.")

        params = self.best_params.copy()
        return GradientBoostingModel(
            backend=self.backend,
            task_type=self.task_type,
            n_estimators=int(params.get("n_estimators", 150)),
            max_depth=int(params.get("max_depth", 2)),
            learning_rate=float(params.get("learning_rate", 0.03)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_alpha=float(params.get("reg_alpha", 0.1)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            early_stopping_rounds=int(params.get("early_stopping_rounds", 15)),
            auto_balance_classes=True,
            random_state=self.random_state,
        )

    def plot_optimization_history(
        self,
        output_dir: Path | str = "reports/tuning",
        save_name: str | None = None,
    ) -> Path:
        """Generate and save the optimization objective history plot."""
        if self.study is None or len(self.study.trials) == 0:
            raise ValueError("No study trials available to plot.")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        filename = save_name or f"{self.study_name}_optimization_history.png"
        filepath = out_path / filename

        fig, ax = plt.subplots(figsize=(10, 5))
        trial_numbers = [t.number for t in self.study.trials if t.value is not None]
        trial_values = [t.value for t in self.study.trials if t.value is not None]

        # Compute running best
        running_best = np.maximum.accumulate(trial_values)

        ax.scatter(trial_numbers, trial_values, color="#1f77b4", alpha=0.6, label="Trial Value")
        ax.plot(
            trial_numbers,
            running_best,
            color="#d62728",
            linewidth=2.5,
            label=f"Best ({self.metric})",
        )
        ax.set_xlabel("Trial Number", fontsize=11)
        ax.set_ylabel(f"Mean OOF {self.metric.upper()}", fontsize=11)
        ax.set_title(
            f"Optuna Optimization History: {self.study_name} (Best = {self.best_value:.4f})",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)

        logger.info("Saved optimization history plot to %s", filepath)
        return filepath

    def plot_param_importances(
        self,
        output_dir: Path | str = "reports/tuning",
        save_name: str | None = None,
    ) -> Path | None:
        """Calculate and save hyperparameter importance chart."""
        if self.study is None or len(self.study.trials) < 10:
            logger.warning("Insufficient trials (<10) to evaluate parameter importance.")
            return None

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        filename = save_name or f"{self.study_name}_param_importances.png"
        filepath = out_path / filename

        try:
            importances = optuna.importance.get_param_importances(self.study)
        except Exception as e:
            logger.warning("Could not calculate parameter importances: %s", e)
            return None

        params = list(importances.keys())
        values = list(importances.values())

        fig, ax = plt.subplots(figsize=(9, 5))
        y_pos = np.arange(len(params))
        ax.barh(y_pos, values, color="#2ca02c", alpha=0.85, edgecolor="black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params, fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score (Mean Variance Reduction)", fontsize=11)
        ax.set_title(
            f"Hyperparameter Importances: {self.study_name}", fontsize=12, fontweight="bold"
        )
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)

        logger.info("Saved parameter importance plot to %s", filepath)
        return filepath
