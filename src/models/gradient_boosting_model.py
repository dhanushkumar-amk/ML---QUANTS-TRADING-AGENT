# ============================================================
# Gradient Boosting Models (XGBoost / LightGBM) (Phase 21)
# ============================================================
"""
Production Gradient Boosting framework supporting XGBoost and LightGBM.

Features:
---------
1. Dual Backend Support:
   Configurable choice of XGBoost or LightGBM for directional classification
   or forward return regression.

2. Automated Class Balancing:
   Inspects empirical label distribution in the training fold rather than assuming
   balance. Computes scale_pos_weight = N_neg / N_pos dynamically to prevent
   majority-class bias in choppy or drifting market regimes.

3. Early Stopping via Temporal Holdout:
   Carves out the final temporal segment (e.g. 15%) of the training fold strictly
   in chronological sequence as an evaluation set. Halts boosting rounds when
   validation logloss fails to improve, preventing overfitting noisy market data.

4. Versioned Model Artifact Persistence:
   Saves model binaries alongside comprehensive metadata JSON (features used,
   hyperparameters, date ranges, best iteration, validation score) to models/artifacts/.
"""

from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import xgboost as xgb

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Optional LightGBM import with graceful fallback
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False


class GradientBoostingModel:
    """Production wrapper for XGBoost and LightGBM quantitative models.

    Parameters
    ----------
    backend : {'xgboost', 'lightgbm'}, default 'xgboost'
    task_type : {'classification', 'regression'}, default 'classification'
    n_estimators : int, default 200
    max_depth : int, default 3
        Shallow tree depth to prevent memorizing market noise.
    learning_rate : float, default 0.03
    subsample : float, default 0.8
    colsample_bytree : float, default 0.8
    reg_alpha : float, default 0.1
        L1 regularization.
    reg_lambda : float, default 1.0
        L2 regularization.
    early_stopping_rounds : int, default 20
    validation_fraction : float, default 0.15
        Temporal fraction at the end of the training partition used for early stopping.
    auto_balance_classes : bool, default True
        Automatically calculate scale_pos_weight from training labels.
    random_state : int, default 42
    """

    def __init__(
        self,
        backend: Literal["xgboost", "lightgbm"] = "xgboost",
        task_type: Literal["classification", "regression"] = "classification",
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 20,
        validation_fraction: float = 0.15,
        auto_balance_classes: bool = True,
        random_state: int = 42,
    ) -> None:
        if backend == "lightgbm" and not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM is not available on this platform; falling back to XGBoost.")
            backend = "xgboost"

        self.backend = backend
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.auto_balance_classes = auto_balance_classes
        self.random_state = random_state

        # Fitted attributes
        self.model_: Any = None
        self.feature_names_: list[str] = []
        self.scale_pos_weight_: float = 1.0
        self.best_iteration_: int | None = None
        self.best_score_: float | None = None
        self.training_start_date_: str | None = None
        self.training_end_date_: str | None = None
        self.is_fitted_: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> GradientBoostingModel:
        """Fit gradient boosting model with automated class balancing and early stopping."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if len(X) != len(y):
            raise ValueError(f"X ({len(X)}) and y ({len(y)}) lengths must match.")

        self.feature_names_ = list(X.columns)
        if isinstance(X.index, pd.DatetimeIndex):
            self.training_start_date_ = X.index[0].strftime("%Y-%m-%d")
            self.training_end_date_ = X.index[-1].strftime("%Y-%m-%d")

        # 1. Temporal Holdout Split for Early Stopping (if not provided externally)
        if X_val is None and self.early_stopping_rounds > 0 and len(X) > 40:
            split_idx = int(len(X) * (1.0 - self.validation_fraction))
            X_tr = X.iloc[:split_idx]
            y_tr = y.iloc[:split_idx]
            X_va = X.iloc[split_idx:]
            y_va = y.iloc[split_idx:]
        else:
            X_tr, y_tr = X, y
            X_va, y_va = X_val, y_val

        # 2. Automated Class Balancing
        if self.task_type == "classification" and self.auto_balance_classes:
            pos_count = int((y_tr == 1.0).sum())
            neg_count = int((y_tr == 0.0).sum())
            if pos_count > 0:
                self.scale_pos_weight_ = float(neg_count / pos_count)
            else:
                self.scale_pos_weight_ = 1.0
            logger.debug(
                "Training class balance: Pos=%d, Neg=%d -> scale_pos_weight=%.3f",
                pos_count,
                neg_count,
                self.scale_pos_weight_,
            )
        else:
            self.scale_pos_weight_ = 1.0

        # 3. Instantiate and Fit Backend
        if self.backend == "xgboost":
            self._fit_xgboost(X_tr, y_tr, X_va, y_va)
        elif self.backend == "lightgbm":
            self._fit_lightgbm(X_tr, y_tr, X_va, y_va)
        else:
            raise ValueError(f"Unsupported backend: '{self.backend}'.")

        self.is_fitted_ = True
        return self

    def _fit_xgboost(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame | None,
        y_va: pd.Series | None,
    ) -> None:
        """Internal fit for XGBoost backend."""
        common_params: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": -1,
        }

        eval_set = (
            [
                (
                    X_va.values,
                    y_va.values.astype(int) if self.task_type == "classification" else y_va.values,
                )
            ]
            if X_va is not None
            else None
        )

        if self.task_type == "classification":
            self.model_ = xgb.XGBClassifier(
                **common_params,
                scale_pos_weight=self.scale_pos_weight_,
                eval_metric="logloss",
                early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
            )
            self.model_.fit(
                X_tr.values,
                y_tr.values.astype(int),
                eval_set=eval_set,
                verbose=False,
            )
        else:
            self.model_ = xgb.XGBRegressor(
                **common_params,
                eval_metric="rmse",
                early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
            )
            self.model_.fit(
                X_tr.values,
                y_tr.values,
                eval_set=eval_set,
                verbose=False,
            )

        if hasattr(self.model_, "best_iteration"):
            self.best_iteration_ = int(self.model_.best_iteration)
        if hasattr(self.model_, "best_score"):
            self.best_score_ = float(self.model_.best_score)

    def _fit_lightgbm(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame | None,
        y_va: pd.Series | None,
    ) -> None:
        """Internal fit for LightGBM backend."""
        common_params: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": -1,
        }

        callbacks = (
            [lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)]
            if (X_va is not None and self.early_stopping_rounds > 0)
            else None
        )
        eval_set = (
            [
                (
                    X_va.values,
                    y_va.values.astype(int) if self.task_type == "classification" else y_va.values,
                )
            ]
            if X_va is not None
            else None
        )

        if self.task_type == "classification":
            self.model_ = lgb.LGBMClassifier(
                **common_params,
                scale_pos_weight=self.scale_pos_weight_,
            )
            self.model_.fit(
                X_tr.values,
                y_tr.values.astype(int),
                eval_set=eval_set,
                callbacks=callbacks,
            )
        else:
            self.model_ = lgb.LGBMRegressor(
                **common_params,
            )
            self.model_.fit(
                X_tr.values,
                y_tr.values,
                eval_set=eval_set,
                callbacks=callbacks,
            )

        if hasattr(self.model_, "best_iteration_"):
            self.best_iteration_ = int(self.model_.best_iteration_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict labels or continuous returns."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict.")
        return self.model_.predict(X[self.feature_names_].values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (only for classification)."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict_proba.")
        if self.task_type != "classification":
            raise ValueError("predict_proba is only supported for classification tasks.")
        return self.model_.predict_proba(X[self.feature_names_].values)

    @property
    def feature_importances(self) -> pd.Series:
        """Feature importances indexed by feature name."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted to access feature importances.")
        return pd.Series(
            self.model_.feature_importances_,
            index=self.feature_names_,
            name="importance",
        ).sort_values(ascending=False)

    # ============================================================
    # Model Artifact Persistence
    # ============================================================

    def save(
        self,
        artifact_dir: str | Path = "models/artifacts",
        model_name: str | None = None,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save model binary and JSON metadata to versioned artifacts directory.

        Parameters
        ----------
        artifact_dir : str or Path
            Destination directory.
        model_name : str, optional
            Identifier string. If None, auto-generated with timestamp.
        metrics : dict, optional
            Evaluation metrics to record in metadata.

        Returns
        -------
        Path
            Path to the saved model file.
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted model.")

        dest_dir = Path(artifact_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        name_prefix = model_name if model_name else f"gbm_{self.backend}_{timestamp_str}_{uid}"

        # 1. Save binary model
        if self.backend == "xgboost":
            model_file = dest_dir / f"{name_prefix}.json"
            self.model_.save_model(str(model_file))
        else:  # lightgbm
            model_file = dest_dir / f"{name_prefix}.txt"
            self.model_.booster_.save_model(str(model_file))

        # 2. Save metadata
        metadata = {
            "model_id": name_prefix,
            "backend": self.backend,
            "task_type": self.task_type,
            "saved_at": datetime.datetime.now().isoformat(),
            "training_date_range": {
                "start": self.training_start_date_,
                "end": self.training_end_date_,
            },
            "feature_names": self.feature_names_,
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda,
                "early_stopping_rounds": self.early_stopping_rounds,
                "scale_pos_weight": self.scale_pos_weight_,
                "best_iteration": self.best_iteration_,
                "best_score": self.best_score_,
            },
            "metrics": metrics if metrics is not None else {},
        }

        meta_file = dest_dir / f"{name_prefix}.meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Model saved to '%s' with metadata in '%s'", model_file, meta_file)
        return model_file

    @classmethod
    def load(cls, filepath: str | Path) -> GradientBoostingModel:
        """Restore GradientBoostingModel from saved binary and companion metadata.

        Parameters
        ----------
        filepath : str or Path
            Path to the saved model file (.json or .txt) or metadata file (.meta.json).

        Returns
        -------
        GradientBoostingModel
        """
        p = Path(filepath)
        if p.name.endswith(".meta.json"):
            meta_file = p
            base_stem = p.name[:-10]  # strip '.meta.json'
            model_file_json = p.parent / f"{base_stem}.json"
            model_file_txt = p.parent / f"{base_stem}.txt"
            model_file = model_file_json if model_file_json.exists() else model_file_txt
        else:
            model_file = p
            base_stem = p.stem
            meta_file = p.parent / f"{base_stem}.meta.json"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file '{model_file}' not found.")

        metadata: dict[str, Any] = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        backend = metadata.get("backend", "xgboost" if model_file.suffix == ".json" else "lightgbm")
        task_type = metadata.get("task_type", "classification")
        hparams = metadata.get("hyperparameters", {})

        inst = cls(
            backend=backend,
            task_type=task_type,
            n_estimators=hparams.get("n_estimators", 200),
            max_depth=hparams.get("max_depth", 3),
            learning_rate=hparams.get("learning_rate", 0.03),
            subsample=hparams.get("subsample", 0.8),
            colsample_bytree=hparams.get("colsample_bytree", 0.8),
            reg_alpha=hparams.get("reg_alpha", 0.1),
            reg_lambda=hparams.get("reg_lambda", 1.0),
        )

        inst.feature_names_ = metadata.get("feature_names", [])
        inst.scale_pos_weight_ = hparams.get("scale_pos_weight", 1.0)
        inst.best_iteration_ = hparams.get("best_iteration")
        inst.best_score_ = hparams.get("best_score")

        tr_dates = metadata.get("training_date_range", {})
        inst.training_start_date_ = tr_dates.get("start")
        inst.training_end_date_ = tr_dates.get("end")

        if backend == "xgboost":
            if task_type == "classification":
                inst.model_ = xgb.XGBClassifier()
            else:
                inst.model_ = xgb.XGBRegressor()
            inst.model_.load_model(str(model_file))
        else:
            if not LIGHTGBM_AVAILABLE:
                raise RuntimeError("LightGBM must be installed to load LightGBM artifacts.")
            booster = lgb.Booster(model_file=str(model_file))
            if task_type == "classification":
                inst.model_ = lgb.LGBMClassifier()
            else:
                inst.model_ = lgb.LGBMRegressor()
            inst.model_._Booster = booster
            inst.model_.fitted_ = True

        inst.is_fitted_ = True
        logger.info("Loaded %s model from '%s'", backend, model_file)
        return inst
