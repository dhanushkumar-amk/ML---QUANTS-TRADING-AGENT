# ============================================================
# Ensemble Strategies & Stacking Meta-Learner (Phase 29)
# ============================================================
"""
Ensemble models combining tabular gradient boosters, linear baselines, and deep learning.

=============================================================================
THEORETICAL FOUNDATION: ENSEMBLING & OUT-OF-FOLD STACKING INTEGRITY
-----------------------------------------------------------------------------
1. Why Ensembles Work:
   Ensemble variance reduction (Brown et al., 2005):
   \\text{Error}(\\text{Ensemble}) = \\bar{E} - \\bar{A},
   where \\bar{E} is average individual model error and \\bar{A} is ensemble diversity.
   When models make uncorrelated prediction errors (e.g., tree splits on tabular
   microstructure vs. recurrent sequence memory), ensembling dampens idiosyncratic
   model variance.

2. Critical Anti-Leakage Protocol: Strictly Out-of-Fold (OOF) Stacking:
   In Stacking, a meta-learner (e.g. L2-regularized Logistic Regression) learns to
   weight predictions from base models:
   \\hat{y}_{\\text{meta}} = \\sigma(w_1 \\hat{p}_1 + w_2 \\hat{p}_2 + \\dots + b).

   CRITICAL ANTI-LEAKAGE RULE:
   The meta-learner must NEVER be trained on the base models' in-sample training
   predictions. Base models fit in-sample data near-perfectly, leading the
   meta-learner to assign overconfident weights to the most complex, overfitted base
   model. Stacking meta-learners MUST be trained strictly on concatenated
   OUT-OF-FOLD (OOF) cross-validation predictions.

3. Ensemble Strategies Implemented:
   - Simple Probability Averaging: Robust equal-weight baseline.
   - Stacking Meta-Learner: OOF-trained linear combination.
   - Confidence-Weighted Ensemble: Weights votes dynamically by model conviction
     or trailing historical fold accuracy.
=============================================================================
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Simple Probability Averaging Ensemble
# ============================================================


class SimpleAverageEnsemble:
    """Ensemble blending base model probabilities with equal weighting."""

    def __init__(self) -> None:
        self.is_fitted_ = True

    def predict_proba(self, model_probs_list: list[np.ndarray]) -> np.ndarray:
        """Compute arithmetic mean of predicted probabilities across base models.

        Parameters
        ----------
        model_probs_list : list of np.ndarray
            Each array of shape (N, 2) or (N,).

        Returns
        -------
        np.ndarray of shape (N, 2)
        """
        if not model_probs_list:
            raise ValueError("model_probs_list must contain at least one probability array.")

        standardized = []
        for p in model_probs_list:
            arr = np.asarray(p, dtype=float)
            if arr.ndim == 1:
                arr = np.column_stack([1.0 - arr, arr])
            standardized.append(arr)

        mean_probs = np.mean(standardized, axis=0)
        return mean_probs

    def predict(self, model_probs_list: list[np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """Predict binary directional classes."""
        probs = self.predict_proba(model_probs_list)
        return (probs[:, 1] >= threshold).astype(int)


# ============================================================
# 2. Out-of-Fold Stacking Meta-Learner
# ============================================================


class StackingMetaLearner:
    """Super Learner meta-classifier trained strictly on Out-of-Fold base predictions.

    Parameters
    ----------
    C : float, default 1.0
        Inverse L2 regularization strength.
    random_state : int, default 42
    """

    def __init__(self, C: float = 1.0, random_state: int = 42) -> None:
        self.C = C
        self.random_state = random_state
        self.meta_model = LogisticRegression(C=C, random_state=random_state, penalty="l2")
        self.is_fitted_ = False
        self.weights_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(
        self,
        oof_base_probabilities: list[np.ndarray],
        y_true_oof: np.ndarray | pd.Series,
    ) -> StackingMetaLearner:
        """Fit meta-learner on concatenated out-of-fold validation predictions.

        Parameters
        ----------
        oof_base_probabilities : list of np.ndarray
            Out-of-fold probability vectors from each base model.
        y_true_oof : array-like of shape (N,)
            Ground truth target aligned with the OOF predictions.
        """
        if not oof_base_probabilities:
            raise ValueError("oof_base_probabilities list cannot be empty.")

        n_samples = len(oof_base_probabilities[0])
        for idx, p in enumerate(oof_base_probabilities):
            if len(p) != n_samples:
                raise ValueError(
                    f"All OOF probability arrays must have identical length. "
                    f"Base model {idx} has length {len(p)} vs expected {n_samples}."
                )

        y_oof = np.asarray(y_true_oof, dtype=int)
        if n_samples != len(y_oof):
            raise ValueError(
                f"Sample size mismatch: OOF predictions ({n_samples}) vs y_true_oof ({len(y_oof)})."
            )

        feats = []
        for p in oof_base_probabilities:
            arr = np.asarray(p, dtype=float)
            p1 = arr[:, 1] if arr.ndim == 2 else arr
            feats.append(p1)

        X_meta = np.column_stack(feats)  # shape (N, n_base_models)

        self.meta_model.fit(X_meta, y_oof)
        self.weights_ = self.meta_model.coef_.ravel()
        self.intercept_ = float(self.meta_model.intercept_[0])
        self.is_fitted_ = True

        logger.info(
            "Fitted Stacking Meta-Learner across %d base models: Weights=%s, Intercept=%.3f",
            len(oof_base_probabilities),
            np.round(self.weights_, 3),
            self.intercept_,
        )
        return self

    def predict_proba(self, test_base_probabilities: list[np.ndarray]) -> np.ndarray:
        """Predict class probabilities for unseen test samples."""
        if not self.is_fitted_:
            raise ValueError("StackingMetaLearner must be fitted before predict_proba.")

        feats = []
        for p in test_base_probabilities:
            arr = np.asarray(p, dtype=float)
            p1 = arr[:, 1] if arr.ndim == 2 else arr
            feats.append(p1)

        X_test_meta = np.column_stack(feats)
        return self.meta_model.predict_proba(X_test_meta)

    def predict(
        self, test_base_probabilities: list[np.ndarray], threshold: float = 0.5
    ) -> np.ndarray:
        """Predict binary directional classes."""
        probs = self.predict_proba(test_base_probabilities)
        return (probs[:, 1] >= threshold).astype(int)


# ============================================================
# 3. Confidence / Regime-Weighted Ensemble
# ============================================================


class ConfidenceWeightedEnsemble:
    """Ensemble weighting votes dynamically by prediction conviction or trailing accuracy.

    Parameters
    ----------
    weighting_mode : {'conviction', 'historical_accuracy'}, default 'conviction'
        - 'conviction': Weights by distance from random coin flip $|p - 0.5|$.
        - 'historical_accuracy': Weights by historical out-of-fold accuracy.
    """

    def __init__(
        self,
        weighting_mode: Literal["conviction", "historical_accuracy"] = "conviction",
        base_model_accuracies: list[float] | None = None,
    ) -> None:
        self.weighting_mode = weighting_mode
        self.base_model_accuracies = base_model_accuracies
        self.is_fitted_ = True

    def predict_proba(self, model_probs_list: list[np.ndarray]) -> np.ndarray:
        """Compute weighted probabilities.

        Parameters
        ----------
        model_probs_list : list of np.ndarray
        """
        standardized = []
        for p in model_probs_list:
            arr = np.asarray(p, dtype=float)
            if arr.ndim == 1:
                arr = np.column_stack([1.0 - arr, arr])
            standardized.append(arr)

        n_models = len(standardized)
        N = standardized[0].shape[0]

        if self.weighting_mode == "conviction":
            # Per-sample dynamic weighting: weight proportional to |p1 - 0.5|
            weights = np.empty((n_models, N), dtype=float)
            for m_idx, p in enumerate(standardized):
                conviction = np.abs(p[:, 1] - 0.5) + 1e-4
                weights[m_idx] = conviction
            weights /= np.sum(weights, axis=0, keepdims=True)

            weighted_probs = np.zeros((N, 2), dtype=float)
            for m_idx in range(n_models):
                w = weights[m_idx, :, None]
                weighted_probs += w * standardized[m_idx]
            return weighted_probs

        else:
            # Historical accuracy weighting
            accs = (
                np.asarray(self.base_model_accuracies, dtype=float)
                if self.base_model_accuracies
                else np.ones(n_models)
            )
            # Softmax or normalized excess accuracy over 0.5
            excess = np.maximum(accs - 0.45, 0.01)
            norm_weights = excess / np.sum(excess)

            weighted_probs = np.zeros((N, 2), dtype=float)
            for m_idx, w in enumerate(norm_weights):
                weighted_probs += w * standardized[m_idx]
            return weighted_probs

    def predict(self, model_probs_list: list[np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """Predict binary directional classes."""
        probs = self.predict_proba(model_probs_list)
        return (probs[:, 1] >= threshold).astype(int)
