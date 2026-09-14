# tests/test_ensemble_model.py
"""Unit tests for ensemble strategies and stacking meta-learner with anti-leakage checks."""

import numpy as np
import pytest

from src.models.ensemble_model import (
    ConfidenceWeightedEnsemble,
    SimpleAverageEnsemble,
    StackingMetaLearner,
)


def test_simple_average_ensemble_2d_and_1d():
    """SimpleAverageEnsemble should correctly average 1D and 2D probability arrays."""
    p_xgb = np.array([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    p_dl = np.array([0.9, 0.3, 0.5])  # 1D positive class probas

    ensemble = SimpleAverageEnsemble()
    probs = ensemble.predict_proba([p_xgb, p_dl])

    assert probs.shape == (3, 2)
    # Row 0: xgb=[0.2, 0.8], dl=[0.1, 0.9] -> avg = [0.15, 0.85]
    assert np.isclose(probs[0, 1], 0.85)
    assert np.isclose(probs[0, 0], 0.15)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    preds = ensemble.predict([p_xgb, p_dl], threshold=0.5)
    assert np.array_equal(preds, [1, 0, 1])


def test_stacking_meta_learner_anti_leakage_and_fit():
    """Stacking meta-learner must be trained strictly on out-of-fold predictions."""
    N_oof = 120
    np.random.seed(42)
    y_oof = np.random.binomial(1, 0.5, size=N_oof)

    # Base model 1 OOF predictions (decently calibrated)
    oof_model1 = np.where(y_oof == 1, 0.75, 0.25) + np.random.normal(0, 0.05, N_oof)
    oof_model1 = np.clip(oof_model1, 0.01, 0.99)

    # Base model 2 OOF predictions (weaker)
    oof_model2 = np.random.uniform(0.4, 0.6, size=N_oof)

    stacker = StackingMetaLearner(C=1.0)
    assert not stacker.is_fitted_

    # Predict before fit must raise ValueError
    with pytest.raises(ValueError, match="must be fitted"):
        stacker.predict_proba([oof_model1, oof_model2])

    # Fit exclusively on OOF predictions
    stacker.fit([oof_model1, oof_model2], y_oof)
    assert stacker.is_fitted_
    assert stacker.weights_ is not None
    assert len(stacker.weights_) == 2
    # Model 1 should receive higher meta-weight than random Model 2
    assert stacker.weights_[0] > stacker.weights_[1]

    # Evaluate on held-out test predictions
    N_test = 30
    test_m1 = np.full(N_test, 0.8)
    test_m2 = np.full(N_test, 0.5)
    test_probs = stacker.predict_proba([test_m1, test_m2])

    assert test_probs.shape == (N_test, 2)
    assert np.all(test_probs[:, 1] > 0.5)
    test_preds = stacker.predict([test_m1, test_m2])
    assert np.all(test_preds == 1)


def test_stacking_meta_learner_mismatched_lengths():
    """Test dimension validation in StackingMetaLearner."""
    stacker = StackingMetaLearner()
    oof1 = np.zeros(50)
    oof2 = np.zeros(40)
    y = np.zeros(50)

    with pytest.raises(ValueError, match="must have identical length"):
        stacker.fit([oof1, oof2], y)

    with pytest.raises(ValueError, match="Sample size mismatch"):
        stacker.fit([oof1, np.zeros(50)], np.zeros(40))


def test_confidence_weighted_ensemble_conviction():
    """Confidence-weighted ensemble weights higher-conviction predictions more heavily."""
    # Model 1 is near 0.5 (low conviction), Model 2 is 0.95 (high conviction)
    p_m1 = np.array([[0.49, 0.51]])
    p_m2 = np.array([[0.05, 0.95]])

    ensemble = ConfidenceWeightedEnsemble(weighting_mode="conviction")
    probs = ensemble.predict_proba([p_m1, p_m2])

    assert probs.shape == (1, 2)
    # Model 2 has conviction |0.95 - 0.5| = 0.45; Model 1 has |0.51 - 0.5| = 0.01
    # Model 2 should dominate, so blended p1 should be close to 0.95 (> 0.85)
    assert probs[0, 1] > 0.85
    assert np.isclose(np.sum(probs[0]), 1.0)


def test_confidence_weighted_ensemble_historical_accuracy():
    """Historical accuracy mode weights models by their out-of-fold accuracy track record."""
    p_m1 = np.array([[0.2, 0.8], [0.8, 0.2]])
    p_m2 = np.array([[0.8, 0.2], [0.2, 0.8]])

    # Model 1 has 65% historical accuracy, Model 2 has 50%
    ensemble = ConfidenceWeightedEnsemble(
        weighting_mode="historical_accuracy",
        base_model_accuracies=[0.65, 0.50],
    )
    probs = ensemble.predict_proba([p_m1, p_m2])

    assert probs.shape == (2, 2)
    # On row 0, Model 1 predicts Class 1 (0.8) and Model 2 predicts Class 0 (0.2).
    # Since Model 1 has much higher historical accuracy, Class 1 should have probability > 0.5
    assert probs[0, 1] > 0.5
    assert np.allclose(np.sum(probs, axis=1), 1.0)
