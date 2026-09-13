# ============================================================
# Unit Tests: Regime Detection (Phase 16)
# ============================================================
"""
Unit tests for Hidden Markov Model (HMM) and Gaussian Mixture Model (GMM)
regime detection, state alignment, and rolling out-of-sample extraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.regime_detection import (
    RegimeFeatureExtractor,
    fit_gmm_regimes,
    fit_hmm_regimes,
    rolling_regime_features,
)


@pytest.fixture
def synthetic_regime_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic 2-regime price series with KNOWN true regime sequence.

    Regime 0 (Low-Vol Trending Bull): N = 300, mu = +0.0010, sigma = 0.005
    Regime 1 (High-Vol Choppy Crisis): N = 200, mu = -0.0025, sigma = 0.030
    Regime 0 (Low-Vol Trending Bull): N = 300, mu = +0.0010, sigma = 0.005
    """
    rng = np.random.RandomState(42)

    r_calm_1 = rng.normal(0.0010, 0.005, 300)
    r_crisis = rng.normal(-0.0025, 0.030, 200)
    r_calm_2 = rng.normal(0.0010, 0.005, 300)

    returns = np.concatenate([r_calm_1, r_crisis, r_calm_2])
    true_states = np.array([0] * 300 + [1] * 200 + [0] * 300)

    prices = 100.0 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({"close": prices})
    return df, true_states


# ============================================================
# 1. HMM Synthetic Parameter & State Recovery Test
# ============================================================


def test_hmm_known_regime_recovery(synthetic_regime_data: tuple[pd.DataFrame, np.ndarray]):
    """Verify HMM recovers true latent regimes with > 85% accuracy on distinct synthetic regimes."""
    df, true_states = synthetic_regime_data
    res = fit_hmm_regimes(df, n_regimes=2, seed=42)

    assert res.model_name == "GaussianHMM"
    assert res.n_regimes == 2

    # Verify state standardization: State 0 volatility mean < State 1 volatility mean
    vol_mean_0 = res.state_means.loc["regime_0", "volatility"]
    vol_mean_1 = res.state_means.loc["regime_1", "volatility"]
    assert vol_mean_0 < vol_mean_1

    # Check accuracy of predicted states against ground truth on common index
    pred_states = res.regime_labels.values
    # Note: the first 20 observations are dropped due to rolling 20d volatility warmup
    valid_true_states = true_states[len(true_states) - len(pred_states) :]
    accuracy = np.mean(pred_states == valid_true_states)
    assert (
        accuracy > 0.85
    ), f"HMM regime accuracy ({accuracy:.2%}) failed to exceed 85% on synthetic data."

    # Posterior probabilities must sum to 1.0 everywhere
    prob_sum = res.regime_probabilities.sum(axis=1)
    np.testing.assert_allclose(prob_sum.values, 1.0, rtol=1e-5)

    # Transition matrix persistence: both states should be highly persistent (> 0.80)
    trans = res.transition_matrix
    assert trans.loc["from_regime_0", "to_regime_0"] > 0.80
    assert trans.loc["from_regime_1", "to_regime_1"] > 0.80


# ============================================================
# 2. GMM Synthetic Regime Recovery Test
# ============================================================


def test_gmm_known_regime_recovery(synthetic_regime_data: tuple[pd.DataFrame, np.ndarray]):
    """Verify GMM separates distinct volatility clusters with state standardization."""
    df, true_states = synthetic_regime_data
    res = fit_gmm_regimes(df, n_regimes=2, seed=42)

    assert res.model_name == "GaussianMixture"
    # State 0 must have lower volatility variance
    vol_var_0 = res.state_covariances[0][1, 1]
    vol_var_1 = res.state_covariances[1][1, 1]
    assert vol_var_0 < vol_var_1

    # Check accuracy
    pred_states = res.regime_labels.values
    valid_true_states = true_states[len(true_states) - len(pred_states) :]
    accuracy = np.mean(pred_states == valid_true_states)
    assert accuracy > 0.85


# ============================================================
# 3. Rolling Zero-Lookahead Regime Generation
# ============================================================


def test_rolling_regime_features():
    """Verify rolling out-of-sample regime extraction produces valid features."""
    rng = np.random.RandomState(42)
    n = 180
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    df = pd.DataFrame({"close": prices})

    out = rolling_regime_features(df, window=80, refit_frequency=30, method="hmm", n_regimes=2)

    assert "regime_label" in out.columns
    assert "regime_prob_0" in out.columns
    assert "regime_prob_1" in out.columns
    assert "regime_entropy" in out.columns

    # Valid values after warmup should be populated
    valid_labels = out["regime_label"].dropna()
    assert len(valid_labels) > 0
    assert set(valid_labels.unique()).issubset({0.0, 1.0})

    valid_entropy = out["regime_entropy"].dropna()
    assert (valid_entropy >= 0).all()


# ============================================================
# 4. Extractor & Error Handling
# ============================================================


def test_regime_feature_extractor():
    """Verify RegimeFeatureExtractor integrates into FeatureBase pipeline."""
    n = 150
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    df = pd.DataFrame({"close": prices})

    extractor = RegimeFeatureExtractor(window=70, refit_frequency=25, n_regimes=2)
    out = extractor.transform(df, append=True)

    assert "regime_label" in out.columns
    assert "regime_prob_0" in out.columns
    assert "regime_entropy" in out.columns


def test_short_series_raises():
    """Series too short for regime estimation must raise ValueError."""
    short_df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    with pytest.raises(ValueError, match="Insufficient valid data points"):
        fit_hmm_regimes(short_df)
