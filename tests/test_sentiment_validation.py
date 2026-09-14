# ============================================================
# Unit Tests: Sentiment Feature Validation (Phase 33)
# ============================================================

import numpy as np
import pandas as pd

from src.nlp.sentiment_validation import (
    build_multimodal_dataset,
    evaluate_sentiment_incremental_value,
    generate_sentiment_verdict,
    run_sentiment_feature_diagnostics,
)


def _generate_synthetic_market_data(n_bars: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate realistic OHLCV dataframe for validation tests."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="B", tz="UTC")

    returns = rng.normal(loc=0.0005, scale=0.015, size=n_bars)
    price_paths = 100.0 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": price_paths * (1 + rng.normal(0, 0.002, n_bars)),
            "high": price_paths * (1 + np.abs(rng.normal(0, 0.005, n_bars))),
            "low": price_paths * (1 - np.abs(rng.normal(0, 0.005, n_bars))),
            "close": price_paths,
            "volume": rng.uniform(1e6, 5e6, n_bars),
        },
        index=dates,
    )
    return df


def test_build_multimodal_dataset_schema():
    """Verify build_multimodal_dataset constructs properly aligned features and target."""
    ohlcv = _generate_synthetic_market_data(150)
    feats, target, quant_cols, sentiment_cols = build_multimodal_dataset(
        ohlcv_df=ohlcv,
        ticker="SPY",  # SPY has no earnings calls -> earnings features will be neutral 0
    )

    assert len(feats) == len(target)
    assert len(feats) > 100  # After warmup drop
    assert set(quant_cols).issubset(feats.columns)
    assert set(sentiment_cols).issubset(feats.columns)
    assert target.isin([0, 1]).all()
    # SPY earnings features should all be 0.0
    assert (feats["earnings_overall_sentiment"] == 0.0).all()


def test_run_sentiment_feature_diagnostics():
    """Verify diagnostics table reports Spearman correlation, mutual info, and VIF."""
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame(
        {
            "q1": rng.normal(0, 1, n),
            "q2": rng.normal(0, 1, n),
            "sent1": rng.normal(0, 1, n),
        }
    )
    y = pd.Series(rng.binomial(1, 0.5, n))

    diag = run_sentiment_feature_diagnostics(
        feature_matrix=X,
        target=y,
        sentiment_cols=["sent1"],
        quant_cols=["q1", "q2"],
    )

    assert len(diag) == 3
    assert "spearman_corr" in diag.columns
    assert "mutual_info" in diag.columns
    assert "vif" in diag.columns
    assert "category" in diag.columns
    assert diag.loc[diag["feature"] == "sent1", "category"].iloc[0] == "NLP Sentiment"


def test_evaluate_known_informative_feature():
    """Verify framework correctly detects and rewards a KNOWN highly predictive feature."""
    rng = np.random.RandomState(123)
    n = 250
    dates = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")

    # Quant baseline: weak noisy features
    q1 = rng.normal(0, 1, n)
    q2 = rng.normal(0, 1, n)

    # True latent direction
    y_raw = rng.binomial(1, 0.5, n)
    y = pd.Series(y_raw, index=dates)

    # Informative sentiment feature: strongly correlated with target (85% alignment)
    sent_signal = np.where(y == 1, rng.normal(1.5, 0.5, n), rng.normal(-1.5, 0.5, n))

    X = pd.DataFrame(
        {
            "q1": q1,
            "q2": q2,
            "sentiment_signal": sent_signal,
        },
        index=dates,
    )

    results = evaluate_sentiment_incremental_value(
        feature_matrix=X,
        target=y,
        quant_cols=["q1", "q2"],
        sentiment_cols=["sentiment_signal"],
        n_splits=3,
        embargo_bars=2,
        random_state=123,
    )

    assert results["multimodal_accuracy"] > results["quant_accuracy"]
    assert results["sharpe_delta"] > 0.0

    verdict = generate_sentiment_verdict(results)
    assert verdict["decision"] in ("ACCEPTED", "PARTIAL")
    assert verdict["delta_sharpe"] > 0.0


def test_evaluate_known_noise_feature():
    """Verify framework correctly flags a KNOWN pure noise feature and avoids false positives."""
    rng = np.random.RandomState(999)
    n = 250
    dates = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")

    # Quant baseline with slight predictive edge
    y_raw = rng.binomial(1, 0.5, n)
    y = pd.Series(y_raw, index=dates)

    # Baseline quant feature that weakly predicts target
    q1 = np.where(y == 1, rng.normal(0.5, 1.0, n), rng.normal(-0.5, 1.0, n))
    q2 = rng.normal(0, 1, n)

    # Pure white noise sentiment feature
    sent_noise = rng.normal(0, 1, n)

    X = pd.DataFrame(
        {
            "q1": q1,
            "q2": q2,
            "sentiment_noise": sent_noise,
        },
        index=dates,
    )

    results = evaluate_sentiment_incremental_value(
        feature_matrix=X,
        target=y,
        quant_cols=["q1", "q2"],
        sentiment_cols=["sentiment_noise"],
        n_splits=3,
        embargo_bars=2,
        random_state=999,
    )

    verdict = generate_sentiment_verdict(results)
    # The noise feature should fail statistical significance
    # Either REJECTED (delta Sharpe <= 0) or PARTIAL (insignificant)
    assert verdict["decision"] in ("PARTIAL", "REJECTED")
    # CI should span zero or p-value should be >= 0.05
    ci_lower, ci_upper = verdict["bootstrap_sharpe_ci"]
    assert ci_lower <= 0.0 or verdict["diebold_mariano_pvalue"] >= 0.05


def test_verdict_decision_logic():
    """Verify generate_sentiment_verdict rules for ACCEPTED, PARTIAL, and REJECTED."""
    # Case 1: Significant outperformance
    res_acc = {
        "sharpe_delta": 0.45,
        "diebold_mariano_pvalue": 0.01,
        "bootstrap_sharpe_ci": (0.10, 0.80),
        "accuracy_delta": 0.05,
        "brier_delta": -0.02,
    }
    v_acc = generate_sentiment_verdict(res_acc)
    assert v_acc["decision"] == "ACCEPTED"

    # Case 2: Positive delta but insignificant
    res_part = {
        "sharpe_delta": 0.15,
        "diebold_mariano_pvalue": 0.35,
        "bootstrap_sharpe_ci": (-0.20, 0.50),
        "accuracy_delta": 0.01,
        "brier_delta": -0.005,
    }
    v_part = generate_sentiment_verdict(res_part)
    assert v_part["decision"] == "PARTIAL"

    # Case 3: Negative delta
    res_rej = {
        "sharpe_delta": -0.20,
        "diebold_mariano_pvalue": 0.80,
        "bootstrap_sharpe_ci": (-0.60, 0.10),
        "accuracy_delta": -0.02,
        "brier_delta": 0.015,
    }
    v_rej = generate_sentiment_verdict(res_rej)
    assert v_rej["decision"] == "REJECTED"
