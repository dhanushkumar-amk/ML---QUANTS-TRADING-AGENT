# tests.test_mean_variance_optimizer — Unit tests for Mean-Variance Portfolio Optimization
"""Unit tests for Markowitz Mean-Variance Optimization, Ledoit-Wolf shrinkage, and constraints."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio.mean_variance_optimizer import (
    compute_efficient_frontier,
    compute_model_expected_returns,
    estimate_covariance,
    optimize_maximum_sharpe,
    optimize_minimum_variance,
    portfolio_performance,
    run_rolling_mean_variance_rebalance,
)


class TestMeanVarianceOptimizer:
    """Test suite for Markowitz mean-variance optimization and shrinkage estimation."""

    @pytest.fixture
    def synthetic_two_asset_system(self):
        """Analytical 2-asset system with known minimum variance solution."""
        sigma1 = 0.20
        sigma2 = 0.30
        rho = 0.25
        cov12 = rho * sigma1 * sigma2
        cov_matrix = np.array(
            [
                [sigma1**2, cov12],
                [cov12, sigma2**2],
            ]
        )
        expected_returns = np.array([0.10, 0.15])
        # Analytical unconstrained minimum variance weight for asset 1:
        # w1 = (sigma2^2 - cov12) / (sigma1^2 + sigma2^2 - 2 * cov12)
        var1 = sigma1**2
        var2 = sigma2**2
        analytical_w1 = (var2 - cov12) / (var1 + var2 - 2 * cov12)
        analytical_w2 = 1.0 - analytical_w1
        return {
            "cov": cov_matrix,
            "returns": expected_returns,
            "analytical_w": np.array([analytical_w1, analytical_w2]),
        }

    def test_analytical_minimum_variance_two_assets(self, synthetic_two_asset_system):
        """Test that numerical optimizer matches analytical 2-asset minimum variance solution."""
        cov = synthetic_two_asset_system["cov"]
        expected_w = synthetic_two_asset_system["analytical_w"]

        w_opt, info = optimize_minimum_variance(cov)

        assert info["success"] is True
        np.testing.assert_allclose(w_opt, expected_w, atol=1e-4)
        assert np.isclose(np.sum(w_opt), 1.0)
        assert np.all(w_opt >= 0.0)

    def test_position_cap_constraints(self, synthetic_two_asset_system):
        """Test that position caps (e.g. max_weight=0.40) are strictly enforced."""
        cov = synthetic_two_asset_system["cov"]
        returns = synthetic_two_asset_system["returns"]

        # In unconstrained 2-asset, w1 is ~0.64. Setting max_weight=0.50 must cap it.
        w_mv, _ = optimize_minimum_variance(cov, max_weight=0.50)
        assert np.all(w_mv <= 0.50001)
        assert np.all(w_mv >= -1e-6)
        assert np.isclose(np.sum(w_mv), 1.0)

        # In max Sharpe, asset 2 has higher return; test max_weight=0.60
        w_ms, _ = optimize_maximum_sharpe(returns, cov, max_weight=0.60)
        assert np.all(w_ms <= 0.60001)
        assert np.all(w_ms >= -1e-6)
        assert np.isclose(np.sum(w_ms), 1.0)

    def test_ledoit_wolf_shrinkage_reduces_condition_number(self):
        """Test that Ledoit-Wolf shrinkage produces a better conditioned covariance matrix on noisy data."""
        np.random.seed(42)
        # Few observations, many assets -> ill-conditioned sample covariance
        n_samples = 20
        n_assets = 10
        noisy_returns = np.random.randn(n_samples, n_assets) * 0.02

        sample_cov, _ = estimate_covariance(noisy_returns, method="sample")
        lw_cov, delta = estimate_covariance(noisy_returns, method="ledoit_wolf")

        # Check shrinkage intensity is strictly between 0 and 1
        assert delta is not None
        assert 0.0 < delta <= 1.0

        # Condition number: ratio of largest to smallest singular value
        cond_sample = np.linalg.cond(sample_cov)
        cond_lw = np.linalg.cond(lw_cov)

        # Shrinkage pulls eigenvalues toward target, strictly reducing condition number
        assert cond_lw < cond_sample
        assert np.all(np.linalg.eigvalsh(lw_cov) > 0)  # Positive definite

    def test_compute_model_expected_returns(self):
        """Test conversion of directional probabilities into expected returns."""
        probs = np.array([0.60, 0.40, 0.50])
        vols = np.array([0.20, 0.25, 0.15])

        exp_ret = compute_model_expected_returns(probs, vols, scaling_factor=1.0)

        # Asset 0 (p=0.60 > 0.5) must have positive expected return
        assert exp_ret[0] > 0.0
        assert np.isclose(exp_ret[0], (2 * 0.60 - 1.0) * 0.20)

        # Asset 1 (p=0.40 < 0.5) must have negative expected return
        assert exp_ret[1] < 0.0
        assert np.isclose(exp_ret[1], (2 * 0.40 - 1.0) * 0.25)

        # Asset 2 (p=0.50 neutral) must have 0 expected return
        assert np.isclose(exp_ret[2], 0.0)

        # Test dimension mismatch
        with pytest.raises(ValueError, match="Mismatched lengths"):
            compute_model_expected_returns([0.6], [0.2, 0.3])

    def test_compute_efficient_frontier(self, synthetic_two_asset_system):
        """Test that efficient frontier returns valid monotonically ordered or bounded points."""
        cov = synthetic_two_asset_system["cov"]
        returns = synthetic_two_asset_system["returns"]

        frontier = compute_efficient_frontier(returns, cov, num_points=10)

        assert "returns" in frontier
        assert "volatilities" in frontier
        assert "weights" in frontier
        assert len(frontier["returns"]) > 0
        assert frontier["max_sharpe"]["sharpe"] >= frontier["min_variance"]["sharpe"]

    def test_portfolio_performance(self):
        """Test portfolio expected return, volatility, and Sharpe calculation."""
        weights = np.array([0.5, 0.5])
        returns = np.array([0.10, 0.20])
        cov = np.diag([0.04, 0.09])  # sigma1=0.20, sigma2=0.30, zero correlation

        ret, vol, sharpe = portfolio_performance(weights, returns, cov, risk_free_rate=0.02)

        expected_ret = 0.5 * 0.10 + 0.5 * 0.20  # 0.15
        expected_var = (0.5**2) * 0.04 + (0.5**2) * 0.09  # 0.01 + 0.0225 = 0.0325
        expected_vol = np.sqrt(expected_var)

        assert np.isclose(ret, expected_ret)
        assert np.isclose(vol, expected_vol)
        assert np.isclose(sharpe, (expected_ret - 0.02) / expected_vol)

    def test_run_rolling_mean_variance_rebalance(self):
        """Test rolling walk-forward mean-variance rebalancing on synthetic returns."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=200, freq="B")
        ret_df = pd.DataFrame(
            np.random.randn(200, 3) * 0.01 + 0.0005,
            index=dates,
            columns=["AAPL", "MSFT", "SPY"],
        )

        res = run_rolling_mean_variance_rebalance(
            ret_df,
            lookback_window=60,
            rebalance_freq="M",
            cov_method="ledoit_wolf",
            max_weight=0.50,
        )

        weights = res["weights"]
        daily_ret = res["returns"]

        assert not weights.empty
        assert not daily_ret.empty
        # Check sum to 1
        np.testing.assert_allclose(weights.sum(axis=1).values, 1.0, atol=1e-4)
        # Check caps
        assert np.all(weights.values <= 0.50001)
        assert np.all(weights.values >= -1e-6)
