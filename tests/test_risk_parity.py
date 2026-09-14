# tests.test_risk_parity — Unit tests for Risk Parity (Equal Risk Contribution) Portfolio
"""Unit tests for Risk Parity allocation, risk contribution decomposition, and comparison."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio.risk_parity import (
    calculate_risk_contributions,
    compare_allocations,
    optimize_risk_parity,
    run_rolling_risk_parity_rebalance,
)


class TestRiskParity:
    """Test suite for Equal Risk Contribution solver and risk accounting."""

    @pytest.fixture
    def synthetic_three_asset_cov(self):
        """Synthetic 3-asset covariance matrix with realistic correlation structure."""
        vols = np.array([0.15, 0.25, 0.35])
        corr = np.array(
            [
                [1.0, 0.4, 0.2],
                [0.4, 1.0, 0.5],
                [0.2, 0.5, 1.0],
            ]
        )
        cov_matrix = np.outer(vols, vols) * corr
        return cov_matrix

    def test_equal_risk_contribution_condition(self, synthetic_three_asset_cov):
        """Test that solved weights produce strictly equal risk contributions within numerical tolerance."""
        cov = synthetic_three_asset_cov
        weights, info = optimize_risk_parity(cov)

        assert info["success"] is True
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights > 0.0)

        # Check fractional risk contributions equal 1/3 (1/N)
        frc = info["fractional_risk_contributions"]
        np.testing.assert_allclose(frc, 1.0 / 3.0, atol=1e-4)

        # Check total risk contributions are equal
        rc = info["risk_contributions"]
        max_rc_diff = np.max(rc) - np.min(rc)
        assert max_rc_diff < 1e-4

    def test_analytical_uncorrelated_inverse_volatility(self):
        """In an uncorrelated 2-asset system, risk parity weights are strictly inversely proportional to volatility."""
        sigma1 = 0.10
        sigma2 = 0.20
        cov = np.diag([sigma1**2, sigma2**2])

        # Analytical ERC weights: w1 = (1/0.10) / (1/0.10 + 1/0.20) = 10 / 15 = 2/3
        expected_w1 = (1.0 / sigma1) / ((1.0 / sigma1) + (1.0 / sigma2))
        expected_w2 = 1.0 - expected_w1

        weights, info = optimize_risk_parity(cov)

        assert info["success"] is True
        np.testing.assert_allclose(weights, [expected_w1, expected_w2], atol=1e-4)
        assert weights[0] > weights[1]  # Lower vol asset gets higher capital allocation

    def test_euler_risk_decomposition_identity(self, synthetic_three_asset_cov):
        """Test Euler's homogeneous function identity: sum(RC_i) == portfolio_volatility."""
        cov = synthetic_three_asset_cov
        weights = np.array([0.4, 0.3, 0.3])

        mrc, rc, port_vol = calculate_risk_contributions(weights, cov)

        sum_rc = float(np.sum(rc))
        assert np.isclose(sum_rc, port_vol, atol=1e-8)

        # Marginal risk contribution should equal derivative d(vol)/d(w)
        eps = 1e-6
        for i in range(len(weights)):
            w_plus = weights.copy()
            w_plus[i] += eps
            vol_plus = np.sqrt(np.dot(w_plus.T, np.dot(cov, w_plus)))
            num_grad = (vol_plus - port_vol) / eps
            assert np.isclose(num_grad, mrc[i], atol=1e-4)

    def test_compare_allocations_dataframe(self, synthetic_three_asset_cov):
        """Test that compare_allocations generates a comprehensive multi-strategy DataFrame."""
        cov = synthetic_three_asset_cov
        expected_returns = np.array([0.08, 0.12, 0.16])
        tickers = ["SPY", "AAPL", "MSFT"]

        df = compare_allocations(expected_returns, cov, tickers=tickers, max_weight=0.60)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 1/N, Min Var, Max Sharpe, Risk Parity
        assert "Strategy" in df.columns
        assert "Weight_SPY" in df.columns
        assert "RiskContrib_AAPL" in df.columns

    def test_rolling_risk_parity_rebalance(self):
        """Test rolling walk-forward risk parity rebalancer on multi-asset returns."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=180, freq="B")
        ret_df = pd.DataFrame(
            np.random.randn(180, 3) * [0.01, 0.015, 0.02],
            index=dates,
            columns=["SPY", "AAPL", "MSFT"],
        )

        res = run_rolling_risk_parity_rebalance(
            ret_df,
            lookback_window=60,
            rebalance_freq="M",
            cov_method="ledoit_wolf",
            max_weight=0.60,
        )

        weights = res["weights"]
        daily_ret = res["returns"]

        assert not weights.empty
        assert not daily_ret.empty
        np.testing.assert_allclose(weights.sum(axis=1).values, 1.0, atol=1e-4)
        assert np.all(weights.values <= 0.60001)
        assert np.all(weights.values >= 0.0)
