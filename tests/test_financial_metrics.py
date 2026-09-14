# ============================================================
# Unit Tests for Financial Metrics (Phase 23)
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


class TestFinancialMetricsCalculations:
    """Test mathematical accuracy of core financial metrics against hand-calculated values."""

    def test_sharpe_ratio_known_values(self):
        """Test Sharpe ratio calculation on a known constant excess return series."""
        # Mean = 0.01, Std = 0.0
        rets = np.array([0.01, 0.01, 0.01, 0.01])
        # Constant returns have 0 std -> returns 0.0
        assert calculate_sharpe_ratio(rets) == 0.0

        # Mean = 0.01, Std > 0
        rets = np.array([0.02, 0.00, 0.02, 0.00])
        # Mean = 0.01, std (sample ddof=1) = sqrt(((0.01^2 * 4) / 3)) = 0.0115470
        expected_sharpe = np.sqrt(252) * (0.01 / np.std(rets, ddof=1))
        sharpe = calculate_sharpe_ratio(rets, periods_per_year=252)
        assert pytest.approx(expected_sharpe, rel=1e-4) == sharpe

    def test_sortino_ratio_downside_only(self):
        """Test Sortino only penalizes negative returns."""
        # Series A: has upside volatility (+0.05, +0.01, +0.01, -0.01)
        rets_a = np.array([0.05, 0.01, 0.01, -0.01])
        # Downside deviations below 0: [0, 0, 0, -0.01]
        # downside variance = 0.01^2 / 4 = 0.000025 -> std = 0.005
        # mean return = 0.015
        expected_sortino = np.sqrt(252) * (0.015 / 0.005)
        sortino = calculate_sortino_ratio(rets_a, periods_per_year=252)
        assert pytest.approx(expected_sortino, rel=1e-4) == sortino

        # Series with all positive returns has 0 downside std -> returns 0.0
        rets_pos = np.array([0.01, 0.02, 0.03])
        assert calculate_sortino_ratio(rets_pos) == 0.0

    def test_drawdown_series_and_max_drawdown(self):
        """Test drawdown series and max drawdown calculation."""
        # Returns: +10%, -20%, +10%
        # Wealth: 1.0 -> 1.10 -> 0.88 -> 0.968
        # Peak:   1.0 -> 1.10 -> 1.10 -> 1.10
        # Drawdown: 0.0 -> 0.0 -> (0.88 - 1.10)/1.10 = -0.20 -> (0.968 - 1.10)/1.10 = -0.12
        rets = pd.Series([0.10, -0.20, 0.10])
        wealth, peaks, dd = calculate_drawdown_series(rets)

        assert pytest.approx(1.10) == wealth.iloc[0]
        assert pytest.approx(0.88) == wealth.iloc[1]
        assert pytest.approx(0.968) == wealth.iloc[2]
        assert pytest.approx(-0.20) == dd.iloc[1]

        max_dd, duration = calculate_max_drawdown(rets)
        assert pytest.approx(-0.20) == max_dd
        assert duration >= 1

    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        # 252 days: Cumulative return 20%, max dd = -10% -> Calmar = 2.0
        # Let's construct a synthetic return series
        daily_ret = 0.20 / 252.0
        rets = np.full(252, daily_ret)
        # Introduce a drop on day 100: -10% and recover next day
        rets[100] = -0.10
        rets[101] = 0.111111  # recover

        calmar = calculate_calmar_ratio(rets, periods_per_year=252)
        assert calmar > 0.0

    def test_win_rate_and_profit_factor(self):
        """Test win rate and profit factor calculation."""
        # 3 wins of +0.02, 1 loss of -0.03 -> Total 4 trades
        # Gross profit = 0.06, Gross loss = 0.03 -> Profit Factor = 2.0, Win Rate = 0.75
        rets = np.array([0.02, 0.02, -0.03, 0.02, 0.0])  # zero return excluded from active
        win_rate, pf = calculate_win_rate_and_profit_factor(rets)

        assert pytest.approx(0.75) == win_rate
        assert pytest.approx(2.0) == pf


class TestStrategyReturnsAndReporting:
    """Test converting predictions to strategy returns and financial report generation."""

    def test_convert_predictions_to_strategy_returns_long_short(self):
        """Test long/short direction conversion: 1 -> +1, 0 -> -1."""
        preds = np.array([1, 0, 1, 0])
        actual = np.array([0.02, -0.01, -0.03, 0.02])
        # Expected strat returns:
        # Pred 1, actual +0.02 -> +0.02
        # Pred 0, actual -0.01 -> -1 * -0.01 = +0.01
        # Pred 1, actual -0.03 -> -0.03
        # Pred 0, actual +0.02 -> -1 * +0.02 = -0.02
        strat_rets = convert_predictions_to_strategy_returns(
            preds, actual, signal_type="long_short"
        )
        expected = np.array([0.02, 0.01, -0.03, -0.02])
        np.testing.assert_allclose(strat_rets.values, expected)

    def test_convert_predictions_to_strategy_returns_long_only(self):
        """Test long only direction conversion: 1 -> +1, 0 -> 0."""
        preds = np.array([1, 0, 1, 0])
        actual = np.array([0.02, -0.01, -0.03, 0.02])
        # Expected strat returns: [+0.02, 0.0, -0.03, 0.0]
        strat_rets = convert_predictions_to_strategy_returns(preds, actual, signal_type="long_only")
        expected = np.array([0.02, 0.0, -0.03, 0.0])
        np.testing.assert_allclose(strat_rets.values, expected)

    def test_compute_financial_metrics_with_benchmark(self):
        """Test full metrics battery with benchmark beta & alpha."""
        pd.date_range("2023-01-01", periods=100, freq="B")
        np.random.seed(42)
        bench = np.random.normal(0.0005, 0.01, size=100)
        strat = 0.8 * bench + np.random.normal(0.0002, 0.005, size=100)

        metrics = compute_financial_metrics(strat, benchmark_returns=bench)
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "calmar_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "alpha" in metrics
        assert "beta" in metrics
        assert pytest.approx(0.8, abs=0.2) == metrics["beta"]

    def test_financial_evaluation_report(self):
        """Test per-fold financial evaluation report creation."""
        dates = pd.date_range("2023-01-01", periods=20, freq="B")
        oof_df = pd.DataFrame(
            {
                "y_pred": [1, 0] * 10,
                "fold": [1] * 10 + [2] * 10,
            },
            index=dates,
        )
        actual_returns = pd.Series([0.01, -0.01] * 10, index=dates)

        scorecard, strat_rets = financial_evaluation_report(
            oof_df, actual_returns, signal_type="long_short", model_name="TestModel"
        )
        assert isinstance(scorecard, pd.DataFrame)
        assert "Fold 1" in scorecard.index
        assert "Fold 2" in scorecard.index
        assert "AGGREGATE" in scorecard.index
        assert scorecard.loc["AGGREGATE", "model"] == "TestModel"
        assert len(strat_rets) == 20
