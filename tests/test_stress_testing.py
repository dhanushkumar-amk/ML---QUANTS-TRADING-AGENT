# ============================================================
# Unit Tests — Stress Testing & Monte Carlo Engine
# ============================================================
"""
Tests for Phase 43 Stress Testing module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestResult
from src.backtest.stress_testing import (
    analyze_regime_performance,
    circular_block_bootstrap,
    evaluate_crisis_periods,
    run_monte_carlo_simulation,
    run_scenario_stress_tests,
)
from src.portfolio.risk_engine import RiskConfig, RiskEngine


@pytest.fixture
def sample_returns():
    """Create a reproducible 500-day return series."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    rets = rng.normal(0.0005, 0.015, size=500)
    return pd.Series(rets, index=dates, name="returns")


def test_circular_block_bootstrap_statistical_preservation(sample_returns):
    """Test that block bootstrap preserves the mean and variance of the underlying series."""
    rets = sample_returns.values
    sim_paths = circular_block_bootstrap(rets, n_simulations=500, block_size=10, random_state=42)

    assert sim_paths.shape == (500, len(rets))
    orig_mean = np.mean(rets)
    orig_var = np.var(rets)

    sim_mean = np.mean(sim_paths)
    sim_var = np.mean(np.var(sim_paths, axis=1))

    # Resampled mean and variance should align closely with the empirical sample
    assert np.isclose(sim_mean, orig_mean, atol=0.001)
    assert np.isclose(sim_var, orig_var, rtol=0.15)


def test_monte_carlo_percentile_ordering(sample_returns):
    """Test that Monte Carlo percentiles are strictly ordered: 5th <= 50th <= 95th."""
    mc_res = run_monte_carlo_simulation(
        sample_returns, n_simulations=200, block_size=10, method="block", random_state=42
    )

    p5 = mc_res.percentile_terminal_return[5]
    p50 = mc_res.percentile_terminal_return[50]
    p95 = mc_res.percentile_terminal_return[95]

    assert p5 <= p50 <= p95, f"Percentiles misordered: {p5=}, {p50=}, {p95=}"

    # Verify equity paths shapes
    assert mc_res.simulated_equity_paths.shape == (200, len(sample_returns) + 1)
    assert mc_res.percentile_equity_paths[50].shape == (len(sample_returns) + 1,)


def test_evaluate_crisis_periods(sample_returns):
    """Test crisis replay correctly isolates specified dates and computes metrics."""
    equity = 100000.0 * (1.0 + sample_returns).cumprod()
    dates = sample_returns.index

    b_res = BacktestResult(
        strategy_name="CrisisTestStrat",
        initial_capital=100000.0,
        portfolio_equity=equity,
        daily_returns=sample_returns,
        positions=pd.DataFrame(index=dates, columns=["AAPL"], data=1.0),
        trades=[],
        cash=pd.Series(index=dates, data=10000.0),
        turnover=pd.Series(index=dates, data=0.0),
        commission_paid=pd.Series(index=dates, data=0.0),
        slippage_paid=pd.Series(index=dates, data=0.0),
    )

    custom_crisis = {
        "TestCrashWindow": ("2020-02-19", "2020-03-23"),
    }

    results = evaluate_crisis_periods(
        b_res,
        benchmark_returns=sample_returns * 0.9,
        crisis_periods=custom_crisis,
    )

    assert "TestCrashWindow" in results
    res = results["TestCrashWindow"]
    assert res.trading_bars > 0
    assert "strategy_max_dd_pct" in res.to_dict()
    assert "excess_return_pct" in res.to_dict()


def test_regime_conditional_breakdown():
    """Test that analyze_regime_performance partitions returns accurately by regime labels."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    # Low-vol positive drift for regime 0, high-vol negative drift for regime 1
    r0 = np.full(50, 0.005)
    r1 = np.full(50, -0.008)
    rets = pd.Series(np.concatenate([r0, r1]), index=dates)
    labels = pd.Series(np.array([0] * 50 + [1] * 50), index=dates)

    df_reg = analyze_regime_performance(
        rets, labels, regime_names={0: "Bull_Calm", 1: "Bear_Turbulent"}
    )

    assert len(df_reg) == 2
    assert df_reg.loc[df_reg["regime_id"] == 0, "annualized_return_pct"].iloc[0] > 0
    assert df_reg.loc[df_reg["regime_id"] == 1, "annualized_return_pct"].iloc[0] < 0
    assert df_reg.loc[df_reg["regime_id"] == 0, "bars"].iloc[0] == 50


def test_scenario_stress_testing_risk_engine_responses():
    """Test that synthetic shock scenarios trigger appropriate Phase 39 RiskEngine safety gates."""
    config = RiskConfig(
        max_drawdown_pct=0.15,
        daily_loss_limit_pct=0.03,
        vol_spike_threshold=1.50,
    )
    risk_engine = RiskEngine(config=config)

    scenarios = run_scenario_stress_tests(
        risk_engine=risk_engine,
        base_portfolio_equity=100000.0,
        tickers=("AAPL", "MSFT"),
    )

    assert "volatility_doubling" in scenarios
    assert "flash_gap_down" in scenarios
    assert "drawdown_kill_switch" in scenarios

    # Volatility doubling scenario must activate VOLATILITY_DERISKING
    assert scenarios["volatility_doubling"].is_capital_protected
    assert "VOLATILITY_DERISKING" in scenarios["volatility_doubling"].safety_systems_activated

    # Flash gap-down (-10% loss) must trip DAILY_LOSS_LIMIT
    assert scenarios["flash_gap_down"].is_capital_protected
    assert "DAILY_LOSS_LIMIT" in scenarios["flash_gap_down"].safety_systems_activated

    # 18% drawdown breach must activate DRAWDOWN_KILL_SWITCH
    assert scenarios["drawdown_kill_switch"].is_capital_protected
    assert "DRAWDOWN_KILL_SWITCH" in scenarios["drawdown_kill_switch"].safety_systems_activated
