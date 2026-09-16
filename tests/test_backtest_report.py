# ============================================================
# Unit Tests — Backtest Reporting & Tearsheet Generator
# ============================================================
"""
Tests for Phase 42 Backtest Reporting module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.backtest_report import (
    calculate_information_ratio,
    calculate_tracking_error,
    compute_benchmark_metrics,
    compute_drawdown_series,
    compute_rolling_metrics,
    compute_trade_statistics,
    generate_tearsheet,
)
from src.backtest.engine import BacktestResult, TradeRecord


@pytest.fixture
def synthetic_equity_and_returns():
    """Create a synthetic 252-day equity and returns series with positive drift."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    # Daily returns with ~10% annual return and ~15% annual vol
    daily_rets = rng.normal(0.10 / 252.0, 0.15 / np.sqrt(252.0), size=252)
    daily_rets[0] = 0.0
    returns_series = pd.Series(daily_rets, index=dates, name="daily_returns")
    equity_series = 100000.0 * (1.0 + returns_series).cumprod()
    return equity_series, returns_series


@pytest.fixture
def synthetic_trades():
    """Create synthetic trade log with known properties:

    - 4 trades: 3 wins (+1000, +2000, +500), 1 loss (-1000)
    - Win rate: 75%
    - Holding periods: 5, 10, 2, 8 days
    """
    return [
        TradeRecord(
            ticker="AAPL",
            entry_date="2023-01-05",
            exit_date="2023-01-12",
            side="LONG",
            quantity=50.0,
            entry_price=150.0,
            exit_price=170.0,
            pnl=1000.0,
            pnl_pct=0.133,
            holding_period_days=5,
        ),
        TradeRecord(
            ticker="MSFT",
            entry_date="2023-02-01",
            exit_date="2023-02-15",
            side="LONG",
            quantity=40.0,
            entry_price=250.0,
            exit_price=300.0,
            pnl=2000.0,
            pnl_pct=0.20,
            holding_period_days=10,
        ),
        TradeRecord(
            ticker="AAPL",
            entry_date="2023-03-01",
            exit_date="2023-03-03",
            side="LONG",
            quantity=50.0,
            entry_price=160.0,
            exit_price=170.0,
            pnl=500.0,
            pnl_pct=0.0625,
            holding_period_days=2,
        ),
        TradeRecord(
            ticker="MSFT",
            entry_date="2023-04-01",
            exit_date="2023-04-11",
            side="LONG",
            quantity=50.0,
            entry_price=280.0,
            exit_price=260.0,
            pnl=-1000.0,
            pnl_pct=-0.0714,
            holding_period_days=8,
        ),
    ]


def test_trade_statistics_computation(synthetic_trades):
    """Test trade-level metrics computation against known ground-truth values."""
    trades_dict = [t.to_dict() for t in synthetic_trades]
    stats = compute_trade_statistics(trades_dict)

    assert stats["total_trades"] == 4
    assert stats["winning_trades"] == 3
    assert stats["losing_trades"] == 1
    assert stats["win_rate_pct"] == 75.0
    # Gross win = 3500, Gross loss = 1000 => profit factor = 3.5
    assert stats["profit_factor"] == 3.5
    assert stats["best_trade_pnl"] == 2000.0
    assert stats["worst_trade_pnl"] == -1000.0
    # Average holding days: (5 + 10 + 2 + 8) / 4 = 6.25
    assert np.isclose(stats["average_holding_days"], 6.25, atol=0.1)


def test_drawdown_series_properties(synthetic_equity_and_returns):
    """Test that drawdown is always <= 0 and peaks match 0."""
    equity, _ = synthetic_equity_and_returns
    dd = compute_drawdown_series(equity)

    assert len(dd) == len(equity)
    assert (dd <= 1e-9).all(), "Drawdown series must always be non-positive."
    assert dd.iloc[0] == 0.0


def test_rolling_metrics_computation(synthetic_equity_and_returns):
    """Test rolling Sharpe, volatility, and win rate calculation."""
    _, returns = synthetic_equity_and_returns
    rolling_df = compute_rolling_metrics(returns, window=50)

    assert "rolling_sharpe" in rolling_df.columns
    assert "rolling_volatility" in rolling_df.columns
    assert "rolling_win_rate" in rolling_df.columns
    # First 49 values should be NaN
    assert rolling_df["rolling_sharpe"].iloc[:49].isna().all()
    # Values from index 50 onward should be valid finite floats
    valid_vals = rolling_df["rolling_sharpe"].iloc[50:].dropna()
    assert len(valid_vals) > 0
    assert np.isfinite(valid_vals).all()


def test_benchmark_metrics():
    """Test tracking error, information ratio, and beta on known synthetic data."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    bench = pd.Series(np.full(100, 0.001), index=dates)
    # Strat outperforms bench by 0.0005 per day with zero relative variance
    strat = pd.Series(np.full(100, 0.0015), index=dates)

    te = calculate_tracking_error(strat, bench)
    ir = calculate_information_ratio(strat, bench)
    metrics = compute_benchmark_metrics(strat, bench)

    assert np.isclose(te, 0.0, atol=1e-8)
    assert np.isclose(ir, 0.0, atol=1e-8)
    assert "alpha_annualized_pct" in metrics
    assert "beta" in metrics


def test_generate_tearsheet_and_html_export(synthetic_equity_and_returns, synthetic_trades, tmp_path):
    """Test end-to-end tearsheet generation and standalone HTML output."""
    equity, returns = synthetic_equity_and_returns
    dates = equity.index

    # Construct BacktestResult container
    b_res = BacktestResult(
        strategy_name="UnitTestStrategy",
        initial_capital=100000.0,
        portfolio_equity=equity,
        daily_returns=returns,
        positions=pd.DataFrame(index=dates, columns=["AAPL"], data=10.0),
        trades=synthetic_trades,
        cash=pd.Series(index=dates, data=50000.0),
        turnover=pd.Series(index=dates, data=1000.0),
        commission_paid=pd.Series(index=dates, data=1.0),
        slippage_paid=pd.Series(index=dates, data=1.0),
    )

    out_file = tmp_path / "test_tearsheet.html"
    res = generate_tearsheet(
        backtest_results=b_res,
        benchmark_returns=returns * 0.8,
        title="Unit Test Tearsheet",
        output_html_path=str(out_file),
    )

    assert "summary_metrics" in res
    assert "trade_stats" in res
    assert "figures" in res
    assert "equity_and_drawdown" in res["figures"]
    assert "rolling_metrics" in res["figures"]
    assert "trade_analysis" in res["figures"]
    assert "underwater_chart" in res["figures"]

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Unit Test Tearsheet" in content
    assert "TOTAL RETURN" in content.upper()
    assert "PROFIT FACTOR" in content.upper()
