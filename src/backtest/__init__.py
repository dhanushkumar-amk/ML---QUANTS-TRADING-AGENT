# ============================================================
# src.backtest — Backtesting Engine, Reporting & Stress Testing
# ============================================================
"""
Backtesting engine: strategy simulation, performance analytics, reporting, and stress testing.
"""

from src.backtest.backtest_report import (
    calculate_information_ratio,
    calculate_tracking_error,
    compute_benchmark_metrics,
    compute_drawdown_series,
    compute_rolling_metrics,
    compute_trade_statistics,
    generate_html_report,
    generate_tearsheet,
    plot_benchmark_comparison,
    plot_equity_and_drawdown,
    plot_rolling_metrics,
    plot_trade_analysis,
    plot_underwater_chart,
)
from src.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    TradeRecord,
    TransactionCostModel,
)
from src.backtest.stress_testing import (
    CrisisPeriodResult,
    MonteCarloResult,
    ScenarioStressTestResult,
    analyze_regime_performance,
    circular_block_bootstrap,
    evaluate_crisis_periods,
    plot_monte_carlo_distribution,
    run_monte_carlo_simulation,
    run_scenario_stress_tests,
)

__all__ = [
    # Engine & Models
    "BacktestEngine",
    "BacktestResult",
    "TradeRecord",
    "TransactionCostModel",
    # Reporting & Tearsheet
    "generate_tearsheet",
    "generate_html_report",
    "compute_rolling_metrics",
    "compute_trade_statistics",
    "compute_benchmark_metrics",
    "compute_drawdown_series",
    "calculate_tracking_error",
    "calculate_information_ratio",
    "plot_equity_and_drawdown",
    "plot_rolling_metrics",
    "plot_trade_analysis",
    "plot_underwater_chart",
    "plot_benchmark_comparison",
    # Stress Testing & Monte Carlo
    "CrisisPeriodResult",
    "MonteCarloResult",
    "ScenarioStressTestResult",
    "evaluate_crisis_periods",
    "run_monte_carlo_simulation",
    "circular_block_bootstrap",
    "plot_monte_carlo_distribution",
    "analyze_regime_performance",
    "run_scenario_stress_tests",
]
