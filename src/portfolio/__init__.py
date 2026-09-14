# src.portfolio — Portfolio construction & risk theory
"""Portfolio optimization, position sizing, and risk management modules."""

from src.portfolio.kelly_sizing import (
    adjust_multi_asset_kelly,
    derive_kelly_parameters,
    fractional_kelly,
    generate_kelly_allocation_series,
    kelly_criterion,
    position_size,
)
from src.portfolio.mean_variance_optimizer import (
    compute_efficient_frontier,
    compute_model_expected_returns,
    estimate_covariance,
    optimize_maximum_sharpe,
    optimize_minimum_variance,
    portfolio_performance,
    run_rolling_mean_variance_rebalance,
)
from src.portfolio.risk_parity import (
    calculate_risk_contributions,
    compare_allocations,
    optimize_risk_parity,
    run_rolling_risk_parity_rebalance,
)

__all__ = [
    # Kelly Sizing
    "kelly_criterion",
    "fractional_kelly",
    "derive_kelly_parameters",
    "position_size",
    "adjust_multi_asset_kelly",
    "generate_kelly_allocation_series",
    # Mean-Variance Optimization
    "estimate_covariance",
    "compute_model_expected_returns",
    "portfolio_performance",
    "optimize_minimum_variance",
    "optimize_maximum_sharpe",
    "compute_efficient_frontier",
    "run_rolling_mean_variance_rebalance",
    # Risk Parity
    "calculate_risk_contributions",
    "optimize_risk_parity",
    "compare_allocations",
    "run_rolling_risk_parity_rebalance",
]
