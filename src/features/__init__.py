# src.features — feature engineering & selection
"""Feature engineering: technical indicators, rolling stats, selection, and EDA."""

from src.features.eda_utils import (
    compute_daily_returns,
    plot_correlation_matrix,
    plot_price_series,
    plot_returns_distribution,
    plot_rolling_volatility,
    plot_seasonality,
    summary_stats_table,
)

__all__ = [
    "compute_daily_returns",
    "summary_stats_table",
    "plot_price_series",
    "plot_returns_distribution",
    "plot_correlation_matrix",
    "plot_rolling_volatility",
    "plot_seasonality",
]
