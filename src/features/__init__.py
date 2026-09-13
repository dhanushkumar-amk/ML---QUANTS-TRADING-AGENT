# src.features — feature engineering & selection
"""Feature engineering: technical indicators, rolling stats, selection, EDA, and stationarity."""

from src.features.eda_utils import (
    compute_daily_returns,
    plot_correlation_matrix,
    plot_price_series,
    plot_returns_distribution,
    plot_rolling_volatility,
    plot_seasonality,
    summary_stats_table,
)
from src.features.stationarity import (
    ADFResult,
    KPSSResult,
    StationarityReport,
    adf_test,
    difference_series,
    kpss_test,
    stationarity_report,
    test_multiple_series,
)

__all__ = [
    # EDA utilities
    "compute_daily_returns",
    "summary_stats_table",
    "plot_price_series",
    "plot_returns_distribution",
    "plot_correlation_matrix",
    "plot_rolling_volatility",
    "plot_seasonality",
    # Stationarity testing
    "ADFResult",
    "KPSSResult",
    "StationarityReport",
    "adf_test",
    "kpss_test",
    "stationarity_report",
    "test_multiple_series",
    "difference_series",
]
