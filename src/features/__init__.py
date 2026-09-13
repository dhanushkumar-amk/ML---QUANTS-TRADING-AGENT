# src.features — feature engineering & selection
"""Feature engineering: technical indicators, rolling stats, selection, EDA, and stationarity."""

from src.features.autocorrelation_diagnostics import (
    AutocorrelationReport,
    LjungBoxRawResult,
    RunLengthResult,
    VarianceRatioResult,
    autocorrelation_report,
    compute_acf_pacf,
    ljung_box_raw,
    multi_asset_autocorrelation_summary,
    multi_horizon_autocorrelation_analysis,
    run_length_analysis,
    variance_ratio_test,
)
from src.features.eda_utils import (
    compute_daily_returns,
    plot_acf_squared_returns,
    plot_correlation_matrix,
    plot_price_series,
    plot_returns_distribution,
    plot_rolling_volatility,
    plot_seasonality,
    plot_volatility_regimes,
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
from src.features.volatility_analysis import (
    ARCHLMResult,
    LjungBoxResult,
    VolatilityClusteringReport,
    arch_lm_test,
    batch_volatility_clustering,
    compute_acf_series,
    compute_volatility_proxies,
    ljung_box_test,
    volatility_clustering_report,
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
    "plot_acf_squared_returns",
    "plot_volatility_regimes",
    # Stationarity testing
    "ADFResult",
    "KPSSResult",
    "StationarityReport",
    "adf_test",
    "kpss_test",
    "stationarity_report",
    "test_multiple_series",
    "difference_series",
    # Volatility clustering analysis
    "ARCHLMResult",
    "LjungBoxResult",
    "VolatilityClusteringReport",
    "compute_volatility_proxies",
    "compute_acf_series",
    "ljung_box_test",
    "arch_lm_test",
    "volatility_clustering_report",
    "batch_volatility_clustering",
    # Autocorrelation & momentum diagnostics
    "AutocorrelationReport",
    "LjungBoxRawResult",
    "RunLengthResult",
    "VarianceRatioResult",
    "compute_acf_pacf",
    "ljung_box_raw",
    "variance_ratio_test",
    "run_length_analysis",
    "autocorrelation_report",
    "multi_horizon_autocorrelation_analysis",
    "multi_asset_autocorrelation_summary",
]
