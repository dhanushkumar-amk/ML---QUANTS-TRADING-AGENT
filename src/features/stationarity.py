# ============================================================
# Time-Series Stationarity Testing Module (Phase 9)
# ============================================================
"""
Formal stationarity testing module for financial time series.

Provides a dual-hypothesis testing framework:
  1. Augmented Dickey-Fuller (ADF) test:
       - H0: Series possesses a unit root (non-stationary).
       - H1: Series is stationary.
       - Rejection (p < alpha) indicates stationarity.

  2. Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test:
       - H0: Series is level or trend stationary.
       - H1: Series possesses a unit root (non-stationary).
       - Rejection (p < alpha) indicates non-stationarity.

Why run both tests?
  ADF and KPSS have opposite null hypotheses. Running them in tandem provides
  a much more robust evaluation than either test alone:
    - Case 1: ADF rejects H0 (p < alpha) AND KPSS fails to reject H0 (p >= alpha)
              --> Strong consensus: "stationary"
    - Case 2: ADF fails to reject H0 (p >= alpha) AND KPSS rejects H0 (p < alpha)
              --> Strong consensus: "non-stationary"
    - Case 3: Both fail to reject
              --> Inconclusive (often low test power or near-unit-root series)
    - Case 4: Both reject
              --> Inconclusive (often indicates deterministic trend or structural break)

Functions:
  - adf_test: Runs Augmented Dickey-Fuller test with critical values.
  - kpss_test: Runs KPSS test with critical values.
  - stationarity_report: Synthesizes ADF and KPSS into a unified consensus verdict.
  - test_multiple_series: Batch evaluates multiple series into a summary DataFrame.
  - difference_series: Applies d-th order differencing to transform non-stationary series.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ADFResult:
    """Structured result from an Augmented Dickey-Fuller (ADF) test."""

    test_statistic: float
    p_value: float
    used_lag: int
    n_observations: int
    critical_values: dict[str, float] = field(default_factory=dict)
    max_lag: int | None = None
    autolag: str | None = None

    @property
    def is_stationary_1pct(self) -> bool:
        """True if ADF rejects unit root null at 1% significance level."""
        return self.test_statistic < self.critical_values.get("1%", -np.inf)

    @property
    def is_stationary_5pct(self) -> bool:
        """True if ADF rejects unit root null at 5% significance level."""
        return self.test_statistic < self.critical_values.get("5%", -np.inf)

    @property
    def is_stationary_10pct(self) -> bool:
        """True if ADF rejects unit root null at 10% significance level."""
        return self.test_statistic < self.critical_values.get("10%", -np.inf)

    def is_stationary(self, alpha: float = 0.05) -> bool:
        """Check stationarity at given significance level alpha."""
        return bool(self.p_value < alpha)


@dataclass
class KPSSResult:
    """Structured result from a Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test."""

    test_statistic: float
    p_value: float
    lags_used: int
    critical_values: dict[str, float] = field(default_factory=dict)
    regression: str = "c"

    @property
    def is_stationary_1pct(self) -> bool:
        """True if KPSS fails to reject stationarity null at 1% significance level."""
        return self.test_statistic < self.critical_values.get("1%", np.inf)

    @property
    def is_stationary_5pct(self) -> bool:
        """True if KPSS fails to reject stationarity null at 5% significance level."""
        return self.test_statistic < self.critical_values.get("5%", np.inf)

    @property
    def is_stationary_10pct(self) -> bool:
        """True if KPSS fails to reject stationarity null at 10% significance level."""
        return self.test_statistic < self.critical_values.get("10%", np.inf)

    def is_stationary(self, alpha: float = 0.05) -> bool:
        """Check stationarity at given significance level alpha.

        Note: KPSS null hypothesis H0 is that the series is stationary.
        Thus, series is considered stationary when p_value >= alpha (failure to reject H0).
        """
        return bool(self.p_value >= alpha)


@dataclass
class StationarityReport:
    """Unified consensus report combining ADF and KPSS dual-test results."""

    name: str
    alpha: float
    adf: ADFResult
    kpss: KPSSResult
    conclusion: str  # "stationary", "non-stationary", "inconclusive"
    explanation: str

    @property
    def is_stationary(self) -> bool:
        """True if consensus conclusion is strictly 'stationary'."""
        return self.conclusion == "stationary"

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "name": self.name,
            "alpha": self.alpha,
            "conclusion": self.conclusion,
            "explanation": self.explanation,
            "adf_statistic": self.adf.test_statistic,
            "adf_pvalue": self.adf.p_value,
            "adf_stationary": self.adf.is_stationary(self.alpha),
            "adf_lags": self.adf.used_lag,
            "kpss_statistic": self.kpss.test_statistic,
            "kpss_pvalue": self.kpss.p_value,
            "kpss_stationary": self.kpss.is_stationary(self.alpha),
            "kpss_lags": self.kpss.lags_used,
        }


def _clean_series(series: pd.Series | np.ndarray) -> np.ndarray:
    """Ensure series is a 1D numeric array without NaNs or Infs."""
    if isinstance(series, pd.Series):
        clean = series.dropna().replace([np.inf, -np.inf], np.nan).dropna().values
    else:
        clean = np.asarray(series, dtype=float)
        clean = clean[~np.isnan(clean) & ~np.isinf(clean)]

    if len(clean) < 10:
        raise ValueError(
            f"Series length ({len(clean)}) is too short for stationarity testing. Minimum required is 10."
        )
    return clean


# ============================================================
# 1. ADF Test Wrapper
# ============================================================


def adf_test(
    series: pd.Series | np.ndarray,
    maxlag: int | None = None,
    autolag: str | None = "AIC",
    regression: str = "c",
) -> ADFResult:
    """Run Augmented Dickey-Fuller (ADF) test for unit root non-stationarity.

    Null Hypothesis (H0):
      The series possesses a unit root and is non-stationary.
    Alternative Hypothesis (H1):
      The series is stationary (no unit root).

    Decision Rule:
      Reject H0 if test statistic < critical value (or p-value < alpha).

    Parameters
    ----------
    series : pd.Series | np.ndarray
        Time series observations.
    maxlag : int | None
        Maximum lag order included in test.
    autolag : str | None
        Criterion for automatic lag selection ('AIC', 'BIC', 't-stat', or None).
    regression : str
        Constant and trend order to include in regression ('c', 'ct', 'ctt', 'n').

    Returns
    -------
    ADFResult
    """
    clean_vals = _clean_series(series)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        res = adfuller(
            clean_vals,
            maxlag=maxlag,
            autolag=autolag,
            regression=regression,
        )

    test_stat = float(res[0])
    p_val = float(res[1])
    used_lag = int(res[2])
    n_obs = int(res[3])
    crit_vals = {str(k): float(v) for k, v in res[4].items()}

    return ADFResult(
        test_statistic=test_stat,
        p_value=p_val,
        used_lag=used_lag,
        n_observations=n_obs,
        critical_values=crit_vals,
        max_lag=maxlag,
        autolag=autolag,
    )


# ============================================================
# 2. KPSS Test Wrapper
# ============================================================


def kpss_test(
    series: pd.Series | np.ndarray,
    regression: str = "c",
    nlags: str | int = "auto",
) -> KPSSResult:
    """Run Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity.

    Null Hypothesis (H0):
      The series is level-stationary (regression='c') or trend-stationary (regression='ct').
    Alternative Hypothesis (H1):
      The series possesses a unit root and is non-stationary.

    Decision Rule:
      Reject H0 if test statistic > critical value (or p-value < alpha).
      Series is stationary when we FAIL to reject H0 (p-value >= alpha).

    Parameters
    ----------
    series : pd.Series | np.ndarray
        Time series observations.
    regression : str
        'c' for level-stationarity (default), 'ct' for trend-stationarity.
    nlags : str | int
        Lag truncation parameter. 'auto' selects based on Hobijn et al. (1998).

    Returns
    -------
    KPSSResult
    """
    clean_vals = _clean_series(series)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        res = kpss(
            clean_vals,
            regression=regression,
            nlags=nlags,
        )

    test_stat = float(res[0])
    p_val = float(res[1])
    lags = int(res[2])
    crit_vals = {str(k): float(v) for k, v in res[3].items()}

    return KPSSResult(
        test_statistic=test_stat,
        p_value=p_val,
        lags_used=lags,
        critical_values=crit_vals,
        regression=regression,
    )


# ============================================================
# 3. Combined Stationarity Report (Dual-Test Consensus)
# ============================================================


def stationarity_report(
    series: pd.Series | np.ndarray,
    name: str = "",
    alpha: float = 0.05,
    regression: str = "c",
) -> StationarityReport:
    """Run both ADF and KPSS tests and synthesize a unified consensus verdict.

    Decision Matrix at significance level alpha:
      - Case 1: ADF stationary (p < alpha) & KPSS stationary (p >= alpha)
        --> "stationary"
      - Case 2: ADF non-stationary (p >= alpha) & KPSS non-stationary (p < alpha)
        --> "non-stationary"
      - Case 3: ADF non-stationary (p >= alpha) & KPSS stationary (p >= alpha)
        --> "inconclusive" (tests lack power; likely near-unit-root or structural break)
      - Case 4: ADF stationary (p < alpha) & KPSS non-stationary (p < alpha)
        --> "inconclusive" (heteroskedasticity, regime shift, or trend stationarity)

    Parameters
    ----------
    series : pd.Series | np.ndarray
        Time series observations.
    name : str
        Identifier or description of series.
    alpha : float
        Significance level threshold (default: 0.05 for 5%).
    regression : str
        Regression specification ('c' for constant, 'ct' for constant + trend).

    Returns
    -------
    StationarityReport
    """
    adf_res = adf_test(series, regression=regression)
    kpss_res = kpss_test(series, regression=regression)

    adf_stat_pass = adf_res.is_stationary(alpha=alpha)
    kpss_stat_pass = kpss_res.is_stationary(alpha=alpha)

    if adf_stat_pass and kpss_stat_pass:
        conclusion = "stationary"
        explanation = (
            f"Both tests agree: ADF rejects unit root (p={adf_res.p_value:.4e} < {alpha}) "
            f"and KPSS fails to reject stationarity (p={kpss_res.p_value:.4f} >= {alpha})."
        )
    elif not adf_stat_pass and not kpss_stat_pass:
        conclusion = "non-stationary"
        explanation = (
            f"Both tests agree: ADF fails to reject unit root (p={adf_res.p_value:.4f} >= {alpha}) "
            f"and KPSS rejects stationarity (p={kpss_res.p_value:.4f} < {alpha})."
        )
    elif not adf_stat_pass and kpss_stat_pass:
        conclusion = "inconclusive"
        explanation = (
            f"Disagreement: ADF indicates non-stationarity (p={adf_res.p_value:.4f} >= {alpha}) "
            f"while KPSS indicates stationarity (p={kpss_res.p_value:.4f} >= {alpha}). "
            f"Series may exhibit near-unit-root behavior or low test power."
        )
    else:  # adf_stat_pass and not kpss_stat_pass
        conclusion = "inconclusive"
        explanation = (
            f"Disagreement: ADF indicates stationarity (p={adf_res.p_value:.4e} < {alpha}) "
            f"while KPSS indicates non-stationarity (p={kpss_res.p_value:.4f} < {alpha}). "
            f"Series may be trend-stationary or contain structural regime shifts."
        )

    series_name = name or (
        series.name if isinstance(series, pd.Series) and series.name else "Series"
    )

    return StationarityReport(
        name=str(series_name),
        alpha=alpha,
        adf=adf_res,
        kpss=kpss_res,
        conclusion=conclusion,
        explanation=explanation,
    )


# ============================================================
# 4. Batch Stationarity Testing
# ============================================================


def test_multiple_series(
    dict_of_series: dict[str, pd.Series | np.ndarray],
    alpha: float = 0.05,
    regression: str = "c",
) -> pd.DataFrame:
    """Batch-test multiple series and return a structured summary DataFrame.

    Parameters
    ----------
    dict_of_series : dict[str, pd.Series | np.ndarray]
        Mapping from series identifier (e.g. 'AAPL_price', 'AAPL_return') to series data.
    alpha : float
        Significance level threshold (default: 0.05).
    regression : str
        Regression model ('c' for constant).

    Returns
    -------
    pd.DataFrame
        Summary table indexed by series name.
    """
    records: list[dict[str, Any]] = []

    for name, s in dict_of_series.items():
        try:
            report = stationarity_report(s, name=name, alpha=alpha, regression=regression)
            d = report.to_dict()
            records.append(
                {
                    "series": name,
                    "adf_stat": d["adf_statistic"],
                    "adf_pvalue": d["adf_pvalue"],
                    "adf_stationary": d["adf_stationary"],
                    "kpss_stat": d["kpss_statistic"],
                    "kpss_pvalue": d["kpss_pvalue"],
                    "kpss_stationary": d["kpss_stationary"],
                    "conclusion": d["conclusion"],
                }
            )
        except Exception as exc:
            logger.error("Failed stationarity test on series '%s': %s", name, exc)
            records.append(
                {
                    "series": name,
                    "adf_stat": np.nan,
                    "adf_pvalue": np.nan,
                    "adf_stationary": False,
                    "kpss_stat": np.nan,
                    "kpss_pvalue": np.nan,
                    "kpss_stationary": False,
                    "conclusion": f"ERROR: {exc}",
                }
            )

    df_out = pd.DataFrame(records).set_index("series")
    return df_out


# Prevent pytest from discovering this function as a test case
test_multiple_series.__test__ = False


# ============================================================
# 5. Differencing Utility
# ============================================================


def difference_series(
    series: pd.Series,
    order: int = 1,
) -> pd.Series:
    """Apply d-th order differencing to a time series to achieve stationarity.

    Formula for order=1:
      Delta y_t = y_t - y_{t-1}

    Parameters
    ----------
    series : pd.Series
        Input time series (e.g., raw prices or log prices).
    order : int
        Order of differencing (default: 1).

    Returns
    -------
    pd.Series
        Differenced series with initial NaNs dropped.
    """
    if order < 1:
        raise ValueError(f"Differencing order must be >= 1, got {order}.")

    diff = series.copy()
    for _ in range(order):
        diff = diff.diff()

    diff = diff.dropna()
    diff.name = f"{series.name or 'series'}_diff{order}"
    return diff
