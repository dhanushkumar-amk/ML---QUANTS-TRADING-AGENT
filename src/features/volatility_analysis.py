# ============================================================
# Volatility Clustering Analysis Module (Phase 10)
# ============================================================
"""
Formal characterization of volatility clustering in financial return series.

Volatility clustering—the stylized empirical fact that "large changes tend to be
followed by large changes, of either sign, and small changes and by small changes"
(Mandelbrot, 1963)—is the foundational motivation for autoregressive conditional
heteroskedasticity (ARCH/GARCH) modeling.

Features:
  1. Volatility Proxies:
     - Squared returns: r_t^2
     - Absolute returns: |r_t|
  2. Autocorrelation Function (ACF):
     - Demonstrating negligible linear autocorrelation in raw returns: Corr(r_t, r_{t-k}) ~ 0
     - Demonstrating strong, persistent autocorrelation in squared returns: Corr(r_t^2, r_{t-k}^2) > 0
  3. Ljung-Box Q-Test on Squared Returns:
     - Formally tests H0: no autocorrelation in squared returns up to lag m.
  4. Engle's ARCH Lagrange Multiplier (ARCH-LM) Test:
     - Formally tests H0: no autoregressive conditional heteroskedasticity.
  5. Volatility Clustering Report:
     - Synthesizes ACF, Ljung-Box, and ARCH-LM into a structured pass/fail consensus verdict.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _clean_series(series: pd.Series | np.ndarray) -> np.ndarray:
    """Ensure series is a 1D numeric array without NaNs or Infs."""
    if isinstance(series, pd.Series):
        clean = series.dropna().replace([np.inf, -np.inf], np.nan).dropna().values
    else:
        clean = np.asarray(series, dtype=float)
        clean = clean[~np.isnan(clean) & ~np.isinf(clean)]

    if len(clean) < 15:
        raise ValueError(
            f"Series length ({len(clean)}) is too short for volatility clustering analysis. Minimum required is 15."
        )
    return clean


# ============================================================
# 1. Volatility Proxies
# ============================================================


def compute_volatility_proxies(returns: pd.Series) -> pd.DataFrame:
    """Compute standard volatility proxies for a return series.

    When integrated/realized volatility is not directly observable at high frequency,
    squared returns (r_t^2) and absolute returns (|r_t|) serve as unbiased and robust
    unobservable volatility proxies, respectively.

    Parameters
    ----------
    returns : pd.Series
        Clean arithmetic or log return series.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
          - 'raw_returns': r_t
          - 'squared_returns': r_t^2
          - 'abs_returns': |r_t|
    """
    clean_series = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    df = pd.DataFrame(
        {
            "raw_returns": clean_series,
            "squared_returns": clean_series**2,
            "abs_returns": clean_series.abs(),
        },
        index=clean_series.index,
    )
    return df


# ============================================================
# 2. Autocorrelation Function (ACF)
# ============================================================


def compute_acf_series(
    series: pd.Series | np.ndarray,
    nlags: int = 20,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute sample autocorrelation function and Bartlett asymptotic confidence intervals.

    Parameters
    ----------
    series : pd.Series | np.ndarray
        Input series (e.g. raw returns or squared returns).
    nlags : int
        Number of lags to compute (default: 20).
    alpha : float
        Significance level for confidence bounds (default: 0.05 for 95% interval).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (acf_values, confint) where confint is shape (nlags + 1, 2).
    """
    clean_vals = _clean_series(series)
    max_lags = min(nlags, len(clean_vals) - 1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        res = acf(clean_vals, nlags=max_lags, alpha=alpha)

    # In statsmodels, acf with alpha returns (acf_vals, confint)
    if isinstance(res, tuple) and len(res) >= 2:
        return res[0], res[1]
    return np.asarray(res), np.empty((0, 2))


# ============================================================
# 3. Ljung-Box Test on Squared Returns
# ============================================================


@dataclass
class LjungBoxResult:
    """Structured result for Ljung-Box autocorrelation test."""

    lags: list[int]
    statistics: dict[int, float] = field(default_factory=dict)
    p_values: dict[int, float] = field(default_factory=dict)

    def is_autocorrelated(self, lag: int | None = None, alpha: float = 0.05) -> bool:
        """True if test rejects H0 (no autocorrelation) at significance level alpha."""
        if lag is not None:
            pval = self.p_values.get(lag)
            if pval is None:
                raise KeyError(f"Lag {lag} not found in test results.")
            return pval < alpha
        # Default: check if any or all tested lags reject
        return all(p < alpha for p in self.p_values.values())


def ljung_box_test(
    series: pd.Series | np.ndarray,
    lags: int | list[int] | tuple[int, ...] = (10, 20),
) -> LjungBoxResult:
    """Run Ljung-Box Q-test for serial correlation in series (typically squared returns).

    Null Hypothesis (H0):
      The data are independently distributed (no autocorrelation up to lag m).
    Alternative Hypothesis (H1):
      The data exhibit serial correlation.

    Parameters
    ----------
    series : pd.Series | np.ndarray
        Series to test (e.g. squared returns r_t^2).
    lags : int | list[int] | tuple[int, ...]
        Specific lag orders or maximum lag order.

    Returns
    -------
    LjungBoxResult
    """
    clean_vals = _clean_series(series)

    if isinstance(lags, int):
        lag_list = [lags]
    else:
        lag_list = [int(lg) for lg in lags]

    # Constrain lags to series size
    max_allowed = len(clean_vals) - 1
    lag_list = [lg for lg in lag_list if lg <= max_allowed]
    if not lag_list:
        lag_list = [min(10, max_allowed)]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lb_df = acorr_ljungbox(clean_vals, lags=lag_list, return_df=True)

    stats = {int(k): float(v) for k, v in lb_df["lb_stat"].items()}
    pvals = {int(k): float(v) for k, v in lb_df["lb_pvalue"].items()}

    return LjungBoxResult(
        lags=lag_list,
        statistics=stats,
        p_values=pvals,
    )


# ============================================================
# 4. Engle's ARCH-LM Test
# ============================================================


@dataclass
class ARCHLMResult:
    """Structured result for Engle's ARCH Lagrange Multiplier test."""

    lags: int
    lm_stat: float
    lm_pvalue: float
    f_stat: float
    f_pvalue: float

    def has_arch_effects(self, alpha: float = 0.05) -> bool:
        """True if test rejects H0 (no ARCH effects) at significance level alpha."""
        return bool(self.lm_pvalue < alpha)


def arch_lm_test(
    returns: pd.Series | np.ndarray,
    lags: int = 10,
) -> ARCHLMResult:
    """Run Engle's ARCH Lagrange Multiplier test for conditional heteroskedasticity.

    Regresses squared residuals e_t^2 on lagged squared residuals:
      e_t^2 = alpha_0 + sum_{i=1}^q alpha_i e_{t-i}^2 + v_t

    Null Hypothesis (H0):
      alpha_1 = alpha_2 = ... = alpha_q = 0 (no ARCH effects / homoskedasticity).
    Alternative Hypothesis (H1):
      At least one alpha_i != 0 (ARCH effects present / volatility clustering).

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Zero-mean or de-meaned return series.
    lags : int
        Number of lags in ARCH-LM auxiliary regression (default: 10).

    Returns
    -------
    ARCHLMResult
    """
    clean_vals = _clean_series(returns)
    demeaned = clean_vals - np.mean(clean_vals)

    effective_lags = min(lags, (len(demeaned) // 3) - 1)
    if effective_lags < 1:
        effective_lags = 1

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        res = het_arch(demeaned, nlags=effective_lags)

    lm_stat = float(res[0])
    lm_pval = float(res[1])
    f_stat = float(res[2])
    f_pval = float(res[3])

    return ARCHLMResult(
        lags=effective_lags,
        lm_stat=lm_stat,
        lm_pvalue=lm_pval,
        f_stat=f_stat,
        f_pvalue=f_pval,
    )


# ============================================================
# 5. Combined Volatility Clustering Report
# ============================================================


@dataclass
class VolatilityClusteringReport:
    """Consensus diagnostic report characterizing volatility clustering in returns."""

    name: str
    alpha: float
    lags: int
    raw_acf_max: float
    sq_acf_max: float
    ljung_box_sq: LjungBoxResult
    arch_lm: ARCHLMResult
    is_clustering_confirmed: bool
    conclusion: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary format for tabular export."""
        lb_stat = next(iter(self.ljung_box_sq.statistics.values()), np.nan)
        lb_pval = next(iter(self.ljung_box_sq.p_values.values()), np.nan)

        return {
            "name": self.name,
            "alpha": self.alpha,
            "lags": self.lags,
            "raw_acf_lag1": self.raw_acf_max,
            "sq_acf_lag1": self.sq_acf_max,
            "ljung_box_stat": lb_stat,
            "ljung_box_pvalue": lb_pval,
            "arch_lm_stat": self.arch_lm.lm_stat,
            "arch_lm_pvalue": self.arch_lm.lm_pvalue,
            "is_clustering_confirmed": self.is_clustering_confirmed,
            "conclusion": self.conclusion,
            "explanation": self.explanation,
        }


def volatility_clustering_report(
    returns: pd.Series | np.ndarray,
    name: str = "",
    lags: int = 10,
    alpha: float = 0.05,
) -> VolatilityClusteringReport:
    """Run full diagnostic battery on returns to detect and verify volatility clustering.

    Combines:
      1. ACF on raw returns (expected ~ 0).
      2. ACF on squared returns (expected > 0).
      3. Ljung-Box test on squared returns.
      4. Engle ARCH-LM test on residuals.

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Clean return series.
    name : str
        Asset name or identifier.
    lags : int
        Diagnostic lag horizon (default: 10).
    alpha : float
        Significance level threshold (default: 0.05).

    Returns
    -------
    VolatilityClusteringReport
    """
    clean_vals = _clean_series(returns)
    series_name = name or (
        returns.name if isinstance(returns, pd.Series) and returns.name else "Series"
    )

    # 1. ACF
    raw_acf, _ = compute_acf_series(clean_vals, nlags=lags, alpha=alpha)
    sq_acf, _ = compute_acf_series(clean_vals**2, nlags=lags, alpha=alpha)

    # Lag 1 ACF (ignoring lag 0 which is 1.0)
    raw_lag1 = float(raw_acf[1]) if len(raw_acf) > 1 else 0.0
    sq_lag1 = float(sq_acf[1]) if len(sq_acf) > 1 else 0.0

    # 2. Ljung-Box on squared returns
    lb_res = ljung_box_test(clean_vals**2, lags=[lags])

    # 3. ARCH-LM test
    arch_res = arch_lm_test(clean_vals, lags=lags)

    lb_reject = lb_res.is_autocorrelated(lag=lags, alpha=alpha)
    arch_reject = arch_res.has_arch_effects(alpha=alpha)

    if lb_reject and arch_reject:
        confirmed = True
        conclusion = "confirmed"
        explanation = (
            f"Strong statistical evidence of volatility clustering: ARCH-LM test decisively rejects "
            f"homoskedasticity (LM stat={arch_res.lm_stat:.2f}, p={arch_res.lm_pvalue:.4e} < {alpha}) "
            f"and Ljung-Box test rejects independence in squared returns (Q stat={lb_res.statistics[lags]:.2f}, "
            f"p={lb_res.p_values[lags]:.4e} < {alpha}). This formally justifies GARCH-family modeling."
        )
    elif not lb_reject and not arch_reject:
        confirmed = False
        conclusion = "absent"
        explanation = (
            f"No statistical evidence of volatility clustering: ARCH-LM fails to reject homoskedasticity "
            f"(LM stat={arch_res.lm_stat:.2f}, p={arch_res.lm_pvalue:.4f} >= {alpha}) and squared returns "
            f"show no serial correlation (Ljung-Box p={lb_res.p_values[lags]:.4f} >= {alpha})."
        )
    else:
        confirmed = False
        conclusion = "inconclusive"
        explanation = (
            f"Mixed evidence: ARCH-LM reject={arch_reject} (p={arch_res.lm_pvalue:.4f}), "
            f"Ljung-Box squared reject={lb_reject} (p={lb_res.p_values[lags]:.4f})."
        )

    return VolatilityClusteringReport(
        name=str(series_name),
        alpha=alpha,
        lags=lags,
        raw_acf_max=raw_lag1,
        sq_acf_max=sq_lag1,
        ljung_box_sq=lb_res,
        arch_lm=arch_res,
        is_clustering_confirmed=confirmed,
        conclusion=conclusion,
        explanation=explanation,
    )


# ============================================================
# 6. Batch Evaluation Utility
# ============================================================


def batch_volatility_clustering(
    dict_of_returns: dict[str, pd.Series | np.ndarray],
    lags: int = 10,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Batch-evaluate multiple return series for volatility clustering.

    Parameters
    ----------
    dict_of_returns : dict[str, pd.Series | np.ndarray]
        Mapping from ticker/symbol to return series.
    lags : int
        Lag horizon for diagnostics (default: 10).
    alpha : float
        Significance level threshold (default: 0.05).

    Returns
    -------
    pd.DataFrame
        Summary table indexed by asset name.
    """
    records: list[dict[str, Any]] = []

    for name, s in dict_of_returns.items():
        try:
            report = volatility_clustering_report(s, name=name, lags=lags, alpha=alpha)
            d = report.to_dict()
            records.append(
                {
                    "ticker": name,
                    "raw_acf_lag1": d["raw_acf_lag1"],
                    "sq_acf_lag1": d["sq_acf_lag1"],
                    "ljung_box_stat": d["ljung_box_stat"],
                    "ljung_box_pvalue": d["ljung_box_pvalue"],
                    "arch_lm_stat": d["arch_lm_stat"],
                    "arch_lm_pvalue": d["arch_lm_pvalue"],
                    "clustering_confirmed": d["is_clustering_confirmed"],
                    "conclusion": d["conclusion"],
                }
            )
        except Exception as exc:
            logger.error("Failed volatility clustering test on '%s': %s", name, exc)
            records.append(
                {
                    "ticker": name,
                    "raw_acf_lag1": np.nan,
                    "sq_acf_lag1": np.nan,
                    "ljung_box_stat": np.nan,
                    "ljung_box_pvalue": np.nan,
                    "arch_lm_stat": np.nan,
                    "arch_lm_pvalue": np.nan,
                    "clustering_confirmed": False,
                    "conclusion": f"ERROR: {exc}",
                }
            )

    df_out = pd.DataFrame(records).set_index("ticker")
    return df_out


# Prevent pytest from treating this function as a test case
batch_volatility_clustering.__test__ = False
