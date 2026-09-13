# ============================================================
# Autocorrelation, Momentum & Mean-Reversion Diagnostics (Phase 11)
# ============================================================
"""
Formal testing for first-moment autocorrelation, directional momentum,
and mean-reversion tendencies across financial return series and multiple time horizons.

Unlike Phase 10 (which tested second-moment nonlinear dependence r_t^2 for volatility clustering),
this module examines whether past returns linearly predict future directional movements:
  - ACF / PACF on raw returns: sample correlation and partial correlation bounds.
  - Ljung-Box test on raw returns: tests H0 of zero linear autocorrelation.
  - Lo-MacKinlay (1988) Variance Ratio test: compares multi-period variance against single-period
    variance to distinguish random walk (VR ~ 1) from mean-reversion (VR < 1) or momentum (VR > 1)
    under both homoskedasticity and heteroskedasticity.
  - Run length & streak analysis: Wald-Wolfowitz runs test measuring consecutive directional runs
    against theoretical i.i.d. benchmarks.
  - Multi-horizon evaluation across 1-day, 5-day (weekly), and 20-day (monthly) horizons.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _clean_series(series: pd.Series | np.ndarray) -> np.ndarray:
    """Ensure series is a 1D numeric array without NaNs or Infs."""
    if isinstance(series, pd.Series):
        clean = series.dropna().replace([np.inf, -np.inf], np.nan).dropna().values
    else:
        clean = np.asarray(series, dtype=float)
        clean = clean[~np.isnan(clean) & ~np.isinf(clean)]

    if len(clean) < 20:
        raise ValueError(
            f"Series length ({len(clean)}) is too short for autocorrelation diagnostics. Minimum required is 20."
        )
    return clean


# ============================================================
# 1. ACF & PACF on Raw Returns
# ============================================================


def compute_acf_pacf(
    returns: pd.Series | np.ndarray,
    nlags: int = 20,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ACF and PACF for raw returns with asymptotic confidence intervals.

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Clean return series.
    nlags : int
        Number of lags to compute (default: 20).
    alpha : float
        Significance level for confidence intervals (default: 0.05).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (acf_values, acf_confint, pacf_values, pacf_confint)
    """
    clean_vals = _clean_series(returns)
    max_lags = min(nlags, len(clean_vals) // 2 - 1)
    if max_lags < 1:
        max_lags = 1

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        acf_res = acf(clean_vals, nlags=max_lags, alpha=alpha)
        pacf_res = pacf(clean_vals, nlags=max_lags, alpha=alpha)

    acf_vals = acf_res[0] if isinstance(acf_res, tuple) else np.asarray(acf_res)
    acf_conf = acf_res[1] if isinstance(acf_res, tuple) else np.empty((0, 2))

    pacf_vals = pacf_res[0] if isinstance(pacf_res, tuple) else np.asarray(pacf_res)
    pacf_conf = pacf_res[1] if isinstance(pacf_res, tuple) else np.empty((0, 2))

    return acf_vals, acf_conf, pacf_vals, pacf_conf


# ============================================================
# 2. Ljung-Box Test on Raw Returns
# ============================================================


@dataclass
class LjungBoxRawResult:
    """Structured result for Ljung-Box test on raw returns."""

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
        return any(p < alpha for p in self.p_values.values())


def ljung_box_raw(
    returns: pd.Series | np.ndarray,
    lags: int | list[int] | tuple[int, ...] = (5, 10, 20),
) -> LjungBoxRawResult:
    """Run Ljung-Box Q-test for serial correlation in raw returns.

    Null Hypothesis (H0):
      Returns are uncorrelated up to lag m (martingale difference / random walk).
    Alternative Hypothesis (H1):
      Returns exhibit significant linear autocorrelation.

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Raw return series.
    lags : int | list[int] | tuple[int, ...]
        Specific lags to evaluate.

    Returns
    -------
    LjungBoxRawResult
    """
    clean_vals = _clean_series(returns)
    if isinstance(lags, int):
        lag_list = [lags]
    else:
        lag_list = [int(lg) for lg in lags]

    max_allowed = len(clean_vals) - 1
    lag_list = [lg for lg in lag_list if lg <= max_allowed]
    if not lag_list:
        lag_list = [min(5, max_allowed)]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lb_df = acorr_ljungbox(clean_vals, lags=lag_list, return_df=True)

    stats_dict = {int(k): float(v) for k, v in lb_df["lb_stat"].items()}
    pvals_dict = {int(k): float(v) for k, v in lb_df["lb_pvalue"].items()}

    return LjungBoxRawResult(
        lags=lag_list,
        statistics=stats_dict,
        p_values=pvals_dict,
    )


# ============================================================
# 3. Lo-MacKinlay (1988) Variance Ratio Test
# ============================================================


@dataclass
class VarianceRatioResult:
    """Structured result of Lo-MacKinlay Variance Ratio test for horizon k."""

    k: int
    vr: float
    z_homo: float
    pval_homo: float
    z_hetero: float
    pval_hetero: float
    interpretation: str  # "mean-reversion", "momentum", "random-walk"


def variance_ratio_test(
    prices_or_returns: pd.Series | np.ndarray,
    k_lags: int | list[int] | tuple[int, ...] = (2, 5, 10, 20),
    is_returns: bool = False,
) -> dict[int, VarianceRatioResult]:
    """Compute Lo & MacKinlay (1988) Variance Ratio test with homoskedastic and heteroskedastic statistics.

    Formula:
      VR(k) = (1/k * Var(r_t(k))) / Var(r_t(1))

    Where r_t(k) = p_t - p_{t-k} is the overlapping k-period return.

    Under the Random Walk Hypothesis:
      - VR(k) = 1.0
      - VR(k) < 1.0 suggests negative autocorrelation / mean-reversion.
      - VR(k) > 1.0 suggests positive autocorrelation / momentum.

    Computes:
      - z_homo: Test statistic assuming i.i.d. homoskedastic errors.
      - z_hetero: Heteroskedasticity-robust test statistic (vital for financial data).

    Parameters
    ----------
    prices_or_returns : pd.Series | np.ndarray
        Price series (default) or log return series.
    k_lags : int | list[int] | tuple[int, ...]
        Aggregation horizons k (e.g. 2, 5, 10, 20).
    is_returns : bool
        If True, treats input as returns and reconstructs cumulative log price.

    Returns
    -------
    dict[int, VarianceRatioResult]
        Mapping from horizon k to structured VarianceRatioResult.
    """
    clean_vals = _clean_series(prices_or_returns)

    if is_returns:
        # Reconstruct log price path: p_0 = 0, p_t = cumsum(returns)
        p = np.concatenate(([0.0], np.cumsum(clean_vals)))
    else:
        # Input is prices: take natural log
        p = np.log(clean_vals)

    n = len(p) - 1
    if n < 30:
        raise ValueError(
            f"Sample size n={n} is too small for variance ratio test. Minimum 30 observations required."
        )

    # 1-period returns
    r1 = p[1:] - p[:-1]
    mu = (p[-1] - p[0]) / n

    # Unbiased 1-period variance: sigma_a^2
    var_1 = np.sum((r1 - mu) ** 2) / (n - 1)

    if var_1 <= 1e-12:
        raise ValueError("Series variance is effectively zero; cannot compute variance ratio.")

    if isinstance(k_lags, int):
        k_list = [k_lags]
    else:
        k_list = [int(k) for k in k_lags]

    results: dict[int, VarianceRatioResult] = {}

    for k in k_list:
        if k < 2 or k >= n // 2:
            logger.warning("Skipping invalid horizon k=%d for n=%d", k, n)
            continue

        # Overlapping k-period returns: r_t(k) = p_t - p_{t-k}
        rk = p[k:] - p[:-k]
        m = k * (n - k + 1) * (1.0 - (k / n))
        var_k = np.sum((rk - (k * mu)) ** 2) / m

        vr = float(var_k / var_1)

        # 1. Homoskedastic variance: phi(k)
        phi_homo = (2.0 * (2 * k - 1) * (k - 1)) / (3.0 * k * n)
        se_homo = np.sqrt(phi_homo)
        z_homo = float((vr - 1.0) / se_homo)
        pval_homo = float(2.0 * (1.0 - stats.norm.cdf(abs(z_homo))))

        # 2. Heteroskedasticity-robust variance: theta(k) (Lo & MacKinlay 1988)
        # delta_j = sum_{t=j+1}^n (r_t - mu)^2 (r_{t-j} - mu)^2 / (sum_{t=1}^n (r_t - mu)^2)^2
        sq_dev = (r1 - mu) ** 2
        denom = np.sum(sq_dev) ** 2

        theta_hetero = 0.0
        for j in range(1, k):
            weight = (2.0 * (k - j) / k) ** 2
            delta_j = np.sum(sq_dev[j:] * sq_dev[:-j]) / denom
            theta_hetero += weight * delta_j

        se_hetero = np.sqrt(theta_hetero) if theta_hetero > 0 else se_homo
        z_hetero = float((vr - 1.0) / se_hetero)
        pval_hetero = float(2.0 * (1.0 - stats.norm.cdf(abs(z_hetero))))

        # Interpretation based on robust z-score at 5% significance (|z| > 1.96)
        if z_hetero < -1.96:
            interp = "mean-reversion"
        elif z_hetero > 1.96:
            interp = "momentum"
        else:
            interp = "random-walk"

        results[k] = VarianceRatioResult(
            k=k,
            vr=vr,
            z_homo=z_homo,
            pval_homo=pval_homo,
            z_hetero=z_hetero,
            pval_hetero=pval_hetero,
            interpretation=interp,
        )

    return results


# ============================================================
# 4. Run Length & Streak Analysis
# ============================================================


@dataclass
class RunLengthResult:
    """Structured result of run length and Wald-Wolfowitz runs test."""

    total_runs: int
    expected_runs: float
    variance_runs: float
    z_stat: float
    p_value: float
    n_positive: int
    n_negative: int
    avg_positive_streak: float
    max_positive_streak: int
    avg_negative_streak: float
    max_negative_streak: int
    theoretical_pos_streak: float
    theoretical_neg_streak: float
    streak_bias: str  # "longer than random", "shorter than random", "consistent with random"


def run_length_analysis(returns: pd.Series | np.ndarray) -> RunLengthResult:
    """Analyze consecutive positive/negative directional streaks and perform Wald-Wolfowitz runs test.

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Return observations.

    Returns
    -------
    RunLengthResult
    """
    clean_vals = _clean_series(returns)

    # Filter zero returns to classify binary sign
    non_zero = clean_vals[clean_vals != 0.0]
    if len(non_zero) < 15:
        non_zero = clean_vals

    signs = np.where(non_zero > 0, 1, -1)
    n = len(signs)
    n_pos = int(np.sum(signs == 1))
    n_neg = int(np.sum(signs == -1))

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "Series contains only positive or only negative returns; runs test is undefined."
        )

    # Count total runs (transitions + 1)
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))

    # Wald-Wolfowitz expected runs and variance under H0 (i.i.d. random walk)
    mu_r = 1.0 + (2.0 * n_pos * n_neg) / n
    var_r = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n)) / (n**2 * (n - 1))
    se_r = np.sqrt(var_r) if var_r > 0 else 1.0

    z_stat = float((runs - mu_r) / se_r)
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))

    # Compute streak lengths
    pos_streaks: list[int] = []
    neg_streaks: list[int] = []
    current_streak = 1
    current_sign = signs[0]

    for s in signs[1:]:
        if s == current_sign:
            current_streak += 1
        else:
            if current_sign == 1:
                pos_streaks.append(current_streak)
            else:
                neg_streaks.append(current_streak)
            current_sign = s
            current_streak = 1
    # Append final streak
    if current_sign == 1:
        pos_streaks.append(current_streak)
    else:
        neg_streaks.append(current_streak)

    avg_pos = float(np.mean(pos_streaks)) if pos_streaks else 0.0
    max_pos = int(np.max(pos_streaks)) if pos_streaks else 0
    avg_neg = float(np.mean(neg_streaks)) if neg_streaks else 0.0
    max_neg = int(np.max(neg_streaks)) if neg_streaks else 0

    # Theoretical average streak under i.i.d. assumption: 1 / (1 - p)
    p_up = n_pos / n
    p_down = n_neg / n
    theo_pos = float(1.0 / (1.0 - p_up)) if p_up < 1.0 else np.nan
    theo_neg = float(1.0 / (1.0 - p_down)) if p_down < 1.0 else np.nan

    # Z-statistic interpretation:
    # z < -1.96 => Fewer runs than expected => Longer streaks => Momentum / clustering of sign
    # z > +1.96 => More runs than expected => Shorter streaks => Frequent alternation / Mean reversion
    if z_stat < -1.96:
        streak_bias = "longer than random (momentum tendency)"
    elif z_stat > 1.96:
        streak_bias = "shorter than random (mean-reversion tendency)"
    else:
        streak_bias = "consistent with random"

    return RunLengthResult(
        total_runs=runs,
        expected_runs=float(mu_r),
        variance_runs=float(var_r),
        z_stat=z_stat,
        p_value=p_val,
        n_positive=n_pos,
        n_negative=n_neg,
        avg_positive_streak=avg_pos,
        max_positive_streak=max_pos,
        avg_negative_streak=avg_neg,
        max_negative_streak=max_neg,
        theoretical_pos_streak=theo_pos,
        theoretical_neg_streak=theo_neg,
        streak_bias=streak_bias,
    )


# ============================================================
# 5. Combined Autocorrelation Report
# ============================================================


@dataclass
class AutocorrelationReport:
    """Comprehensive diagnostic report on autocorrelation, momentum, and mean-reversion."""

    name: str
    horizon_days: int
    alpha: float
    acf_lag1: float
    pacf_lag1: float
    ljung_box: LjungBoxRawResult
    variance_ratios: dict[int, VarianceRatioResult]
    run_length: RunLengthResult
    classification: str  # "momentum-dominant", "mean-reversion-dominant", "no significant structure (random walk)"
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for tabular export."""
        vr_5 = self.variance_ratios.get(5)
        vr_5_val = vr_5.vr if vr_5 else np.nan
        vr_5_z = vr_5.z_hetero if vr_5 else np.nan
        vr_5_p = vr_5.pval_hetero if vr_5 else np.nan

        lb_10_stat = self.ljung_box.statistics.get(10, np.nan)
        lb_10_p = self.ljung_box.p_values.get(10, np.nan)

        return {
            "name": self.name,
            "horizon_days": self.horizon_days,
            "acf_lag1": self.acf_lag1,
            "pacf_lag1": self.pacf_lag1,
            "ljung_box_stat": lb_10_stat,
            "ljung_box_pvalue": lb_10_p,
            "vr_5": vr_5_val,
            "vr_5_z": vr_5_z,
            "vr_5_pvalue": vr_5_p,
            "runs_z": self.run_length.z_stat,
            "runs_pvalue": self.run_length.p_value,
            "classification": self.classification,
            "explanation": self.explanation,
        }


def autocorrelation_report(
    returns: pd.Series | np.ndarray,
    prices: pd.Series | np.ndarray | None = None,
    name: str = "",
    horizon_days: int = 1,
    alpha: float = 0.05,
) -> AutocorrelationReport:
    """Synthesize ACF, PACF, Ljung-Box, Variance Ratio, and Run Length into consensus classification.

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Return series at specified horizon.
    prices : pd.Series | np.ndarray | None
        Underlying price series (if available, used for variance ratio).
    name : str
        Asset name.
    horizon_days : int
        Observation horizon (e.g. 1, 5, 20).
    alpha : float
        Significance level threshold (default: 0.05).

    Returns
    -------
    AutocorrelationReport
    """
    clean_ret = _clean_series(returns)
    series_name = name or (
        returns.name if isinstance(returns, pd.Series) and returns.name else "Series"
    )

    # 1. ACF / PACF
    acf_vals, _, pacf_vals, _ = compute_acf_pacf(clean_ret, nlags=10, alpha=alpha)
    lag1_acf = float(acf_vals[1]) if len(acf_vals) > 1 else 0.0
    lag1_pacf = float(pacf_vals[1]) if len(pacf_vals) > 1 else 0.0

    # 2. Ljung-Box test
    lb_res = ljung_box_raw(clean_ret, lags=(5, 10, 20))

    # 3. Variance Ratio test
    if prices is not None:
        vr_dict = variance_ratio_test(prices, k_lags=(2, 5, 10, 20), is_returns=False)
    else:
        vr_dict = variance_ratio_test(clean_ret, k_lags=(2, 5, 10, 20), is_returns=True)

    # 4. Run length analysis
    runs_res = run_length_analysis(clean_ret)

    # Consensus Decision Logic:
    # Check VR at horizon 5 (standard weekly benchmark)
    vr_k5 = vr_dict.get(5)
    vr_is_mean_rev = bool(vr_k5 and vr_k5.z_hetero < -1.96)
    vr_is_momentum = bool(vr_k5 and vr_k5.z_hetero > 1.96)

    # Check runs test
    runs_is_mean_rev = bool(runs_res.z_stat > 1.96)
    runs_is_momentum = bool(runs_res.z_stat < -1.96)

    # Classification matrix
    if (vr_is_mean_rev or runs_is_mean_rev) and (lag1_acf < 0 or vr_k5.vr < 1.0):
        classification = "mean-reversion-dominant"
        explanation = (
            f"Statistically significant mean-reversion detected at {horizon_days}-day horizon: "
            f"VR(5)={vr_k5.vr:.3f} (z_hetero={vr_k5.z_hetero:.2f}, p={vr_k5.pval_hetero:.4f}), "
            f"lag-1 ACF={lag1_acf:.3f}. Past returns exhibit negative serial dependence."
        )
    elif (vr_is_momentum or runs_is_momentum) and (lag1_acf > 0 or vr_k5.vr > 1.0):
        classification = "momentum-dominant"
        explanation = (
            f"Statistically significant momentum detected at {horizon_days}-day horizon: "
            f"VR(5)={vr_k5.vr:.3f} (z_hetero={vr_k5.z_hetero:.2f}, p={vr_k5.pval_hetero:.4f}), "
            f"lag-1 ACF={lag1_acf:.3f}. Past returns exhibit positive persistence."
        )
    else:
        classification = "no significant structure (random walk)"
        explanation = (
            f"No statistically significant momentum or mean-reversion at {horizon_days}-day horizon: "
            f"VR(5)={vr_k5.vr:.3f} (p_hetero={vr_k5.pval_hetero:.3f}), Ljung-Box(10) p={lb_res.p_values.get(10, 1.0):.3f}, "
            f"lag-1 ACF={lag1_acf:.3f}. Returns are indistinguishable from a martingale difference random walk."
        )

    return AutocorrelationReport(
        name=str(series_name),
        horizon_days=horizon_days,
        alpha=alpha,
        acf_lag1=lag1_acf,
        pacf_lag1=lag1_pacf,
        ljung_box=lb_res,
        variance_ratios=vr_dict,
        run_length=runs_res,
        classification=classification,
        explanation=explanation,
    )


# ============================================================
# 6. Multi-Horizon and Multi-Asset Analysis
# ============================================================


def multi_horizon_autocorrelation_analysis(
    prices: pd.Series,
    name: str = "",
    horizons: tuple[int, ...] = (1, 5, 20),
    alpha: float = 0.05,
    non_overlapping: bool = True,
) -> dict[int, AutocorrelationReport]:
    """Run diagnostics on a single asset across multiple time horizons (e.g. 1d, 5d, 20d).

    Parameters
    ----------
    prices : pd.Series
        Closing price series with DatetimeIndex.
    name : str
        Ticker symbol.
    horizons : tuple[int, ...]
        Return aggregation horizons in days.
    alpha : float
        Significance level threshold.
    non_overlapping : bool
        If True (default), samples prices at step h to avoid mechanical moving-sum
        autocorrelation overlap. If False, computes rolling overlapping returns.

    Returns
    -------
    dict[int, AutocorrelationReport]
        Mapping from horizon (days) to AutocorrelationReport.
    """
    clean_p = prices.dropna()
    results: dict[int, AutocorrelationReport] = {}

    for h in horizons:
        if h == 1:
            price_sub = clean_p
            ret = clean_p.pct_change().dropna()
        else:
            if non_overlapping:
                price_sub = clean_p.iloc[::h]
                ret = price_sub.pct_change().dropna()
            else:
                price_sub = clean_p
                ret = (clean_p / clean_p.shift(h) - 1.0).dropna()

        rep = autocorrelation_report(
            returns=ret,
            prices=price_sub,
            name=name,
            horizon_days=h,
            alpha=alpha,
        )
        results[h] = rep

    return results


def multi_asset_autocorrelation_summary(
    dict_of_prices: dict[str, pd.Series],
    horizons: tuple[int, ...] = (1, 5, 20),
    alpha: float = 0.05,
    non_overlapping: bool = True,
) -> pd.DataFrame:
    """Batch-evaluate multiple assets across multiple horizons into a structured summary table.

    Parameters
    ----------
    dict_of_prices : dict[str, pd.Series]
        Mapping from ticker to closing price Series.
    horizons : tuple[int, ...]
        Return aggregation horizons in days.
    alpha : float
        Significance level threshold.
    non_overlapping : bool
        If True (default), samples non-overlapping periods.

    Returns
    -------
    pd.DataFrame
        Multi-index or tabular summary DataFrame.
    """
    records: list[dict[str, Any]] = []

    for ticker, prices in dict_of_prices.items():
        try:
            horizon_reports = multi_horizon_autocorrelation_analysis(
                prices=prices,
                name=ticker,
                horizons=horizons,
                alpha=alpha,
                non_overlapping=non_overlapping,
            )
            for _h, rep in horizon_reports.items():
                d = rep.to_dict()
                records.append(d)
        except Exception as exc:
            logger.error("Failed autocorrelation diagnostics on '%s': %s", ticker, exc)
            for h in horizons:
                records.append(
                    {
                        "name": ticker,
                        "horizon_days": h,
                        "acf_lag1": np.nan,
                        "pacf_lag1": np.nan,
                        "ljung_box_stat": np.nan,
                        "ljung_box_pvalue": np.nan,
                        "vr_5": np.nan,
                        "vr_5_z": np.nan,
                        "vr_5_pvalue": np.nan,
                        "runs_z": np.nan,
                        "runs_pvalue": np.nan,
                        "classification": f"ERROR: {exc}",
                        "explanation": str(exc),
                    }
                )

    df_out = pd.DataFrame(records).set_index(["name", "horizon_days"])
    return df_out


# Prevent pytest from treating this function as a test case
multi_asset_autocorrelation_summary.__test__ = False
