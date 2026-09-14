# ============================================================
# Model Comparison & Statistical Significance Engine (Phase 28)
# ============================================================
"""
Unified model comparison and statistical hypothesis testing framework.

=============================================================================
THEORETICAL FOUNDATION: STATISTICAL SIGNIFICANCE IN QUANT FORECASTING
-----------------------------------------------------------------------------
In quantitative trading, comparing models solely on point estimates of Sharpe ratio
or directional accuracy is statistically naive and dangerous:
- Small sample sizes ($N \\approx 500 - 1,500$ bars) mean observed differences of
  1-2% accuracy or 0.20 in Sharpe can easily arise from pure chance.
- Financial return series exhibit fat tails, serial correlation in volatility,
  and non-normality, invalidating standard Student's t-tests.

We introduce two rigorous hypothesis testing methodologies:

1. The Diebold-Mariano (DM) Test (1995) with Harvey-Leybourne-Newbold (HLN) Correction:
   Tests the null hypothesis of equal predictive accuracy between two competing
   forecasts: $H_0: \\mathbb{E}[d_t] = 0$, where $d_t = L(e_{1,t}) - L(e_{2,t})$
   is the loss differential (e.g. Brier score or logloss).
   The test accounts for serial correlation in forecast errors up to horizon $h-1$
   via Newey-West autocovariance weighting. The HLN finite-sample correction
   adjusts the asymptotic test statistic for small sample sizes $T$.

2. Block Bootstrap Confidence Intervals for Sharpe Ratio Differences (Ledoit & Wolf 2008):
   Standard Sharpe confidence intervals assume i.i.d. Gaussian returns.
   We implement a moving block bootstrap (block size $b=10$ bars) that preserves
   empirical autocorrelation and volatility clustering.
   We resample $B = 1,000$ paired return paths, compute $\\Delta \\text{Sharpe}^{*(b)}$,
   and calculate the empirical 95% confidence interval $[\\Delta_{\\text{lower}}, \\Delta_{\\text{upper}}]$.
   If 0 falls inside the interval, the outperformance is statistically indistinguishable
   from zero at the $\\alpha = 0.05$ significance level.

3. Production Latency Profiling:
   Measures wall-clock training time and per-bar microsecond inference latency.
   A complex model providing a statistically insignificant +0.5% accuracy but
   requiring 50x more compute/latency is an architectural anti-pattern in production.
=============================================================================
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.models.financial_metrics import (
    calculate_sharpe_ratio,
    compute_financial_metrics,
    convert_predictions_to_strategy_returns,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Statistical Hypothesis Tests (Diebold-Mariano & Bootstrap Sharpe)
# ============================================================


def diebold_mariano_test(
    y_true: np.ndarray | pd.Series,
    y_pred1: np.ndarray | pd.Series,
    y_pred2: np.ndarray | pd.Series,
    loss_type: Literal["brier", "logloss", "directional"] = "brier",
    h: int = 1,
) -> tuple[float, float]:
    """Execute Diebold-Mariano test of equal predictive accuracy with HLN correction.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary directions (0 or 1).
    y_pred1 : array-like
        Forecast probabilities (or class predictions) from Model 1 (e.g. Baseline).
    y_pred2 : array-like
        Forecast probabilities (or class predictions) from Model 2 (e.g. Candidate).
    loss_type : {'brier', 'logloss', 'directional'}, default 'brier'
        Loss function used to compute error differential:
        - 'brier': Squared probability error $(p_t - y_t)^2$
        - 'logloss': Cross-entropy loss $-[y \\log p + (1-y) \\log(1-p)]$
        - 'directional': Binary 0-1 classification loss $I(\\hat{y} \\neq y)$
    h : int, default 1
        Forecast horizon in steps (h=1 for 1-step-ahead forecasts).

    Returns
    -------
    dm_statistic : float
        Harvey-Leybourne-Newbold adjusted test statistic (asymptotically standard normal).
    p_value : float
        Two-sided p-value. $p < 0.05$ rejects the null hypothesis of equal accuracy.
    """
    yt = np.asarray(y_true, dtype=float)
    p1 = np.asarray(y_pred1, dtype=float)
    p2 = np.asarray(y_pred2, dtype=float)

    n = len(yt)
    if len(p1) != n or len(p2) != n:
        raise ValueError(
            f"Length mismatch: y_true ({n}), y_pred1 ({len(p1)}), y_pred2 ({len(p2)})."
        )

    if n < 10:
        raise ValueError(f"Insufficient samples ({n}) for Diebold-Mariano test.")

    # 1. Compute loss series L1 and L2
    if loss_type == "brier":
        e1 = (p1 - yt) ** 2
        e2 = (p2 - yt) ** 2
    elif loss_type == "logloss":
        clip1 = np.clip(p1, 1e-7, 1.0 - 1e-7)
        clip2 = np.clip(p2, 1e-7, 1.0 - 1e-7)
        e1 = -(yt * np.log(clip1) + (1.0 - yt) * np.log(1.0 - clip1))
        e2 = -(yt * np.log(clip2) + (1.0 - yt) * np.log(1.0 - clip2))
    elif loss_type == "directional":
        pred_cls1 = (p1 >= 0.5).astype(int)
        pred_cls2 = (p2 >= 0.5).astype(int)
        e1 = (pred_cls1 != yt).astype(float)
        e2 = (pred_cls2 != yt).astype(float)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Loss differential: d_t = e1 - e2 (Positive d_t means Model 2 has lower error)
    d = e1 - e2
    d_mean = float(np.mean(d))

    # 2. Estimate autocovariance of loss differential (Newey-West truncation at lag h-1)
    gamma0 = float(np.var(d, ddof=0))
    if gamma0 <= 1e-12:
        return 0.0, 1.0

    gamma_sum = 0.0
    for k in range(1, h):
        if k < n:
            gamma_k = float(np.mean((d[k:] - d_mean) * (d[:-k] - d_mean)))
            # Bartlett kernel weighting
            weight = 1.0 - (k / h)
            gamma_sum += 2.0 * weight * gamma_k

    lr_variance = gamma0 + gamma_sum
    if lr_variance <= 1e-12:
        return 0.0, 1.0

    # Asymptotic DM statistic
    dm_stat = d_mean / np.sqrt(lr_variance / n)

    # 3. Harvey, Leybourne, and Newbold (HLN, 1997) small-sample modification factor
    hln_factor = np.sqrt((n + 1 - 2 * h + (h * (h - 1)) / n) / n)
    adjusted_dm = float(dm_stat * hln_factor)

    # Two-sided p-value using Student's t distribution with n-1 degrees of freedom
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(adjusted_dm), df=n - 1)))

    return adjusted_dm, p_val


def bootstrap_sharpe_difference(
    returns_model: np.ndarray | pd.Series,
    returns_baseline: np.ndarray | pd.Series,
    n_bootstrap: int = 1000,
    block_size: int = 10,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, tuple[float, float], float]:
    """Compute moving-block bootstrap confidence interval for the Sharpe ratio difference.

    Parameters
    ----------
    returns_model : array-like
        Periodic strategy return series for candidate model.
    returns_baseline : array-like
        Periodic strategy return series for baseline model.
    n_bootstrap : int, default 1000
        Number of bootstrap replications.
    block_size : int, default 10
        Length of moving blocks to preserve volatility clustering and autocorrelation.
    alpha : float, default 0.05
        Significance level (0.05 for 95% confidence interval).
    random_state : int, default 42

    Returns
    -------
    observed_diff : float
        Observed Sharpe ratio difference: Sharpe(model) - Sharpe(baseline).
    ci : tuple[float, float]
        (lower_bound, upper_bound) of the bootstrap confidence interval.
    p_value : float
        Empirical two-tailed p-value testing null hypothesis H0: Delta Sharpe = 0.
    """
    r_m = np.asarray(returns_model, dtype=float)
    r_b = np.asarray(returns_baseline, dtype=float)

    n = len(r_m)
    if len(r_b) != n or n < 20:
        raise ValueError(
            f"Returns series must be identical length and >= 20 bars (got {n}, {len(r_b)})."
        )

    # Observed difference
    sharpe_m = calculate_sharpe_ratio(r_m)
    sharpe_b = calculate_sharpe_ratio(r_b)
    observed_diff = float(sharpe_m - sharpe_b)

    rng = np.random.RandomState(random_state)
    boot_diffs = np.empty(n_bootstrap, dtype=float)

    n_blocks = int(np.ceil(n / block_size))

    for b in range(n_bootstrap):
        # Draw random block start indices
        start_indices = rng.randint(0, n - block_size + 1, size=n_blocks)
        sampled_indices = np.concatenate([np.arange(s, s + block_size) for s in start_indices])[:n]

        b_m = r_m[sampled_indices]
        b_b = r_b[sampled_indices]

        s_m = calculate_sharpe_ratio(b_m)
        s_b = calculate_sharpe_ratio(b_b)
        boot_diffs[b] = s_m - s_b

    # Percentile confidence interval
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_diffs, lower_pct))
    ci_upper = float(np.percentile(boot_diffs, upper_pct))

    # Empirical two-tailed p-value
    # Fraction of bootstrap differences with opposite sign or zero
    if observed_diff >= 0:
        p_val = float(2.0 * np.mean(boot_diffs <= 0.0))
    else:
        p_val = float(2.0 * np.mean(boot_diffs >= 0.0))
    p_val = min(1.0, max(0.0, p_val))

    return observed_diff, (ci_lower, ci_upper), p_val


# ============================================================
# 2. Computational Latency Profiler
# ============================================================


def profile_model_latency(
    model_or_fn: Any,
    sample_input: np.ndarray,
    y: np.ndarray | None = None,
    n_runs: int = 100,
    warmup_runs: int = 10,
    **kwargs: Any,
) -> float | tuple[float, float]:
    """Measure inference latency (us) and optionally training time (ms).

    Parameters
    ----------
    model_or_fn : object or callable
        Trained model instance or inference function.
    sample_input : np.ndarray
        Single input sample or training matrix X.
    y : np.ndarray, optional
        Target labels. If provided and model_or_fn has a fit method,
        train time is benchmarked in milliseconds.
    n_runs : int, default 100
    warmup_runs : int, default 10

    Returns
    -------
    latency_us : float
        If y is None, returns inference latency per sample in microseconds.
    (train_time_ms, latency_us) : tuple of floats
        If y is provided, returns (train_time_ms, latency_us).
    """
    if "n_inference_runs" in kwargs:
        n_runs = kwargs["n_inference_runs"]

    train_time_ms = 0.0
    if y is not None and hasattr(model_or_fn, "fit"):
        t_fit_start = time.perf_counter()
        model_or_fn.fit(sample_input, y)
        train_time_ms = (time.perf_counter() - t_fit_start) * 1000.0

    # Resolve predict function and test input
    if callable(model_or_fn):
        fn = model_or_fn
        test_sample = sample_input
    elif hasattr(model_or_fn, "predict_proba"):
        fn = model_or_fn.predict_proba
        test_sample = sample_input[:1] if sample_input.ndim >= 2 else sample_input
    elif hasattr(model_or_fn, "predict"):
        fn = model_or_fn.predict
        test_sample = sample_input[:1] if sample_input.ndim >= 2 else sample_input
    else:
        raise ValueError("model_or_fn must be callable or possess predict/predict_proba method.")

    # Warmup
    for _ in range(warmup_runs):
        _ = fn(test_sample)

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = fn(test_sample)
    t_total = time.perf_counter() - t0

    latency_us = (t_total / n_runs) * 1_000_000.0
    if y is not None:
        return float(train_time_ms), float(latency_us)
    return float(latency_us)


# ============================================================
# 3. Unified Model Comparison Framework
# ============================================================


class ModelComparisonHarness:
    """Unified evaluation framework executing side-by-side model tournaments.

    Computes statistical ML metrics, financial strategy metrics,
    significance testing against a designated baseline (e.g. XGBoost),
    and runtime efficiency benchmarks across the same walk-forward splits.
    """

    def __init__(
        self,
        baseline_model_name: str = "Tuned XGBoost",
        significance_alpha: float = 0.05,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.baseline_model_name = baseline_model_name
        self.significance_alpha = significance_alpha
        self.risk_free_rate = risk_free_rate
        self.records_: list[dict[str, Any]] = []

    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        y_prob: np.ndarray | pd.Series | None,
        actual_returns: np.ndarray | pd.Series,
        training_time_sec: float = 0.0,
        inference_latency_us: float = 0.0,
    ) -> dict[str, Any]:
        """Evaluate a candidate model and compute its metric battery."""
        yt = np.asarray(y_true, dtype=int)
        yp = np.asarray(y_pred, dtype=int)
        rets = np.asarray(actual_returns, dtype=float)

        # 1. Statistical ML Metrics
        acc = float(np.mean(yp == yt))
        if y_prob is not None:
            p_arr = np.asarray(y_prob, dtype=float)
            if p_arr.ndim == 2:
                prob1 = p_arr[:, 1]
            else:
                prob1 = p_arr
            try:
                auc = float(
                    stats.rankdata(prob1)[yt == 1].sum()
                    - len(yt[yt == 1]) * (len(yt[yt == 1]) + 1) / 2.0
                )
                auc /= len(yt[yt == 1]) * len(yt[yt == 0])
            except Exception:
                auc = 0.5
            brier = float(np.mean((prob1 - yt) ** 2))
        else:
            prob1 = yp.astype(float)
            auc = 0.5
            brier = float(np.mean((yp - yt) ** 2))

        # 2. Financial Strategy Metrics
        strat_rets = convert_predictions_to_strategy_returns(yp, rets, signal_type="long_short")
        fin_metrics = compute_financial_metrics(strat_rets, risk_free_rate=self.risk_free_rate)

        record = {
            "model_name": model_name,
            "accuracy": acc,
            "roc_auc": auc,
            "brier_score": brier,
            "annualized_return": fin_metrics["annualized_return"],
            "annualized_volatility": fin_metrics["annualized_volatility"],
            "sharpe_ratio": fin_metrics["sharpe_ratio"],
            "sortino_ratio": fin_metrics["sortino_ratio"],
            "calmar_ratio": fin_metrics["calmar_ratio"],
            "max_drawdown": fin_metrics["max_drawdown"],
            "win_rate": fin_metrics["win_rate"],
            "profit_factor": fin_metrics["profit_factor"],
            "training_time_sec": training_time_sec,
            "inference_latency_us": inference_latency_us,
            "_prob": prob1,
            "_strat_rets": strat_rets.values if hasattr(strat_rets, "values") else strat_rets,
            "_y_true": yt,
        }

        self.records_.append(record)
        return record

    def register_model_results(
        self,
        model_name: str,
        y_true: np.ndarray | pd.Series,
        asset_returns: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series | None = None,
        y_pred_proba: np.ndarray | pd.Series | None = None,
        train_time_sec: float | None = None,
        train_time_ms: float | None = None,
        inference_latency_us: float = 0.0,
    ) -> dict[str, Any]:
        """Convenience method registering model predictions and returns."""
        if y_pred is None:
            if y_pred_proba is None:
                raise ValueError("Must provide either y_pred or y_pred_proba.")
            p_arr = np.asarray(y_pred_proba, dtype=float)
            p1 = p_arr[:, 1] if p_arr.ndim == 2 else p_arr
            y_pred = (p1 >= 0.5).astype(int)

        t_sec = train_time_sec if train_time_sec is not None else ((train_time_ms or 0.0) / 1000.0)
        return self.evaluate_model(
            model_name=model_name,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_pred_proba,
            actual_returns=asset_returns,
            training_time_sec=t_sec,
            inference_latency_us=inference_latency_us,
        )

    def build_comparison_report(self) -> pd.DataFrame:
        """Compile comparison table with Diebold-Mariano and Bootstrap significance markers."""
        if not self.records_:
            raise ValueError("No models evaluated yet.")

        # Find baseline record
        baseline_record = next(
            (r for r in self.records_ if r["model_name"] == self.baseline_model_name),
            None,
        )

        rows = []
        for r in self.records_:
            row: dict[str, Any] = {
                "Model": r["model_name"],
                "Accuracy": r["accuracy"],
                "ROC-AUC": r["roc_auc"],
                "Brier Loss": r["brier_score"],
                "Ann. Return": r["annualized_return"],
                "Sharpe": r["sharpe_ratio"],
                "Sortino": r["sortino_ratio"],
                "Calmar": r["calmar_ratio"],
                "Max DD": r["max_drawdown"],
                "Win Rate": r["win_rate"],
                "Profit Factor": r["profit_factor"],
                "Train Time (s)": r["training_time_sec"],
                "Latency (us/bar)": r["inference_latency_us"],
            }

            if baseline_record is not None and r["model_name"] != self.baseline_model_name:
                # 1. Diebold-Mariano test vs baseline
                dm_stat, p_dm = diebold_mariano_test(
                    r["_y_true"],
                    baseline_record["_prob"],
                    r["_prob"],
                    loss_type="brier",
                )

                # 2. Bootstrap Sharpe test vs baseline
                diff, (ci_l, ci_u), p_sharpe = bootstrap_sharpe_difference(
                    r["_strat_rets"],
                    baseline_record["_strat_rets"],
                    n_bootstrap=500,
                    random_state=42,
                )

                sig_mark = ""
                if p_dm < 0.01:
                    sig_mark = "***"
                elif p_dm < 0.05:
                    sig_mark = "**"
                elif p_dm < 0.10:
                    sig_mark = "*"

                row["DM Stat"] = dm_stat
                row["DM p-val"] = p_dm
                row["DM Sig"] = sig_mark
                row["Delta Sharpe"] = diff
                row["Sharpe CI (95%)"] = f"[{ci_l:.2f}, {ci_u:.2f}]"
                row["Sharpe p-val"] = p_sharpe
            elif r["model_name"] == self.baseline_model_name:
                row["DM Stat"] = 0.0
                row["DM p-val"] = 1.0
                row["DM Sig"] = "Baseline"
                row["Delta Sharpe"] = 0.0
                row["Sharpe CI (95%)"] = "[0.00, 0.00]"
                row["Sharpe p-val"] = 1.0
            else:
                row["DM Stat"] = np.nan
                row["DM p-val"] = np.nan
                row["DM Sig"] = "N/A"
                row["Delta Sharpe"] = np.nan
                row["Sharpe CI (95%)"] = "N/A"
                row["Sharpe p-val"] = np.nan

            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info("Compiled model comparison tournament table (%d models).", len(df))
        return df

    compile_tournament_table = build_comparison_report
