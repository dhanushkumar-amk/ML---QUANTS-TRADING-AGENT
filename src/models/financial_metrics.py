# ============================================================
# Financial Evaluation Metrics Engine (Phase 23)
# ============================================================
"""
Production financial evaluation metrics module for quantitative trading models.

=============================================================================
BRIDGING THE GAP: ML METRICS VS. FINANCIAL STRATEGY METRICS
-----------------------------------------------------------------------------
In quantitative trading, pure statistical metrics (Accuracy, ROC-AUC, LogLoss)
are necessary but fundamentally insufficient for assessing strategy viability:

1. Asymmetric Payoffs:
   A model with 60% accuracy can blow up an account if the 40% losing trades
   occur in fat-tailed drawdown regimes (e.g. losing 5% per loss while gaining
   1% per win yields a negative expected value despite 60% accuracy).
   Conversely, a trend-following model with 40% accuracy can be highly profitable
   if winners outpace losers by 3:1.

2. Path Dependency & Volatility Clustering:
   Standard ML evaluation assumes i.i.d. observations. Financial returns are
   serially correlated in second moments (volatility clustering) and path-dependent.
   Maximum drawdown and recovery duration determine whether an institutional or
   retail trader faces liquidation or margin calls.

3. Sharpe vs. Sortino Ratio:
   The Sharpe ratio penalizes both upside and downside volatility equally.
   In trading, upside volatility (large positive windfall gains) is desirable.
   The Sortino ratio replaces total volatility with downside semideviation
   (volatility of negative returns below a minimum acceptable return or risk-free
   rate), making it far more appropriate for skewed return distributions.

=============================================================================
IMPORTANT LIMITATION NOTICE: PRE-TRANSACTION COST RETURNS
-----------------------------------------------------------------------------
The strategy returns computed in this module do NOT include transaction costs,
exchange fees, or market impact / slippage. Real-world execution friction
will be comprehensively modeled in Phase 41 (Backtesting & Execution Engine).
Sharpe ratios reported here are strictly theoretical gross upper bounds.
=============================================================================
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Core Financial Performance Metrics
# ============================================================


def calculate_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate the annualized Sharpe ratio of a return series.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Periodic strategy return series (e.g. daily fractional returns).
    risk_free_rate : float, default 0.0
        Annualized risk-free rate (e.g. 0.04 for 4%).
    periods_per_year : int, default 252
        Annualization factor (252 for daily equity bars, 52 for weekly).

    Returns
    -------
    float
        Annualized Sharpe ratio. Returns 0.0 if standard deviation is zero.
    """
    clean_rets = np.asarray(returns, dtype=float)
    clean_rets = clean_rets[~np.isnan(clean_rets)]

    if len(clean_rets) < 2:
        return 0.0

    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_returns = clean_rets - rf_per_period

    std = float(np.std(excess_returns, ddof=1))
    if std <= 1e-12:
        return 0.0

    mean_excess = float(np.mean(excess_returns))
    return float(np.sqrt(periods_per_year) * (mean_excess / std))


def calculate_sortino_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate the annualized Sortino ratio using downside semideviation.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Periodic strategy return series.
    risk_free_rate : float, default 0.0
        Annualized risk-free rate.
    periods_per_year : int, default 252
        Annualization factor (252 for daily trading bars).

    Returns
    -------
    float
        Annualized Sortino ratio. Returns 0.0 if downside deviation is zero.
    """
    clean_rets = np.asarray(returns, dtype=float)
    clean_rets = clean_rets[~np.isnan(clean_rets)]

    if len(clean_rets) < 2:
        return 0.0

    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_returns = clean_rets - rf_per_period

    # Downside deviation: only penalize negative deviations below target/rf
    downside = np.minimum(excess_returns, 0.0)
    downside_var = float(np.mean(downside**2))
    downside_std = float(np.sqrt(downside_var))

    if downside_std <= 1e-12:
        return 0.0

    mean_excess = float(np.mean(excess_returns))
    return float(np.sqrt(periods_per_year) * (mean_excess / downside_std))


def calculate_drawdown_series(
    returns: pd.Series | np.ndarray,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate the cumulative wealth index, running peak, and percentage drawdown series.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Periodic strategy returns.

    Returns
    -------
    wealth_index : pd.Series
        Cumulative wealth starting at 1.0.
    peaks : pd.Series
        Running maximum of wealth index.
    drawdown : pd.Series
        Percentage drawdown relative to running peak: (wealth - peak) / peak.
    """
    if isinstance(returns, pd.Series):
        clean_rets = returns.dropna()
    else:
        arr = np.asarray(returns, dtype=float)
        valid = ~np.isnan(arr)
        clean_rets = pd.Series(arr[valid])

    if len(clean_rets) == 0:
        empty = pd.Series(dtype=float)
        return empty, empty, empty

    wealth_index = (1.0 + clean_rets).cumprod()
    peaks = wealth_index.cummax()
    drawdown = (wealth_index - peaks) / peaks

    return wealth_index, peaks, drawdown


def calculate_max_drawdown(
    returns: pd.Series | np.ndarray,
) -> tuple[float, int]:
    """Calculate the maximum peak-to-trough drawdown and the longest drawdown duration.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Periodic strategy returns.

    Returns
    -------
    max_drawdown : float
        Worst percentage drawdown (e.g. -0.22 for -22%). Always <= 0.0.
    max_drawdown_duration : int
        Maximum consecutive bars spent in drawdown below the prior high-water mark.
    """
    _, _, dd = calculate_drawdown_series(returns)
    if len(dd) == 0:
        return 0.0, 0

    max_dd = float(dd.min())

    # Calculate max duration under peak
    is_underwater = dd < 0.0
    underwater_runs = (~is_underwater).cumsum()
    durations = is_underwater.groupby(underwater_runs).cumsum()
    max_duration = int(durations.max()) if len(durations) > 0 else 0

    return max_dd, max_duration


def calculate_calmar_ratio(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Calculate the Calmar ratio (annualized return / abs(max drawdown)).

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Periodic return series.
    periods_per_year : int, default 252

    Returns
    -------
    float
        Calmar ratio. If max drawdown is 0, returns 0.0.
    """
    clean_rets = np.asarray(returns, dtype=float)
    clean_rets = clean_rets[~np.isnan(clean_rets)]

    if len(clean_rets) < 2:
        return 0.0

    max_dd, _ = calculate_max_drawdown(clean_rets)
    if abs(max_dd) <= 1e-12:
        return 0.0

    cumulative_return = float(np.prod(1.0 + clean_rets) - 1.0)
    n_years = len(clean_rets) / periods_per_year
    if n_years <= 0:
        return 0.0

    annualized_return = (1.0 + cumulative_return) ** (1.0 / n_years) - 1.0
    return float(annualized_return / abs(max_dd))


def calculate_win_rate_and_profit_factor(
    returns: pd.Series | np.ndarray,
) -> tuple[float, float]:
    """Calculate trading win rate and profit factor.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Periodic strategy return series.

    Returns
    -------
    win_rate : float
        Fraction of non-zero trading periods with positive returns (0.0 to 1.0).
    profit_factor : float
        Gross winning return sum divided by absolute gross losing return sum.
        Returns float('inf') if there are wins and zero losses; 0.0 if no wins.
    """
    clean_rets = np.asarray(returns, dtype=float)
    clean_rets = clean_rets[~np.isnan(clean_rets)]

    active_rets = clean_rets[clean_rets != 0.0]
    if len(active_rets) == 0:
        return 0.0, 0.0

    wins = active_rets[active_rets > 0.0]
    losses = active_rets[active_rets < 0.0]

    win_rate = float(len(wins) / len(active_rets))

    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0

    if gross_loss <= 1e-12:
        profit_factor = float("inf") if gross_profit > 0.0 else 0.0
    else:
        profit_factor = float(gross_profit / gross_loss)

    return win_rate, profit_factor


# ============================================================
# 2. Strategy Returns Generation
# ============================================================


def convert_predictions_to_strategy_returns(
    predictions: pd.Series | np.ndarray,
    actual_returns: pd.Series | np.ndarray,
    signal_type: Literal["long_only", "long_short"] = "long_short",
) -> pd.Series:
    """Convert model directional predictions into a gross strategy return series.

    NOTE:
    This function assumes execution at bar t close or t+1 open using predictions
    computed at bar t. The actual_returns series must represent the strictly
    future return corresponding to that prediction horizon (e.g. forward 1-day return).

    Parameters
    ----------
    predictions : pd.Series or np.ndarray
        Model binary predictions (1 for UP, 0 for DOWN).
    actual_returns : pd.Series or np.ndarray
        Actual realized forward return over the prediction period.
    signal_type : {'long_only', 'long_short'}, default 'long_short'
        - 'long_short': Pred 1 -> Long (+1), Pred 0 -> Short (-1)
        - 'long_only': Pred 1 -> Long (+1), Pred 0 -> Flat (0)

    Returns
    -------
    pd.Series
        Periodic strategy returns aligned by index.
    """
    if isinstance(predictions, pd.Series) and isinstance(actual_returns, pd.Series):
        common_idx = predictions.dropna().index.intersection(actual_returns.dropna().index)
        preds = predictions.loc[common_idx]
        rets = actual_returns.loc[common_idx]
        idx = common_idx
    else:
        preds = np.asarray(predictions, dtype=float)
        rets = np.asarray(actual_returns, dtype=float)
        if len(preds) != len(rets):
            raise ValueError(
                f"Predictions length ({len(preds)}) must match actual_returns ({len(rets)})."
            )
        idx = (
            predictions.index
            if isinstance(predictions, pd.Series)
            else pd.RangeIndex(len(predictions))
        )

    preds_arr = np.asarray(preds)
    rets_arr = np.asarray(rets)

    if signal_type == "long_short":
        # Binary 1 -> +1, Binary 0 -> -1
        positions = np.where(preds_arr == 1.0, 1.0, np.where(preds_arr == 0.0, -1.0, 0.0))
    elif signal_type == "long_only":
        # Binary 1 -> +1, Binary 0 -> 0.0
        positions = np.where(preds_arr == 1.0, 1.0, 0.0)
    else:
        raise ValueError(
            f"Invalid signal_type '{signal_type}'. Choose 'long_only' or 'long_short'."
        )

    strat_rets = positions * rets_arr
    return pd.Series(strat_rets, index=idx, name="strategy_return")


# ============================================================
# 3. Comprehensive Financial Metrics Report
# ============================================================


def compute_financial_metrics(
    strategy_returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute a complete battery of financial metrics for a strategy return series.

    Parameters
    ----------
    strategy_returns : pd.Series or np.ndarray
        Strategy returns series.
    benchmark_returns : pd.Series or np.ndarray, optional
        Buy-and-hold benchmark return series for relative comparisons.
    risk_free_rate : float, default 0.0
    periods_per_year : int, default 252

    Returns
    -------
    dict[str, float]
        Dictionary of financial metrics:
        - total_return: Cumulative gross strategy return
        - annualized_return: Annualized geometric return
        - annualized_volatility: Annualized standard deviation
        - sharpe_ratio: Annualized Sharpe
        - sortino_ratio: Annualized Sortino
        - calmar_ratio: Calmar ratio
        - max_drawdown: Maximum percentage drawdown
        - max_drawdown_duration: Maximum consecutive underwater bars
        - win_rate: Percentage of winning trading periods
        - profit_factor: Gross profit / gross loss
        - alpha: Annualized excess return vs benchmark (if provided)
        - beta: Market beta vs benchmark (if provided)
    """
    clean_rets = np.asarray(strategy_returns, dtype=float)
    clean_rets = clean_rets[~np.isnan(clean_rets)]

    n_bars = len(clean_rets)
    if n_bars < 2:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

    cum_return = float(np.prod(1.0 + clean_rets) - 1.0)
    n_years = n_bars / periods_per_year
    ann_return = float((1.0 + cum_return) ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0

    ann_vol = float(np.std(clean_rets, ddof=1) * np.sqrt(periods_per_year))
    sharpe = calculate_sharpe_ratio(clean_rets, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(clean_rets, risk_free_rate, periods_per_year)
    calmar = calculate_calmar_ratio(clean_rets, periods_per_year)
    max_dd, max_dd_dur = calculate_max_drawdown(clean_rets)
    win_rate, pf = calculate_win_rate_and_profit_factor(clean_rets)

    metrics = {
        "total_return": cum_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "max_drawdown_duration": max_dd_dur,
        "win_rate": win_rate,
        "profit_factor": pf,
    }

    if benchmark_returns is not None:
        b_rets = np.asarray(benchmark_returns, dtype=float)
        b_rets = b_rets[~np.isnan(b_rets)]
        if len(b_rets) == n_bars:
            cov_mat = np.cov(clean_rets, b_rets)
            var_b = cov_mat[1, 1]
            beta = float(cov_mat[0, 1] / var_b) if var_b > 1e-12 else 1.0
            cum_b = float(np.prod(1.0 + b_rets) - 1.0)
            ann_b = float((1.0 + cum_b) ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
            alpha = ann_return - (risk_free_rate + beta * (ann_b - risk_free_rate))
            metrics["alpha"] = float(alpha)
            metrics["beta"] = float(beta)

    return metrics


def financial_evaluation_report(
    oof_predictions_df: pd.DataFrame,
    actual_returns: pd.Series,
    signal_type: Literal["long_only", "long_short"] = "long_short",
    risk_free_rate: float = 0.0,
    model_name: str = "Model",
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate per-fold and aggregate financial metrics scorecard from out-of-sample predictions.

    Parameters
    ----------
    oof_predictions_df : pd.DataFrame
        DataFrame with at least 'y_pred' and 'fold' columns, indexed by datetime.
    actual_returns : pd.Series
        Forward returns corresponding to each prediction row.
    signal_type : {'long_only', 'long_short'}, default 'long_short'
    risk_free_rate : float, default 0.0
    model_name : str, default 'Model'

    Returns
    -------
    scorecard_df : pd.DataFrame
        Table summarizing financial metrics for each fold plus aggregate OOF.
    strat_returns : pd.Series
        Full chronological strategy return series across all test periods.
    """
    if "y_pred" not in oof_predictions_df.columns:
        raise ValueError("oof_predictions_df must contain 'y_pred' column.")

    aligned_idx = oof_predictions_df.index.intersection(actual_returns.index)
    df_aligned = oof_predictions_df.loc[aligned_idx]
    actual_aligned = actual_returns.loc[aligned_idx]

    strat_returns = convert_predictions_to_strategy_returns(
        df_aligned["y_pred"], actual_aligned, signal_type=signal_type
    )

    rows = []
    if "fold" in df_aligned.columns:
        for fold, group in df_aligned.groupby("fold"):
            f_rets = strat_returns.loc[group.index]
            f_metrics = compute_financial_metrics(f_rets, risk_free_rate=risk_free_rate)
            f_metrics["fold"] = f"Fold {fold}"
            f_metrics["n_bars"] = len(f_rets)
            rows.append(f_metrics)

    # Aggregate
    agg_metrics = compute_financial_metrics(strat_returns, risk_free_rate=risk_free_rate)
    agg_metrics["fold"] = "AGGREGATE"
    agg_metrics["n_bars"] = len(strat_returns)
    rows.append(agg_metrics)

    scorecard_df = pd.DataFrame(rows).set_index("fold")
    scorecard_df.insert(0, "model", model_name)

    logger.info(
        "Financial scorecard for %s [%s]: Aggregate Ann. Return=%.2f%%, Sharpe=%.2f, MaxDD=%.2f%%",
        model_name,
        signal_type,
        agg_metrics["annualized_return"] * 100,
        agg_metrics["sharpe_ratio"],
        agg_metrics["max_drawdown"] * 100,
    )

    return scorecard_df, strat_returns
