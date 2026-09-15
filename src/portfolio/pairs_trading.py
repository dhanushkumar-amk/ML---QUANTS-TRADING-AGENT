# ============================================================
# Cointegration & Pairs Trading Engine (Phase 38)
# ============================================================
"""
Production Cointegration and Statistical Arbitrage Pairs Trading Engine.

=============================================================================
THEORETICAL FOUNDATION: COINTEGRATION VS. CORRELATION IN PAIRS TRADING
-----------------------------------------------------------------------------
A foundational mistake in quantitative trading is conflating return correlation
with price-level cointegration:

1. Return Correlation:
   - Measures co-movement of periodic returns: Corr(Delta P_A, Delta P_B).
   - High correlation (e.g., rho = 0.95) does NOT prevent two price series from
     drifting arbitrarily far apart over time.
   - Example: If Asset A returns +10% annually and Asset B returns +5% annually,
     their daily returns can have a correlation of 0.99, yet the ratio P_A / P_B
     will grow exponentially toward infinity. Trading a divergence under the
     naive assumption that "prices must converge" leads to catastrophic losses.

2. Price-Level Cointegration:
   - Formulated by Engle & Granger (1987) and Johansen (1988, 1991).
   - Two series P_A(t) and P_B(t) are each integrated of order 1 (I(1) random walks),
     but there exists a linear combination:
         S_t = P_A(t) - beta * P_B(t) - alpha
     that is integrated of order 0 (I(0) stationary).
   - Because S_t is stationary:
     * It has a constant, time-invariant unconditional mean E[S_t] = mu.
     * It has finite variance and covariance that depends only on lag.
     * Any divergence S_t - mu is guaranteed to be transient and mean-reverting.
   - It is precisely this price-level long-run equilibrium relationship that a pairs
     trading strategy exploits.

3. Static vs. Dynamic (Rolling) Hedge Ratios:
   - In textbook examples, beta is estimated once via full-sample OLS.
   - In reality, the true economic relationship between two companies evolves
     due to differential revenue growth, capital structure changes, or sector shifts.
   - A static beta introduces severe look-ahead bias if fit over the full backtest,
     or structural breakdown risk if fit once on a past window.
   - This module provides rolling OLS hedge ratios (beta_t) computed strictly over
     past observations [t - W, t], ensuring temporal causality and adapting to
     regime changes.
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.features.mean_reversion_features import estimate_half_life
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Data Containers for Cointegration Results
# ============================================================


@dataclass
class CointegrationResult:
    """Container for pairwise cointegration test statistics and diagnostics."""

    asset_x: str
    asset_y: str
    is_cointegrated: bool
    p_value: float
    test_stat: float
    critical_values: dict[str, float]
    hedge_ratio: float
    intercept: float
    half_life: float
    method: str = "engle_granger"

    def summary_dict(self) -> dict[str, Any]:
        """Return clean serializable dictionary of test metrics."""
        return {
            "pair": f"{self.asset_x}/{self.asset_y}",
            "asset_x": self.asset_x,
            "asset_y": self.asset_y,
            "is_cointegrated": self.is_cointegrated,
            "p_value": round(float(self.p_value), 5),
            "test_stat": round(float(self.test_stat), 4),
            "critical_values": {k: round(float(v), 4) for k, v in self.critical_values.items()},
            "hedge_ratio": round(float(self.hedge_ratio), 4),
            "intercept": round(float(self.intercept), 4),
            "half_life": round(float(self.half_life), 2) if np.isfinite(self.half_life) else np.nan,
            "method": self.method,
        }


# ============================================================
# 2. Cointegration Testing & Screening
# ============================================================


def engle_granger_test(
    series_y: pd.Series | np.ndarray,
    series_x: pd.Series | np.ndarray,
    significance_level: float = 0.05,
) -> tuple[float, float, dict[str, float], float, float]:
    """Perform the Engle-Granger two-step cointegration test.

    Step 1: OLS cointegrating regression:
        y_t = alpha + beta * x_t + epsilon_t
    Step 2: Augmented Dickey-Fuller unit-root test on residuals epsilon_t.

    Parameters
    ----------
    series_y : pd.Series | np.ndarray
        Dependent price series (Asset Y).
    series_x : pd.Series | np.ndarray
        Independent price series (Asset X).
    significance_level : float, default 0.05
        Significance threshold for null rejection (e.g. 0.05 for 95% confidence).

    Returns
    -------
    test_stat : float
        ADF test statistic on regression residuals.
    p_value : float
        MacKinnon approximate p-value for the cointegration test.
    critical_values : dict[str, float]
        Critical values at 1%, 5%, and 10% levels.
    hedge_ratio : float
        OLS slope coefficient beta.
    intercept : float
        OLS constant alpha.
    """
    y = np.asarray(series_y, dtype=float)
    x = np.asarray(series_x, dtype=float)

    # Clean missing values pairwise
    valid = ~np.isnan(y) & ~np.isnan(x) & ~np.isinf(y) & ~np.isinf(x)
    y_clean = y[valid]
    x_clean = x[valid]

    if len(y_clean) < 30:
        raise ValueError(
            f"Insufficient data points for cointegration test ({len(y_clean)}). Minimum is 30."
        )

    # Step 1: OLS regression to get beta and alpha
    var_x = np.var(x_clean, ddof=1)
    if var_x == 0.0:
        return 0.0, 1.0, {"1%": 0.0, "5%": 0.0, "10%": 0.0}, 0.0, 0.0

    cov_xy = np.cov(y_clean, x_clean)[0, 1]
    hedge_ratio = float(cov_xy / var_x)
    intercept = float(np.mean(y_clean) - hedge_ratio * np.mean(x_clean))

    # Step 2: statsmodels coint test (y on x)
    stat, p_val, crit = coint(y_clean, x_clean)

    crit_dict = {"1%": float(crit[0]), "5%": float(crit[1]), "10%": float(crit[2])}
    return float(stat), float(p_val), crit_dict, hedge_ratio, intercept


def johansen_test(
    df_pair: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> dict[str, Any]:
    """Perform the Johansen cointegration test on a 2-asset price DataFrame.

    Parameters
    ----------
    df_pair : pd.DataFrame
        DataFrame with 2 price columns.
    det_order : int, default 0
        Deterministic trend order (-1 = no const, 0 = const in cointegration, 1 = const outside).
    k_ar_diff : int, default 1
        Lag order of difference terms in VAR.

    Returns
    -------
    dict
        Dictionary containing trace statistics, eigenvalue statistics,
        critical values (90%, 95%, 99%), and boolean cointegration flags.
    """
    if df_pair.shape[1] != 2:
        raise ValueError(f"Johansen test expects 2 columns for a pair, got {df_pair.shape[1]}")

    clean = df_pair.dropna().astype(float)
    if len(clean) < 30:
        raise ValueError(f"Insufficient samples for Johansen test ({len(clean)}). Minimum is 30.")

    res = coint_johansen(clean.values, det_order=det_order, k_ar_diff=k_ar_diff)

    # r = 0 test: null hypothesis of 0 cointegrating vectors vs >= 1
    trace_stat_r0 = float(res.lr1[0])
    trace_crit_95_r0 = float(res.cvt[0, 1])  # index 1 is 95%
    eigen_stat_r0 = float(res.lr2[0])
    eigen_crit_95_r0 = float(res.cvm[0, 1])

    is_coint_trace = trace_stat_r0 > trace_crit_95_r0
    is_coint_eigen = eigen_stat_r0 > eigen_crit_95_r0

    return {
        "trace_stat": trace_stat_r0,
        "trace_crit_95": trace_crit_95_r0,
        "eigen_stat": eigen_stat_r0,
        "eigen_crit_95": eigen_crit_95_r0,
        "is_cointegrated_trace": is_coint_trace,
        "is_cointegrated_eigen": is_coint_eigen,
        "is_cointegrated": is_coint_trace and is_coint_eigen,
        "eigenvectors": res.evec,
    }


def screen_pairs(
    prices_df: pd.DataFrame,
    method: str = "engle_granger",
    p_value_threshold: float = 0.05,
    min_half_life: float = 1.0,
    max_half_life: float = 126.0,
) -> list[CointegrationResult]:
    """Screen all unique pairs in a price DataFrame for cointegration.

    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame where columns are ticker symbols and rows are synchronized prices.
    method : str, default 'engle_granger'
        Cointegration screening method ('engle_granger' or 'johansen').
    p_value_threshold : float, default 0.05
        Maximum p-value to qualify as statistically cointegrated.
    min_half_life : float, default 1.0
        Minimum allowable mean-reversion half-life (bars).
    max_half_life : float, default 126.0
        Maximum allowable mean-reversion half-life (bars, ~6 months).

    Returns
    -------
    list[CointegrationResult]
        Ranked list of cointegration results (sorted by p-value ascending).
    """
    tickers = list(prices_df.columns)
    n_tickers = len(tickers)
    if n_tickers < 2:
        raise ValueError(f"Need at least 2 tickers to screen pairs, got {n_tickers}")

    results: list[CointegrationResult] = []

    for i in range(n_tickers):
        for j in range(i + 1, n_tickers):
            t_y = tickers[i]
            t_x = tickers[j]

            s_y = prices_df[t_y].dropna()
            s_x = prices_df[t_x].dropna()
            common_idx = s_y.index.intersection(s_x.index)
            if len(common_idx) < 30:
                continue

            y_aligned = s_y.loc[common_idx]
            x_aligned = s_x.loc[common_idx]

            try:
                stat, p_val, crits, hedge_ratio, intercept = engle_granger_test(
                    y_aligned, x_aligned, significance_level=p_value_threshold
                )

                # Compute spread to estimate half-life
                spread = y_aligned - hedge_ratio * x_aligned - intercept
                try:
                    hl = estimate_half_life(spread)
                except Exception:
                    hl = np.inf

                is_coint = False
                if method == "engle_granger":
                    is_coint = (
                        (p_val <= p_value_threshold)
                        and np.isfinite(hl)
                        and (min_half_life <= hl <= max_half_life)
                    )
                elif method == "johansen":
                    joh_res = johansen_test(pd.DataFrame({t_y: y_aligned, t_x: x_aligned}))
                    is_coint = (
                        joh_res["is_cointegrated"]
                        and np.isfinite(hl)
                        and (min_half_life <= hl <= max_half_life)
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")

                result = CointegrationResult(
                    asset_x=t_x,
                    asset_y=t_y,
                    is_cointegrated=is_coint,
                    p_value=float(p_val),
                    test_stat=float(stat),
                    critical_values=crits,
                    hedge_ratio=float(hedge_ratio),
                    intercept=float(intercept),
                    half_life=float(hl),
                    method=method,
                )
                results.append(result)
            except Exception as e:
                logger.warning("Error testing pair (%s, %s): %s", t_y, t_x, e)
                continue

    # Sort results: cointegrated first, then lowest p-value
    results.sort(key=lambda r: (not r.is_cointegrated, r.p_value))
    return results


def pairs_results_to_dataframe(results: Sequence[CointegrationResult]) -> pd.DataFrame:
    """Convert a sequence of CointegrationResult objects to a pandas DataFrame."""
    rows = [r.summary_dict() for r in results]
    if not rows:
        return pd.DataFrame(
            columns=[
                "pair",
                "asset_x",
                "asset_y",
                "is_cointegrated",
                "p_value",
                "test_stat",
                "critical_values",
                "hedge_ratio",
                "intercept",
                "half_life",
                "method",
            ]
        )
    return pd.DataFrame(rows)


# ============================================================
# 3. Dynamic Hedge Ratio & Spread Construction
# ============================================================


def compute_static_hedge_ratio(
    series_y: pd.Series | np.ndarray,
    series_x: pd.Series | np.ndarray,
) -> tuple[float, float]:
    """Compute static full-sample OLS hedge ratio (beta) and intercept (alpha).

    y = alpha + beta * x + epsilon
    """
    y = np.asarray(series_y, dtype=float)
    x = np.asarray(series_x, dtype=float)
    mask = ~np.isnan(y) & ~np.isnan(x)
    y_c = y[mask]
    x_c = x[mask]
    var_x = np.var(x_c, ddof=1)
    if var_x == 0.0:
        return 1.0, 0.0
    beta = float(np.cov(y_c, x_c)[0, 1] / var_x)
    alpha = float(np.mean(y_c) - beta * np.mean(x_c))
    return beta, alpha


def compute_rolling_hedge_ratio(
    series_y: pd.Series,
    series_x: pd.Series,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute rolling OLS hedge ratio (beta_t) and intercept (alpha_t) over time.

    STRICT TEMPORAL INTEGRITY:
    --------------------------
    For bar t, beta_t and alpha_t are fit STRICTLY using observations in [t - window + 1, t].
    This eliminates lookahead bias and adapts to time-varying cointegrating vectors.

    Parameters
    ----------
    series_y : pd.Series
        Dependent asset price series.
    series_x : pd.Series
        Independent asset price series.
    window : int, default 60
        Lookback window in trading bars.
    min_periods : int, optional
        Minimum number of observations required to compute beta. Defaults to window // 2.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['hedge_ratio', 'intercept'] indexed matching series_y.
    """
    if min_periods is None:
        min_periods = max(20, window // 2)

    df = pd.DataFrame({"y": series_y, "x": series_x}).dropna()
    rolling_cov = df["y"].rolling(window=window, min_periods=min_periods).cov(df["x"])
    rolling_var = df["x"].rolling(window=window, min_periods=min_periods).var()

    # Avoid division by zero
    rolling_var_safe = rolling_var.replace(0.0, np.nan)
    beta = rolling_cov / rolling_var_safe

    rolling_mean_y = df["y"].rolling(window=window, min_periods=min_periods).mean()
    rolling_mean_x = df["x"].rolling(window=window, min_periods=min_periods).mean()
    alpha = rolling_mean_y - beta * rolling_mean_x

    # Forward fill initial burn-in period with the first valid estimated beta/alpha
    beta = beta.bfill()
    alpha = alpha.bfill()

    return pd.DataFrame({"hedge_ratio": beta, "intercept": alpha}, index=df.index)


def compute_spread(
    series_y: pd.Series,
    series_x: pd.Series,
    hedge_ratio: float | pd.Series,
    intercept: float | pd.Series = 0.0,
) -> pd.Series:
    """Compute the cointegrated spread series.

    Spread_t = Y_t - (beta_t * X_t + alpha_t)

    Parameters
    ----------
    series_y : pd.Series
        Asset Y price series.
    series_x : pd.Series
        Asset X price series.
    hedge_ratio : float | pd.Series
        Static beta or rolling series of betas.
    intercept : float | pd.Series, default 0.0
        Static alpha or rolling series of alphas.

    Returns
    -------
    pd.Series
        Clean spread series.
    """
    spread = series_y - (hedge_ratio * series_x + intercept)
    spread.name = "spread"
    return spread


def compute_spread_zscore(
    spread: pd.Series,
    window: int = 30,
    min_periods: int | None = None,
) -> pd.Series:
    """Compute rolling Z-score of the spread relative to its rolling mean and std.

    Z_t = (S_t - mu_t) / sigma_t

    Parameters
    ----------
    spread : pd.Series
        Raw spread series.
    window : int, default 30
        Lookback window for mean and standard deviation.
    min_periods : int, optional
        Minimum periods. Defaults to window // 2.

    Returns
    -------
    pd.Series
        Spread Z-score series.
    """
    if min_periods is None:
        min_periods = max(10, window // 2)

    rolling = spread.rolling(window=window, min_periods=min_periods)
    mean = rolling.mean()
    std = rolling.std(ddof=1).replace(0.0, np.nan)

    z = (spread - mean) / std
    z = z.mask(std.isna(), 0.0).fillna(0.0)
    z.name = "zscore"
    return z


# ============================================================
# 4. Pairs Trading Signal Logic & Execution
# ============================================================


def generate_pairs_signals(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_loss_threshold: float = 3.5,
) -> pd.Series:
    """Generate stateful market-neutral pairs trading positions from spread Z-scores.

    State Machine Rules:
    --------------------
    - Spread Long (+1.0):
      When Z_t <= -entry_threshold (Asset Y is cheap relative to Asset X).
      Position: Buy Asset Y, Short beta * Asset X.
    - Spread Short (-1.0):
      When Z_t >= +entry_threshold (Asset Y is rich relative to Asset X).
      Position: Short Asset Y, Buy beta * Asset X.
    - Exit / Mean-Reversion (0.0):
      * Long spread exits when Z_t >= -exit_threshold (reverts towards or crosses mean).
      * Short spread exits when Z_t <= +exit_threshold (reverts towards or crosses mean).
    - Stop-Loss (0.0):
      * Long spread stopped out if divergence widens: Z_t <= -stop_loss_threshold.
      * Short spread stopped out if divergence widens: Z_t >= +stop_loss_threshold.

    Parameters
    ----------
    zscore : pd.Series
        Rolling Z-score of spread.
    entry_threshold : float, default 2.0
        Z-score magnitude to enter position.
    exit_threshold : float, default 0.0
        Z-score magnitude to exit position upon mean reversion.
    stop_loss_threshold : float, default 3.5
        Z-score magnitude to trigger catastrophic risk exit.

    Returns
    -------
    pd.Series
        Signal series taking values in {+1.0, 0.0, -1.0}.
    """
    n = len(zscore)
    signals = np.zeros(n, dtype=float)
    z_vals = zscore.values

    current_pos = 0.0

    for i in range(n):
        z = z_vals[i]

        if np.isnan(z):
            signals[i] = 0.0
            current_pos = 0.0
            continue

        if current_pos == 0.0:
            # Look for entry
            if z <= -entry_threshold:
                current_pos = 1.0  # Go Long Spread
            elif z >= entry_threshold:
                current_pos = -1.0  # Go Short Spread
        elif current_pos == 1.0:
            # Currently Long Spread
            # Stop loss check
            if z <= -stop_loss_threshold:
                current_pos = 0.0
            # Mean-reversion exit check
            elif z >= -exit_threshold:
                current_pos = 0.0
        elif current_pos == -1.0:
            # Currently Short Spread
            # Stop loss check
            if z >= stop_loss_threshold:
                current_pos = 0.0
            # Mean-reversion exit check
            elif z <= exit_threshold:
                current_pos = 0.0

        signals[i] = current_pos

    return pd.Series(signals, index=zscore.index, name="pair_signal")


# ============================================================
# 5. Pairs Strategy Backtesting Engine
# ============================================================


@dataclass
class PairsBacktestResult:
    """Container for backtest results of a pairs trading strategy."""

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    portfolio_equity: pd.Series
    spread_pnl: pd.Series
    weights_y: pd.Series
    weights_x: pd.Series
    trade_log: list[dict[str, Any]]

    def summary_dict(self) -> dict[str, Any]:
        """Summary dictionary for metrics reporting."""
        return {
            "total_return_pct": round(self.total_return * 100, 2),
            "annualized_return_pct": round(self.annualized_return * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "win_rate_pct": round(self.win_rate * 100, 2),
            "total_trades": self.total_trades,
        }


def backtest_pairs_strategy(
    price_y: pd.Series,
    price_x: pd.Series,
    signals: pd.Series,
    hedge_ratio: float | pd.Series,
    initial_capital: float = 100000.0,
    commission_pct: float = 0.0005,
) -> PairsBacktestResult:
    """Run an event-driven / bar-by-bar backtest of a pairs trading strategy.

    Maintains Dollar-Neutral or Beta-Neutral Weights:
    - Long Spread (+1): Weight_Y = +0.5, Weight_X = -0.5 * (beta_t / |beta_t| or proportional)
      Specifically: Dollar-Neutral allocation: w_Y = +0.5, w_X = -0.5.
    - Short Spread (-1): w_Y = -0.5, w_X = +0.5.
    - Flat (0): w_Y = 0.0, w_X = 0.0.

    Parameters
    ----------
    price_y : pd.Series
        Price series of Asset Y.
    price_x : pd.Series
        Price series of Asset X.
    signals : pd.Series
        Trading signal series in {+1, 0, -1}.
    hedge_ratio : float | pd.Series
        Hedge ratio beta.
    initial_capital : float, default 100000.0
        Starting portfolio cash.
    commission_pct : float, default 0.0005
        Transaction cost / slippage per trade turnover (5 bps).

    Returns
    -------
    PairsBacktestResult
        Complete backtest performance statistics and time series.
    """
    df = pd.DataFrame({"y": price_y, "x": price_x, "signal": signals}).dropna()

    if isinstance(hedge_ratio, pd.Series):
        df["beta"] = hedge_ratio.loc[df.index].fillna(1.0)
    else:
        df["beta"] = float(hedge_ratio)

    ret_y = df["y"].pct_change().fillna(0.0)
    ret_x = df["x"].pct_change().fillna(0.0)

    # Shift signal by 1 bar to prevent execution lookahead (signal generated on close t, executed on t+1)
    pos = df["signal"].shift(1).fillna(0.0)

    # Beta-neutral position sizing:
    # We allocate 50% gross capital to Asset Y, and (0.5 * beta) to Asset X normalized by (1 + |beta|)
    beta_abs = df["beta"].abs()
    norm = 1.0 + beta_abs
    w_y = (pos * 1.0) / norm
    w_x = (-pos * df["beta"]) / norm

    # Daily strategy return before costs
    strat_ret = w_y * ret_y + w_x * ret_x

    # Turnover costs on position change
    dw_y = w_y.diff().abs().fillna(w_y.abs())
    dw_x = w_x.diff().abs().fillna(w_x.abs())
    turnover = dw_y + dw_x
    costs = turnover * commission_pct

    net_ret = strat_ret - costs

    equity = initial_capital * (1.0 + net_ret).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    # Trade extraction for win rate
    trades: list[dict[str, Any]] = []
    in_trade = False
    entry_val = initial_capital
    entry_date = df.index[0]

    for t, p_val in pos.items():
        if not in_trade and p_val != 0.0:
            in_trade = True
            entry_val = equity.loc[t]
            entry_date = t
        elif in_trade and p_val == 0.0:
            in_trade = False
            exit_val = equity.loc[t]
            trade_pnl = exit_val - entry_val
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": t,
                    "pnl": trade_pnl,
                    "is_win": trade_pnl > 0,
                }
            )

    total_trades = len(trades)
    wins = sum(1 for tr in trades if tr["is_win"])
    win_rate = float(wins / total_trades) if total_trades > 0 else 0.0

    total_return = float((equity.iloc[-1] / initial_capital) - 1.0)
    n_days = max(1, len(equity))
    ann_return = float((1.0 + total_return) ** (252.0 / n_days) - 1.0)

    std_ret = float(net_ret.std())
    sharpe = float((net_ret.mean() / (std_ret + 1e-9)) * np.sqrt(252.0)) if std_ret > 0 else 0.0
    max_dd = float(drawdown.min())

    return PairsBacktestResult(
        total_return=total_return,
        annualized_return=ann_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        total_trades=total_trades,
        portfolio_equity=equity,
        spread_pnl=net_ret,
        weights_y=w_y,
        weights_x=w_x,
        trade_log=trades,
    )
