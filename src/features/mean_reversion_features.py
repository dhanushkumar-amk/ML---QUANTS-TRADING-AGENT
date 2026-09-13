# ============================================================
# Mean-Reversion Features Library (Phase 13)
# ============================================================
"""
Production-grade mean-reversion feature extraction library grounded in academic
finance literature and empirical autocorrelation/variance ratio diagnostics from Phase 11.

DATA LEAKAGE SAFEGUARDS (MANDATORY REQUIREMENT):
------------------------------------------------
1. Temporal Causality (No Look-Ahead):
   Every feature at row t is computed STRICTLY using market data observable
   at or before time t (information set F_t). Future prices (t+1, t+2, ...) must
   never enter the calculation under any circumstance.
2. Rolling Window Integrity:
   All rolling aggregations use past-only windows (closed='right', center=False).
3. Scale Invariance & Cross-Asset Comparability:
   Price distances and deviations are normalized by rolling volatility (standard deviation
   or Average True Range) rather than raw dollar prices, ensuring cross-asset model readiness.

Theoretical References:
-----------------------
- Bollinger, J. (2001). Bollinger on Bollinger Bands. McGraw-Hill.
- Ornstein, L. S., & Uhlenbeck, G. E. (1930). On the Theory of the Brownian Motion.
  Physical Review, 36(5), 823-841.
- Lane, G. C. (1984). Lane's Stochastics. Technical Analysis of Stocks & Commodities.
- Wilder, J. W. (1978). New Concepts in Technical Trading Systems. Trend Research.
- Lo, A. W., & MacKinlay, A. C. (1988). Stock Market Prices Do Not Follow Random Walks:
  Evidence from a Simple Specification Test. The Review of Financial Studies, 1(1), 41-66.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.features.base import FeatureBase
from src.features.feature_registry import feature_registry
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Rolling Z-Score of Price Relative to Moving Average
# ============================================================


def compute_price_zscore(
    df: pd.DataFrame,
    windows: Sequence[int] = (10, 20, 50),
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute rolling Z-scores of price relative to its moving average.

    Theoretical Foundation:
    -----------------------
    The rolling Z-score quantifies how many historical standard deviations current price
    is extended from its recent rolling mean:
        Z_{t, n} = (P_t - mu_{t, n}) / sigma_{t, n}
    where:
        mu_{t, n} = (1 / n) * sum_{i=0}^{n-1} P_{t-i}
        sigma_{t, n} = sqrt( (1 / (n - 1)) * sum_{i=0}^{n-1} (P_{t-i} - mu_{t, n})^2 )

    Under Gaussian return assumptions, values exceeding +2.0 or -2.0 denote statistical
    overextension (occurring < 5% of the time under stationarity), signaling potential
    mean-reversion opportunities.

    Parameters
    ----------
    df : pd.DataFrame
        Market data containing `price_col`.
    windows : Sequence[int], default (10, 20, 50)
        Lookback windows in trading days.
    price_col : str, default 'close'
        Column name for price.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'zscore_{w}d' aligned with input index.
    """
    p = df[price_col].astype(float)
    res: dict[str, pd.Series] = {}
    for w in windows:
        if w < 2:
            raise ValueError(f"Z-score lookback window must be >= 2, got {w}")
        rolling = p.rolling(window=w, min_periods=w)
        mean = rolling.mean()
        std = rolling.std(ddof=1)
        # Avoid division by zero on flat synthetic segments
        std_safe = std.replace(0.0, np.nan)
        z = (p - mean) / std_safe
        # If std was exactly 0 and price equals mean, z-score is 0.0
        z = z.mask((std == 0.0) & (p == mean), 0.0)
        res[f"zscore_{w}d"] = z
    return pd.DataFrame(res, index=df.index)


# ============================================================
# 2. Bollinger Bands (%B and Bandwidth) — From Scratch
# ============================================================


def compute_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute Bollinger Bands, %B, and Bandwidth from scratch.

    Theoretical Foundation:
    -----------------------
    Developed by John Bollinger (1983), Bollinger Bands place volatility-adaptive envelopes
    around a simple moving average:
        Middle Band_t = SMA_n(P_t)
        Upper Band_t  = Middle Band_t + k * sigma_{t, n}
        Lower Band_t  = Middle Band_t - k * sigma_{t, n}

    Normalized Signals:
    1. %B (Percent B):
       Quantifies price location relative to the bands:
           %B_t = (P_t - Lower Band_t) / (Upper Band_t - Lower Band_t)
       - %B > 1.0  => Price closed above the upper band (statistical overbought / upper stretch).
       - %B < 0.0  => Price closed below the lower band (statistical oversold / lower stretch).
       - %B = 0.5  => Price closed exactly at the middle moving average.
    2. Bandwidth:
       Quantifies volatility-normalized envelope width:
           Bandwidth_t = (Upper Band_t - Lower Band_t) / Middle Band_t = 2 * k * sigma_{t, n} / SMA_n
       Identifies the "Bollinger Squeeze": extreme low bandwidth reflects volatility compression,
       frequently preceding strong regime transitions or explosive breakouts.

    Parameters
    ----------
    df : pd.DataFrame
        Market data containing `price_col`.
    window : int, default 20
        Lookback window for the middle moving average.
    num_std : float, default 2.0
        Number of standard deviations for band envelopes (k).
    price_col : str, default 'close'
        Column name for price.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - bb_middle_{window}
        - bb_upper_{window}_{num_std}
        - bb_lower_{window}_{num_std}
        - bb_pct_b_{window}_{num_std}
        - bb_bandwidth_{window}_{num_std}
    """
    if window < 2:
        raise ValueError(f"Bollinger window must be >= 2, got {window}")
    if num_std <= 0:
        raise ValueError(f"num_std must be positive, got {num_std}")

    p = df[price_col].astype(float)
    rolling = p.rolling(window=window, min_periods=window)
    middle = rolling.mean()
    std = rolling.std(ddof=1)

    upper = middle + num_std * std
    lower = middle - num_std * std

    band_width_diff = upper - lower
    # Avoid zero division
    band_width_safe = band_width_diff.replace(0.0, np.nan)
    pct_b = (p - lower) / band_width_safe
    pct_b = pct_b.mask((band_width_diff == 0.0) & (p == middle), 0.5)

    middle_safe = middle.replace(0.0, np.nan)
    bandwidth = band_width_diff / middle_safe

    tag = f"{window}_{int(num_std) if num_std.is_integer() else num_std}"
    return pd.DataFrame(
        {
            f"bb_middle_{window}": middle,
            f"bb_upper_{tag}": upper,
            f"bb_lower_{tag}": lower,
            f"bb_pct_b_{tag}": pct_b,
            f"bb_bandwidth_{tag}": bandwidth,
        },
        index=df.index,
    )


# ============================================================
# 3. RSI-Based Reversion Signal (Contrasting Momentum Framing)
# ============================================================


def compute_rsi_reversion(
    df: pd.DataFrame,
    window: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute RSI-based mean-reversion signals and continuous oscillator stretch.

    Theoretical Nuance: Momentum Continuation vs. Mean-Reversion:
    --------------------------------------------------------------
    In Phase 12, RSI was constructed as a momentum indicator. Here in Phase 13,
    we re-frame extreme RSI values (>70 or <30) as mean-reversion exhaustion signals.

    *Interview Defense & Quantitative Rationale*:
    - In strong directional trends (low autocorrelation, positive drift, high variance ratio),
      RSI can remain pinned above 70 or below 30 for extended periods. Treating extreme RSI
      as an immediate reversal signal in a trending market is a classic retail trap.
    - However, in range-bound, oscillating, or high-volatility regimes (where Phase 11 showed
      negative autocorrelation and VR < 1), extreme RSI values accurately denote liquidity
      exhaustion and order-flow overextension.
    - Thus, framing RSI as a reversion signal is a deliberate, regime-conditioned feature
      that pairs directly with Phase 10's volatility regime classifiers and Phase 11's VR diagnostics.

    Signals Generated:
    1. Discrete Reversion Signal:
       +1.0 if RSI < oversold (oversold bounce opportunity)
       -1.0 if RSI > overbought (overbought pullback opportunity)
        0.0 otherwise (neutral range)
    2. Continuous Neutral Stretch:
       (RSI - 50.0) / 50.0  => scaled into [-1.0, +1.0]

    Parameters
    ----------
    df : pd.DataFrame
        Market data containing `price_col`.
    window : int, default 14
        RSI lookback period.
    oversold : float, default 30.0
        Oversold threshold.
    overbought : float, default 70.0
        Overbought threshold.
    price_col : str, default 'close'
        Column name for price.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'rsi_reversion_signal_{w}' and 'rsi_stretch_{w}'.
    """
    from src.features.momentum_features import compute_rsi

    rsi_df = compute_rsi(df, window=window, price_col=price_col)
    rsi_vals = rsi_df[f"rsi_{window}"]

    signal = pd.Series(0.0, index=df.index, dtype=float)
    valid = rsi_vals.notna()

    signal[valid & (rsi_vals < oversold)] = 1.0
    signal[valid & (rsi_vals > overbought)] = -1.0
    signal[~valid] = np.nan

    # Continuous normalized distance from neutral 50
    stretch = (rsi_vals - 50.0) / 50.0

    return pd.DataFrame(
        {
            f"rsi_reversion_signal_{window}": signal,
            f"rsi_stretch_{window}": stretch,
        },
        index=df.index,
    )


# ============================================================
# 4. Normalized Distance from Moving Average (ATR & Std Normalization)
# ============================================================


def compute_true_range(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Compute True Range (TR) strictly looking backward.

    TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
    """
    high_s = df[high_col].astype(float)
    low_s = df[low_col].astype(float)
    close_s = df[close_col].astype(float)
    c_prev = close_s.shift(1)

    tr1 = high_s - low_s
    tr2 = (high_s - c_prev).abs()
    tr3 = (low_s - c_prev).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # At index 0, prev close is NaN, so TR is simply high - low
    tr.iloc[0] = high_s.iloc[0] - low_s.iloc[0]
    return tr


def compute_atr(
    df: pd.DataFrame,
    window: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Compute Average True Range (ATR) from scratch using Wilder's smoothing."""
    tr = compute_true_range(df, high_col=high_col, low_col=low_col, close_col=close_col).values
    n = len(tr)
    atr = np.full(n, np.nan, dtype=float)

    if n < window:
        return pd.Series(atr, index=df.index)

    # Initial simple average of TR over first 'window' periods
    atr[window - 1] = np.mean(tr[:window])

    # Wilder's smoothing recursion: ATR_t = (ATR_{t-1} * (w - 1) + TR_t) / w
    for i in range(window, n):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window

    return pd.Series(atr, index=df.index)


def compute_ma_distance(
    df: pd.DataFrame,
    windows: Sequence[int] = (20, 50),
    normalize_by: str = "atr",
    atr_window: int = 14,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.DataFrame:
    """Compute normalized distance of price from its moving average.

    Why Normalization is Essential:
    -------------------------------
    Raw dollar difference (P_t - MA_t) is non-stationary and scale-dependent: a $5 deviation
    means 10% on a $50 stock (AAPL in 2019) but only 1% on a $500 stock (MSFT in 2026).
    Normalizing by Average True Range (ATR) or rolling standard deviation converts the
    deviation into volatility units ("how many typical daily moves is price extended?"),
    yielding a scale-invariant, cross-asset comparable feature.

    Parameters
    ----------
    df : pd.DataFrame
        Market data.
    windows : Sequence[int], default (20, 50)
        Lookback windows for the moving average.
    normalize_by : str, default 'atr'
        'atr' for Average True Range normalization or 'std' for rolling std normalization.
    atr_window : int, default 14
        ATR smoothing period if normalize_by='atr'.
    price_col : str, default 'close'
        Column name for price.
    high_col : str, default 'high'
        Column name for high price.
    low_col : str, default 'low'
        Column name for low price.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized distance columns (e.g. 'ma_dist_atr_20', 'ma_dist_std_20').
    """
    p = df[price_col].astype(float)
    res: dict[str, pd.Series] = {}

    if normalize_by.lower() == "atr":
        # Fall back to high=low=close if high/low not present
        has_hl = high_col in df.columns and low_col in df.columns
        if has_hl:
            denom = compute_atr(
                df, window=atr_window, high_col=high_col, low_col=low_col, close_col=price_col
            )
        else:
            # High-low fallback: rolling std * sqrt(252/trading_days) or simple rolling std
            denom = p.rolling(window=atr_window, min_periods=atr_window).std(ddof=1)
        denom_safe = denom.replace(0.0, np.nan)

        for w in windows:
            ma = p.rolling(window=w, min_periods=w).mean()
            dist = (p - ma) / denom_safe
            dist = dist.mask((denom == 0.0) & (p == ma), 0.0)
            res[f"ma_dist_atr_{w}"] = dist

    elif normalize_by.lower() == "std":
        for w in windows:
            rolling = p.rolling(window=w, min_periods=w)
            ma = rolling.mean()
            std = rolling.std(ddof=1)
            std_safe = std.replace(0.0, np.nan)
            dist = (p - ma) / std_safe
            dist = dist.mask((std == 0.0) & (p == ma), 0.0)
            res[f"ma_dist_std_{w}"] = dist
    else:
        raise ValueError(f"Unsupported normalize_by '{normalize_by}'. Must be 'atr' or 'std'.")

    return pd.DataFrame(res, index=df.index)


# ============================================================
# 5. Half-Life of Mean Reversion (Ornstein-Uhlenbeck / AR(1))
# ============================================================


def estimate_half_life(series: pd.Series | np.ndarray) -> float:
    """Estimate the mean-reversion half-life of a series via an Ornstein-Uhlenbeck fit.

    Theoretical Foundation:
    -----------------------
    The continuous-time Ornstein-Uhlenbeck (OU) process is defined by the SDE:
        dx_t = theta * (mu - x_t) * dt + sigma * dW_t
    where:
        theta > 0 is the speed of mean reversion,
        mu is the long-term equilibrium level,
        sigma is diffusion volatility.

    Discretizing with Delta t = 1 yields the standard AR(1) linear regression:
        x_t - x_{t-1} = Delta x_t = alpha + beta * x_{t-1} + epsilon_t
    where:
        beta = -(1 - e^{-theta}) = e^{-theta} - 1.

    Properties of the Slope Coefficient beta:
    - If beta < 0: The series is stationary and mean-reverting.
      The speed of reversion is:
          theta = -ln(1 + beta)  (or theta approx -beta for small |beta|)
      The Half-Life (time required for a deviation to decay by 50%) is:
          t_{1/2} = ln(2) / theta = -ln(2) / ln(1 + beta)
    - If beta >= 0: The series is non-stationary (unit root random walk or explosive).
      Mean-reversion does NOT exist, and half-life is infinite (returns np.inf).

    Parameters
    ----------
    series : pd.Series | np.ndarray
        Clean numeric 1D series (e.g. price spread, z-score, or stationary residual).

    Returns
    -------
    float
        Half-life in periods (trading days). Returns np.inf if series is non-mean-reverting.
    """
    if isinstance(series, pd.Series):
        vals = series.dropna().values.astype(float)
    else:
        vals = np.asarray(series, dtype=float)
        vals = vals[~np.isnan(vals) & ~np.isinf(vals)]

    if len(vals) < 10:
        raise ValueError(
            f"Series length ({len(vals)}) is too short for half-life estimation. Minimum is 10."
        )

    y = vals[1:]
    y_lag = vals[:-1]
    delta_y = y - y_lag

    # OLS regression of delta_y on y_lag with intercept
    # delta_y = alpha + beta * y_lag
    var_lag = np.var(y_lag, ddof=1)
    if var_lag == 0.0:
        return np.inf

    cov = np.cov(delta_y, y_lag)[0, 1]
    beta = cov / var_lag

    # If beta >= 0, no mean reversion (random walk or explosive)
    if beta >= 0.0 or (1.0 + beta) <= 0.0:
        return np.inf

    theta = -np.log(1.0 + beta)
    if theta <= 0.0:
        return np.inf

    half_life = np.log(2.0) / theta
    return float(half_life)


def compute_rolling_half_life(
    df: pd.DataFrame,
    window: int = 120,
    ma_window: int = 20,
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute rolling half-life of price spread from its moving average.

    Why Spread Half-Life?
    ---------------------
    Raw equity prices P_t have positive long-term drift and unit roots (I(1)), meaning
    a raw OU fit on P_t will produce beta >= 0 (infinite half-life).
    However, the price spread from trend:
        spread_t = P_t - SMA_{ma_window}(P_t)
    is covariance-stationary and exhibits genuine mean-reversion dynamics.

    Parameters
    ----------
    df : pd.DataFrame
        Market data.
    window : int, default 120
        Rolling estimation window in trading days.
    ma_window : int, default 20
        Lookback window for the detrending moving average.
    price_col : str, default 'close'
        Column name for price.

    Returns
    -------
    pd.DataFrame
        DataFrame with column 'half_life_{window}d'.
    """
    if window < 30:
        raise ValueError(f"Rolling half-life window must be >= 30, got {window}")

    p = df[price_col].astype(float)
    ma = p.rolling(window=ma_window, min_periods=ma_window).mean()
    spread = (p - ma).values

    n = len(spread)
    hl_vals = np.full(n, np.nan, dtype=float)

    for i in range(window - 1, n):
        sub_spread = spread[i - window + 1 : i + 1]
        if np.any(np.isnan(sub_spread)):
            continue
        try:
            hl = estimate_half_life(sub_spread)
            # Cap extreme half-lives for numerical stability in ML models
            hl_vals[i] = min(hl, float(window)) if not np.isinf(hl) else float(window)
        except Exception:
            hl_vals[i] = np.nan

    return pd.DataFrame({f"half_life_{window}d": hl_vals}, index=df.index)


# ============================================================
# 6. Stochastic Oscillator (%K, %D) — From Scratch
# ============================================================


def compute_stochastic_oscillator(
    df: pd.DataFrame,
    k_window: int = 14,
    d_window: int = 3,
    slow_k_window: int = 3,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.DataFrame:
    """Compute Fast and Slow Stochastic Oscillators from scratch.

    Theoretical Foundation:
    -----------------------
    Developed by George Lane (1950s), the Stochastic Oscillator compares an asset's
    closing price to its price range over a specified lookback period:
        Fast %K = (C_t - Lowest_Low_n) / (Highest_High_n - Lowest_Low_n) * 100.0
        Fast %D = SMA_d(Fast %K)
        Slow %K = SMA_{slow}(Fast %K) = Fast %D
        Slow %D = SMA_d(Slow %K)

    Reversion Signals:
    - Overbought threshold: %K or %D > 80.0
    - Oversold threshold: %K or %D < 20.0
    - Bullish Crossover: %K crosses above %D while below 20.0 (+1.0)
    - Bearish Crossover: %K crosses below %D while above 80.0 (-1.0)

    Parameters
    ----------
    df : pd.DataFrame
        Market data.
    k_window : int, default 14
        Lookback window for highest high and lowest low.
    d_window : int, default 3
        Smoothing window for %D.
    slow_k_window : int, default 3
        Smoothing window for Slow %K.
    price_col : str, default 'close'
        Column name for close price.
    high_col : str, default 'high'
        Column name for high price.
    low_col : str, default 'low'
        Column name for low price.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - stoch_k_{k_window}
        - stoch_d_{k_window}_{d_window}
        - stoch_slow_k
        - stoch_slow_d
        - stoch_reversion_signal
    """
    if k_window < 2:
        raise ValueError(f"k_window must be >= 2, got {k_window}")
    if d_window < 1 or slow_k_window < 1:
        raise ValueError("d_window and slow_k_window must be >= 1")

    close_s = df[price_col].astype(float)
    high_s = df[high_col].astype(float) if high_col in df.columns else close_s
    low_s = df[low_col].astype(float) if low_col in df.columns else close_s

    lowest_low = low_s.rolling(window=k_window, min_periods=k_window).min()
    highest_high = high_s.rolling(window=k_window, min_periods=k_window).max()

    range_diff = highest_high - lowest_low
    range_safe = range_diff.replace(0.0, np.nan)

    fast_k = ((close_s - lowest_low) / range_safe) * 100.0
    fast_k = fast_k.mask((range_diff == 0.0) & (close_s == lowest_low), 50.0)

    fast_d = fast_k.rolling(window=d_window, min_periods=d_window).mean()
    slow_k = fast_k.rolling(window=slow_k_window, min_periods=slow_k_window).mean()
    slow_d = slow_k.rolling(window=d_window, min_periods=d_window).mean()

    # Reversion crossover event:
    # Bullish: Slow %K crosses above Slow %D while in oversold (< 20)
    # Bearish: Slow %K crosses below Slow %D while in overbought (> 80)
    k_above_d = (slow_k > slow_d).astype(float)
    crossover = k_above_d.diff().fillna(0.0)

    reversion_signal = pd.Series(0.0, index=df.index, dtype=float)
    # Bullish crossover while oversold
    reversion_signal[(crossover == 1.0) & (slow_k < 25.0)] = 1.0
    # Bearish crossover while overbought
    reversion_signal[(crossover == -1.0) & (slow_k > 75.0)] = -1.0

    return pd.DataFrame(
        {
            f"stoch_k_{k_window}": fast_k,
            f"stoch_d_{k_window}_{d_window}": fast_d,
            "stoch_slow_k": slow_k,
            "stoch_slow_d": slow_d,
            "stoch_reversion_signal": reversion_signal,
        },
        index=df.index,
    )


# ============================================================
# 7. Unified MeanReversionFeatureExtractor (FeatureBase Pattern)
# ============================================================


class MeanReversionFeatureExtractor(FeatureBase):
    """Unified mean-reversion feature extractor computing a full suite of reversion signals.

    Features Produced:
    - zscore_10d, zscore_20d, zscore_50d (Price Z-scores)
    - bb_middle_20, bb_upper_20_2, bb_lower_20_2, bb_pct_b_20_2, bb_bandwidth_20_2 (Bollinger Bands)
    - rsi_reversion_signal_14, rsi_stretch_14 (RSI Reversion Framing)
    - ma_dist_atr_20, ma_dist_atr_50 (Normalized MA Distances)
    - stoch_k_14, stoch_d_14_3, stoch_slow_k, stoch_slow_d, stoch_reversion_signal (Stochastic)
    - half_life_120d (Rolling Ornstein-Uhlenbeck Half-Life on Spread)
    """

    def __init__(
        self,
        name: str = "mean_reversion_suite",
        zscore_windows: Sequence[int] = (10, 20, 50),
        bb_window: int = 20,
        bb_std: float = 2.0,
        rsi_window: int = 14,
        ma_dist_windows: Sequence[int] = (20, 50),
        stoch_k: int = 14,
        stoch_d: int = 3,
        half_life_window: int = 120,
        price_col: str = "close",
        high_col: str = "high",
        low_col: str = "low",
    ) -> None:
        super().__init__(
            name=name,
            category="mean_reversion",
            required_columns=[price_col],
        )
        self.zscore_windows = tuple(zscore_windows)
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rsi_window = rsi_window
        self.ma_dist_windows = tuple(ma_dist_windows)
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.half_life_window = half_life_window
        self.price_col = price_col
        self.high_col = high_col
        self.low_col = low_col

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all single-asset mean-reversion features strictly respecting temporal boundaries."""
        features_list = []

        # 1. Price Z-scores
        features_list.append(
            compute_price_zscore(df, windows=self.zscore_windows, price_col=self.price_col)
        )

        # 2. Bollinger Bands
        features_list.append(
            compute_bollinger_bands(
                df,
                window=self.bb_window,
                num_std=self.bb_std,
                price_col=self.price_col,
            )
        )

        # 3. RSI Reversion Framing
        features_list.append(
            compute_rsi_reversion(df, window=self.rsi_window, price_col=self.price_col)
        )

        # 4. Normalized Moving Average Distance
        features_list.append(
            compute_ma_distance(
                df,
                windows=self.ma_dist_windows,
                normalize_by="atr",
                price_col=self.price_col,
                high_col=self.high_col,
                low_col=self.low_col,
            )
        )

        # 5. Stochastic Oscillator
        features_list.append(
            compute_stochastic_oscillator(
                df,
                k_window=self.stoch_k,
                d_window=self.stoch_d,
                price_col=self.price_col,
                high_col=self.high_col,
                low_col=self.low_col,
            )
        )

        # 6. Rolling Half-Life of Mean Reversion on Price Spread
        features_list.append(
            compute_rolling_half_life(
                df,
                window=self.half_life_window,
                ma_window=self.bb_window,
                price_col=self.price_col,
            )
        )

        return pd.concat(features_list, axis=1)


# ============================================================
# 8. Register Mean-Reversion Features in Central Registry
# ============================================================


def _register_mean_reversion_features() -> None:
    """Register canonical mean-reversion features in the global feature_registry."""
    feature_registry.register(
        name="zscore_10d",
        category="mean_reversion",
        description="10-day rolling price Z-score: (close - mean) / std",
        lookback_horizon=10,
        compute_fn=lambda df: compute_price_zscore(df, windows=(10,)),
        tags=["zscore", "statistical_stretch", "short_term"],
    )
    feature_registry.register(
        name="zscore_20d",
        category="mean_reversion",
        description="20-day rolling price Z-score: (close - mean) / std",
        lookback_horizon=20,
        compute_fn=lambda df: compute_price_zscore(df, windows=(20,)),
        tags=["zscore", "statistical_stretch", "intermediate"],
    )
    feature_registry.register(
        name="zscore_50d",
        category="mean_reversion",
        description="50-day rolling price Z-score: (close - mean) / std",
        lookback_horizon=50,
        compute_fn=lambda df: compute_price_zscore(df, windows=(50,)),
        tags=["zscore", "statistical_stretch", "medium_term"],
    )
    feature_registry.register(
        name="bb_pct_b_20_2",
        category="mean_reversion",
        description="20-day 2-std Bollinger %B: (close - lower) / (upper - lower)",
        lookback_horizon=20,
        compute_fn=lambda df: compute_bollinger_bands(df, window=20, num_std=2.0)[
            ["bb_pct_b_20_2"]
        ],
        tags=["bollinger", "oscillator", "bounded"],
    )
    feature_registry.register(
        name="bb_bandwidth_20_2",
        category="mean_reversion",
        description="20-day 2-std Bollinger Bandwidth: (upper - lower) / middle (volatility normalized)",
        lookback_horizon=20,
        compute_fn=lambda df: compute_bollinger_bands(df, window=20, num_std=2.0)[
            ["bb_bandwidth_20_2"]
        ],
        tags=["bollinger", "volatility", "squeeze"],
    )
    feature_registry.register(
        name="rsi_reversion_signal_14",
        category="mean_reversion",
        description="14-day RSI exhaustion signal: +1 if oversold (<30), -1 if overbought (>70), 0 neutral",
        lookback_horizon=14,
        compute_fn=lambda df: compute_rsi_reversion(df, window=14)[["rsi_reversion_signal_14"]],
        tags=["rsi", "exhaustion", "discrete"],
    )
    feature_registry.register(
        name="ma_dist_atr_20",
        category="mean_reversion",
        description="Distance from 20-day SMA normalized by 14-day ATR: (close - SMA20) / ATR14",
        lookback_horizon=20,
        compute_fn=lambda df: compute_ma_distance(df, windows=(20,), normalize_by="atr"),
        tags=["moving_average", "atr_normalized", "scale_invariant"],
    )
    feature_registry.register(
        name="stoch_slow_k",
        category="mean_reversion",
        description="14-day Slow Stochastic %K oscillator",
        lookback_horizon=17,
        compute_fn=lambda df: compute_stochastic_oscillator(df)[["stoch_slow_k"]],
        tags=["stochastic", "oscillator", "bounded"],
    )
    feature_registry.register(
        name="half_life_120d",
        category="mean_reversion",
        description="120-day rolling Ornstein-Uhlenbeck half-life on price spread from SMA20",
        lookback_horizon=120,
        compute_fn=lambda df: compute_rolling_half_life(df, window=120, ma_window=20),
        tags=["ornstein_uhlenbeck", "half_life", "persistence"],
    )


# Automatically execute registration upon module import
_register_mean_reversion_features()
