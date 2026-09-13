# ============================================================
# Momentum Features Library (Phase 12)
# ============================================================
"""
Production-grade momentum feature extraction library grounded in academic finance
literature and empirical diagnostics from Phase 11.

DATA LEAKAGE SAFEGUARDS (MANDATORY REQUIREMENT):
------------------------------------------------
1. Temporal Causality (No Look-Ahead):
   Every feature at row t must be computed STRICTLY using market data observable
   at or before time t (information set F_t). Future prices (t+1, t+2, ...) must
   never enter the calculation under any circumstance.
2. Rolling Window Integrity:
   All rolling aggregations use past-only windows (closed='right', center=False).
3. Exponential Smoothing Initialization:
   Exponential moving averages (EMAs) and Wilder's smoothing rely solely on recursive
   updates from past bars without forward adjustment.
4. Scale Invariance & Cross-Asset Comparability:
   Price differences are normalized by price levels (percentage spreads, returns, or
   z-scores) to prevent high-priced assets from dominating model weights.

Academic Literature References:
-------------------------------
- Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling Losers:
  Implications for Stock Market Efficiency. The Journal of Finance, 48(1), 65-91.
- Novy-Marx, R. (2012). Is momentum really momentum? Journal of Financial Economics,
  103(3), 429-453.
- Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum.
  Journal of Financial Economics, 104(2), 228-250.
- Wilder, J. W. (1978). New Concepts in Technical Trading Systems. Trend Research.
- Appel, G. (2005). Technical Analysis: Power Tools for Active Investors. FT Press.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.data_pipeline.universe_builder import UniverseBuilder
from src.features.base import FeatureBase
from src.features.feature_registry import feature_registry
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Simple Price Momentum & 12-1 Month Academic Momentum
# ============================================================


def compute_price_momentum(
    df: pd.DataFrame,
    windows: Sequence[int] = (5, 10, 20, 60, 120, 252),
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute multi-horizon simple price momentum (percentage returns).

    Parameters
    ----------
    df : pd.DataFrame
        Market data containing at least `price_col`.
    windows : Sequence[int], default (5, 10, 20, 60, 120, 252)
        Lookback horizons in trading days.
    price_col : str, default 'close'
        Column name for price.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'mom_{w}d' aligned with input index.
    """
    p = df[price_col].astype(float)
    res: dict[str, pd.Series] = {}
    for w in windows:
        if w < 1:
            raise ValueError(f"Lookback window must be >= 1, got {w}")
        # R_{t, w} = (P_t - P_{t-w}) / P_{t-w}
        # Strictly uses price at t and price at t-w (no future bars)
        res[f"mom_{w}d"] = p.pct_change(periods=w)
    return pd.DataFrame(res, index=df.index)


def compute_jegadeesh_titman_momentum(
    df: pd.DataFrame,
    total_window: int = 252,
    skip_window: int = 21,
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute the classic academic "12-1 month" momentum feature.

    Theoretical Foundation & Short-Term Reversal Exclusion:
    -------------------------------------------------------
    In their seminal work, Jegadeesh and Titman (1993) documented that intermediate-term
    past winners continue to outperform past losers over 3 to 12 month holding periods.
    Crucially, they discovered that skipping the most recent month (approx 21 trading days)
    significantly enhances the momentum factor's Sharpe ratio and risk-adjusted return.

    Why skip the most recent month?
    1. Microstructure Bounce: Order flow imbalances and bid-ask bounce create negative
       serial correlation (mean-reversion) at ultra-short horizons (1-5 days), as shown
       empirically in Phase 11's variance ratio and ACF diagnostics.
    2. Short-Term Overreaction: Market participants frequently overreact to immediate
       earnings releases and news catalysts, inducing a temporary 1-month reversal
       (Lehmann 1990; Jegadeesh 1990).
    3. Purity of Intermediate Trend: Skipping [t-21, t] isolates the sustained intermediate
       trend from short-term liquidity noise.

    Mathematical Formulation:
    -------------------------
    R_{t}^{12-1m} = (P_{t - skip_window} - P_{t - total_window}) / P_{t - total_window}

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing `price_col`.
    total_window : int, default 252
        Total lookback window (approx 12 trading months / 252 days).
    skip_window : int, default 21
        Number of most recent days to exclude (approx 1 trading month / 21 days).
    price_col : str, default 'close'
        Column name for price.

    Returns
    -------
    pd.DataFrame
        DataFrame with column 'mom_12_1m' (or 'mom_skip{skip}_{total}d').
    """
    if total_window <= skip_window:
        raise ValueError(
            f"total_window ({total_window}) must be strictly greater than skip_window ({skip_window})."
        )
    p = df[price_col].astype(float)
    p_lag_skip = p.shift(skip_window)
    p_lag_total = p.shift(total_window)
    mom_12_1 = (p_lag_skip - p_lag_total) / p_lag_total

    col_name = (
        "mom_12_1m"
        if (total_window == 252 and skip_window == 21)
        else f"mom_skip{skip_window}_{total_window}d"
    )
    return pd.DataFrame({col_name: mom_12_1}, index=df.index)


# ============================================================
# 2. Rate of Change (ROC) Indicator
# ============================================================


def compute_rate_of_change(
    df: pd.DataFrame,
    windows: Sequence[int] = (10, 20),
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute the Rate of Change (ROC) momentum oscillator.

    ROC measures the percentage change between current price and the price
    n periods ago:
        ROC_t = ((P_t - P_{t-n}) / P_{t-n}) * 100.0

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing `price_col`.
    windows : Sequence[int], default (10, 20)
        Lookback horizons.
    price_col : str, default 'close'
        Column name for price.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'roc_{w}' aligned with input index.
    """
    p = df[price_col].astype(float)
    res: dict[str, pd.Series] = {}
    for w in windows:
        if w < 1:
            raise ValueError(f"ROC window must be >= 1, got {w}")
        res[f"roc_{w}"] = p.pct_change(periods=w) * 100.0
    return pd.DataFrame(res, index=df.index)


# ============================================================
# 3. Moving Average Crossover Signals & Continuous Spread
# ============================================================


def compute_ma_crossover(
    df: pd.DataFrame,
    fast_window: int = 50,
    slow_window: int = 200,
    ma_type: str = "sma",
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute Moving Average crossover signals, binary trend regimes, and continuous spreads.

    Features Produced:
    ------------------
    1. Continuous MA Spread:
       ma_spread_{fast}_{slow} = (MA_fast - MA_slow) / MA_slow
       Scale-invariant indicator of trend strength and extension.
    2. Binary Regime State:
       ma_bullish_{fast}_{slow} = 1 if MA_fast > MA_slow, else 0.
    3. Crossover Event Marker:
       ma_cross_{fast}_{slow}:
         +1 on Golden Cross (fast crosses above slow at time t).
         -1 on Death Cross (fast crosses below slow at time t).
          0 otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing `price_col`.
    fast_window : int, default 50
        Fast moving average window.
    slow_window : int, default 200
        Slow moving average window.
    ma_type : str, default 'sma'
        'sma' for Simple Moving Average or 'ema' for Exponential Moving Average.
    price_col : str, default 'close'
        Column name for price.

    Returns
    -------
    pd.DataFrame
        DataFrame containing continuous spread, bullish state, and crossover event markers.
    """
    if fast_window >= slow_window:
        raise ValueError(
            f"fast_window ({fast_window}) must be strictly less than slow_window ({slow_window})."
        )
    p = df[price_col].astype(float)

    if ma_type.lower() == "sma":
        fast_ma = p.rolling(window=fast_window, min_periods=fast_window).mean()
        slow_ma = p.rolling(window=slow_window, min_periods=slow_window).mean()
    elif ma_type.lower() == "ema":
        fast_ma = p.ewm(span=fast_window, min_periods=fast_window, adjust=False).mean()
        slow_ma = p.ewm(span=slow_window, min_periods=slow_window).mean()
    else:
        raise ValueError(f"Unsupported ma_type '{ma_type}'. Must be 'sma' or 'ema'.")

    # 1. Continuous normalized spread
    spread = (fast_ma - slow_ma) / slow_ma

    # 2. Binary bullish state (1 if fast > slow, 0 otherwise; NaN if warmup incomplete)
    is_valid = fast_ma.notna() & slow_ma.notna()
    bullish = pd.Series(np.nan, index=df.index, dtype=float)
    bullish[is_valid] = (fast_ma[is_valid] > slow_ma[is_valid]).astype(float)

    # 3. Crossover event: diff of binary state (+1 = golden cross, -1 = death cross)
    cross = bullish.diff().fillna(0.0)

    prefix = f"{ma_type}_{fast_window}_{slow_window}"
    return pd.DataFrame(
        {
            f"{prefix}_spread": spread,
            f"{prefix}_bullish": bullish,
            f"{prefix}_cross": cross,
        },
        index=df.index,
    )


# ============================================================
# 4. Relative Strength Index (RSI) — From Scratch Implementation
# ============================================================


def compute_rsi(
    df: pd.DataFrame,
    window: int = 14,
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute the Relative Strength Index (RSI) from scratch using Wilder's smoothing.

    Theoretical Foundation & Interview Derivation:
    ---------------------------------------------
    Developed by J. Welles Wilder Jr. (1978), the Relative Strength Index (RSI) is a
    bounded momentum oscillator [0, 100] quantifying the velocity and magnitude of directional
    price movements.

    Step 1: Compute price change:
        Delta P_t = P_t - P_{t-1}
        U_t = max(Delta P_t, 0)       (Upward gain)
        D_t = max(-Delta P_t, 0)      (Downward loss)

    Step 2: Wilder's Smoothed Averages:
        Unlike a standard Simple Moving Average (which abruptly drops observations older than N),
        Wilder used a modified exponential smoothing with alpha = 1 / N:
            AvgGain_N = (1 / N) * sum_{i=1}^N U_i    (Initial seed)
            AvgGain_t = (AvgGain_{t-1} * (N - 1) + U_t) / N,  for t > N
            AvgLoss_t = (AvgLoss_{t-1} * (N - 1) + D_t) / N,  for t > N

        *Interview Note*: Wilder's smoothing with lookback N is mathematically identical
        to an Exponential Moving Average (EMA) with span = 2N - 1 (smoothing factor alpha = 1/N).

    Step 3: Relative Strength & Bounded Normalization:
        RS_t = AvgGain_t / AvgLoss_t
        RSI_t = 100 - (100 / (1 + RS_t)) = 100 * (AvgGain_t / (AvgGain_t + AvgLoss_t))

    Boundary & Edge-Case Handling:
    - If AvgLoss_t == 0 and AvgGain_t > 0: RSI_t = 100.0 (unbroken upward trend)
    - If AvgGain_t == 0 and AvgLoss_t > 0: RSI_t = 0.0   (unbroken downward trend)
    - If AvgGain_t == 0 and AvgLoss_t == 0: RSI_t = 50.0  (completely flat price series)

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing `price_col`.
    window : int, default 14
        Smoothing window (Wilder's standard is 14).
    price_col : str, default 'close'
        Column name for price.

    Returns
    -------
    pd.DataFrame
        DataFrame with column 'rsi_{window}' aligned with input index.
    """
    if window < 1:
        raise ValueError(f"RSI window must be >= 1, got {window}")

    p = df[price_col].astype(float).values
    n = len(p)
    rsi_vals = np.full(n, np.nan, dtype=float)

    if n <= window:
        return pd.DataFrame({f"rsi_{window}": rsi_vals}, index=df.index)

    # Compute deltas
    diff = np.diff(p)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)

    # Initial simple averages for the first 'window' periods (indices 1 to window)
    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])

    if avg_gain + avg_loss == 0.0:
        rsi_vals[window] = 50.0
    elif avg_loss == 0.0:
        rsi_vals[window] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_vals[window] = 100.0 - (100.0 / (1.0 + rs))

    # Wilder's smoothing recursion for remaining periods
    alpha_inv = window
    for i in range(window, len(gains)):
        curr_gain = gains[i]
        curr_loss = losses[i]

        avg_gain = (avg_gain * (alpha_inv - 1) + curr_gain) / alpha_inv
        avg_loss = (avg_loss * (alpha_inv - 1) + curr_loss) / alpha_inv

        idx = i + 1  # diff array is offset by 1 relative to price array
        if avg_gain + avg_loss == 0.0:
            rsi_vals[idx] = 50.0
        elif avg_loss == 0.0:
            rsi_vals[idx] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_vals[idx] = 100.0 - (100.0 / (1.0 + rs))

    return pd.DataFrame({f"rsi_{window}": rsi_vals}, index=df.index)


# ============================================================
# 5. Moving Average Convergence Divergence (MACD)
# ============================================================


def compute_macd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute Moving Average Convergence Divergence (MACD) indicators and crossovers.

    Components:
    -----------
    1. MACD Line:
       MACD = EMA_{fast}(P) - EMA_{slow}(P)
    2. Signal Line:
       Signal = EMA_{signal}(MACD)
    3. MACD Histogram:
       Histogram = MACD - Signal
    4. Normalized MACD Spread:
       macd_norm = MACD / P_t (percentage of price, scale-independent)
    5. Binary Bullish State:
       macd_bullish = 1 if MACD > Signal, else 0
    6. Crossover Event:
       macd_cross = +1 on bullish signal cross, -1 on bearish signal cross, 0 otherwise

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing `price_col`.
    fast_period : int, default 12
        Fast EMA span.
    slow_period : int, default 26
        Slow EMA span.
    signal_period : int, default 9
        Signal line EMA span.
    price_col : str, default 'close'
        Column name for price.

    Returns
    -------
    pd.DataFrame
        DataFrame with MACD line, signal, histogram, normalized line, and crossover indicators.
    """
    if fast_period >= slow_period:
        raise ValueError(
            f"fast_period ({fast_period}) must be strictly less than slow_period ({slow_period})."
        )
    p = df[price_col].astype(float)

    # Strictly backward-looking recursive EMAs
    ema_fast = p.ewm(span=fast_period, min_periods=fast_period, adjust=False).mean()
    ema_slow = p.ewm(span=slow_period, min_periods=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, min_periods=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # Normalized MACD by price level for cross-asset scale consistency
    macd_norm = macd_line / p

    # State & Crossover
    is_valid = macd_line.notna() & signal_line.notna()
    bullish = pd.Series(np.nan, index=df.index, dtype=float)
    bullish[is_valid] = (macd_line[is_valid] > signal_line[is_valid]).astype(float)
    cross = bullish.diff().fillna(0.0)

    tag = f"{fast_period}_{slow_period}_{signal_period}"
    return pd.DataFrame(
        {
            f"macd_line_{tag}": macd_line,
            f"macd_signal_{tag}": signal_line,
            f"macd_hist_{tag}": macd_hist,
            f"macd_norm_{tag}": macd_norm,
            f"macd_bullish_{tag}": bullish,
            f"macd_cross_{tag}": cross,
        },
        index=df.index,
    )


# ============================================================
# 6. Rank-Based Cross-Sectional Momentum
# ============================================================


def compute_cross_sectional_momentum(
    prices: pd.DataFrame | dict[str, pd.DataFrame | pd.Series],
    window: int = 252,
    skip_window: int = 21,
    universe_builder: UniverseBuilder | None = None,
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute rank-based cross-sectional momentum across an asset universe.

    Theoretical Distinction: Time-Series vs. Cross-Sectional Momentum:
    -------------------------------------------------------------------
    1. Time-Series (Absolute) Momentum (Moskowitz, Ooi, Pedersen 2012):
       - Evaluates an asset strictly against its own past return history:
         R_{i, t} > 0 => Long Asset i; R_{i, t} < 0 => Short / Flat Asset i.
       - Inherently directional and takes long/short market beta exposures.
       - Governed by asset-specific autocorrelation and persistence.

    2. Cross-Sectional (Relative) Momentum (Jegadeesh & Titman 1993):
       - Ranks each asset's momentum at time t relative to the cross-section of peers:
         Rank_{i, t} = Rank(R_{i, t} in {R_{1, t}, R_{2, t}, ..., R_{N, t}}).
       - Zero-cost long-short portfolio: long top decile (winners), short bottom decile (losers).
       - Market-neutral: removes common market drift / beta, isolating relative winners.
       - Robust against broad market regime shifts.

    Survivorship-Bias-Free Point-in-Time Universe Integration:
    ---------------------------------------------------------
    If `universe_builder` is provided, constituents active at date t are filtered
    dynamically using point-in-time index records, preventing look-ahead and survivorship bias.

    Parameters
    ----------
    prices : pd.DataFrame | dict[str, pd.DataFrame | pd.Series]
        Either a wide DataFrame of prices (columns = tickers, index = dates),
        or a dictionary mapping ticker -> DataFrame/Series.
    window : int, default 252
        Lookback window in trading days.
    skip_window : int, default 21
        Skip window to avoid short-term reversal (1 month).
    universe_builder : UniverseBuilder | None
        Optional point-in-time universe engine to filter active constituents per date.
    price_col : str, default 'close'
        Price column name if dict of DataFrames is passed.

    Returns
    -------
    pd.DataFrame
        DataFrame of cross-sectional percentile ranks [0.0, 1.0] across time (index = dates, columns = tickers).
    """
    if window <= skip_window:
        raise ValueError(
            f"window ({window}) must be strictly greater than skip_window ({skip_window})."
        )

    # Align prices into wide format (index = dates, columns = tickers)
    if isinstance(prices, dict):
        series_dict = {}
        for ticker, data in prices.items():
            if isinstance(data, pd.DataFrame):
                p_s = data[price_col].copy()
                if "date" in data.columns and not isinstance(data.index, pd.DatetimeIndex):
                    p_s.index = pd.to_datetime(data["date"])
                series_dict[ticker] = p_s
            else:
                series_dict[ticker] = data
        df_wide = pd.DataFrame(series_dict)
    else:
        df_wide = prices.copy()

    # Compute skipped momentum return per ticker
    lag_skip = df_wide.shift(skip_window)
    lag_total = df_wide.shift(window)
    returns = (lag_skip - lag_total) / lag_total

    # Optional point-in-time universe masking
    if universe_builder is not None:
        for dt in returns.index:
            active = set(universe_builder.get_universe(dt))
            inactive = [c for c in returns.columns if c not in active]
            returns.loc[dt, inactive] = np.nan

    # Cross-sectional percentile rank across rows (axis=1)
    # pct=True normalizes ranks into [0.0, 1.0]
    rank_pct = returns.rank(axis=1, pct=True, ascending=True)

    return rank_pct


# ============================================================
# 7. Unified MomentumFeatureExtractor (FeatureBase Pattern)
# ============================================================


class MomentumFeatureExtractor(FeatureBase):
    """Unified momentum feature extractor computing a full suite of single-asset momentum signals.

    Features Produced:
    - mom_{w}d for w in (5, 10, 20, 60, 120, 252)
    - mom_12_1m (Jegadeesh-Titman 12-1 month momentum)
    - roc_10, roc_20
    - sma_50_200_spread, sma_50_200_bullish, sma_50_200_cross
    - rsi_14 (scratch Wilder smoothing)
    - macd_line_12_26_9, macd_signal_12_26_9, macd_hist_12_26_9, macd_norm_12_26_9, macd_cross_12_26_9
    """

    def __init__(
        self,
        name: str = "momentum_suite",
        momentum_windows: Sequence[int] = (5, 10, 20, 60, 120, 252),
        roc_windows: Sequence[int] = (10, 20),
        fast_ma: int = 50,
        slow_ma: int = 200,
        rsi_window: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        price_col: str = "close",
    ) -> None:
        super().__init__(
            name=name,
            category="momentum",
            required_columns=[price_col],
        )
        self.momentum_windows = tuple(momentum_windows)
        self.roc_windows = tuple(roc_windows)
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.rsi_window = rsi_window
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.price_col = price_col

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all single-asset momentum features strictly respecting temporal boundaries."""
        features_list = []

        # 1. Multi-horizon simple momentum
        features_list.append(
            compute_price_momentum(df, windows=self.momentum_windows, price_col=self.price_col)
        )

        # 2. Jegadeesh-Titman 12-1 month momentum
        features_list.append(compute_jegadeesh_titman_momentum(df, price_col=self.price_col))

        # 3. Rate of Change
        features_list.append(
            compute_rate_of_change(df, windows=self.roc_windows, price_col=self.price_col)
        )

        # 4. MA crossover & spread
        features_list.append(
            compute_ma_crossover(
                df,
                fast_window=self.fast_ma,
                slow_window=self.slow_ma,
                ma_type="sma",
                price_col=self.price_col,
            )
        )

        # 5. Relative Strength Index (RSI)
        features_list.append(compute_rsi(df, window=self.rsi_window, price_col=self.price_col))

        # 6. MACD
        features_list.append(
            compute_macd(
                df,
                fast_period=self.macd_fast,
                slow_period=self.macd_slow,
                signal_period=self.macd_signal,
                price_col=self.price_col,
            )
        )

        out = pd.concat(features_list, axis=1)
        return out


# ============================================================
# 8. Register Momentum Features in Central Registry
# ============================================================


def _register_momentum_features() -> None:
    """Register canonical momentum features in the global feature_registry."""
    feature_registry.register(
        name="mom_5d",
        category="momentum",
        description="5-day simple price momentum (weekly return)",
        lookback_horizon=5,
        compute_fn=lambda df: compute_price_momentum(df, windows=(5,)),
        tags=["short_term", "price_return"],
    )
    feature_registry.register(
        name="mom_10d",
        category="momentum",
        description="10-day simple price momentum (bi-weekly return)",
        lookback_horizon=10,
        compute_fn=lambda df: compute_price_momentum(df, windows=(10,)),
        tags=["short_term", "price_return"],
    )
    feature_registry.register(
        name="mom_20d",
        category="momentum",
        description="20-day simple price momentum (1-month return)",
        lookback_horizon=20,
        compute_fn=lambda df: compute_price_momentum(df, windows=(20,)),
        tags=["intermediate", "price_return"],
    )
    feature_registry.register(
        name="mom_60d",
        category="momentum",
        description="60-day simple price momentum (quarterly return)",
        lookback_horizon=60,
        compute_fn=lambda df: compute_price_momentum(df, windows=(60,)),
        tags=["intermediate", "price_return"],
    )
    feature_registry.register(
        name="mom_120d",
        category="momentum",
        description="120-day simple price momentum (semi-annual return)",
        lookback_horizon=120,
        compute_fn=lambda df: compute_price_momentum(df, windows=(120,)),
        tags=["long_term", "price_return"],
    )
    feature_registry.register(
        name="mom_252d",
        category="momentum",
        description="252-day simple price momentum (annual return)",
        lookback_horizon=252,
        compute_fn=lambda df: compute_price_momentum(df, windows=(252,)),
        tags=["long_term", "price_return"],
    )
    feature_registry.register(
        name="mom_12_1m",
        category="momentum",
        description="12-1 month momentum skipping most recent 21 days to avoid reversal contamination",
        lookback_horizon=252,
        compute_fn=lambda df: compute_jegadeesh_titman_momentum(
            df, total_window=252, skip_window=21
        ),
        tags=["academic", "long_term", "reversal_clean"],
    )
    feature_registry.register(
        name="roc_10",
        category="momentum",
        description="10-day Rate of Change percentage oscillator",
        lookback_horizon=10,
        compute_fn=lambda df: compute_rate_of_change(df, windows=(10,)),
        tags=["oscillator", "short_term"],
    )
    feature_registry.register(
        name="roc_20",
        category="momentum",
        description="20-day Rate of Change percentage oscillator",
        lookback_horizon=20,
        compute_fn=lambda df: compute_rate_of_change(df, windows=(20,)),
        tags=["oscillator", "intermediate"],
    )
    feature_registry.register(
        name="sma_50_200_spread",
        category="momentum",
        description="50-day vs 200-day moving average continuous spread: (SMA50 - SMA200) / SMA200",
        lookback_horizon=200,
        compute_fn=lambda df: compute_ma_crossover(df, fast_window=50, slow_window=200)[
            ["sma_50_200_spread"]
        ],
        tags=["trend", "moving_average"],
    )
    feature_registry.register(
        name="rsi_14",
        category="momentum",
        description="14-day Relative Strength Index (Wilder smoothing from scratch)",
        lookback_horizon=14,
        compute_fn=lambda df: compute_rsi(df, window=14),
        tags=["oscillator", "wilder"],
    )
    feature_registry.register(
        name="macd_12_26_9",
        category="momentum",
        description="MACD (12, 26, 9) indicators: line, signal, hist, norm spread, and crossover",
        lookback_horizon=35,
        compute_fn=lambda df: compute_macd(df, fast_period=12, slow_period=26, signal_period=9),
        tags=["trend", "oscillator", "ema"],
    )


# Automatically execute registration upon module import
_register_momentum_features()
