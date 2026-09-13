# ============================================================
# Volume Features Module (Phase 15)
# ============================================================
"""
Volume-based technical indicators and price impact proxies for quantitative trading.

Honest Data Source Disclosure:
------------------------------
True market microstructure analysis requires Level 2 / tick-level order book feeds
(depth of book, bid/ask quote queues, cancellation ratios, and order flow imbalance).
In this module, all features are constructed strictly from daily or bar-level OHLCV data.
While measures like Amihud Illiquidity are well-established academic proxies for market depth
and price impact (Amihud 2002), they are statistical proxies rather than direct order book
measurements.

Implemented Features:
---------------------
1. On-Balance Volume (OBV) (Granville 1963)
2. Volume-Weighted Average Price (VWAP) - both session-cumulative and rolling daily
3. Accumulation/Distribution Line (ADL) (Chaikin 1970s)
4. Chaikin Money Flow (CMF) (Chaikin 1980s)
5. Volume Rate of Change (Volume ROC) & Volume Z-Score
6. Amihud (2002) Illiquidity Ratio (Price Impact Proxy)
7. VolumeFeatureExtractor adhering to FeatureBase
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.features.base import FeatureBase
from src.features.feature_registry import feature_registry
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. On-Balance Volume (OBV)
# ============================================================


def compute_obv(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
) -> pd.Series:
    """Compute On-Balance Volume (OBV) developed by Joseph Granville (1963).

    Theoretical Foundation:
    -----------------------
    OBV relates price momentum to trading volume under the premise that smart money
    institutional accumulation or distribution precedes price trends:
        OBV_t = OBV_{t-1} + sign(P_t - P_{t-1}) * V_t
    where sign(x) = +1 if x > 0, -1 if x < 0, and 0 if x == 0.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price and volume columns.
    price_col : str, default 'close'
        Price column name.
    volume_col : str, default 'volume'
        Volume column name.

    Returns
    -------
    pd.Series
        Cumulative OBV series aligned with df.index.
    """
    if price_col not in df.columns or volume_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' and '{volume_col}' columns.")

    price = df[price_col].astype(float)
    volume = df[volume_col].astype(float)

    delta_price = price.diff()
    direction = np.sign(delta_price).fillna(0.0)

    # First bar volume is retained as initial baseline
    signed_volume = direction * volume
    signed_volume.iloc[0] = volume.iloc[0]

    obv = signed_volume.cumsum()
    obv.name = "obv"
    return obv


# ============================================================
# 2. Volume-Weighted Average Price (VWAP)
# ============================================================


def compute_vwap(
    df: pd.DataFrame,
    window: int | None = 20,
    price_type: Literal["typical", "close"] = "typical",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.Series:
    """Compute Volume-Weighted Average Price (VWAP).

    Implementation Details:
    -----------------------
    - If `window` is specified (e.g. 20), computes rolling daily VWAP:
          VWAP_{t, n} = sum_{i=0}^{n-1} (P_{t-i} * V_{t-i}) / sum_{i=0}^{n-1} V_{t-i}
    - If `window` is None and index is a DatetimeIndex with multiple bars per day (intraday),
      computes standard cumulative intraday session VWAP resetting each day at market open.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price and volume data.
    window : int | None, default 20
        Rolling window size. If None, computes session-resetting cumulative VWAP.
    price_type : 'typical' | 'close', default 'typical'
        Price used: typical price (H+L+C)/3 or close price.
    high_col, low_col, close_col, volume_col : str
        Column names.

    Returns
    -------
    pd.Series
        VWAP series aligned with df.index.
    """
    if volume_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{volume_col}' column.")

    volume = df[volume_col].astype(float)

    if price_type == "typical":
        if not all(col in df.columns for col in (high_col, low_col, close_col)):
            raise ValueError(
                f"Typical price requires '{high_col}', '{low_col}', '{close_col}' columns."
            )
        price = (
            df[high_col].astype(float) + df[low_col].astype(float) + df[close_col].astype(float)
        ) / 3.0
    else:
        if close_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{close_col}' column.")
        price = df[close_col].astype(float)

    pv = price * volume

    if window is not None:
        # Rolling daily VWAP
        roll_pv = pv.rolling(window=window, min_periods=window).sum()
        roll_vol = volume.rolling(window=window, min_periods=window).sum()
        vwap = roll_pv / roll_vol.replace(0.0, np.nan)
        vwap.name = f"vwap_{window}"
        return vwap

    # Session-resetting cumulative VWAP (intraday)
    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.normalize()
        cum_pv = pv.groupby(dates).cumsum()
        cum_vol = volume.groupby(dates).cumsum()
        vwap = cum_pv / cum_vol.replace(0.0, np.nan)
        vwap.name = "vwap_session"
        return vwap

    # Fallback to full-sample cumulative if not DatetimeIndex
    vwap = pv.cumsum() / volume.cumsum().replace(0.0, np.nan)
    vwap.name = "vwap_cumulative"
    return vwap


# ============================================================
# 3. Accumulation/Distribution Line (ADL)
# ============================================================


def compute_adl(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.Series:
    """Compute the Chaikin Accumulation/Distribution Line (ADL).

    Theoretical Foundation:
    -----------------------
    Developed by Marc Chaikin, ADL gauges supply and demand by evaluating where the close
    finishes relative to the bar's high-low range (Close Location Value, CLV):
        CLV_t = ((Close_t - Low_t) - (High_t - Close_t)) / (High_t - Low_t)
              = (2 * Close_t - High_t - Low_t) / (High_t - Low_t)
        Money Flow Volume (MFV)_t = CLV_t * Volume_t
        ADL_t = ADL_{t-1} + MFV_t

    CLV ranges from +1 (close at high = pure accumulation) to -1 (close at low = pure distribution).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    high_col, low_col, close_col, volume_col : str
        Column names.

    Returns
    -------
    pd.Series
        Cumulative ADL series aligned with df.index.
    """
    for col in (high_col, low_col, close_col, volume_col):
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column '{col}'.")

    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)
    volume = df[volume_col].astype(float)

    hl_range = high - low
    # When High == Low (zero range bar), CLV is neutral (0.0)
    clv = np.where(hl_range > 0, (2.0 * close - high - low) / hl_range, 0.0)
    mfv = pd.Series(clv * volume.values, index=df.index)

    adl = mfv.cumsum()
    adl.name = "adl"
    return adl


# ============================================================
# 4. Chaikin Money Flow (CMF)
# ============================================================


def compute_cmf(
    df: pd.DataFrame,
    window: int = 20,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.Series:
    """Compute Chaikin Money Flow (CMF) over a rolling lookback window.

    Theoretical Foundation:
    -----------------------
    CMF normalizes Money Flow Volume over a rolling window N by the total volume over that window:
        CMF_{t, n} = sum_{i=0}^{n-1} (CLV_{t-i} * V_{t-i}) / sum_{i=0}^{n-1} V_{t-i}
    CMF oscillates between -1.0 (sustained distribution) and +1.0 (sustained accumulation).
    Values > +0.05 signify net institutional buying; < -0.05 signify net selling.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    window : int, default 20
        Lookback window in bars.
    high_col, low_col, close_col, volume_col : str
        Column names.

    Returns
    -------
    pd.Series
        Rolling CMF series.
    """
    for col in (high_col, low_col, close_col, volume_col):
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column '{col}'.")

    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)
    volume = df[volume_col].astype(float)

    hl_range = high - low
    clv = np.where(hl_range > 0, (2.0 * close - high - low) / hl_range, 0.0)
    mfv = pd.Series(clv * volume.values, index=df.index)

    roll_mfv = mfv.rolling(window=window, min_periods=window).sum()
    roll_vol = volume.rolling(window=window, min_periods=window).sum()

    cmf = roll_mfv / roll_vol.replace(0.0, np.nan)
    cmf.name = f"cmf_{window}"
    return cmf


# ============================================================
# 5. Volume Rate of Change & Volume Z-Score
# ============================================================


def compute_volume_roc(
    df: pd.DataFrame,
    window: int = 10,
    volume_col: str = "volume",
) -> pd.Series:
    """Compute Volume Rate of Change (Volume ROC).

    VROC_{t, n} = (Volume_t - Volume_{t-n}) / Volume_{t-n}

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with volume column.
    window : int, default 10
        Lag horizon.
    volume_col : str, default 'volume'
        Volume column name.

    Returns
    -------
    pd.Series
        Volume ROC series.
    """
    if volume_col not in df.columns:
        raise ValueError(f"DataFrame missing volume column '{volume_col}'.")

    vol = df[volume_col].astype(float)
    v_lag = vol.shift(window)
    vroc = (vol - v_lag) / v_lag.replace(0.0, np.nan)
    vroc.name = f"volume_roc_{window}"
    return vroc


def compute_volume_zscore(
    df: pd.DataFrame,
    window: int = 20,
    volume_col: str = "volume",
) -> pd.Series:
    """Compute Volume Z-Score relative to recent rolling volume distribution.

    Z_{V, t, n} = (Volume_t - Mean_V(n)) / Std_V(n)

    A Z-score > 2.0 indicates an abnormal volume surge (breakout or capitulation confirmation).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with volume column.
    window : int, default 20
        Rolling lookback window.
    volume_col : str, default 'volume'
        Volume column name.

    Returns
    -------
    pd.Series
        Volume Z-score series.
    """
    if volume_col not in df.columns:
        raise ValueError(f"DataFrame missing volume column '{volume_col}'.")

    vol = df[volume_col].astype(float)
    roll_mean = vol.rolling(window=window, min_periods=window).mean()
    roll_std = vol.rolling(window=window, min_periods=window).std(ddof=1)

    zscore = (vol - roll_mean) / roll_std.replace(0.0, np.nan)
    zscore.name = f"volume_zscore_{window}"
    return zscore


# ============================================================
# 6. Amihud (2002) Illiquidity Measure (Price Impact Proxy)
# ============================================================


def compute_amihud_illiquidity(
    df: pd.DataFrame,
    window: int = 20,
    scale: float = 1e6,
    price_col: str = "close",
    volume_col: str = "volume",
) -> tuple[pd.Series, pd.Series]:
    """Compute Amihud (2002) Illiquidity Ratio (Price Impact Proxy).

    Theoretical Foundation:
    -----------------------
    Yakov Amihud (2002, Journal of Financial Markets) proposed the daily ratio of
    absolute return to dollar volume as a rough proxy for Kyle's lambda (price impact):
        ILLIQ_t = |Return_t| / DollarVolume_t
                = |P_t - P_{t-1}| / (P_{t-1} * P_t * V_t)
    scaled by a multiplier (default: 1,000,000) to represent basis points of price impact
    per million dollars traded.

    Honest Microstructure Qualification:
    ------------------------------------
    Amihud illiquidity is a low-frequency macro proxy for order book depth. It captures
    how much price moved per unit of traded turnover. It does NOT observe limit order book
    depth, quotes, or spread directly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price and volume columns.
    window : int, default 20
        Rolling window for smoothed illiquidity.
    scale : float, default 1e6
        Scaling multiplier (1e6 expresses in price change per million $ dollar volume).
    price_col, volume_col : str
        Column names.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (daily_amihud, rolling_smoothed_amihud)
    """
    if price_col not in df.columns or volume_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' and '{volume_col}' columns.")

    price = df[price_col].astype(float)
    volume = df[volume_col].astype(float)

    abs_ret = price.pct_change().abs()
    dollar_volume = price * volume

    # Avoid division by zero
    daily_illiq = (abs_ret / dollar_volume.replace(0.0, np.nan)) * scale
    daily_illiq.name = "amihud_illiquidity_daily"

    rolling_illiq = daily_illiq.rolling(window=window, min_periods=window).mean()
    rolling_illiq.name = f"amihud_illiquidity_{window}"

    return daily_illiq, rolling_illiq


# ============================================================
# 7. Unified Volume Feature Extractor (FeatureBase)
# ============================================================


class VolumeFeatureExtractor(FeatureBase):
    """Unified volume-based feature extractor conforming to the FeatureBase architecture.

    Features Generated:
    - obv: On-Balance Volume
    - vwap_20: 20-day rolling Volume-Weighted Average Price
    - adl: Chaikin Accumulation/Distribution Line
    - cmf_20: 20-day Chaikin Money Flow
    - volume_roc_10: 10-day Volume Rate of Change
    - volume_zscore_20: 20-day Volume Z-Score
    - amihud_illiquidity_20: 20-day Rolling Amihud Illiquidity (scaled 1e6)
    """

    def __init__(
        self,
        name: str = "volume_suite",
        vwap_window: int = 20,
        cmf_window: int = 20,
        roc_window: int = 10,
        zscore_window: int = 20,
        amihud_window: int = 20,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
    ) -> None:
        super().__init__(
            name=name,
            category="volume",
            required_columns=[high_col, low_col, close_col, volume_col],
        )
        self.vwap_window = vwap_window
        self.cmf_window = cmf_window
        self.roc_window = roc_window
        self.zscore_window = zscore_window
        self.amihud_window = amihud_window
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all volume features for the input DataFrame."""
        obv = compute_obv(df, price_col=self.close_col, volume_col=self.volume_col)
        vwap = compute_vwap(
            df,
            window=self.vwap_window,
            price_type="typical",
            high_col=self.high_col,
            low_col=self.low_col,
            close_col=self.close_col,
            volume_col=self.volume_col,
        )
        adl = compute_adl(
            df,
            high_col=self.high_col,
            low_col=self.low_col,
            close_col=self.close_col,
            volume_col=self.volume_col,
        )
        cmf = compute_cmf(
            df,
            window=self.cmf_window,
            high_col=self.high_col,
            low_col=self.low_col,
            close_col=self.close_col,
            volume_col=self.volume_col,
        )
        roc = compute_volume_roc(df, window=self.roc_window, volume_col=self.volume_col)
        zscore = compute_volume_zscore(df, window=self.zscore_window, volume_col=self.volume_col)
        _, amihud = compute_amihud_illiquidity(
            df,
            window=self.amihud_window,
            price_col=self.close_col,
            volume_col=self.volume_col,
        )

        return pd.DataFrame(
            {
                "obv": obv,
                f"vwap_{self.vwap_window}": vwap,
                "adl": adl,
                f"cmf_{self.cmf_window}": cmf,
                f"volume_roc_{self.roc_window}": roc,
                f"volume_zscore_{self.zscore_window}": zscore,
                f"amihud_illiquidity_{self.amihud_window}": amihud,
            },
            index=df.index,
        )


# ============================================================
# 8. Central Feature Registry Registration
# ============================================================


def _register_volume_features() -> None:
    """Register volume features in the global feature_registry."""
    feature_registry.register(
        name="obv",
        category="volume",
        description="On-Balance Volume: cumulative signed volume based on price changes (Granville 1963)",
        lookback_horizon=1,
        compute_fn=lambda df: compute_obv(df).to_frame(name="obv"),
        tags=["volume", "momentum", "granville"],
    )
    feature_registry.register(
        name="vwap_20",
        category="volume",
        description="20-day rolling Volume-Weighted Average Price based on typical price",
        lookback_horizon=20,
        compute_fn=lambda df: compute_vwap(df, window=20).to_frame(name="vwap_20"),
        tags=["volume", "vwap", "benchmark"],
    )
    feature_registry.register(
        name="adl",
        category="volume",
        description="Accumulation/Distribution Line based on Close Location Value (Chaikin)",
        lookback_horizon=1,
        compute_fn=lambda df: compute_adl(df).to_frame(name="adl"),
        tags=["volume", "accumulation", "distribution"],
    )
    feature_registry.register(
        name="cmf_20",
        category="volume",
        description="20-day Chaikin Money Flow: volume-weighted accumulation/distribution oscillator",
        lookback_horizon=20,
        compute_fn=lambda df: compute_cmf(df, window=20).to_frame(name="cmf_20"),
        tags=["volume", "money_flow", "oscillator"],
    )
    feature_registry.register(
        name="volume_roc_10",
        category="volume",
        description="10-day Volume Rate of Change",
        lookback_horizon=10,
        compute_fn=lambda df: compute_volume_roc(df, window=10).to_frame(name="volume_roc_10"),
        tags=["volume", "rate_of_change", "activity"],
    )
    feature_registry.register(
        name="volume_zscore_20",
        category="volume",
        description="20-day Volume Z-Score: standardized deviation from rolling volume mean",
        lookback_horizon=20,
        compute_fn=lambda df: compute_volume_zscore(df, window=20).to_frame(
            name="volume_zscore_20"
        ),
        tags=["volume", "zscore", "unusual_volume"],
    )
    feature_registry.register(
        name="amihud_illiquidity_20",
        category="volume",
        description="20-day Rolling Amihud (2002) Illiquidity: daily |return| / dollar volume scaled by 1e6",
        lookback_horizon=20,
        compute_fn=lambda df: compute_amihud_illiquidity(df, window=20)[1].to_frame(
            name="amihud_illiquidity_20"
        ),
        tags=["volume", "illiquidity", "price_impact", "microstructure_proxy"],
    )


_register_volume_features()
