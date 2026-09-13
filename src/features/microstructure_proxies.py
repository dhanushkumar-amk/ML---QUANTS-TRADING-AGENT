# ============================================================
# Microstructure Proxies Module (Phase 15)
# ============================================================
"""
Low-frequency microstructure and liquidity proxies derived from OHLCV bar data.

CRITICAL METHODOLOGICAL DISCLOSURE (PROXY VS. TRUE MICROSTRUCTURE):
-------------------------------------------------------------------
True market microstructure metrics (such as the Kyle lambda price impact, Hasbrouck order
flow imbalance, limit order book depth, quote queue dynamics, and trade-by-trade effective
spreads) fundamentally require continuous Level 2/Level 3 tick and quote data feeds (TAQ/ITCH).

Retail quantitative researchers and standard backtesting workflows often only have access to
daily or minute-level OHLCV bars. The financial econometrics literature has developed clever
closed-form proxies to estimate microstructure characteristics strictly from bar data:
  1. Corwin-Schultz (2012): Estimates effective bid-ask spread from high-low price ratios.
  2. Roll (1984): Estimates effective bid-ask spread from serial covariance of price changes.
  3. Bulk Volume VPIN (Easley et al. 2011/2012): Proxies order flow toxicity from bar volume.
  4. Garman-Klass (1980) & Parkinson (1980): Extreme-value intraday volatility estimators.

These features MUST be understood and presented as statistical PROXIES rather than ground-truth
order book observations. They are noisy at the single-day level and should generally be smoothed
over rolling windows (e.g., 20 days).

Academic References:
--------------------
- Corwin, S. A., & Schultz, P. (2012). A Simple Way to Estimate Bid-Ask Spreads from Daily
  High and Low Prices. The Journal of Finance, 67(2), 719-760.
- Roll, R. (1984). A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient
  Market. The Journal of Finance, 39(4), 1127-1139.
- Easley, D., López de Prado, M. M., & O'Hara, M. (2011). The Microstructure of the "Flash
  Crash": Flow Toxicity, Liquidity Deficits, and the Probability of Informed Trading.
  The Journal of Portfolio Management, 37(2), 118-128.
- Garman, M. B., & Klass, M. J. (1980). On the Estimation of Security Price Volatilities from
  Historical Data. Journal of Business, 53(1), 67-78.
- Parkinson, M. (1980). The Extreme Value Method for Estimating the Variance of the Rate of
  Return. Journal of Business, 53(1), 61-65.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import FeatureBase
from src.features.feature_registry import feature_registry
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Corwin-Schultz (2012) High-Low Bid-Ask Spread Estimator
# ============================================================


def compute_corwin_schultz_spread(
    df: pd.DataFrame,
    window: int = 20,
    clamp_negative: bool = True,
    high_col: str = "high",
    low_col: str = "low",
) -> tuple[pd.Series, pd.Series]:
    """Compute Corwin-Schultz (2012) High-Low Bid-Ask Spread Estimator.

    Theoretical Foundation:
    -----------------------
    Daily high prices are typically buyer-initiated trades (at the ask) and daily low prices
    are seller-initiated trades (at the bid). Therefore, the high-to-low ratio reflects both
    underlying asset volatility and the bid-ask spread.

    By comparing the sum of single-day high-low variances over two consecutive days against
    the high-low variance over the combined two-day period, asset volatility cancels out,
    leaving an analytical estimate of the spread:
        beta_t = [ln(H_t / L_t)]^2 + [ln(H_{t-1} / L_{t-1})]^2
        gamma_t = [ln(max(H_t, H_{t-1}) / min(L_t, L_{t-1}))]^2
        alpha_t = (sqrt(2 * beta_t) - sqrt(beta_t)) / (3 - 2 * sqrt(2)) - sqrt(gamma_t / (3 - 2 * sqrt(2)))
        Spread_t = 2 * (exp(alpha_t) - 1) / (1 + exp(alpha_t))

    Negative Spread Handling:
    -------------------------
    Due to overnight price jumps or small-sample estimation noise, alpha_t can occasionally
    be negative. As recommended by Corwin & Schultz (2012), negative daily spreads are clamped
    to 0.0 before computing rolling moving averages.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with high and low price columns.
    window : int, default 20
        Rolling window for smoothed spread percentage.
    clamp_negative : bool, default True
        Whether to clamp negative daily spread estimates to 0.0.
    high_col, low_col : str
        Column names.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (daily_spread, rolling_smoothed_spread) as fractional spread (e.g. 0.0010 = 10 bps).
    """
    if high_col not in df.columns or low_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{high_col}' and '{low_col}' columns.")

    high = df[high_col].astype(float)
    low = df[low_col].astype(float)

    if (high <= 0).any() or (low <= 0).any():
        raise ValueError("High and low prices must be strictly positive.")

    # Single-day log high/low squared
    log_hl = np.log(high / low)
    log_hl_sq = log_hl**2

    # beta_t = log_hl_sq_t + log_hl_sq_{t-1}
    beta = log_hl_sq + log_hl_sq.shift(1)

    # 2-day high and low
    high_2d = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
    low_2d = pd.concat([low, low.shift(1)], axis=1).min(axis=1)
    gamma = np.log(high_2d / low_2d) ** 2

    # Denominator constant: 3 - 2 * sqrt(2)
    denom = 3.0 - 2.0 * np.sqrt(2.0)

    # alpha_t calculation
    sqrt_2 = np.sqrt(2.0)
    alpha = ((sqrt_2 - 1.0) * np.sqrt(beta)) / denom - np.sqrt(gamma / denom)

    # Spread_t = 2 * (exp(alpha) - 1) / (1 + exp(alpha))
    exp_alpha = np.exp(alpha)
    daily_spread = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)

    if clamp_negative:
        daily_spread = daily_spread.clip(lower=0.0)

    daily_spread.name = "corwin_schultz_spread_daily"
    rolling_spread = daily_spread.rolling(window=window, min_periods=window).mean()
    rolling_spread.name = f"corwin_schultz_spread_{window}"

    return daily_spread, rolling_spread


# ============================================================
# 2. Roll (1984) Serial Covariance Implied Spread Estimator
# ============================================================


def compute_roll_spread(
    df: pd.DataFrame,
    window: int = 20,
    price_col: str = "close",
) -> tuple[pd.Series, pd.Series]:
    """Compute Roll (1984) Serial Covariance Bid-Ask Spread Estimator.

    Theoretical Foundation:
    -----------------------
    Richard Roll (1984) established that under market efficiency, transaction prices bounce
    between the bid and the ask price. This "bid-ask bounce" produces negative first-order
    serial covariance in price changes:
        Delta P_t = Delta m_t + (s / 2) * Delta Q_t
        Cov(Delta P_t, Delta P_{t-1}) = -(s^2) / 4
    where s is the effective dollar spread and Q_t in {-1, +1} is trade direction.

    Solving for the effective dollar spread s:
        s = 2 * sqrt(-Cov(Delta P_t, Delta P_{t-1}))   when Cov < 0
        s = 0.0                                        when Cov >= 0

    The fractional percentage spread is obtained by dividing by current price P_t:
        Roll_Spread_t = s / P_t

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price column.
    window : int, default 20
        Rolling window for covariance calculation.
    price_col : str, default 'close'
        Price column name.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (dollar_spread, fractional_spread)
    """
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column.")

    price = df[price_col].astype(float)
    dp = price.diff()
    dp_lag = dp.shift(1)

    # Rolling covariance between dp and dp_lag
    # cov(x, y) = E[xy] - E[x]E[y]
    roll_cov = dp.rolling(window=window, min_periods=window).cov(dp_lag)

    # Roll formula: 2 * sqrt(-cov) if cov < 0 else 0.0
    neg_cov = np.maximum(0.0, -roll_cov.fillna(0.0))
    dollar_spread = np.where(roll_cov < 0, 2.0 * np.sqrt(neg_cov), 0.0)
    s_dollar = pd.Series(dollar_spread, index=df.index, name=f"roll_dollar_spread_{window}")

    # Set leading warmup NaNs
    s_dollar.iloc[: window + 1] = np.nan

    # Fractional spread
    s_pct = s_dollar / price.replace(0.0, np.nan)
    s_pct.name = f"roll_spread_{window}"

    return s_dollar, s_pct


# ============================================================
# 3. VPIN (Volume-Synchronized Probability of Informed Trading) Proxy
# ============================================================


def compute_vpin_proxy(
    df: pd.DataFrame,
    window: int = 20,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.Series:
    """Compute VPIN (Volume-Synchronized Probability of Informed Trading) Bar Proxy.

    Theoretical Foundation:
    -----------------------
    VPIN (Easley, Lopez de Prado, O'Hara 2011/2012) measures order flow toxicity by
    calculating the imbalance between buyer-initiated and seller-initiated volume.
    High VPIN indicates that one side of the market possesses superior private information,
    leading liquidity providers to withdraw and widening spreads (e.g. prior to flash crashes).

    Bar-Level Approximation (Bulk Volume Classification):
    -----------------------------------------------------
    True VPIN is computed on trade-by-trade tick feeds grouped into equal-volume buckets.
    On bar data, we approximate trade direction via the Close Location Value (CLV):
        Buy Volume Fraction:  f_B = (Close - Low) / (High - Low)
        V_t^B = Volume_t * f_B
        V_t^S = Volume_t * (1 - f_B)
        Order Imbalance_t = |V_t^B - V_t^S|
        VPIN_{t, n} = sum_{i=0}^{n-1} |V_{t-i}^B - V_{t-i}^S| / sum_{i=0}^{n-1} Volume_{t-i}

    VPIN takes values in [0, 1]. High values (> 0.60) signify toxic one-sided order flow.

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
        VPIN proxy series in [0, 1].
    """
    for col in (high_col, low_col, close_col, volume_col):
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column '{col}'.")

    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)
    volume = df[volume_col].astype(float)

    hl_range = high - low
    buy_fraction = np.where(hl_range > 0, (close - low) / hl_range, 0.5)

    v_buy = volume * buy_fraction
    v_sell = volume * (1.0 - buy_fraction)
    imbalance = np.abs(v_buy - v_sell)

    roll_imbalance = imbalance.rolling(window=window, min_periods=window).sum()
    roll_volume = volume.rolling(window=window, min_periods=window).sum()

    vpin = roll_imbalance / roll_volume.replace(0.0, np.nan)
    vpin.name = f"vpin_proxy_{window}"
    return vpin


# ============================================================
# 4. Intraday Volatility Proxies (Garman-Klass & Parkinson)
# ============================================================


def compute_garman_klass_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualized: bool = True,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Compute Garman-Klass (1980) High-Low-Open-Close Volatility Estimator.

    Theoretical Foundation:
    -----------------------
    Under geometric Brownian motion with zero drift, Garman & Klass (1980) showed that an
    optimal linear combination of high, low, open, and close prices is ~7.4 times more
    statistically efficient than the traditional close-to-close return variance:
        sigma_{GK, t}^2 = 0.5 * [ln(H_t / L_t)]^2 - (2 * ln(2) - 1) * [ln(C_t / O_t)]^2
    where (2 * ln(2) - 1) approx 0.386294.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame.
    window : int, default 20
        Rolling window in bars.
    annualized : bool, default True
        If True, multiplies daily volatility by sqrt(252).
    open_col, high_col, low_col, close_col : str
        Column names.

    Returns
    -------
    pd.Series
        Rolling Garman-Klass volatility.
    """
    for col in (open_col, high_col, low_col, close_col):
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column '{col}'.")

    op = df[open_col].astype(float)
    hi = df[high_col].astype(float)
    lo = df[low_col].astype(float)
    cl = df[close_col].astype(float)

    term1 = 0.5 * (np.log(hi / lo) ** 2)
    term2 = (2.0 * np.log(2.0) - 1.0) * (np.log(cl / op) ** 2)
    var_gk = term1 - term2

    # Daily GK variance clamped to >= 0
    var_gk = var_gk.clip(lower=0.0)

    roll_var = var_gk.rolling(window=window, min_periods=window).mean()
    vol_gk = np.sqrt(roll_var)

    if annualized:
        vol_gk = vol_gk * np.sqrt(252.0)
        vol_gk.name = f"garman_klass_vol_{window}_ann"
    else:
        vol_gk.name = f"garman_klass_vol_{window}"

    return vol_gk


def compute_parkinson_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualized: bool = True,
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    """Compute Parkinson (1980) High-Low Volatility Estimator.

    Theoretical Foundation:
    -----------------------
    Michael Parkinson (1980) developed the extreme-value variance estimator based on the
    continuous Brownian maximum and minimum:
        sigma_{P, t}^2 = [ln(H_t / L_t)]^2 / (4 * ln(2))
    where 4 * ln(2) approx 2.772589.

    Efficiency:
    -----------
    Parkinson volatility is ~5.0 times more statistically efficient than standard
    close-to-close variance because it utilizes the extreme price path realized throughout
    the entire trading session.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with high and low price columns.
    window : int, default 20
        Rolling window in bars.
    annualized : bool, default True
        If True, multiplies daily volatility by sqrt(252).
    high_col, low_col : str
        Column names.

    Returns
    -------
    pd.Series
        Rolling Parkinson volatility.
    """
    if high_col not in df.columns or low_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{high_col}' and '{low_col}' columns.")

    hi = df[high_col].astype(float)
    lo = df[low_col].astype(float)

    var_p = (np.log(hi / lo) ** 2) / (4.0 * np.log(2.0))
    roll_var = var_p.rolling(window=window, min_periods=window).mean()
    vol_p = np.sqrt(roll_var)

    if annualized:
        vol_p = vol_p * np.sqrt(252.0)
        vol_p.name = f"parkinson_vol_{window}_ann"
    else:
        vol_p.name = f"parkinson_vol_{window}"

    return vol_p


# ============================================================
# 5. Unified Microstructure Proxy Feature Extractor (FeatureBase)
# ============================================================


class MicrostructureProxyFeatureExtractor(FeatureBase):
    """Unified microstructure proxy feature extractor conforming to FeatureBase.

    Features Generated:
    - corwin_schultz_spread_20: 20-day Corwin-Schultz (2012) High-Low bid-ask spread proxy
    - roll_spread_20: 20-day Roll (1984) serial covariance bid-ask spread proxy
    - vpin_proxy_20: 20-day Volume-Synchronized Probability of Informed Trading bar proxy
    - garman_klass_vol_20: 20-day Annualized Garman-Klass (1980) OHLC volatility
    - parkinson_vol_20: 20-day Annualized Parkinson (1980) High-Low volatility
    """

    def __init__(
        self,
        name: str = "microstructure_proxy_suite",
        spread_window: int = 20,
        vpin_window: int = 20,
        vol_window: int = 20,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
    ) -> None:
        super().__init__(
            name=name,
            category="microstructure",
            required_columns=[open_col, high_col, low_col, close_col, volume_col],
        )
        self.spread_window = spread_window
        self.vpin_window = vpin_window
        self.vol_window = vol_window
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all microstructure proxy features for the input DataFrame."""
        _, cs_spread = compute_corwin_schultz_spread(
            df,
            window=self.spread_window,
            clamp_negative=True,
            high_col=self.high_col,
            low_col=self.low_col,
        )
        _, roll_spread = compute_roll_spread(
            df,
            window=self.spread_window,
            price_col=self.close_col,
        )
        vpin = compute_vpin_proxy(
            df,
            window=self.vpin_window,
            high_col=self.high_col,
            low_col=self.low_col,
            close_col=self.close_col,
            volume_col=self.volume_col,
        )
        gk_vol = compute_garman_klass_volatility(
            df,
            window=self.vol_window,
            annualized=True,
            open_col=self.open_col,
            high_col=self.high_col,
            low_col=self.low_col,
            close_col=self.close_col,
        )
        parkinson_vol = compute_parkinson_volatility(
            df,
            window=self.vol_window,
            annualized=True,
            high_col=self.high_col,
            low_col=self.low_col,
        )

        return pd.DataFrame(
            {
                f"corwin_schultz_spread_{self.spread_window}": cs_spread,
                f"roll_spread_{self.spread_window}": roll_spread,
                f"vpin_proxy_{self.vpin_window}": vpin,
                f"garman_klass_vol_{self.vol_window}": gk_vol,
                f"parkinson_vol_{self.vol_window}": parkinson_vol,
            },
            index=df.index,
        )


# ============================================================
# 6. Central Feature Registry Registration
# ============================================================


def _register_microstructure_proxies() -> None:
    """Register microstructure proxies in the global feature_registry."""
    feature_registry.register(
        name="corwin_schultz_spread_20",
        category="microstructure",
        description="20-day Corwin-Schultz (2012) High-Low Bid-Ask Spread Estimator Proxy (fractional)",
        lookback_horizon=20,
        compute_fn=lambda df: compute_corwin_schultz_spread(df, window=20)[1].to_frame(
            name="corwin_schultz_spread_20"
        ),
        tags=["microstructure", "spread", "liquidity", "corwin_schultz", "proxy"],
    )
    feature_registry.register(
        name="roll_spread_20",
        category="microstructure",
        description="20-day Roll (1984) Serial Covariance Implied Bid-Ask Spread Proxy (fractional)",
        lookback_horizon=20,
        compute_fn=lambda df: compute_roll_spread(df, window=20)[1].to_frame(name="roll_spread_20"),
        tags=["microstructure", "spread", "roll", "bid_ask_bounce", "proxy"],
    )
    feature_registry.register(
        name="vpin_proxy_20",
        category="microstructure",
        description="20-day Volume-Synchronized Probability of Informed Trading (VPIN) Bulk Volume Proxy",
        lookback_horizon=20,
        compute_fn=lambda df: compute_vpin_proxy(df, window=20).to_frame(name="vpin_proxy_20"),
        tags=["microstructure", "vpin", "informed_trading", "order_flow_toxicity", "proxy"],
    )
    feature_registry.register(
        name="garman_klass_vol_20",
        category="volatility",
        description="20-day Annualized Garman-Klass (1980) OHLC Volatility (~7.4x more efficient than close-to-close)",
        lookback_horizon=20,
        compute_fn=lambda df: compute_garman_klass_volatility(
            df, window=20, annualized=True
        ).to_frame(name="garman_klass_vol_20"),
        tags=["volatility", "garman_klass", "efficiency", "intraday_proxy"],
    )
    feature_registry.register(
        name="parkinson_vol_20",
        category="volatility",
        description="20-day Annualized Parkinson (1980) High-Low Volatility (~5.0x more efficient than close-to-close)",
        lookback_horizon=20,
        compute_fn=lambda df: compute_parkinson_volatility(df, window=20, annualized=True).to_frame(
            name="parkinson_vol_20"
        ),
        tags=["volatility", "parkinson", "efficiency", "extreme_value"],
    )


_register_microstructure_proxies()
