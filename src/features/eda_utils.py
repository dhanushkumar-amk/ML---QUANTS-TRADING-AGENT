# ============================================================
# Exploratory Data Analysis Utilities (Phase 8)
# ============================================================
"""
Reusable statistical analysis and visualization functions for exploratory
financial data analysis (EDA).

Functions:
  - summary_stats_table: Returns DataFrame of return moments & risk metrics.
  - plot_price_series: Price & volume subplot with SMA overlays.
  - plot_returns_distribution: Empirical returns histogram vs. theoretical Normal curve.
  - plot_correlation_matrix: Cross-asset returns correlation heatmap.
  - plot_rolling_volatility: Rolling annualized volatility to spot volatility clustering.
  - plot_seasonality: Day-of-week and month-of-year return seasonalities.
"""

from __future__ import annotations

import sys
import types
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Safeguard for environments where Application Control restricts C-extensions
if "matplotlib._c_internal_utils" not in sys.modules:
    try:
        import matplotlib._c_internal_utils  # noqa: F401
    except ImportError:
        sys.modules["matplotlib._c_internal_utils"] = types.ModuleType(
            "matplotlib._c_internal_utils"
        )

import matplotlib

# Use non-interactive backend for headless environments / CI
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_pipeline.data_access import get_data_access
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_EDA_REPORT_DIR = _PROJECT_ROOT / "reports" / "eda"


def _ensure_dir(path: Path | str) -> Path:
    """Ensure parent directory of a path exists and return Path object."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_ticker_df(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV DataFrame from DataAccessLayer if not provided directly."""
    if df is not None and not df.empty:
        data = df.copy()
    else:
        dal = get_data_access()
        data = dal.get_ohlcv(ticker, start=start, end=end)

    if not isinstance(data.index, pd.DatetimeIndex):
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
            data = data.set_index("date")
        else:
            data.index = pd.to_datetime(data.index)

    return data.sort_index()


def compute_daily_returns(
    series_or_df: pd.Series | pd.DataFrame,
    col: str = "close",
) -> pd.Series:
    """Compute simple percentage daily returns, dropping initial NaN."""
    if isinstance(series_or_df, pd.DataFrame):
        price_series = series_or_df[col]
    else:
        price_series = series_or_df

    returns = price_series.pct_change().dropna()
    returns.name = "return"
    return returns


# ============================================================
# 1. Summary Statistics Table
# ============================================================


def summary_stats_table(
    tickers: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    dfs: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Compute summary statistics and return moments for a collection of tickers.

    Metrics returned:
      - Count (number of return observations)
      - Mean Daily Return (%)
      - Annualized Return (%) [Mean * 252]
      - Daily Volatility (%)
      - Annualized Volatility (%) [Std * sqrt(252)]
      - Skewness (Fisher-Pearson skew)
      - Kurtosis (Fisher excess kurtosis where Normal = 0)
      - Min Daily Return (%)
      - Max Daily Return (%)
      - Annualized Sharpe Ratio (assuming risk-free rate = 0%)

    Parameters
    ----------
    tickers : Sequence[str] | None
        List of ticker symbols.
    start : str | None
        Start date string (YYYY-MM-DD).
    end : str | None
        End date string (YYYY-MM-DD).
    dfs : dict[str, pd.DataFrame] | None
        Optional dictionary of pre-loaded DataFrames keyed by ticker symbol.

    Returns
    -------
    pd.DataFrame
        Summary metrics indexed by ticker symbol.
    """
    if tickers is None:
        tickers = list(dfs.keys()) if dfs else ["AAPL", "MSFT", "SPY"]

    records: list[dict[str, Any]] = []

    for ticker in tickers:
        df_ticker = dfs.get(ticker) if dfs else None
        data = _load_ticker_df(ticker, start=start, end=end, df=df_ticker)
        if data.empty:
            continue

        returns = compute_daily_returns(data)
        ret_vals = returns.values
        n = len(ret_vals)
        if n < 2:
            continue

        mean_daily = float(np.mean(ret_vals))
        std_daily = float(np.std(ret_vals, ddof=1))
        ann_return = mean_daily * 252.0
        ann_vol = std_daily * np.sqrt(252.0)

        # Fisher-Pearson skewness and Fisher excess kurtosis
        skew_val = float(returns.skew())
        kurt_val = float(returns.kurtosis())

        min_val = float(np.min(ret_vals))
        max_val = float(np.max(ret_vals))
        sharpe = (ann_return / ann_vol) if ann_vol > 0 else 0.0

        records.append(
            {
                "ticker": ticker,
                "count": n,
                "mean_daily_pct": mean_daily * 100.0,
                "ann_return_pct": ann_return * 100.0,
                "daily_vol_pct": std_daily * 100.0,
                "ann_vol_pct": ann_vol * 100.0,
                "skewness": skew_val,
                "excess_kurtosis": kurt_val,
                "min_return_pct": min_val * 100.0,
                "max_return_pct": max_val * 100.0,
                "sharpe_ratio": sharpe,
            }
        )

    if not records:
        return pd.DataFrame()

    stats_df = pd.DataFrame(records).set_index("ticker")
    return stats_df


# ============================================================
# 2. Price & Volume Series Subplot
# ============================================================


def plot_price_series(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    df: pd.DataFrame | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot daily close price with 50-day and 200-day SMAs, plus daily volume.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    start : str | None
        Start date filter.
    end : str | None
        End date filter.
    df : pd.DataFrame | None
        Optional pre-loaded DataFrame.
    save_path : str | Path | None
        File path to save the generated plot PNG.
    """
    data = _load_ticker_df(ticker, start=start, end=end, df=df)

    fig, (ax_price, ax_vol) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # 1. Price panel
    close = data["close"]
    sma50 = close.rolling(window=50).mean()
    sma200 = close.rolling(window=200).mean()

    ax_price.plot(data.index, close, label=f"{ticker} Close", color="#1f77b4", linewidth=1.5)
    ax_price.plot(
        data.index, sma50, label="50-Day SMA", color="#ff7f0e", linestyle="--", linewidth=1.2
    )
    ax_price.plot(
        data.index, sma200, label="200-Day SMA", color="#2ca02c", linestyle="--", linewidth=1.2
    )

    ax_price.set_title(
        f"{ticker} Price & Volume Trajectory ({data.index[0].date()} to {data.index[-1].date()})",
        fontsize=14,
        fontweight="bold",
    )
    ax_price.set_ylabel("Price ($)", fontsize=11)
    ax_price.grid(True, linestyle=":", alpha=0.6)
    ax_price.legend(loc="upper left")

    # 2. Volume panel with color coding
    if "volume" in data.columns:
        if "open" in data.columns:
            colors = np.where(data["close"] >= data["open"], "#2ca02c", "#d62728")
        else:
            colors = "#1f77b4"
        ax_vol.bar(data.index, data["volume"] / 1e6, color=colors, width=1.0, alpha=0.7)
        ax_vol.set_ylabel("Vol (M)", fontsize=11)
        ax_vol.grid(True, linestyle=":", alpha=0.6)

    ax_vol.xaxis.set_major_locator(mdates.YearLocator())
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_vol.set_xlabel("Date", fontsize=11)

    fig.tight_layout()

    if save_path:
        out_p = _ensure_dir(save_path)
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        logger.info("[%s] Saved price series plot -> %s", ticker, out_p)

    return fig, (ax_price, ax_vol)


# ============================================================
# 3. Returns Distribution & Normal Overlay
# ============================================================


def plot_returns_distribution(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    df: pd.DataFrame | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot histogram of empirical daily returns overlaid with theoretical Normal fit.

    Displays empirical Skewness and Excess Kurtosis to directly demonstrate fat tails.
    """
    data = _load_ticker_df(ticker, start=start, end=end, df=df)
    returns = compute_daily_returns(data)
    ret_pct = returns * 100.0  # work in percentage

    mu = float(ret_pct.mean())
    sigma = float(ret_pct.std())
    skew_val = float(ret_pct.dropna().skew())
    kurt_val = float(ret_pct.dropna().kurtosis())

    fig, ax = plt.subplots(figsize=(10, 6))

    # Empirical histogram & KDE
    n_bins = min(100, max(30, int(len(ret_pct) / 25)))
    count, bins, _ = ax.hist(
        ret_pct,
        bins=n_bins,
        density=True,
        alpha=0.6,
        color="#3b82f6",
        edgecolor="#1d4ed8",
        label=f"Empirical Returns ({len(ret_pct)} bars)",
    )

    # Theoretical normal curve with same mean and std
    x_grid = np.linspace(ret_pct.min(), ret_pct.max(), 300)
    if sigma > 0:
        gaussian_pdf = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * ((x_grid - mu) / sigma) ** 2
        )
    else:
        gaussian_pdf = np.zeros_like(x_grid)
    ax.plot(
        x_grid,
        gaussian_pdf,
        color="#dc2626",
        linewidth=2.2,
        label=f"Fitted Gaussian N({mu:.2f}%, {sigma:.2f}%)",
    )

    # Annotate moments & fat tails
    stats_text = (
        f"Mean: {mu:.2f}%\n"
        f"Std Dev: {sigma:.2f}%\n"
        f"Skewness: {skew_val:.2f}\n"
        f"Excess Kurtosis: {kurt_val:.2f}\n"
        f"Fat-Tailed: {'YES (Leptokurtic)' if kurt_val > 0 else 'NO'}"
    )
    ax.text(
        0.03,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        fontfamily="monospace",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "#94a3b8",
            "alpha": 0.9,
        },
    )

    ax.set_title(
        f"{ticker} Daily Returns Distribution vs. Gaussian Fit", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Daily Return (%)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    fig.tight_layout()

    if save_path:
        out_p = _ensure_dir(save_path)
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        logger.info("[%s] Saved returns distribution plot -> %s", ticker, out_p)

    return fig, ax


# ============================================================
# 4. Cross-Asset Correlation Matrix Heatmap
# ============================================================


def plot_correlation_matrix(
    tickers: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    dfs: dict[str, pd.DataFrame] | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot correlation heatmap across ticker daily returns."""
    if tickers is None:
        tickers = list(dfs.keys()) if dfs else ["AAPL", "MSFT", "SPY"]

    returns_dict: dict[str, pd.Series] = {}
    for ticker in tickers:
        df_ticker = dfs.get(ticker) if dfs else None
        data = _load_ticker_df(ticker, start=start, end=end, df=df_ticker)
        if not data.empty:
            returns_dict[ticker] = compute_daily_returns(data)

    ret_matrix = pd.DataFrame(returns_dict).dropna()
    corr = ret_matrix.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    tickers_list = list(corr.columns)
    ax.set_xticks(range(len(tickers_list)))
    ax.set_yticks(range(len(tickers_list)))
    ax.set_xticklabels(tickers_list, fontsize=11, fontweight="bold")
    ax.set_yticklabels(tickers_list, fontsize=11, fontweight="bold")

    # Annotate correlation numbers inside cells
    for i in range(len(tickers_list)):
        for j in range(len(tickers_list)):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
                fontweight="bold",
            )

    ax.set_title(
        f"Pairwise Daily Returns Correlation Matrix ({len(ret_matrix)} bars)",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    fig.tight_layout()

    if save_path:
        out_p = _ensure_dir(save_path)
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        logger.info("Saved correlation matrix plot -> %s", out_p)

    return fig, ax


# ============================================================
# 5. Rolling Volatility Plot (Volatility Clustering)
# ============================================================


def plot_rolling_volatility(
    ticker: str,
    window: int = 30,
    start: str | None = None,
    end: str | None = None,
    df: pd.DataFrame | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot rolling annualized standard deviation of returns to highlight volatility clustering."""
    data = _load_ticker_df(ticker, start=start, end=end, df=df)
    returns = compute_daily_returns(data)

    roll_short = returns.rolling(window=window).std() * np.sqrt(252.0) * 100.0
    roll_long = returns.rolling(window=90).std() * np.sqrt(252.0) * 100.0
    overall_vol = float(returns.std()) * np.sqrt(252.0) * 100.0

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        returns.index,
        roll_short,
        label=f"{window}-Day Rolling Volatility",
        color="#e11d48",
        linewidth=1.5,
    )
    ax.plot(
        returns.index,
        roll_long,
        label="90-Day Baseline Volatility",
        color="#4f46e5",
        linestyle="--",
        linewidth=1.2,
    )
    ax.axhline(
        overall_vol,
        color="#64748b",
        linestyle=":",
        label=f"Unconditional Mean Vol ({overall_vol:.1f}%)",
    )

    ax.set_title(
        f"{ticker} Rolling Annualized Volatility (Volatility Clustering)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("Annualized Volatility (%)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    fig.tight_layout()

    if save_path:
        out_p = _ensure_dir(save_path)
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        logger.info("[%s] Saved rolling volatility plot -> %s", ticker, out_p)

    return fig, ax


# ============================================================
# 6. Seasonality Analysis Subplots
# ============================================================


def plot_seasonality(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    df: pd.DataFrame | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Analyze calendar return patterns across Day-of-Week and Month-of-Year."""
    data = _load_ticker_df(ticker, start=start, end=end, df=df)
    returns = compute_daily_returns(data) * 100.0  # in percentage

    ret_df = pd.DataFrame({"return": returns})
    ret_df["day_name"] = ret_df.index.day_name()
    ret_df["day_num"] = ret_df.index.dayofweek
    ret_df["month_name"] = ret_df.index.month_name()
    ret_df["month_num"] = ret_df.index.month

    # 1. Day of Week
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    dow_stats = (
        ret_df[ret_df["day_name"].isin(dow_order)]
        .groupby("day_name")["return"]
        .agg(["mean", "std", "count"])
        .reindex(dow_order)
    )
    dow_stats["sem"] = dow_stats["std"] / np.sqrt(dow_stats["count"])

    # 2. Month of Year
    moy_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    moy_stats = (
        ret_df.groupby("month_name")["return"].agg(["mean", "std", "count"]).reindex(moy_order)
    )
    moy_stats["sem"] = moy_stats["std"] / np.sqrt(moy_stats["count"])

    fig, (ax_dow, ax_moy) = plt.subplots(1, 2, figsize=(14, 5))

    # Day of week bar chart
    colors_dow = ["#22c55e" if x >= 0 else "#ef4444" for x in dow_stats["mean"]]
    ax_dow.bar(
        [d[:3] for d in dow_order],
        dow_stats["mean"],
        yerr=dow_stats["sem"],
        capsize=4,
        color=colors_dow,
        alpha=0.75,
        edgecolor="#334155",
    )
    ax_dow.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_dow.set_title(f"{ticker} Day-of-Week Average Return", fontsize=12, fontweight="bold")
    ax_dow.set_ylabel("Mean Daily Return (%) \u00b1 SEM", fontsize=11)
    ax_dow.grid(True, linestyle=":", alpha=0.6)

    # Month of year bar chart
    colors_moy = ["#22c55e" if x >= 0 else "#ef4444" for x in moy_stats["mean"]]
    ax_moy.bar(
        [m[:3] for m in moy_order],
        moy_stats["mean"],
        yerr=moy_stats["sem"],
        capsize=3,
        color=colors_moy,
        alpha=0.75,
        edgecolor="#334155",
    )
    ax_moy.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_moy.set_title(f"{ticker} Month-of-Year Average Return", fontsize=12, fontweight="bold")
    ax_moy.set_ylabel("Mean Return (%) \u00b1 SEM", fontsize=11)
    ax_moy.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(f"{ticker} Calendar Seasonality Patterns", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        out_p = _ensure_dir(save_path)
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        logger.info("[%s] Saved seasonality plot -> %s", ticker, out_p)

    return fig, (ax_dow, ax_moy)


# ============================================================
# 7. ACF of Squared Returns Plot (Phase 10)
# ============================================================


def plot_acf_squared_returns(
    ticker: str,
    nlags: int = 20,
    start: str | None = None,
    end: str | None = None,
    df: pd.DataFrame | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plot sample Autocorrelation Function (ACF) of raw vs. squared returns.

    Demonstrates the signature stylized fact of financial time series:
    raw returns have near-zero linear autocorrelation (no simple momentum/mean reversion),
    while squared returns show persistent positive autocorrelation (volatility clustering).

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    nlags : int
        Number of lags to evaluate (default: 20).
    start : str | None
        Start date filter.
    end : str | None
        End date filter.
    df : pd.DataFrame | None
        Optional pre-loaded dataframe.
    save_path : str | Path | None
        Destination image path.

    Returns
    -------
    tuple[plt.Figure, Sequence[plt.Axes]]
        (fig, (ax_raw, ax_sq))
    """
    from statsmodels.tsa.stattools import acf

    data = _load_ticker_df(ticker, start=start, end=end, df=df)
    returns = compute_daily_returns(data)
    sq_returns = returns**2

    clean_ret = returns.values
    clean_sq = sq_returns.values

    # Compute ACF with 95% Bartlett confidence bands
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        raw_acf, raw_conf = acf(clean_ret, nlags=nlags, alpha=0.05)
        sq_acf, sq_conf = acf(clean_sq, nlags=nlags, alpha=0.05)

    lags = np.arange(len(raw_acf))

    fig, (ax_raw, ax_sq) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 1. Raw returns ACF
    ax_raw.vlines(lags, [0], raw_acf, color="#2563eb", linewidth=2.0)
    ax_raw.plot(lags, raw_acf, "o", color="#1d4ed8", markersize=5)
    # 95% confidence interval band around 0
    band_upper = raw_conf[:, 1] - raw_acf
    band_lower = raw_conf[:, 0] - raw_acf
    ax_raw.fill_between(
        lags, band_lower, band_upper, color="#93c5fd", alpha=0.35, label="95% Confidence Band"
    )
    ax_raw.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax_raw.set_title(
        f"{ticker} Raw Daily Returns ACF — Minimal Linear Dependence (EMH)",
        fontsize=12,
        fontweight="bold",
    )
    ax_raw.set_ylabel("Autocorrelation", fontsize=11)
    ax_raw.grid(True, linestyle=":", alpha=0.6)
    ax_raw.legend(loc="upper right")

    # 2. Squared returns ACF
    ax_sq.vlines(lags, [0], sq_acf, color="#dc2626", linewidth=2.0)
    ax_sq.plot(lags, sq_acf, "o", color="#b91c1c", markersize=5)
    sq_band_upper = sq_conf[:, 1] - sq_acf
    sq_band_lower = sq_conf[:, 0] - sq_acf
    ax_sq.fill_between(
        lags,
        sq_band_lower,
        sq_band_upper,
        color="#fca5a5",
        alpha=0.35,
        label="95% Confidence Band",
    )
    ax_sq.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax_sq.set_title(
        f"{ticker} Squared Returns ($r_t^2$) ACF — Significant Volatility Clustering",
        fontsize=12,
        fontweight="bold",
    )
    ax_sq.set_xlabel("Lag (Trading Days)", fontsize=11)
    ax_sq.set_ylabel("Autocorrelation", fontsize=11)
    ax_sq.grid(True, linestyle=":", alpha=0.6)
    ax_sq.legend(loc="upper right")

    fig.suptitle(f"{ticker} Volatility Memory Diagnostic", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        out_p = _ensure_dir(save_path)
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        logger.info("[%s] Saved ACF squared returns plot -> %s", ticker, out_p)

    return fig, (ax_raw, ax_sq)


# ============================================================
# 8. Volatility Regimes Plot (Phase 10)
# ============================================================


def plot_volatility_regimes(
    ticker: str,
    window: int = 21,
    regime_quantile: float = 0.5,
    start: str | None = None,
    end: str | None = None,
    df: pd.DataFrame | None = None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot rolling volatility with shaded high/low volatility regimes.

    Shades background periods:
      - Low Volatility Regime: Rolling volatility <= threshold (e.g. median).
      - High Volatility Regime: Rolling volatility > threshold.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    window : int
        Rolling window in trading days (default: 21 ~ 1 month).
    regime_quantile : float
        Quantile separating low from high volatility regime (default: 0.5 for median).
    start : str | None
        Start date filter.
    end : str | None
        End date filter.
    df : pd.DataFrame | None
        Optional pre-loaded dataframe.
    save_path : str | Path | None
        Destination image path.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        (fig, ax)
    """
    data = _load_ticker_df(ticker, start=start, end=end, df=df)
    returns = compute_daily_returns(data)

    roll_vol = returns.rolling(window=window).std() * np.sqrt(252.0) * 100.0
    valid_vol = roll_vol.dropna()
    threshold = float(valid_vol.quantile(regime_quantile))

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(
        valid_vol.index,
        valid_vol,
        color="#0f172a",
        linewidth=1.4,
        label=f"{window}-Day Rolling Volatility (% ann.)",
    )
    ax.axhline(
        threshold,
        color="#dc2626",
        linestyle="--",
        linewidth=1.2,
        label=f"Regime Threshold ({regime_quantile:.0%}: {threshold:.1f}%)",
    )

    # Shading high vs low regimes
    ax.fill_between(
        valid_vol.index,
        0,
        valid_vol.max() * 1.1,
        where=(valid_vol > threshold),
        color="#fee2e2",
        alpha=0.6,
        label="High Volatility Regime",
    )
    ax.fill_between(
        valid_vol.index,
        0,
        valid_vol.max() * 1.1,
        where=(valid_vol <= threshold),
        color="#f0fdf4",
        alpha=0.6,
        label="Low Volatility Regime",
    )

    ax.set_ylim(0, valid_vol.max() * 1.12)
    ax.set_title(
        f"{ticker} Volatility Regimes ({window}-Day Rolling Ann. Volatility vs {regime_quantile:.0%} Quantile)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Annualized Volatility (%)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

    fig.tight_layout()

    if save_path:
        out_p = _ensure_dir(save_path)
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        logger.info("[%s] Saved volatility regimes plot -> %s", ticker, out_p)

    return fig, ax
