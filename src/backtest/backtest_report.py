# ============================================================
# Backtest Reporting & Quantitative Tearsheet Generator (Phase 42)
# ============================================================
"""
Comprehensive, institutional-grade quantitative report and tearsheet generator.

=============================================================================
THE QUANTITATIVE "TEARSHEET" CONVENTION
-----------------------------------------------------------------------------
In quantitative finance and institutional portfolio management (pioneered by
Barra, FactSet, and open-source frameworks like pyfolio), a "Tearsheet" refers to
a dense, standardized 1-to-2 page summary document that tears through the surface
of a single aggregate Sharpe ratio.

Investment committees and risk officers look past headline numbers because:
1. Aggregate Sharpe Masquerade:
   A strategy can boast a 1.5 Sharpe across 5 years while experiencing an 18-month
   flat-to-negative period with decaying alpha, or blow up during crisis regimes.
2. Underwater Path Dependency:
   The duration of a drawdown (time spent underwater before reaching a new high-water
   mark) is psychologically and operationally critical; long underwater recoveries
   lead to investor redemptions.
3. Factor & Benchmark Realities:
   A strategy with 15% annualized return is unimpressive if the benchmark (e.g. SPY)
   gained 20% with lower beta. Benchmark-relative metrics (Alpha, Beta, Information
   Ratio) quantify genuine value-add over passive indexing.
=============================================================================
"""

from __future__ import annotations

import base64
import io
from dataclasses import asdict, is_dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.financial_metrics import (
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_tracking_error(
    strategy_returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized tracking error between strategy and benchmark."""
    diff = np.asarray(strategy_returns) - np.asarray(benchmark_returns)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2:
        return 0.0
    return float(np.std(diff, ddof=1) * np.sqrt(periods_per_year))


def calculate_information_ratio(
    strategy_returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized information ratio (excess return / tracking error)."""
    diff = np.asarray(strategy_returns) - np.asarray(benchmark_returns)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2:
        return 0.0
    te = calculate_tracking_error(strategy_returns, benchmark_returns, periods_per_year)
    if te <= 1e-12:
        return 0.0
    excess_mean = float(np.mean(diff) * periods_per_year)
    return float(excess_mean / te)


# ============================================================
# 1. Helper Functions & Metric Calculations
# ============================================================


def extract_backtest_components(
    backtest_results: Any,
) -> tuple[pd.Series, pd.Series, list[dict[str, Any]], pd.Series | None, pd.Series | None]:
    """Extract standard equity, returns, trade log, and benchmark series from

    various backtest result containers (BacktestResult, PairsBacktestResult, or dict).
    """
    equity: pd.Series | None = None
    returns: pd.Series | None = None
    trades: list[dict[str, Any]] = []
    benchmark_equity: pd.Series | None = None
    benchmark_returns: pd.Series | None = None

    # Case 1: Object with attributes
    if hasattr(backtest_results, "portfolio_equity"):
        equity = backtest_results.portfolio_equity
    elif hasattr(backtest_results, "equity"):
        equity = backtest_results.equity

    if hasattr(backtest_results, "daily_returns"):
        returns = backtest_results.daily_returns
    elif hasattr(backtest_results, "returns"):
        returns = backtest_results.returns
    elif hasattr(backtest_results, "spread_pnl"):
        returns = backtest_results.spread_pnl

    if hasattr(backtest_results, "trades"):
        raw_trades = backtest_results.trades
        for tr in raw_trades:
            if hasattr(tr, "to_dict"):
                trades.append(tr.to_dict())
            elif isinstance(tr, dict):
                trades.append(tr)
            elif is_dataclass(tr):
                trades.append(asdict(tr))
    elif hasattr(backtest_results, "trade_log"):
        raw_trades = backtest_results.trade_log
        for tr in raw_trades:
            if isinstance(tr, dict):
                trades.append(tr)
            elif hasattr(tr, "to_dict"):
                trades.append(tr.to_dict())

    if hasattr(backtest_results, "benchmark_equity"):
        benchmark_equity = backtest_results.benchmark_equity
    if hasattr(backtest_results, "benchmark_returns"):
        benchmark_returns = backtest_results.benchmark_returns

    # Case 2: Dictionary input
    if isinstance(backtest_results, dict):
        equity = backtest_results.get("portfolio_equity", backtest_results.get("equity", equity))
        returns = backtest_results.get("daily_returns", backtest_results.get("returns", returns))
        raw_trades = backtest_results.get("trades", backtest_results.get("trade_log", []))
        for tr in raw_trades:
            if isinstance(tr, dict):
                trades.append(tr)
            elif hasattr(tr, "to_dict"):
                trades.append(tr.to_dict())
        benchmark_equity = backtest_results.get("benchmark_equity", benchmark_equity)
        benchmark_returns = backtest_results.get("benchmark_returns", benchmark_returns)

    # Derive returns if only equity is present
    if returns is None and equity is not None:
        returns = equity.pct_change().fillna(0.0)

    # Derive equity if only returns are present
    if equity is None and returns is not None:
        equity = 100000.0 * (1.0 + returns).cumprod()

    if equity is None or returns is None:
        raise ValueError("Unable to extract equity or returns series from backtest_results.")

    return equity, returns, trades, benchmark_equity, benchmark_returns


def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute underwater drawdown fraction from cumulative peak equity."""
    peaks = equity.cummax()
    drawdown = (equity - peaks) / peaks
    return drawdown


def compute_rolling_metrics(
    returns: pd.Series,
    window: int = 126,  # ~6 months
    trades: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Compute rolling 6-month Sharpe ratio, annualized volatility, and win rate.

    Parameters
    ----------
    returns : pd.Series
        Daily strategy return series.
    window : int, default 126
        Rolling window length (126 bars = ~6 calendar months).
    trades : list of dicts, optional
        Trade history for rolling win rate computation.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'rolling_sharpe', 'rolling_volatility', 'rolling_win_rate'.
    """
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()

    rolling_sharpe = (rolling_mean / (rolling_std + 1e-9)) * np.sqrt(252.0)
    rolling_vol = rolling_std * np.sqrt(252.0)

    # Rolling win rate of daily returns
    rolling_win_rate = (returns > 0).astype(float).rolling(window=window).mean()

    return pd.DataFrame(
        {
            "rolling_sharpe": rolling_sharpe,
            "rolling_volatility": rolling_vol,
            "rolling_win_rate": rolling_win_rate,
        },
        index=returns.index,
    )


def compute_trade_statistics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute comprehensive trade-level metrics."""
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "average_trade_pnl": 0.0,
            "average_win_pnl": 0.0,
            "average_loss_pnl": 0.0,
            "payoff_ratio": 0.0,
            "best_trade_pnl": 0.0,
            "worst_trade_pnl": 0.0,
            "average_holding_days": 0.0,
            "max_holding_days": 0,
        }

    pnls = [float(tr.get("pnl", 0.0)) for tr in trades]
    holding_days = [int(tr.get("holding_period_days", 1)) for tr in trades]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_trades = len(pnls)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total_trades if total_trades > 0 else 0.0

    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win > 0 else 1.0)

    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    payoff = abs(avg_win / avg_loss) if abs(avg_loss) > 0 else avg_win

    return {
        "total_trades": total_trades,
        "winning_trades": win_count,
        "losing_trades": loss_count,
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": round(float(profit_factor), 2),
        "average_trade_pnl": round(float(np.mean(pnls)), 2),
        "average_win_pnl": round(float(avg_win), 2),
        "average_loss_pnl": round(float(avg_loss), 2),
        "payoff_ratio": round(float(payoff), 2),
        "best_trade_pnl": round(float(max(pnls)), 2),
        "worst_trade_pnl": round(float(min(pnls)), 2),
        "average_holding_days": round(float(np.mean(holding_days)), 1),
        "max_holding_days": int(max(holding_days)) if holding_days else 0,
    }


def compute_benchmark_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict[str, Any]:
    """Compute benchmark comparison metrics (Alpha, Beta, Tracking Error, Information Ratio)."""
    aligned = pd.DataFrame({"strat": strategy_returns, "bench": benchmark_returns}).dropna()
    if len(aligned) < 2:
        return {
            "alpha_annualized": 0.0,
            "beta": 0.0,
            "tracking_error": 0.0,
            "information_ratio": 0.0,
            "correlation": 0.0,
        }

    cov = np.cov(aligned["strat"], aligned["bench"])
    bench_var = cov[1, 1]
    cov_sb = cov[0, 1]
    beta = float(cov_sb / bench_var) if bench_var > 1e-12 else 1.0

    mean_s = float(aligned["strat"].mean() * 252.0)
    mean_b = float(aligned["bench"].mean() * 252.0)
    alpha = mean_s - (beta * mean_b)

    te = calculate_tracking_error(aligned["strat"], aligned["bench"])
    ir = calculate_information_ratio(aligned["strat"], aligned["bench"])
    corr = float(aligned["strat"].corr(aligned["bench"]))

    return {
        "alpha_annualized_pct": round(alpha * 100, 2),
        "beta": round(beta, 2),
        "tracking_error_pct": round(te * 100, 2),
        "information_ratio": round(ir, 2),
        "correlation": round(corr, 2),
    }


# ============================================================
# 2. Visualization Plotters
# ============================================================


def plot_equity_and_drawdown(
    equity: pd.Series,
    benchmark_equity: pd.Series | None = None,
    title: str = "Strategy Equity & Drawdown Profile",
) -> plt.Figure:
    """Create combined equity curve with underwater drawdown shaded beneath."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Upper panel: Equity
    norm_eq = (equity / equity.iloc[0]) * 100.0
    ax1.plot(norm_eq.index, norm_eq.values, label="Strategy", color="#00E5FF", linewidth=2.0)

    if benchmark_equity is not None:
        norm_bm = (benchmark_equity / benchmark_equity.iloc[0]) * 100.0
        ax1.plot(
            norm_bm.index,
            norm_bm.values,
            label="SPY Benchmark",
            color="#FFD600",
            linestyle="--",
            alpha=0.85,
            linewidth=1.5,
        )

    ax1.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Normalized Equity (Base 100)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", framealpha=0.9)

    # Lower panel: Drawdown
    dd = compute_drawdown_series(equity) * 100.0
    ax2.plot(dd.index, dd.values, color="#FF5252", linewidth=1.2)
    ax2.fill_between(dd.index, dd.values, 0, color="#FF5252", alpha=0.35)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_ylim([min(-25.0, float(dd.min()) * 1.1), 1.0])

    fig.tight_layout()
    return fig


def plot_benchmark_comparison(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    title: str = "Strategy vs Benchmark (SPY) Comparison",
) -> plt.Figure:
    """Plot direct normalized equity comparison between strategy and benchmark."""
    fig, ax = plt.subplots(figsize=(12, 5))
    norm_s = (strategy_equity / strategy_equity.iloc[0]) * 100.0
    norm_b = (benchmark_equity / benchmark_equity.iloc[0]) * 100.0
    ax.plot(norm_s.index, norm_s.values, label="Strategy", color="#00E5FF", linewidth=2.0)
    ax.plot(norm_b.index, norm_b.values, label="SPY Benchmark", color="#FFD600", linestyle="--", linewidth=1.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Normalized Growth (Base 100)", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_rolling_metrics(
    rolling_df: pd.DataFrame,
    title: str = "Rolling Performance & Risk (6-Month / 126-Day Window)",
) -> plt.Figure:
    """Plot rolling Sharpe ratio, volatility, and win rate over time."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Rolling Sharpe
    ax1.plot(rolling_df.index, rolling_df["rolling_sharpe"], color="#00E676", linewidth=1.8)
    ax1.axhline(0.0, color="gray", linestyle="--", alpha=0.7)
    ax1.axhline(1.0, color="#00E676", linestyle=":", alpha=0.6, label="Sharpe = 1.0")
    ax1.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Rolling Sharpe", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Rolling Volatility
    ax2.plot(rolling_df.index, rolling_df["rolling_volatility"] * 100.0, color="#FF9100", linewidth=1.8)
    ax2.set_ylabel("Annualized Vol (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Rolling Win Rate
    ax3.plot(rolling_df.index, rolling_df["rolling_win_rate"] * 100.0, color="#2979FF", linewidth=1.8)
    ax3.axhline(50.0, color="gray", linestyle="--", alpha=0.7, label="50% Benchmark")
    ax3.set_ylabel("Daily Win Rate (%)", fontsize=10)
    ax3.set_xlabel("Date", fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper left")

    fig.tight_layout()
    return fig


def plot_trade_analysis(trades: list[dict[str, Any]], title: str = "Trade PnL Distribution") -> plt.Figure:
    """Plot distribution of trade returns and holding period relationship."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    pnls = [float(tr.get("pnl", 0.0)) for tr in trades] if trades else [0.0]
    pnl_pcts = [float(tr.get("pnl_pct", 0.0)) for tr in trades] if trades else [0.0]
    holding = [int(tr.get("holding_period_days", 1)) for tr in trades] if trades else [1]

    # Histogram of PnLs
    colors = ["#00E676" if p > 0 else "#FF5252" for p in pnls]
    ax1.hist(pnls, bins=25, color="#304FFE", edgecolor="white", alpha=0.75)
    ax1.axvline(0.0, color="red", linestyle="--", linewidth=1.5)
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.set_xlabel("Trade PnL ($)", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Scatter: Holding Period vs Return
    ax2.scatter(holding, pnl_pcts, c=colors, alpha=0.7, edgecolors="none", s=40)
    ax2.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax2.set_title("Holding Period vs Return", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Holding Period (Days)", fontsize=10)
    ax2.set_ylabel("Trade Return (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    return fig


def plot_underwater_chart(equity: pd.Series, title: str = "Isolated Underwater Drawdown Plot") -> plt.Figure:
    """Plot dedicated underwater drawdown plot to highlight recovery duration."""
    fig, ax = plt.subplots(figsize=(12, 4))
    dd = compute_drawdown_series(equity) * 100.0

    ax.plot(dd.index, dd.values, color="#D50000", linewidth=1.5)
    ax.fill_between(dd.index, dd.values, 0, color="#D50000", alpha=0.3)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Drawdown from Peak (%)", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    return fig


def figure_to_base64(fig: plt.Figure) -> str:
    """Encode matplotlib figure into base64 PNG string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ============================================================
# 3. HTML Tearsheet Template & Generator
# ============================================================


def generate_html_report(
    summary_metrics: dict[str, Any],
    trade_stats: dict[str, Any],
    benchmark_metrics: dict[str, Any],
    fig_equity_b64: str,
    fig_rolling_b64: str,
    fig_trades_b64: str,
    fig_underwater_b64: str,
    title: str = "Quantitative Strategy Tearsheet",
) -> str:
    """Generate a clean, modern, standalone HTML tearsheet report."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --border-color: #334155;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 16px;
        }}
        .header h1 {{
            margin: 0;
            color: var(--accent-cyan);
            font-size: 28px;
            letter-spacing: -0.5px;
        }}
        .header p {{
            margin: 6px 0 0 0;
            color: var(--text-secondary);
            font-size: 14px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .metric-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        .metric-label {{
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 22px;
            font-weight: 700;
            margin-top: 6px;
            color: var(--text-primary);
        }}
        .value-green {{ color: var(--accent-green); }}
        .value-red {{ color: var(--accent-red); }}
        .chart-container {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 24px;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .section-title {{
            font-size: 18px;
            color: var(--accent-cyan);
            margin: 24px 0 12px 0;
            border-left: 4px solid var(--accent-cyan);
            padding-left: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: var(--bg-card);
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 24px;
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
            font-size: 13px;
        }}
        th {{
            background-color: #1a2333;
            color: var(--text-secondary);
            font-weight: 600;
        }}
        .footer {{
            text-align: center;
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 40px;
            border-top: 1px solid var(--border-color);
            padding-top: 16px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Institutional Performance Evaluation & Risk Architecture Tearsheet</p>
    </div>

    <!-- Primary KPIs -->
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Total Return</div>
            <div class="metric-value {'value-green' if summary_metrics.get('total_return_pct', 0) >= 0 else 'value-red'}">
                {summary_metrics.get('total_return_pct', 0.0)}%
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Annualized Return (CAGR)</div>
            <div class="metric-value">
                {summary_metrics.get('annualized_return_pct', 0.0)}%
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value {'value-green' if summary_metrics.get('sharpe_ratio', 0) >= 1.0 else ''}">
                {summary_metrics.get('sharpe_ratio', 0.0)}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sortino Ratio</div>
            <div class="metric-value">
                {summary_metrics.get('sortino_ratio', 0.0)}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value value-red">
                {summary_metrics.get('max_drawdown_pct', 0.0)}%
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Calmar Ratio</div>
            <div class="metric-value">
                {summary_metrics.get('calmar_ratio', 0.0)}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">
                {trade_stats.get('win_rate_pct', 0.0)}%
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Profit Factor</div>
            <div class="metric-value {'value-green' if trade_stats.get('profit_factor', 0) > 1.2 else ''}">
                {trade_stats.get('profit_factor', 0.0)}
            </div>
        </div>
    </div>

    <!-- Charts -->
    <div class="section-title">1. Cumulative Performance & Drawdown</div>
    <div class="chart-container">
        <img src="{fig_equity_b64}" alt="Equity Curve & Drawdown">
    </div>

    <div class="section-title">2. Rolling Risk & Performance (6-Month Window)</div>
    <div class="chart-container">
        <img src="{fig_rolling_b64}" alt="Rolling Sharpe and Volatility">
    </div>

    <div class="section-title">3. Underwater Drawdown Profile (Recovery Analysis)</div>
    <div class="chart-container">
        <img src="{fig_underwater_b64}" alt="Underwater Drawdown Plot">
    </div>

    <div class="section-title">4. Trade PnL Distribution & Execution Statistics</div>
    <div class="chart-container">
        <img src="{fig_trades_b64}" alt="Trade Distribution">
    </div>

    <!-- Tabular Breakdown -->
    <div class="section-title">5. Benchmark Comparison & Trade Execution Statistics</div>
    <table>
        <thead>
            <tr>
                <th>Benchmark Metric (vs SPY)</th>
                <th>Value</th>
                <th>Trade Execution Metric</th>
                <th>Value</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Annualized Alpha</td>
                <td>{benchmark_metrics.get('alpha_annualized_pct', 'N/A')}%</td>
                <td>Total Completed Trades</td>
                <td>{trade_stats.get('total_trades', 0)}</td>
            </tr>
            <tr>
                <td>Market Beta</td>
                <td>{benchmark_metrics.get('beta', 'N/A')}</td>
                <td>Average Holding Period</td>
                <td>{trade_stats.get('average_holding_days', 0)} days</td>
            </tr>
            <tr>
                <td>Tracking Error</td>
                <td>{benchmark_metrics.get('tracking_error_pct', 'N/A')}%</td>
                <td>Best Trade PnL</td>
                <td class="value-green">${trade_stats.get('best_trade_pnl', 0)}</td>
            </tr>
            <tr>
                <td>Information Ratio</td>
                <td>{benchmark_metrics.get('information_ratio', 'N/A')}</td>
                <td>Worst Trade PnL</td>
                <td class="value-red">${trade_stats.get('worst_trade_pnl', 0)}</td>
            </tr>
            <tr>
                <td>Return Correlation</td>
                <td>{benchmark_metrics.get('correlation', 'N/A')}</td>
                <td>Payoff Ratio (Avg Win / Avg Loss)</td>
                <td>{trade_stats.get('payoff_ratio', 0)}</td>
            </tr>
        </tbody>
    </table>

    <div class="footer">
        Generated by Antigravity Quantitative Trading Bot Backtest Reporting Engine (Phase 42)
    </div>
</body>
</html>
"""
    return html


# ============================================================
# 4. Master Entry Point: generate_tearsheet()
# ============================================================


def generate_tearsheet(
    backtest_results: Any,
    benchmark_returns: pd.Series | None = None,
    title: str = "Quantitative Strategy Tearsheet",
    output_html_path: str | None = None,
    rolling_window: int = 126,
) -> dict[str, Any]:
    """Generate a comprehensive quantitative tearsheet report from any backtest result.

    =============================================================================
    THE QUANTITATIVE TEARSHEET CONVENTION:
    -----------------------------------------------------------------------------
    A "Tearsheet" is the gold standard deliverable in professional quantitative
    finance (popularized by pyfolio and institutional risk attribution packages).
    Rather than evaluating a strategy via an isolated aggregate statistic, a
    tearsheet tears into the underlying mechanics:
      - Equity curve overlay against benchmark (SPY buy-and-hold)
      - Shaded and isolated underwater drawdown profiles
      - 6-month rolling Sharpe ratio, rolling volatility, and rolling win rate
      - Trade duration, win/loss skew, payoff ratio, and profit factor
      - Benchmark-relative factor exposures (Beta, Alpha, Tracking Error, IR)

    Parameters
    ----------
    backtest_results : BacktestResult, PairsBacktestResult, or dict
        Simulation result container containing equity, returns, and trade logs.
    benchmark_returns : pd.Series, optional
        Returns series for market benchmark (SPY). If None, checks if already
        contained in backtest_results.
    title : str, default "Quantitative Strategy Tearsheet"
        Header title for the generated report.
    output_html_path : str, optional
        Filepath to write standalone HTML report (e.g. 'reports/tearsheet.html').
    rolling_window : int, default 126
        Rolling evaluation window in bars (~6 months).

    Returns
    -------
    dict
        Dictionary containing figures, summary metrics, trade stats, and html content.
    """
    (
        equity,
        returns,
        trades,
        b_equity,
        b_returns,
    ) = extract_backtest_components(backtest_results)

    if benchmark_returns is not None:
        b_returns = benchmark_returns
        b_equity = equity.iloc[0] * (1.0 + b_returns).cumprod()

    # 1. Compute summary statistics
    n_days = max(1, len(equity))
    tot_ret = float((equity.iloc[-1] / equity.iloc[0]) - 1.0)
    ann_ret = float((1.0 + tot_ret) ** (252.0 / n_days) - 1.0) if tot_ret > -1.0 else -1.0
    sharpe = float(calculate_sharpe_ratio(returns))
    sortino = float(calculate_sortino_ratio(returns))
    max_dd = float(calculate_max_drawdown(returns)[0])
    calmar = float(calculate_calmar_ratio(returns))

    summary_metrics = {
        "total_return_pct": round(tot_ret * 100, 2),
        "annualized_return_pct": round(ann_ret * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar_ratio": round(calmar, 2),
    }

    # 2. Trade statistics
    trade_stats = compute_trade_statistics(trades)

    # 3. Rolling metrics
    rolling_df = compute_rolling_metrics(returns, window=rolling_window)

    # 4. Benchmark metrics
    benchmark_metrics = (
        compute_benchmark_metrics(returns, b_returns)
        if b_returns is not None
        else {
            "alpha_annualized_pct": "N/A",
            "beta": "N/A",
            "tracking_error_pct": "N/A",
            "information_ratio": "N/A",
            "correlation": "N/A",
        }
    )

    # 5. Generate Figures
    fig_equity = plot_equity_and_drawdown(equity, b_equity, title=f"{title} — Equity & Drawdown")
    fig_rolling = plot_rolling_metrics(rolling_df, title=f"{title} — Rolling Risk & Stability")
    fig_trades = plot_trade_analysis(trades, title="Trade PnL Distribution & Holding Duration")
    fig_underwater = plot_underwater_chart(equity, title=f"{title} — Isolated Underwater Profile")

    # 6. Convert figures to base64 for HTML export
    fig_equity_b64 = figure_to_base64(fig_equity)
    fig_rolling_b64 = figure_to_base64(fig_rolling)
    fig_trades_b64 = figure_to_base64(fig_trades)
    fig_underwater_b64 = figure_to_base64(fig_underwater)

    # Recreate figures for in-notebook rendering return
    fig_equity_display = plot_equity_and_drawdown(equity, b_equity, title=f"{title} — Equity & Drawdown")
    fig_rolling_display = plot_rolling_metrics(rolling_df, title=f"{title} — Rolling Risk & Stability")
    fig_trades_display = plot_trade_analysis(trades, title="Trade PnL Distribution & Holding Duration")
    fig_underwater_display = plot_underwater_chart(equity, title=f"{title} — Isolated Underwater Profile")

    html_content = generate_html_report(
        summary_metrics=summary_metrics,
        trade_stats=trade_stats,
        benchmark_metrics=benchmark_metrics,
        fig_equity_b64=fig_equity_b64,
        fig_rolling_b64=fig_rolling_b64,
        fig_trades_b64=fig_trades_b64,
        fig_underwater_b64=fig_underwater_b64,
        title=title,
    )

    if output_html_path:
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info("Saved standalone HTML tearsheet to %s", output_html_path)

    return {
        "summary_metrics": summary_metrics,
        "trade_stats": trade_stats,
        "rolling_metrics": rolling_df,
        "benchmark_metrics": benchmark_metrics,
        "html_content": html_content,
        "figures": {
            "equity_and_drawdown": fig_equity_display,
            "rolling_metrics": fig_rolling_display,
            "trade_analysis": fig_trades_display,
            "underwater_chart": fig_underwater_display,
        },
    }
