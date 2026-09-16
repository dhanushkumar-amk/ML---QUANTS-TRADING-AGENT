# ============================================================
# Quantitative Stress Testing & Monte Carlo Engine (Phase 43)
# ============================================================
"""
Production stress testing, crisis replay, and Monte Carlo simulation engine.

Theoretical Principles & Crisis Architecture:
---------------------------------------------
1. Crisis Asymmetry:
   Aggregate metrics (e.g. full-sample Sharpe) are often dominated by benign bull regimes.
   A strategy that bleeds capital during the 2020 COVID panic or the 2022 rate-hike
   drawdown is unviable for institutional capital regardless of calm-period alpha.
2. Autocorrelation & Volatility Clustering in Resampling:
   Standard IID bootstrap destroys serial dependence and volatility clustering (Phase 10).
   We implement Circular Block Bootstrap to preserve local volatility persistence
   and regime duration when simulating synthetic equity paths.
3. Regime-Conditional Breakdown:
   Performance is segmented by Phase 16 HMM / GMM detected market regimes
   (Low-Vol Trending, Medium-Vol Transition, High-Vol Crisis) to detect structural
   regime degradation honestly.
4. Active Safety Gate Verification:
   Synthetic stress shocks (volatility doubling, -10% flash gaps) verify that the
   Phase 39 RiskEngine actively intervenes to protect portfolio capital.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest.backtest_report import extract_backtest_components
from src.models.financial_metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from src.portfolio.risk_engine import (
    DecisionStatus,
    OrderAction,
    OrderDecision,
    OrderIntent,
    PortfolioState,
    RiskEngine,
    RiskRule,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Historical Crisis Windows & Evaluation
# ============================================================

STANDARD_CRISIS_PERIODS: dict[str, tuple[str, str]] = {
    "2018_Q4_Tech_Selloff": ("2018-10-01", "2018-12-24"),
    "2020_COVID_Crash": ("2020-02-19", "2020-03-23"),
    "2020_COVID_Full_Cycle": ("2020-02-19", "2020-05-29"),
    "2022_Rate_Hike_Bear_Market": ("2022-01-03", "2022-10-12"),
    "2022_Full_Year": ("2022-01-03", "2022-12-30"),
}


@dataclass
class CrisisPeriodResult:
    """Performance evaluation of a strategy within a specific stress window."""

    period_name: str
    start_date: str
    end_date: str
    trading_bars: int
    strategy_total_return: float
    strategy_annualized_return: float
    strategy_max_drawdown: float
    strategy_sharpe_ratio: float
    benchmark_total_return: float | None = None
    benchmark_max_drawdown: float | None = None
    excess_return: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "period_name": self.period_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "trading_bars": self.trading_bars,
            "strategy_return_pct": round(self.strategy_total_return * 100, 2),
            "strategy_cagr_pct": round(self.strategy_annualized_return * 100, 2),
            "strategy_max_dd_pct": round(self.strategy_max_drawdown * 100, 2),
            "strategy_sharpe": round(self.strategy_sharpe_ratio, 2),
            "benchmark_return_pct": (
                round(self.benchmark_total_return * 100, 2)
                if self.benchmark_total_return is not None
                else "N/A"
            ),
            "benchmark_max_dd_pct": (
                round(self.benchmark_max_drawdown * 100, 2)
                if self.benchmark_max_drawdown is not None
                else "N/A"
            ),
            "excess_return_pct": (
                round(self.excess_return * 100, 2) if self.excess_return is not None else "N/A"
            ),
        }


def evaluate_crisis_periods(
    backtest_results: Any,
    benchmark_returns: pd.Series | None = None,
    crisis_periods: dict[str, tuple[str, str]] | None = None,
) -> dict[str, CrisisPeriodResult]:
    """Replay and evaluate backtest performance strictly within historical crisis windows.

    Parameters
    ----------
    backtest_results : BacktestResult, PairsBacktestResult, or dict
        Simulation result container.
    benchmark_returns : pd.Series, optional
        Benchmark daily return series.
    crisis_periods : dict, optional
        Dictionary mapping period name to (start_date, end_date) strings.

    Returns
    -------
    dict[str, CrisisPeriodResult]
        Evaluated metrics for each stress period.
    """
    equity, returns, _, _, b_returns = extract_backtest_components(backtest_results)
    if benchmark_returns is not None:
        b_returns = benchmark_returns

    periods = crisis_periods or STANDARD_CRISIS_PERIODS
    results: dict[str, CrisisPeriodResult] = {}

    for name, (start_str, end_str) in periods.items():
        # Mask slice
        sub_equity = equity.loc[start_str:end_str]
        sub_returns = returns.loc[start_str:end_str]

        if len(sub_equity) < 5:
            logger.warning(
                "Period %s has insufficient data in backtest (%d bars). Skipping.",
                name,
                len(sub_equity),
            )
            continue

        n_bars = len(sub_equity)
        strat_tot_ret = float((sub_equity.iloc[-1] / sub_equity.iloc[0]) - 1.0)
        strat_ann_ret = (
            float((1.0 + strat_tot_ret) ** (252.0 / n_bars) - 1.0) if strat_tot_ret > -1.0 else -1.0
        )
        strat_max_dd = float(calculate_max_drawdown(sub_returns)[0])
        strat_sharpe = float(calculate_sharpe_ratio(sub_returns))

        bench_tot_ret = None
        bench_max_dd = None
        excess = None

        if b_returns is not None:
            sub_b_ret = b_returns.loc[start_str:end_str]
            if len(sub_b_ret) > 0:
                b_cum = (1.0 + sub_b_ret).cumprod()
                bench_tot_ret = float(b_cum.iloc[-1] - 1.0)
                bench_max_dd = float(calculate_max_drawdown(sub_b_ret)[0])
                excess = strat_tot_ret - bench_tot_ret

        results[name] = CrisisPeriodResult(
            period_name=name,
            start_date=str(sub_equity.index[0]),
            end_date=str(sub_equity.index[-1]),
            trading_bars=n_bars,
            strategy_total_return=strat_tot_ret,
            strategy_annualized_return=strat_ann_ret,
            strategy_max_drawdown=strat_max_dd,
            strategy_sharpe_ratio=strat_sharpe,
            benchmark_total_return=bench_tot_ret,
            benchmark_max_drawdown=bench_max_dd,
            excess_return=excess,
        )

    return results


# ============================================================
# 2. Monte Carlo Simulation (IID & Circular Block Bootstrap)
# ============================================================


@dataclass
class MonteCarloResult:
    """Results container for Monte Carlo return resampling simulation."""

    simulated_equity_paths: np.ndarray  # Shape: (n_simulations, n_bars)
    percentile_equity_paths: dict[int, np.ndarray]  # 5, 25, 50, 75, 95
    terminal_equity_distribution: np.ndarray
    max_drawdown_distribution: np.ndarray
    percentile_terminal_return: dict[int, float]
    percentile_max_drawdown: dict[int, float]
    method: str
    n_simulations: int
    block_size: int

    def summary_dict(self) -> dict[str, Any]:
        """Return key statistical summary."""
        return {
            "method": self.method,
            "n_simulations": self.n_simulations,
            "block_size": self.block_size,
            "terminal_return_5th_pct": round(self.percentile_terminal_return[5] * 100, 2),
            "terminal_return_50th_pct": round(self.percentile_terminal_return[50] * 100, 2),
            "terminal_return_95th_pct": round(self.percentile_terminal_return[95] * 100, 2),
            "max_drawdown_5th_pct": round(self.percentile_max_drawdown[5] * 100, 2),
            "max_drawdown_50th_pct": round(self.percentile_max_drawdown[50] * 100, 2),
            "max_drawdown_95th_pct": round(self.percentile_max_drawdown[95] * 100, 2),
        }


def circular_block_bootstrap(
    returns: np.ndarray,
    n_simulations: int = 1000,
    block_size: int = 10,
    random_state: int | None = 42,
) -> np.ndarray:
    """Resample returns using a Circular Block Bootstrap to preserve autocorrelation

    and volatility clustering.

    Parameters
    ----------
    returns : np.ndarray
        1D array of historical periodic returns.
    n_simulations : int, default 1000
        Number of synthetic paths to generate.
    block_size : int, default 10
        Length of contiguous blocks (in bars).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Simulated returns array of shape (n_simulations, n_bars).
    """
    rng = np.random.default_rng(random_state)
    n = len(returns)
    n_blocks = int(np.ceil(n / block_size))

    # Extended array to handle wrap-around circular sampling
    extended = np.concatenate([returns, returns[:block_size]])

    simulated_paths = np.empty((n_simulations, n), dtype=float)

    for i in range(n_simulations):
        # Pick random starting indices for blocks
        start_indices = rng.integers(0, n, size=n_blocks)
        sampled_blocks = [extended[idx : idx + block_size] for idx in start_indices]
        concatenated = np.concatenate(sampled_blocks)[:n]
        simulated_paths[i, :] = concatenated

    return simulated_paths


def run_monte_carlo_simulation(
    returns: pd.Series | np.ndarray,
    n_simulations: int = 1000,
    block_size: int = 10,
    initial_capital: float = 100000.0,
    method: str = "block",
    random_state: int | None = 42,
) -> MonteCarloResult:
    """Run Monte Carlo simulation over historical returns.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Historical strategy daily returns.
    n_simulations : int, default 1000
        Number of simulation paths.
    block_size : int, default 10
        Block size for block bootstrap (ignored if method="iid").
    initial_capital : float, default 100000.0
        Starting portfolio cash.
    method : {"block", "iid"}, default "block"
        Resampling algorithm. "block" preserves volatility clustering.
    random_state : int, optional
        Seed.

    Returns
    -------
    MonteCarloResult
        Simulated paths, percentile trajectories, and drawdown distributions.
    """
    clean_rets = np.asarray(returns, dtype=float)
    clean_rets = clean_rets[~np.isnan(clean_rets)]
    n = len(clean_rets)

    if method == "block":
        sim_rets = circular_block_bootstrap(
            clean_rets,
            n_simulations=n_simulations,
            block_size=block_size,
            random_state=random_state,
        )
    elif method == "iid":
        rng = np.random.default_rng(random_state)
        indices = rng.integers(0, n, size=(n_simulations, n))
        sim_rets = clean_rets[indices]
    else:
        raise ValueError(f"Unknown Monte Carlo method: {method}. Must be 'block' or 'iid'.")

    # Compute equity curves: shape (n_simulations, n + 1)
    cum_factors = np.cumprod(1.0 + sim_rets, axis=1)
    # prepend initial capital
    init_col = np.ones((n_simulations, 1)) * initial_capital
    sim_equity = np.hstack([init_col, initial_capital * cum_factors])

    # Terminal returns
    terminal_eq = sim_equity[:, -1]
    terminal_rets = (terminal_eq / initial_capital) - 1.0

    # Max drawdowns per simulation path
    peaks = np.maximum.accumulate(sim_equity, axis=1)
    drawdowns = (sim_equity - peaks) / peaks
    max_dds = np.min(drawdowns, axis=1)

    # Percentiles
    percentiles = [5, 25, 50, 75, 95]
    pct_paths = {p: np.percentile(sim_equity, p, axis=0) for p in percentiles}
    pct_terminal = {p: float(np.percentile(terminal_rets, p)) for p in percentiles}
    pct_max_dd = {p: float(np.percentile(max_dds, p)) for p in percentiles}

    return MonteCarloResult(
        simulated_equity_paths=sim_equity,
        percentile_equity_paths=pct_paths,
        terminal_equity_distribution=terminal_rets,
        max_drawdown_distribution=max_dds,
        percentile_terminal_return=pct_terminal,
        percentile_max_drawdown=pct_max_dd,
        method=method,
        n_simulations=n_simulations,
        block_size=block_size,
    )


def plot_monte_carlo_distribution(
    mc_result: MonteCarloResult,
    dates: pd.DatetimeIndex | None = None,
    title: str = "Monte Carlo Equity Curve Simulation (Block Bootstrap)",
) -> plt.Figure:
    """Plot Monte Carlo fan chart showing 5th, 25th, 50th, 75th, and 95th percentiles."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]})

    x_axis = np.arange(mc_result.simulated_equity_paths.shape[1])
    if dates is not None and len(dates) == len(x_axis) - 1:
        x_axis = [dates[0]] + list(dates)

    p5 = mc_result.percentile_equity_paths[5]
    p25 = mc_result.percentile_equity_paths[25]
    p50 = mc_result.percentile_equity_paths[50]
    p75 = mc_result.percentile_equity_paths[75]
    p95 = mc_result.percentile_equity_paths[95]

    # Shaded confidence bands
    ax1.fill_between(x_axis, p5, p95, color="#00E5FF", alpha=0.15, label="5th–95th Percentile Band")
    ax1.fill_between(
        x_axis, p25, p75, color="#00E5FF", alpha=0.30, label="25th–75th Percentile Band"
    )
    ax1.plot(x_axis, p50, color="#FFD600", linewidth=2.0, label="Median Path (50th %ile)")
    ax1.plot(
        x_axis, p5, color="#FF5252", linestyle="--", linewidth=1.2, label="5th %ile (Adverse Tail)"
    )
    ax1.plot(
        x_axis, p95, color="#00E676", linestyle="--", linewidth=1.2, label="95th %ile (Bull Tail)"
    )

    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.set_ylabel("Portfolio Equity ($)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", framealpha=0.9)

    # Histogram of Terminal Returns
    rets_pct = mc_result.terminal_equity_distribution * 100.0
    ax2.hist(rets_pct, bins=30, color="#7C4DFF", edgecolor="white", alpha=0.75)
    ax2.axvline(
        mc_result.percentile_terminal_return[5] * 100.0,
        color="#FF5252",
        linestyle="--",
        label=f"5th: {mc_result.percentile_terminal_return[5]*100:.1f}%",
    )
    ax2.axvline(
        mc_result.percentile_terminal_return[50] * 100.0,
        color="#FFD600",
        linestyle="-",
        label=f"Median: {mc_result.percentile_terminal_return[50]*100:.1f}%",
    )
    ax2.set_title("Terminal Return Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Total Return (%)", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig


# ============================================================
# 3. Regime-Conditional Performance Breakdown
# ============================================================


def analyze_regime_performance(
    returns: pd.Series,
    regime_labels: pd.Series,
    regime_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Segment and evaluate performance metrics conditionally across detected market regimes.

    Parameters
    ----------
    returns : pd.Series
        Strategy daily returns.
    regime_labels : pd.Series
        Regime state indices from Phase 16 (0, 1, ..., K-1).
    regime_names : dict, optional
        Mapping of regime state integer to descriptive name
        (e.g. {0: "Low-Vol Trending", 1: "Medium-Vol Transition", 2: "High-Vol Crisis"}).

    Returns
    -------
    pd.DataFrame
        Table summarizing bars, return, Sharpe, volatility, and max drawdown by regime.
    """
    aligned = pd.DataFrame({"returns": returns, "regime": regime_labels}).dropna()
    if aligned.empty:
        raise ValueError("No overlapping data between returns and regime_labels.")

    unique_regimes = sorted(aligned["regime"].unique())
    default_names = {
        0: "Low-Vol Trending",
        1: "Medium-Vol Transition",
        2: "High-Vol Crisis",
    }
    names = regime_names or default_names

    rows = []
    total_bars = len(aligned)

    for r in unique_regimes:
        sub = aligned[aligned["regime"] == r]["returns"]
        n_bars = len(sub)
        pct_time = (n_bars / total_bars) * 100.0

        cum_ret = float((1.0 + sub).prod() - 1.0)
        ann_ret = (
            float((1.0 + cum_ret) ** (252.0 / max(1, n_bars)) - 1.0) if cum_ret > -1.0 else -1.0
        )
        ann_vol = float(sub.std() * np.sqrt(252.0))
        sharpe = float(calculate_sharpe_ratio(sub))
        sortino = float(calculate_sortino_ratio(sub))
        max_dd = float(calculate_max_drawdown(sub)[0])
        win_rate = float((sub > 0).mean() * 100.0)

        rows.append(
            {
                "regime_id": int(r),
                "regime_name": names.get(int(r), f"Regime {r}"),
                "bars": n_bars,
                "pct_of_time": round(pct_time, 1),
                "annualized_return_pct": round(ann_ret * 100, 2),
                "annualized_vol_pct": round(ann_vol * 100, 2),
                "sharpe_ratio": round(sharpe, 2),
                "sortino_ratio": round(sortino, 2),
                "max_drawdown_pct": round(max_dd * 100, 2),
                "win_rate_pct": round(win_rate, 2),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 4. Synthetic Scenario Stress Tests (Phase 39 Safety Gate)
# ============================================================


@dataclass
class ScenarioStressTestResult:
    """Result of a synthetic stress shock applied to the portfolio safety gate."""

    scenario_name: str
    description: str
    pre_shock_equity: float
    post_shock_equity: float
    simulated_drawdown_pct: float
    volatility_ratio: float
    risk_decisions: list[OrderDecision]
    safety_systems_activated: list[str]
    is_capital_protected: bool

    def summary_dict(self) -> dict[str, Any]:
        """Summary dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "simulated_drawdown_pct": round(self.simulated_drawdown_pct * 100, 2),
            "volatility_ratio": round(self.volatility_ratio, 2),
            "safety_systems_activated": self.safety_systems_activated,
            "is_capital_protected": self.is_capital_protected,
            "decisions_count": len(self.risk_decisions),
        }


def run_scenario_stress_tests(
    risk_engine: RiskEngine,
    base_portfolio_equity: float = 100000.0,
    tickers: Sequence[str] = ("AAPL", "MSFT"),
    prices: dict[str, float] | None = None,
) -> dict[str, ScenarioStressTestResult]:
    """Execute synthetic stress shock scenarios to probe the Phase 39 RiskEngine safety gates.

    Scenarios Tested:
    1. Volatility Doubling Shock (+100% vol spike overnight):
       Tests whether GARCH volatility de-risking logic actively scales down order sizes.
    2. Flash Crash / Gap-Down Shock (-10% overnight gap):
       Tests whether Daily Loss Limit circuit breaker and Drawdown Kill Switch halt new risk.
    3. Severe Drawdown Regime (>16% accumulated drawdown):
       Tests whether new position openings are completely blocked while de-risking orders are permitted.

    Parameters
    ----------
    risk_engine : RiskEngine
        Configured Phase 39 RiskEngine instance.
    base_portfolio_equity : float, default 100000.0
        Starting portfolio equity.
    tickers : Sequence of str
        Tickers to simulate orders for.
    prices : dict, optional
        Current market prices.

    Returns
    -------
    dict[str, ScenarioStressTestResult]
        Stress scenario diagnostic results.
    """
    px = prices or {"AAPL": 150.0, "MSFT": 300.0, "SPY": 400.0}
    results = {}

    # -------------------------------------------------------------
    # Scenario 1: Overnight Volatility Doubling (GARCH De-risking)
    # -------------------------------------------------------------
    state_vol = PortfolioState(
        current_equity=base_portfolio_equity,
        peak_equity=base_portfolio_equity,
        daily_start_equity=base_portfolio_equity,
        current_date="2026-09-16",
        positions={},
    )
    baseline_vol = 0.15
    spiked_vol = 0.35  # >2x baseline

    decisions_vol = []
    for t in tickers:
        p = px.get(t, 100.0)
        intent = OrderIntent(
            ticker=t,
            action=OrderAction.BUY,
            quantity=100.0,
            price=p,
        )
        dec = risk_engine.evaluate_order(
            intent=intent,
            portfolio=state_vol,
            market_volatility={t: (spiked_vol, baseline_vol)},
        )
        decisions_vol.append(dec)

    # Verify de-risking
    vol_resized = any(
        d.status == DecisionStatus.RESIZED and d.rule_triggered == RiskRule.VOLATILITY_DERISKING
        for d in decisions_vol
    )

    results["volatility_doubling"] = ScenarioStressTestResult(
        scenario_name="Volatility Doubling (+100% overnight vol jump)",
        description="Tests dynamic volatility de-risking haircut when GARCH forecast spikes above baseline.",
        pre_shock_equity=base_portfolio_equity,
        post_shock_equity=base_portfolio_equity,
        simulated_drawdown_pct=0.0,
        volatility_ratio=spiked_vol / baseline_vol,
        risk_decisions=decisions_vol,
        safety_systems_activated=["VOLATILITY_DERISKING"] if vol_resized else [],
        is_capital_protected=vol_resized,
    )

    # -------------------------------------------------------------
    # Scenario 2: Flash Crash / Overnight Gap Down (-10% loss)
    # -------------------------------------------------------------
    crashed_equity = base_portfolio_equity * 0.90
    state_crash = PortfolioState(
        current_equity=crashed_equity,
        peak_equity=base_portfolio_equity,
        daily_start_equity=base_portfolio_equity,
        current_date="2026-09-16",
        realized_pnl_today=-10000.0,
        unrealized_pnl_today=0.0,
        positions={},
    )

    decisions_crash = []
    for t in tickers:
        p = px.get(t, 100.0) * 0.90
        # Intend to buy dip
        intent = OrderIntent(
            ticker=t,
            action=OrderAction.BUY,
            quantity=100.0,
            price=p,
        )
        dec = risk_engine.evaluate_order(
            intent=intent,
            portfolio=state_crash,
        )
        decisions_crash.append(dec)

    daily_loss_blocked = any(
        d.status == DecisionStatus.BLOCKED and d.rule_triggered == RiskRule.DAILY_LOSS_LIMIT
        for d in decisions_crash
    )

    results["flash_gap_down"] = ScenarioStressTestResult(
        scenario_name="Flash Gap Down (-10% overnight crash)",
        description="Tests whether the Daily Loss Circuit Breaker halts all new risk opening.",
        pre_shock_equity=base_portfolio_equity,
        post_shock_equity=crashed_equity,
        simulated_drawdown_pct=0.10,
        volatility_ratio=1.0,
        risk_decisions=decisions_crash,
        safety_systems_activated=["DAILY_LOSS_LIMIT"] if daily_loss_blocked else [],
        is_capital_protected=daily_loss_blocked,
    )

    # -------------------------------------------------------------
    # Scenario 3: Maximum Drawdown Breach (>16% accumulated drawdown)
    # -------------------------------------------------------------
    drawdown_equity = base_portfolio_equity * 0.82  # 18% drawdown
    state_dd = PortfolioState(
        current_equity=drawdown_equity,
        peak_equity=base_portfolio_equity,
        daily_start_equity=drawdown_equity,
        current_date="2026-09-16",
        positions={"AAPL": 20000.0},
    )

    decisions_dd = []
    # New risk order: should be blocked
    intent_new = OrderIntent(
        ticker="MSFT",
        action=OrderAction.BUY,
        quantity=50.0,
        price=px.get("MSFT", 300.0),
        is_closing=False,
    )
    dec_new = risk_engine.evaluate_order(intent=intent_new, portfolio=state_dd)
    decisions_dd.append(dec_new)

    # De-risking order: closing existing AAPL exposure should be APPROVED
    intent_close = OrderIntent(
        ticker="AAPL",
        action=OrderAction.SELL,
        quantity=50.0,
        price=px.get("AAPL", 150.0),
        is_closing=True,
    )
    dec_close = risk_engine.evaluate_order(intent=intent_close, portfolio=state_dd)
    decisions_dd.append(dec_close)

    dd_kill_switch_active = (
        dec_new.status == DecisionStatus.BLOCKED
        and dec_new.rule_triggered == RiskRule.DRAWDOWN_KILL_SWITCH
        and dec_close.status == DecisionStatus.APPROVED
    )

    results["drawdown_kill_switch"] = ScenarioStressTestResult(
        scenario_name="Drawdown Breach (18% drawdown)",
        description="Tests whether Max Drawdown Kill Switch halts new risk while permitting de-risking trades.",
        pre_shock_equity=base_portfolio_equity,
        post_shock_equity=drawdown_equity,
        simulated_drawdown_pct=0.18,
        volatility_ratio=1.0,
        risk_decisions=decisions_dd,
        safety_systems_activated=["DRAWDOWN_KILL_SWITCH"] if dd_kill_switch_active else [],
        is_capital_protected=dd_kill_switch_active,
    )

    return results
