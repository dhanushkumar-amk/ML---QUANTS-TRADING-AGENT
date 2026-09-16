# ============================================================
# Backtesting Engine & Execution Simulation (Phases 40 + 41)
# ============================================================
"""
Production backtesting simulation engine with realistic transaction friction.

Features:
1. Event/Bar-driven simulation with strict no-lookahead execution (orders generated
   at bar t are executed at bar t+1).
2. Friction Modeling: Linear & non-linear slippage, proportional commissions,
   and market impact.
3. Architectural Safety Integration: Directly integrates with Phase 39 RiskEngine,
   routing all target orders through risk evaluation before execution.
4. Comprehensive Trade Logging: Tracks trade-level entries, exits, holding periods,
   realized PnL, win/loss stats, and portfolio equity series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.financial_metrics import (
    calculate_calmar_ratio,
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
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Data Structures & Results Containers
# ============================================================


@dataclass
class TradeRecord:
    """Detailed record of an executed trade from entry to exit."""

    ticker: str
    entry_date: pd.Timestamp | str
    exit_date: pd.Timestamp | str
    side: str  # "LONG" or "SHORT"
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    holding_period_days: int
    exit_reason: str = "SIGNAL"
    commission: float = 0.0
    slippage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert trade record to dictionary."""
        return {
            "ticker": self.ticker,
            "entry_date": str(self.entry_date),
            "exit_date": str(self.exit_date),
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct * 100, 2),
            "holding_period_days": self.holding_period_days,
            "exit_reason": self.exit_reason,
            "commission": round(self.commission, 2),
            "slippage": round(self.slippage, 2),
        }


@dataclass
class TransactionCostModel:
    """Configurable friction model for commission, spread, and market impact."""

    commission_bps: float = 5.0  # 5 bps = 0.05%
    slippage_bps: float = 5.0  # 5 bps = 0.05%
    impact_exponent: float = 0.5  # Square-root impact law

    def calculate_cost(
        self,
        trade_value: float,
        volume: float | None = None,
        trade_shares: float | None = None,
    ) -> tuple[float, float]:
        """Calculate commission and slippage costs.

        Returns (commission_dollar, slippage_dollar).
        """
        comm = abs(trade_value) * (self.commission_bps / 10000.0)
        base_slip = abs(trade_value) * (self.slippage_bps / 10000.0)

        if volume is not None and volume > 0 and trade_shares is not None and trade_shares > 0:
            participation = trade_shares / volume
            impact_scale = max(1.0, (participation / 0.01) ** self.impact_exponent)
            slip = base_slip * impact_scale
        else:
            slip = base_slip

        return comm, slip


@dataclass
class BacktestResult:
    """Comprehensive container for backtest performance and telemetry."""

    strategy_name: str
    initial_capital: float
    portfolio_equity: pd.Series
    daily_returns: pd.Series
    positions: pd.DataFrame
    trades: list[TradeRecord]
    cash: pd.Series
    turnover: pd.Series
    commission_paid: pd.Series
    slippage_paid: pd.Series
    benchmark_equity: pd.Series | None = None
    benchmark_returns: pd.Series | None = None
    risk_decisions: list[OrderDecision] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        """Total strategy return over backtest period."""
        if len(self.portfolio_equity) < 2:
            return 0.0
        return float((self.portfolio_equity.iloc[-1] / self.initial_capital) - 1.0)

    @property
    def annualized_return(self) -> float:
        """Annualized compounded return (CAGR)."""
        n_days = max(1, len(self.portfolio_equity))
        if self.total_return <= -1.0:
            return -1.0
        return float((1.0 + self.total_return) ** (252.0 / n_days) - 1.0)

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio (risk-free rate = 0%)."""
        return calculate_sharpe_ratio(self.daily_returns)

    @property
    def sortino_ratio(self) -> float:
        """Annualized Sortino ratio."""
        return calculate_sortino_ratio(self.daily_returns)

    @property
    def max_drawdown(self) -> float:
        """Maximum peak-to-trough portfolio drawdown."""
        return float(calculate_max_drawdown(self.daily_returns)[0])

    @property
    def calmar_ratio(self) -> float:
        """Annualized return divided by absolute maximum drawdown."""
        return calculate_calmar_ratio(self.daily_returns)

    @property
    def total_trades(self) -> int:
        """Total completed roundtrip trades."""
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        """Fraction of profitable completed trades."""
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return float(wins / len(self.trades))

    @property
    def profit_factor(self) -> float:
        """Gross winning dollar PnL divided by gross losing dollar PnL."""
        gross_win = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if gross_loss <= 1e-9:
            return float(gross_win if gross_win > 0 else 1.0)
        return float(gross_win / gross_loss)

    @property
    def average_holding_period(self) -> float:
        """Average holding period across completed trades in trading days."""
        if not self.trades:
            return 0.0
        return float(np.mean([t.holding_period_days for t in self.trades]))

    def summary_dict(self) -> dict[str, Any]:
        """Serializable dictionary summarizing backtest key metrics."""
        return {
            "strategy_name": self.strategy_name,
            "total_return_pct": round(self.total_return * 100, 2),
            "annualized_return_pct": round(self.annualized_return * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            "total_trades": self.total_trades,
            "win_rate_pct": round(self.win_rate * 100, 2),
            "profit_factor": round(self.profit_factor, 2),
            "avg_holding_period_days": round(self.average_holding_period, 1),
            "total_commission": round(float(self.commission_paid.sum()), 2),
            "total_slippage": round(float(self.slippage_paid.sum()), 2),
        }


# ============================================================
# 2. Backtest Engine Implementation
# ============================================================


class BacktestEngine:
    """Production backtesting engine supporting single/multi-asset strategies,

    friction modeling, and Phase 39 RiskEngine safety enforcement.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        cost_model: TransactionCostModel | None = None,
        risk_engine: RiskEngine | None = None,
        strategy_name: str = "QuantitativeStrategy",
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.cost_model = cost_model or TransactionCostModel()
        self.risk_engine = risk_engine
        self.strategy_name = strategy_name

    def run(
        self,
        prices: pd.DataFrame | pd.Series,
        weights: pd.DataFrame | pd.Series,
        benchmark_prices: pd.Series | None = None,
        volumes: pd.DataFrame | pd.Series | None = None,
        volatilities: pd.DataFrame | pd.Series | None = None,
    ) -> BacktestResult:
        """Run backtest simulation.

        Parameters
        ----------
        prices : pd.DataFrame or pd.Series
            Close prices for assets over the backtest index.
        weights : pd.DataFrame or pd.Series
            Target allocation weights for each asset. Signals are generated at close t
            and executed at bar t+1 (strictly shifted by 1 bar).
        benchmark_prices : pd.Series, optional
            Close prices for benchmark (e.g., SPY) aligned with backtest index.
        volumes : pd.DataFrame or pd.Series, optional
            Trading volumes for non-linear impact estimation.
        volatilities : pd.DataFrame or pd.Series, optional
            Current volatility estimates for GARCH de-risking gate.

        Returns
        -------
        BacktestResult
            Full backtest metrics and series.
        """
        # Standardize prices and weights to DataFrames
        if isinstance(prices, pd.Series):
            df_prices = prices.to_frame(name=prices.name or "ASSET")
        else:
            df_prices = prices.copy()

        if isinstance(weights, pd.Series):
            df_weights = weights.to_frame(name=weights.name or df_prices.columns[0])
        else:
            df_weights = weights.copy()

        # Align columns and index
        common_cols = [c for c in df_prices.columns if c in df_weights.columns]
        if not common_cols:
            raise ValueError("No overlapping ticker columns between prices and weights.")

        df_prices = df_prices[common_cols].dropna(how="all")
        df_weights = df_weights[common_cols].reindex(df_prices.index).fillna(0.0)

        # STRICT NO-LOOKAHEAD: target weights generated on close t are executed at t+1
        exec_weights = df_weights.shift(1).fillna(0.0)

        dates = df_prices.index
        tickers = common_cols

        # Telemetry containers
        equity_series = pd.Series(index=dates, dtype=float)
        cash_series = pd.Series(index=dates, dtype=float)
        returns_series = pd.Series(index=dates, dtype=float)
        turnover_series = pd.Series(index=dates, dtype=float)
        comm_series = pd.Series(index=dates, dtype=float)
        slip_series = pd.Series(index=dates, dtype=float)
        position_shares = pd.DataFrame(index=dates, columns=tickers, dtype=float)

        trades: list[TradeRecord] = []
        risk_decisions: list[OrderDecision] = []

        # State tracking
        current_cash = self.initial_capital
        current_shares: dict[str, float] = dict.fromkeys(tickers, 0.0)
        peak_equity = self.initial_capital
        day_start_equity = self.initial_capital

        # Open trade positions tracker: ticker -> dict(entry_date, entry_price, quantity, side)
        open_trades: dict[str, dict[str, Any]] = {}

        for i, dt in enumerate(dates):
            current_bar_prices = df_prices.iloc[i]

            # 1. Update valuation at current bar prices
            positions_value = sum(
                current_shares[t] * float(current_bar_prices[t])
                for t in tickers
                if not np.isnan(current_bar_prices[t])
            )
            current_equity = current_cash + positions_value
            if current_equity > peak_equity:
                peak_equity = current_equity

            # 2. Daily reset for risk engine
            if i > 0 and (getattr(dt, "date", None) != getattr(dates[i - 1], "date", None)):
                day_start_equity = current_equity

            # 3. Portfolio state for RiskEngine evaluation
            portfolio_state = PortfolioState(
                current_equity=current_equity,
                peak_equity=peak_equity,
                daily_start_equity=day_start_equity,
                current_date=dt,
                positions={
                    t: current_shares[t] * float(current_bar_prices[t])
                    for t in tickers
                    if not np.isnan(current_bar_prices[t])
                },
            )

            # 4. Target allocation and order evaluation
            bar_comm = 0.0
            bar_slip = 0.0
            bar_turnover = 0.0

            target_w = exec_weights.iloc[i]

            for t in tickers:
                p = float(current_bar_prices[t])
                if np.isnan(p) or p <= 0:
                    continue

                target_dollar = float(target_w[t]) * current_equity
                target_share_count = target_dollar / p
                share_diff = target_share_count - current_shares[t]

                if abs(share_diff) < 1e-4:
                    continue

                # Prepare order intent
                action = OrderAction.BUY if share_diff > 0 else OrderAction.SELL
                # If buying while short or selling while long, it is reducing/closing exposure
                is_closing = (current_shares[t] > 0 and share_diff < 0) or (
                    current_shares[t] < 0 and share_diff > 0
                )
                intent = OrderIntent(
                    ticker=t,
                    action=action,
                    quantity=abs(share_diff),
                    price=p,
                    timestamp=dt,
                    is_closing=is_closing,
                    metadata={"target_allocation": float(target_w[t])},
                )

                approved_shares = abs(share_diff)
                if self.risk_engine is not None:
                    # Retrieve volatility if available
                    curr_vol = None
                    if volatilities is not None:
                        if isinstance(volatilities, pd.DataFrame) and t in volatilities.columns:
                            curr_vol = float(volatilities.loc[dt, t])
                        elif isinstance(volatilities, pd.Series):
                            curr_vol = float(volatilities.loc[dt])

                    market_vol = None
                    if curr_vol is not None:
                        market_vol = {t: (curr_vol, 0.15)}

                    decision = self.risk_engine.evaluate_order(
                        intent=intent,
                        portfolio=portfolio_state,
                        market_volatility=market_vol,
                    )
                    risk_decisions.append(decision)

                    if decision.status == DecisionStatus.BLOCKED:
                        continue
                    approved_shares = decision.approved_quantity

                executed_diff = approved_shares if share_diff > 0 else -approved_shares
                trade_val = abs(executed_diff * p)

                # Friction modeling
                vol = None
                if volumes is not None:
                    if isinstance(volumes, pd.DataFrame) and t in volumes.columns:
                        vol = float(volumes.loc[dt, t])
                    elif isinstance(volumes, pd.Series):
                        vol = float(volumes.loc[dt])

                comm, slip = self.cost_model.calculate_cost(
                    trade_value=trade_val,
                    volume=vol,
                    trade_shares=approved_shares,
                )

                # Update cash and shares
                execution_price = (
                    p + (slip / approved_shares)
                    if executed_diff > 0
                    else p - (slip / approved_shares)
                )
                current_cash -= (executed_diff * execution_price) + comm
                bar_comm += comm
                bar_slip += slip
                bar_turnover += trade_val

                # Trade logging: check for open/close transitions
                old_shares = current_shares[t]
                new_shares = old_shares + executed_diff
                current_shares[t] = new_shares

                # Check if closing or opening trade
                if t in open_trades:
                    ot = open_trades[t]
                    # Partial or full close
                    if (ot["side"] == "LONG" and executed_diff < 0) or (
                        ot["side"] == "SHORT" and executed_diff > 0
                    ):
                        closed_qty = min(ot["quantity"], abs(executed_diff))
                        side_mult = 1.0 if ot["side"] == "LONG" else -1.0
                        trade_pnl = (
                            closed_qty * (execution_price - ot["entry_price"]) * side_mult
                            - comm
                            - slip
                        )
                        trade_pnl_pct = (
                            (trade_pnl / (closed_qty * ot["entry_price"]))
                            if ot["entry_price"] > 0
                            else 0.0
                        )

                        entry_idx = dates.get_loc(ot["entry_date"])
                        curr_idx = i
                        h_days = max(1, curr_idx - entry_idx)

                        trades.append(
                            TradeRecord(
                                ticker=t,
                                entry_date=ot["entry_date"],
                                exit_date=dt,
                                side=ot["side"],
                                quantity=closed_qty,
                                entry_price=ot["entry_price"],
                                exit_price=execution_price,
                                pnl=trade_pnl,
                                pnl_pct=trade_pnl_pct,
                                holding_period_days=h_days,
                                exit_reason="REBALANCE",
                                commission=comm,
                                slippage=slip,
                            )
                        )

                        if abs(new_shares) < 1e-4:
                            del open_trades[t]
                        else:
                            ot["quantity"] = abs(new_shares)
                elif abs(new_shares) > 1e-4:
                    open_trades[t] = {
                        "entry_date": dt,
                        "entry_price": execution_price,
                        "quantity": abs(new_shares),
                        "side": "LONG" if new_shares > 0 else "SHORT",
                    }

            # 5. End of bar accounting
            positions_value = sum(
                current_shares[t] * float(current_bar_prices[t])
                for t in tickers
                if not np.isnan(current_bar_prices[t])
            )
            bar_equity = current_cash + positions_value

            equity_series.iloc[i] = bar_equity
            cash_series.iloc[i] = current_cash
            turnover_series.iloc[i] = bar_turnover
            comm_series.iloc[i] = bar_comm
            slip_series.iloc[i] = bar_slip
            for t in tickers:
                position_shares.iloc[i][t] = current_shares[t]

            if i == 0:
                returns_series.iloc[i] = (bar_equity / self.initial_capital) - 1.0
            else:
                prev_eq = equity_series.iloc[i - 1]
                returns_series.iloc[i] = (bar_equity / prev_eq) - 1.0 if prev_eq > 0 else 0.0

        # Benchmark calculation if provided
        b_equity = None
        b_returns = None
        if benchmark_prices is not None:
            common_idx = dates.intersection(benchmark_prices.index)
            b_aligned = benchmark_prices.loc[common_idx]
            b_returns = b_aligned.pct_change().fillna(0.0)
            b_equity = self.initial_capital * (1.0 + b_returns).cumprod()

        return BacktestResult(
            strategy_name=self.strategy_name,
            initial_capital=self.initial_capital,
            portfolio_equity=equity_series,
            daily_returns=returns_series,
            positions=position_shares,
            trades=trades,
            cash=cash_series,
            turnover=turnover_series,
            commission_paid=comm_series,
            slippage_paid=slip_series,
            benchmark_equity=b_equity,
            benchmark_returns=b_returns,
            risk_decisions=risk_decisions,
        )
