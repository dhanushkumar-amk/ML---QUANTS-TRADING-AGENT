# ============================================================
# Live Paper Trading Orchestration Loop (Phase 46)
# ============================================================
"""
Main orchestration loop for live / paper quantitative trading execution.

=============================================================================
ARCHITECTURAL BLUEPRINT: END-TO-END SYSTEM INTEGRATION
-----------------------------------------------------------------------------
This loop represents the operational culmination of the trading bot platform:
  [Data Feed / Ingestion] (Phase 3: RealtimeBuffer / Polling / Alpaca Stream)
             \u2193
  [Feature Computation] (Phases 11-20: Technical, Volatility, Microstructure)
             \u2193
  [Model Inference & Signal] (Phases 21-34: Directional ML / Momentum / Pairs)
             \u2193
  [Portfolio Sizing] (Phases 35-37: Kelly Criterion / Risk Parity Allocation)
             \u2193
  [Phase 39 Risk Engine] (Hard Gate: Kill switches, Exposure limits, GARCH de-risking)
             \u2193
  [Phase 45 Order Manager] (Idempotency, Limit offset, Partial fill policies)
             \u2193
  [Phase 44 Broker Client] (Alpaca Paper Trading API with Enforced Safety Rail)
             \u2193
  [Phase 47 Audit Trail] (Structured JSONL Logging for 100% Forensic Reconstructibility)

Operational Protocols:
----------------------
1. Market Calendar & Hours Gating:
   Trades strictly during regular exchange sessions (09:30 - 16:00 ET).
   Idles outside market hours to avoid useless polling and fee generation.
2. Startup Position Reconciliation:
   Audits the internal portfolio against the broker's ledger before placing any orders,
   preventing duplicate positions upon bot restarts.
3. Graceful Signal-Trapping & Shutdown:
   Intercepts SIGINT / SIGTERM to cancel open orphan limit orders and reconcile state.
4. Heartbeat & Feed Stall Detection:
   Pulses heartbeats and warns if market data feed stops delivering ticks during market hours.
=============================================================================
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from src.data_pipeline.data_access import DataAccessLayer
from src.data_pipeline.market_hours import is_market_open, market_status
from src.execution.audit_trail import AuditEventType, AuditTrail
from src.execution.broker_client import (
    AlpacaBrokerClient,
    BrokerClient,
    OrderType,
)
from src.execution.order_manager import OrderManager, PartialFillPolicy
from src.portfolio.risk_engine import (
    DecisionStatus,
    OrderAction,
    OrderIntent,
    PortfolioState,
    RiskConfig,
    RiskEngine,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Configuration Container
# ============================================================


@dataclass
class LiveTradingConfig:
    """Runtime configuration for the live paper trading engine."""

    tickers: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "SPY"])
    cycle_interval_sec: float = 60.0  # Frequency of bar evaluation (1 minute)
    market_hours_only: bool = True  # If True, sleep outside NYSE trading sessions
    heartbeat_interval_sec: float = 60.0  # Periodic heartbeat pulse frequency
    max_feed_stall_sec: float = 300.0  # 5 minutes without price updates triggers stall alert
    paper_mode: bool = True  # Strict paper-trading mode
    allow_live: bool = False  # Explicit override flag for real money
    audit_log_dir: str = "logs/audit"
    max_position_size_pct: float = 0.35  # Single-asset cap
    max_drawdown_pct: float = 0.15  # 15% drawdown halt


# ============================================================
# 2. Main Live Trading Loop Orchestrator
# ============================================================


class LiveTradingLoop:
    """Production execution orchestrator tying together data, models, risk, and broker execution."""

    def __init__(
        self,
        config: LiveTradingConfig | None = None,
        broker_client: BrokerClient | None = None,
        risk_engine: RiskEngine | None = None,
        order_manager: OrderManager | None = None,
        audit_trail: AuditTrail | None = None,
        signal_generator: Callable[[dict[str, pd.DataFrame]], dict[str, float]] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        config : LiveTradingConfig, optional
            Runtime operational settings.
        broker_client : BrokerClient, optional
            Broker API client. Defaults to AlpacaBrokerClient in paper mode.
        risk_engine : RiskEngine, optional
            Phase 39 RiskEngine.
        order_manager : OrderManager, optional
            Phase 45 OrderManager.
        audit_trail : AuditTrail, optional
            Phase 47 centralized audit logger.
        signal_generator : Callable, optional
            Function returning target allocation weights {ticker: target_weight} from price bars.
        """
        self.config = config or LiveTradingConfig()
        self.audit = audit_trail or AuditTrail(log_dir=self.config.audit_log_dir)

        # 1. Initialize Broker Client with Safety Rails
        if broker_client is not None:
            self.broker = broker_client
        else:
            self.broker = AlpacaBrokerClient(
                paper=self.config.paper_mode,
                allow_live=self.config.allow_live,
            )

        # 2. Initialize Risk Engine
        self.risk_engine = risk_engine or RiskEngine(
            config=RiskConfig(
                max_drawdown_pct=self.config.max_drawdown_pct,
                max_position_size_pct=self.config.max_position_size_pct,
            )
        )

        # 3. Initialize Order Manager
        self.order_mgr = order_manager or OrderManager(
            broker_client=self.broker,
            risk_engine=self.risk_engine,
            default_partial_fill_policy=PartialFillPolicy.WAIT,
        )

        # 4. Strategy Signal Engine
        self.signal_generator = signal_generator or self._default_trend_hedge_strategy

        # State Telemetry
        self._is_running = False
        self._last_heartbeat_time = 0.0
        self._last_bar_time = time.time()
        self._total_cycles_executed = 0
        self._dal = DataAccessLayer()

    # ---- Graceful Startup & Shutdown Handlers -----------------------

    def startup(self) -> None:
        """Execute pre-flight checks, verify paper mode safety, and reconcile initial positions."""
        logger.info(
            "Initializing Live Trading Loop (Mode: %s)...",
            "PAPER" if self.config.paper_mode else "LIVE",
        )

        # Log system startup event
        self.audit.log_event(
            event_type=AuditEventType.SYSTEM_EVENT,
            details={
                "action": "STARTUP",
                "tickers": self.config.tickers,
                "paper_mode": self.config.paper_mode,
                "cycle_interval_sec": self.config.cycle_interval_sec,
            },
        )

        # Pre-flight broker connection check
        if not self.broker.health_check():
            raise ConnectionError("Broker client health check failed during startup.")

        account = self.broker.get_account_info()
        logger.info(
            "Broker Connected: Cash=$%.2f, Equity=$%.2f, BuyingPower=$%.2f (Paper=%s)",
            account.cash,
            account.equity,
            account.buying_power,
            account.is_paper,
        )

        # Pre-flight Reconciliation: Sync internal accounting with actual broker holdings
        recon_report = self.order_mgr.reconcile()
        self.audit.log_event(
            event_type=AuditEventType.RECONCILIATION,
            details={
                "trigger": "STARTUP",
                "is_synchronized": recon_report.is_synchronized,
                "discrepancy_count": len(recon_report.discrepancies),
                "broker_positions": recon_report.broker_positions,
            },
        )
        logger.info(
            "Initial reconciliation complete: %s (Broker Positions: %s)",
            "SYNCHRONIZED" if recon_report.is_synchronized else "DRIFT DETECTED & ADJUSTED",
            recon_report.broker_positions,
        )

    def shutdown(self, reason: str = "USER_TERMINATION") -> None:
        """Perform orderly shutdown, cancel pending open orders, and record final audit reconciliation."""
        logger.info("Shutting down Live Trading Loop (Reason: %s)...", reason)
        self._is_running = False

        # Clean up any pending / active open orders to prevent orphan execution
        open_orders = [o for o in self.order_mgr.submitted_orders.values() if not o.is_terminal]
        cancelled_count = 0
        for o in open_orders:
            try:
                self.broker.cancel_order(o.order_id)
                cancelled_count += 1
                self.audit.log_event(
                    event_type=AuditEventType.ORDER_CANCELLED,
                    ticker=o.ticker,
                    details={
                        "order_id": o.order_id,
                        "reason": f"SHUTDOWN_CLEANUP ({reason})",
                    },
                )
            except Exception as err:
                logger.warning(
                    "Failed to cancel open order %s during shutdown: %s", o.order_id, err
                )

        # Final reconciliation audit
        final_recon = self.order_mgr.reconcile()
        self.audit.log_event(
            event_type=AuditEventType.SYSTEM_EVENT,
            details={
                "action": "SHUTDOWN",
                "reason": reason,
                "cycles_executed": self._total_cycles_executed,
                "cancelled_open_orders": cancelled_count,
                "final_positions": final_recon.broker_positions,
            },
        )
        logger.info(
            "Shutdown complete. Total cycles: %d. Cancelled open orders: %d.",
            self._total_cycles_executed,
            cancelled_count,
        )

    # ---- Execution Cycle --------------------------------------------

    def execute_cycle(self, cycle_id: str | None = None) -> dict[str, Any]:
        """Execute a single end-to-end trading bar cycle.

        Ingests data -> computes signals -> evaluates risk -> submits orders -> logs audit trail.
        """
        cid = cycle_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._total_cycles_executed += 1
        self._last_bar_time = time.time()

        # 1. Fetch latest prices for configured universe
        price_history: dict[str, pd.DataFrame] = {}
        latest_prices: dict[str, float] = {}

        for t in self.config.tickers:
            try:
                df = self._dal.get_ohlcv(t)
            except Exception as err:
                logger.warning("Could not fetch OHLCV for %s: %s", t, err)
                continue

            if not df.empty:
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date").set_index("date")
                df = df.tail(100)
                price_history[t] = df
                latest_prices[t] = float(df["close"].iloc[-1])

        if not latest_prices:
            logger.warning("No price data available for cycle %s. Skipping.", cid)
            return {"cycle_id": cid, "status": "NO_DATA"}

        # 2. Strategy Inference / Signal Generation
        target_weights = self.signal_generator(price_history)

        # 3. Portfolio State & Account Balance
        account = self.broker.get_account_info()
        positions = self.broker.get_positions()
        pos_dict = {p.ticker.upper(): p.market_value for p in positions}

        state = PortfolioState(
            current_equity=account.equity,
            peak_equity=account.equity,
            daily_start_equity=account.equity,
            current_date=datetime.now(timezone.utc).date(),
            positions=pos_dict,
        )

        cycle_results: dict[str, Any] = {
            "cycle_id": cid,
            "target_weights": target_weights,
            "orders_placed": [],
        }

        # 4. Generate OrderIntents and Submit via RiskEngine & OrderManager
        for t in self.config.tickers:
            p = latest_prices.get(t)
            if p is None or p <= 0:
                continue

            target_w = target_weights.get(t, 0.0)
            target_dollar = target_w * account.equity
            target_shares = target_dollar / p

            # Current shares from broker
            curr_shares = 0.0
            for pos in positions:
                if pos.ticker.upper() == t:
                    curr_shares = pos.quantity
                    break

            share_diff = target_shares - curr_shares
            if abs(share_diff) < 0.01:
                continue

            action = OrderAction.BUY if share_diff > 0 else OrderAction.SELL
            is_closing = (curr_shares > 0 and share_diff < 0) or (
                curr_shares < 0 and share_diff > 0
            )

            intent = OrderIntent(
                ticker=t,
                action=action,
                quantity=abs(share_diff),
                price=p,
                is_closing=is_closing,
                metadata={"target_weight": target_w},
            )

            # Log Signal Generation in Audit Trail
            self.audit.log_event(
                event_type=AuditEventType.SIGNAL_GENERATED,
                ticker=t,
                cycle_id=cid,
                details={
                    "direction": action.value,
                    "target_weight": target_w,
                    "requested_shares": round(abs(share_diff), 4),
                    "price": p,
                },
            )

            # Evaluate Risk Engine Gating & Log Decision
            risk_decision = self.risk_engine.evaluate_order(intent=intent, portfolio=state)
            self.audit.log_event(
                event_type=AuditEventType.RISK_DECISION,
                ticker=t,
                cycle_id=cid,
                details={
                    "status": risk_decision.status.value,
                    "approved_quantity": risk_decision.approved_quantity,
                    "rule_triggered": risk_decision.rule_triggered.value,
                    "reasons": risk_decision.reasons,
                },
            )

            if risk_decision.status == DecisionStatus.BLOCKED:
                logger.warning("Order for %s BLOCKED by Risk Engine: %s", t, risk_decision.reasons)
                continue

            # Submit order through unified pipeline
            order = self.order_mgr.submit_intent(
                intent=intent,
                portfolio_state=state,
                order_type=OrderType.MARKET,
                cycle_id=cid,
            )

            if order is not None:
                cycle_results["orders_placed"].append(order.order_id)
                self.audit.log_event(
                    event_type=AuditEventType.ORDER_SUBMITTED,
                    ticker=t,
                    cycle_id=cid,
                    details={
                        "order_id": order.order_id,
                        "client_order_id": order.client_order_id,
                        "side": order.side.value,
                        "quantity": order.quantity,
                        "status": order.status.value,
                    },
                )

        return cycle_results

    # ---- Continuous Loop & Health Monitoring ------------------------

    def check_heartbeat(self) -> None:
        """Emit periodic health pulse and detect data feed stalls."""
        now = time.time()
        if now - self._last_heartbeat_time >= self.config.heartbeat_interval_sec:
            self._last_heartbeat_time = now
            m_status = market_status()

            self.audit.log_event(
                event_type=AuditEventType.HEARTBEAT,
                details={
                    "total_cycles": self._total_cycles_executed,
                    "market_open": m_status.get("is_open", False),
                    "active_open_orders": len(
                        [o for o in self.order_mgr.submitted_orders.values() if not o.is_terminal]
                    ),
                },
            )
            logger.debug(
                "Heartbeat pulse recorded: %d cycles executed.", self._total_cycles_executed
            )

            # Feed Stall Detection (during market hours only)
            if m_status.get("is_open", False) and (
                now - self._last_bar_time > self.config.max_feed_stall_sec
            ):
                logger.warning(
                    "DATA FEED STALL ALERT: No price updates received in the last %.1f seconds!",
                    now - self._last_bar_time,
                )
                self.audit.log_event(
                    event_type=AuditEventType.FEED_STALL_ALERT,
                    details={
                        "stall_duration_sec": round(now - self._last_bar_time, 1),
                        "max_allowed_sec": self.config.max_feed_stall_sec,
                    },
                )

    def run(
        self,
        max_cycles: int | None = None,
        duration_sec: float | None = None,
    ) -> None:
        """Run the main trading loop continuously until stopped, duration expires, or max cycles reached.

        Parameters
        ----------
        max_cycles : int, optional
            Stop after executing this many cycles.
        duration_sec : float, optional
            Stop after this many seconds elapsed.
        """
        # Register graceful termination handlers if running on main thread
        try:
            signal.signal(signal.SIGINT, lambda sig, frame: self.shutdown("SIGINT"))
            signal.signal(signal.SIGTERM, lambda sig, frame: self.shutdown("SIGTERM"))
        except (ValueError, AttributeError):
            pass

        self.startup()
        self._is_running = True
        start_time = time.time()

        logger.info(
            "Live Trading Loop RUNNING (Interval=%.1fs, MarketHoursOnly=%s)...",
            self.config.cycle_interval_sec,
            self.config.market_hours_only,
        )

        try:
            while self._is_running:
                # Check duration limit
                if duration_sec is not None and (time.time() - start_time >= duration_sec):
                    logger.info("Duration limit reached (%.1fs). Ending loop.", duration_sec)
                    break

                # Check max cycles limit
                if max_cycles is not None and (self._total_cycles_executed >= max_cycles):
                    logger.info("Max cycles limit reached (%d). Ending loop.", max_cycles)
                    break

                # Market Hours Gating Check
                if self.config.market_hours_only and not is_market_open():
                    logger.debug("US Market is currently CLOSED. Idling until next session...")
                    self.check_heartbeat()
                    time.sleep(min(15.0, self.config.cycle_interval_sec))
                    continue

                # Execute active cycle
                self.execute_cycle()
                self.check_heartbeat()

                # Sleep until next bar
                time.sleep(self.config.cycle_interval_sec)

        except Exception as err:
            logger.error("Unhandled exception in live trading loop: %s", err, exc_info=True)
            self.shutdown(reason=f"EXCEPTION: {err}")
            raise
        finally:
            if self._is_running:
                self.shutdown(reason="COMPLETED")

    # ---- Default Built-in Strategy ----------------------------------

    @staticmethod
    def _default_trend_hedge_strategy(
        price_history: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """Default multi-asset trend momentum strategy with dynamic SPY hedge.

        Allocates 30% each to AAPL / MSFT if 20d SMA > 60d SMA, and hedges short SPY (-15%)
        if SPY falls below its 50d SMA.
        """
        weights: dict[str, float] = {}

        for t in ["AAPL", "MSFT"]:
            if t in price_history and len(price_history[t]) >= 60:
                closes = price_history[t]["close"]
                sma20 = closes.rolling(20).mean().iloc[-1]
                sma60 = closes.rolling(60).mean().iloc[-1]
                weights[t] = 0.30 if sma20 > sma60 else 0.0
            else:
                weights[t] = 0.0

        if "SPY" in price_history and len(price_history["SPY"]) >= 50:
            closes = price_history["SPY"]["close"]
            sma50 = closes.rolling(50).mean().iloc[-1]
            last_p = closes.iloc[-1]
            weights["SPY"] = -0.15 if last_p < sma50 else 0.0
        else:
            weights["SPY"] = 0.0

        return weights


# ============================================================
# 3. CLI Command-Line Entrypoint
# ============================================================


def main() -> None:
    """CLI entrypoint for running the live paper trading loop."""
    import argparse

    parser = argparse.ArgumentParser(description="Live Paper Trading Engine (Phases 46 & 47)")
    parser.add_argument(
        "--symbols",
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "NVDA", "SPY"],
        help="Universe of ticker symbols to trade",
    )
    parser.add_argument(
        "--bar-interval",
        type=float,
        default=60.0,
        help="Bar cadence in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Maximum cycles to execute before exiting (default: infinite)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Maximum loop run duration in seconds",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Bypass NYSE market hours gating (for testing/after-hours simulation)",
    )
    parser.add_argument(
        "--audit-dir",
        type=str,
        default="logs/audit",
        help="Directory to persist structured daily audit logs",
    )

    args = parser.parse_args()

    cfg = LiveTradingConfig(
        tickers=args.symbols,
        cycle_interval_sec=args.bar_interval,
        market_hours_only=not args.force_run,
        audit_log_dir=args.audit_dir,
    )

    loop = LiveTradingLoop(config=cfg)
    loop.run(max_cycles=args.max_cycles, duration_sec=args.duration)


if __name__ == "__main__":
    main()
