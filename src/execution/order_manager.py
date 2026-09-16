# ============================================================
# Order Execution Manager & Position Reconciliation (Phase 45)
# ============================================================
"""
Production order management, execution routing, idempotency, and reconciliation.

=============================================================================
ARCHITECTURAL PRINCIPLES: UNIFIED EXECUTION PIPELINE & IDEMPOTENCY
-----------------------------------------------------------------------------
1. Unified Pipeline (Shared Backtest vs. Live Parity):
   The strategy logic, Kelly sizing, and Phase 39 RiskEngine evaluation remain
   strictly identical across simulation and live trading.
   The `OrderManager` consumes the exact same `OrderIntent` and `OrderDecision`
   constructs produced during backtesting, eliminating subtle translation bugs
   and behavioral divergence between research and execution.

2. Idempotency Safeguard (Client Order IDs):
   In networked trading environments, network timeouts or process restarts
   frequently cause duplicate order submissions—one of the most devastating failure
   modes in algorithmic execution.
   The `OrderManager` computes a deterministic SHA256 `client_order_id` based on
   `(cycle_id, ticker, action, quantity)`. If an order is retransmitted, the broker
   recognizes the duplicate client order ID and prevents redundant execution.

3. Partial Fill Policies:
   Orders on illiquid tickers or during volatility spikes can fill partially:
     - `WAIT`: Keep the remaining unfilled quantity active in the market.
     - `CANCEL_AND_RESUBMIT`: Cancel the remaining open tail and resubmit at the latest price.
     - `ACCEPT_PARTIAL`: Cancel the open tail and accept the executed portion as the final fill.

4. Position & Order Reconciliation:
   A live trading bot's internal state can drift from broker reality due to
   missed WebSocket events, manual broker console overrides, stock splits, or fees.
   The `reconcile()` method audits internal books against actual broker ledger
   holdings, immediately alerting on discrepancies.
=============================================================================
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from src.execution.broker_client import (
    BrokerClient,
    BrokerOrder,
    OrderSide,
    OrderStatus,
    OrderType,
)
from src.portfolio.risk_engine import (
    DecisionStatus,
    OrderDecision,
    OrderIntent,
    PortfolioState,
    RiskEngine,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Enums & Data Containers
# ============================================================


class PartialFillPolicy(str, Enum):
    """Handling policy for partially filled orders."""

    WAIT = "wait"
    CANCEL_AND_RESUBMIT = "cancel_and_resubmit"
    ACCEPT_PARTIAL = "accept_partial"


@dataclass
class ReconciliationDiscrepancy:
    """Discrepancy between internal records and broker ledger."""

    ticker: str
    internal_quantity: float
    broker_quantity: float
    discrepancy_delta: float
    message: str
    severity: str = "HIGH"


@dataclass
class ReconciliationReport:
    """Comprehensive audit report comparing internal book with broker state."""

    timestamp: datetime
    is_synchronized: bool
    internal_positions: dict[str, float]
    broker_positions: dict[str, float]
    discrepancies: list[ReconciliationDiscrepancy]
    open_orders_count: int

    def summary_dict(self) -> dict[str, Any]:
        """Convert report to serializable summary dictionary."""
        return {
            "timestamp": str(self.timestamp),
            "is_synchronized": self.is_synchronized,
            "discrepancy_count": len(self.discrepancies),
            "discrepancies": [
                {
                    "ticker": d.ticker,
                    "internal_qty": d.internal_quantity,
                    "broker_qty": d.broker_quantity,
                    "delta": d.discrepancy_delta,
                    "message": d.message,
                }
                for d in self.discrepancies
            ],
            "open_orders_count": self.open_orders_count,
        }


# ============================================================
# 2. Order Execution Manager Implementation
# ============================================================


class OrderManager:
    """Central order lifecycle manager, risk gatekeeper, and state reconciler."""

    def __init__(
        self,
        broker_client: BrokerClient,
        risk_engine: RiskEngine | None = None,
        default_partial_fill_policy: PartialFillPolicy = PartialFillPolicy.WAIT,
        max_retries: int = 3,
        retry_delay_sec: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        broker_client : BrokerClient
            Target broker communication client.
        risk_engine : RiskEngine, optional
            Phase 39 RiskEngine. If provided, all orders are gated through risk evaluation.
        default_partial_fill_policy : PartialFillPolicy, default WAIT
            Default handling policy for partial fills.
        max_retries : int, default 3
            Maximum retry attempts for transient broker submission failures.
        retry_delay_sec : float, default 0.5
            Base backoff delay in seconds between retries.
        """
        self.broker = broker_client
        self.risk_engine = risk_engine
        self.partial_fill_policy = default_partial_fill_policy
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec

        # Internal state tracking
        self.internal_positions: dict[str, float] = {}  # ticker -> shares
        self.submitted_orders: dict[str, BrokerOrder] = {}  # client_order_id -> BrokerOrder
        self.audit_trail: list[dict[str, Any]] = []

    # ---- Idempotency & Order ID Generation ---------------------------

    @staticmethod
    def generate_client_order_id(
        ticker: str,
        action: str,
        quantity: float,
        cycle_id: str | None = None,
    ) -> str:
        """Generate a deterministic client order ID for idempotency.

        Using SHA256 of cycle/date, ticker, action, and rounded quantity.
        """
        cycle = cycle_id or datetime.now().strftime("%Y%m%d_%H")
        seed_str = f"{cycle}:{ticker.upper()}:{action.upper()}:{round(float(quantity), 4)}"
        hash_hex = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:24]
        return f"agy_{hash_hex}"

    # ---- Order Submission Pipeline ----------------------------------

    def submit_intent(
        self,
        intent: OrderIntent,
        portfolio_state: PortfolioState | None = None,
        order_type: OrderType | str = OrderType.MARKET,
        limit_price: float | None = None,
        limit_offset_pct: float = 0.001,  # 10 bps offset from market
        cycle_id: str | None = None,
        market_volatility: dict[str, tuple[float, float]] | None = None,
    ) -> BrokerOrder | None:
        """Process an OrderIntent through the RiskEngine and route to the broker.

        Parameters
        ----------
        intent : OrderIntent
            Proposed trade intent.
        portfolio_state : PortfolioState, optional
            Current dynamic state. If None, derived from broker account info.
        order_type : OrderType or str, default MARKET
            Market or Limit order.
        limit_price : float, optional
            Explicit limit price. If None and order_type is LIMIT, computed using limit_offset_pct.
        limit_offset_pct : float, default 0.001
            Offset percentage added/subtracted to intent.price for limit orders.
        cycle_id : str, optional
            Rebalance cycle tag for deterministic idempotency.
        market_volatility : dict, optional
            GARCH volatility estimates for de-risking gates.

        Returns
        -------
        BrokerOrder or None
            Executed order details, or None if blocked by RiskEngine.
        """
        norm_action = str(intent.action).upper()
        if "." in norm_action:
            norm_action = norm_action.split(".")[-1]

        # 1. Evaluate against Phase 39 RiskEngine (if attached)
        approved_qty = intent.quantity
        if self.risk_engine is not None:
            # Build portfolio state if not provided
            if portfolio_state is None:
                acc = self.broker.get_account_info()
                pos_list = self.broker.get_positions()
                portfolio_state = PortfolioState(
                    current_equity=acc.equity,
                    peak_equity=acc.equity,
                    daily_start_equity=acc.equity,
                    current_date=datetime.now().date(),
                    positions={p.ticker: p.market_value for p in pos_list},
                )

            decision: OrderDecision = self.risk_engine.evaluate_order(
                intent=intent,
                portfolio=portfolio_state,
                market_volatility=market_volatility,
            )

            if decision.status == DecisionStatus.BLOCKED:
                logger.warning(
                    "Order BLOCKED by Risk Engine rule: %s (Ticker=%s, Reasons=%s)",
                    decision.rule_triggered.value,
                    intent.ticker,
                    decision.reasons,
                )
                self.audit_trail.append(decision.to_audit_dict())
                return None

            approved_qty = decision.approved_quantity
            logger.info(
                "Order %s by Risk Engine (Approved Qty: %.2f / Requested: %.2f)",
                decision.status.value,
                approved_qty,
                intent.quantity,
            )

        if approved_qty <= 0:
            logger.warning("Order quantity after risk evaluation is 0. Aborting.")
            return None

        # 2. Derive Limit Price if required
        type_str = order_type.value if hasattr(order_type, "value") else str(order_type)
        if "." in type_str:
            type_str = type_str.split(".")[-1]
        norm_type = OrderType(type_str.lower())
        calculated_limit = limit_price
        if norm_type == OrderType.LIMIT and calculated_limit is None:
            base_px = intent.price
            if norm_action == "BUY":
                # Willing to pay slightly above mid/current quote
                calculated_limit = round(base_px * (1.0 + limit_offset_pct), 2)
            else:
                # Willing to sell slightly below mid/current quote
                calculated_limit = round(base_px * (1.0 - limit_offset_pct), 2)

        # 3. Idempotency Check
        client_order_id = self.generate_client_order_id(
            ticker=intent.ticker,
            action=norm_action,
            quantity=approved_qty,
            cycle_id=cycle_id,
        )

        if client_order_id in self.submitted_orders:
            existing = self.submitted_orders[client_order_id]
            logger.warning(
                "IDEMPOTENCY SAFEGUARD: Order with client_order_id=%s already submitted. "
                "Returning existing order (Status: %s)",
                client_order_id,
                existing.status.value,
            )
            return existing

        # 4. Submit Order with Backoff Retry on Transient Errors
        side_enum = OrderSide.BUY if norm_action == "BUY" else OrderSide.SELL
        order: BrokerOrder | None = None
        delay = self.retry_delay_sec

        for attempt in range(1, self.max_retries + 1):
            try:
                order = self.broker.place_order(
                    ticker=intent.ticker,
                    quantity=approved_qty,
                    side=side_enum,
                    order_type=norm_type,
                    limit_price=calculated_limit,
                    client_order_id=client_order_id,
                )
                break
            except Exception as err:
                if attempt == self.max_retries:
                    logger.error(
                        "Order placement FAILED after %d attempts: %s",
                        self.max_retries,
                        err,
                    )
                    raise
                logger.warning(
                    "Order placement transient failure (attempt %d/%d): %s. Retrying in %.2fs...",
                    attempt,
                    self.max_retries,
                    err,
                    delay,
                )
                time.sleep(delay)
                delay *= 2.0

        if order is not None:
            self.submitted_orders[client_order_id] = order
            # Update internal positions book tentatively
            qty_delta = approved_qty if side_enum == OrderSide.BUY else -approved_qty
            self.internal_positions[intent.ticker] = (
                self.internal_positions.get(intent.ticker, 0.0) + qty_delta
            )
            self.audit_trail.append(
                {
                    "timestamp": str(datetime.now()),
                    "action": "SUBMIT_ORDER",
                    "client_order_id": client_order_id,
                    "order_id": order.order_id,
                    "ticker": order.ticker,
                    "quantity": order.quantity,
                    "side": order.side.value,
                    "status": order.status.value,
                }
            )

        return order

    # ---- Partial Fill Management ------------------------------------

    def handle_partial_fill(
        self,
        order_id: str,
        policy: PartialFillPolicy | None = None,
        resubmit_limit_price: float | None = None,
    ) -> BrokerOrder:
        """Inspect and handle a partially filled order according to configured policy.

        Policies:
        - WAIT: Keep order open.
        - CANCEL_AND_RESUBMIT: Cancel remaining open shares and resubmit as new order.
        - ACCEPT_PARTIAL: Cancel remaining open shares and accept filled portion.
        """
        active_policy = policy or self.partial_fill_policy
        current_order = self.broker.get_order_status(order_id)

        if current_order.status != OrderStatus.PARTIALLY_FILLED:
            logger.info(
                "Order %s status is %s (not partially filled). No action needed.",
                order_id,
                current_order.status.value,
            )
            return current_order

        rem_qty = current_order.remaining_quantity
        logger.info(
            "Order %s is partially filled (Filled: %.2f / Remaining: %.2f). Applying policy: %s",
            order_id,
            current_order.filled_quantity,
            rem_qty,
            active_policy.value,
        )

        if active_policy == PartialFillPolicy.WAIT:
            return current_order

        if active_policy in {
            PartialFillPolicy.ACCEPT_PARTIAL,
            PartialFillPolicy.CANCEL_AND_RESUBMIT,
        }:
            # Cancel the open remainder
            self.broker.cancel_order(order_id)

            if active_policy == PartialFillPolicy.CANCEL_AND_RESUBMIT and rem_qty > 0:
                logger.info(
                    "Resubmitting remaining %.2f shares for order %s",
                    rem_qty,
                    order_id,
                )
                resub_order = self.broker.place_order(
                    ticker=current_order.ticker,
                    quantity=rem_qty,
                    side=current_order.side,
                    order_type=current_order.order_type,
                    limit_price=resubmit_limit_price or current_order.limit_price,
                    client_order_id=f"{current_order.client_order_id}_resub",
                )
                return resub_order

        # Return latest order status after cancellation
        return self.broker.get_order_status(order_id)

    # ---- State & Position Reconciliation ----------------------------

    def reconcile(
        self,
        internal_positions: dict[str, float] | None = None,
        tolerance_shares: float = 1e-4,
    ) -> ReconciliationReport:
        """Audit and compare the system's internal position records with the broker's ledger.

        Parameters
        ----------
        internal_positions : dict, optional
            Internal position book {ticker: shares}. If None, uses self.internal_positions.
        tolerance_shares : float, default 1e-4
            Acceptable floating-point delta before flagging a discrepancy.

        Returns
        -------
        ReconciliationReport
            Comprehensive audit report indicating synchronization status and discrepancies.
        """
        internal_book = (
            internal_positions if internal_positions is not None else self.internal_positions
        )
        broker_pos_list = self.broker.get_positions()
        broker_book = {p.ticker.upper(): float(p.quantity) for p in broker_pos_list}

        all_tickers = set(internal_book.keys()) | set(broker_book.keys())
        discrepancies: list[ReconciliationDiscrepancy] = []

        for t in sorted(all_tickers):
            i_qty = float(internal_book.get(t, 0.0))
            b_qty = float(broker_book.get(t, 0.0))
            delta = b_qty - i_qty

            if abs(delta) > tolerance_shares:
                if i_qty == 0.0 and b_qty != 0.0:
                    msg = f"Ghost Position: Broker holds {b_qty:.2f} shares of {t}, but internal book has 0."
                elif i_qty != 0.0 and b_qty == 0.0:
                    msg = f"Missing Position: Internal book records {i_qty:.2f} shares of {t}, but broker has 0."
                else:
                    msg = f"Quantity Mismatch: Internal={i_qty:.2f}, Broker={b_qty:.2f} (Delta={delta:+.2f})."

                logger.warning("RECONCILIATION DISCREPANCY: %s", msg)
                discrepancies.append(
                    ReconciliationDiscrepancy(
                        ticker=t,
                        internal_quantity=i_qty,
                        broker_quantity=b_qty,
                        discrepancy_delta=delta,
                        message=msg,
                    )
                )

        is_sync = len(discrepancies) == 0

        # Sync internal book to broker if discrepancy detected
        if not is_sync:
            logger.info("Synchronizing internal position records to match broker ledger.")
            self.internal_positions = dict(broker_book)

        report = ReconciliationReport(
            timestamp=datetime.now(),
            is_synchronized=is_sync,
            internal_positions=dict(internal_book),
            broker_positions=dict(broker_book),
            discrepancies=discrepancies,
            open_orders_count=len([o for o in self.submitted_orders.values() if not o.is_terminal]),
        )

        return report
