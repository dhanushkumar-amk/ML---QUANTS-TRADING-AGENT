# ============================================================
# Unit Tests — Order Execution Manager & Reconciliation (Phase 45)
# ============================================================
"""
Tests for OrderManager:
- Translation of OrderIntent via RiskEngine
- Idempotent order submission via deterministic client_order_id
- Partial fill handling under various policies
- Position ledger reconciliation
"""

from __future__ import annotations

from src.execution.broker_client import (
    BrokerAccount,
    BrokerClient,
    BrokerOrder,
    BrokerPosition,
    OrderSide,
    OrderStatus,
    OrderType,
)
from src.execution.order_manager import (
    OrderManager,
    PartialFillPolicy,
)
from src.portfolio.risk_engine import (
    OrderAction,
    OrderIntent,
    PortfolioState,
    RiskConfig,
    RiskEngine,
)


class MockBroker(BrokerClient):
    """In-memory mock broker for testing OrderManager."""

    def __init__(self) -> None:
        self.orders: dict[str, BrokerOrder] = {}
        self.positions: list[BrokerPosition] = []
        self.account = BrokerAccount(
            account_id="test_acc",
            status="ACTIVE",
            currency="USD",
            cash=100000.0,
            portfolio_value=100000.0,
            buying_power=400000.0,
            equity=100000.0,
            is_paper=True,
        )

    def get_account_info(self) -> BrokerAccount:
        return self.account

    def get_positions(self) -> list[BrokerPosition]:
        return self.positions

    def place_order(
        self,
        ticker: str,
        quantity: float,
        side: OrderSide | str,
        order_type: OrderType | str = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        client_order_id: str | None = None,
        time_in_force: str = "day",
    ) -> BrokerOrder:
        ord_id = f"ord_{len(self.orders) + 1}"
        cid = client_order_id or f"cid_{len(self.orders) + 1}"
        side_str = side.value if hasattr(side, "value") else str(side)
        if "." in side_str:
            side_str = side_str.split(".")[-1]
        norm_side = OrderSide(side_str.lower())

        type_str = order_type.value if hasattr(order_type, "value") else str(order_type)
        if "." in type_str:
            type_str = type_str.split(".")[-1]
        norm_type = OrderType(type_str.lower())

        order = BrokerOrder(
            order_id=ord_id,
            client_order_id=cid,
            ticker=ticker,
            side=norm_side,
            order_type=norm_type,
            quantity=quantity,
            filled_quantity=quantity if norm_type == OrderType.MARKET else 0.0,
            filled_avg_price=150.0 if norm_type == OrderType.MARKET else None,
            status=OrderStatus.FILLED if norm_type == OrderType.MARKET else OrderStatus.ACCEPTED,
            created_at="2026-09-16T10:00:00Z",
            limit_price=limit_price,
        )
        self.orders[ord_id] = order
        return order

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            o = self.orders[order_id]
            self.orders[order_id] = BrokerOrder(
                order_id=o.order_id,
                client_order_id=o.client_order_id,
                ticker=o.ticker,
                side=o.side,
                order_type=o.order_type,
                quantity=o.quantity,
                filled_quantity=o.filled_quantity,
                filled_avg_price=o.filled_avg_price,
                status=OrderStatus.CANCELLED,
                created_at=o.created_at,
            )
            return True
        return False

    def get_order_status(self, order_id: str) -> BrokerOrder:
        return self.orders[order_id]

    def health_check(self) -> bool:
        return True


def test_order_translation_and_submission():
    """Verify OrderIntent is correctly translated and sent to broker."""
    broker = MockBroker()
    manager = OrderManager(broker_client=broker)

    intent = OrderIntent(
        ticker="AAPL",
        action=OrderAction.BUY,
        quantity=50.0,
        price=150.0,
    )

    order = manager.submit_intent(intent, order_type=OrderType.MARKET)
    assert order is not None
    assert order.ticker == "AAPL"
    assert order.quantity == 50.0
    assert order.status == OrderStatus.FILLED
    assert manager.internal_positions["AAPL"] == 50.0


def test_risk_engine_blocks_order():
    """Verify that if Phase 39 RiskEngine blocks an order, it never reaches the broker."""
    broker = MockBroker()
    # Configure 5% max drawdown and simulate 10% drawdown state
    risk_cfg = RiskConfig(max_drawdown_pct=0.05)
    risk_engine = RiskEngine(config=risk_cfg)
    manager = OrderManager(broker_client=broker, risk_engine=risk_engine)

    state = PortfolioState(
        current_equity=80000.0,
        peak_equity=100000.0,  # 20% drawdown > 5% limit
        daily_start_equity=80000.0,
        current_date="2026-09-16",
        positions={},
    )

    intent = OrderIntent(
        ticker="MSFT",
        action=OrderAction.BUY,
        quantity=20.0,
        price=300.0,
        is_closing=False,
    )

    result = manager.submit_intent(intent, portfolio_state=state)
    assert result is None  # Blocked by RiskEngine
    assert len(broker.orders) == 0  # Zero orders submitted to broker


def test_idempotency_safeguard():
    """Verify submitting duplicate intent returns original order and does not re-submit."""
    broker = MockBroker()
    manager = OrderManager(broker_client=broker)

    intent = OrderIntent(
        ticker="SPY",
        action=OrderAction.BUY,
        quantity=10.0,
        price=400.0,
    )

    # First submission
    order1 = manager.submit_intent(intent, cycle_id="20260916_rebalance_1")
    assert order1 is not None
    assert len(broker.orders) == 1

    # Second submission with identical cycle tag
    order2 = manager.submit_intent(intent, cycle_id="20260916_rebalance_1")
    assert order2 is not None
    assert order1.client_order_id == order2.client_order_id
    assert order1.order_id == order2.order_id
    assert len(broker.orders) == 1  # Broker called only once!


def test_partial_fill_handling_accept_partial():
    """Verify handle_partial_fill under ACCEPT_PARTIAL cancels remaining tail."""
    broker = MockBroker()
    manager = OrderManager(broker_client=broker)

    # Simulate partially filled order at broker
    order = BrokerOrder(
        order_id="part_ord_1",
        client_order_id="c_part",
        ticker="NVDA",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100.0,
        filled_quantity=40.0,
        filled_avg_price=120.0,
        status=OrderStatus.PARTIALLY_FILLED,
        created_at="2026-09-16T10:00:00Z",
    )
    broker.orders["part_ord_1"] = order

    res = manager.handle_partial_fill(
        order_id="part_ord_1", policy=PartialFillPolicy.ACCEPT_PARTIAL
    )
    # Order remainder should be cancelled
    assert res.status == OrderStatus.CANCELLED
    assert res.filled_quantity == 40.0


def test_reconciliation_detects_mismatches():
    """Verify reconcile() detects quantity delta, ghost positions, and missing positions."""
    broker = MockBroker()
    # Broker holds: AAPL=50, MSFT=20
    broker.positions = [
        BrokerPosition(
            ticker="AAPL",
            quantity=50.0,
            side="long",
            market_value=7500.0,
            cost_basis=7000.0,
            current_price=150.0,
            unrealized_pnl=500.0,
            unrealized_pnl_pct=0.07,
        ),
        BrokerPosition(
            ticker="MSFT",
            quantity=20.0,
            side="long",
            market_value=6000.0,
            cost_basis=6000.0,
            current_price=300.0,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
        ),
    ]

    manager = OrderManager(broker_client=broker)
    # Internal book records: AAPL=40 (quantity delta), GOOGL=10 (missing on broker), MSFT=0 (ghost on broker)
    internal_book = {"AAPL": 40.0, "GOOGL": 10.0}

    report = manager.reconcile(internal_positions=internal_book)

    assert not report.is_synchronized
    assert len(report.discrepancies) == 3

    tickers_flagged = {d.ticker for d in report.discrepancies}
    assert tickers_flagged == {"AAPL", "GOOGL", "MSFT"}

    # Internal book synchronized to broker post-reconciliation
    assert manager.internal_positions["AAPL"] == 50.0
    assert manager.internal_positions["MSFT"] == 20.0
