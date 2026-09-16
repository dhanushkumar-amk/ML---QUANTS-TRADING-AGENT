# ============================================================
# Unit Tests — Broker Client Abstraction & Safety Gate (Phase 44)
# ============================================================
"""
Tests for BrokerClient abstraction and AlpacaBrokerClient.
All external broker API interactions are 100% mocked to guarantee hermetic CI test runs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.execution.broker_client import (
    AlpacaBrokerClient,
    BrokerAccount,
    BrokerClient,
    BrokerOrder,
    BrokerPosition,
    LiveTradingSafetyError,
    OrderSide,
    OrderStatus,
    OrderType,
)


class DummyBroker(BrokerClient):
    """Concrete dummy broker for testing abstract interface contract."""

    def get_account_info(self) -> BrokerAccount:
        return BrokerAccount(
            account_id="dummy_acc",
            status="ACTIVE",
            currency="USD",
            cash=100000.0,
            portfolio_value=100000.0,
            buying_power=400000.0,
            equity=100000.0,
            is_paper=True,
        )

    def get_positions(self) -> list[BrokerPosition]:
        return []

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
        return BrokerOrder(
            order_id="dummy_ord_1",
            client_order_id=client_order_id or "c1",
            ticker=ticker,
            side=OrderSide.BUY if str(side).lower() == "buy" else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            filled_quantity=quantity,
            filled_avg_price=100.0,
            status=OrderStatus.FILLED,
            created_at="2026-09-16T10:00:00Z",
        )

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_order_status(self, order_id: str) -> BrokerOrder:
        return self.place_order("AAPL", 10.0, "buy")

    def health_check(self) -> bool:
        return True


def test_abstract_interface_contract():
    """Verify that BrokerClient enforces abstract methods on incomplete subclasses."""
    with pytest.raises(TypeError):
        # Cannot instantiate abstract class without implementing all abstract methods
        class IncompleteBroker(BrokerClient):
            pass

        IncompleteBroker()

    dummy = DummyBroker()
    assert dummy.health_check() is True
    acc = dummy.get_account_info()
    assert acc.cash == 100000.0
    assert acc.is_paper is True


def test_live_trading_safety_rail_endpoint_check():
    """Verify that paper=False without allow_live=True raises LiveTradingSafetyError."""
    with pytest.raises(LiveTradingSafetyError) as excinfo:
        AlpacaBrokerClient(
            api_key="PK_TEST_KEY",
            secret_key="SECRET",
            paper=False,
            allow_live=False,
        )
    assert "CRITICAL SAFETY VIOLATION" in str(excinfo.value)
    assert "paper=False" in str(excinfo.value)


def test_live_trading_safety_rail_api_key_check():
    """Verify that live API key prefix ('AK...') without allow_live=True raises LiveTradingSafetyError."""
    with pytest.raises(LiveTradingSafetyError) as excinfo:
        AlpacaBrokerClient(
            api_key="AK_LIVE_REAL_MONEY_KEY",
            secret_key="SECRET",
            paper=True,
            allow_live=False,
        )
    assert "CRITICAL SAFETY VIOLATION" in str(excinfo.value)
    assert "live Alpaca API key prefix" in str(excinfo.value)


def test_live_trading_safety_rail_override():
    """Verify that allow_live=True deliberately permits live configuration."""
    mock_raw = MagicMock()
    client = AlpacaBrokerClient(
        api_key="AK_LIVE_KEY",
        secret_key="SECRET",
        paper=False,
        allow_live=True,
        raw_client=mock_raw,
    )
    assert client.allow_live is True
    assert client.paper is False


def test_alpaca_get_account_info_mocked():
    """Verify get_account_info normalizes raw Alpaca Account response."""
    mock_raw = MagicMock()
    mock_acc = MagicMock()
    mock_acc.id = "acc_12345"
    mock_acc.status = "ACTIVE"
    mock_acc.currency = "USD"
    mock_acc.cash = "50000.50"
    mock_acc.portfolio_value = "75000.00"
    mock_acc.buying_power = "200000.00"
    mock_acc.equity = "75000.00"
    mock_raw.get_account.return_value = mock_acc

    client = AlpacaBrokerClient(
        api_key="PK_TEST",
        secret_key="SECRET",
        paper=True,
        raw_client=mock_raw,
    )
    account = client.get_account_info()

    assert account.account_id == "acc_12345"
    assert account.cash == 50000.50
    assert account.portfolio_value == 75000.00
    assert account.buying_power == 200000.00
    assert account.is_paper is True


def test_alpaca_get_positions_mocked():
    """Verify get_positions normalizes raw Alpaca Position list."""
    mock_raw = MagicMock()
    p1 = MagicMock()
    p1.symbol = "AAPL"
    p1.qty = "10"
    p1.side = "long"
    p1.market_value = "1500.0"
    p1.cost_basis = "1400.0"
    p1.current_price = "150.0"
    p1.unrealized_pl = "100.0"
    p1.unrealized_plpc = "0.0714"

    mock_raw.get_all_positions.return_value = [p1]

    client = AlpacaBrokerClient(
        api_key="PK_TEST",
        secret_key="SECRET",
        paper=True,
        raw_client=mock_raw,
    )
    positions = client.get_positions()

    assert len(positions) == 1
    pos = positions[0]
    assert pos.ticker == "AAPL"
    assert pos.quantity == 10.0
    assert pos.side == "long"
    assert pos.market_value == 1500.0
    assert pos.unrealized_pnl == 100.0


def test_alpaca_place_market_and_limit_order_mocked():
    """Verify place_order handles Market and Limit order requests."""
    mock_raw = MagicMock()
    mock_order = MagicMock()
    mock_order.id = "ord_abc_123"
    mock_order.client_order_id = "client_ord_1"
    mock_order.symbol = "MSFT"
    mock_order.side = "buy"
    mock_order.order_type = "market"
    mock_order.qty = "25"
    mock_order.filled_qty = "25"
    mock_order.filled_avg_price = "300.0"
    mock_order.status = "filled"
    mock_order.created_at = "2026-09-16T12:00:00Z"
    mock_order.limit_price = None
    mock_order.stop_price = None

    mock_raw.submit_order.return_value = mock_order

    client = AlpacaBrokerClient(
        api_key="PK_TEST",
        secret_key="SECRET",
        paper=True,
        raw_client=mock_raw,
    )

    # 1. Market Order
    res_mkt = client.place_order(
        ticker="MSFT",
        quantity=25.0,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        client_order_id="client_ord_1",
    )
    assert res_mkt.order_id == "ord_abc_123"
    assert res_mkt.status == OrderStatus.FILLED
    assert res_mkt.filled_quantity == 25.0

    # 2. Limit order missing limit_price must fail validation
    with pytest.raises(ValueError) as exc:
        client.place_order(ticker="MSFT", quantity=10, side="buy", order_type="limit")
    assert "limit_price must be positive" in str(exc.value)


def test_alpaca_cancel_and_status_mocked():
    """Verify cancel_order and get_order_status delegates properly to SDK."""
    mock_raw = MagicMock()
    mock_raw.cancel_order_by_id.return_value = None

    client = AlpacaBrokerClient(
        api_key="PK_TEST",
        secret_key="SECRET",
        paper=True,
        raw_client=mock_raw,
    )

    # Cancel
    assert client.cancel_order("ord_999") is True
    mock_raw.cancel_order_by_id.assert_called_once_with("ord_999")

    # Status
    mock_ord = MagicMock()
    mock_ord.id = "ord_999"
    mock_ord.client_order_id = "c_999"
    mock_ord.symbol = "SPY"
    mock_ord.side = "buy"
    mock_ord.order_type = "limit"
    mock_ord.qty = "5"
    mock_ord.filled_qty = "0"
    mock_ord.filled_avg_price = None
    mock_ord.status = "canceled"
    mock_ord.limit_price = "400.0"
    mock_raw.get_order_by_id.return_value = mock_ord

    status = client.get_order_status("ord_999")
    assert status.order_id == "ord_999"
    assert status.status == OrderStatus.CANCELLED
    assert status.remaining_quantity == 5.0
    assert status.is_terminal is True
