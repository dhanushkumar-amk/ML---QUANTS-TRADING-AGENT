# ============================================================
# tests.test_live_trading_loop — End-to-End Orchestration Tests
# ============================================================
"""
Hermetic unit tests for LiveTradingLoop.
Tests end-to-end orchestration against fully mocked broker, data feed,
model inference, risk engine, and order manager.
"""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.execution.audit_trail import AuditEventType, AuditTrail
from src.execution.broker_client import (
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
    OrderSide,
    OrderStatus,
    OrderType,
)
from src.execution.live_trading_loop import LiveTradingConfig, LiveTradingLoop
from src.execution.order_manager import OrderManager


class TestLiveTradingLoop(unittest.TestCase):
    """Test suite for LiveTradingLoop orchestration, gating, and teardown."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.temp_dir.name)

        # Build mock broker
        self.mock_broker = MagicMock()
        self.mock_broker.health_check.return_value = True
        self.mock_account = BrokerAccount(
            account_id="test_acc_123",
            status="ACTIVE",
            currency="USD",
            cash=100000.0,
            portfolio_value=100000.0,
            buying_power=200000.0,
            equity=100000.0,
            is_paper=True,
        )
        self.mock_broker.get_account_info.return_value = self.mock_account
        self.mock_broker.get_positions.return_value = []

        # Build mock audit trail
        self.audit_trail = AuditTrail(log_dir=self.log_dir)

        # Build mock order manager
        self.order_manager = OrderManager(
            broker_client=self.mock_broker,
        )

        # Build config
        self.config = LiveTradingConfig(
            tickers=["AAPL", "MSFT"],
            cycle_interval_sec=1.0,
            market_hours_only=True,
            heartbeat_interval_sec=10.0,
            max_feed_stall_sec=30.0,
            audit_log_dir=str(self.log_dir),
        )

        # Build loop instance
        self.loop = LiveTradingLoop(
            config=self.config,
            broker_client=self.mock_broker,
            order_manager=self.order_manager,
            audit_trail=self.audit_trail,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_startup_reconciliation(self) -> None:
        """Verify startup queries broker and runs reconciliation before trading."""
        self.mock_broker.get_positions.return_value = [
            BrokerPosition(
                ticker="AAPL",
                quantity=10.0,
                side="long",
                market_value=1750.0,
                cost_basis=1700.0,
                current_price=175.0,
                unrealized_pnl=50.0,
                unrealized_pnl_pct=0.0294,
            )
        ]
        self.loop.startup()

        self.mock_broker.health_check.assert_called_once()
        self.mock_broker.get_account_info.assert_called()

        # Confirm reconciliation was recorded in audit trail
        events = self.audit_trail.read_events()
        recon_events = [
            e for e in events if e.get("event_type") == AuditEventType.RECONCILIATION.value
        ]
        self.assertEqual(len(recon_events), 1)
        self.assertIn("AAPL", recon_events[0]["details"]["broker_positions"])

    @patch("src.execution.live_trading_loop.is_market_open")
    def test_market_hours_gating(self, mock_is_market_open: MagicMock) -> None:
        """Verify loop idles outside regular market hours when configured."""
        mock_is_market_open.return_value = False

        # Run loop for 1 iteration with market_hours_only=True
        # It should check market hours and not invoke execute_cycle
        with patch.object(self.loop, "execute_cycle") as mock_exec:
            # We run with max_cycles=1, but because market is closed, it idles
            # Let's test with run() limited by duration_sec=0.1
            self.loop.run(duration_sec=0.1)
            mock_exec.assert_not_called()

    def test_end_to_end_cycle_execution(self) -> None:
        """Verify full pipeline execution: bar -> signal -> risk gate -> order submission."""
        # Mock DAL to return synthetic OHLCV bars
        dates = pd.date_range("2026-01-01", periods=100, freq="D")
        mock_df = pd.DataFrame(
            {
                "date": dates,
                "open": 150.0,
                "high": 155.0,
                "low": 149.0,
                "close": 152.0,
                "volume": 1000000,
            }
        )
        self.loop._dal = MagicMock()
        self.loop._dal.get_ohlcv.return_value = mock_df

        # Strategy allocates 20% to AAPL
        self.loop.signal_generator = MagicMock(return_value={"AAPL": 0.20, "MSFT": 0.0})

        # Mock broker order placement via order manager
        mock_placed_order = BrokerOrder(
            order_id="ord_live_test_1",
            client_order_id="cli_123",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=131.0,
            filled_quantity=131.0,
            filled_avg_price=152.0,
            status=OrderStatus.FILLED,
            created_at=datetime.now(timezone.utc),
        )
        self.mock_broker.place_order.return_value = mock_placed_order

        # Execute single cycle
        results = self.loop.execute_cycle(cycle_id="cycle_test_42")

        self.assertEqual(results["cycle_id"], "cycle_test_42")
        self.assertIn("ord_live_test_1", results["orders_placed"])

        # Check audit trail captured SIGNAL_GENERATED and ORDER_SUBMITTED
        events = self.audit_trail.read_events()
        signal_events = [
            e for e in events if e.get("event_type") == AuditEventType.SIGNAL_GENERATED.value
        ]
        self.assertTrue(len(signal_events) >= 1)
        self.assertEqual(signal_events[0]["ticker"], "AAPL")

        order_events = [
            e for e in events if e.get("event_type") == AuditEventType.ORDER_SUBMITTED.value
        ]
        self.assertTrue(len(order_events) >= 1)
        self.assertEqual(order_events[0]["ticker"], "AAPL")

    def test_graceful_shutdown_cancels_open_orders(self) -> None:
        """Verify graceful shutdown cancels any unexecuted / active orders."""
        mock_open_order = BrokerOrder(
            order_id="open_ord_999",
            client_order_id="cli_999",
            ticker="MSFT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=50.0,
            filled_quantity=0.0,
            filled_avg_price=None,
            status=OrderStatus.ACCEPTED,
            created_at=datetime.now(timezone.utc),
            limit_price=400.0,
        )
        self.order_manager.submitted_orders["open_ord_999"] = mock_open_order

        # Trigger shutdown
        self.loop.shutdown(reason="TEST_SHUTDOWN")

        # Confirm cancel_order was invoked for the dangling order
        self.mock_broker.cancel_order.assert_called_with("open_ord_999")

        # Confirm shutdown and order cancellation events logged
        events = self.audit_trail.read_events()
        cancel_events = [
            e for e in events if e.get("event_type") == AuditEventType.ORDER_CANCELLED.value
        ]
        self.assertEqual(len(cancel_events), 1)
        self.assertEqual(cancel_events[0]["details"]["order_id"], "open_ord_999")

    def test_heartbeat_and_stall_detection(self) -> None:
        """Verify heartbeat records health pulse and detects feed stalls during market hours."""
        self.loop._last_heartbeat_time = 0.0
        self.loop._last_bar_time = (
            datetime.now(timezone.utc).timestamp() - 500.0
        )  # 500s ago (> 30s threshold)

        with patch("src.execution.live_trading_loop.market_status", return_value={"is_open": True}):
            self.loop.check_heartbeat()

        events = self.audit_trail.read_events()
        heartbeats = [e for e in events if e.get("event_type") == AuditEventType.HEARTBEAT.value]
        self.assertEqual(len(heartbeats), 1)

        stall_alerts = [
            e for e in events if e.get("event_type") == AuditEventType.FEED_STALL_ALERT.value
        ]
        self.assertEqual(len(stall_alerts), 1)
        self.assertGreater(stall_alerts[0]["details"]["stall_duration_sec"], 30.0)


if __name__ == "__main__":
    unittest.main()
