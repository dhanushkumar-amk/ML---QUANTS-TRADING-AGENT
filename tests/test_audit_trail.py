# ============================================================
# tests.test_audit_trail — Comprehensive Audit Trail Unit Tests
# ============================================================
"""
Unit tests for AuditTrail, AuditEventType, AuditEvent, and audit_trail_summary.
Runs hermetically without external network access or storage.
"""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.execution.audit_trail import (
    AuditEvent,
    AuditEventType,
    AuditTrail,
    audit_trail_summary,
)


class TestAuditTrail(unittest.TestCase):
    """Test suite for AuditTrail logging and query summary utilities."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.temp_dir.name)
        self.audit = AuditTrail(log_dir=self.log_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_log_and_jsonl_persistence(self) -> None:
        """Test that events are persisted accurately in daily JSONL format."""
        event = self.audit.log_event(
            event_type=AuditEventType.SIGNAL_GENERATED,
            ticker="AAPL",
            cycle_id="cycle_001",
            details={"direction": "BUY", "confidence": 0.85, "features": {"rsi": 32.5}},
        )

        self.assertIsInstance(event, AuditEvent)
        self.assertEqual(event.event_type, AuditEventType.SIGNAL_GENERATED)
        self.assertEqual(event.ticker, "AAPL")
        self.assertEqual(event.cycle_id, "cycle_001")

        # Verify file exists on disk
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        expected_file = self.log_dir / f"{today_str}_audit.jsonl"
        self.assertTrue(expected_file.exists())

        # Verify json contents
        with open(expected_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["event_type"], "SIGNAL_GENERATED")
        self.assertEqual(record["ticker"], "AAPL")
        self.assertEqual(record["cycle_id"], "cycle_001")
        self.assertEqual(record["details"]["confidence"], 0.85)

    def test_all_major_event_types_logging(self) -> None:
        """Test logging across all major event types with required fields."""
        event_types = [
            AuditEventType.SIGNAL_GENERATED,
            AuditEventType.RISK_DECISION,
            AuditEventType.ORDER_SUBMITTED,
            AuditEventType.ORDER_FILLED,
            AuditEventType.ORDER_CANCELLED,
            AuditEventType.ORDER_REJECTED,
            AuditEventType.RECONCILIATION,
            AuditEventType.HEARTBEAT,
            AuditEventType.FEED_STALL_ALERT,
            AuditEventType.SYSTEM_EVENT,
        ]

        for et in event_types:
            evt = self.audit.log_event(
                event_type=et,
                ticker="MSFT" if "ORDER" in et.value or "SIGNAL" in et.value else None,
                cycle_id="cycle_test",
                details={"metric": "test_value"},
            )
            self.assertEqual(evt.event_type, et)

        events = self.audit.read_events()
        self.assertEqual(len(events), len(event_types))

    def test_audit_trail_summary_with_synthetic_log(self) -> None:
        """Test audit_trail_summary aggregation against synthetic event stream."""
        # 1. Generate 3 signals
        self.audit.log_event(
            AuditEventType.SIGNAL_GENERATED,
            ticker="AAPL",
            cycle_id="c1",
            details={"direction": "BUY", "confidence": 0.92, "features": {"rsi": 28.0}},
        )
        self.audit.log_event(
            AuditEventType.SIGNAL_GENERATED,
            ticker="MSFT",
            cycle_id="c1",
            details={"direction": "SELL", "confidence": 0.74, "features": {"rsi": 75.0}},
        )
        self.audit.log_event(
            AuditEventType.SIGNAL_GENERATED,
            ticker="NVDA",
            cycle_id="c1",
            details={"direction": "HOLD", "confidence": 0.50, "features": {"rsi": 50.0}},
        )

        # 2. Risk decisions: 1 approved, 2 blocked
        self.audit.log_event(
            AuditEventType.RISK_DECISION,
            ticker="AAPL",
            cycle_id="c1",
            details={"status": "APPROVED", "rule_triggered": "NONE"},
        )
        self.audit.log_event(
            AuditEventType.RISK_DECISION,
            ticker="MSFT",
            cycle_id="c1",
            details={"status": "BLOCKED", "rule_triggered": "DailyDrawdownExceeded"},
        )
        self.audit.log_event(
            AuditEventType.RISK_DECISION,
            ticker="NVDA",
            cycle_id="c1",
            details={"status": "BLOCKED", "rule_triggered": "PositionCapExceeded"},
        )

        # 3. Order submissions & fills
        self.audit.log_event(
            AuditEventType.ORDER_SUBMITTED,
            ticker="AAPL",
            cycle_id="c1",
            details={"order_id": "ord_001", "side": "buy", "quantity": 10.0},
        )
        self.audit.log_event(
            AuditEventType.ORDER_FILLED,
            ticker="AAPL",
            cycle_id="c1",
            details={"order_id": "ord_001", "filled_qty": 10.0, "price": 175.50},
        )
        self.audit.log_event(
            AuditEventType.ORDER_REJECTED,
            ticker="MSFT",
            cycle_id="c1",
            details={"order_id": "ord_002", "reason": "Insufficient margin"},
        )

        # 4. Reconciliation with 1 discrepancy
        self.audit.log_event(
            AuditEventType.RECONCILIATION,
            cycle_id="c1",
            details={"discrepancies": [{"ticker": "TSLA", "diff": 5.0}]},
        )

        # 5. Heartbeat & Feed Stall
        self.audit.log_event(
            AuditEventType.HEARTBEAT,
            cycle_id="c1",
            details={"total_cycles": 5, "market_open": True},
        )
        self.audit.log_event(
            AuditEventType.FEED_STALL_ALERT,
            cycle_id="c1",
            details={"stall_duration_sec": 350.0},
        )

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        summary = audit_trail_summary(target_date=today_str, log_dir=self.log_dir)

        # Validate structured metrics
        self.assertEqual(summary["total_events"], 12)
        self.assertEqual(summary["signals"]["total"], 3)
        self.assertEqual(summary["signals"]["by_direction"]["BUY"], 1)
        self.assertEqual(summary["signals"]["by_direction"]["SELL"], 1)
        self.assertEqual(summary["signals"]["by_direction"]["HOLD"], 1)
        self.assertEqual(summary["risk_decisions"]["approved"], 1)
        self.assertEqual(summary["risk_decisions"]["blocked"], 2)
        self.assertEqual(summary["orders"]["submitted"], 1)
        self.assertEqual(summary["orders"]["filled"], 1)
        self.assertEqual(summary["orders"]["rejected"], 1)
        self.assertEqual(summary["reconciliation"]["checks_run"], 1)
        self.assertEqual(summary["reconciliation"]["discrepancies_detected"], 1)
        self.assertEqual(summary["heartbeats"], 1)
        self.assertEqual(summary["feed_stall_alerts"], 1)

        # Validate formatted text output
        report_text = summary["report_text"]
        self.assertIn("DAILY AUDIT TRAIL SUMMARY REPORT", report_text)
        self.assertIn("Total Signals:     3", report_text)
        self.assertIn("Approved:          1", report_text)
        self.assertIn("Blocked (Halted):  2", report_text)
        self.assertIn("Submitted:         1", report_text)
        self.assertIn("Filled:            1", report_text)

    def test_read_events_empty_when_no_log(self) -> None:
        """Test read_events returns empty list when no log exists for target date."""
        events = self.audit.read_events(target_date="2020-01-01")
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
