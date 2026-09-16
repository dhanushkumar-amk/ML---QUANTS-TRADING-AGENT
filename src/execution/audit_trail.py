# ============================================================
# Centralized Audit Trail & Structured Event Logger (Phase 47)
# ============================================================
"""
Production-grade, centralized, structured audit trail for live quantitative trading.

=============================================================================
ARCHITECTURAL PRINCIPLE: FORENSIC AUDITABILITY & TRACEABILITY
-----------------------------------------------------------------------------
In institutional quantitative trading, every single execution must be 100%
reconstructible after the fact. When an unexpected loss, margin call, or risk halt
occurs, risk officers and regulators ask:
  "Why did the trading system enter or exit position X at timestamp T?"

To answer this comprehensively, this module formalizes an immutable JSON Lines
audit trail capturing:
1. Signal Generation:
   Timestamp, ticker, signal direction, model confidence, raw features, and
   primary feature drivers (SHAP / interpretability values).
2. Risk Engine Decision:
   Verdict (APPROVED, RESIZED, BLOCKED), specific rule evaluated (drawdown halt,
   daily loss, Kelly cap, GARCH volatility de-risking), original vs approved size.
3. Order Lifecycle:
   Client order ID, broker order ID, order type, limit price, submission status,
   fill details, and cancellation reasons.
4. State Reconciliation:
   Discrepancy flags between internal accounting books and broker ledgers.
5. System Events & Heartbeats:
   Loop startup, graceful shutdown, health pulses, and feed stall alerts.
=============================================================================
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Audit Event Types & Data Structures
# ============================================================


class AuditEventType(str, Enum):
    """Categorical classification of logged trading events."""

    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    RISK_DECISION = "RISK_DECISION"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    RECONCILIATION = "RECONCILIATION"
    HEARTBEAT = "HEARTBEAT"
    FEED_STALL_ALERT = "FEED_STALL_ALERT"
    SYSTEM_EVENT = "SYSTEM_EVENT"


@dataclass
class AuditEvent:
    """Individual structured audit record."""

    timestamp: str  # ISO-8601 UTC
    event_type: AuditEventType | str
    cycle_id: str
    ticker: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize audit event to a single JSON string line."""
        d = {
            "timestamp": self.timestamp,
            "event_type": str(
                self.event_type.value
                if isinstance(self.event_type, AuditEventType)
                else self.event_type
            ),
            "cycle_id": self.cycle_id,
            "ticker": self.ticker,
            "details": self.details,
            "metadata": self.metadata,
        }
        return json.dumps(d, default=str)


# ============================================================
# 2. Centralized Audit Trail Engine
# ============================================================


class AuditTrail:
    """Thread-safe, durable audit logger persisting structured events to daily rotated JSONL."""

    def __init__(self, log_dir: str | Path = "logs/audit") -> None:
        """
        Parameters
        ----------
        log_dir : str or Path, default 'logs/audit'
            Directory path to persist daily audit files.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._events_logged = 0

    def _get_current_log_path(self, target_date: date | None = None) -> Path:
        """Generate log filepath with daily rotation (YYYY-MM-DD_audit.jsonl)."""
        dt = target_date or datetime.now(timezone.utc).date()
        filename = f"{dt.strftime('%Y-%m-%d')}_audit.jsonl"
        return self.log_dir / filename

    def log_event(
        self,
        event_type: AuditEventType | str,
        details: dict[str, Any],
        ticker: str | None = None,
        cycle_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> AuditEvent:
        """Record and immediately persist a structured audit event to disk.

        Parameters
        ----------
        event_type : AuditEventType or str
            Categorical event type.
        details : dict
            Event payload and parameters.
        ticker : str, optional
            Associated asset ticker symbol.
        cycle_id : str, optional
            Rebalance or execution loop cycle identifier.
        metadata : dict, optional
            Additional contextual telemetry.
        timestamp : datetime, optional
            Explicit event timestamp. Defaults to current UTC time.

        Returns
        -------
        AuditEvent
            The recorded audit event instance.
        """
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        cid = cycle_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        event = AuditEvent(
            timestamp=ts,
            event_type=event_type,
            cycle_id=cid,
            ticker=ticker.upper() if ticker else None,
            details=details,
            metadata=metadata or {},
        )

        json_line = event.to_json() + "\n"
        log_path = self._get_current_log_path()

        with self._lock:
            # Open in append mode and flush immediately to ensure durability across crashes
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json_line)
                f.flush()
            self._events_logged += 1

        logger.debug(
            "Audit event logged: [%s] %s %s",
            event.event_type,
            event.ticker or "ALL",
            event.cycle_id,
        )
        return event

    def read_events(
        self,
        target_date: str | date | datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Read all parsed JSON audit events for a given calendar date."""
        if target_date is None:
            d = datetime.now(timezone.utc).date()
        elif isinstance(target_date, str):
            d = datetime.strptime(target_date, "%Y-%m-%d").date()
        elif isinstance(target_date, datetime):
            d = target_date.date()
        else:
            d = target_date

        log_path = self._get_current_log_path(d)
        if not log_path.exists():
            return []

        events = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                clean_line = line.strip()
                if clean_line:
                    try:
                        events.append(json.loads(clean_line))
                    except json.JSONDecodeError:
                        continue
        return events


# ============================================================
# 3. Audit Trail Forensic Query & Summary Utility
# ============================================================


def audit_trail_summary(
    target_date: str | date | datetime | None = None,
    log_dir: str | Path = "logs/audit",
) -> dict[str, Any]:
    """Analyze the audit trail for a trading session and compile a human-readable summary.

    Parameters
    ----------
    target_date : str, date, or datetime, optional
        Date to analyze (YYYY-MM-DD). Defaults to today.
    log_dir : str or Path, default 'logs/audit'
        Path to audit directory.

    Returns
    -------
    dict
        Structured summary containing metric counts and narrative report string.
    """
    audit = AuditTrail(log_dir=log_dir)
    events = audit.read_events(target_date)

    summary: dict[str, Any] = {
        "date": str(target_date or datetime.now(timezone.utc).date()),
        "total_events": len(events),
        "signals": {
            "total": 0,
            "by_ticker": {},
            "by_direction": {"BUY": 0, "SELL": 0, "HOLD": 0},
        },
        "risk_decisions": {
            "total": 0,
            "approved": 0,
            "resized": 0,
            "blocked": 0,
            "rules_triggered": {},
        },
        "orders": {
            "submitted": 0,
            "filled": 0,
            "cancelled": 0,
            "rejected": 0,
        },
        "reconciliation": {
            "checks_run": 0,
            "discrepancies_detected": 0,
        },
        "heartbeats": 0,
        "feed_stall_alerts": 0,
        "system_events": 0,
    }

    for ev in events:
        etype = ev.get("event_type")
        details = ev.get("details", {})
        ticker = ev.get("ticker")

        if etype == AuditEventType.SIGNAL_GENERATED.value:
            summary["signals"]["total"] += 1
            if ticker:
                summary["signals"]["by_ticker"][ticker] = (
                    summary["signals"]["by_ticker"].get(ticker, 0) + 1
                )
            direction = str(details.get("direction", "HOLD")).upper()
            if direction in summary["signals"]["by_direction"]:
                summary["signals"]["by_direction"][direction] += 1

        elif etype == AuditEventType.RISK_DECISION.value:
            summary["risk_decisions"]["total"] += 1
            status = str(details.get("status", "")).upper()
            if status == "APPROVED":
                summary["risk_decisions"]["approved"] += 1
            elif status == "RESIZED":
                summary["risk_decisions"]["resized"] += 1
            elif status == "BLOCKED":
                summary["risk_decisions"]["blocked"] += 1

            rule = details.get("rule_triggered", "NONE")
            if rule != "NONE":
                summary["risk_decisions"]["rules_triggered"][rule] = (
                    summary["risk_decisions"]["rules_triggered"].get(rule, 0) + 1
                )

        elif etype == AuditEventType.ORDER_SUBMITTED.value:
            summary["orders"]["submitted"] += 1

        elif etype == AuditEventType.ORDER_FILLED.value:
            summary["orders"]["filled"] += 1

        elif etype == AuditEventType.ORDER_CANCELLED.value:
            summary["orders"]["cancelled"] += 1

        elif etype == AuditEventType.ORDER_REJECTED.value:
            summary["orders"]["rejected"] += 1

        elif etype == AuditEventType.RECONCILIATION.value:
            summary["reconciliation"]["checks_run"] += 1
            discrepancies = details.get("discrepancies", [])
            if discrepancies:
                summary["reconciliation"]["discrepancies_detected"] += len(discrepancies)

        elif etype == AuditEventType.HEARTBEAT.value:
            summary["heartbeats"] += 1

        elif etype == AuditEventType.FEED_STALL_ALERT.value:
            summary["feed_stall_alerts"] += 1

        elif etype == AuditEventType.SYSTEM_EVENT.value:
            summary["system_events"] += 1

    # Compile human-readable explanation report
    report_lines = [
        "=" * 64,
        f"DAILY AUDIT TRAIL SUMMARY REPORT: {summary['date']}",
        "=" * 64,
        f"Total Logged Events: {summary['total_events']}",
        f"System Heartbeats:   {summary['heartbeats']} | Feed Stall Alerts: {summary['feed_stall_alerts']}",
        "",
        "1. SIGNALS GENERATED:",
        f"   Total Signals:     {summary['signals']['total']}",
        f"   By Direction:      BUY={summary['signals']['by_direction']['BUY']}, SELL={summary['signals']['by_direction']['SELL']}, HOLD={summary['signals']['by_direction']['HOLD']}",
        f"   By Asset:          {json.dumps(summary['signals']['by_ticker'])}",
        "",
        "2. RISK ENGINE DECISIONS (Safety Layer):",
        f"   Total Evaluated:   {summary['risk_decisions']['total']}",
        f"   Approved:          {summary['risk_decisions']['approved']}",
        f"   Resized (Haircut): {summary['risk_decisions']['resized']}",
        f"   Blocked (Halted):  {summary['risk_decisions']['blocked']}",
        f"   Rules Triggered:   {json.dumps(summary['risk_decisions']['rules_triggered'])}",
        "",
        "3. BROKER ORDERS & EXECUTION:",
        f"   Submitted:         {summary['orders']['submitted']}",
        f"   Filled:            {summary['orders']['filled']}",
        f"   Cancelled:         {summary['orders']['cancelled']}",
        f"   Rejected:          {summary['orders']['rejected']}",
        "",
        "4. LEDGER RECONCILIATION:",
        f"   Audits Executed:   {summary['reconciliation']['checks_run']}",
        f"   Discrepancies:     {summary['reconciliation']['discrepancies_detected']}",
        "=" * 64,
    ]
    summary["report_text"] = "\n".join(report_lines)
    return summary
