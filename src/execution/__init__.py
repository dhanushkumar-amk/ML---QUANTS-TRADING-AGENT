# ============================================================
# src.execution — Order Management & Broker Integration
# ============================================================
"""
Execution layer: order management, broker abstraction, Alpaca integration, and reconciliation.
"""

from src.execution.audit_trail import (
    AuditEvent,
    AuditEventType,
    AuditTrail,
    audit_trail_summary,
)
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
from src.execution.live_trading_loop import (
    LiveTradingConfig,
    LiveTradingLoop,
)
from src.execution.order_manager import (
    OrderManager,
    PartialFillPolicy,
    ReconciliationDiscrepancy,
    ReconciliationReport,
)

__all__ = [
    # Broker Abstraction & Client
    "BrokerClient",
    "AlpacaBrokerClient",
    "LiveTradingSafetyError",
    "BrokerAccount",
    "BrokerPosition",
    "BrokerOrder",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    # Order Manager & Reconciliation
    "OrderManager",
    "PartialFillPolicy",
    "ReconciliationReport",
    "ReconciliationDiscrepancy",
    # Audit Trail & Logging
    "AuditEventType",
    "AuditEvent",
    "AuditTrail",
    "audit_trail_summary",
    # Live Trading Orchestration
    "LiveTradingConfig",
    "LiveTradingLoop",
]
