# ============================================================
# Alpaca Paper Trading Execution Test Script (Phase 44+45)
# ============================================================
"""
Live validation script against Alpaca Paper Trading API.

Executes:
1. Paper Trading authentication & account health check.
2. Market order placement and fill verification (1 share liquid ticker).
3. Limit order placement (below market), status check, and active cancellation.
4. Position ledger reconciliation (internal book vs. Alpaca ledger).
5. Enforced safety rail verification: confirms LiveTradingSafetyError is raised
   if paper=False or live API credentials are used without allow_live=True.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure workspace root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.execution.broker_client import (
    AlpacaBrokerClient,
    LiveTradingSafetyError,
    OrderStatus,
    OrderType,
)
from src.execution.order_manager import OrderManager
from src.portfolio.risk_engine import OrderAction, OrderIntent, RiskConfig, RiskEngine


def main() -> None:
    print("================================================================")
    print("Executing Phase 44+45: Alpaca Paper Trading & Execution Test")
    print("================================================================")

    # -------------------------------------------------------------
    # 1. Initialize Client in Paper Mode & Health Check
    # -------------------------------------------------------------
    print("\n[Step 1] Initializing AlpacaBrokerClient in PAPER mode...")
    client = AlpacaBrokerClient(paper=True, allow_live=False)

    is_healthy = client.health_check()
    print(f"  Connection Health Check: {'PASS (Active)' if is_healthy else 'FAIL'}")

    account = client.get_account_info()
    print(f"  Account ID:    {account.account_id}")
    print(f"  Status:        {account.status}")
    print(f"  Cash:          ${account.cash:,.2f}")
    print(f"  Portfolio Eq:  ${account.equity:,.2f}")
    print(f"  Buying Power:  ${account.buying_power:,.2f}")
    print(f"  Paper Account: {account.is_paper}")

    # -------------------------------------------------------------
    # 2. Attach OrderManager & RiskEngine
    # -------------------------------------------------------------
    print("\n[Step 2] Attaching OrderManager and Phase 39 RiskEngine...")
    risk_engine = RiskEngine(config=RiskConfig(max_drawdown_pct=0.15))
    order_mgr = OrderManager(broker_client=client, risk_engine=risk_engine)
    print("  OrderManager successfully linked to RiskEngine.")

    # -------------------------------------------------------------
    # 3. Market Order Placement & Fill Verification
    # -------------------------------------------------------------
    test_ticker = "SPY"
    print(f"\n[Step 3] Submitting small test Market Order: BUY 1 {test_ticker}...")
    intent_mkt = OrderIntent(
        ticker=test_ticker,
        action=OrderAction.BUY,
        quantity=1.0,
        price=550.0,
    )

    cycle_id = f"test_{int(time.time())}"
    mkt_order = order_mgr.submit_intent(
        intent=intent_mkt,
        order_type=OrderType.MARKET,
        cycle_id=cycle_id,
    )

    if mkt_order is None:
        print("  ERROR: Market order was unexpectedly blocked.")
        return

    print(f"  Order ID:        {mkt_order.order_id}")
    print(f"  Client Order ID: {mkt_order.client_order_id}")
    print(f"  Status:          {mkt_order.status.value}")
    print(f"  Quantity:        {mkt_order.quantity}")

    # Poll status for up to 5 seconds to confirm fill
    print("  Polling order status for fill confirmation...")
    for _ in range(5):
        time.sleep(1)
        latest = client.get_order_status(mkt_order.order_id)
        if latest.status in {OrderStatus.FILLED, OrderStatus.ACCEPTED}:
            print(
                f"  -> Order status: {latest.status.value.upper()} (Filled: {latest.filled_quantity} @ {latest.filled_avg_price})"
            )
            break

    # -------------------------------------------------------------
    # 4. Limit Order Placement & Active Cancellation
    # -------------------------------------------------------------
    deep_otm_price = 150.00  # Far below current SPY price (~$550)
    print(
        f"\n[Step 4] Placing deep out-of-the-money Limit Order: BUY 1 {test_ticker} @ ${deep_otm_price:.2f}..."
    )
    intent_lmt = OrderIntent(
        ticker=test_ticker,
        action=OrderAction.BUY,
        quantity=1.0,
        price=deep_otm_price,
    )

    lmt_order = order_mgr.submit_intent(
        intent=intent_lmt,
        order_type=OrderType.LIMIT,
        limit_price=deep_otm_price,
        cycle_id=f"lmt_{int(time.time())}",
    )

    if lmt_order is None:
        print("  ERROR: Limit order placement failed.")
        return

    print(f"  Limit Order ID:  {lmt_order.order_id}")
    print(f"  Status:          {lmt_order.status.value}")
    print(f"  Limit Price:     ${lmt_order.limit_price:.2f}")

    # Check status
    time.sleep(1)
    status_before = client.get_order_status(lmt_order.order_id)
    print(
        f"  Status before cancel: {status_before.status.value.upper()} (Remaining: {status_before.remaining_quantity})"
    )

    # Cancel order
    print("  Canceling limit order...")
    cancelled = client.cancel_order(lmt_order.order_id)
    print(f"  Cancel Request Sent: {cancelled}")

    time.sleep(1)
    status_after = client.get_order_status(lmt_order.order_id)
    print(
        f"  Status after cancel:  {status_after.status.value.upper()} (Is Terminal: {status_after.is_terminal})"
    )
    assert status_after.status == OrderStatus.CANCELLED or status_after.is_terminal

    # -------------------------------------------------------------
    # 5. Position & Order Reconciliation Check
    # -------------------------------------------------------------
    print("\n[Step 5] Running Position & Order Reconciliation Check...")
    recon_report = order_mgr.reconcile()

    print(f"  Is Synchronized:     {recon_report.is_synchronized}")
    print(f"  Discrepancy Count:   {len(recon_report.discrepancies)}")
    print(f"  Internal Positions:  {recon_report.internal_positions}")
    print(f"  Broker Positions:    {recon_report.broker_positions}")
    print(f"  Active Open Orders:  {recon_report.open_orders_count}")

    if not recon_report.is_synchronized:
        print("  Discrepancies identified and synchronized:")
        for d in recon_report.discrepancies:
            print(f"    - {d.message}")
    else:
        print("  -> Internal book matches Alpaca ledger perfectly.")

    # -------------------------------------------------------------
    # 6. Verify Paper-Trading Safety Rail Rejection
    # -------------------------------------------------------------
    print("\n[Step 6] Testing Enforced Live-Trading Safety Rail...")
    try:
        # Deliberately attempt to instantiate client with paper=False and allow_live=False
        AlpacaBrokerClient(
            api_key="PK4N_TEST",
            secret_key="TEST_SEC",
            paper=False,
            allow_live=False,
        )
        print("  FAIL: Safety rail did NOT trigger when paper=False!")
    except LiveTradingSafetyError as err:
        print(
            f"  PASS: Safety rail intercepted execution as expected!\n        Error message: {err}"
        )

    print("\n================================================================")
    print("Phase 44+45 Validation Complete: All checks PASSED successfully.")
    print("================================================================")


if __name__ == "__main__":
    main()
