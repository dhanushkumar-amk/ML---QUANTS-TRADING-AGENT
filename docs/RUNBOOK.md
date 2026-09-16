# Live Paper Trading System Runbook (Phases 46 & 47)

> **Audience**: Quant Traders, Production Support Engineers, and System Operators.  
> **Target System**: End-to-end Automated ML/DL/NLP Trading Bot (Paper Trading Environment).  
> **Primary Interfaces**: Alpaca Paper Trading API (`paper-api.alpaca.markets`), Structured Audit Trail (`logs/audit/`).

---

## 1. System Overview & Architecture

The live trading loop (`src.execution.live_trading_loop`) is the core continuous orchestration engine that binds the system's modular layers together:

```
[Real-Time Market Data Feed (Phase 3)]
                  │
                  ▼
[Feature Engineering Pipeline (Phase 17/34)]
                  │
                  ▼
[Production ML/DL Model Inference (Phase 34)]
                  │
                  ▼
[Position Sizing (Phase 35/36)]
                  │
                  ▼
[Risk Engine Hard Gating (Phase 39)]
                  │
                  ▼
[Order Manager & Execution (Phase 44/45)]
                  │
                  ▼
[Alpaca Broker API (Paper Rail)]
                  │
                  ▼
[Structured Audit Trail (Phase 47 JSONL)]
```

### Core Operational Invariants
1. **Paper-Trading Guardrail**: The broker client rejects any API key or base URL configured for real-money execution unless explicit live flags are asserted.
2. **Startup Reconciliation**: On startup, the bot queries the broker for all open orders and open positions, synchronizing internal state before generating signals.
3. **Market Hours Gating**: Cycles only evaluate signals during regular equity market hours (09:30 - 16:00 ET Monday–Friday, excluding NYSE holidays). Outside market hours, the engine enters low-power idle sleep.
4. **Graceful Shutdown**: On `SIGINT` or `SIGTERM`, all pending orders are cancelled, open positions remain un-corrupted, and audit trail buffers are flushed cleanly to disk.

---

## 2. Deployment & Execution Options

### Option A: Systemd Service (Linux Server / Cloud VM)
Recommended for dedicated VPS instances (e.g. AWS EC2, DigitalOcean Droplet, Hetzner) where native OS process supervision, journald logging, and zero container overhead are desired.

#### Start the Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
```

#### Check Status & Heartbeat
```bash
sudo systemctl status trading-bot
# Real-time tailing of stdout/stderr and audit logs via journald
journalctl -u trading-bot -f -o cat
```

#### Stop the Service
```bash
# Sends SIGTERM, triggering graceful order cancellation and audit flush (30s timeout)
sudo systemctl stop trading-bot
```

---

### Option B: Docker & Docker Compose (Recommended)
Recommended for cross-platform portability, reproducible environment parity (Python 3.11 with exact dependencies), resource isolation, and local workstation operation.

#### Start the Container
```bash
# Build and start detached
docker compose up --build -d
```

#### Monitor Container & Logs
```bash
docker compose ps
docker compose logs -f trading-bot
```

#### Stop the Container
```bash
# Sends SIGTERM with 30s grace period for clean teardown
docker compose stop -t 30
```

---

### Option C: Standalone Interactive CLI (Development / Testing)
```bash
# Run one or more symbols with 60s bar cadence
python -m src.execution.live_trading_loop --symbols AAPL MSFT NVDA GOOGL --bar-interval 60

# Run in debug/after-hours testing mode (bypasses market hours gating)
python -m src.execution.live_trading_loop --symbols AAPL MSFT --bar-interval 5 --force-run
```

---

## 3. Monitoring & Heartbeat Mechanism

### Heartbeat Operation
- **Cadence**: Emits a `HEARTBEAT` audit event and info log every **60 seconds** (configurable via `heartbeat_interval_sec`).
- **Metrics Logged**:
  - Broker Account Equity & Buying Power.
  - Number of Active Positions and Open Orders.
  - Lifetime Cycle Count and Errors Encountered.
  - Market State (Open / Closed / Pre-market / After-hours).
  - Time Elapsed Since Last Market Bar (`seconds_since_last_bar`).

### Stall Detection Alert
If the market is **open** and no new bar or cycle completion has occurred for longer than `stall_alert_threshold_sec` (default: 300 seconds / 5 minutes), the heartbeat monitors raises a **CRITICAL STALL WARNING**:
```json
{
  "event_type": "HEARTBEAT",
  "details": {
    "stall_alert": true,
    "seconds_since_last_bar": 342.1,
    "threshold": 300.0,
    "status": "DATA_FEED_OR_PIPELINE_STALLED"
  }
}
```

---

## 4. Triage & Troubleshooting Procedures

### Scenario 1: Heartbeat Alerts `DATA_FEED_OR_PIPELINE_STALLED`
- **Root Cause**: Websocket disconnected, Alpaca data rate limit hit, or feature computation thread hung.
- **Triage Steps**:
  1. Check network connectivity: `curl -I https://paper-api.alpaca.markets/v2/clock`
  2. Inspect the latest log entries in `logs/audit/YYYY-MM-DD_audit.jsonl` for unhandled exceptions or timeout messages.
  3. Verify whether the market is genuinely trading (check for halt / LULD suspension on the target ticker).
  4. If feed is unresponsive, restart the service: `docker compose restart trading-bot` or `sudo systemctl restart trading-bot`.

### Scenario 2: Startup Reconciliation Discrepancy Alert
- **Root Cause**: Manual trade placed via Alpaca mobile/web dashboard, external bot instance running concurrently, or unrecorded liquidation.
- **Triage Steps**:
  1. Inspect the `RECONCILIATION` event in the audit trail:
     ```bash
     grep '"event_type": "RECONCILIATION"' logs/audit/*_audit.jsonl
     ```
  2. Review `untracked_remote_orders` and `untracked_remote_positions`.
  3. If untracked positions exist, determine if they belong to the automated strategy or a discretionary trade. The bot automatically imports untracked positions into `OrderManager` tracking by default.

### Scenario 3: Order Rejection or Circuit Breaker Triggered
- **Root Cause**: Daily drawdown limit reached (Phase 39 Risk Engine), max gross exposure exceeded, or Alpaca buying power exhausted.
- **Triage Steps**:
  1. Run audit trail summary utility:
     ```python
     from src.execution.audit_trail import audit_trail_summary
     summary = audit_trail_summary()
     print(summary)
     ```
  2. Check `risk_blocks_by_reason` in the summary output.
  3. If risk engine triggered a maximum drawdown halt, verify portfolio PnL in Alpaca dashboard before manual intervention. Do NOT bypass risk thresholds without documented supervisor sign-off.

### Scenario 4: Emergency Halt / Liquidation
If erratic trading or an out-of-control feedback loop is detected:
1. **Immediate Service Termination**:
   ```bash
   docker compose down   # or: sudo systemctl stop trading-bot
   ```
2. **Execute Emergency Order Cancellation**:
   ```python
   from src.execution.broker_client import AlpacaBrokerClient
   broker = AlpacaBrokerClient(paper=True)
   broker.cancel_all_orders()
   ```
3. **Check Remaining Positions**:
   ```python
   for pos in broker.get_positions():
       print(pos.symbol, pos.qty, pos.market_value)
   ```

---

## 5. Audit Trail Investigation & Decision Reconstruction

Every decision made by the system is permanently captured as a JSON object in `/logs/audit/YYYY-MM-DD_audit.jsonl`. This schema enables deterministic post-trade reconstruction:

```json
{
  "event_id": "c1f7a08b-4a52-4411-9a71-6ad3ec7521ab",
  "timestamp": "2026-09-16T14:32:00.124851+00:00",
  "event_type": "SIGNAL_GENERATED",
  "symbol": "NVDA",
  "cycle_id": "cycle_00042",
  "details": {
    "signal": 1,
    "confidence": 0.784,
    "features": {"rsi_14": 31.2, "macd_hist": 0.45, "atr_14": 2.15}
  }
}
```

### Inspecting a Specific Decision
To answer: *"Why did the bot buy NVDA at 14:32?"*
1. Search by symbol and timestamp:
   ```bash
   grep '"symbol": "NVDA"' logs/audit/2026-09-16_audit.jsonl
   ```
2. Trace the exact lifecycle for that `cycle_id`:
   - `SIGNAL_GENERATED`: Inspect model probability and contributing technical/NLP features.
   - `RISK_DECISION`: Confirm risk limits passed and position sizing calculations.
   - `ORDER_SUBMITTED`: Confirm client order ID, quantity, and limit price sent to Alpaca.
   - `ORDER_FILLED`: Check fill price, slippage vs bar close, and execution timestamp.

---

## 6. Daily Operations Checklist
| Time (ET) | Action | Method |
|---|---|---|
| **09:15** | Pre-Market Health Check | Verify bot is running in idle state; review overnight broker account equity. |
| **09:30** | Market Open Validation | Verify log confirms transition from idle to active bar evaluation. |
| **12:00** | Midday Heartbeat Review | Inspect `audit_trail_summary()` to check signal-to-order conversion rate and risk blocks. |
| **16:00** | Market Close Reconciliation | Confirm all day orders are expired/cancelled; reconcile closing book. |
| **16:05** | Daily Audit Summary Generation | Archive daily audit log to cloud / object storage for compliance. |
