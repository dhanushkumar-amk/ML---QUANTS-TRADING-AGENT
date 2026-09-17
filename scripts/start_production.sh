#!/bin/bash
set -e

# 1. Start live paper trading loop in background if Alpaca credentials are configured
if [ -n "$APCA_API_KEY_ID" ] && [ -n "$APCA_API_SECRET_KEY" ]; then
    echo "[INFO] Alpaca API credentials detected. Starting background live trading loop..."
    python -m src.execution.live_trading_loop --symbols AAPL MSFT NVDA GOOGL --bar-interval 60 &
else
    echo "[WARN] No APCA_API_KEY_ID / APCA_API_SECRET_KEY detected. Running in API-only observation mode."
fi

# 2. Start FastAPI Server bound to Render's dynamic PORT
PORT="${PORT:-8000}"
echo "[INFO] Starting FastAPI on 0.0.0.0:${PORT}..."
exec uvicorn src.api.server:app --host 0.0.0.0 --port "${PORT}"
