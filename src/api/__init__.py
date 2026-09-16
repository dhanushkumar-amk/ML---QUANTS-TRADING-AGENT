# ============================================================
# src.api — FastAPI Backend Services for Trading Dashboard
# ============================================================
"""
FastAPI application exposing quantitative trading engine data,
broker status, backtest metrics, and audit trail logs to the Next.js frontend.
"""

from src.api.server import app

__all__ = ["app"]
