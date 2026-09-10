# ============================================================
# Alpha Vantage Adapter — STUB
# ============================================================
"""
Fallback / secondary market-data source via Alpha Vantage.

This is a **stub** for Phase 2.  The interface mirrors
``YFinanceLoader`` so callers can swap sources transparently.
Implementation will be completed in a later phase once an
Alpha Vantage API key is provisioned.

Usage:
    from src.data_pipeline.alphavantage_loader import AlphaVantageLoader

    loader = AlphaVantageLoader(cfg["data"])
    df = loader.fetch("AAPL")   # → NotImplementedError for now
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlphaVantageLoader:
    """Stub adapter for Alpha Vantage TIME_SERIES_DAILY_ADJUSTED.

    TODO (future phase):
        - Accept API key from .env via get_env("ALPHA_VANTAGE_KEY")
        - Hit the /query endpoint with retries
        - Normalise response JSON → OHLCV DataFrame
        - Respect the free-tier rate limit (5 req/min, 500 req/day)
    """

    def __init__(self, data_cfg: dict[str, Any]) -> None:
        self.tickers: list[str] = data_cfg.get("tickers", [])
        self.start: str = data_cfg.get("start_date", "2018-01-01")
        self.end: str = data_cfg.get("end_date", "2026-09-01")
        logger.info("AlphaVantageLoader initialised (STUB — not yet implemented).")

    def fetch(self, ticker: str, **kwargs: Any) -> pd.DataFrame | None:
        """Fetch OHLCV for *ticker*. Currently raises NotImplementedError."""
        raise NotImplementedError(
            f"AlphaVantageLoader.fetch('{ticker}') is a stub. "
            "Implement in a future phase with a valid API key."
        )

    def fetch_batch(
        self, tickers: list[str] | None = None, **kwargs: Any
    ) -> dict[str, pd.DataFrame]:
        """Batch fetch — delegates to ``fetch`` per ticker."""
        raise NotImplementedError(
            "AlphaVantageLoader.fetch_batch() is a stub. "
            "Implement in a future phase."
        )
