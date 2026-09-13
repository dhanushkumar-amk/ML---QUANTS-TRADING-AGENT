# ============================================================
# Delisted Tickers Registry — Survivorship Bias Prevention
# ============================================================
"""
Maintains a registry of historical delisted, acquired, and bankrupt tickers.

Why This Matters (Survivorship Bias):
    In quantitative research, testing a strategy exclusively on currently active
    stocks artificially inflates returns (often by 1-4% annualized Sharpe drag),
    because losers that went bankrupt (e.g., Enron, Lehman Brothers, Silicon Valley
    Bank) or companies acquired at a discount are purged from modern ticker lists.
    A valid point-in-time backtest must evaluate the exact universe of investable
    assets that existed on that trading day.

Data Source & Known Limitations:
    1. Source: Curated reference catalog compiled from SEC Form 25 filings (delisting),
       S&P Dow Jones index change announcements, and corporate merger records.
    2. Gaps & Limitations:
       • Free retail market APIs (Yahoo Finance, Alpha Vantage) aggressively purge
         or overwrite historical data for delisted symbols once their ticker is
         retired or reassigned.
       • Complete survivorship-bias-free daily tick/bar data historically requires
         institutional feeds (such as CRSP, Compustat, or Norgate Data).
       • This module provides the authoritative reference registry to flag, track,
         and reconstruct active windows for delisted assets in the backtesting universe.

Usage:
    from src.data_pipeline.delisted_tickers import DelistedRegistry

    registry = DelistedRegistry()
    info = registry.get_delisting_info("YHOO")
    print(registry.is_delisted("YHOO", as_of_date="2016-01-01"))  # False (was active)
    print(registry.is_delisted("YHOO", as_of_date="2018-01-01"))  # True (delisted in 2017)
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "universe" / "delisted_tickers.csv"
)


@dataclass
class DelistedInfo:
    """Metadata describing a delisted, acquired, or liquidated security."""

    ticker: str
    name: str
    delisting_date: datetime.date
    reason: str  # 'acquired', 'bankruptcy', 'privatized', 'merger', 'regulatory'
    acquirer: str | None = None
    last_price: float | None = None
    notes: str = ""


class DelistedRegistry:
    """Catalog of delisted securities used to filter or audit universe constituents."""

    def __init__(self, data_path: Path | str | None = None) -> None:
        self.data_path = Path(data_path) if data_path else _DEFAULT_REGISTRY_PATH
        self._registry: dict[str, DelistedInfo] = {}
        self._load()

    def _load(self) -> None:
        """Load registry from CSV if it exists."""
        if not self.data_path.exists():
            logger.warning(
                "Delisted tickers database not found at %s. Initializing empty registry.",
                self.data_path,
            )
            return

        try:
            df = pd.read_csv(self.data_path)
            for _, row in df.iterrows():
                ticker = str(row["ticker"]).strip().upper()
                d_date = pd.to_datetime(row["delisting_date"]).date()
                acquirer = (
                    str(row["acquirer"])
                    if pd.notna(row.get("acquirer")) and str(row.get("acquirer")).lower() != "none"
                    else None
                )
                last_price = float(row["last_price"]) if pd.notna(row.get("last_price")) else None
                notes = str(row["notes"]) if pd.notna(row.get("notes")) else ""

                self._registry[ticker] = DelistedInfo(
                    ticker=ticker,
                    name=str(row["name"]).strip(),
                    delisting_date=d_date,
                    reason=str(row["reason"]).strip().lower(),
                    acquirer=acquirer,
                    last_price=last_price,
                    notes=notes,
                )
            logger.info("Loaded %d delisted tickers from %s", len(self._registry), self.data_path)
        except Exception as exc:
            logger.error("Failed to load delisted registry from %s: %s", self.data_path, exc)

    def get_delisting_info(self, ticker: str) -> DelistedInfo | None:
        """Retrieve delisting metadata for a specific ticker."""
        return self._registry.get(ticker.upper())

    def is_delisted(
        self,
        ticker: str,
        as_of_date: str | datetime.date | pd.Timestamp | None = None,
    ) -> bool:
        """Check if a ticker was delisted.

        If `as_of_date` is provided, returns True ONLY if the ticker was delisted
        on or before `as_of_date`. If None, returns True if the ticker is in the
        delisted catalog at all.
        """
        info = self.get_delisting_info(ticker)
        if info is None:
            return False

        if as_of_date is None:
            return True

        target_date = pd.to_datetime(as_of_date).date()
        return info.delisting_date <= target_date

    def get_delisted_between(
        self,
        start_date: str | datetime.date | pd.Timestamp,
        end_date: str | datetime.date | pd.Timestamp,
    ) -> list[DelistedInfo]:
        """Return all tickers delisted within the specified date range [start, end]."""
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()

        matches = [info for info in self._registry.values() if start <= info.delisting_date <= end]
        return sorted(matches, key=lambda x: x.delisting_date)

    def add_delisted_ticker(self, info: DelistedInfo) -> None:
        """Add or update a delisted ticker record in the registry."""
        self._registry[info.ticker.upper()] = info

    def all_delisted_tickers(self) -> list[str]:
        """Return list of all registered delisted symbols."""
        return sorted(self._registry.keys())

    def to_dataframe(self) -> pd.DataFrame:
        """Export registry as a pandas DataFrame."""
        if not self._registry:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "name",
                    "delisting_date",
                    "reason",
                    "acquirer",
                    "last_price",
                    "notes",
                ]
            )

        records = [
            {
                "ticker": info.ticker,
                "name": info.name,
                "delisting_date": info.delisting_date,
                "reason": info.reason,
                "acquirer": info.acquirer,
                "last_price": info.last_price,
                "notes": info.notes,
            }
            for info in self._registry.values()
        ]
        return pd.DataFrame(records)
