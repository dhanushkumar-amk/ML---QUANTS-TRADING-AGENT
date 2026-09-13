# ============================================================
# Universe Builder — Survivorship-Bias-Free Universe Engine
# ============================================================
"""
Constructs point-in-time investable universes (e.g. S&P 500) to eliminate
survivorship bias in quant research and backtesting.

Why Point-in-Time Universe is Critical:
    Standard backtests suffer from survivorship bias if they simply evaluate
    today's S&P 500 index members backwards in time. A backtest starting in 2015
    must include companies that were actually in the index on 2015-01-01 (such as
    Yahoo, Monsanto, Celgene, Rockwell Collins) rather than only companies that
    survived through 2026.

Data Sources & Methodology:
    1. Base Snapshot: Current/reference constituents stored in `data/universe/sp500_constituents.csv`.
    2. Historical Changes Table: Point-in-time index additions and removals stored
       in `data/universe/sp500_historical_changes.csv`. This tracks historical dates,
       added tickers, removed tickers, and reasons.
    3. Delisted Registry: Integrated with `DelistedRegistry` to identify which removed
       tickers ceased trading entirely vs. which were merely rebalanced out.
    4. Reconstruction Algorithm:
       Given an `as_of_date`, the engine applies historical changes backward from
       the reference snapshot (or forward across recorded addition/removal intervals).
       Stocks removed *after* `as_of_date` are restored to the universe; stocks added
       *after* `as_of_date` are excluded.

Limitations:
    • Public records of S&P 500 changes are highly accurate from ~2000 onward, but
      sparse before the late 1990s.
    • Ticker symbol renames and dual-class share splits (e.g., GOOG vs. GOOGL)
      must be reconciled to match historical price data identifiers.

Usage:
    from src.data_pipeline.universe_builder import UniverseBuilder

    builder = UniverseBuilder()
    tickers_2015 = builder.get_universe("2015-01-01")
    constituents_2015 = builder.get_constituents("2015-01-01")
    # 'YHOO' will be present in 2015-01-01, but absent in 2024-01-01
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.data_pipeline.delisted_tickers import DelistedInfo, DelistedRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_UNIVERSE_DIR = Path(__file__).resolve().parents[2] / "data" / "universe"
_DEFAULT_CONSTITUENTS_PATH = _DEFAULT_UNIVERSE_DIR / "sp500_constituents.csv"
_DEFAULT_CHANGES_PATH = _DEFAULT_UNIVERSE_DIR / "sp500_historical_changes.csv"


@dataclass
class UniverseConstituent:
    """Represents an asset's point-in-time status in an index/universe."""

    ticker: str
    name: str
    active_from: datetime.date | None = None
    active_to: datetime.date | None = None  # None = active through present
    is_delisted: bool = False
    delisting_date: datetime.date | None = None
    delisting_reason: str | None = None
    notes: str = ""


class UniverseBuilder:
    """Builds point-in-time constituent lists avoiding survivorship bias."""

    def __init__(
        self,
        constituents_path: Path | str | None = None,
        changes_path: Path | str | None = None,
        delisted_registry: DelistedRegistry | None = None,
    ) -> None:
        self.constituents_path = (
            Path(constituents_path) if constituents_path else _DEFAULT_CONSTITUENTS_PATH
        )
        self.changes_path = Path(changes_path) if changes_path else _DEFAULT_CHANGES_PATH
        self.delisted_registry = delisted_registry or DelistedRegistry()

        self._base_constituents: dict[str, str] = {}  # ticker -> name
        self._changes: list[dict] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load base constituents snapshot and historical changes table."""
        if self.constituents_path.exists():
            try:
                df_base = pd.read_csv(self.constituents_path)
                for _, row in df_base.iterrows():
                    ticker = str(row["ticker"]).strip().upper()
                    name = str(row.get("name", ticker)).strip()
                    self._base_constituents[ticker] = name
                logger.info(
                    "Loaded %d base constituents from %s",
                    len(self._base_constituents),
                    self.constituents_path,
                )
            except Exception as exc:
                logger.error(
                    "Failed to load base constituents from %s: %s", self.constituents_path, exc
                )
        else:
            logger.warning("Base constituents file not found at %s", self.constituents_path)

        if self.changes_path.exists():
            try:
                df_changes = pd.read_csv(self.changes_path)
                df_changes["date"] = pd.to_datetime(df_changes["date"]).dt.date
                df_changes = df_changes.sort_values("date").reset_index(drop=True)

                for _, row in df_changes.iterrows():
                    t_add = (
                        str(row["ticker_added"]).strip().upper()
                        if pd.notna(row.get("ticker_added"))
                        else None
                    )
                    t_rem = (
                        str(row["ticker_removed"]).strip().upper()
                        if pd.notna(row.get("ticker_removed"))
                        else None
                    )

                    self._changes.append(
                        {
                            "date": row["date"],
                            "ticker_added": t_add,
                            "name_added": str(row.get("name_added", t_add or "")).strip(),
                            "ticker_removed": t_rem,
                            "name_removed": str(row.get("name_removed", t_rem or "")).strip(),
                            "reason": str(row.get("reason", "")).strip(),
                        }
                    )
                logger.info(
                    "Loaded %d historical index changes from %s",
                    len(self._changes),
                    self.changes_path,
                )
            except Exception as exc:
                logger.error(
                    "Failed to load historical changes from %s: %s", self.changes_path, exc
                )
        else:
            logger.warning("Historical changes file not found at %s", self.changes_path)

    # ---- Point-in-Time Reconstruction -----------------------------------

    def get_universe(self, as_of_date: str | datetime.date | pd.Timestamp) -> list[str]:
        """Return the sorted list of tickers active in the universe on as_of_date."""
        constituents = self.get_constituents(as_of_date)
        return sorted(c.ticker for c in constituents)

    def get_constituents(
        self,
        as_of_date: str | datetime.date | pd.Timestamp,
    ) -> list[UniverseConstituent]:
        """Reconstruct full constituent records as of a specific historical date.

        Uses backward reconstruction from the modern base snapshot:
          • Any stock added to the index AFTER as_of_date was not yet in the index.
          • Any stock removed from the index AFTER as_of_date WAS in the index.
        """
        target_date = pd.to_datetime(as_of_date).date()

        # Start with current/modern base snapshot
        active_tickers: dict[str, str] = dict(self._base_constituents)

        # Apply changes that occurred AFTER target_date in reverse chronological order
        changes_after = [c for c in self._changes if c["date"] > target_date]
        changes_after_rev = sorted(changes_after, key=lambda c: c["date"], reverse=True)

        for change in changes_after_rev:
            t_add = change["ticker_added"]
            t_rem = change["ticker_removed"]

            # If added after target_date, it wasn't there yet on target_date -> remove
            if t_add and t_add in active_tickers:
                del active_tickers[t_add]

            # If removed after target_date, it WAS present on target_date -> restore
            if t_rem:
                active_tickers[t_rem] = change["name_removed"] or t_rem

        # Build detailed UniverseConstituent objects
        results: list[UniverseConstituent] = []
        for ticker, name in sorted(active_tickers.items()):
            # Look up delisting status
            delist_info: DelistedInfo | None = self.delisted_registry.get_delisting_info(ticker)
            is_delisted = False
            delisting_date = None
            delisting_reason = None
            notes = ""

            if delist_info:
                is_delisted = True
                delisting_date = delist_info.delisting_date
                delisting_reason = delist_info.reason
                notes = delist_info.notes

            # Find active_to date if removed later in changes table
            removal_event = next(
                (
                    c
                    for c in self._changes
                    if c["ticker_removed"] == ticker and c["date"] > target_date
                ),
                None,
            )
            active_to = removal_event["date"] if removal_event else None

            # Find addition date if added prior to target_date
            addition_event = next(
                (
                    c
                    for c in self._changes
                    if c["ticker_added"] == ticker and c["date"] <= target_date
                ),
                None,
            )
            active_from = addition_event["date"] if addition_event else None

            results.append(
                UniverseConstituent(
                    ticker=ticker,
                    name=name,
                    active_from=active_from,
                    active_to=active_to,
                    is_delisted=is_delisted,
                    delisting_date=delisting_date,
                    delisting_reason=delisting_reason,
                    notes=notes,
                )
            )

        logger.debug(
            "Reconstructed universe as of %s: %d constituents (%d flagged as delisted later)",
            target_date,
            len(results),
            sum(1 for c in results if c.is_delisted),
        )
        return results

    def get_universe_history(
        self,
        dates: Sequence[str | datetime.date | pd.Timestamp],
    ) -> dict[datetime.date, list[str]]:
        """Return point-in-time universe snapshots for a sequence of dates."""
        history: dict[datetime.date, list[str]] = {}
        for dt in dates:
            d = pd.to_datetime(dt).date()
            history[d] = self.get_universe(d)
        return history

    def to_dataframe(
        self,
        as_of_date: str | datetime.date | pd.Timestamp,
    ) -> pd.DataFrame:
        """Export universe as of a specific date as a pandas DataFrame."""
        constituents = self.get_constituents(as_of_date)
        records = [
            {
                "ticker": c.ticker,
                "name": c.name,
                "active_from": c.active_from,
                "active_to": c.active_to,
                "is_delisted": c.is_delisted,
                "delisting_date": c.delisting_date,
                "delisting_reason": c.delisting_reason,
                "notes": c.notes,
            }
            for c in constituents
        ]
        return pd.DataFrame(records)
