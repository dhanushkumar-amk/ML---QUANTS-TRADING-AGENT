# ============================================================
# Unified Data Access Layer — Single Source of Truth
# ============================================================
"""
Unified data access layer sitting atop the storage backend and universe engine.

Architecture:
-------------
All downstream phases (feature engineering, machine learning models, backtesting,
and execution) must interact EXCLUSIVELY through this module rather than reading
raw or processed files directly.

Provides:
  • `get_ohlcv(ticker, start, end, columns)`: Efficient time-sliced OHLCV queries.
  • `save_ohlcv(ticker, df)`: Partitioned persistence.
  • `get_universe(date)`: Survivorship-bias-free point-in-time constituent lists.
  • `get_constituents(date)`: Point-in-time constituent records with metadata.
  • `get_corporate_actions(ticker)`: Stock split and dividend events.
  • `get_delisted_info(ticker)`: Delisting metadata and reasons.
"""

from __future__ import annotations

import datetime
from typing import Any, Sequence

import pandas as pd

from src.data_pipeline.corporate_actions import CorporateActionsAdjuster, SplitEvent
from src.data_pipeline.delisted_tickers import DelistedInfo, DelistedRegistry
from src.data_pipeline.storage_backend import ParquetBackend, StorageBackend
from src.data_pipeline.universe_builder import UniverseBuilder, UniverseConstituent
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataAccessLayer:
    """Unified query and storage interface for quant research and modeling."""

    def __init__(
        self,
        backend: StorageBackend | None = None,
        universe_builder: UniverseBuilder | None = None,
        delisted_registry: DelistedRegistry | None = None,
        actions_adjuster: CorporateActionsAdjuster | None = None,
    ) -> None:
        self.backend = backend or ParquetBackend()
        self.universe_builder = universe_builder or UniverseBuilder()
        self.delisted_registry = delisted_registry or DelistedRegistry()
        self.actions_adjuster = actions_adjuster or CorporateActionsAdjuster()

    # ---- OHLCV Data Access ----------------------------------------------

    def get_ohlcv(
        self,
        ticker: str,
        start: str | datetime.date | pd.Timestamp | None = None,
        end: str | datetime.date | pd.Timestamp | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Retrieve historical OHLCV data for a ticker over a given date range.

        Parameters
        ----------
        ticker : str
            Symbol (e.g. 'AAPL').
        start : str | date | Timestamp | None
            Start date inclusive.
        end : str | date | Timestamp | None
            End date inclusive.
        columns : Sequence[str] | None
            Specific columns to project (e.g., ['date', 'close', 'volume']).

        Returns
        -------
        pd.DataFrame
            Sorted, filtered OHLCV dataframe.
        """
        df = self.backend.query(
            key=ticker,
            start=start,
            end=end,
            columns=columns,
        )
        if df.empty:
            logger.debug("[%s] get_ohlcv returned 0 rows for range [%s, %s]", ticker, start, end)
        return df

    def save_ohlcv(
        self,
        ticker: str,
        df: pd.DataFrame,
        date_col: str = "date",
        overwrite: bool = True,
    ) -> Any:
        """Persist OHLCV data for a ticker via the configured storage backend."""
        return self.backend.save(
            df=df,
            key=ticker,
            date_col=date_col,
            overwrite=overwrite,
        )

    def list_available_tickers(self, prefix: str = "") -> list[str]:
        """List all tickers stored in the data backend."""
        return self.backend.list_keys(prefix=prefix)

    # ---- Universe Data Access -------------------------------------------

    def get_universe(self, date: str | datetime.date | pd.Timestamp) -> list[str]:
        """Get the point-in-time list of active constituent tickers on a specific date.

        Guaranteed free of survivorship bias.
        """
        return self.universe_builder.get_universe(date)

    def get_constituents(
        self,
        date: str | datetime.date | pd.Timestamp,
    ) -> list[UniverseConstituent]:
        """Get point-in-time constituent records including delisting status."""
        return self.universe_builder.get_constituents(date)

    # ---- Corporate Actions & Delistings ---------------------------------

    def get_corporate_actions(
        self,
        ticker: str,
        price_col: str = "close",
        use_authoritative: bool = True,
    ) -> list[SplitEvent]:
        """Detect and return corporate actions (splits) for the ticker."""
        df = self.get_ohlcv(ticker)
        if df.empty:
            return []
        return self.actions_adjuster.detect_splits(
            df=df,
            ticker=ticker,
            price_col=price_col,
            use_authoritative=use_authoritative,
        )

    def get_delisted_info(self, ticker: str) -> DelistedInfo | None:
        """Query metadata if the ticker was ever delisted, acquired, or liquidated."""
        return self.delisted_registry.get_delisting_info(ticker)


# Global default instance
_DEFAULT_DATA_ACCESS: DataAccessLayer | None = None


def get_data_access() -> DataAccessLayer:
    """Return the global singleton DataAccessLayer instance."""
    global _DEFAULT_DATA_ACCESS
    if _DEFAULT_DATA_ACCESS is None:
        _DEFAULT_DATA_ACCESS = DataAccessLayer()
    return _DEFAULT_DATA_ACCESS
