# ============================================================
# Historical OHLCV Loader — yfinance
# ============================================================
"""
Fetch historical OHLCV data from Yahoo Finance via yfinance.

Supports single-ticker and multi-ticker batch fetching with
exponential-backoff retry logic and graceful missing-data handling.

Usage:
    from src.data_pipeline.historical_loader import YFinanceLoader

    loader = YFinanceLoader(cfg["data"])
    df = loader.fetch("AAPL")
    batch = loader.fetch_batch(["AAPL", "MSFT", "SPY"])
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import yfinance as yf

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ----- defaults ---------------------------------------------------------
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF_BASE = 2.0  # seconds


class YFinanceLoader:
    """Download OHLCV bars from Yahoo Finance with retry + validation."""

    def __init__(self, data_cfg: dict[str, Any]) -> None:
        """
        Parameters
        ----------
        data_cfg : dict
            The ``data`` section of default.yaml, expecting keys:
            tickers, start_date, end_date, interval.
        """
        self.tickers: list[str] = data_cfg.get("tickers", [])
        self.start: str = data_cfg.get("start_date", "2018-01-01")
        self.end: str = data_cfg.get("end_date", "2026-09-01")
        self.interval: str = data_cfg.get("interval", "1d")
        self.retries: int = data_cfg.get("retries", _DEFAULT_RETRIES)
        self.backoff_base: float = data_cfg.get("backoff_base", _DEFAULT_BACKOFF_BASE)

    # ---- single ticker -------------------------------------------------
    def fetch(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
        interval: str | None = None,
    ) -> pd.DataFrame | None:
        """Fetch OHLCV for a single ticker with retry + backoff.

        Returns ``None`` (instead of raising) if all retries are exhausted
        so that batch runs can continue with partial results.
        """
        _start = start or self.start
        _end = end or self.end
        _interval = interval or self.interval

        for attempt in range(1, self.retries + 1):
            try:
                logger.info(
                    "Fetching %s  [%s → %s, %s]  attempt %d/%d",
                    ticker,
                    _start,
                    _end,
                    _interval,
                    attempt,
                    self.retries,
                )
                df: pd.DataFrame = yf.download(
                    ticker,
                    start=_start,
                    end=_end,
                    interval=_interval,
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )

                if df.empty:
                    logger.warning("%s returned an empty DataFrame — skipping.", ticker)
                    return None

                # Flatten MultiIndex columns if present (yfinance ≥ 0.2.31)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Standardise column names
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                df.index.name = "date"
                df = df.reset_index()

                # Ensure date column is datetime (not DatetimeIndex)
                df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

                # Tag with metadata
                df["ticker"] = ticker
                df["source"] = "yfinance"

                logger.info(
                    "%s fetched OK — %d rows, %s → %s",
                    ticker,
                    len(df),
                    df["date"].min().date(),
                    df["date"].max().date(),
                )
                return df

            except Exception as exc:
                wait = self.backoff_base**attempt
                logger.warning(
                    "%s attempt %d failed (%s). Retrying in %.1fs …",
                    ticker,
                    attempt,
                    exc,
                    wait,
                )
                time.sleep(wait)

        logger.error("%s — all %d retries exhausted. Returning None.", ticker, self.retries)
        return None

    # ---- batch ----------------------------------------------------------
    def fetch_batch(
        self,
        tickers: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch multiple tickers. Returns a {ticker: df} mapping.

        Tickers that fail or return empty data are logged and excluded
        from the result dict (no crash).
        """
        _tickers = tickers or self.tickers
        results: dict[str, pd.DataFrame] = {}

        for t in _tickers:
            df = self.fetch(t, start=start, end=end)
            if df is not None:
                results[t] = df

        logger.info(
            "Batch complete — %d/%d tickers succeeded.",
            len(results),
            len(_tickers),
        )
        return results

    # ---- point-in-time universe -----------------------------------------
    def fetch_universe(
        self,
        universe: Any,
        as_of_date: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical bars for a point-in-time universe.

        Supports UniverseBuilder instances, lists of UniverseConstituent, or
        standard ticker lists. Respects active tenure windows and logs clear,
        transparent warnings when delisted securities cannot be retrieved from
        free retail data providers (yfinance).

        Parameters
        ----------
        universe : UniverseBuilder | list[UniverseConstituent] | list[str]
            Point-in-time universe representation.
        as_of_date : str | None
            Historical date when querying a UniverseBuilder instance.
        start : str | None
            Override query start date.
        end : str | None
            Override query end date.

        Returns
        -------
        dict[str, pd.DataFrame]
        """
        # Resolve constituents
        if hasattr(universe, "get_constituents"):
            target_date = as_of_date or self.start
            constituents = universe.get_constituents(target_date)
        elif isinstance(universe, list):
            constituents = universe
        else:
            raise TypeError(
                f"Unsupported universe type: {type(universe)}. Expected UniverseBuilder or list."
            )

        results: dict[str, pd.DataFrame] = {}

        for item in constituents:
            if hasattr(item, "ticker"):
                ticker = item.ticker
                is_delisted = getattr(item, "is_delisted", False)
                delist_date = getattr(item, "delisting_date", None)
                delist_reason = getattr(item, "delisting_reason", None)
                active_from = getattr(item, "active_from", None)
            else:
                ticker = str(item)
                is_delisted = False
                delist_date = None
                delist_reason = None
                active_from = None

            # Determine appropriate query date window
            req_start = start or (str(active_from) if active_from else self.start)
            req_end = end or (str(delist_date) if delist_date else self.end)

            logger.info(
                "Fetching universe constituent %s (delisted=%s, range: %s -> %s)",
                ticker,
                is_delisted,
                req_start,
                req_end,
            )

            df = self.fetch(ticker, start=req_start, end=req_end)

            if df is not None and not df.empty:
                results[ticker] = df
            else:
                if is_delisted:
                    logger.warning(
                        "[%s] Delisted ticker data unavailable on yfinance (delisted %s, reason: %s). "
                        "Known limitation: free retail feeds purge historical prices for delisted symbols. "
                        "Institutional survivorship-free feeds (e.g. CRSP, Norgate) are needed for full tick coverage.",
                        ticker,
                        delist_date,
                        delist_reason,
                    )
                else:
                    logger.warning(
                        "[%s] Constituent data could not be fetched from yfinance.", ticker
                    )

        logger.info(
            "Universe fetch complete — %d/%d constituents successfully fetched.",
            len(results),
            len(constituents),
        )
        return results
