# ============================================================
# Polling Fallback Feed — yfinance near-real-time data
# ============================================================
"""
Polling feed that periodically fetches recent intraday bars (1m, 5m, etc.)
via yfinance and feeds them into the RealtimeBuffer.

Acts as a lightweight, API-key-free fallback to live WebSockets,
particularly useful when markets are closed (fetches latest available bars),
during development, or when Alpaca credentials are not configured.

Usage:
    from src.data_pipeline.polling_feed import PollingFeed
    from src.data_pipeline.realtime_buffer import RealtimeBuffer

    buffer = RealtimeBuffer(["AAPL", "MSFT"])
    feed = PollingFeed(tickers=["AAPL", "MSFT"], buffer=buffer, poll_interval=10)
    feed.run(duration=60)
"""

from __future__ import annotations

import datetime
import threading
import time
from typing import Any

import pandas as pd
import yfinance as yf

from src.data_pipeline.realtime_buffer import RealtimeBuffer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PollingFeed:
    """Periodically queries yfinance for the latest intraday bars."""

    def __init__(
        self,
        tickers: list[str],
        buffer: RealtimeBuffer,
        interval: str = "1m",
        poll_interval: int = 15,
        lookback_period: str = "1d",
    ) -> None:
        """
        Parameters
        ----------
        tickers : list[str]
            List of ticker symbols to track.
        buffer : RealtimeBuffer
            Thread-safe buffer instance to push bars into.
        interval : str
            Intraday bar interval for yfinance (e.g., '1m', '2m', '5m').
        poll_interval : int
            Seconds to sleep between polling rounds.
        lookback_period : str
            History window to request each poll (e.g. '1d' or '5d').
        """
        self.tickers = [t.upper().strip() for t in tickers]
        self.buffer = buffer
        self.interval = interval
        self.poll_interval = poll_interval
        self.lookback_period = lookback_period

        self._stop_event = threading.Event()
        self._poll_count = 0
        self._new_bars_count = 0

    def stop(self) -> None:
        """Signal the polling loop to stop."""
        self._stop_event.set()

    def poll_once(self) -> dict[str, int]:
        """Perform one polling round across all configured tickers.

        Returns
        -------
        dict[str, int]
            Mapping of ticker -> number of new bars appended.
        """
        new_counts: dict[str, int] = {}

        for ticker in self.tickers:
            added = self._poll_ticker(ticker)
            new_counts[ticker] = added
            self._new_bars_count += added

        self._poll_count += 1
        return new_counts

    def _poll_ticker(self, ticker: str) -> int:
        """Fetch latest bars for a single ticker and push new bars to buffer."""
        try:
            # yfinance download with minimal output
            df = yf.download(
                ticker,
                period=self.lookback_period,
                interval=self.interval,
                progress=False,
                auto_adjust=False,
            )

            if df is None or df.empty:
                logger.debug("[%s] Polling returned empty dataframe.", ticker)
                return 0

            # Handle potential MultiIndex columns from newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                    "Adj Close": "adj_close",
                }
            )

            # Standardize index to timestamp column
            df = df.reset_index()
            # The timestamp col could be 'Datetime' or 'Date'
            date_col = next((c for c in df.columns if str(c).lower() in ("datetime", "date")), None)
            if not date_col:
                logger.warning("[%s] Could not identify datetime column in yfinance response.", ticker)
                return 0

            df = df.rename(columns={date_col: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Keep only standard columns
            keep_cols = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[keep_cols]

            new_bars_added = 0
            # Iterate through the most recent bars (up to last 10)
            recent_df = df.tail(10)
            for _, row in recent_df.iterrows():
                bar_dict: dict[str, Any] = {
                    "ticker": ticker,
                    "timestamp": row["timestamp"],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0)),
                }

                appended = self.buffer.append(ticker, bar_dict)
                if appended:
                    new_bars_added += 1
                    logger.debug(
                        "[%s] Appended new bar: %s | Close: %.2f | Vol: %.0f",
                        ticker,
                        bar_dict["timestamp"],
                        bar_dict["close"],
                        bar_dict["volume"],
                    )

            return new_bars_added

        except Exception as exc:
            logger.warning("[%s] Error during yfinance poll: %s", ticker, exc)
            return 0

    def run(self, duration: int | None = None) -> None:
        """Run the polling loop continuously or until duration expires.

        Parameters
        ----------
        duration : int | None
            Maximum number of seconds to run. If None, runs until stop()
            or KeyboardInterrupt.
        """
        start_time = time.time()
        logger.info(
            "Starting PollingFeed for %s (interval=%s, poll_frequency=%ds, duration=%s)",
            self.tickers,
            self.interval,
            self.poll_interval,
            f"{duration}s" if duration else "forever",
        )

        try:
            while not self._stop_event.is_set():
                round_start = time.time()

                # Poll tickers
                counts = self.poll_once()
                total_round = sum(counts.values())

                logger.debug(
                    "Poll round #%d completed: %d new bars across %d tickers.",
                    self._poll_count,
                    total_round,
                    len(self.tickers),
                )

                # Check if buffer needs periodic Parquet flush
                self.buffer.maybe_flush()

                # Check duration
                elapsed = time.time() - start_time
                if duration is not None and elapsed >= duration:
                    logger.info("Duration limit (%ds) reached. Stopping feed.", duration)
                    break

                # Sleep remaining time of poll interval
                process_duration = time.time() - round_start
                sleep_time = max(0.5, self.poll_interval - process_duration)

                # Responsive sleep checking stop_event every 0.5s
                sleep_end = time.time() + sleep_time
                while time.time() < sleep_end and not self._stop_event.is_set():
                    if duration is not None and (time.time() - start_time) >= duration:
                        break
                    time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt — shutting down polling feed.")
        finally:
            logger.info("Polling feed stopped. Flushing remaining buffer to disk...")
            self.buffer.flush_all()
