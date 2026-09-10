# ============================================================
# Real-Time Data Buffer — in-memory + periodic Parquet flush
# ============================================================
"""
Thread-safe, per-ticker rolling buffer that:
  • Accumulates incoming bars/quotes in memory (deque).
  • Periodically flushes to ``data/raw/realtime/{ticker}.parquet``.
  • De-duplicates on restart by loading existing Parquet timestamps.
  • Tracks per-ticker tick counts and staleness for health monitoring.

Usage:
    from src.data_pipeline.realtime_buffer import RealtimeBuffer

    buf = RealtimeBuffer(["AAPL", "MSFT", "SPY"])
    buf.append("AAPL", {"timestamp": ..., "open": ..., ...})
    buf.maybe_flush()          # call periodically
    buf.flush_all()            # call on shutdown
    buf.print_stats()          # periodic health report
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REALTIME_DIR = _PROJECT_ROOT / "data" / "raw" / "realtime"


class RealtimeBuffer:
    """Thread-safe per-ticker data buffer with Parquet persistence."""

    def __init__(
        self,
        tickers: list[str],
        data_dir: str | Path | None = None,
        flush_interval: int = 60,
        max_buffer_size: int = 10_000,
        stale_threshold: int = 120,
    ) -> None:
        """
        Parameters
        ----------
        tickers : list[str]
            Ticker symbols to track.
        data_dir : str | Path | None
            Override for the realtime data directory.
        flush_interval : int
            Seconds between automatic Parquet flushes.
        max_buffer_size : int
            Max bars per ticker held in memory (oldest are dropped).
        stale_threshold : int
            Warn if no data for a ticker for this many seconds.
        """
        self.data_dir = Path(data_dir or _REALTIME_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.flush_interval = flush_interval
        self.stale_threshold = stale_threshold

        self._lock = threading.Lock()
        self._buffers: dict[str, deque[dict[str, Any]]] = {
            t: deque(maxlen=max_buffer_size) for t in tickers
        }
        self._tick_counts: dict[str, int] = {t: 0 for t in tickers}
        self._report_counts: dict[str, int] = {t: 0 for t in tickers}
        self._last_received: dict[str, float] = {t: 0.0 for t in tickers}
        self._last_flush: float = time.time()
        self._total_flushed: dict[str, int] = {t: 0 for t in tickers}

        # Load existing timestamps for de-duplication
        self._existing_max_ts: dict[str, pd.Timestamp | None] = {}
        self._load_existing_timestamps()

    # ---- de-duplication -------------------------------------------------

    def _load_existing_timestamps(self) -> None:
        """Scan existing Parquet files to find the latest timestamp per
        ticker so we can skip duplicates on restart."""
        for ticker in self._buffers:
            filepath = self.data_dir / f"{ticker}.parquet"
            if filepath.exists():
                try:
                    df = pd.read_parquet(filepath, columns=["timestamp"])
                    if not df.empty:
                        max_ts = pd.to_datetime(df["timestamp"]).max()
                        self._existing_max_ts[ticker] = max_ts
                        logger.info(
                            "Buffer: %s existing data up to %s -- will skip older bars.",
                            ticker, max_ts,
                        )
                        continue
                except Exception as exc:
                    logger.warning("Could not read %s: %s", filepath, exc)
            self._existing_max_ts[ticker] = None

    # ---- append ---------------------------------------------------------

    def append(self, ticker: str, bar: dict[str, Any]) -> bool:
        """Append a bar dict to the ticker's buffer.

        Returns ``True`` if the bar was accepted, ``False`` if it was
        a duplicate (timestamp ≤ existing max).
        """
        ts = bar.get("timestamp")
        if ts is not None:
            ts = pd.Timestamp(ts)
            max_ts = self._existing_max_ts.get(ticker)
            if max_ts is not None and ts <= max_ts:
                return False  # duplicate

        with self._lock:
            if ticker not in self._buffers:
                self._buffers[ticker] = deque(maxlen=10_000)
                self._tick_counts[ticker] = 0
                self._report_counts[ticker] = 0
                self._last_received[ticker] = 0.0
                self._total_flushed[ticker] = 0

            self._buffers[ticker].append(bar)
            self._tick_counts[ticker] += 1
            self._last_received[ticker] = time.time()

        return True

    # ---- flush ----------------------------------------------------------

    def maybe_flush(self) -> None:
        """Flush to Parquet if ``flush_interval`` seconds have elapsed."""
        if time.time() - self._last_flush >= self.flush_interval:
            self.flush_all()

    def flush_all(self) -> None:
        """Flush all tickers' buffers to Parquet."""
        for ticker in list(self._buffers):
            self._flush_ticker(ticker)
        self._last_flush = time.time()

    def _flush_ticker(self, ticker: str) -> None:
        """Flush a single ticker's buffer to its Parquet file."""
        with self._lock:
            if not self._buffers[ticker]:
                return
            new_rows = list(self._buffers[ticker])
            self._buffers[ticker].clear()

        new_df = pd.DataFrame(new_rows)

        # Ensure timestamp column
        if "timestamp" in new_df.columns:
            new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])

        filepath = self.data_dir / f"{ticker}.parquet"

        try:
            if filepath.exists():
                existing = pd.read_parquet(filepath, engine="pyarrow")
                combined = pd.concat([existing, new_df], ignore_index=True)
                # De-duplicate by timestamp
                if "timestamp" in combined.columns:
                    combined = (
                        combined.drop_duplicates(subset=["timestamp"])
                        .sort_values("timestamp")
                        .reset_index(drop=True)
                    )
            else:
                combined = new_df

            combined.to_parquet(filepath, index=False, engine="pyarrow")

            # Update max timestamp for future de-dup
            if "timestamp" in combined.columns and not combined.empty:
                self._existing_max_ts[ticker] = combined["timestamp"].max()

            flushed_count = len(new_rows)
            self._total_flushed[ticker] = self._total_flushed.get(ticker, 0) + flushed_count
            logger.info(
                "Flushed %d bars for %s -> %s (total on disk: %d)",
                flushed_count, ticker, filepath, len(combined),
            )
        except Exception as exc:
            logger.error("Failed to flush %s: %s — data kept in memory.", ticker, exc)
            # Put the data back
            with self._lock:
                for row in reversed(new_rows):
                    self._buffers[ticker].appendleft(row)

    # ---- health monitoring ----------------------------------------------

    def print_stats(self) -> None:
        """Print per-ticker stats and stale-data warnings."""
        now = time.time()
        border = "-" * 60
        lines = [f"\n{border}", "  [STATS] Real-Time Feed Health", border]

        for ticker in sorted(self._buffers):
            count = self._tick_counts[ticker]
            since_report = count - self._report_counts.get(ticker, 0)
            last_recv = self._last_received[ticker]
            elapsed = now - last_recv if last_recv > 0 else float("inf")
            flushed = self._total_flushed.get(ticker, 0)

            status = "[OK]"
            if last_recv == 0:
                status = "[WAITING]"
            elif elapsed > self.stale_threshold:
                status = f"[STALE ({elapsed:.0f}s ago)]"
                logger.warning(
                    "%s: no data for %.0f seconds — possible feed issue!",
                    ticker, elapsed,
                )

            lines.append(
                f"  {ticker:<6}  new: {since_report:>4}  total: {count:>6}  "
                f"flushed: {flushed:>6}  {status}"
            )
            self._report_counts[ticker] = count

        lines.append(border)
        print("\n".join(lines))

    def get_buffer_sizes(self) -> dict[str, int]:
        """Return current in-memory buffer size per ticker."""
        with self._lock:
            return {t: len(buf) for t, buf in self._buffers.items()}
