# ============================================================
# Alpaca Real-Time Market Data Streamer
# ============================================================
"""
WebSocket streaming feed for real-time market data from Alpaca.

Supports:
  • Real-time 1-minute bars, trades, and quotes for configured tickers.
  • Exponential backoff reconnection if the websocket connection drops.
  • Thread-safe feeding into RealtimeBuffer.
  • Duration-based execution or continuous operation.
  • Debug logging for incoming ticks, warning/error logging for connection issues.

Usage:
    from src.data_pipeline.alpaca_stream import AlpacaStreamer
    from src.data_pipeline.realtime_buffer import RealtimeBuffer

    buf = RealtimeBuffer(["AAPL", "MSFT"])
    streamer = AlpacaStreamer(tickers=["AAPL", "MSFT"], buffer=buf)
    streamer.run(duration=60)
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pandas as pd

from src.data_pipeline.realtime_buffer import RealtimeBuffer
from src.utils.config_loader import get_env
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlpacaStreamer:
    """Stream real-time bars, trades, and quotes from Alpaca market data API."""

    def __init__(
        self,
        tickers: list[str],
        buffer: RealtimeBuffer,
        feed: str = "iex",  # 'iex' (free) or 'sip' (paid)
        subscribe_to: tuple[str, ...] = ("bars",),  # bars, trades, quotes
        max_reconnect_retries: int = 10,
        base_backoff_sec: float = 1.0,
        max_backoff_sec: float = 60.0,
    ) -> None:
        """
        Parameters
        ----------
        tickers : list[str]
            Stock symbols to subscribe to.
        buffer : RealtimeBuffer
            Thread-safe buffer instance to push received data into.
        feed : str
            Data feed: 'iex' (default for paper/free accounts) or 'sip'.
        subscribe_to : tuple[str, ...]
            Data types to stream ('bars', 'trades', 'quotes').
        max_reconnect_retries : int
            Max consecutive reconnection attempts before giving up.
        base_backoff_sec : float
            Initial reconnect delay in seconds.
        max_backoff_sec : float
            Maximum reconnect delay in seconds.
        """
        self.tickers = [t.upper().strip() for t in tickers]
        self.buffer = buffer
        self.feed_name = feed.lower()
        self.subscribe_to = subscribe_to
        self.max_reconnect_retries = max_reconnect_retries
        self.base_backoff_sec = base_backoff_sec
        self.max_backoff_sec = max_backoff_sec

        self._api_key = get_env("ALPACA_API_KEY", "")
        self._secret_key = get_env("ALPACA_SECRET_KEY", "")

        if not self._api_key or self._api_key.startswith("your_"):
            raise ValueError(
                "Alpaca API credentials missing or placeholder. "
                "Please set valid ALPACA_API_KEY and ALPACA_SECRET_KEY in .env, "
                "or use the polling feed instead."
            )

        self._stop_event = threading.Event()
        self._stream: Any = None
        self._stream_task: Any = None
        self._bars_received = 0
        self._trades_received = 0
        self._quotes_received = 0

    def _create_stream(self) -> Any:
        """Initialize the Alpaca StockDataStream."""
        from alpaca.data.enums import DataFeed
        from alpaca.data.live.stock import StockDataStream

        feed_enum = DataFeed.SIP if self.feed_name == "sip" else DataFeed.IEX

        stream = StockDataStream(
            api_key=self._api_key,
            secret_key=self._secret_key,
            feed=feed_enum,
        )
        return stream

    # ---- Async handlers -------------------------------------------------

    async def _handle_bar(self, bar: Any) -> None:
        """Handle incoming bar data."""
        try:
            symbol = getattr(bar, "symbol", None)
            timestamp = getattr(bar, "timestamp", None)
            open_px = getattr(bar, "open", None)
            high_px = getattr(bar, "high", None)
            low_px = getattr(bar, "low", None)
            close_px = getattr(bar, "close", None)
            volume = getattr(bar, "volume", 0)

            # If bar is dict
            if isinstance(bar, dict):
                symbol = bar.get("S", bar.get("symbol", symbol))
                timestamp = bar.get("t", bar.get("timestamp", timestamp))
                open_px = bar.get("o", bar.get("open", open_px))
                high_px = bar.get("h", bar.get("high", high_px))
                low_px = bar.get("l", bar.get("low", low_px))
                close_px = bar.get("c", bar.get("close", close_px))
                volume = bar.get("v", bar.get("volume", volume))

            if symbol and timestamp:
                bar_dict = {
                    "ticker": symbol,
                    "timestamp": pd.Timestamp(timestamp),
                    "open": float(open_px) if open_px is not None else 0.0,
                    "high": float(high_px) if high_px is not None else 0.0,
                    "low": float(low_px) if low_px is not None else 0.0,
                    "close": float(close_px) if close_px is not None else 0.0,
                    "volume": float(volume) if volume is not None else 0.0,
                }
                appended = self.buffer.append(symbol, bar_dict)
                if appended:
                    self._bars_received += 1
                    logger.debug(
                        "[Alpaca Bar] %s | %s | Close: %.2f | Vol: %.0f",
                        symbol,
                        bar_dict["timestamp"],
                        bar_dict["close"],
                        bar_dict["volume"],
                    )
        except Exception as exc:
            logger.error("Error processing Alpaca bar: %s", exc)

    async def _handle_trade(self, trade: Any) -> None:
        """Handle incoming trade tick."""
        try:
            symbol = getattr(trade, "symbol", getattr(trade, "S", None))
            ts = getattr(trade, "timestamp", getattr(trade, "t", None))
            px = getattr(trade, "price", getattr(trade, "p", None))
            size = getattr(trade, "size", getattr(trade, "s", 0))

            self._trades_received += 1
            logger.debug("[Alpaca Trade] %s | %s | Price: %s | Size: %s", symbol, ts, px, size)
        except Exception as exc:
            logger.error("Error processing Alpaca trade: %s", exc)

    async def _handle_quote(self, quote: Any) -> None:
        """Handle incoming quote tick."""
        try:
            symbol = getattr(quote, "symbol", getattr(quote, "S", None))
            ts = getattr(quote, "timestamp", getattr(quote, "t", None))
            bid = getattr(quote, "bid_price", getattr(quote, "bp", None))
            ask = getattr(quote, "ask_price", getattr(quote, "ap", None))

            self._quotes_received += 1
            logger.debug("[Alpaca Quote] %s | %s | Bid: %s | Ask: %s", symbol, ts, bid, ask)
        except Exception as exc:
            logger.error("Error processing Alpaca quote: %s", exc)

    # ---- Control methods ------------------------------------------------

    def stop(self) -> None:
        """Signal streamer to stop and shut down WebSocket connection."""
        self._stop_event.set()
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception as exc:
                logger.debug("Error stopping Alpaca stream: %s", exc)

    def run(self, duration: int | None = None) -> None:
        """Run the Alpaca WebSocket stream with exponential backoff reconnect logic.

        Parameters
        ----------
        duration : int | None
            Maximum duration in seconds to run. If None, runs indefinitely.
        """
        retries = 0

        logger.info(
            "Starting Alpaca Streamer for %s (feed=%s, subs=%s, duration=%s)",
            self.tickers,
            self.feed_name,
            self.subscribe_to,
            f"{duration}s" if duration else "forever",
        )

        # Background duration watcher thread if duration is specified
        if duration is not None:

            def _duration_timer():
                time.sleep(duration)
                logger.info("Duration limit of %ds reached -- stopping Alpaca stream.", duration)
                self.stop()

            timer_thread = threading.Thread(target=_duration_timer, daemon=True)
            timer_thread.start()

        # Background periodic Parquet flusher thread
        def _periodic_flusher():
            while not self._stop_event.is_set():
                time.sleep(self.buffer.flush_interval)
                self.buffer.maybe_flush()

        flusher_thread = threading.Thread(target=_periodic_flusher, daemon=True)
        flusher_thread.start()

        while not self._stop_event.is_set():
            try:
                self._stream = self._create_stream()

                if "bars" in self.subscribe_to:
                    self._stream.subscribe_bars(self._handle_bar, *self.tickers)
                if "trades" in self.subscribe_to:
                    self._stream.subscribe_trades(self._handle_trade, *self.tickers)
                if "quotes" in self.subscribe_to:
                    self._stream.subscribe_quotes(self._handle_quote, *self.tickers)

                logger.info("Connected to Alpaca WebSocket feed (%s). Listening...", self.feed_name)
                retries = 0  # reset retry counter on successful connection

                # stream.run() blocks until stopped or connection drops
                self._stream.run()

            except Exception as exc:
                if self._stop_event.is_set():
                    break

                retries += 1
                if retries > self.max_reconnect_retries:
                    logger.error(
                        "Max reconnect retries (%d) reached. Aborting Alpaca stream.",
                        self.max_reconnect_retries,
                    )
                    break

                backoff = min(
                    self.max_backoff_sec,
                    self.base_backoff_sec * (2 ** (retries - 1)),
                )
                logger.warning(
                    "Alpaca WebSocket error: %s. Reconnecting in %.1fs (attempt %d/%d)...",
                    exc,
                    backoff,
                    retries,
                    self.max_reconnect_retries,
                )
                time.sleep(backoff)

        logger.info("Alpaca streamer stopped. Final flush to disk...")
        self.buffer.flush_all()
