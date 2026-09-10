#!/usr/bin/env python
# ============================================================
# CLI Entry Point: Real-Time Market Data Streamer
# ============================================================
"""
Stream or poll real-time intraday market data and persist to Parquet.

Usage examples:
    # Run Alpaca WebSocket stream for 60 seconds:
    python scripts/stream_data.py --source alpaca --duration 60

    # Run polling fallback for 30 seconds with custom tickers:
    python scripts/stream_data.py --source polling --duration 30 --tickers AAPL MSFT

    # Run continuously (Ctrl+C to stop):
    python scripts/stream_data.py --source polling
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.alpaca_stream import AlpacaStreamer
from src.data_pipeline.market_hours import (
    is_market_open,
    is_market_open_alpaca,
    market_status,
    wait_for_market_open,
)
from src.data_pipeline.polling_feed import PollingFeed
from src.data_pipeline.realtime_buffer import RealtimeBuffer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ML + Quants Real-Time Market Data Feed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["alpaca", "polling"],
        default=None,
        help="Feed source ('alpaca' WebSocket or 'polling' yfinance fallback). Defaults to config.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Ticker symbols to stream (e.g. AAPL MSFT SPY).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Duration to run the feed in seconds before cleanly stopping (default: run indefinitely).",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default=None,
        help="Bar interval for polling (e.g. 1m, 5m).",
    )
    parser.add_argument(
        "--poll-freq",
        type=int,
        default=None,
        help="Seconds between polling rounds in polling mode.",
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=None,
        help="Seconds between automatic Parquet flushes.",
    )
    parser.add_argument(
        "--stale-threshold",
        type=int,
        default=None,
        help="Seconds of silence before logging a stale-data warning.",
    )
    parser.add_argument(
        "--ignore-market-hours",
        action="store_true",
        help="Bypass market hours check and stream/poll regardless of market status.",
    )
    parser.add_argument(
        "--on-closed",
        choices=["wait", "exit", "poll_recent"],
        default=None,
        help="Action if market is closed ('wait' for open, 'exit', or 'poll_recent' available data).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    stream_cfg = config.get("streaming", {})
    source = args.source or stream_cfg.get("source", "polling")
    tickers = args.tickers or stream_cfg.get("tickers", ["AAPL", "MSFT", "SPY"])
    interval = args.interval or stream_cfg.get("interval", "1m")
    poll_freq = args.poll_freq or stream_cfg.get("poll_frequency_sec", 10)
    flush_interval = args.flush_interval or stream_cfg.get("flush_interval_sec", 30)
    stale_thresh = args.stale_threshold or stream_cfg.get("stale_threshold_sec", 60)
    on_closed = args.on_closed or stream_cfg.get("on_market_closed", "poll_recent")
    feed_type = stream_cfg.get("feed", "iex")
    subscribe_to = tuple(stream_cfg.get("subscribe_to", ["bars"]))

    print("\n" + "=" * 65)
    print("  >> ML + QUANTS TRADING AGENT -- REAL-TIME FEED")
    print("=" * 65)
    print(f"  Source          : {source.upper()}")
    print(f"  Tickers         : {', '.join(tickers)}")
    print(f"  Duration        : {args.duration}s" if args.duration else "  Duration        : Indefinite (Ctrl+C to stop)")
    print(f"  Flush Interval  : Every {flush_interval}s")
    print(f"  Stale Threshold : {stale_thresh}s")
    print("=" * 65 + "\n")

    # 1. Market hours check
    status = market_status()
    print(f"  [*] Current ET Time : {status['current_time_et']}")
    print(f"  [*] Market Session  : {status['market_open']} -> {status['market_close']} ET")
    is_open = status["is_open"]

    # Alpaca authoritative check if available
    try:
        alpaca_open = is_market_open_alpaca()
        is_open = alpaca_open
    except Exception:
        pass

    if is_open:
        print("  [+] Market Status   : OPEN\n")
    else:
        print("  [-] Market Status   : CLOSED\n")
        if not args.ignore_market_hours:
            if on_closed == "exit":
                logger.info("Market is closed and on_closed=exit. Exiting.")
                print("  [!] Market is closed. Exiting as configured (--on-closed exit).")
                return 0
            elif on_closed == "wait":
                logger.info("Market is closed. Waiting for open...")
                print("  [*] Waiting for market to open...")
                wait_for_market_open(check_interval=60)
            elif on_closed == "poll_recent":
                print("  [*] Market is closed. Switching to polling recent available bars...\n")
                source = "polling"

    # 2. Initialize RealtimeBuffer
    buffer = RealtimeBuffer(
        tickers=tickers,
        flush_interval=flush_interval,
        stale_threshold=stale_thresh,
    )

    # 3. Launch Feed
    start_time = time.time()
    try:
        if source == "alpaca":
            print(f"  [*] Connecting to Alpaca WebSocket ({feed_type.upper()} feed)...")
            streamer = AlpacaStreamer(
                tickers=tickers,
                buffer=buffer,
                feed=feed_type,
                subscribe_to=subscribe_to,
            )
            streamer.run(duration=args.duration)
        else:
            print(f"  [*] Starting yfinance polling feed ({interval} bars, every {poll_freq}s)...")
            feed = PollingFeed(
                tickers=tickers,
                buffer=buffer,
                interval=interval,
                poll_interval=poll_freq,
            )
            feed.run(duration=args.duration)

    except KeyboardInterrupt:
        print("\n\n  [!] Feed interrupted by user.")
    except Exception as exc:
        logger.error("Feed error: %s", exc, exc_info=True)
        print(f"\n  [ERROR] Error running feed: {exc}")
        return 1

    # 4. Final summary and sanity checks
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"  == FEED SUMMARY ({total_elapsed:.1f}s elapsed) ==")
    print("=" * 65)
    buffer.print_stats()

    # Verify Parquet files on disk
    realtime_dir = buffer.data_dir
    print(f"\n  [*] Persisted Parquet Files in {realtime_dir}:")
    for ticker in tickers:
        p = realtime_dir / f"{ticker}.parquet"
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"    - {ticker:<6} -> {p.name} ({size_kb:.1f} KB)")
        else:
            print(f"    - {ticker:<6} -> (no file created)")
    print("=" * 65 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
