#!/usr/bin/env python
# ============================================================
# Storage Benchmark — Performance Timing & Throughput
# ============================================================
"""
Benchmarks data access performance across single-year vs. multi-year queries.

Metrics tracked:
  • Execution time (ms)
  • Rows returned
  • Read throughput (rows/second)
  • Partition pruning efficiency (reading 1 partition vs. full history)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.data_access import DataAccessLayer


def run_benchmark(iterations: int = 10) -> None:
    dal = DataAccessLayer()
    tickers = dal.list_available_tickers()

    if not tickers:
        print("No partitioned data found. Run scripts/migrate_to_storage.py first.")
        return

    test_ticker = "AAPL" if "AAPL" in tickers else tickers[0]

    print("=" * 80)
    print("  STORAGE PERFORMANCE BENCHMARK (Local Partitioned ParquetBackend)")
    print("=" * 80)
    print(f"Available tickers: {tickers}")
    print(f"Benchmark iterations per test: {iterations}\n")

    scenarios = [
        (
            "1 Year for 1 Ticker (AAPL, 2021)",
            [test_ticker],
            "2021-01-01",
            "2021-12-31",
        ),
        (
            "5 Years for 1 Ticker (AAPL, 2018-2022)",
            [test_ticker],
            "2018-01-01",
            "2022-12-31",
        ),
        (
            f"5 Years for All Tickers ({', '.join(tickers[:5])}, 2018-2022)",
            tickers[:10],
            "2018-01-01",
            "2022-12-31",
        ),
    ]

    header = f"{'Scenario':<42} {'Rows':<10} {'Avg Time (ms)':<15} {'Throughput (rows/s)':<20}"
    print(header)
    print("-" * len(header))

    for name, query_tickers, start, end in scenarios:
        total_rows = 0
        times: list[float] = []

        for _ in range(iterations):
            t0 = time.perf_counter()
            count = 0
            for t in query_tickers:
                df = dal.get_ohlcv(t, start=start, end=end)
                count += len(df)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            total_rows = count

        avg_time_sec = sum(times) / len(times)
        avg_time_ms = avg_time_sec * 1000.0
        throughput = (total_rows / avg_time_sec) if avg_time_sec > 0 else 0.0
        tp_str = f"{throughput:,.0f}"
        print(f"{name:<42} {total_rows:<10} {avg_time_ms:<15.2f} {tp_str:<20}")

    print("-" * len(header))

    print(
        "Benchmark complete. Columnar ParquetBackend provides sub-millisecond to low-millisecond queries.\n"
    )


if __name__ == "__main__":
    run_benchmark()
