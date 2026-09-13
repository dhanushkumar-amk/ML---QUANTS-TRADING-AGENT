#!/usr/bin/env python
# ============================================================
# Migration Script — Migrate Existing Data to Partitioned Storage
# ============================================================
"""
One-time migration script that takes historical data from:
    data/processed/yfinance/ (or data/raw/yfinance/)
and migrates it into the new partitioned storage backend:
    data/processed/{ticker}/{year}.parquet

Performs strict row-count verification before and after migration to ensure zero data loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.data_pipeline.storage_backend import ParquetBackend
from src.utils.logger import get_logger

logger = get_logger(__name__)

_RAW_YF_DIR = _PROJECT_ROOT / "data" / "raw" / "yfinance"
_PROCESSED_YF_DIR = _PROJECT_ROOT / "data" / "processed" / "yfinance"
_PARTITIONED_DIR = _PROJECT_ROOT / "data" / "processed"


def migrate() -> dict[str, dict[str, int | str]]:
    """Run data migration into partitioned storage backend with row count verification."""
    backend = ParquetBackend(root_dir=_PARTITIONED_DIR)

    # 1. Discover source Parquet files
    source_files: dict[str, Path] = {}

    # Prefer processed/yfinance, fall back to raw/yfinance
    if _PROCESSED_YF_DIR.exists():
        for p in _PROCESSED_YF_DIR.glob("*.parquet"):
            source_files[p.stem.upper()] = p

    if _RAW_YF_DIR.exists():
        for p in _RAW_YF_DIR.glob("*.parquet"):
            ticker = p.stem.upper()
            if ticker not in source_files:
                source_files[ticker] = p

    if not source_files:
        print("No source Parquet files found to migrate.")
        return {}

    print("=" * 75)
    print("  MIGRATING DATA TO PARTITIONED STORAGE (Ticker + Year Layout)")
    print("=" * 75)
    print(f"Target directory: {_PARTITIONED_DIR}\n")

    results: dict[str, dict[str, int | str]] = {}

    for ticker, filepath in sorted(source_files.items()):
        print(f"--> Migrating {ticker} from {filepath.name} ...")
        df_src = pd.read_parquet(filepath, engine="pyarrow")
        orig_count = len(df_src)

        # Write to partitioned backend
        written_files = backend.save(df_src, key=ticker, date_col="date", overwrite=True)

        # Verify by reloading from partitioned backend
        df_loaded = backend.load(key=ticker)
        migrated_count = len(df_loaded)

        match = orig_count == migrated_count
        status = "MATCH (0 Loss)" if match else "MISMATCH"

        results[ticker] = {
            "orig_rows": orig_count,
            "migrated_rows": migrated_count,
            "partitions": len(written_files),
            "status": status,
        }

    # Summary table
    print("\n" + "=" * 75)
    print("  DATA MIGRATION & ROW-COUNT INTEGRITY VERIFICATION")
    print("=" * 75)
    header = f"{'Ticker':<8} {'Original Rows':<16} {'Migrated Rows':<16} {'Partitions':<12} {'Status':<16}"
    print(header)
    print("-" * len(header))

    all_matched = True
    for ticker, res in sorted(results.items()):
        if res["orig_rows"] != res["migrated_rows"]:
            all_matched = False
        print(
            f"{ticker:<8} {res['orig_rows']:<16} {res['migrated_rows']:<16} "
            f"{res['partitions']:<12} {res['status']:<16}"
        )

    print("-" * len(header))
    if all_matched:
        print("  ALL TICKERS VERIFIED: 100% data integrity preserved with 0 row loss.\n")
    else:
        print("  WARNING: Row count mismatch detected during migration!\n")

    return results


if __name__ == "__main__":
    migrate()
