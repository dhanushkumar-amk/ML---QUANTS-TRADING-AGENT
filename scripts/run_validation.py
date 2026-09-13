#!/usr/bin/env python
# ============================================================
# CLI: Run Automated Data Validation Gatekeeper
# ============================================================
"""
Runs data validation pipeline against existing datasets, outputs console
summary reports, and saves machine-readable JSON reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.data_access import get_data_access
from src.data_pipeline.storage import load_dataframe
from src.data_pipeline.validation_pipeline import ValidationPipeline, get_validation_trends
from src.data_pipeline.validation_rules import (
    ContinuityCheckRule,
    FreshnessCheckRule,
    RangeCheckRule,
    SchemaValidationRule,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Data Validation Gatekeeper.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "SPY"],
        help="Tickers to validate.",
    )
    parser.add_argument(
        "--max-lag-days",
        type=int,
        default=365 * 5,  # Historical datasets span back; set lag or evaluate rules
        help="Max acceptable lag in days for freshness check.",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="data/validation_reports",
        help="Directory to save JSON reports.",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    dal = get_data_access()
    pipeline = ValidationPipeline(
        rules=[
            SchemaValidationRule(),
            RangeCheckRule(),
            ContinuityCheckRule(),
            FreshnessCheckRule(max_lag_days=args.max_lag_days),
        ],
        raise_on_critical=False,
        reports_dir=reports_dir,
        save_reports=True,
    )

    print("=" * 88)
    print("  RUNNING DATA QUALITY GATEKEEPER VALIDATION (AAPL, MSFT, SPY)")
    print("=" * 88)

    generated_reports = []
    for ticker in args.tickers:
        try:
            df = dal.get_ohlcv(ticker)
        except Exception:
            # Fallback to load_dataframe
            try:
                df = load_dataframe(source="yfinance", name=ticker)
            except Exception as exc:
                print(f"  [!] Could not load dataset for {ticker}: {exc}")
                continue

        report = pipeline.validate(df, ticker=ticker)
        print(report.summary_table())
        generated_reports.append(report)

    # Trend summary
    trends = get_validation_trends(reports_dir=reports_dir)
    print("\n" + "=" * 88)
    print("  VALIDATION QUALITY TREND SUMMARY")
    print("=" * 88)
    print(f"  Total Reports Ingested : {trends['total_reports']}")
    print(f"  Passed / Valid Datasets: {trends['valid_count']}")
    print(f"  Critical Failures      : {trends['total_critical_failures']}")
    print(f"  Warnings Encountered   : {trends['total_warnings']}")
    print(f"  Failures by Rule       : {trends['failures_by_rule']}")
    print("=" * 88 + "\n")

    # Output JSON sample of first report
    if generated_reports:
        print("SAMPLE GENERATED JSON REPORT (AAPL):")
        print(json.dumps(generated_reports[0].to_dict(), indent=2))


if __name__ == "__main__":
    main()
