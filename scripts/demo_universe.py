#!/usr/bin/env python
# ============================================================
# Demo — Survivorship-Bias-Free Point-in-Time Universe
# ============================================================
"""Demonstrates point-in-time universe reconstruction and survivorship bias prevention."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.delisted_tickers import DelistedRegistry
from src.data_pipeline.universe_builder import UniverseBuilder


def run_demo() -> None:
    builder = UniverseBuilder()
    registry = DelistedRegistry()

    # 1. Query point-in-time universe as of 2015-01-01
    constituents_2015 = builder.get_constituents("2015-01-01")
    const_2015_dict = {c.ticker: c for c in constituents_2015}

    # 2. Query point-in-time universe as of 2024-01-01
    constituents_2024 = builder.get_constituents("2024-01-01")
    const_2024_dict = {c.ticker: c for c in constituents_2024}

    print("=" * 80)
    print("  SURVIVORSHIP-BIAS FIX DEMONSTRATION: S&P 500 POINT-IN-TIME UNIVERSES")
    print("=" * 80)

    print(
        f"\n[1] Point-in-Time Universe as of 2015-01-01 (Total: {len(constituents_2015)} constituents):"
    )
    header = f"{'Ticker':<8} {'Name':<32} {'Delisted?':<12} {'Delisting Date':<16} {'Reason':<15}"
    print(header)
    print("-" * len(header))

    delisted_in_2015 = [c for c in constituents_2015 if c.is_delisted]
    for c in delisted_in_2015:
        d_date = str(c.delisting_date) if c.delisting_date else "N/A"
        reason = str(c.delisting_reason) if c.delisting_reason else "N/A"
        print(f"{c.ticker:<8} {c.name:<32} {str(c.is_delisted):<12} {d_date:<16} {reason:<15}")

    print("\n[2] Comparison for Specific Historical Tickers (2015-01-01 vs. 2024-01-01):")
    cmp_header = f"{'Ticker':<8} {'Name':<30} {'In 2015 Universe?':<20} {'In 2024 Universe?':<20}"
    print(cmp_header)
    print("-" * len(cmp_header))

    test_tickers = [
        ("YHOO", "Yahoo! Inc."),
        ("MON", "Monsanto Company"),
        ("CELG", "Celgene Corporation"),
        ("WFM", "Whole Foods Market Inc."),
        ("DOW", "The Dow Chemical Company"),
        ("COL", "Rockwell Collins Inc."),
        ("TWTR", "Twitter Inc."),
        ("TSLA", "Tesla Inc."),
        ("NOW", "ServiceNow Inc."),
        ("AAPL", "Apple Inc."),
        ("MSFT", "Microsoft Corporation"),
    ]

    for ticker, default_name in test_tickers:
        in_2015 = ticker in const_2015_dict
        in_2024 = ticker in const_2024_dict
        c = const_2015_dict.get(ticker) or const_2024_dict.get(ticker)
        name = c.name if c else default_name
        print(f"{ticker:<8} {name:<30} {str(in_2015):<20} {str(in_2024):<20}")

    print("\n[3] Delisted Registry Inspection for YHOO (Yahoo):")
    yhoo_info = registry.get_delisting_info("YHOO")
    if yhoo_info:
        print(f"  • Ticker          : {yhoo_info.ticker}")
        print(f"  • Name            : {yhoo_info.name}")
        print(f"  • Delisting Date  : {yhoo_info.delisting_date}")
        print(f"  • Reason          : {yhoo_info.reason}")
        print(f"  • Acquirer        : {yhoo_info.acquirer}")
        print(f"  • Last Price      : ${yhoo_info.last_price:.2f}")
        print(f"  • Is delisted on 2015-01-01? : {registry.is_delisted('YHOO', '2015-01-01')}")
        print(f"  • Is delisted on 2024-01-01? : {registry.is_delisted('YHOO', '2024-01-01')}")

    print("\n[4] Quant Takeaway:")
    print("  * A naive backtest querying today's S&P 500 would OMIT Yahoo, Monsanto, and Celgene")
    print("    from 2015, introducing severe survivorship bias.")
    print("  * The point-in-time UniverseBuilder restores them to the 2015 universe while")
    print("    excluding stocks added later (e.g. Tesla added Dec 2020).")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
