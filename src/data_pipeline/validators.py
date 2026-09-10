# ============================================================
# Sanity Checks — post-fetch data validation
# ============================================================
"""
Run basic integrity checks on fetched DataFrames:
  • No duplicate timestamps (per ticker)
  • No negative prices or volumes
  • Summary statistics table

Usage:
    from src.data_pipeline.validators import validate_ohlcv, print_summary

    issues = validate_ohlcv(df, ticker="AAPL")
    print_summary(results)
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns expected in OHLCV data
_PRICE_COLS = ["open", "high", "low", "close"]
_VOLUME_COL = "volume"


def validate_ohlcv(df: pd.DataFrame, ticker: str = "?") -> list[str]:
    """Run sanity checks on an OHLCV DataFrame.

    Returns a list of human-readable issue strings (empty = all good).
    """
    issues: list[str] = []

    if df.empty:
        issues.append(f"[{ticker}] DataFrame is empty.")
        return issues

    # 1. Duplicate timestamps
    if "date" in df.columns:
        dupes = df["date"].duplicated().sum()
        if dupes > 0:
            issues.append(f"[{ticker}] {dupes} duplicate timestamps found.")
            logger.warning("%s: %d duplicate timestamps", ticker, dupes)
        else:
            logger.info("%s: no duplicate timestamps ✓", ticker)

    # 2. Negative prices
    for col in _PRICE_COLS:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                issues.append(f"[{ticker}] {neg} negative values in '{col}'.")
                logger.warning("%s: %d negative values in %s", ticker, neg, col)

    # 3. Negative volume
    if _VOLUME_COL in df.columns:
        neg_vol = (df[_VOLUME_COL] < 0).sum()
        if neg_vol > 0:
            issues.append(
                f"[{ticker}] {neg_vol} negative values in 'volume'."
            )
            logger.warning("%s: %d negative volumes", ticker, neg_vol)

    if not issues:
        logger.info("%s: all sanity checks passed ✓", ticker)

    return issues


def build_summary_table(
    results: dict[str, pd.DataFrame],
    source: str = "yfinance",
) -> pd.DataFrame:
    """Build a summary table from a {name: df} mapping.

    Columns: name, source, rows, columns, date_min, date_max, issues
    """
    rows: list[dict] = []

    for name, df in results.items():
        issues = validate_ohlcv(df, ticker=name)
        row: dict = {
            "name": name,
            "source": source,
            "rows": len(df),
            "columns": len(df.columns),
            "issues": "; ".join(issues) if issues else "none",
        }
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce").dropna()
            row["date_min"] = str(dates.min().date()) if not dates.empty else "N/A"
            row["date_max"] = str(dates.max().date()) if not dates.empty else "N/A"
        else:
            row["date_min"] = "N/A"
            row["date_max"] = "N/A"

        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary


def print_summary(summary_df: pd.DataFrame, title: str = "Fetch Summary") -> None:
    """Pretty-print a summary DataFrame to the console."""
    border = "=" * 72
    print(f"\n{border}")
    print(f"  {title}")
    print(border)
    print(summary_df.to_string(index=False))
    print(f"{border}\n")
