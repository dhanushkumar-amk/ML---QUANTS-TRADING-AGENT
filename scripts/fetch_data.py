#!/usr/bin/env python
# ============================================================
# CLI — Fetch Raw Data
# ============================================================
"""
Entry point for data acquisition.

Examples::

    # Fetch OHLCV from yfinance for configured tickers
    python scripts/fetch_data.py --source yfinance

    # Fetch from Hugging Face only
    python scripts/fetch_data.py --source huggingface

    # Fetch from all sources
    python scripts/fetch_data.py --source all

    # Override tickers (yfinance)
    python scripts/fetch_data.py --source yfinance --tickers AAPL MSFT SPY
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so ``src`` is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.historical_loader import YFinanceLoader
from src.data_pipeline.huggingface_loader import HuggingFaceLoader
from src.data_pipeline.storage import save_dataframe
from src.data_pipeline.validators import build_summary_table, print_summary
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch raw market / NLP data and save as Parquet.",
    )
    parser.add_argument(
        "--source",
        choices=["yfinance", "huggingface", "all"],
        default="all",
        help="Which data source(s) to pull from (default: all).",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Override tickers for yfinance (space-separated).",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Run corporate actions adjustment and data cleaning, saving to data/processed.",
    )
    parser.add_argument(
        "--clean-strategy",
        choices=["flag_only", "forward_fill", "drop"],
        default="flag_only",
        help="Remediation strategy for anomalies (default: flag_only).",
    )
    return parser.parse_args()


_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


def fetch_yfinance(
    cfg: dict,
    tickers_override: list[str] | None,
    clean: bool = False,
    clean_strategy: str = "flag_only",
) -> None:
    """Fetch OHLCV data via yfinance, save to Parquet, and optionally clean."""
    from src.data_pipeline.corporate_actions import CorporateActionsAdjuster
    from src.data_pipeline.data_cleaner import DataCleaner

    data_cfg = cfg["data"]
    loader = YFinanceLoader(data_cfg)

    tickers = tickers_override or data_cfg.get("tickers", [])
    logger.info("=== yfinance fetch: %s ===", tickers)

    results = loader.fetch_batch(tickers=tickers)

    # 1. Save raw data (immutable source of truth)
    for ticker, df in results.items():
        save_dataframe(df, source="yfinance", name=ticker)

    # Summary of raw fetch
    summary = build_summary_table(results, source="yfinance")
    print_summary(summary, title="yfinance Raw Ingestion Summary")

    # 2. Optionally clean and adjust corporate actions
    if clean:
        print("\n" + "=" * 65)
        print(f"  RUNNING DATA CLEANING PIPELINE (strategy={clean_strategy})")
        print("=" * 65)

        adjuster = CorporateActionsAdjuster()
        cleaner = DataCleaner(strategy=clean_strategy)

        for ticker, df in results.items():
            # A. Corporate Actions (Splits & Dividends)
            splits = adjuster.detect_splits(df, ticker=ticker)
            adjusted_df = adjuster.adjust_for_splits(df, splits)

            # B. Data Cleaning (Duplicates, Gaps, Outliers)
            cleaned_df, report = cleaner.clean(adjusted_df, ticker=ticker)
            print(report.summary_table())

            # C. Save cleaned data to /data/processed
            processed_path = save_dataframe(
                cleaned_df,
                source="yfinance",
                name=ticker,
                raw_dir=_PROCESSED_DIR,
            )
            print(f"  --> Cleaned dataset saved to: {processed_path}\n")


def fetch_huggingface(cfg: dict) -> None:
    """Fetch HF dataset(s), save to Parquet, print summary."""
    data_cfg = cfg.get("data", {})
    loader = HuggingFaceLoader(data_cfg)

    logger.info("=== Hugging Face fetch ===")
    df = loader.fetch()  # uses default from catalogue

    if df is not None:
        # Use the dataset key as the filename
        ds_key = data_cfg.get("hf_dataset", "twitter_financial_sentiment")
        save_dataframe(df, source="huggingface", name=ds_key)

        # Build a simple summary (HF datasets may not have date/price cols)
        summary = build_summary_table({ds_key: df}, source="huggingface")
        print_summary(summary, title="Hugging Face Fetch Summary")
    else:
        logger.error("Hugging Face fetch returned no data.")


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    if args.source in ("yfinance", "all"):
        fetch_yfinance(
            cfg,
            args.tickers,
            clean=args.clean,
            clean_strategy=args.clean_strategy,
        )

    if args.source in ("huggingface", "all"):
        fetch_huggingface(cfg)

    logger.info("Data fetch complete.")


if __name__ == "__main__":
    main()
