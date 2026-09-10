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
    return parser.parse_args()


def fetch_yfinance(cfg: dict, tickers_override: list[str] | None) -> None:
    """Fetch OHLCV data via yfinance, save to Parquet, print summary."""
    data_cfg = cfg["data"]
    loader = YFinanceLoader(data_cfg)

    tickers = tickers_override or data_cfg.get("tickers", [])
    logger.info("=== yfinance fetch: %s ===", tickers)

    results = loader.fetch_batch(tickers=tickers)

    # Save each ticker
    for ticker, df in results.items():
        save_dataframe(df, source="yfinance", name=ticker)

    # Summary
    summary = build_summary_table(results, source="yfinance")
    print_summary(summary, title="yfinance Fetch Summary")


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
        fetch_yfinance(cfg, args.tickers)

    if args.source in ("huggingface", "all"):
        fetch_huggingface(cfg)

    logger.info("✅ Data fetch complete.")


if __name__ == "__main__":
    main()
