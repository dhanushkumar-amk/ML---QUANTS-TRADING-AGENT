# src.data_pipeline — data ingestion, cleaning, storage
"""Data pipeline: fetching, cleaning, and persisting market data."""

from src.data_pipeline.historical_loader import YFinanceLoader
from src.data_pipeline.huggingface_loader import HuggingFaceLoader
from src.data_pipeline.alphavantage_loader import AlphaVantageLoader
from src.data_pipeline.storage import save_dataframe, load_dataframe, load_metadata
from src.data_pipeline.validators import validate_ohlcv, build_summary_table, print_summary
from src.data_pipeline.market_hours import (
    is_market_open,
    market_status,
    is_market_open_alpaca,
    wait_for_market_open,
)
from src.data_pipeline.realtime_buffer import RealtimeBuffer
from src.data_pipeline.polling_feed import PollingFeed
from src.data_pipeline.alpaca_stream import AlpacaStreamer

__all__ = [
    "YFinanceLoader",
    "HuggingFaceLoader",
    "AlphaVantageLoader",
    "save_dataframe",
    "load_dataframe",
    "load_metadata",
    "validate_ohlcv",
    "build_summary_table",
    "print_summary",
    "is_market_open",
    "market_status",
    "is_market_open_alpaca",
    "wait_for_market_open",
    "RealtimeBuffer",
    "PollingFeed",
    "AlpacaStreamer",
]
