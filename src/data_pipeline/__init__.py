# src.data_pipeline — data ingestion, cleaning, storage
"""Data pipeline: fetching, cleaning, and persisting market data."""

from src.data_pipeline.alpaca_stream import AlpacaStreamer
from src.data_pipeline.alphavantage_loader import AlphaVantageLoader
from src.data_pipeline.corporate_actions import (
    CorporateActionsAdjuster,
    DividendEvent,
    SplitEvent,
)
from src.data_pipeline.data_access import DataAccessLayer, get_data_access
from src.data_pipeline.data_cleaner import DataCleaner, DataQualityReport
from src.data_pipeline.delisted_tickers import DelistedInfo, DelistedRegistry
from src.data_pipeline.historical_loader import YFinanceLoader
from src.data_pipeline.huggingface_loader import HuggingFaceLoader
from src.data_pipeline.market_calendar import MarketCalendar
from src.data_pipeline.market_hours import (
    is_market_open,
    is_market_open_alpaca,
    market_status,
    wait_for_market_open,
)
from src.data_pipeline.polling_feed import PollingFeed
from src.data_pipeline.realtime_buffer import RealtimeBuffer
from src.data_pipeline.storage import load_dataframe, load_metadata, save_dataframe
from src.data_pipeline.storage_backend import (
    ParquetBackend,
    StorageBackend,
    TimescaleDBBackend,
)
from src.data_pipeline.universe_builder import UniverseBuilder, UniverseConstituent
from src.data_pipeline.validators import build_summary_table, print_summary, validate_ohlcv

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
    "MarketCalendar",
    "CorporateActionsAdjuster",
    "SplitEvent",
    "DividendEvent",
    "DataCleaner",
    "DataQualityReport",
    "DelistedRegistry",
    "DelistedInfo",
    "UniverseBuilder",
    "UniverseConstituent",
    "StorageBackend",
    "ParquetBackend",
    "TimescaleDBBackend",
    "DataAccessLayer",
    "get_data_access",
]
