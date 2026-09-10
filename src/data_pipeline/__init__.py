# src.data_pipeline — data ingestion, cleaning, storage
"""Data pipeline: fetching, cleaning, and persisting market data."""

from src.data_pipeline.historical_loader import YFinanceLoader
from src.data_pipeline.huggingface_loader import HuggingFaceLoader
from src.data_pipeline.alphavantage_loader import AlphaVantageLoader
from src.data_pipeline.storage import save_dataframe, load_dataframe, load_metadata
from src.data_pipeline.validators import validate_ohlcv, build_summary_table, print_summary

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
]
