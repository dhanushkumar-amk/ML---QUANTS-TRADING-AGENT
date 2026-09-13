# ============================================================
# Unit Tests — Historical Data Loader (Phase 2)
# ============================================================

import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from src.data_pipeline.historical_loader import YFinanceLoader
from src.data_pipeline.validators import validate_ohlcv


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create a mock raw DataFrame matching yfinance output structure."""
    dates = pd.date_range("2026-09-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "Open": [150.0, 151.0, 152.0, 153.0, 154.0],
            "High": [152.0, 153.0, 154.0, 155.0, 156.0],
            "Low": [149.0, 150.0, 151.0, 152.0, 153.0],
            "Close": [151.0, 152.0, 153.0, 154.0, 155.0],
            "Volume": [100000, 110000, 120000, 130000, 140000],
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def test_loader_init():
    cfg = {"tickers": ["AAPL", "MSFT"], "start_date": "2026-01-01", "end_date": "2026-09-01"}
    loader = YFinanceLoader(cfg)
    assert loader.tickers == ["AAPL", "MSFT"]
    assert loader.start == "2026-01-01"
    assert loader.end == "2026-09-01"


@patch("yfinance.download")
def test_fetch_success(mock_download, sample_ohlcv_df):
    mock_download.return_value = sample_ohlcv_df.copy()

    loader = YFinanceLoader({"retries": 1})
    df = loader.fetch("AAPL")

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "date" in df.columns
    assert "ticker" in df.columns
    assert "source" in df.columns
    assert df["ticker"].iloc[0] == "AAPL"
    assert df["source"].iloc[0] == "yfinance"

    # Verify validation passes (no issues returned)
    issues = validate_ohlcv(df, "AAPL")
    assert issues == []


@patch("yfinance.download")
def test_fetch_multiindex_columns(mock_download):
    """Test handling of MultiIndex columns emitted by some yfinance versions."""
    dates = pd.date_range("2026-09-01", periods=3, freq="D")
    tuples = [
        ("Open", "AAPL"),
        ("High", "AAPL"),
        ("Low", "AAPL"),
        ("Close", "AAPL"),
        ("Volume", "AAPL"),
    ]
    index = pd.MultiIndex.from_tuples(tuples)
    mock_df = pd.DataFrame([[150, 155, 149, 152, 50000]] * 3, index=dates, columns=index)
    mock_download.return_value = mock_df

    loader = YFinanceLoader({"retries": 1})
    df = loader.fetch("AAPL")

    assert df is not None
    assert "close" in df.columns
    assert "open" in df.columns
    assert len(df) == 3


@patch("yfinance.download")
def test_fetch_empty_dataframe(mock_download):
    mock_download.return_value = pd.DataFrame()

    loader = YFinanceLoader({"retries": 1})
    df = loader.fetch("INVALID")
    assert df is None


@patch("yfinance.download")
def test_fetch_retry_on_exception(mock_download, sample_ohlcv_df):
    # First attempt raises ConnectionError, second succeeds
    mock_download.side_effect = [ConnectionError("Timeout"), sample_ohlcv_df.copy()]

    loader = YFinanceLoader({"retries": 2, "backoff_base": 0.01})
    df = loader.fetch("AAPL")

    assert df is not None
    assert len(df) == 5
    assert mock_download.call_count == 2


@patch("yfinance.download")
def test_fetch_exhaust_all_retries(mock_download):
    mock_download.side_effect = RuntimeError("Service Unavailable")

    loader = YFinanceLoader({"retries": 2, "backoff_base": 0.01})
    df = loader.fetch("FAIL")

    assert df is None
    assert mock_download.call_count == 2


@patch("yfinance.download")
def test_fetch_batch(mock_download, sample_ohlcv_df):
    mock_download.return_value = sample_ohlcv_df.copy()

    loader = YFinanceLoader({"tickers": ["AAPL", "MSFT"], "retries": 1})
    batch = loader.fetch_batch(["AAPL", "MSFT"])

    assert isinstance(batch, dict)
    assert "AAPL" in batch
    assert "MSFT" in batch
    assert len(batch["AAPL"]) == 5


@patch("yfinance.download")
def test_fetch_universe(mock_download, sample_ohlcv_df):
    from src.data_pipeline.universe_builder import UniverseConstituent

    mock_download.return_value = sample_ohlcv_df.copy()

    loader = YFinanceLoader({"retries": 1})
    constituents = [
        UniverseConstituent(ticker="AAPL", name="Apple Inc."),
        UniverseConstituent(ticker="MSFT", name="Microsoft Corp"),
    ]

    results = loader.fetch_universe(constituents)
    assert "AAPL" in results
    assert "MSFT" in results
    assert len(results["AAPL"]) == 5


@patch("yfinance.download")
def test_fetch_universe_delisted_handling(mock_download, sample_ohlcv_df):
    from src.data_pipeline.universe_builder import UniverseConstituent

    # Return valid df for AAPL, empty df for delisted ticker
    def side_effect(ticker, **kwargs):
        if ticker == "AAPL":
            return sample_ohlcv_df.copy()
        return pd.DataFrame()

    mock_download.side_effect = side_effect

    loader = YFinanceLoader({"retries": 1})
    constituents = [
        UniverseConstituent(ticker="AAPL", name="Apple Inc."),
        UniverseConstituent(
            ticker="YHOO",
            name="Yahoo! Inc.",
            is_delisted=True,
            delisting_date=datetime.date(2017, 6, 19),
            delisting_reason="acquired",
        ),
    ]

    results = loader.fetch_universe(constituents)
    assert "AAPL" in results
    assert "YHOO" not in results  # Unavailable in yfinance, gracefully logged
