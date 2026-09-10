# ============================================================
# Unit Tests — Polling Feed & Market Hours Advanced (Phase 3)
# ============================================================

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_pipeline.market_hours import is_market_open_alpaca
from src.data_pipeline.polling_feed import PollingFeed
from src.data_pipeline.realtime_buffer import RealtimeBuffer


@pytest.fixture
def mock_intraday_df() -> pd.DataFrame:
    dates = pd.date_range("2026-09-10 10:00:00", periods=5, freq="min")
    df = pd.DataFrame(
        {
            "Open": [150.0, 150.2, 150.5, 150.3, 150.6],
            "High": [150.5, 150.6, 150.8, 150.7, 150.9],
            "Low": [149.8, 150.0, 150.3, 150.1, 150.4],
            "Close": [150.2, 150.5, 150.4, 150.6, 150.8],
            "Volume": [1000, 1500, 1200, 1300, 1600],
        },
        index=dates,
    )
    df.index.name = "Datetime"
    return df


def test_polling_feed_poll_ticker(tmp_path: Path, mock_intraday_df: pd.DataFrame):
    """Test polling a single ticker and appending to buffer."""
    buf = RealtimeBuffer(["AAPL"], data_dir=tmp_path)
    feed = PollingFeed(["AAPL"], buffer=buf, poll_interval=1)

    with patch("yfinance.download", return_value=mock_intraday_df):
        added = feed._poll_ticker("AAPL")
        assert added > 0
        assert buf.get_buffer_sizes()["AAPL"] == added


def test_polling_feed_poll_once(tmp_path: Path, mock_intraday_df: pd.DataFrame):
    """Test poll_once across multiple tickers."""
    buf = RealtimeBuffer(["AAPL", "MSFT"], data_dir=tmp_path)
    feed = PollingFeed(["AAPL", "MSFT"], buffer=buf, poll_interval=1)

    with patch("yfinance.download", return_value=mock_intraday_df):
        counts = feed.poll_once()
        assert "AAPL" in counts
        assert "MSFT" in counts
        assert counts["AAPL"] > 0
        assert counts["MSFT"] > 0


def test_polling_feed_run_duration(tmp_path: Path, mock_intraday_df: pd.DataFrame):
    """Test that run(duration=1) runs and exits cleanly."""
    buf = RealtimeBuffer(["AAPL"], data_dir=tmp_path)
    feed = PollingFeed(["AAPL"], buffer=buf, poll_interval=1)

    with patch("yfinance.download", return_value=mock_intraday_df):
        feed.run(duration=1)

    # Verify Parquet flushed
    assert (tmp_path / "AAPL.parquet").exists()


def test_is_market_open_alpaca_mocked(monkeypatch):
    """Test Alpaca clock API check with mocked TradingClient."""
    monkeypatch.setenv("ALPACA_API_KEY", "mock_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "mock_secret")

    mock_clock = MagicMock(is_open=True)
    mock_client = MagicMock()
    mock_client.get_clock.return_value = mock_clock

    with patch("alpaca.trading.client.TradingClient", return_value=mock_client):
        assert is_market_open_alpaca() is True


def test_is_market_open_alpaca_fallback_on_error(monkeypatch):
    """Test fallback to local calendar if Alpaca client raises exception."""
    monkeypatch.setenv("ALPACA_API_KEY", "mock_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "mock_secret")

    with patch("alpaca.trading.client.TradingClient", side_effect=Exception("API Error")):
        # Should not raise, falls back to local is_market_open()
        result = is_market_open_alpaca()
        assert isinstance(result, bool)
