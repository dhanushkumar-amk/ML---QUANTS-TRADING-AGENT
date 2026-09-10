# ============================================================
# Unit Tests — Alpaca WebSocket Streamer (Phase 3)
# ============================================================

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data_pipeline.alpaca_stream import AlpacaStreamer
from src.data_pipeline.realtime_buffer import RealtimeBuffer


def test_alpaca_streamer_missing_credentials(monkeypatch, tmp_path: Path):
    """Verify ValueError when API keys are missing or placeholder."""
    monkeypatch.setenv("ALPACA_API_KEY", "your_alpaca_api_key_here")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "your_alpaca_secret_key_here")

    buf = RealtimeBuffer(["AAPL"], data_dir=tmp_path)
    with pytest.raises(ValueError, match="credentials missing or placeholder"):
        AlpacaStreamer(tickers=["AAPL"], buffer=buf)


def test_alpaca_streamer_init_valid(monkeypatch, tmp_path: Path):
    """Verify initialization succeeds when API keys are present."""
    monkeypatch.setenv("ALPACA_API_KEY", "valid_mock_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "valid_mock_secret")

    buf = RealtimeBuffer(["AAPL", "MSFT"], data_dir=tmp_path)
    streamer = AlpacaStreamer(
        tickers=["AAPL", "MSFT"],
        buffer=buf,
        feed="iex",
        subscribe_to=("bars", "trades"),
    )

    assert streamer.tickers == ["AAPL", "MSFT"]
    assert streamer.feed_name == "iex"
    assert "bars" in streamer.subscribe_to


def test_alpaca_handle_bar_object(monkeypatch, tmp_path: Path):
    """Test _handle_bar with a simulated Bar object."""
    monkeypatch.setenv("ALPACA_API_KEY", "valid_mock_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "valid_mock_secret")

    buf = RealtimeBuffer(["AAPL"], data_dir=tmp_path)
    streamer = AlpacaStreamer(tickers=["AAPL"], buffer=buf)

    mock_bar = MagicMock()
    mock_bar.symbol = "AAPL"
    mock_bar.timestamp = pd.Timestamp("2026-09-10 10:30:00")
    mock_bar.open = 150.0
    mock_bar.high = 152.0
    mock_bar.low = 149.5
    mock_bar.close = 151.5
    mock_bar.volume = 5000.0

    asyncio.run(streamer._handle_bar(mock_bar))
    assert streamer._bars_received == 1

    sizes = buf.get_buffer_sizes()
    assert sizes["AAPL"] == 1


def test_alpaca_handle_bar_dict(monkeypatch, tmp_path: Path):
    """Test _handle_bar with raw dictionary format."""
    monkeypatch.setenv("ALPACA_API_KEY", "valid_mock_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "valid_mock_secret")

    buf = RealtimeBuffer(["MSFT"], data_dir=tmp_path)
    streamer = AlpacaStreamer(tickers=["MSFT"], buffer=buf)

    bar_dict = {
        "S": "MSFT",
        "t": "2026-09-10T10:31:00Z",
        "o": 320.0,
        "h": 321.0,
        "l": 319.5,
        "c": 320.5,
        "v": 2500,
    }

    asyncio.run(streamer._handle_bar(bar_dict))
    assert streamer._bars_received == 1
    assert buf.get_buffer_sizes()["MSFT"] == 1


def test_alpaca_handle_trade_and_quote(monkeypatch, tmp_path: Path):
    """Test trade and quote tick handling."""
    monkeypatch.setenv("ALPACA_API_KEY", "valid_mock_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "valid_mock_secret")

    buf = RealtimeBuffer(["AAPL"], data_dir=tmp_path)
    streamer = AlpacaStreamer(tickers=["AAPL"], buffer=buf)

    trade = MagicMock(symbol="AAPL", timestamp="2026-09-10T10:32:00Z", price=151.0, size=100)
    asyncio.run(streamer._handle_trade(trade))
    assert streamer._trades_received == 1

    quote = MagicMock(
        symbol="AAPL", timestamp="2026-09-10T10:32:01Z", bid_price=150.9, ask_price=151.1
    )
    asyncio.run(streamer._handle_quote(quote))
    assert streamer._quotes_received == 1


def test_alpaca_streamer_stop(monkeypatch, tmp_path: Path):
    """Test that stop() sets event and stops internal stream."""
    monkeypatch.setenv("ALPACA_API_KEY", "valid_mock_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "valid_mock_secret")

    buf = RealtimeBuffer(["AAPL"], data_dir=tmp_path)
    streamer = AlpacaStreamer(tickers=["AAPL"], buffer=buf)

    mock_stream = MagicMock()
    streamer._stream = mock_stream
    streamer.stop()

    assert streamer._stop_event.is_set()
    mock_stream.stop.assert_called_once()
