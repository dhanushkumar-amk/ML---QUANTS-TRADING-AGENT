# ============================================================
# Unit Tests — Real-Time Market Data Pipeline (Phase 3)
# ============================================================

import datetime
from pathlib import Path

import pandas as pd

from src.data_pipeline.market_hours import NYSE_TZ, is_market_open, market_status
from src.data_pipeline.realtime_buffer import RealtimeBuffer


def test_market_hours_open_and_closed():
    # Wednesday at 11:00 AM ET (open)
    open_dt = datetime.datetime(2026, 9, 9, 11, 0, 0, tzinfo=NYSE_TZ)
    assert is_market_open(open_dt) is True

    # Wednesday at 8:00 AM ET (pre-market -> regular session closed)
    pre_dt = datetime.datetime(2026, 9, 9, 8, 0, 0, tzinfo=NYSE_TZ)
    assert is_market_open(pre_dt) is False

    # Wednesday at 5:00 PM ET (after-hours -> regular session closed)
    after_dt = datetime.datetime(2026, 9, 9, 17, 0, 0, tzinfo=NYSE_TZ)
    assert is_market_open(after_dt) is False

    # Saturday (weekend -> closed)
    weekend_dt = datetime.datetime(2026, 9, 12, 12, 0, 0, tzinfo=NYSE_TZ)
    assert is_market_open(weekend_dt) is False

    # Labor Day 2026 (holiday -> closed)
    holiday_dt = datetime.datetime(2026, 9, 7, 12, 0, 0, tzinfo=NYSE_TZ)
    assert is_market_open(holiday_dt) is False


def test_market_status_dict():
    status = market_status()
    assert "is_open" in status
    assert "current_time_et" in status
    assert "market_open" in status
    assert "market_close" in status
    assert "weekday" in status


def test_realtime_buffer_deduplication(tmp_path: Path):
    buffer = RealtimeBuffer(
        tickers=["AAPL"],
        data_dir=tmp_path,
        flush_interval=1,
    )

    now = pd.Timestamp("2026-09-10 10:00:00")
    bar1 = {
        "ticker": "AAPL",
        "timestamp": now,
        "open": 150.0,
        "high": 151.0,
        "low": 149.5,
        "close": 150.5,
        "volume": 1000.0,
    }

    # First append should succeed
    assert buffer.append("AAPL", bar1) is True

    # Flush to parquet
    buffer.flush_all()
    parquet_path = tmp_path / "AAPL.parquet"
    assert parquet_path.exists()

    df = pd.read_parquet(parquet_path)
    assert len(df) == 1
    assert df.iloc[0]["close"] == 150.5

    # Simulate restart with fresh buffer reading from disk
    new_buffer = RealtimeBuffer(
        tickers=["AAPL"],
        data_dir=tmp_path,
    )

    # Appending same timestamp should be rejected as duplicate
    assert new_buffer.append("AAPL", bar1) is False

    # Appending older timestamp should also be rejected
    older_bar = dict(bar1, timestamp=pd.Timestamp("2026-09-10 09:59:00"))
    assert new_buffer.append("AAPL", older_bar) is False

    # Appending newer timestamp should succeed
    newer_bar = dict(bar1, timestamp=pd.Timestamp("2026-09-10 10:01:00"), close=152.0)
    assert new_buffer.append("AAPL", newer_bar) is True

    new_buffer.flush_all()
    df2 = pd.read_parquet(parquet_path)
    assert len(df2) == 2
    assert df2.iloc[1]["close"] == 152.0
