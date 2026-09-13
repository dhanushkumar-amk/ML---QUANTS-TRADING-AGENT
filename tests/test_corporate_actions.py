# ============================================================
# Unit Tests — Corporate Actions (Splits & Dividends) (Phase 4)
# ============================================================

import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_pipeline.corporate_actions import CorporateActionsAdjuster, SplitEvent


@pytest.fixture
def adjuster():
    return CorporateActionsAdjuster()


def test_detect_forward_split_2_to_1(adjuster):
    """Test heuristic detection of a 2:1 forward stock split."""
    dates = pd.date_range("2026-09-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            # Days 1-3 at ~$100, Day 4 drops to $50 (2:1 split)
            "open": [99.0, 100.0, 101.0, 50.0, 51.0, 50.5],
            "high": [101.0, 102.0, 102.0, 51.0, 52.0, 51.5],
            "low": [98.0, 99.0, 100.0, 49.0, 50.0, 49.5],
            "close": [100.0, 100.0, 100.0, 50.0, 51.0, 50.5],
            "volume": [1_000_000, 1_000_000, 1_000_000, 2_000_000, 2_000_000, 2_000_000],
        }
    )

    splits = adjuster.detect_splits(df, ticker="TEST", use_authoritative=False)
    assert len(splits) == 1
    split = splits[0]
    assert split.date == datetime.date(2026, 9, 4)
    assert split.ratio == 2.0
    assert split.ratio_str == "2:1"


def test_detect_forward_split_4_to_1(adjuster):
    """Test heuristic detection of a 4:1 forward stock split (like AAPL 2020)."""
    dates = pd.date_range("2020-08-27", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "close": [500.0, 499.0, 125.0, 126.0],  # 4:1 split on Day 3
            "open": [495.0, 500.0, 124.0, 125.0],
            "volume": [50_000_000, 50_000_000, 200_000_000, 200_000_000],
        }
    )

    splits = adjuster.detect_splits(df, ticker="AAPL", use_authoritative=False)
    assert len(splits) == 1
    assert splits[0].ratio == 4.0
    assert splits[0].ratio_str == "4:1"


def test_detect_reverse_split_1_to_10(adjuster):
    """Test heuristic detection of a 1:10 reverse stock split."""
    dates = pd.date_range("2026-09-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            # Jumps from $2.00 to $20.00 (1:10 reverse split)
            "close": [2.00, 2.05, 20.00, 20.50],
            "volume": [10_000_000, 10_000_000, 1_000_000, 1_000_000],
        }
    )

    splits = adjuster.detect_splits(df, ticker="REVERSE", use_authoritative=False)
    assert len(splits) == 1
    split = splits[0]
    assert split.ratio == 0.1
    assert split.ratio_str == "1:10"


def test_adjust_for_splits(adjuster):
    """Test retroactive adjustment of historical prices and volumes."""
    dates = pd.date_range("2026-09-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": [200.0, 200.0, 100.0, 100.0],
            "high": [205.0, 205.0, 105.0, 105.0],
            "low": [195.0, 195.0, 95.0, 95.0],
            "close": [200.0, 200.0, 100.0, 100.0],
            "volume": [1_000_000, 1_000_000, 2_000_000, 2_000_000],
        }
    )

    split = SplitEvent(
        ticker="TEST",
        date=datetime.date(2026, 9, 3),
        ratio=2.0,
        ratio_str="2:1",
        source="test",
    )

    df_adj = adjuster.adjust_for_splits(df, [split])

    # Pre-split prices should be divided by 2.0 (200 -> 100)
    assert df_adj.loc[0, "close"] == 100.0
    assert df_adj.loc[1, "close"] == 100.0
    # Post-split prices remain unchanged
    assert df_adj.loc[2, "close"] == 100.0
    assert df_adj.loc[3, "close"] == 100.0

    # Pre-split volume should be multiplied by 2.0 (1M -> 2M)
    assert df_adj.loc[0, "volume"] == 2_000_000
    assert df_adj.loc[1, "volume"] == 2_000_000
    assert df_adj.loc[2, "volume"] == 2_000_000


def test_reconcile_dividends(adjuster):
    """Test dividend factor calculation and ex-dividend event detection."""
    dates = pd.date_range("2026-09-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "close": [100.0, 100.0, 100.0, 100.0],
            # Day 3 pays a dividend: prior adjusted closes are discounted
            "adj_close": [98.0, 98.0, 100.0, 100.0],
        }
    )

    df_res, events = adjuster.reconcile_dividends(df, ticker="DIV_TEST")
    assert "dividend_factor" in df_res.columns
    assert df_res.loc[0, "dividend_factor"] == 0.98
    assert df_res.loc[2, "dividend_factor"] == 1.0

    assert len(events) == 1
    event = events[0]
    assert event.date == datetime.date(2026, 9, 3)
    assert event.estimated_dividend == 2.0


def test_authoritative_splits_mock(adjuster):
    """Test yfinance splits fetching with mocked Ticker object."""
    mock_splits = pd.Series(
        [4.0],
        index=[pd.Timestamp("2020-08-31 09:30:00-04:00")],
        name="Stock Splits",
    )

    mock_ticker = MagicMock()
    mock_ticker.splits = mock_splits

    with patch("yfinance.Ticker", return_value=mock_ticker):
        splits = adjuster.fetch_authoritative_splits("AAPL")
        assert len(splits) == 1
        assert splits[0].ratio == 4.0
        assert splits[0].ratio_str == "4:1"
        assert splits[0].date == datetime.date(2020, 8, 31)


def test_adjust_for_splits_already_adjusted(adjuster):
    """Test that if price series is already split-adjusted, adjustment is skipped."""
    dates = pd.date_range("2020-08-28", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": [124.0, 124.5, 125.0, 126.0],
            "high": [125.0, 125.5, 126.0, 127.0],
            "low": [123.0, 123.5, 124.0, 125.0],
            "close": [124.0, 124.5, 125.0, 126.0],
            "volume": [50_000_000] * 4,
        }
    )

    split = SplitEvent(
        ticker="AAPL",
        date=datetime.date(2020, 8, 30),
        ratio=4.0,
        ratio_str="4:1",
        source="yfinance",
    )

    df_adj = adjuster.adjust_for_splits(df, [split], check_already_adjusted=True)

    # Pre-split prices should remain ~124, NOT divided by 4 to 31
    assert df_adj.loc[0, "close"] == 124.0
    assert df_adj.loc[1, "close"] == 124.5
