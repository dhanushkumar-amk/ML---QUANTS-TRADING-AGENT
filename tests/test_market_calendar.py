# ============================================================
# Unit Tests — Market Calendar Utility (Phase 4)
# ============================================================

import datetime

import pandas as pd
import pytest

from src.data_pipeline.market_calendar import MarketCalendar


@pytest.fixture
def nyse_calendar():
    return MarketCalendar("NYSE")


def test_market_calendar_init(nyse_calendar):
    assert nyse_calendar.exchange == "NYSE"


def test_regular_trading_day(nyse_calendar):
    # Wednesday, September 9, 2026 was a regular trading day
    assert nyse_calendar.is_trading_day("2026-09-09") is True


def test_weekend_excluded(nyse_calendar):
    # Saturday and Sunday
    assert nyse_calendar.is_trading_day("2026-09-12") is False
    assert nyse_calendar.is_trading_day("2026-09-13") is False


def test_known_holidays_excluded(nyse_calendar):
    # Independence Day observed (Friday, July 3, 2026)
    assert nyse_calendar.is_trading_day("2026-07-03") is False

    # Labor Day (Monday, September 7, 2026)
    assert nyse_calendar.is_trading_day("2026-09-07") is False

    # Christmas (Friday, December 25, 2026)
    assert nyse_calendar.is_trading_day("2026-12-25") is False

    # Check holiday list contains July 3rd
    holidays = nyse_calendar.get_holidays("2026-07-01", "2026-07-07")
    assert datetime.date(2026, 7, 3) in holidays


def test_find_missing_trading_days(nyse_calendar):
    # In a full trading week (Mon Aug 17 to Fri Aug 21, 2026): 5 sessions
    all_expected = nyse_calendar.get_trading_days("2026-08-17", "2026-08-21")
    assert len(all_expected) == 5

    # Simulate missing Wednesday Aug 19
    actual = [d for d in all_expected if d != pd.Timestamp("2026-08-19")]
    missing = nyse_calendar.find_missing_trading_days(actual, "2026-08-17", "2026-08-21")

    assert len(missing) == 1
    assert missing[0] == datetime.date(2026, 8, 19)


def test_find_missing_trading_days_empty_actual(nyse_calendar):
    missing = nyse_calendar.find_missing_trading_days([], "2026-08-17", "2026-08-21")
    assert len(missing) == 5
