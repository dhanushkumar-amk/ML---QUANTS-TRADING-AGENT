# ============================================================
# Market Calendar Utility — NYSE Trading Sessions & Holidays
# ============================================================
"""
Provides authoritative market calendar schedules for US exchanges (NYSE / NASDAQ)
using `pandas_market_calendars`.

Used by data cleaners to differentiate between expected market closures
(weekends, holidays) and genuine missing data gaps.

Usage:
    from src.data_pipeline.market_calendar import MarketCalendar

    cal = MarketCalendar("NYSE")
    trading_days = cal.get_trading_days("2026-01-01", "2026-09-01")
    missing_days = cal.find_missing_trading_days(df["date"])
"""

from __future__ import annotations

import datetime
from typing import Any, Sequence

import pandas as pd
import pandas_market_calendars as mcal

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MarketCalendar:
    """Wrapper around exchange market calendars to query valid trading sessions."""

    def __init__(self, exchange: str = "NYSE") -> None:
        self.exchange = exchange.upper()
        try:
            self._cal = mcal.get_calendar(self.exchange)
        except Exception as exc:
            logger.error("Failed to load exchange calendar '%s': %s", exchange, exc)
            raise

    def get_trading_days(
        self,
        start_date: str | datetime.date | pd.Timestamp,
        end_date: str | datetime.date | pd.Timestamp,
    ) -> pd.DatetimeIndex:
        """Get all valid exchange trading days between start_date and end_date (inclusive).

        Returns normalized datetime64 (midnight).
        """
        start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")

        schedule = self._cal.schedule(start_date=start_str, end_date=end_str)
        if schedule.empty:
            return pd.DatetimeIndex([])

        # Convert index of schedule to normalized DatetimeIndex
        days = pd.to_datetime(schedule.index).tz_localize(None).normalize()
        return days

    def is_trading_day(self, date: str | datetime.date | pd.Timestamp) -> bool:
        """Check whether a specific date is a valid trading session."""
        target_date = pd.to_datetime(date).normalize()
        days = self.get_trading_days(target_date, target_date)
        return len(days) > 0

    def get_holidays(
        self,
        start_date: str | datetime.date | pd.Timestamp,
        end_date: str | datetime.date | pd.Timestamp,
    ) -> list[datetime.date]:
        """Return weekday dates within the range where the exchange was closed for a holiday."""
        start_dt = pd.to_datetime(start_date).normalize()
        end_dt = pd.to_datetime(end_date).normalize()

        # All calendar weekdays (Mon-Fri)
        all_weekdays = pd.date_range(start_dt, end_dt, freq="B")
        trading_days = set(self.get_trading_days(start_dt, end_dt).date)

        holidays = [d.date() for d in all_weekdays if d.date() not in trading_days]
        return holidays

    def find_missing_trading_days(
        self,
        actual_dates: Sequence[Any],
        start_date: str | datetime.date | pd.Timestamp | None = None,
        end_date: str | datetime.date | pd.Timestamp | None = None,
    ) -> list[datetime.date]:
        """Identify missing trading sessions by comparing actual dates against expected schedule.

        Weekends and exchange holidays are automatically excluded from the missing list.
        """
        if len(actual_dates) == 0:
            if start_date and end_date:
                return list(self.get_trading_days(start_date, end_date).date)
            return []

        ts_series = pd.to_datetime(list(actual_dates)).tz_localize(None).normalize()
        min_date = start_date or ts_series.min()
        max_date = end_date or ts_series.max()

        expected_days = self.get_trading_days(min_date, max_date)
        actual_set = set(ts_series.date)

        missing = [d.date() for d in expected_days if d.date() not in actual_set]
        return missing
