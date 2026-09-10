# ============================================================
# Market Hours Utility
# ============================================================
"""
Check whether US equity markets (NYSE / NASDAQ) are currently open.

Provides two methods:
  1. **Local calendar check** — timezone-based, no API calls, handles
     weekends but NOT exchange holidays.
  2. **Alpaca clock API** — authoritative source, requires credentials.

Usage:
    from src.data_pipeline.market_hours import is_market_open, market_status

    if is_market_open():
        start_feed()

    status = market_status()  # dict with is_open, next_open, next_close
"""

from __future__ import annotations

import datetime
from typing import Any
from zoneinfo import ZoneInfo

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ----- NYSE timezone & schedule -----------------------------------------
NYSE_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = datetime.time(9, 30)
MARKET_CLOSE = datetime.time(16, 0)

# Major US market holidays (static list — extend yearly as needed).
# Dates where NYSE is fully closed.
_HOLIDAYS_2026 = {
    datetime.date(2026, 1, 1),    # New Year's Day
    datetime.date(2026, 1, 19),   # MLK Day
    datetime.date(2026, 2, 16),   # Presidents' Day
    datetime.date(2026, 4, 3),    # Good Friday
    datetime.date(2026, 5, 25),   # Memorial Day
    datetime.date(2026, 7, 3),    # Independence Day (observed)
    datetime.date(2026, 9, 7),    # Labor Day
    datetime.date(2026, 11, 26),  # Thanksgiving
    datetime.date(2026, 12, 25),  # Christmas
}


def is_market_open(now: datetime.datetime | None = None) -> bool:
    """Check if US equity markets are currently open (simple calendar).

    Parameters
    ----------
    now : datetime.datetime | None
        Override the current time (for testing). Must be tz-aware or
        will be treated as UTC.

    Returns
    -------
    bool
    """
    if now is None:
        now = datetime.datetime.now(NYSE_TZ)
    else:
        now = now.astimezone(NYSE_TZ)

    # Weekend
    if now.weekday() >= 5:
        return False

    # Holiday
    if now.date() in _HOLIDAYS_2026:
        return False

    # Regular session
    return MARKET_OPEN <= now.time() < MARKET_CLOSE


def market_status(now: datetime.datetime | None = None) -> dict[str, Any]:
    """Return a dict with market status info.

    Keys: is_open, current_time_et, market_open, market_close
    """
    if now is None:
        now = datetime.datetime.now(NYSE_TZ)
    else:
        now = now.astimezone(NYSE_TZ)

    return {
        "is_open": is_market_open(now),
        "current_time_et": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "market_open": str(MARKET_OPEN),
        "market_close": str(MARKET_CLOSE),
        "weekday": now.strftime("%A"),
    }


def is_market_open_alpaca() -> bool:
    """Check market status via Alpaca's clock endpoint (authoritative).

    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in .env.
    Falls back to the local calendar check on failure.
    """
    try:
        from alpaca.trading.client import TradingClient
        from src.utils.config_loader import get_env

        api_key = get_env("ALPACA_API_KEY", "")
        secret_key = get_env("ALPACA_SECRET_KEY", "")

        if not api_key or api_key.startswith("your_"):
            logger.debug("No valid Alpaca credentials — using local calendar.")
            return is_market_open()

        client = TradingClient(api_key, secret_key, paper=True)
        clock = client.get_clock()
        return clock.is_open  # type: ignore[return-value]

    except Exception as exc:
        logger.warning("Alpaca clock check failed (%s) — falling back to local.", exc)
        return is_market_open()


def wait_for_market_open(check_interval: int = 60) -> None:
    """Block until the market opens (useful for overnight scripts).

    Parameters
    ----------
    check_interval : int
        Seconds between re-checks.
    """
    import time

    while not is_market_open():
        status = market_status()
        logger.info(
            "Market closed (%s, %s). Next check in %ds …",
            status["weekday"],
            status["current_time_et"],
            check_interval,
        )
        time.sleep(check_interval)

    logger.info("Market is OPEN — proceeding.")
