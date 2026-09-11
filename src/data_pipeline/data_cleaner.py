# ============================================================
# Data Cleaning Engine — Outlier, Gap & Duplicate Resolution
# ============================================================
"""
Production-grade data sanitizer for raw market data:
  • Removes or flags duplicate timestamps (keeps first, logs duplicates).
  • Distinguishes true market gaps from weekends and exchange holidays.
  • Detects bad ticks: zero/negative prices, zero volume, and reverting price spikes.
  • Configurable remediation strategy: 'flag_only' (safe default), 'forward_fill', 'drop'.
  • Produces comprehensive DataQualityReport per ticker.

Usage:
    from src.data_pipeline.data_cleaner import DataCleaner

    cleaner = DataCleaner(strategy="flag_only")
    cleaned_df, report = cleaner.clean(df, ticker="AAPL")
    print(report.summary_table())
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data_pipeline.market_calendar import MarketCalendar
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataQualityReport:
    """Detailed summary of data quality metrics for a single ticker."""

    ticker: str
    total_raw_rows: int
    cleaned_rows: int
    duplicate_rows: int = 0
    missing_trading_days: list[datetime.date] = field(default_factory=list)
    zero_or_negative_prices: int = 0
    zero_volumes: int = 0
    reverting_spikes: int = 0
    strategy_used: str = "flag_only"

    @property
    def total_issues(self) -> int:
        return (
            self.duplicate_rows
            + len(self.missing_trading_days)
            + self.zero_or_negative_prices
            + self.zero_volumes
            + self.reverting_spikes
        )

    @property
    def affected_percentage(self) -> float:
        if self.total_raw_rows == 0:
            return 0.0
        return (self.total_issues / self.total_raw_rows) * 100.0

    def summary_table(self) -> str:
        """Render a formatted ASCII quality table."""
        border = "-" * 60
        lines = [
            f"\n{border}",
            f"  DATA QUALITY REPORT: {self.ticker}",
            border,
            f"  Total Raw Rows       : {self.total_raw_rows:>8}",
            f"  Cleaned Output Rows  : {self.cleaned_rows:>8}",
            f"  Duplicate Timestamps : {self.duplicate_rows:>8}",
            f"  Missing Trading Days : {len(self.missing_trading_days):>8}",
            f"  Zero/Negative Prices : {self.zero_or_negative_prices:>8}",
            f"  Zero Volume Bars     : {self.zero_volumes:>8}",
            f"  Reverting Spikes     : {self.reverting_spikes:>8}",
            border,
            f"  Total Issues Found   : {self.total_issues:>8}",
            f"  Data Affected        : {self.affected_percentage:>7.2f}%",
            f"  Remediation Strategy : {self.strategy_used:>8}",
            border,
        ]
        if self.missing_trading_days:
            missing_str = ", ".join(d.strftime("%Y-%m-%d") for d in self.missing_trading_days[:5])
            if len(self.missing_trading_days) > 5:
                missing_str += f" ... (+{len(self.missing_trading_days) - 5} more)"
            lines.append(f"  Missing Sessions     : {missing_str}")
            lines.append(border)

        return "\n".join(lines)


class DataCleaner:
    """Sanitizes raw market data and flags anomalies."""

    def __init__(
        self,
        strategy: str = "flag_only",  # 'flag_only' | 'forward_fill' | 'drop'
        spike_threshold: float = 0.15,  # 15% return spike
        spike_reversion_tolerance: float = 0.05,  # must revert within 5% of pre-spike price
        calendar_exchange: str = "NYSE",
    ) -> None:
        """
        Parameters
        ----------
        strategy : str
            Action to take on anomalies: 'flag_only', 'forward_fill', or 'drop'.
        spike_threshold : float
            Fractional price return threshold to identify transient spikes.
        spike_reversion_tolerance : float
            Maximum deviation from baseline price to classify as mean-reverting spike.
        calendar_exchange : str
            Market calendar name for exchange holiday / session checks.
        """
        self.strategy = strategy.lower()
        if self.strategy not in ("flag_only", "forward_fill", "drop"):
            raise ValueError(f"Invalid strategy '{strategy}'. Choose 'flag_only', 'forward_fill', or 'drop'.")

        self.spike_threshold = spike_threshold
        self.spike_reversion_tolerance = spike_reversion_tolerance
        self.calendar = MarketCalendar(calendar_exchange)

    def clean(
        self,
        df: pd.DataFrame,
        ticker: str = "UNKNOWN",
        date_col: str = "date",
        price_cols: tuple[str, ...] = ("open", "high", "low", "close"),
        volume_col: str = "volume",
    ) -> tuple[pd.DataFrame, DataQualityReport]:
        """Clean an OHLCV DataFrame according to configured strategy.

        Returns
        -------
        tuple[pd.DataFrame, DataQualityReport]
        """
        if df.empty:
            report = DataQualityReport(
                ticker=ticker,
                total_raw_rows=0,
                cleaned_rows=0,
                strategy_used=self.strategy,
            )
            return df.copy(), report

        df_work = df.copy()

        # Standardize date column to datetime
        if date_col in df_work.columns:
            df_work[date_col] = pd.to_datetime(df_work[date_col]).dt.tz_localize(None)
            df_work = df_work.sort_values(date_col).reset_index(drop=True)

        total_raw = len(df_work)

        # 1. Duplicate Timestamps
        dup_mask = df_work.duplicated(subset=[date_col], keep="first")
        dup_count = int(dup_mask.sum())
        if dup_count > 0:
            logger.warning("[%s] Found %d duplicate timestamps.", ticker, dup_count)

        # 2. Trading Day Gaps (Calendar Check)
        missing_days = []
        if date_col in df_work.columns and total_raw > 1:
            actual_dates = df_work[date_col].dt.date.values
            missing_days = self.calendar.find_missing_trading_days(
                actual_dates=actual_dates,
                start_date=actual_dates[0],
                end_date=actual_dates[-1],
            )
            if missing_days:
                logger.info(
                    "[%s] Found %d missing trading days against %s calendar.",
                    ticker,
                    len(missing_days),
                    self.calendar.exchange,
                )

        # 3. Negative / Zero Prices
        neg_price_mask = pd.Series(False, index=df_work.index)
        for col in price_cols:
            if col in df_work.columns:
                neg_price_mask |= df_work[col] <= 0
        neg_price_count = int(neg_price_mask.sum())

        # 4. Zero Volume on Trading Day
        zero_vol_mask = pd.Series(False, index=df_work.index)
        if volume_col in df_work.columns:
            zero_vol_mask = df_work[volume_col] <= 0
        zero_vol_count = int(zero_vol_mask.sum())

        # 5. Reverting Price Spikes (Bad Ticks)
        spike_mask = pd.Series(False, index=df_work.index)
        if "close" in df_work.columns and total_raw >= 3:
            closes = df_work["close"].values
            for i in range(1, len(closes) - 1):
                p_prev = closes[i - 1]
                p_curr = closes[i]
                p_next = closes[i + 1]

                if p_prev <= 0 or p_curr <= 0 or p_next <= 0:
                    continue

                # Return on bar i
                r1 = (p_curr - p_prev) / p_prev
                # Return on bar i+1
                r2 = (p_next - p_curr) / p_curr
                # Total deviation from baseline
                baseline_diff = abs(p_next - p_prev) / p_prev

                # Spike condition: sharp jump/drop and immediate reversal back near baseline
                if abs(r1) >= self.spike_threshold and (r1 * r2 < 0) and baseline_diff <= self.spike_reversion_tolerance:
                    spike_mask.iloc[i] = True
                    logger.warning(
                        "[%s] Detected reverting price spike on %s: %.2f -> %.2f -> %.2f (ret=%.1f%%)",
                        ticker,
                        df_work[date_col].iloc[i].strftime("%Y-%m-%d"),
                        p_prev,
                        p_curr,
                        p_next,
                        r1 * 100,
                    )

        spike_count = int(spike_mask.sum())

        report = DataQualityReport(
            ticker=ticker,
            total_raw_rows=total_raw,
            cleaned_rows=total_raw,
            duplicate_rows=dup_count,
            missing_trading_days=missing_days,
            zero_or_negative_prices=neg_price_count,
            zero_volumes=zero_vol_count,
            reverting_spikes=spike_count,
            strategy_used=self.strategy,
        )

        # Apply Remediation Strategy
        if self.strategy == "flag_only":
            df_work["flag_duplicate"] = dup_mask
            df_work["flag_bad_price"] = neg_price_mask
            df_work["flag_zero_volume"] = zero_vol_mask
            df_work["flag_spike"] = spike_mask
            report.cleaned_rows = len(df_work)
            return df_work, report

        elif self.strategy == "drop":
            drop_mask = dup_mask | neg_price_mask | spike_mask
            df_work = df_work[~drop_mask].reset_index(drop=True)
            report.cleaned_rows = len(df_work)
            logger.info("[%s] Strategy 'drop': %d bad rows removed.", ticker, total_raw - len(df_work))
            return df_work, report

        elif self.strategy == "forward_fill":
            # 1. Deduplicate by keeping first
            df_work = df_work[~dup_mask].copy()

            # 2. Forward fill bad prices and spikes with previous valid price
            bad_price_or_spike = neg_price_mask | spike_mask
            for col in price_cols:
                if col in df_work.columns:
                    df_work.loc[bad_price_or_spike, col] = np.nan
                    df_work[col] = df_work[col].ffill().bfill()

            # 3. Forward fill zero volume
            if volume_col in df_work.columns:
                df_work.loc[zero_vol_mask, volume_col] = np.nan
                df_work[volume_col] = df_work[volume_col].ffill().bfill()

            df_work = df_work.reset_index(drop=True)
            report.cleaned_rows = len(df_work)
            logger.info("[%s] Strategy 'forward_fill': cleaned %d affected entries.", ticker, report.total_issues)
            return df_work, report

        return df_work, report
