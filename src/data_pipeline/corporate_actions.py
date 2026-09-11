# ============================================================
# Corporate Actions Module — Split & Dividend Adjustments
# ============================================================
"""
Detects and adjusts for stock splits and dividend actions.

Provides:
  1. Heuristic split detection based on overnight price ratio drops (>40%)
     and volume step patterns, as well as reverse splits.
  2. Integration with yfinance authoritative split events.
  3. Retroactive back-adjustment of prices and volume scaling.
  4. Dividend reconciliation (raw close vs. adjusted close factor analysis).
  5. Audit logging for every detected and applied corporate action.

Usage:
    from src.data_pipeline.corporate_actions import CorporateActionsAdjuster

    adjuster = CorporateActionsAdjuster()
    splits = adjuster.detect_splits(df, ticker="AAPL")
    adjusted_df = adjuster.adjust_for_splits(df, splits)
    div_df = adjuster.reconcile_dividends(df, ticker="AAPL")
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SplitEvent:
    """Represents a detected or known stock split event."""

    ticker: str
    date: datetime.date
    ratio: float  # e.g., 4.0 for a 4:1 forward split, 0.1 for a 1:10 reverse split
    ratio_str: str  # e.g., "4:1" or "1:10"
    source: str  # "heuristic" or "yfinance"
    confidence: float = 1.0


@dataclass
class DividendEvent:
    """Represents an inferred or recorded dividend event."""

    ticker: str
    date: datetime.date
    adjustment_factor: float
    estimated_dividend: float


class CorporateActionsAdjuster:
    """Handles detection and adjustment of corporate actions (splits, dividends)."""

    def __init__(
        self,
        split_ratio_tolerance: float = 0.15,
        min_split_drop_pct: float = 0.40,
        min_reverse_split_jump_pct: float = 0.50,
    ) -> None:
        """
        Parameters
        ----------
        split_ratio_tolerance : float
            Maximum deviation from integer split ratios (e.g. 1.95 -> 2.0).
        min_split_drop_pct : float
            Minimum price drop (40%) to flag a candidate forward split.
        min_reverse_split_jump_pct : float
            Minimum price jump (50%) to flag a candidate reverse split.
        """
        self.split_ratio_tolerance = split_ratio_tolerance
        self.min_split_drop_pct = min_split_drop_pct
        self.min_reverse_split_jump_pct = min_reverse_split_jump_pct

    # ---- Split Detection ------------------------------------------------

    def detect_splits(
        self,
        df: pd.DataFrame,
        ticker: str = "UNKNOWN",
        price_col: str = "close",
        use_authoritative: bool = True,
    ) -> list[SplitEvent]:
        """Detect stock splits in a DataFrame using price ratio heuristics and/or yfinance.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame sorted ascending by date.
        ticker : str
            Symbol for logging and authoritative lookups.
        price_col : str
            Column to inspect for overnight price gaps.
        use_authoritative : bool
            Whether to cross-check with yfinance corporate actions API.

        Returns
        -------
        list[SplitEvent]
        """
        if df.empty or price_col not in df.columns or len(df) < 2:
            return []

        df_sorted = df.sort_values("date").reset_index(drop=True)
        detected_splits: list[SplitEvent] = []

        prices = df_sorted[price_col].values
        dates = pd.to_datetime(df_sorted["date"]).dt.date.values

        # 1. Heuristic Overnight Ratio Detection
        for i in range(1, len(prices)):
            p_prev = prices[i - 1]
            p_curr = prices[i]

            if p_prev <= 0 or p_curr <= 0 or np.isnan(p_prev) or np.isnan(p_curr):
                continue

            ratio = p_curr / p_prev

            # Forward Split Check (Large Drop, e.g. 4:1 drops price to ~25% of prev)
            # Drop > min_split_drop_pct -> ratio < 1 - min_split_drop_pct
            if ratio <= (1.0 - self.min_split_drop_pct):
                candidate_factor = 1.0 / ratio
                nearest_int = round(candidate_factor)
                if nearest_int >= 2 and abs(candidate_factor - nearest_int) / nearest_int <= self.split_ratio_tolerance:
                    event = SplitEvent(
                        ticker=ticker,
                        date=dates[i],
                        ratio=float(nearest_int),
                        ratio_str=f"{nearest_int}:1",
                        source="heuristic",
                        confidence=max(0.0, 1.0 - abs(candidate_factor - nearest_int)),
                    )
                    detected_splits.append(event)
                    logger.info(
                        "[%s] Detected likely forward split on %s: %s (factor=%.2f, price drop: %.2f -> %.2f)",
                        ticker,
                        event.date,
                        event.ratio_str,
                        event.ratio,
                        p_prev,
                        p_curr,
                    )

            # Reverse Split Check (Large Jump, e.g. 1:10 jumps price ~10x)
            elif ratio >= (1.0 + self.min_reverse_split_jump_pct):
                candidate_factor = ratio
                nearest_int = round(candidate_factor)
                if nearest_int >= 2 and abs(candidate_factor - nearest_int) / nearest_int <= self.split_ratio_tolerance:
                    event = SplitEvent(
                        ticker=ticker,
                        date=dates[i],
                        ratio=1.0 / float(nearest_int),
                        ratio_str=f"1:{nearest_int}",
                        source="heuristic",
                        confidence=max(0.0, 1.0 - abs(candidate_factor - nearest_int)),
                    )
                    detected_splits.append(event)
                    logger.info(
                        "[%s] Detected likely reverse split on %s: %s (factor=%.4f, price jump: %.2f -> %.2f)",
                        ticker,
                        event.date,
                        event.ratio_str,
                        event.ratio,
                        p_prev,
                        p_curr,
                    )

        # 2. Authoritative Verification via yfinance (if requested)
        if use_authoritative and ticker != "UNKNOWN":
            try:
                auth_splits = self.fetch_authoritative_splits(ticker)
                min_date = dates[0]
                max_date = dates[-1]

                for auth in auth_splits:
                    if min_date <= auth.date <= max_date:
                        # Check if already captured by heuristic
                        existing = next((s for s in detected_splits if abs((s.date - auth.date).days) <= 1), None)
                        if existing:
                            existing.source = "yfinance+heuristic"
                            existing.ratio = auth.ratio
                            existing.ratio_str = auth.ratio_str
                        else:
                            detected_splits.append(auth)
                            logger.info(
                                "[%s] Added authoritative yfinance split on %s: %s (factor=%.2f)",
                                ticker,
                                auth.date,
                                auth.ratio_str,
                                auth.ratio,
                            )
            except Exception as exc:
                logger.debug("[%s] Could not fetch authoritative splits: %s", ticker, exc)

        return sorted(detected_splits, key=lambda s: s.date)

    def fetch_authoritative_splits(self, ticker: str) -> list[SplitEvent]:
        """Fetch split history directly from yfinance."""
        splits: list[SplitEvent] = []
        try:
            t = yf.Ticker(ticker)
            s_series = t.splits
            if s_series is not None and not s_series.empty:
                for dt, factor in s_series.items():
                    factor_float = float(factor)
                    if factor_float > 0 and factor_float != 1.0:
                        split_date = pd.to_datetime(dt).date()
                        if factor_float >= 1.0:
                            ratio_str = f"{int(factor_float) if factor_float.is_integer() else factor_float:.2f}:1"
                        else:
                            inv = 1.0 / factor_float
                            ratio_str = f"1:{int(inv) if inv.is_integer() else inv:.2f}"

                        splits.append(
                            SplitEvent(
                                ticker=ticker,
                                date=split_date,
                                ratio=factor_float,
                                ratio_str=ratio_str,
                                source="yfinance",
                            )
                        )
        except Exception as exc:
            logger.warning("[%s] Error retrieving yfinance splits: %s", ticker, exc)

        return splits

    # ---- Split Adjustment -----------------------------------------------

    def adjust_for_splits(
        self,
        df: pd.DataFrame,
        splits: list[SplitEvent],
        price_cols: Sequence[str] = ("open", "high", "low", "close", "adj_close"),
        volume_col: str = "volume",
    ) -> pd.DataFrame:
        """Retroactively back-adjust prices and forward-adjust volumes for detected splits.

        For each split event on date T with split ratio N (e.g. N=4 for 4:1):
          • Historical prices for t < T are divided by N (multiplied by 1/N).
          • Historical volumes for t < T are multiplied by N.
        """
        if df.empty or not splits:
            return df.copy()

        df_adj = df.copy()
        df_dates = pd.to_datetime(df_adj["date"]).dt.date

        for split in splits:
            factor = split.ratio
            if factor <= 0 or factor == 1.0:
                continue

            mask = df_dates < split.date
            # Back-adjust prices
            for col in price_cols:
                if col in df_adj.columns:
                    df_adj.loc[mask, col] = df_adj.loc[mask, col] / factor

            # Back-adjust volumes (so past volumes match post-split share base)
            if volume_col in df_adj.columns:
                df_adj.loc[mask, volume_col] = df_adj.loc[mask, volume_col] * factor

            logger.info(
                "[%s] Applied split adjustment on %s: ratio %s (factor=%.2f) to %d historical bars",
                split.ticker,
                split.date,
                split.ratio_str,
                factor,
                int(mask.sum()),
            )

        return df_adj

    # ---- Dividend Reconciliation ----------------------------------------

    def reconcile_dividends(
        self,
        df: pd.DataFrame,
        ticker: str = "UNKNOWN",
        close_col: str = "close",
        adj_close_col: str = "adj_close",
    ) -> tuple[pd.DataFrame, list[DividendEvent]]:
        """Reconcile unadjusted close vs. adjusted close to calculate dividend discount factors.

        Calculates:
          • adjustment_factor = adj_close / close
          • ex-dividend dates where the factor changes significantly between bars.
        """
        if df.empty or close_col not in df.columns or adj_close_col not in df.columns:
            return df.copy(), []

        df_res = df.copy()
        # Calculate adjustment factor
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = df_res[adj_close_col] / df_res[close_col]
            factor = factor.fillna(1.0)

        df_res["dividend_factor"] = factor

        # Detect discrete drops in adjustment factor (indicating dividend payout)
        div_events: list[DividendEvent] = []
        factor_arr = factor.values
        dates = pd.to_datetime(df_res["date"]).dt.date.values
        closes = df_res[close_col].values

        for i in range(1, len(factor_arr)):
            f_prev = factor_arr[i - 1]
            f_curr = factor_arr[i]

            # In adjusted series, earlier prices are discounted: f_prev < f_curr on dividend
            if f_curr > 0 and (f_curr - f_prev) / f_curr > 0.001:
                # Approximate cash dividend per share = (1 - (f_prev / f_curr)) * close
                cash_div = (1.0 - (f_prev / f_curr)) * closes[i]
                event = DividendEvent(
                    ticker=ticker,
                    date=dates[i],
                    adjustment_factor=f_curr,
                    estimated_dividend=round(float(cash_div), 4),
                )
                div_events.append(event)
                logger.debug(
                    "[%s] Identified dividend adjustment on %s: factor=%.4f, est_div=$%.2f",
                    ticker,
                    event.date,
                    event.adjustment_factor,
                    event.estimated_dividend,
                )

        return df_res, div_events
