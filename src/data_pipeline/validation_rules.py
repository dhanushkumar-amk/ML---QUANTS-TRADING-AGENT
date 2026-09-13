# ============================================================
# Composable Data Validation Rules (Phase 7)
# ============================================================
"""
Composable validation rules acting as a quality gatekeeper before data reaches
downstream feature engineering or ML models.

Rules included:
  1. SchemaValidationRule: Required columns, dtypes, non-empty, non-all-NaN.
  2. RangeCheckRule: Prices > 0, Volume >= 0, OHLC geometry (high >= low, etc.).
  3. ContinuityCheckRule: Validates against MarketCalendar for unexpected gaps.
  4. FreshnessCheckRule: Verifies latest timestamp is within allowable staleness lag.
  5. CrossSourceConsistencyRule: Cross-reconciles prices between data vendors (e.g. yfinance vs HF).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data_pipeline.market_calendar import MarketCalendar
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Structured result returned by each validation rule."""

    rule_name: str
    passed: bool
    severity: str = "info"  # "info", "warning", "critical"
    affected_rows: int = 0
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class ValidationRule(ABC):
    """Abstract base class for all composable data validation rules."""

    name: str = "base_rule"
    default_severity: str = "critical"

    def __init__(self, name: str | None = None, default_severity: str | None = None) -> None:
        if name:
            self.name = name
        if default_severity:
            self.default_severity = default_severity

    @abstractmethod
    def validate(
        self,
        df: pd.DataFrame,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Evaluate rule against DataFrame and return ValidationResult."""
        pass


class SchemaValidationRule(ValidationRule):
    """Validates required columns, valid dtypes, and absence of all-NaN columns."""

    name = "schema_validation"
    default_severity = "critical"

    def __init__(
        self,
        name: str = "schema_validation",
        required_columns: Sequence[str] = ("open", "high", "low", "close", "volume"),
        numeric_columns: Sequence[str] = ("open", "high", "low", "close", "volume"),
        date_col: str = "date",
        require_datetime_index: bool = False,
    ) -> None:
        super().__init__(name=name)
        self.required_columns = set(required_columns)
        self.numeric_columns = set(numeric_columns)
        self.date_col = date_col
        self.require_datetime_index = require_datetime_index

    def validate(
        self,
        df: pd.DataFrame,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        if df.empty:
            return ValidationResult(
                rule_name=self.name,
                passed=False,
                severity=self.default_severity,
                affected_rows=0,
                message="Empty dataset: DataFrame has 0 rows.",
            )

        if self.require_datetime_index and not isinstance(df.index, pd.DatetimeIndex):
            return ValidationResult(
                rule_name=self.name,
                passed=False,
                severity=self.default_severity,
                affected_rows=len(df),
                message=f"Expected DatetimeIndex, but found {type(df.index).__name__}.",
            )

        # Check required columns
        missing_cols = self.required_columns - set(df.columns)
        if missing_cols:
            return ValidationResult(
                rule_name=self.name,
                passed=False,
                severity=self.default_severity,
                affected_rows=len(df),
                message=f"Missing required columns: {sorted(missing_cols)}",
                details={"missing_columns": list(missing_cols)},
            )

        # Check for all-NaN columns
        all_nan_cols = [c for c in self.required_columns if df[c].isna().all()]
        if all_nan_cols:
            return ValidationResult(
                rule_name=self.name,
                passed=False,
                severity=self.default_severity,
                affected_rows=len(df),
                message=f"Columns contain 100% NaN values: {all_nan_cols}",
                details={"all_nan_columns": all_nan_cols},
            )

        # Check numeric types
        non_numeric = []
        for col in self.numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric.append(col)

        if non_numeric:
            return ValidationResult(
                rule_name=self.name,
                passed=False,
                severity=self.default_severity,
                affected_rows=len(df),
                message=f"Expected numeric dtype for columns: {non_numeric}",
                details={"non_numeric_columns": non_numeric},
            )

        return ValidationResult(
            rule_name=self.name,
            passed=True,
            severity="info",
            affected_rows=0,
            message="Schema check passed: all required columns and dtypes valid.",
        )


class RangeCheckRule(ValidationRule):
    """Validates price positivity, non-negative volume, and bar geometric consistency."""

    name = "range_check"
    default_severity = "critical"

    def __init__(
        self,
        name: str = "range_check",
        price_cols: Sequence[str] = ("open", "high", "low", "close"),
        volume_col: str = "volume",
        enforce_geometry: bool = True,
    ) -> None:
        super().__init__(name=name)
        self.price_cols = price_cols
        self.volume_col = volume_col
        self.enforce_geometry = enforce_geometry

    def validate(
        self,
        df: pd.DataFrame,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        if df.empty:
            return ValidationResult(
                rule_name=self.name, passed=True, severity="info", message="Empty DataFrame."
            )

        affected_mask = pd.Series(False, index=df.index)
        issues: list[str] = []

        # 1. Non-positive prices
        for col in self.price_cols:
            if col in df.columns:
                bad_prices = df[col] <= 0
                if bad_prices.any():
                    affected_mask |= bad_prices
                    issues.append(f"Non-positive price in '{col}': {bad_prices.sum()} occurrences")

        # 2. Negative volume
        if self.volume_col in df.columns:
            neg_vol = df[self.volume_col] < 0
            if neg_vol.any():
                affected_mask |= neg_vol
                issues.append(f"Negative volume: {neg_vol.sum()} occurrences")

        # 3. Bar Geometric Constraints
        if self.enforce_geometry and all(c in df.columns for c in ("high", "low", "open", "close")):
            geom_hl = df["high"] < df["low"]
            geom_ho = (df["high"] < df["open"]) | (df["high"] < df["close"])
            geom_lo = (df["low"] > df["open"]) | (df["low"] > df["close"])

            if geom_hl.any():
                affected_mask |= geom_hl
                issues.append(f"high < low violation: {geom_hl.sum()} bars")
            if geom_ho.any():
                affected_mask |= geom_ho
                issues.append(f"high < max(open, close) violation: {geom_ho.sum()} bars")
            if geom_lo.any():
                affected_mask |= geom_lo
                issues.append(f"low > min(open, close) violation: {geom_lo.sum()} bars")

        total_affected = int(affected_mask.sum())
        if total_affected > 0:
            return ValidationResult(
                rule_name=self.name,
                passed=False,
                severity=self.default_severity,
                affected_rows=total_affected,
                message="; ".join(issues),
                details={"issues": issues, "affected_count": total_affected},
            )

        return ValidationResult(
            rule_name=self.name,
            passed=True,
            severity="info",
            affected_rows=0,
            message="Range and bar geometry check passed.",
        )


class ContinuityCheckRule(ValidationRule):
    """Detects unexpected trading day gaps against exchange market calendar."""

    name = "continuity_check"
    default_severity = "warning"

    def __init__(
        self,
        name: str = "continuity_check",
        exchange: str = "NYSE",
        date_col: str = "date",
        max_consecutive_missing: int = 0,
    ) -> None:
        super().__init__(name=name)
        self.calendar = MarketCalendar(exchange)
        self.date_col = date_col
        self.max_consecutive_missing = max_consecutive_missing

    def validate(
        self,
        df: pd.DataFrame,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        if df.empty or len(df) < 2:
            return ValidationResult(
                rule_name=self.name, passed=True, severity="info", message="Insufficient rows."
            )

        if self.date_col in df.columns:
            date_series = pd.to_datetime(df[self.date_col])
        elif isinstance(df.index, pd.DatetimeIndex):
            date_series = pd.Series(df.index, index=df.index)
        else:
            return ValidationResult(
                rule_name=self.name,
                passed=False,
                severity="critical",
                message="No DatetimeIndex or date column found for continuity check.",
            )

        if not date_series.is_monotonic_increasing:
            return ValidationResult(
                rule_name=self.name,
                passed=False,
                severity="critical",
                affected_rows=len(df),
                message="Dates are not monotonically increasing.",
            )

        dates = date_series.dt.date.values
        missing_sessions = self.calendar.find_missing_trading_days(
            actual_dates=dates,
            start_date=dates[0],
            end_date=dates[-1],
        )

        if len(missing_sessions) > self.max_consecutive_missing:
            msg = (
                f"Found {len(missing_sessions)} unexpected trading session gap(s) "
                f"against {self.calendar.exchange} calendar."
            )
            return ValidationResult(
                rule_name=self.name,
                passed=False,
                severity=self.default_severity,
                affected_rows=len(missing_sessions),
                message=msg,
                details={
                    "missing_sessions": [d.strftime("%Y-%m-%d") for d in missing_sessions[:10]],
                    "total_missing": len(missing_sessions),
                },
            )

        return ValidationResult(
            rule_name=self.name,
            passed=True,
            severity="info",
            affected_rows=0,
            message=f"Continuity check passed: no missing trading days against {self.calendar.exchange} calendar.",
        )


class FreshnessCheckRule(ValidationRule):
    """Verifies that the latest timestamp is within an allowable staleness lag."""

    name = "freshness_check"
    default_severity = "warning"

    def __init__(
        self,
        name: str = "freshness_check",
        max_lag_days: int = 5,  # 5 days default for daily data
        max_lag_seconds: float | None = None,  # for intraday / live streaming
        date_col: str = "date",
    ) -> None:
        super().__init__(name=name)
        self.max_lag_days = max_lag_days
        self.max_lag_seconds = max_lag_seconds
        self.date_col = date_col

    def validate(
        self,
        df: pd.DataFrame,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        if df.empty:
            return ValidationResult(
                rule_name=self.name, passed=False, severity="critical", message="Empty DataFrame."
            )

        if self.date_col in df.columns:
            latest_dt = pd.to_datetime(df[self.date_col]).max()
        elif isinstance(df.index, pd.DatetimeIndex):
            latest_dt = df.index.max()
        else:
            return ValidationResult(
                rule_name=self.name,
                passed=False,
                severity="critical",
                message="No DatetimeIndex or date column found for freshness check.",
            )

        ref_now = (
            pd.to_datetime(context["now"])
            if context and "now" in context
            else pd.Timestamp.now(tz=latest_dt.tz if latest_dt.tz else None)
        )

        if latest_dt.tz is None and ref_now.tz is not None:
            ref_now = ref_now.tz_localize(None)
        elif latest_dt.tz is not None and ref_now.tz is None:
            latest_dt = latest_dt.tz_localize(None)

        lag = ref_now - latest_dt

        if self.max_lag_seconds is not None:
            lag_sec = lag.total_seconds()
            if lag_sec > self.max_lag_seconds:
                return ValidationResult(
                    rule_name=self.name,
                    passed=False,
                    severity=self.default_severity,
                    affected_rows=1,
                    message=f"Data is stale: lag of {lag_sec:.1f}s exceeds threshold ({self.max_lag_seconds}s).",
                    details={"lag_seconds": lag_sec, "latest_timestamp": str(latest_dt)},
                )
        else:
            lag_days = lag.days
            if lag_days > self.max_lag_days:
                return ValidationResult(
                    rule_name=self.name,
                    passed=False,
                    severity=self.default_severity,
                    affected_rows=1,
                    message=f"Data is stale: lag of {lag_days} days exceeds threshold ({self.max_lag_days} days).",
                    details={"lag_days": lag_days, "latest_date": str(latest_dt.date())},
                )

        return ValidationResult(
            rule_name=self.name,
            passed=True,
            severity="info",
            affected_rows=0,
            message=f"Freshness check passed: latest bar is {latest_dt}.",
        )


class CrossSourceConsistencyRule(ValidationRule):
    """Reconciles price discrepancies across two sources for the same ticker/dates."""

    name = "cross_source_consistency"
    default_severity = "warning"

    def __init__(
        self,
        name: str = "cross_source_consistency",
        ref_df: pd.DataFrame | None = None,
        discrepancy_threshold_pct: float = 0.05,  # 5% max acceptable price delta
        max_divergence_pct: float | None = None,
        price_col: str = "close",
        date_col: str = "date",
    ) -> None:
        super().__init__(name=name)
        if max_divergence_pct is not None:
            self.discrepancy_threshold_pct = (
                max_divergence_pct / 100.0 if max_divergence_pct > 1.0 else max_divergence_pct
            )
        else:
            self.discrepancy_threshold_pct = discrepancy_threshold_pct
        self.ref_df = ref_df
        self.price_col = price_col
        self.date_col = date_col

    def _extract_price_series(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.date_col in df.columns:
            sub = df[[self.date_col, self.price_col]].copy()
            sub[self.date_col] = pd.to_datetime(sub[self.date_col]).dt.date
        elif isinstance(df.index, pd.DatetimeIndex):
            sub = pd.DataFrame(
                {
                    self.date_col: pd.to_datetime(df.index).date,
                    self.price_col: df[self.price_col].values,
                }
            )
        else:
            sub = pd.DataFrame()
        return sub

    def validate(
        self,
        df: pd.DataFrame,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        ref = (
            self.ref_df
            if self.ref_df is not None
            else (context.get("reference_df") if context else None)
        )
        if ref is None:
            return ValidationResult(
                rule_name=self.name,
                passed=True,
                severity="info",
                message="Skipped: No reference dataset provided.",
            )

        df1 = self._extract_price_series(df)
        df2 = self._extract_price_series(ref)

        if df1.empty or df2.empty:
            return ValidationResult(
                rule_name=self.name,
                passed=False,
                severity=self.default_severity,
                message="Empty comparison data.",
            )

        merged = pd.merge(df1, df2, on=self.date_col, suffixes=("_primary", "_ref")).dropna()

        if merged.empty:
            return ValidationResult(
                rule_name=self.name,
                passed=False,
                severity=self.default_severity,
                message="No overlapping dates between primary and reference datasets.",
            )

        p1 = merged[f"{self.price_col}_primary"].values
        p2 = merged[f"{self.price_col}_ref"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.abs(p1 - p2) / np.where(p2 > 0, p2, 1.0)

        discrepancies = delta > self.discrepancy_threshold_pct
        num_divergent = int(np.sum(discrepancies))

        if num_divergent > 0:
            max_delta = float(np.max(delta[discrepancies]))
            return ValidationResult(
                rule_name=self.name,
                passed=False,
                severity=self.default_severity,
                affected_rows=num_divergent,
                message=(
                    f"Found {num_divergent} price discrepancies (> {self.discrepancy_threshold_pct*100:.1f}%, "
                    f"max: {max_delta*100:.1f}%)."
                ),
                details={"divergent_bars": num_divergent, "max_discrepancy_pct": max_delta * 100},
            )

        return ValidationResult(
            rule_name=self.name,
            passed=True,
            severity="info",
            affected_rows=0,
            message="Cross-source consistency check passed: prices align within threshold.",
        )
