# ============================================================
# Unit Tests: Validation Rules (Phase 7)
# ============================================================
"""
Tests for individual validation rules:
  - SchemaValidationRule
  - RangeCheckRule
  - ContinuityCheckRule
  - FreshnessCheckRule
  - CrossSourceConsistencyRule
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.validation_rules import (
    ContinuityCheckRule,
    CrossSourceConsistencyRule,
    FreshnessCheckRule,
    RangeCheckRule,
    SchemaValidationRule,
)


@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    """Fixture providing 5 days of valid daily OHLCV data."""
    dates = pd.bdate_range(start="2024-01-08", periods=5, freq="B")
    return pd.DataFrame(
        {
            "open": [100.0, 102.0, 101.5, 103.0, 104.0],
            "high": [105.0, 106.0, 104.0, 107.0, 108.0],
            "low": [98.0, 101.0, 100.0, 102.0, 103.0],
            "close": [103.0, 104.0, 103.5, 106.0, 107.0],
            "volume": [1_000_000, 1_200_000, 950_000, 1_100_000, 1_300_000],
        },
        index=dates,
    )


# ============================================================
# SchemaValidationRule Tests
# ============================================================


def test_schema_valid(sample_valid_df: pd.DataFrame) -> None:
    rule = SchemaValidationRule()
    res = rule.validate(sample_valid_df, ticker="TEST")
    assert res.passed is True
    assert res.rule_name == "schema_validation"
    assert res.severity == "info"


def test_schema_missing_column(sample_valid_df: pd.DataFrame) -> None:
    df = sample_valid_df.drop(columns=["volume"])
    rule = SchemaValidationRule()
    res = rule.validate(df, ticker="TEST")
    assert res.passed is False
    assert res.severity == "critical"
    assert "Missing required columns" in res.message
    assert "volume" in res.details.get("missing_columns", [])


def test_schema_empty_df() -> None:
    rule = SchemaValidationRule()
    res = rule.validate(pd.DataFrame(), ticker="TEST")
    assert res.passed is False
    assert res.severity == "critical"
    assert "Empty dataset" in res.message


def test_schema_all_null_column(sample_valid_df: pd.DataFrame) -> None:
    df = sample_valid_df.copy()
    df["open"] = np.nan
    rule = SchemaValidationRule()
    res = rule.validate(df, ticker="TEST")
    assert res.passed is False
    assert res.severity == "critical"
    assert "100% NaN" in res.message


def test_schema_invalid_index(sample_valid_df: pd.DataFrame) -> None:
    df = sample_valid_df.reset_index(drop=True)
    rule = SchemaValidationRule(require_datetime_index=True)
    res = rule.validate(df, ticker="TEST")
    assert res.passed is False
    assert res.severity == "critical"
    assert "DatetimeIndex" in res.message


# ============================================================
# RangeCheckRule Tests
# ============================================================


def test_range_valid(sample_valid_df: pd.DataFrame) -> None:
    rule = RangeCheckRule()
    res = rule.validate(sample_valid_df, ticker="TEST")
    assert res.passed is True
    assert res.affected_rows == 0


def test_range_negative_price(sample_valid_df: pd.DataFrame) -> None:
    df = sample_valid_df.copy()
    df.iloc[0, df.columns.get_loc("close")] = -10.0
    rule = RangeCheckRule()
    res = rule.validate(df, ticker="TEST")
    assert res.passed is False
    assert res.severity == "critical"
    assert res.affected_rows >= 1
    assert "Non-positive price" in res.message


def test_range_negative_volume(sample_valid_df: pd.DataFrame) -> None:
    df = sample_valid_df.copy()
    df.iloc[1, df.columns.get_loc("volume")] = -500
    rule = RangeCheckRule()
    res = rule.validate(df, ticker="TEST")
    assert res.passed is False
    assert "Negative volume" in res.message


def test_range_high_lower_than_low(sample_valid_df: pd.DataFrame) -> None:
    df = sample_valid_df.copy()
    # High < Low violation
    df.iloc[2, df.columns.get_loc("high")] = 90.0
    df.iloc[2, df.columns.get_loc("low")] = 110.0
    rule = RangeCheckRule()
    res = rule.validate(df, ticker="TEST")
    assert res.passed is False
    assert res.severity == "critical"
    assert "high < low" in res.message


def test_range_high_lower_than_open(sample_valid_df: pd.DataFrame) -> None:
    df = sample_valid_df.copy()
    # High < Open violation
    df.iloc[0, df.columns.get_loc("high")] = 99.0
    rule = RangeCheckRule()
    res = rule.validate(df, ticker="TEST")
    assert res.passed is False
    assert "high < max(open, close)" in res.message


# ============================================================
# ContinuityCheckRule Tests
# ============================================================


def test_continuity_valid(sample_valid_df: pd.DataFrame) -> None:
    rule = ContinuityCheckRule()
    res = rule.validate(sample_valid_df, ticker="TEST")
    assert res.passed is True
    assert res.affected_rows == 0


def test_continuity_unexpected_gap() -> None:
    # Skip a week of trading days (e.g. 2024-01-08 to 2024-01-22)
    dates = pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-22"])
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [98.0, 99.0, 100.0],
            "close": [103.0, 104.0, 105.0],
            "volume": [1000, 1000, 1000],
        },
        index=dates,
    )
    rule = ContinuityCheckRule(max_consecutive_missing=1)
    res = rule.validate(df, ticker="TEST")
    assert res.passed is False
    assert res.severity == "warning"
    assert res.affected_rows > 0
    assert "gap(s)" in res.message


def test_continuity_unsorted_index(sample_valid_df: pd.DataFrame) -> None:
    df = sample_valid_df.iloc[::-1]  # Reverse order
    rule = ContinuityCheckRule()
    res = rule.validate(df, ticker="TEST")
    assert res.passed is False
    assert res.severity == "critical"
    assert "monotonically increasing" in res.message


# ============================================================
# FreshnessCheckRule Tests
# ============================================================


def test_freshness_daily_pass() -> None:
    now = pd.Timestamp.now()
    dates = pd.date_range(end=now, periods=5, freq="D")
    df = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0, 103.0, 104.0]},
        index=dates,
    )
    rule = FreshnessCheckRule(max_lag_days=7)
    res = rule.validate(df, ticker="TEST")
    assert res.passed is True


def test_freshness_daily_fail_stale() -> None:
    # 30 days old
    stale_date = pd.Timestamp.now() - pd.Timedelta(days=30)
    dates = pd.date_range(end=stale_date, periods=5, freq="D")
    df = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0, 103.0, 104.0]},
        index=dates,
    )
    rule = FreshnessCheckRule(max_lag_days=5)
    res = rule.validate(df, ticker="TEST")
    assert res.passed is False
    assert res.severity == "warning"
    assert "stale" in res.message


def test_freshness_seconds_lag() -> None:
    now = pd.Timestamp.now()
    # 50 seconds old
    dates = [now - pd.Timedelta(seconds=50)]
    df = pd.DataFrame({"close": [100.0]}, index=dates)

    # Within 60 seconds -> pass
    rule_pass = FreshnessCheckRule(max_lag_seconds=60)
    assert rule_pass.validate(df, ticker="TEST").passed is True

    # Within 30 seconds -> fail
    rule_fail = FreshnessCheckRule(max_lag_seconds=30)
    res = rule_fail.validate(df, ticker="TEST")
    assert res.passed is False
    assert "stale" in res.message


# ============================================================
# CrossSourceConsistencyRule Tests
# ============================================================


def test_cross_source_agreement(sample_valid_df: pd.DataFrame) -> None:
    ref_df = sample_valid_df.copy()
    ref_df["close"] = ref_df["close"] * 1.005  # 0.5% difference

    rule = CrossSourceConsistencyRule(ref_df=ref_df, max_divergence_pct=2.0)
    res = rule.validate(sample_valid_df, ticker="TEST")
    assert res.passed is True
    assert res.affected_rows == 0


def test_cross_source_divergence(sample_valid_df: pd.DataFrame) -> None:
    ref_df = sample_valid_df.copy()
    ref_df.iloc[2, ref_df.columns.get_loc("close")] *= 1.25  # 25% discrepancy

    rule = CrossSourceConsistencyRule(ref_df=ref_df, max_divergence_pct=5.0)
    res = rule.validate(sample_valid_df, ticker="TEST")
    assert res.passed is False
    assert res.severity == "warning"
    assert res.affected_rows == 1
    assert "discrepancies" in res.message


def test_cross_source_no_overlap(sample_valid_df: pd.DataFrame) -> None:
    ref_df = sample_valid_df.copy()
    ref_df.index = ref_df.index + pd.DateOffset(years=10)

    rule = CrossSourceConsistencyRule(ref_df=ref_df)
    res = rule.validate(sample_valid_df, ticker="TEST")
    assert res.passed is False
    assert "No overlapping dates" in res.message
